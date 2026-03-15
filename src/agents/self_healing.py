# src/agents/self_healing.py
"""
2026 Standard — Agentic Self-Healing Infrastructure
=====================================================
Implements the "Shadow-to-Active Governor": an autonomous agent that
monitors shadow A/B results and promotes models without human intervention.

Components:
  ShadowGovernor         — watches shadow eval metrics; auto-promotes when
                           candidate outperforms production by >2% for 48h.
  SelfHealingOrchestrator— diagnoses drift root cause and selects repair action.
  ResourceAdvisor        — detects memory spikes and recommends instance upgrade.
  PRPromoter             — creates a promotion record (simulates a GitHub PR).

Netflix 2026 standard: Zero-touch model promotion within 48h of sustained lift.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ── Enums & config ────────────────────────────────────────────────────────────

class PromotionDecision(str, Enum):
    PROMOTE   = "promote"
    HOLD      = "hold"
    ROLLBACK  = "rollback"
    ESCALATE  = "escalate"   # human needed


PROMOTION_LIFT_THRESHOLD  = 0.02    # >2% nDCG@10 lift required
PROMOTION_HOURS_REQUIRED  = 48      # must sustain lift for 48 h
ROLLBACK_DROP_THRESHOLD   = 0.05    # auto-rollback if live model drops >5%
MEMORY_SPIKE_THRESHOLD_GB = 1.5     # auto-scale if peak RAM exceeds this


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ShadowObservation:
    """One hourly shadow A/B measurement."""
    timestamp: float
    production_ndcg: float
    candidate_ndcg: float
    n_queries: int
    all_unit_tests_pass: bool = True

    @property
    def lift(self) -> float:
        base = max(1e-9, self.production_ndcg)
        return (self.candidate_ndcg - self.production_ndcg) / base


@dataclass
class PromotionRecord:
    decision: PromotionDecision
    reason: str
    lift_pct: float
    sustained_hours: float
    candidate_model_path: str
    timestamp: float = field(default_factory=time.time)
    pr_url: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "lift_pct": round(self.lift_pct * 100, 3),
            "sustained_hours": round(self.sustained_hours, 1),
            "candidate_model_path": self.candidate_model_path,
            "timestamp": self.timestamp,
            "pr_url": self.pr_url,
            **self.metadata,
        }


# ── Shadow Governor ───────────────────────────────────────────────────────────

class ShadowGovernor:
    """
    Continuously ingests ShadowObservations and decides when to promote.

    Algorithm:
      1. Track a rolling window of observations.
      2. If lift > PROMOTION_LIFT_THRESHOLD AND all unit tests pass AND
         sustained for >= PROMOTION_HOURS_REQUIRED → PROMOTE.
      3. If production nDCG drops > ROLLBACK_DROP_THRESHOLD → ROLLBACK.
      4. If lift is positive but unstable → HOLD.
    """

    def __init__(
        self,
        lift_threshold: float = PROMOTION_LIFT_THRESHOLD,
        hours_required: float = PROMOTION_HOURS_REQUIRED,
        rollback_drop: float = ROLLBACK_DROP_THRESHOLD,
    ) -> None:
        self.lift_threshold = lift_threshold
        self.hours_required = hours_required
        self.rollback_drop = rollback_drop
        self._window: list[ShadowObservation] = []

    def ingest(self, obs: ShadowObservation) -> None:
        self._window.append(obs)
        # keep 7-day window
        cutoff = time.time() - 7 * 24 * 3600
        self._window = [o for o in self._window if o.timestamp >= cutoff]

    def evaluate(self, candidate_model_path: str) -> PromotionRecord:
        if not self._window:
            return PromotionRecord(
                decision=PromotionDecision.HOLD,
                reason="No shadow observations yet",
                lift_pct=0.0, sustained_hours=0.0,
                candidate_model_path=candidate_model_path,
            )

        # Check rollback: production degraded?
        baseline = self._window[0].production_ndcg
        latest_prod = self._window[-1].production_ndcg
        prod_drop = (baseline - latest_prod) / max(1e-9, baseline)
        if prod_drop > self.rollback_drop:
            return PromotionRecord(
                decision=PromotionDecision.ROLLBACK,
                reason=f"Production nDCG dropped {prod_drop:.1%} > threshold {self.rollback_drop:.1%}",
                lift_pct=-prod_drop, sustained_hours=0.0,
                candidate_model_path=candidate_model_path,
            )

        # Find longest run where lift >= threshold AND tests pass
        sustained = self._sustained_hours_above_threshold()
        avg_lift = sum(o.lift for o in self._window) / len(self._window)
        all_tests = all(o.all_unit_tests_pass for o in self._window[-5:])

        if sustained >= self.hours_required and all_tests and avg_lift >= self.lift_threshold:
            pr = PRPromoter().create_pr(candidate_model_path, avg_lift, sustained)
            return PromotionRecord(
                decision=PromotionDecision.PROMOTE,
                reason=f"Candidate sustained {sustained:.1f}h lift={avg_lift:.1%} > {self.lift_threshold:.1%}, all tests pass",
                lift_pct=avg_lift, sustained_hours=sustained,
                candidate_model_path=candidate_model_path,
                pr_url=pr,
            )

        if avg_lift >= self.lift_threshold and not all_tests:
            return PromotionRecord(
                decision=PromotionDecision.HOLD,
                reason=f"Lift {avg_lift:.1%} is good but unit tests failing — holding",
                lift_pct=avg_lift, sustained_hours=sustained,
                candidate_model_path=candidate_model_path,
            )

        if avg_lift < 0:
            return PromotionRecord(
                decision=PromotionDecision.HOLD,
                reason=f"Candidate underperforming production (lift={avg_lift:.1%})",
                lift_pct=avg_lift, sustained_hours=0.0,
                candidate_model_path=candidate_model_path,
            )

        return PromotionRecord(
            decision=PromotionDecision.HOLD,
            reason=f"Lift {avg_lift:.1%} positive but need {self.hours_required}h sustained (have {sustained:.1f}h)",
            lift_pct=avg_lift, sustained_hours=sustained,
            candidate_model_path=candidate_model_path,
        )

    def _sustained_hours_above_threshold(self) -> float:
        if not self._window:
            return 0.0
        # Walk backwards from latest
        run_start: float | None = None
        prev_ts: float | None = None
        total = 0.0
        for obs in reversed(self._window):
            if obs.lift >= self.lift_threshold and obs.all_unit_tests_pass:
                if prev_ts is not None:
                    total += (prev_ts - obs.timestamp) / 3600
                else:
                    run_start = obs.timestamp
                prev_ts = obs.timestamp
            else:
                break
        return total


# ── Self-Healing Orchestrator ─────────────────────────────────────────────────

class DriftCause(str, Enum):
    DATA_SHIFT      = "data_shift"
    MODEL_STALENESS = "model_staleness"
    FEATURE_SKEW    = "feature_skew"
    INFRA_ISSUE     = "infra_issue"
    UNKNOWN         = "unknown"


@dataclass
class HealingAction:
    cause: DriftCause
    action: str
    priority: int   # 1=immediate, 2=scheduled, 3=advisory
    metadata: dict = field(default_factory=dict)


class SelfHealingOrchestrator:
    """
    Diagnoses drift root cause from metric signals and selects a repair action.

    Repair menu:
      - data_shift      → trigger incremental retraining on fresh data
      - model_staleness → full retrain + OPE gate
      - feature_skew    → reload feature pipeline, validate distributions
      - infra_issue     → alert + scale-up recommendation
    """

    def diagnose(
        self,
        metrics: dict[str, Any],
        drift_report: dict[str, Any],
    ) -> HealingAction:
        ndcg_drop     = float(drift_report.get("ndcg_drop", 0))
        latency_spike = float(drift_report.get("p95_ms_delta", 0))
        feature_drift = float(drift_report.get("feature_psi", 0))   # PSI > 0.2 = severe

        if latency_spike > 100:
            return HealingAction(
                cause=DriftCause.INFRA_ISSUE,
                action="scale_up_api_replicas",
                priority=1,
                metadata={"p95_delta_ms": latency_spike, "recommended_replicas": 4},
            )
        if feature_drift > 0.2:
            return HealingAction(
                cause=DriftCause.FEATURE_SKEW,
                action="reload_feature_pipeline",
                priority=1,
                metadata={"psi": feature_drift},
            )
        if ndcg_drop > 0.05:
            return HealingAction(
                cause=DriftCause.DATA_SHIFT,
                action="trigger_incremental_retrain",
                priority=1,
                metadata={"ndcg_drop": ndcg_drop},
            )
        if ndcg_drop > 0.02:
            return HealingAction(
                cause=DriftCause.MODEL_STALENESS,
                action="schedule_full_retrain",
                priority=2,
                metadata={"ndcg_drop": ndcg_drop},
            )
        return HealingAction(
            cause=DriftCause.UNKNOWN,
            action="monitor_and_alert",
            priority=3,
            metadata={"ndcg_drop": ndcg_drop},
        )

    def execute(self, action: HealingAction, out_dir: Path | None = None) -> dict[str, Any]:
        result = {
            "action": action.action,
            "cause": action.cause.value,
            "priority": action.priority,
            "executed_at": time.time(),
            **action.metadata,
        }
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "healing_action.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        return result


# ── Resource Advisor ──────────────────────────────────────────────────────────

class ResourceAdvisor:
    """
    Detects memory spikes during Metaflow training steps and recommends
    instance type upgrades. Integrates with @netflix_standard decorator.
    """

    def __init__(self, spike_threshold_gb: float = MEMORY_SPIKE_THRESHOLD_GB) -> None:
        self.threshold = spike_threshold_gb

    def check(self, peak_memory_gb: float, current_instance: str = "m5.xlarge") -> dict[str, Any]:
        if peak_memory_gb < self.threshold:
            return {"action": "none", "current": current_instance, "peak_gb": peak_memory_gb}

        # Simple upgrade ladder
        UPGRADE = {
            "m5.xlarge": "m5.2xlarge",
            "m5.2xlarge": "m5.4xlarge",
            "m5.4xlarge": "r5.4xlarge",
        }
        recommended = UPGRADE.get(current_instance, "r5.8xlarge")
        return {
            "action": "upgrade_instance",
            "current": current_instance,
            "recommended": recommended,
            "peak_gb": round(peak_memory_gb, 2),
            "threshold_gb": self.threshold,
            "reason": f"Peak RAM {peak_memory_gb:.1f}GB exceeded threshold {self.threshold}GB",
        }


# ── PR Promoter ───────────────────────────────────────────────────────────────

class PRPromoter:
    """
    Simulates creating a GitHub Pull Request to promote a model.
    In production: replace with PyGithub or GitHub Actions API call.
    """

    def create_pr(
        self,
        model_path: str,
        lift: float,
        sustained_hours: float,
    ) -> str:
        title = f"[AutoPromote] LTR model — +{lift:.1%} nDCG@10 sustained {sustained_hours:.0f}h"
        body = (
            f"**Automated promotion by ShadowGovernor**\n\n"
            f"- Candidate model: `{model_path}`\n"
            f"- Lift: `{lift:.4f}` ({lift:.1%}) over production\n"
            f"- Sustained: `{sustained_hours:.1f}` hours\n"
            f"- All unit tests: ✅\n\n"
            f"*This PR was created automatically. Review shadow metrics before merging.*"
        )
        # In production: POST to GitHub API
        # For now: write to reports/
        pr_log = Path("reports/latest/auto_pr.json")
        pr_log.parent.mkdir(parents=True, exist_ok=True)
        pr_log.write_text(json.dumps({"title": title, "body": body, "created_at": time.time()}, indent=2))
        return f"reports/latest/auto_pr.json"
