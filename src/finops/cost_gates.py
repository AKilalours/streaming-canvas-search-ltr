# src/finops/cost_gates.py
"""
2026 Standard — FinOps & Unit Economics for AI Deployments
===========================================================
Cost-Aware AI: a deployment is rejected if its inference cost exceeds
the expected revenue lift it provides.

Components:
  CostEstimator       — estimates per-query inference cost (compute + memory).
  RevenueCalculator   — converts nDCG lift → estimated revenue per user.
  FinOpsGate          — pre-deployment gate: ROI = revenue_lift / cost.
                        Rejects if ROI < threshold.
  ArtifactLifecycle   — automates S3 tiering: moves artifacts > 30 days
                        to Glacier, keeps only winning models hot.

Netflix standard: Every model deployment must have ROI > 1.0 (revenue > cost).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── Config ────────────────────────────────────────────────────────────────────

# These are illustrative unit costs — replace with your cloud billing data
COST_PER_CPU_SEC   = 0.000048    # $/CPU-second (c5.2xlarge on-demand)
COST_PER_GB_RAM_HR = 0.006       # $/GB-hour
REVENUE_PER_USER   = 0.45        # $/user/month (avg Netflix ARPU / 30 / 24h)
NDCG_TO_RETENTION  = 0.12        # 1 unit nDCG@10 lift ≈ 12% retention improvement
RETENTION_TO_REV   = 0.08        # 1% retention ≈ 0.08% revenue per user


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class InferenceCost:
    model_name: str
    p50_latency_ms: float
    p99_latency_ms: float
    queries_per_day: int
    cpu_cores: float
    ram_gb: float
    cost_per_query_usd: float
    cost_per_day_usd: float
    cost_per_user_usd: float


@dataclass
class RevenueLift:
    ndcg_lift: float
    retention_lift_pct: float
    revenue_lift_per_user_usd: float
    revenue_lift_per_day_usd: float
    n_users: int


@dataclass
class FinOpsDecision:
    approved: bool
    roi: float                     # revenue_lift / cost — must be > threshold
    roi_threshold: float
    reason: str
    cost: InferenceCost
    revenue: RevenueLift
    recommendation: str


# ── Cost Estimator ────────────────────────────────────────────────────────────

class CostEstimator:
    """
    Estimates inference cost from latency + resource allocation.
    Uses your actual latency benchmark output (reports/latest/latency.json).
    """

    def __init__(
        self,
        cpu_cores: float = 2.0,
        ram_gb: float = 4.0,
        queries_per_day: int = 1_000_000,
    ) -> None:
        self.cpu_cores = cpu_cores
        self.ram_gb = ram_gb
        self.queries_per_day = queries_per_day

    def estimate_from_latency_report(
        self,
        latency_report_path: str | Path,
        model_name: str = "hybrid_ltr",
    ) -> InferenceCost:
        path = Path(latency_report_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            p50 = float(data.get("p50_ms", 50))
            p99 = float(data.get("p99_ms", 200))
        else:
            p50, p99 = 50.0, 200.0   # fallback estimates

        # Cost per query: CPU time * cores * rate + RAM * rate
        cpu_sec = p50 / 1000 * self.cpu_cores
        ram_hr  = (p50 / 1000 / 3600) * self.ram_gb
        cpq = cpu_sec * COST_PER_CPU_SEC + ram_hr * COST_PER_GB_RAM_HR
        cpd = cpq * self.queries_per_day

        return InferenceCost(
            model_name=model_name,
            p50_latency_ms=p50,
            p99_latency_ms=p99,
            queries_per_day=self.queries_per_day,
            cpu_cores=self.cpu_cores,
            ram_gb=self.ram_gb,
            cost_per_query_usd=round(cpq, 8),
            cost_per_day_usd=round(cpd, 4),
            cost_per_user_usd=round(cpd / max(1, self.queries_per_day / 3), 6),
        )


# ── Revenue Calculator ────────────────────────────────────────────────────────

class RevenueCalculator:
    """
    Converts nDCG@10 lift into expected revenue per user.

    Formula (simplified from Netflix internal elasticity model):
      retention_lift = ndcg_lift * NDCG_TO_RETENTION * 100
      revenue_lift   = retention_lift * RETENTION_TO_REV * REVENUE_PER_USER
    """

    def __init__(self, n_users: int = 238_000_000) -> None:
        self.n_users = n_users

    def calculate(self, ndcg_lift: float) -> RevenueLift:
        retention_lift_pct = ndcg_lift * NDCG_TO_RETENTION * 100
        rev_per_user = retention_lift_pct * RETENTION_TO_REV * REVENUE_PER_USER
        rev_per_day  = rev_per_user * self.n_users / 30   # monthly → daily

        return RevenueLift(
            ndcg_lift=round(ndcg_lift, 4),
            retention_lift_pct=round(retention_lift_pct, 4),
            revenue_lift_per_user_usd=round(rev_per_user, 6),
            revenue_lift_per_day_usd=round(rev_per_day, 2),
            n_users=self.n_users,
        )


# ── FinOps Gate ───────────────────────────────────────────────────────────────

class FinOpsGate:
    """
    Pre-deployment cost gate. Integrated into @netflix_standard decorator
    and ProductionLTRFlow gate_check step.

    Rejects deployment if:
      ROI = revenue_lift_per_day / cost_per_day < roi_threshold
    """

    def __init__(self, roi_threshold: float = 1.5) -> None:
        self.roi_threshold = roi_threshold

    def evaluate(
        self,
        ndcg_lift: float,
        latency_report_path: str | Path = "reports/latest/latency.json",
        n_users: int = 238_000_000,
        queries_per_day: int = 1_000_000,
    ) -> FinOpsDecision:
        cost_est = CostEstimator(queries_per_day=queries_per_day)
        cost = cost_est.estimate_from_latency_report(latency_report_path)
        rev  = RevenueCalculator(n_users=n_users).calculate(ndcg_lift)

        roi = rev.revenue_lift_per_day_usd / max(1e-9, cost.cost_per_day_usd)
        approved = roi >= self.roi_threshold

        if approved:
            rec = f"Deploy approved. ROI={roi:.1f}x exceeds threshold {self.roi_threshold}x."
        elif roi > 0.5:
            rec = f"Marginal ROI={roi:.1f}x. Consider reducing model complexity or batching inference."
        else:
            rec = f"Rejected. ROI={roi:.2f}x is below threshold. Optimise latency or increase lift first."

        return FinOpsDecision(
            approved=approved, roi=round(roi, 3),
            roi_threshold=self.roi_threshold,
            reason=f"nDCG lift={ndcg_lift:.4f}, rev/day=${rev.revenue_lift_per_day_usd:.2f}, cost/day=${cost.cost_per_day_usd:.4f}",
            cost=cost, revenue=rev, recommendation=rec,
        )

    def to_dict(self, decision: FinOpsDecision) -> dict[str, Any]:
        return {
            "approved": decision.approved,
            "roi": decision.roi,
            "roi_threshold": decision.roi_threshold,
            "reason": decision.reason,
            "recommendation": decision.recommendation,
            "cost_per_day_usd": decision.cost.cost_per_day_usd,
            "revenue_lift_per_day_usd": decision.revenue.revenue_lift_per_day_usd,
            "ndcg_lift": decision.revenue.ndcg_lift,
            "p50_latency_ms": decision.cost.p50_latency_ms,
        }


# ── Artifact Lifecycle Manager ────────────────────────────────────────────────

@dataclass
class ArtifactTieringResult:
    archived: list[str]
    kept_hot: list[str]
    freed_bytes: int
    policy: str


class ArtifactLifecycle:
    """
    Automates artifact tiering:
      - Models > 30 days AND not tagged 'winner' → archive (simulates S3 Glacier move)
      - Latest winning model always stays hot
      - Training datasets > 30 days → archive

    In production: replace _archive() with boto3 S3 copy + delete to Glacier.
    """

    def __init__(
        self,
        artifacts_dir: str | Path = "artifacts",
        archive_after_days: int = 30,
        dry_run: bool = False,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.archive_after_days = archive_after_days
        self.dry_run = dry_run
        self._cutoff = time.time() - archive_after_days * 86400

    def run(self, winning_model_path: str | None = None) -> ArtifactTieringResult:
        archived, kept_hot = [], []
        freed = 0

        for path in self.artifacts_dir.rglob("*"):
            if not path.is_file():
                continue
            age = time.time() - path.stat().st_mtime
            is_winner = winning_model_path and str(path) == winning_model_path
            is_old = age > self.archive_after_days * 86400
            size = path.stat().st_size

            if is_winner:
                kept_hot.append(str(path))
            elif is_old:
                freed += size
                archived.append(str(path))
                if not self.dry_run:
                    self._archive(path)
            else:
                kept_hot.append(str(path))

        return ArtifactTieringResult(
            archived=archived,
            kept_hot=kept_hot,
            freed_bytes=freed,
            policy=f">={self.archive_after_days}d → glacier, winner always hot",
        )

    def _archive(self, path: Path) -> None:
        """
        Production: boto3.client('s3').copy_object(... StorageClass='GLACIER')
        Development: rename to .archived suffix as simulation.
        """
        archived_path = path.with_suffix(path.suffix + ".archived")
        try:
            path.rename(archived_path)
        except Exception:
            pass   # best-effort
