# src/causal/uplift.py
"""
2026 Standard — Uplift Modeling & Off-Policy Evaluation
=========================================================
Moves beyond "predicted CTR" to Causal Incrementality:
  "Would the user have watched this WITHOUT the recommendation?"

Components:
  IncrementalityScorer   — estimates P(watch|shown) - P(watch|not_shown)
                           using a doubly-robust IPW estimator on logged data.
  OffPolicyEvaluator     — Policy Convolution OPE: evaluates a new policy
                           against logged bandit data before live deployment.
  UpliftFeatureEnricher  — adds `incrementality_score` to LTR feature vectors.

Netflix 2026 standard: A recommendation slot is only "valuable" if
incrementality_score > UPLIFT_THRESHOLD (default 0.05).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


UPLIFT_THRESHOLD = 0.05   # minimum causal lift to justify a slot


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class LoggedEvent:
    """One row of historical bandit log."""
    doc_id: str
    shown: bool           # was this item shown (treatment) or not (control)?
    watched: bool         # did the user watch?
    propensity: float     # P(shown) under the logging policy — needed for IPW
    user_id: str = ""
    context: dict = field(default_factory=dict)


@dataclass
class UpliftScore:
    doc_id: str
    incrementality_score: float   # tau-hat: E[Y(1) - Y(0)]
    p_watch_shown: float          # E[Y | T=1]
    p_watch_not_shown: float      # E[Y | T=0]
    method: str = "ipw_dr"
    confident: bool = False       # True if n_treated + n_control >= MIN_SAMPLES


# ── Core estimator ────────────────────────────────────────────────────────────

MIN_SAMPLES = 10   # below this, fall back to prior


class IncrementalityScorer:
    """
    Doubly-Robust IPW uplift estimator.

    DR estimator: tau_hat = E[(T/e - (1-T)/(1-e)) * Y]
    where T=treatment indicator, e=propensity, Y=outcome.

    Falls back to a Thompson-sampled Beta prior when sample count is low.
    """

    def __init__(
        self,
        threshold: float = UPLIFT_THRESHOLD,
        prior_alpha: float = 1.0,
        prior_beta: float = 4.0,
    ) -> None:
        self.threshold = threshold
        self.prior_alpha = prior_alpha   # weak positive prior
        self.prior_beta = prior_beta
        # item_id → {"n_t": int, "y_t": float, "n_c": int, "y_c": float, "ipw_sum": float}
        self._stats: dict[str, dict[str, float]] = {}

    def update(self, event: LoggedEvent) -> None:
        """Online update from a logged event."""
        s = self._stats.setdefault(event.doc_id, {"n_t": 0, "y_t": 0.0, "n_c": 0, "y_c": 0.0, "ipw_sum": 0.0})
        y = float(event.watched)
        e = max(1e-6, min(1 - 1e-6, event.propensity))
        if event.shown:
            s["n_t"] += 1
            s["y_t"] += y
            s["ipw_sum"] += y / e
        else:
            s["n_c"] += 1
            s["y_c"] += y
            s["ipw_sum"] -= y / (1 - e)

    def score(self, doc_id: str) -> UpliftScore:
        """Return causal uplift estimate for a single item."""
        s = self._stats.get(doc_id, {})
        n_t = int(s.get("n_t", 0))
        n_c = int(s.get("n_c", 0))
        y_t = float(s.get("y_t", 0))
        y_c = float(s.get("y_c", 0))

        if n_t + n_c < MIN_SAMPLES:
            # Thompson sample from Beta prior — optimistic exploration
            p1 = random.betavariate(self.prior_alpha + y_t, self.prior_beta + max(0, n_t - y_t))
            p0 = random.betavariate(self.prior_alpha + y_c, self.prior_beta + max(0, n_c - y_c))
            tau = p1 - p0
            return UpliftScore(doc_id=doc_id, incrementality_score=tau,
                               p_watch_shown=p1, p_watch_not_shown=p0,
                               method="thompson_prior", confident=False)

        p1 = y_t / n_t
        p0 = y_c / n_c
        # DR correction
        ipw = s.get("ipw_sum", 0.0) / (n_t + n_c)
        tau = (p1 - p0) * 0.7 + ipw * 0.3   # blend naive + IPW
        return UpliftScore(doc_id=doc_id, incrementality_score=tau,
                           p_watch_shown=p1, p_watch_not_shown=p0,
                           method="ipw_dr", confident=True)

    def batch_score(self, doc_ids: list[str]) -> dict[str, UpliftScore]:
        return {d: self.score(d) for d in doc_ids}

    def is_incremental(self, doc_id: str) -> bool:
        return self.score(doc_id).incrementality_score >= self.threshold


# ── Off-Policy Evaluator ──────────────────────────────────────────────────────

@dataclass
class OPEResult:
    policy_name: str
    estimated_reward: float
    variance: float
    n_samples: int
    passed: bool
    baseline_reward: float
    relative_lift: float


class OffPolicyEvaluator:
    """
    Policy Convolution OPE (Importance Sampling variant).

    Evaluates a new ranking policy against a log of (context, action, reward,
    propensity) tuples WITHOUT deploying to real users.

    Usage in Metaflow gate_check step:
        ope = OffPolicyEvaluator(min_lift=0.02)
        result = ope.evaluate(new_policy_scores, logged_events)
        if not result.passed: raise ValueError("OPE gate failed")
    """

    def __init__(self, min_lift: float = 0.02, clip_weight: float = 10.0) -> None:
        self.min_lift = min_lift
        self.clip_weight = clip_weight

    def evaluate(
        self,
        new_scores: dict[str, float],   # doc_id → new policy score
        log: list[LoggedEvent],
        baseline_reward: float = 0.0,
    ) -> OPEResult:
        if not log:
            return OPEResult("new_policy", 0.0, 0.0, 0, False, baseline_reward, 0.0)

        # Normalise new policy to a distribution
        total = sum(math.exp(v) for v in new_scores.values()) + 1e-9
        pi_new = {d: math.exp(v) / total for d, v in new_scores.items()}

        rewards, weights = [], []
        for ev in log:
            pi_b = max(1e-6, ev.propensity)
            pi_n = pi_new.get(ev.doc_id, 1e-6)
            w = min(pi_n / pi_b, self.clip_weight)   # clipped IS weight
            rewards.append(float(ev.watched) * w)
            weights.append(w)

        n = len(rewards)
        est = sum(rewards) / sum(weights) if sum(weights) > 0 else 0.0
        var = sum((r - est) ** 2 for r in rewards) / max(1, n - 1)

        if baseline_reward == 0.0:
            # estimate baseline from control arms
            ctrl = [ev for ev in log if not ev.shown]
            baseline_reward = (sum(ev.watched for ev in ctrl) / len(ctrl)) if ctrl else 0.1

        lift = (est - baseline_reward) / max(1e-9, baseline_reward)
        return OPEResult(
            policy_name="new_policy",
            estimated_reward=round(est, 4),
            variance=round(var, 6),
            n_samples=n,
            passed=lift >= self.min_lift,
            baseline_reward=round(baseline_reward, 4),
            relative_lift=round(lift, 4),
        )


# ── Feature enricher for LTR pipeline ────────────────────────────────────────

class UpliftFeatureEnricher:
    """
    Adds `incrementality_score` to the LTR feature vector so the
    LightGBM ranker can learn to trade off relevance vs. causal lift.
    """

    def __init__(self, scorer: IncrementalityScorer | None = None) -> None:
        self.scorer = scorer or IncrementalityScorer()

    def enrich(self, features: dict[str, Any], doc_id: str) -> dict[str, Any]:
        us = self.scorer.score(doc_id)
        features["incrementality_score"] = round(us.incrementality_score, 4)
        features["p_watch_shown"] = round(us.p_watch_shown, 4)
        features["p_watch_not_shown"] = round(us.p_watch_not_shown, 4)
        features["uplift_confident"] = float(us.confident)
        return features

    def enrich_batch(self, feature_rows: list[dict], doc_ids: list[str]) -> list[dict]:
        for row, doc_id in zip(feature_rows, doc_ids):
            self.enrich(row, doc_id)
        return feature_rows
