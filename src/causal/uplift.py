# src/causal/uplift.py
"""
Incrementality and off-policy diagnostics built from logged randomized data.

The estimator implemented here is clipped, self-normalized inverse propensity
weighting (Hajek), not doubly robust. A doubly robust estimator requires an
outcome model that this module does not fit. Estimates are marked uncertain
until both arms have adequate effective sample size and propensity overlap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

UPLIFT_THRESHOLD = 0.05   # minimum causal lift to justify a slot


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class LoggedEvent:
    """One row of historical bandit log."""
    doc_id: str
    shown: bool           # was this item shown (treatment) or not (control)?
    watched: float | bool # bounded observed reward, usually normalized watch time
    propensity: float     # P(shown) under the logging policy — needed for IPW
    user_id: str = ""
    context: dict = field(default_factory=dict)


@dataclass
class UpliftScore:
    doc_id: str
    incrementality_score: float   # tau-hat: E[Y(1) - Y(0)]
    p_watch_shown: float          # E[Y | T=1]
    p_watch_not_shown: float      # E[Y | T=0]
    method: str = "clipped_ipw_hajek"
    confident: bool = False
    n_treated: int = 0
    n_control: int = 0
    effective_sample_size_treated: float = 0.0
    effective_sample_size_control: float = 0.0


# ── Core estimator ────────────────────────────────────────────────────────────

MIN_SAMPLES_PER_ARM = 10


class IncrementalityScorer:
    """Per-item clipped Hajek estimates with explicit overlap diagnostics."""

    def __init__(
        self,
        threshold: float = UPLIFT_THRESHOLD,
        prior_alpha: float = 1.0,
        prior_beta: float = 4.0,
        max_inverse_weight: float = 20.0,
    ) -> None:
        if not math.isfinite(threshold) or not math.isfinite(prior_alpha) \
                or not math.isfinite(prior_beta) or not math.isfinite(max_inverse_weight):
            raise ValueError("uplift configuration must be finite")
        if prior_alpha <= 0 or prior_beta <= 0 or max_inverse_weight < 1:
            raise ValueError("prior parameters must be positive and max_inverse_weight >= 1")
        self.threshold = threshold
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.max_inverse_weight = max_inverse_weight
        # item_id -> arm counts, capped weighted outcomes, weight sums/squares,
        # and the number of observations outside the conservative overlap band.
        self._stats: dict[str, dict[str, float]] = {}

    def update(self, event: LoggedEvent) -> None:
        """Online update from a logged event."""
        if not math.isfinite(float(event.propensity)) or not 0 < event.propensity < 1:
            raise ValueError("propensity must be finite and strictly between 0 and 1")
        s = self._stats.setdefault(event.doc_id, {
            "n_t": 0, "n_c": 0, "wy_t": 0.0, "wy_c": 0.0,
            "w_t": 0.0, "w_c": 0.0, "w2_t": 0.0, "w2_c": 0.0,
            "overlap_violations": 0,
        })
        y = float(event.watched)
        e = float(event.propensity)
        s["overlap_violations"] += int(e < 0.05 or e > 0.95)
        if event.shown:
            s["n_t"] += 1
            weight = min(self.max_inverse_weight, 1.0 / e)
            s["wy_t"] += weight * y
            s["w_t"] += weight
            s["w2_t"] += weight * weight
        else:
            s["n_c"] += 1
            weight = min(self.max_inverse_weight, 1.0 / (1.0 - e))
            s["wy_c"] += weight * y
            s["w_c"] += weight
            s["w2_c"] += weight * weight

    def score(self, doc_id: str) -> UpliftScore:
        """Return causal uplift estimate for a single item."""
        s = self._stats.get(doc_id, {})
        n_t = int(s.get("n_t", 0))
        n_c = int(s.get("n_c", 0))
        w_t, w_c = float(s.get("w_t", 0.0)), float(s.get("w_c", 0.0))
        prior_strength = self.prior_alpha + self.prior_beta
        prior_mean = self.prior_alpha / prior_strength
        p1 = (float(s.get("wy_t", 0.0)) + self.prior_alpha) / (w_t + prior_strength)
        p0 = (float(s.get("wy_c", 0.0)) + self.prior_alpha) / (w_c + prior_strength)
        ess_t = w_t * w_t / max(float(s.get("w2_t", 0.0)), 1e-12)
        ess_c = w_c * w_c / max(float(s.get("w2_c", 0.0)), 1e-12)
        overlap_ok = int(s.get("overlap_violations", 0)) == 0
        confident = (
            n_t >= MIN_SAMPLES_PER_ARM
            and n_c >= MIN_SAMPLES_PER_ARM
            and ess_t >= MIN_SAMPLES_PER_ARM
            and ess_c >= MIN_SAMPLES_PER_ARM
            and overlap_ok
        )
        # A symmetric prior has exactly zero estimated lift before evidence;
        # it must not inject a new random pseudo-outcome on every request.
        if n_t + n_c == 0:
            p1 = p0 = prior_mean
        return UpliftScore(
            doc_id=doc_id,
            incrementality_score=p1 - p0,
            p_watch_shown=p1,
            p_watch_not_shown=p0,
            method="clipped_ipw_hajek",
            confident=confident,
            n_treated=n_t,
            n_control=n_c,
            effective_sample_size_treated=ess_t,
            effective_sample_size_control=ess_c,
        )

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
    effective_sample_size: float = 0.0
    weight_clipping_fraction: float = 0.0
    standard_error: float = 0.0
    estimator: str = "snips"


class OffPolicyEvaluator:
    """Self-normalized IPS for observed actions with known logging propensities.

    This estimator requires an exact behavior-policy action probability and
    support for the target policy. Its result is a diagnostic, not a promotion
    decision. It does not become doubly robust by supplying a baseline mean.
    """

    def __init__(
        self,
        min_lift: float = 0.02,
        clip_weight: float = 10.0,
        min_effective_sample_size: float = 30.0,
    ) -> None:
        if not all(math.isfinite(v) for v in (min_lift, clip_weight, min_effective_sample_size)):
            raise ValueError("OPE configuration must be finite")
        if clip_weight < 1 or min_effective_sample_size < 1:
            raise ValueError("clip_weight and min_effective_sample_size must be >= 1")
        self.min_lift = min_lift
        self.clip_weight = clip_weight
        self.min_effective_sample_size = min_effective_sample_size

    def evaluate(
        self,
        new_scores: dict[str, float] | None,   # global target scores, or per-event context scores
        log: list[LoggedEvent],
        baseline_reward: float | None = None,
    ) -> OPEResult:
        new_scores = new_scores or {}
        if any(not math.isfinite(float(v)) for v in new_scores.values()):
            raise ValueError("new_scores must be finite")
        shown = [event for event in log if event.shown]
        for event in shown:
            if not math.isfinite(float(event.propensity)) or not 0 < event.propensity <= 1:
                raise ValueError("logged action propensities must be finite and in (0, 1]")
        if not shown:
            return OPEResult("new_policy", 0.0, 0.0, 0, False, 0.0, 0.0)
        raw_weights_list = []
        for event in shown:
            target_scores = event.context.get("target_scores") if event.context else None
            if target_scores:
                if (not isinstance(target_scores, dict)
                        or any(not math.isfinite(float(value)) for value in target_scores.values())):
                    raise ValueError("per-event target_scores must be a finite mapping")
                scores = np.asarray(list(map(float, target_scores.values())), dtype=np.float64)
                ids = list(target_scores)
            else:
                scores = np.asarray([float(value) for value in new_scores.values()], dtype=np.float64)
                ids = list(new_scores)
            if not ids:
                raise ValueError("each logged event needs target_scores or global new_scores")
            exp_scores = np.exp(np.clip(scores - np.max(scores), -60.0, 0.0))
            target_probs = exp_scores / exp_scores.sum()
            target_probability = dict(zip(ids, target_probs, strict=True))
            raw_weights_list.append(
                target_probability.get(event.doc_id, 0.0) / float(event.propensity)
            )

        rewards = np.asarray([float(event.watched) for event in shown], dtype=np.float64)
        if np.any(~np.isfinite(rewards)) or np.any((rewards < 0) | (rewards > 1)):
            raise ValueError("logged rewards must be finite and in [0, 1]")
        raw_weights = np.asarray(raw_weights_list, dtype=np.float64)
        weights = np.minimum(raw_weights, self.clip_weight)
        weight_sum = float(weights.sum())
        n = len(rewards)
        if weight_sum <= 0:
            return OPEResult("new_policy", 0.0, 0.0, n, False, 0.0, 0.0)

        normalized_weights = weights / weight_sum
        estimate = float(np.dot(normalized_weights, rewards))
        ess = float(weight_sum * weight_sum / max(float(np.dot(weights, weights)), 1e-12))
        variance = float(np.dot(normalized_weights**2, (rewards - estimate) ** 2))
        standard_error = math.sqrt(max(variance, 0.0))
        if baseline_reward is None:
            baseline_reward = float(rewards.mean())
        if not math.isfinite(float(baseline_reward)) or baseline_reward < 0:
            raise ValueError("baseline_reward must be finite and non-negative")
        lift = (estimate - baseline_reward) / max(1e-9, baseline_reward)
        clip_fraction = float(np.mean(raw_weights > self.clip_weight))
        return OPEResult(
            policy_name="new_policy",
            estimated_reward=round(estimate, 4),
            variance=round(variance, 6),
            n_samples=n,
            passed=(lift >= self.min_lift and ess >= self.min_effective_sample_size),
            baseline_reward=round(baseline_reward, 4),
            relative_lift=round(lift, 4),
            effective_sample_size=round(ess, 2),
            weight_clipping_fraction=round(clip_fraction, 4),
            standard_error=round(standard_error, 6),
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
        for row, doc_id in zip(feature_rows, doc_ids, strict=True):
            self.enrich(row, doc_id)
        return feature_rows
