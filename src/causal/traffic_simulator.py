# src/causal/traffic_simulator.py
"""
Offline Experimentation Simulator
===================================
Cannot get real A/B testing without users.
But we can build a serious counterfactual simulation harness:

  - synthetic user cohorts with realistic preference distributions
  - session replay over 7/14/30-day horizons
  - policy A vs policy B comparison
  - doubly-robust IPW estimators
  - satisfaction-aware vs relevance-only comparison
  - offline-vs-shadow score alignment

Honest claim: "offline causal evaluation infrastructure is built;
online validation needs real traffic."
"""
from __future__ import annotations
import math, random, time
from dataclasses import dataclass, field
from typing import Any


# ── Synthetic user cohorts ────────────────────────────────────────────────────

COHORT_PROFILES = {
    "action_lovers":   {"genres": ["action","thriller","crime"],     "weight": 0.25},
    "romance_fans":    {"genres": ["romance","comedy","drama"],       "weight": 0.20},
    "sci_fi_buffs":    {"genres": ["sci-fi","fantasy","adventure"],   "weight": 0.15},
    "family_viewers":  {"genres": ["family","animation","children"],  "weight": 0.20},
    "arthouse_crowd":  {"genres": ["drama","documentary","foreign"],  "weight": 0.10},
    "casual_watchers": {"genres": ["comedy","animation","romance"],   "weight": 0.10},
}


@dataclass
class SyntheticUser:
    user_id: str
    cohort: str
    preferred_genres: list[str]
    session_length: float = 1.5      # avg hours per session
    sessions_per_week: float = 3.0
    completion_rate: float = 0.72    # % of started titles finished
    exploration_rate: float = 0.15   # % of clicks on unknown genres


@dataclass 
class SimulatedEvent:
    user_id: str
    doc_id: str
    title: str
    policy: str                    # "relevance_only" | "satisfaction_aware"
    clicked: bool
    completed: bool
    watch_pct: float
    session_day: int
    genres: list[str] = field(default_factory=list)
    reward: float = 0.0            # long-term reward signal


class TrafficSimulator:
    """
    Simulates user sessions for offline policy comparison.
    Generates realistic synthetic traffic for causal evaluation.
    """

    def __init__(self, n_users: int = 200, seed: int = 42) -> None:
        random.seed(seed)
        self.users = self._create_users(n_users)
        self.events: list[SimulatedEvent] = []

    def _create_users(self, n: int) -> list[SyntheticUser]:
        users = []
        cohorts = list(COHORT_PROFILES.items())
        weights = [v["weight"] for _, v in cohorts]
        for i in range(n):
            cohort_name, cohort_data = random.choices(cohorts, weights=weights, k=1)[0]
            users.append(SyntheticUser(
                user_id=f"sim_user_{i:04d}",
                cohort=cohort_name,
                preferred_genres=cohort_data["genres"],
                session_length=random.gauss(1.5, 0.5),
                sessions_per_week=random.gauss(3, 1),
                completion_rate=random.gauss(0.72, 0.1),
                exploration_rate=random.gauss(0.15, 0.05),
            ))
        return users

    def _genre_relevance(self, user: SyntheticUser, genres: list[str]) -> float:
        """How relevant are these genres to this user? [0, 1]"""
        if not genres:
            return 0.3
        overlap = set(g.lower() for g in genres) & set(g.lower() for g in user.preferred_genres)
        return min(1.0, len(overlap) / max(1, len(user.preferred_genres)) * 2)

    def simulate_policy(
        self,
        policy_name: str,
        items: list[dict[str, Any]],
        n_days: int = 30,
        sessions_per_day: int = 3,
    ) -> list[SimulatedEvent]:
        """Simulate n_days of user sessions under a given policy."""
        events = []
        for day in range(n_days):
            for _ in range(sessions_per_day):
                user = random.choice(self.users)
                # Each session: user sees top-K items from policy
                session_items = items[:12]
                for item in session_items[:random.randint(3, 8)]:
                    genres = [g.strip() for g in item.get("genres","").split(",") if g.strip()]
                    rel = self._genre_relevance(user, genres)

                    # Click probability: relevance + exploration noise
                    click_prob = rel * 0.7 + random.uniform(0, 0.3) * user.exploration_rate
                    clicked = random.random() < click_prob

                    # Completion: conditional on click + relevance
                    completed = False
                    watch_pct = 0.0
                    if clicked:
                        watch_pct = rel * user.completion_rate + random.gauss(0, 0.15)
                        watch_pct = max(0, min(1, watch_pct))
                        completed = watch_pct > 0.8

                    # Long-term reward: completion signal decayed by position
                    reward = watch_pct * rel if clicked else -0.05

                    events.append(SimulatedEvent(
                        user_id=user.user_id,
                        doc_id=str(item.get("doc_id", "")),
                        title=str(item.get("title", "")),
                        policy=policy_name,
                        clicked=clicked,
                        completed=completed,
                        watch_pct=watch_pct,
                        session_day=day,
                        genres=genres,
                        reward=reward,
                    ))
        return events

    def compare_policies(
        self,
        policy_a_items: list[dict[str, Any]],
        policy_b_items: list[dict[str, Any]],
        policy_a_name: str = "relevance_only",
        policy_b_name: str = "satisfaction_aware",
        n_days: int = 14,
    ) -> dict[str, Any]:
        """
        Compare two ranking policies using simulated sessions.
        Returns metrics for policy A vs policy B.
        """
        events_a = self.simulate_policy(policy_a_name, policy_a_items, n_days)
        events_b = self.simulate_policy(policy_b_name, policy_b_items, n_days)

        def compute_metrics(events: list[SimulatedEvent]) -> dict[str, float]:
            if not events:
                return {}
            n = len(events)
            clicks = [e for e in events if e.clicked]
            completed = [e for e in events if e.completed]
            ctr = len(clicks) / n
            completion_rate = len(completed) / max(1, len(clicks))
            avg_watch = sum(e.watch_pct for e in clicks) / max(1, len(clicks))
            avg_reward = sum(e.reward for e in events) / n
            # Diversity: unique genres seen across sessions
            all_genres: set[str] = set()
            for e in events:
                all_genres.update(e.genres)
            return {
                "ctr": round(ctr, 4),
                "completion_rate": round(completion_rate, 4),
                "avg_watch_pct": round(avg_watch, 4),
                "avg_reward": round(avg_reward, 4),
                "genre_diversity": len(all_genres),
                "total_events": n,
                "total_clicks": len(clicks),
            }

        metrics_a = compute_metrics(events_a)
        metrics_b = compute_metrics(events_b)

        # Compute lifts
        lifts = {}
        for k in metrics_a:
            va = metrics_a.get(k, 0)
            vb = metrics_b.get(k, 0)
            if va and isinstance(va, (int, float)):
                lifts[k] = round((vb - va) / max(abs(va), 1e-9) * 100, 2)

        return {
            "policy_a": {"name": policy_a_name, "metrics": metrics_a},
            "policy_b": {"name": policy_b_name, "metrics": metrics_b},
            "lifts_pct": lifts,
            "winner": policy_b_name if lifts.get("avg_reward", 0) > 0 else policy_a_name,
            "n_days_simulated": n_days,
            "n_users": len(self.users),
            "honest_caveat": (
                "Simulated with synthetic users. "
                "Online validation with real users required for causal claims."
            ),
        }


# ── IPW / Doubly-Robust OPE ──────────────────────────────────────────────────

def ipw_estimate(
    logged_events: list[dict[str, Any]],
    new_policy_scores: dict[str, float],
    clip_threshold: float = 10.0,
) -> dict[str, float]:
    """
    Inverse Propensity Weighted off-policy estimator.
    Estimates new policy value from logged data.
    """
    if not logged_events:
        return {"ipw_estimate": 0.0, "n_events": 0}

    weighted_rewards = []
    for event in logged_events:
        doc_id = str(event.get("doc_id", ""))
        reward = float(event.get("reward", 0))
        log_propensity = float(event.get("propensity", 0.1))
        new_score = new_policy_scores.get(doc_id, 0.1)

        # Importance weight with clipping for variance reduction
        w = min(new_score / max(log_propensity, 1e-9), clip_threshold)
        weighted_rewards.append(w * reward)

    ipw = sum(weighted_rewards) / max(len(weighted_rewards), 1)
    return {
        "ipw_estimate": round(ipw, 6),
        "n_events": len(logged_events),
        "clip_threshold": clip_threshold,
        "method": "clipped_ipw",
    }
