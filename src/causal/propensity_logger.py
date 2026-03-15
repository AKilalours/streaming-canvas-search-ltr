# src/causal/propensity_logger.py
"""
Real Propensity Logger — records every impression for causal inference
======================================================================
This is a REAL implementation, not a simulation.

Records EVERY recommendation impression with:
  - which item was shown (treatment)
  - which position it appeared in
  - the policy that generated it (logging policy)
  - the propensity score P(shown | context) under that policy
  - whether the user watched (outcome, recorded on playback event)

This is the foundational data required for:
  - Doubly-robust IPW uplift estimation
  - Off-policy evaluation
  - Counterfactual learning

Storage: Redis (live) + JSONL file (persistent)
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


LOG_DIR = Path("reports/propensity_logs")


@dataclass
class ImpressionEvent:
    """One recommendation impression — the atomic unit of causal logging."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    timestamp: float = field(default_factory=time.time)

    # Context
    user_id: str = ""
    session_id: str = ""
    page_type: str = ""           # "feed" | "search" | "detail_row"
    row_title: str = ""
    position: int = 0             # 0-indexed position in feed

    # Item
    doc_id: str = ""
    title: str = ""

    # Policy
    policy_name: str = "hybrid_ltr"
    ltr_score: float = 0.0
    propensity: float = 0.0       # P(shown | context, policy) — CRITICAL for IPW

    # Outcome (filled in later on watch event)
    clicked: bool = False
    watch_pct: float = 0.0        # 0.0 = not watched, 1.0 = completed
    outcome_recorded: bool = False
    outcome_ts: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WatchEvent:
    event_id: str
    doc_id: str
    user_id: str
    watch_pct: float
    timestamp: float = field(default_factory=time.time)


class PropensityCalculator:
    """
    Computes propensity scores P(item shown | context, policy).

    For a softmax ranking policy:
      P(item_i shown at position k) = softmax(score_i) * P(position k shown)

    Position exposure model: position 0 = 1.0, decays with position.
    """

    POSITION_EXPOSURE = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3]

    def compute(
        self,
        item_score: float,
        all_scores: list[float],
        position: int,
        temperature: float = 1.0,
    ) -> float:
        """
        Propensity = P(item selected) * P(position exposed)

        P(item selected) via softmax:
          softmax_i = exp(score_i / T) / sum(exp(score_j / T))
        """
        import math
        if not all_scores:
            return 0.1

        # Numerically stable softmax
        max_s = max(all_scores)
        exps = [math.exp((s - max_s) / max(temperature, 0.01)) for s in all_scores]
        total = sum(exps)

        item_idx = None
        for i, s in enumerate(all_scores):
            if abs(s - item_score) < 1e-6:
                item_idx = i
                break
        if item_idx is None:
            item_idx = 0

        p_selected = exps[item_idx] / total if total > 0 else 0.1
        p_exposed  = self.POSITION_EXPOSURE[min(position, len(self.POSITION_EXPOSURE)-1)]
        propensity = p_selected * p_exposed

        # Clip to avoid extreme weights in IPW
        return round(max(0.01, min(0.99, propensity)), 6)


class PropensityLogger:
    """
    Real propensity logger. Writes every impression to:
      1. Redis (for real-time OPE and uplift scoring)
      2. JSONL file (for offline training data)

    Usage in main.py feed/search endpoints:
        logger = PropensityLogger()
        for pos, hit in enumerate(hits):
            event = logger.log_impression(
                user_id=user_id,
                doc_id=hit.doc_id,
                ltr_score=hit.score,
                all_scores=[h.score for h in hits],
                position=pos,
                policy_name="hybrid_ltr",
            )

        # Later, when user watches:
        logger.record_outcome(event_id, watch_pct=0.85)
    """

    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._log_path = LOG_DIR / "impressions.jsonl"
        self._outcome_path = LOG_DIR / "outcomes.jsonl"
        self._calculator = PropensityCalculator()
        self._redis = self._connect_redis()
        self._buffer: list[ImpressionEvent] = []
        self._buffer_size = 50

    def _connect_redis(self):
        try:
            import redis
            r = redis.Redis.from_url(
                os.environ.get("REDIS_URL", "redis://redis:6379/0"),
                decode_responses=True, socket_timeout=1,
            )
            r.ping()
            return r
        except Exception:
            return None

    def log_impression(
        self,
        user_id: str,
        doc_id: str,
        ltr_score: float,
        all_scores: list[float],
        position: int,
        policy_name: str = "hybrid_ltr",
        session_id: str = "",
        page_type: str = "feed",
        row_title: str = "",
        title: str = "",
    ) -> ImpressionEvent:
        propensity = self._calculator.compute(ltr_score, all_scores, position)
        event = ImpressionEvent(
            user_id=user_id,
            session_id=session_id or f"sess_{user_id}_{int(time.time()//300)}",
            page_type=page_type,
            row_title=row_title,
            position=position,
            doc_id=doc_id,
            title=title,
            policy_name=policy_name,
            ltr_score=round(ltr_score, 6),
            propensity=propensity,
        )

        self._write(event)
        return event

    def record_outcome(
        self,
        event_id: str,
        watch_pct: float,
        clicked: bool = True,
    ) -> None:
        outcome = {
            "event_id": event_id,
            "watch_pct": round(watch_pct, 4),
            "clicked": clicked,
            "outcome_ts": time.time(),
        }
        with self._outcome_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(outcome) + "\n")

        # Update Redis
        if self._redis:
            try:
                key = f"outcome:{event_id}"
                self._redis.setex(key, 86400 * 7, json.dumps(outcome))
            except Exception:
                pass

    def _write(self, event: ImpressionEvent) -> None:
        line = json.dumps(event.to_dict())
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self._redis:
            try:
                key = f"impression:{event.event_id}"
                self._redis.setex(key, 86400 * 7, line)
                # Push to user stream
                self._redis.lpush(f"user_impressions:{event.user_id}", event.event_id)
                self._redis.ltrim(f"user_impressions:{event.user_id}", 0, 999)
            except Exception:
                pass

    def load_events_for_ope(
        self,
        n: int = 10000,
        min_propensity: float = 0.01,
    ) -> list[dict]:
        """
        Load logged events for OPE / uplift training.
        Joins impressions with outcomes. Returns list of
        {doc_id, shown, watched, propensity, user_id} dicts.
        """
        if not self._log_path.exists():
            return []

        # Load outcomes
        outcomes: dict[str, dict] = {}
        if self._outcome_path.exists():
            for line in self._outcome_path.read_text().splitlines():
                try:
                    o = json.loads(line)
                    outcomes[o["event_id"]] = o
                except Exception:
                    pass

        events = []
        lines = self._log_path.read_text().splitlines()
        for line in lines[-n:]:
            try:
                ev = json.loads(line)
                if ev.get("propensity", 0) < min_propensity:
                    continue
                outcome = outcomes.get(ev["event_id"], {})
                events.append({
                    "doc_id": ev["doc_id"],
                    "user_id": ev["user_id"],
                    "shown": True,
                    "watched": outcome.get("watch_pct", 0) >= 0.25,
                    "watch_pct": outcome.get("watch_pct", 0.0),
                    "propensity": ev["propensity"],
                    "position": ev["position"],
                    "policy_name": ev["policy_name"],
                    "timestamp": ev["timestamp"],
                })
            except Exception:
                pass

        return events

    def stats(self) -> dict[str, Any]:
        n_impressions = 0
        n_outcomes = 0
        if self._log_path.exists():
            n_impressions = sum(1 for _ in self._log_path.open())
        if self._outcome_path.exists():
            n_outcomes = sum(1 for _ in self._outcome_path.open())
        return {
            "n_impressions": n_impressions,
            "n_outcomes": n_outcomes,
            "outcome_rate": round(n_outcomes / max(1, n_impressions), 4),
            "log_path": str(self._log_path),
            "redis_connected": self._redis is not None,
        }
