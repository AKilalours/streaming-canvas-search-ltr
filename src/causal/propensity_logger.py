# src/causal/propensity_logger.py
"""
Propensity/event logger
======================

Records recommendation impressions with:
  - which item was shown (treatment)
  - which position it appeared in
  - the policy that generated it (logging policy)
  - an exact propensity score supplied by the serving randomizer, if available
  - whether the user watched (outcome, recorded on playback event)

This is the foundational data required for:
  - diagnostic per-decision SNIPS when exact randomized propensities exist

This logger does not infer action probabilities or random assignment. The feed
serving layer supplies the exact action distribution and marks verified events;
the joined log supports diagnostic OPE, not policy promotion.

Storage: Redis (live) + JSONL file (persistent)
"""

from __future__ import annotations

import json
import math
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
    page_type: str = ""  # "feed" | "search" | "detail_row"
    row_title: str = ""
    position: int = 0  # 0-indexed position in feed

    # Item
    doc_id: str = ""
    title: str = ""

    # Policy
    policy_name: str = "hybrid_ltr"
    ltr_score: float = 0.0
    propensity: float | None = None  # exact P(action | context, logging policy)
    randomized: bool = False
    serving_verified: bool = False
    ope_eligible: bool = False
    context: dict[str, Any] = field(default_factory=dict)

    # Outcome (filled in later on watch event)
    clicked: bool = False
    watch_pct: float = 0.0  # 0.0 = not watched, 1.0 = completed
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
    """Validate probabilities supplied by the actual serving policy.

    A deterministic score or an assumed position exposure curve is not an
    action propensity. This class intentionally does not infer probabilities.
    """

    @staticmethod
    def validate(selection_probability: float | None) -> float | None:
        if selection_probability is None:
            return None
        probability = float(selection_probability)
        if not 0 < probability <= 1:
            raise ValueError("selection_probability must be in (0, 1]")
        return probability


class PropensityLogger:
    """
    Stores every impression in JSONL and, when available, Redis. Only events
    with exact supplied propensity, explicit randomized assignment, and a
    recorded outcome are included in ``load_events_for_ope``.
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
                decode_responses=True,
                socket_timeout=1,
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
        all_scores: list[float] | None = None,
        position: int = 0,
        propensity: float | None = None,
        randomized: bool = False,
        serving_verified: bool = False,
        policy_name: str = "hybrid_ltr",
        session_id: str = "",
        page_type: str = "feed",
        row_title: str = "",
        title: str = "",
        event_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ImpressionEvent:
        if not user_id or not doc_id or len(user_id) > 128 or len(doc_id) > 128:
            raise ValueError("user_id and doc_id are required and must be at most 128 characters")
        if position < 0 or not math.isfinite(float(ltr_score)):
            raise ValueError("position must be non-negative and ltr_score finite")
        propensity = self._calculator.validate(propensity)
        if context is not None:
            try:
                if len(json.dumps(context, allow_nan=False)) > 100_000:
                    raise ValueError("context must be at most 100 KB")
            except (TypeError, ValueError) as exc:
                raise ValueError("context must be JSON-serializable and finite") from exc
        event = ImpressionEvent(
            event_id=event_id or str(uuid.uuid4())[:16],
            user_id=user_id,
            session_id=session_id or f"sess_{user_id}_{int(time.time() // 300)}",
            page_type=page_type,
            row_title=row_title,
            position=position,
            doc_id=doc_id,
            title=title,
            policy_name=policy_name,
            ltr_score=round(ltr_score, 6),
            propensity=propensity,
            randomized=bool(randomized),
            serving_verified=bool(serving_verified),
            ope_eligible=bool(randomized and serving_verified and propensity is not None),
            context=context or {},
        )

        self._write(event)
        return event

    def record_outcome(
        self,
        event_id: str,
        watch_pct: float,
        clicked: bool = True,
    ) -> None:
        watch_pct = float(watch_pct)
        if not math.isfinite(watch_pct) or not 0.0 <= watch_pct <= 1.0:
            raise ValueError("watch_pct must be in [0, 1]")
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
                propensity = ev.get("propensity")
                if (
                    not ev.get("ope_eligible")
                    or propensity is None
                    or not ev.get("serving_verified")
                    or not ev.get("randomized")
                    or propensity < min_propensity
                    or not 0 < propensity <= 1
                ):
                    continue
                outcome = outcomes.get(ev["event_id"], {})
                if not outcome:
                    continue
                context = ev.get("context", {})
                behavior = (
                    context.get("behavior_probabilities", {}) if isinstance(context, dict) else {}
                )
                target_scores = (
                    context.get("target_scores", {}) if isinstance(context, dict) else {}
                )
                selected_probability = (
                    behavior.get(ev.get("doc_id")) if isinstance(behavior, dict) else None
                )
                if (
                    not isinstance(selected_probability, (int, float))
                    or not isinstance(target_scores, dict)
                    or ev.get("doc_id") not in target_scores
                    or not math.isclose(
                        float(selected_probability), float(propensity), rel_tol=1e-8
                    )
                    or not math.isclose(sum(map(float, behavior.values())), 1.0, rel_tol=1e-6)
                ):
                    continue
                events.append(
                    {
                        "event_id": ev["event_id"],
                        "doc_id": ev["doc_id"],
                        "user_id": ev["user_id"],
                        "shown": True,
                        "watched": float(outcome.get("watch_pct", 0.0)),
                        "clicked": bool(outcome.get("clicked", False)),
                        "watch_pct": outcome.get("watch_pct", 0.0),
                        "propensity": propensity,
                        "position": ev["position"],
                        "policy_name": ev["policy_name"],
                        "timestamp": ev["timestamp"],
                        "context": context,
                    }
                )
            except Exception:
                pass

        return events

    def stats(self) -> dict[str, Any]:
        n_impressions = 0
        n_outcomes = 0
        n_ope_eligible = 0
        if self._log_path.exists():
            lines = self._log_path.read_text(encoding="utf-8").splitlines()
            n_impressions = len(lines)
            for line in lines:
                try:
                    event = json.loads(line)
                    n_ope_eligible += int(bool(event.get("ope_eligible")))
                except (json.JSONDecodeError, TypeError):
                    continue
        if self._outcome_path.exists():
            n_outcomes = sum(1 for _ in self._outcome_path.open(encoding="utf-8"))
        return {
            "n_impressions": n_impressions,
            "n_outcomes": n_outcomes,
            "n_ope_eligible_impressions": n_ope_eligible,
            "outcome_rate": round(n_outcomes / max(1, n_impressions), 4),
            "log_path": str(self._log_path),
            "redis_connected": self._redis is not None,
        }
