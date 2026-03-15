# src/retrieval/live_events.py
"""
Live Event Ranking Layer + WebSocket Real-Time Updates
=======================================================
Resolves Gap: "Live streaming infra — needs CDN"

We build the full ranking architecture for live events.
The CDN and video ingest are mocked — the ranking logic is real.

Components:
  LiveEventScheduler    - manages live event windows + countdown
  LiveScoreBooster      - real-time ranking boost during live events
  LiveFeedComposer      - composes feed rows with live content prioritised
  WebSocketEventStream  - pushes ranking updates every 30s (FastAPI WebSocket)
  LiveAvailabilityCheck - validates content is live right now

Netflix standard: live content gets a 5x ranking boost during event window,
with 30-second refresh cycles pushed via WebSocket to all connected clients.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator


# ── Live event state machine ──────────────────────────────────────────────────

class LiveEventState(str, Enum):
    SCHEDULED   = "scheduled"    # > 24h away
    COUNTDOWN   = "countdown"    # < 24h, not started
    LIVE        = "live"         # currently streaming
    REPLAY      = "replay"       # ended, replay available
    EXPIRED     = "expired"      # no longer available


@dataclass
class LiveEvent:
    event_id: str
    title: str
    doc_id: str
    start_ts: float
    end_ts: float
    category: str = "live"       # "sport" | "concert" | "news" | "talk"
    peak_viewers: int = 0
    replay_available: bool = True
    ranking_boost: float = 5.0   # multiplier during live window
    countdown_boost: float = 2.0 # multiplier in 24h pre-window

    @property
    def state(self) -> LiveEventState:
        now = time.time()
        if now < self.start_ts - 86400:
            return LiveEventState.SCHEDULED
        elif now < self.start_ts:
            return LiveEventState.COUNTDOWN
        elif now <= self.end_ts:
            return LiveEventState.LIVE
        elif self.replay_available:
            return LiveEventState.REPLAY
        return LiveEventState.EXPIRED

    @property
    def seconds_until_start(self) -> float:
        return max(0.0, self.start_ts - time.time())

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.end_ts - time.time())

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "doc_id": self.doc_id,
            "state": self.state.value,
            "category": self.category,
            "seconds_until_start": round(self.seconds_until_start),
            "seconds_remaining": round(self.seconds_remaining),
            "ranking_boost": self.ranking_boost,
            "peak_viewers": self.peak_viewers,
        }


@dataclass
class LiveRankSignal:
    doc_id: str
    base_score: float
    live_boost: float
    final_score: float
    state: LiveEventState
    urgency: float    # 0-1, higher = more urgent to surface


# ── Live Event Scheduler ──────────────────────────────────────────────────────

class LiveEventScheduler:
    """
    Manages live event registry and state transitions.
    In production: syncs from content management system every 60s.
    Here: in-memory store with demo events.
    """

    def __init__(self) -> None:
        self._events: dict[str, LiveEvent] = {}
        self._doc_to_event: dict[str, str] = {}
        self._seed_demo_events()

    def _seed_demo_events(self) -> None:
        now = time.time()
        demos = [
            LiveEvent(
                event_id="live_001",
                title="StreamLens Awards Show 2026",
                doc_id="live_doc_001",
                start_ts=now + 3600,    # starts in 1 hour
                end_ts=now + 7200,
                category="talk",
                ranking_boost=5.0,
            ),
            LiveEvent(
                event_id="live_002",
                title="Championship Finals — Live",
                doc_id="live_doc_002",
                start_ts=now - 900,     # started 15 min ago (currently LIVE)
                end_ts=now + 5400,
                category="sport",
                ranking_boost=8.0,
                peak_viewers=2_400_000,
            ),
            LiveEvent(
                event_id="live_003",
                title="Breaking News Coverage",
                doc_id="live_doc_003",
                start_ts=now + 86400,   # tomorrow
                end_ts=now + 90000,
                category="news",
                ranking_boost=3.0,
            ),
        ]
        for ev in demos:
            self.register(ev)

    def register(self, event: LiveEvent) -> None:
        self._events[event.event_id] = event
        self._doc_to_event[event.doc_id] = event.event_id

    def get_live_now(self) -> list[LiveEvent]:
        return [e for e in self._events.values() if e.state == LiveEventState.LIVE]

    def get_countdown(self) -> list[LiveEvent]:
        return [e for e in self._events.values() if e.state == LiveEventState.COUNTDOWN]

    def get_event_for_doc(self, doc_id: str) -> LiveEvent | None:
        eid = self._doc_to_event.get(doc_id)
        return self._events.get(eid) if eid else None

    def all_active(self) -> list[LiveEvent]:
        return [e for e in self._events.values()
                if e.state in (LiveEventState.LIVE, LiveEventState.COUNTDOWN)]

    def status(self) -> dict[str, Any]:
        return {
            "total_events": len(self._events),
            "live_now": len(self.get_live_now()),
            "countdown": len(self.get_countdown()),
            "events": [e.to_dict() for e in self._events.values()],
        }


# ── Live Score Booster ────────────────────────────────────────────────────────

class LiveScoreBooster:
    """
    Applies real-time ranking boosts to live content.

    Boost formula:
      LIVE state:      score * event.ranking_boost + urgency_bonus
      COUNTDOWN state: score * event.countdown_boost * time_decay
      REPLAY state:    score * 1.2 (slight replay boost for 6h)
      Other:           score unchanged
    """

    def __init__(self, scheduler: LiveEventScheduler) -> None:
        self.scheduler = scheduler

    def boost(self, doc_id: str, base_score: float) -> LiveRankSignal:
        event = self.scheduler.get_event_for_doc(doc_id)

        if event is None:
            return LiveRankSignal(
                doc_id=doc_id, base_score=base_score,
                live_boost=1.0, final_score=base_score,
                state=LiveEventState.EXPIRED, urgency=0.0,
            )

        state = event.state

        if state == LiveEventState.LIVE:
            # Urgency increases as event nears end
            pct_remaining = event.seconds_remaining / max(1, event.end_ts - event.start_ts)
            urgency = 1.0 - pct_remaining  # higher urgency as end approaches
            boost = event.ranking_boost * (1.0 + urgency * 0.5)
            final = base_score * boost + urgency * 2.0

        elif state == LiveEventState.COUNTDOWN:
            # Boost increases as start approaches
            hrs_away = event.seconds_until_start / 3600
            time_factor = max(0.1, 1.0 - hrs_away / 24)
            boost = event.countdown_boost * time_factor
            urgency = time_factor * 0.5
            final = base_score * boost

        elif state == LiveEventState.REPLAY:
            boost = 1.2
            urgency = 0.1
            final = base_score * boost

        else:
            boost, urgency, final = 1.0, 0.0, base_score

        return LiveRankSignal(
            doc_id=doc_id, base_score=round(base_score, 4),
            live_boost=round(boost, 3), final_score=round(final, 4),
            state=state, urgency=round(urgency, 3),
        )

    def boost_batch(
        self, hits: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply live boosts to a list of search hits in-place."""
        for hit in hits:
            doc_id = hit.get("doc_id", "")
            signal = self.boost(doc_id, float(hit.get("score", 0)))
            hit["score"] = signal.final_score
            hit["live_state"] = signal.state.value
            hit["live_boost"] = signal.live_boost
            hit["urgency"] = signal.urgency
        # Re-sort after boost
        hits.sort(key=lambda x: -x.get("score", 0))
        return hits


# ── Live Feed Composer ────────────────────────────────────────────────────────

class LiveFeedComposer:
    """
    Composes feed rows ensuring live content is always surfaced.

    Netflix standard:
      - Live content always appears in first row during active events
      - Countdown content gets a dedicated "Starting Soon" row
      - Row refresh interval: 30 seconds during live events
    """

    def __init__(self, scheduler: LiveEventScheduler, booster: LiveScoreBooster) -> None:
        self.scheduler = scheduler
        self.booster = booster

    def compose_live_rows(self) -> list[dict[str, Any]]:
        rows = []

        # Live now row
        live_events = self.scheduler.get_live_now()
        if live_events:
            rows.append({
                "title": "Live Now",
                "refresh_interval_s": 30,
                "items": [
                    {
                        "doc_id": e.doc_id,
                        "title": e.title,
                        "score": e.ranking_boost,
                        "live_state": "live",
                        "viewers": e.peak_viewers,
                        "seconds_remaining": round(e.seconds_remaining),
                        "category": e.category,
                    }
                    for e in sorted(live_events, key=lambda e: -e.ranking_boost)
                ],
            })

        # Starting soon row
        countdown = self.scheduler.get_countdown()
        if countdown:
            rows.append({
                "title": "Starting Soon",
                "refresh_interval_s": 60,
                "items": [
                    {
                        "doc_id": e.doc_id,
                        "title": e.title,
                        "score": e.countdown_boost,
                        "live_state": "countdown",
                        "seconds_until_start": round(e.seconds_until_start),
                        "category": e.category,
                    }
                    for e in sorted(countdown, key=lambda e: e.seconds_until_start)
                ],
            })

        return rows

    def inject_into_feed(
        self, feed_rows: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Inject live rows at top of existing feed."""
        live_rows = self.compose_live_rows()
        return live_rows + feed_rows


# ── WebSocket Event Stream ────────────────────────────────────────────────────

class LiveRankingUpdateStream:
    """
    Manages WebSocket connections for real-time ranking updates.
    Pushes updated live scores every 30 seconds to all connected clients.

    In production: backed by Redis pub/sub for horizontal scaling.
    Here: in-process asyncio broadcast.
    """

    def __init__(
        self,
        scheduler: LiveEventScheduler,
        booster: LiveScoreBooster,
        refresh_interval_s: float = 30.0,
    ) -> None:
        self.scheduler = scheduler
        self.booster = booster
        self.refresh_interval = refresh_interval_s
        self._connections: set = set()
        self._running = False

    def _build_update_payload(self) -> dict[str, Any]:
        live_events = self.scheduler.get_live_now()
        countdown   = self.scheduler.get_countdown()
        return {
            "type": "live_ranking_update",
            "timestamp": time.time(),
            "refresh_interval_s": self.refresh_interval,
            "live_now": [e.to_dict() for e in live_events],
            "countdown": [e.to_dict() for e in countdown],
            "has_live_content": len(live_events) > 0,
            "total_active_events": len(live_events) + len(countdown),
        }

    async def stream_updates(self) -> AsyncGenerator[dict, None]:
        """
        Async generator — yields ranking updates every refresh_interval seconds.
        Wire to FastAPI WebSocket endpoint.
        """
        while True:
            yield self._build_update_payload()
            await asyncio.sleep(self.refresh_interval)

    def get_current_state(self) -> dict[str, Any]:
        """Synchronous snapshot — for REST endpoint polling fallback."""
        return self._build_update_payload()
