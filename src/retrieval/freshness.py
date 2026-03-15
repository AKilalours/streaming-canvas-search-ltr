# src/retrieval/freshness.py
"""
Netflix-grade Freshness & Live-Aware Retrieval
================================================
Resolves Gap #8: "not solving live, launch, and freshness properly"

Components:
  FreshnessScorer     - decay-based freshness signal for new content
  LaunchTracker       - monitors title launch health (discovery within 72h)
  LiveEventRanker     - real-time ranking boost for live events
  AvailabilityFilter  - market/rights/device/maturity filtering BEFORE rank
  FreshnessAwareMerger - integrates freshness into hybrid retrieval

Netflix standard: freshness is a first-class ranking signal, not a post-filter.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


LAUNCH_WINDOW_HOURS = 72      # "new" titles get boosted for 72h
LIVE_BOOST = 5.0              # multiplier for live content during event window
FRESHNESS_HALF_LIFE_DAYS = 30  # freshness decays with 30-day half-life


class ContentState(str, Enum):
    LIVE        = "live"         # currently streaming live
    LAUNCHING   = "launching"    # within launch window
    RECENT      = "recent"       # < 30 days old
    CATALOG     = "catalog"      # standard catalog
    EXPIRING    = "expiring"     # rights expiring soon (urgency signal)


@dataclass
class ContentMetadata:
    doc_id: str
    title: str
    release_ts: float | None = None     # unix epoch of release/launch
    live_start_ts: float | None = None  # unix epoch of live event start
    live_end_ts: float | None = None    # unix epoch of live event end
    available_markets: set[str] = field(default_factory=set)
    maturity_rating: str = "PG-13"
    available_plans: set[str] = field(default_factory=set)  # {"standard", "ads", "premium"}
    rights_expiry_ts: float | None = None
    content_type: str = "film"          # "film" | "series" | "live" | "podcast" | "game"


@dataclass
class FreshnessSignal:
    doc_id: str
    state: ContentState
    freshness_score: float    # 0-1, higher = fresher
    live_boost: float         # multiplier if live
    urgency_boost: float      # multiplier if expiring soon
    hours_since_launch: float | None = None


# ── Freshness Scorer ──────────────────────────────────────────────────────────

class FreshnessScorer:
    """
    Computes freshness signal with exponential decay.

    freshness(t) = 0.5 ^ (age_days / HALF_LIFE) for catalog
    Launch window: flat 1.0 for first 72h
    Live: LIVE_BOOST multiplier during event
    """

    def score(self, meta: ContentMetadata) -> FreshnessSignal:
        now = time.time()
        state = ContentState.CATALOG
        freshness = 0.1
        live_boost = 1.0
        urgency_boost = 1.0
        hours_since_launch = None

        # Live content
        if meta.live_start_ts and meta.live_end_ts:
            if meta.live_start_ts <= now <= meta.live_end_ts:
                state = ContentState.LIVE
                freshness = 1.0
                live_boost = LIVE_BOOST

        # Launch window
        elif meta.release_ts:
            age_hours = (now - meta.release_ts) / 3600
            hours_since_launch = age_hours
            age_days = age_hours / 24

            if age_hours <= LAUNCH_WINDOW_HOURS:
                state = ContentState.LAUNCHING
                freshness = 1.0 - (age_hours / LAUNCH_WINDOW_HOURS) * 0.2
            elif age_days <= 30:
                state = ContentState.RECENT
                freshness = 0.5 ** (age_days / FRESHNESS_HALF_LIFE_DAYS) * 0.8
            else:
                state = ContentState.CATALOG
                freshness = max(0.05, 0.5 ** (age_days / FRESHNESS_HALF_LIFE_DAYS))

        # Expiry urgency
        if meta.rights_expiry_ts:
            days_until_expiry = (meta.rights_expiry_ts - now) / 86400
            if 0 < days_until_expiry <= 14:
                state = ContentState.EXPIRING
                urgency_boost = 1.0 + (1.0 - days_until_expiry / 14) * 0.5

        return FreshnessSignal(
            doc_id=meta.doc_id, state=state,
            freshness_score=round(freshness, 4),
            live_boost=live_boost, urgency_boost=urgency_boost,
            hours_since_launch=hours_since_launch,
        )

    def apply_to_score(self, base_score: float, signal: FreshnessSignal) -> float:
        """Blend freshness signal into existing retrieval/LTR score."""
        boosted = base_score * signal.live_boost * signal.urgency_boost
        fresh_component = signal.freshness_score * 0.15
        return boosted + fresh_component


# ── Launch Tracker ────────────────────────────────────────────────────────────

@dataclass
class LaunchHealthReport:
    doc_id: str
    title: str
    hours_since_launch: float
    impressions: int
    clicks: int
    ctr: float
    p25_watches: int          # users who watched >= 25%
    target_ctr: float
    health_status: str        # "healthy" | "at_risk" | "failing"
    recommendation: str


class LaunchTracker:
    """
    Monitors title discovery health within the launch window.
    Netflix standard: a title should hit minimum CTR within 24h of launch.
    If not, surface in more rows and lower the relevance threshold.
    """

    MIN_CTR_24H = 0.02      # 2% CTR minimum at 24h
    MIN_CTR_72H = 0.05      # 5% CTR minimum at 72h

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def record_impression(self, doc_id: str) -> None:
        self._data.setdefault(doc_id, {"impressions": 0, "clicks": 0, "p25": 0})
        self._data[doc_id]["impressions"] += 1

    def record_click(self, doc_id: str) -> None:
        self._data.setdefault(doc_id, {"impressions": 0, "clicks": 0, "p25": 0})
        self._data[doc_id]["clicks"] += 1

    def report(self, meta: ContentMetadata) -> LaunchHealthReport:
        data = self._data.get(meta.doc_id, {"impressions": 0, "clicks": 0, "p25": 0})
        now = time.time()
        hours = (now - (meta.release_ts or now)) / 3600
        impressions = data["impressions"]
        clicks = data["clicks"]
        ctr = clicks / max(1, impressions)
        target = self.MIN_CTR_24H if hours <= 24 else self.MIN_CTR_72H

        if ctr >= target:
            status = "healthy"
            rec = "No action needed."
        elif ctr >= target * 0.5:
            status = "at_risk"
            rec = f"Increase row placements and lower relevance threshold for 12h."
        else:
            status = "failing"
            rec = f"Emergency surface: add to hero row, trending row, and reduce K threshold to 50."

        return LaunchHealthReport(
            doc_id=meta.doc_id, title=meta.title,
            hours_since_launch=round(hours, 1),
            impressions=impressions, clicks=clicks,
            ctr=round(ctr, 4), p25_watches=data["p25"],
            target_ctr=target, health_status=status, recommendation=rec,
        )


# ── Availability Filter ───────────────────────────────────────────────────────

class AvailabilityFilter:
    """
    Hard filter applied BEFORE ranking to remove unavailable content.
    Resolves Gap #2: "availability constraints by market, rights window, device, plan"

    This must run before retrieval results are scored — not as a post-filter —
    to avoid wasting compute on unavailable content.
    """

    def filter(
        self,
        candidates: list[dict[str, Any]],
        metadata: dict[str, ContentMetadata],
        user_market: str = "US",
        user_plan: str = "standard",
        user_maturity: str = "PG-13",
    ) -> list[dict[str, Any]]:
        MATURITY_ORDER = {"G": 0, "PG": 1, "PG-13": 2, "R": 3, "NC-17": 4, "TV-MA": 4}
        user_level = MATURITY_ORDER.get(user_maturity, 2)
        passed = []

        for item in candidates:
            doc_id = item.get("doc_id", "")
            meta = metadata.get(doc_id)

            if meta is None:
                passed.append(item)   # no metadata = assume available
                continue

            # Market check
            if meta.available_markets and user_market not in meta.available_markets:
                continue

            # Plan check
            if meta.available_plans and user_plan not in meta.available_plans:
                continue

            # Maturity check
            content_level = MATURITY_ORDER.get(meta.maturity_rating, 2)
            if content_level > user_level:
                continue

            # Rights expiry (expired = unavailable)
            if meta.rights_expiry_ts and meta.rights_expiry_ts < time.time():
                continue

            passed.append(item)

        return passed


# ── Multi-format Content Support ──────────────────────────────────────────────

class MultiFormatRanker:
    """
    Resolves Gap #10: "Games, podcasts, and multi-format content are not solved"

    Handles cross-format retrieval by normalising engagement semantics:
      - Film/Series: watch time, completion rate
      - Podcast: listen completion, episode return rate
      - Game: session length, session return rate, achievement rate
      - Live: concurrent viewers, replay rate
    """

    FORMAT_WEIGHTS = {
        "film":    {"completion": 0.5, "return": 0.3, "social": 0.2},
        "series":  {"completion": 0.3, "return": 0.5, "social": 0.2},
        "podcast": {"completion": 0.4, "return": 0.4, "social": 0.2},
        "game":    {"completion": 0.2, "return": 0.5, "social": 0.3},
        "live":    {"completion": 0.3, "return": 0.3, "social": 0.4},
    }

    def normalise_engagement(
        self,
        raw_engagement: dict[str, float],
        content_type: str,
    ) -> float:
        """
        Converts format-specific engagement metrics to a universal 0-1 score.
        Allows cross-format ranking in shared rows.
        """
        weights = self.FORMAT_WEIGHTS.get(content_type, self.FORMAT_WEIGHTS["film"])
        score = 0.0
        for metric, weight in weights.items():
            score += raw_engagement.get(metric, 0.5) * weight
        return min(1.0, max(0.0, score))

    def cross_format_score(
        self,
        ltr_score: float,
        engagement_score: float,
        content_type: str,
        user_format_preference: dict[str, float] | None = None,
    ) -> float:
        """
        Final cross-format ranking score.
        Boosts content types the user has engaged with before.
        """
        base = 0.7 * ltr_score + 0.3 * engagement_score
        if user_format_preference:
            pref = user_format_preference.get(content_type, 0.5)
            base = base * (0.8 + 0.4 * pref)
        return round(base, 4)
