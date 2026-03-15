# src/retrieval/cross_format.py
"""
Cross-Format Ranking Support
==============================
Unified content schema and ranking across:
  film | series | live_event | podcast | game

Honest claim: "cross-format ranking support exists;
production delivery for those formats is out of scope."
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContentFormat(str, Enum):
    FILM        = "film"
    SERIES      = "series"
    LIVE_EVENT  = "live_event"
    PODCAST     = "podcast"
    GAME        = "game"


# Format-specific reward weights
FORMAT_REWARD_WEIGHTS = {
    ContentFormat.FILM:       {"completion": 1.5, "rewatch": 1.2, "share": 0.8},
    ContentFormat.SERIES:     {"completion": 2.0, "binge_next": 2.5, "return_7d": 2.0},
    ContentFormat.LIVE_EVENT: {"concurrent_viewers": 3.0, "chat_activity": 1.5},
    ContentFormat.PODCAST:    {"listen_pct": 1.0, "subscribe": 2.0, "share": 1.5},
    ContentFormat.GAME:       {"session_length": 1.8, "return_daily": 2.2},
}

# Surface-aware ranking: what goes on which row
SURFACE_FORMAT_MIX = {
    "home":        {ContentFormat.FILM: 0.5, ContentFormat.SERIES: 0.35,
                    ContentFormat.LIVE_EVENT: 0.1, ContentFormat.PODCAST: 0.05},
    "new_popular": {ContentFormat.FILM: 0.4, ContentFormat.SERIES: 0.4,
                    ContentFormat.LIVE_EVENT: 0.15, ContentFormat.PODCAST: 0.05},
    "live":        {ContentFormat.LIVE_EVENT: 0.8, ContentFormat.FILM: 0.2},
    "games":       {ContentFormat.GAME: 0.9, ContentFormat.FILM: 0.1},
}

# Intent-format affinity
INTENT_FORMAT_BOOST = {
    "background_noise":   {ContentFormat.SERIES: 1.3, ContentFormat.PODCAST: 1.2},
    "dedicated_viewing":  {ContentFormat.FILM: 1.4, ContentFormat.SERIES: 1.1},
    "social_watching":    {ContentFormat.LIVE_EVENT: 1.8, ContentFormat.FILM: 1.2},
    "discovery_mode":     {ContentFormat.PODCAST: 1.3, ContentFormat.GAME: 1.2},
}


@dataclass
class UnifiedContent:
    """Unified content item across all formats."""
    doc_id: str
    title: str
    format: ContentFormat
    genres: list[str] = field(default_factory=list)
    score: float = 0.0
    # Format-specific fields
    episode_count: int | None = None      # series
    live_start_time: float | None = None  # live_event
    episode_duration_min: int | None = None  # podcast
    game_platform: str | None = None      # game


class CrossFormatRanker:
    """
    Ranks content across all formats for a given surface and intent.
    """

    def rerank(
        self,
        items: list[dict[str, Any]],
        surface: str = "home",
        intent: str | None = None,
        alpha: float = 0.1,
    ) -> list[dict[str, Any]]:
        mix = SURFACE_FORMAT_MIX.get(surface, SURFACE_FORMAT_MIX["home"])
        intent_boosts = INTENT_FORMAT_BOOST.get(intent or "", {}) if intent else {}

        for item in items:
            fmt_str = item.get("content_type", "film")
            try:
                fmt = ContentFormat(fmt_str)
            except ValueError:
                fmt = ContentFormat.FILM

            # Surface mix boost
            surface_boost = mix.get(fmt, 0.1) * alpha
            # Intent boost
            intent_boost = intent_boosts.get(fmt, 1.0) * alpha * 0.5

            item["score"] = round(
                float(item.get("score", 0)) * (1 + surface_boost + intent_boost), 4
            )
            item["format_boost"] = round(surface_boost + intent_boost, 4)
            item["content_format"] = fmt.value

        return sorted(items, key=lambda x: -x.get("score", 0))

    def mixed_format_row(
        self,
        items_by_format: dict[str, list[dict[str, Any]]],
        surface: str = "home",
        k: int = 12,
    ) -> list[dict[str, Any]]:
        """Build a mixed-format row respecting surface mix ratios."""
        mix = SURFACE_FORMAT_MIX.get(surface, SURFACE_FORMAT_MIX["home"])
        result = []
        for fmt, ratio in sorted(mix.items(), key=lambda x: -x[1]):
            n = max(1, int(k * ratio))
            bucket = items_by_format.get(fmt.value, [])
            result.extend(bucket[:n])
        return result[:k]
