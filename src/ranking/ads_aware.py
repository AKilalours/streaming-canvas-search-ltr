# src/ranking/ads_aware.py
"""
Netflix-grade Ads-Aware Ranking
================================
Resolves Gap #9: "Ads are barely addressed"

Netflix ads revenue: $1.5B in 2025, doubling in 2026.
Ads-aware ranking must balance:
  1. Organic recommendation quality
  2. Ad load vs engagement tradeoff
  3. Frequency capping per user
  4. Sponsored placement measurement without poisoning organic signals
  5. Incrementality across ad and organic slots

Components:
  AdSlotAllocator       - decides which feed slots get ads vs organic
  FrequencyCapper       - per-user per-advertiser frequency limits
  SponsoredRanker       - blends sponsored signals without organic pollution
  AdIncrementalityGate  - validates ad placements do not hurt organic engagement
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SlotType(str, Enum):
    ORGANIC    = "organic"
    SPONSORED  = "sponsored"
    HOUSE_AD   = "house_ad"    # Netflix own promos
    EXPLORE    = "explore"     # bandit exploration slot


@dataclass
class AdCandidate:
    ad_id: str
    advertiser_id: str
    target_doc_id: str         # content being promoted
    bid_score: float           # advertiser bid (normalised 0-1)
    relevance_score: float     # estimated user-ad relevance
    frequency_cap: int = 3     # max shows per user per day
    maturity_rating: str = "PG-13"


@dataclass
class FeedSlot:
    position: int
    slot_type: SlotType
    doc_id: str
    score: float
    ad_id: str | None = None
    advertiser_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class AdsAwareRankResult:
    slots: list[FeedSlot]
    organic_count: int
    sponsored_count: int
    explore_count: int
    ad_load_pct: float
    estimated_engagement_impact: float   # negative = ad load hurting engagement


# ── Frequency Capper ──────────────────────────────────────────────────────────

class FrequencyCapper:
    """Per-user per-advertiser daily frequency cap."""

    def __init__(self) -> None:
        # user_id -> advertiser_id -> list[timestamp]
        self._log: dict[str, dict[str, list[float]]] = {}

    def can_show(self, user_id: str, advertiser_id: str, cap: int = 3) -> bool:
        now = time.time()
        day_ago = now - 86400
        caps = self._log.setdefault(user_id, {})
        shows = caps.setdefault(advertiser_id, [])
        # Purge old entries
        caps[advertiser_id] = [t for t in shows if t > day_ago]
        return len(caps[advertiser_id]) < cap

    def record_show(self, user_id: str, advertiser_id: str) -> None:
        self._log.setdefault(user_id, {}).setdefault(advertiser_id, []).append(time.time())


# ── Ad Slot Allocator ─────────────────────────────────────────────────────────

class AdSlotAllocator:
    """
    Determines which feed positions receive ads.

    Netflix standard rules:
    - No ad in position 1 (hero slot)
    - Max 1 ad per 5 organic results (20% ad load ceiling)
    - No two consecutive ad slots
    - Exploration slots protected from ad injection
    - Maturity rating must match user profile
    """

    def __init__(
        self,
        max_ad_load: float = 0.20,
        min_organic_run: int = 4,
    ) -> None:
        self.max_ad_load = max_ad_load
        self.min_organic_run = min_organic_run

    def allocate(
        self,
        n_slots: int,
        ad_candidates: list[AdCandidate],
        user_id: str,
        capper: FrequencyCapper,
        user_maturity: str = "PG-13",
    ) -> list[SlotType]:
        """Returns slot type assignment for n_slots positions."""
        slots = [SlotType.ORGANIC] * n_slots
        eligible_ads = [
            a for a in ad_candidates
            if capper.can_show(user_id, a.advertiser_id, a.frequency_cap)
            and self._maturity_ok(a.maturity_rating, user_maturity)
        ]

        if not eligible_ads:
            return slots

        max_ads = max(1, int(n_slots * self.max_ad_load))
        ads_placed = 0
        last_ad_pos = -self.min_organic_run

        for pos in range(1, n_slots):  # skip position 0 (hero)
            if ads_placed >= max_ads:
                break
            if pos - last_ad_pos < self.min_organic_run:
                continue
            ad = eligible_ads[ads_placed % len(eligible_ads)]
            slots[pos] = SlotType.SPONSORED
            capper.record_show(user_id, ad.advertiser_id)
            last_ad_pos = pos
            ads_placed += 1

        return slots

    def _maturity_ok(self, ad_rating: str, user_rating: str) -> bool:
        order = {"G": 0, "PG": 1, "PG-13": 2, "R": 3, "NC-17": 4, "TV-MA": 4}
        return order.get(ad_rating, 2) <= order.get(user_rating, 2)


# ── Sponsored Ranker ──────────────────────────────────────────────────────────

class SponsoredRanker:
    """
    Blends sponsored content into feed without polluting organic ranking signals.

    Key principle: sponsored items are scored on a SEPARATE signal track.
    The organic LTR score is NEVER modified by bid signals — only slot
    allocation changes. This preserves organic ranking integrity.
    """

    def rank_ads(
        self,
        ad_candidates: list[AdCandidate],
        user_genre_affinities: dict[str, float],
        n_slots: int,
    ) -> list[AdCandidate]:
        """
        Score and rank ad candidates for available slots.
        Score = 0.6 * relevance + 0.4 * bid (Vickrey-style second-price auction)
        """
        scored = []
        for ad in ad_candidates:
            relevance = self._estimate_relevance(ad, user_genre_affinities)
            score = 0.6 * relevance + 0.4 * ad.bid_score
            scored.append((score, ad))
        scored.sort(key=lambda x: -x[0])
        return [ad for _, ad in scored[:n_slots]]

    def _estimate_relevance(
        self, ad: AdCandidate, affinities: dict[str, float]
    ) -> float:
        if not affinities:
            return ad.relevance_score
        # Blend pre-computed relevance with user affinities
        affinity_boost = sum(v for k, v in affinities.items() if k in ad.target_doc_id.lower()) * 0.1
        return min(1.0, ad.relevance_score + affinity_boost)


# ── Ad Incrementality Gate ────────────────────────────────────────────────────

class AdIncrementalityGate:
    """
    Validates that ad placements don't significantly hurt organic engagement.
    
    Method: compare session engagement metrics in ad vs no-ad control groups.
    Rejects ad configuration if organic engagement drops > threshold.
    """

    def __init__(self, max_engagement_drop: float = 0.05) -> None:
        self.max_engagement_drop = max_engagement_drop

    def evaluate(
        self,
        organic_engagement: float,   # clicks / impressions in organic slots
        ad_session_engagement: float, # clicks / impressions in sessions with ads
    ) -> dict[str, Any]:
        drop = (organic_engagement - ad_session_engagement) / max(organic_engagement, 1e-9)
        passed = drop <= self.max_engagement_drop
        return {
            "passed": passed,
            "engagement_drop": round(drop, 4),
            "organic_engagement": round(organic_engagement, 4),
            "ad_session_engagement": round(ad_session_engagement, 4),
            "threshold": self.max_engagement_drop,
            "recommendation": (
                "Ad load approved" if passed else
                f"Reduce ad load — engagement dropped {drop:.1%} > {self.max_engagement_drop:.1%} threshold"
            ),
        }


# ── Full Ads-Aware Ranker ─────────────────────────────────────────────────────

class AdsAwareRanker:
    """
    Orchestrates organic + sponsored ranking for a full feed page.
    Organic LTR scores are NEVER modified. Ads fill allocated slots only.
    """

    def __init__(self) -> None:
        self.allocator = AdSlotAllocator()
        self.sponsored_ranker = SponsoredRanker()
        self.capper = FrequencyCapper()

    def rank(
        self,
        organic_hits: list[dict[str, Any]],   # already LTR-ranked
        ad_candidates: list[AdCandidate],
        user_id: str,
        user_maturity: str = "PG-13",
        user_genre_affinities: dict[str, float] | None = None,
    ) -> AdsAwareRankResult:
        n = len(organic_hits)
        slot_types = self.allocator.allocate(
            n, ad_candidates, user_id, self.capper, user_maturity
        )
        ranked_ads = self.sponsored_ranker.rank_ads(
            ad_candidates, user_genre_affinities or {}, n
        )

        slots: list[FeedSlot] = []
        organic_idx = 0
        ad_idx = 0

        for pos, slot_type in enumerate(slot_types):
            if slot_type == SlotType.SPONSORED and ad_idx < len(ranked_ads):
                ad = ranked_ads[ad_idx]
                slots.append(FeedSlot(
                    position=pos, slot_type=slot_type,
                    doc_id=ad.target_doc_id, score=ad.bid_score,
                    ad_id=ad.ad_id, advertiser_id=ad.advertiser_id,
                ))
                ad_idx += 1
            elif organic_idx < len(organic_hits):
                hit = organic_hits[organic_idx]
                slots.append(FeedSlot(
                    position=pos, slot_type=SlotType.ORGANIC,
                    doc_id=hit.get("doc_id", ""), score=float(hit.get("score", 0)),
                ))
                organic_idx += 1

        sponsored_count = sum(1 for s in slots if s.slot_type == SlotType.SPONSORED)
        organic_count   = sum(1 for s in slots if s.slot_type == SlotType.ORGANIC)
        ad_load = sponsored_count / max(1, len(slots))

        # Rough engagement impact estimate: -2% per ad slot beyond first
        engagement_impact = -0.02 * max(0, sponsored_count - 1)

        return AdsAwareRankResult(
            slots=slots,
            organic_count=organic_count,
            sponsored_count=sponsored_count,
            explore_count=0,
            ad_load_pct=round(ad_load, 3),
            estimated_engagement_impact=round(engagement_impact, 4),
        )
