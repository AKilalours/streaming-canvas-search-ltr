# src/ranking/slate_optimizer.py
"""
Page-Level Slate Optimization + Long-Term Satisfaction
=======================================================
Moves ranking from "nDCG machine" to "satisfaction machine".

Components:
  SlateOptimizer         - page-level optimization (not just item-level)
  LongTermSatisfaction   - models retention signals beyond immediate clicks
  IntentAwareReranker    - session intent × content type matching
  UncertaintyExplorer    - calibrated exploration under uncertainty

Netflix standard: optimize for 30-day retention, not just next-click CTR.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Long-term satisfaction signals ────────────────────────────────────────────

class SatisfactionSignal(str, Enum):
    COMPLETED       = "completed"      # watched >80% — strong positive
    BINGE_NEXT      = "binge_next"     # immediately watched next episode
    RETURNED_7D     = "returned_7d"    # came back within 7 days after watching
    SHARED          = "shared"         # shared/recommended to others
    ABANDONED_10PCT = "abandoned_10"   # stopped at <10% — strong negative
    THUMBS_DOWN     = "thumbs_down"    # explicit negative
    SKIP_INTRO      = "skip_intro"     # positive engagement signal
    REWATCH         = "rewatch"        # watched again — very positive


SIGNAL_WEIGHTS = {
    SatisfactionSignal.COMPLETED:       +1.5,
    SatisfactionSignal.BINGE_NEXT:      +2.0,
    SatisfactionSignal.RETURNED_7D:     +1.8,
    SatisfactionSignal.SHARED:          +1.2,
    SatisfactionSignal.ABANDONED_10PCT: -2.0,
    SatisfactionSignal.THUMBS_DOWN:     -2.5,
    SatisfactionSignal.SKIP_INTRO:      +0.4,
    SatisfactionSignal.REWATCH:         +2.2,
}


class LongTermSatisfactionModel:
    """
    Models 30-day retention signal instead of immediate CTR.
    
    score = Σ(signal_weight × recency_decay × confidence)
    
    This is the difference between:
      - CTR model: "will they click?"
      - Satisfaction model: "will this make them come back next week?"
    """

    HALF_LIFE_DAYS = 30.0

    def __init__(self) -> None:
        # user_id → list of (timestamp, signal, doc_id)
        self._log: dict[str, list[tuple[float, SatisfactionSignal, str]]] = {}

    def record(self, user_id: str, doc_id: str,
               signal: SatisfactionSignal, timestamp: float) -> None:
        self._log.setdefault(user_id, []).append((timestamp, signal, doc_id))

    def satisfaction_score(self, user_id: str, doc_id: str,
                           now: float | None = None) -> float:
        import time
        now = now or time.time()
        events = [(ts, sig, did) for ts, sig, did in self._log.get(user_id, [])
                  if did == doc_id]
        score = 0.0
        for ts, sig, _ in events:
            age_days = (now - ts) / 86400
            decay = 0.5 ** (age_days / self.HALF_LIFE_DAYS)
            score += SIGNAL_WEIGHTS[sig] * decay
        return round(score, 4)

    def user_satisfaction_profile(self, user_id: str) -> dict[str, Any]:
        import time
        now = time.time()
        events = self._log.get(user_id, [])
        if not events:
            return {"score": 0.0, "dominant_signal": None, "n_events": 0}
        total = sum(SIGNAL_WEIGHTS[sig] * (0.5 ** ((now-ts)/86400/self.HALF_LIFE_DAYS))
                    for ts, sig, _ in events)
        counts: dict[str, int] = {}
        for _, sig, _ in events:
            counts[sig.value] = counts.get(sig.value, 0) + 1
        dominant = max(counts, key=lambda k: counts[k]) if counts else None
        return {
            "score": round(total, 4),
            "dominant_signal": dominant,
            "n_events": len(events),
            "signal_counts": counts,
        }


# ── Slate (page-level) optimizer ──────────────────────────────────────────────

@dataclass
class SlateItem:
    doc_id: str
    title: str
    score: float                    # LTR score
    genres: list[str] = field(default_factory=list)
    content_type: str = "film"      # film|series|live|podcast|game
    satisfaction_score: float = 0.0
    visual_similarity: float = 0.0
    uncertainty: float = 0.5       # model uncertainty (0=certain, 1=uncertain)
    slot_type: str = "exploit"     # exploit|explore|diversity|business


@dataclass
class Slate:
    items: list[SlateItem]
    diversity_score: float
    satisfaction_estimate: float
    exploration_slots: int
    page_coherence: float   # how well the page tells a story


class SlateOptimizer:
    """
    Optimizes a full page of recommendations, not just individual items.
    
    Standard item-level ranking: score each item independently → sort.
    Slate optimization: score the PAGE as a whole → maximize joint value.
    
    Objectives balanced:
      1. Relevance     (LTR score)          weight=0.45
      2. Satisfaction  (30-day signal)       weight=0.25
      3. Diversity     (genre coverage)      weight=0.15
      4. Exploration   (uncertainty bonus)   weight=0.10
      5. Business      (originals/margins)   weight=0.05
    """

    W_RELEVANCE    = 0.45
    W_SATISFACTION = 0.25
    W_DIVERSITY    = 0.15
    W_EXPLORE      = 0.10
    W_BUSINESS     = 0.05

    def __init__(
        self,
        satisfaction_model: LongTermSatisfactionModel | None = None,
        explore_rate: float = 0.15,
    ) -> None:
        self.satisfaction = satisfaction_model or LongTermSatisfactionModel()
        self.explore_rate = explore_rate

    def optimize(
        self,
        candidates: list[dict[str, Any]],
        user_id: str,
        k: int = 12,
        surface: str = "home",
    ) -> Slate:
        """
        Build an optimized slate from candidates.
        Uses greedy marginal gain maximization.
        """
        if not candidates:
            return Slate([], 0.0, 0.0, 0, 0.0)

        items = [self._to_slate_item(c, user_id) for c in candidates]
        selected: list[SlateItem] = []
        remaining = items.copy()
        covered_genres: set[str] = set()

        while len(selected) < k and remaining:
            best_item = None
            best_score = -float("inf")

            for item in remaining:
                marginal = self._marginal_score(item, selected, covered_genres)
                if marginal > best_score:
                    best_score = marginal
                    best_item = item

            if best_item is None:
                break

            # Assign slot type
            if len(selected) < int(k * self.explore_rate):
                best_item.slot_type = "explore"
            elif len(selected) % 5 == 4:
                best_item.slot_type = "diversity"
            else:
                best_item.slot_type = "exploit"

            selected.append(best_item)
            covered_genres.update(best_item.genres)
            remaining.remove(best_item)

        diversity = self._diversity_score(selected)
        sat_est   = sum(i.satisfaction_score for i in selected) / max(1, len(selected))
        coherence = self._coherence_score(selected)
        explore_n = sum(1 for i in selected if i.slot_type == "explore")

        return Slate(
            items=selected,
            diversity_score=round(diversity, 4),
            satisfaction_estimate=round(sat_est, 4),
            exploration_slots=explore_n,
            page_coherence=round(coherence, 4),
        )

    def _to_slate_item(self, c: dict[str, Any], user_id: str) -> SlateItem:
        doc_id = str(c.get("doc_id", ""))
        text = str(c.get("text", ""))
        genres = self._parse_genres(text)
        sat = self.satisfaction.satisfaction_score(user_id, doc_id)
        return SlateItem(
            doc_id=doc_id,
            title=str(c.get("title", "")),
            score=float(c.get("score", 0)),
            genres=genres,
            content_type=str(c.get("content_type", "film")),
            satisfaction_score=sat,
            uncertainty=float(c.get("uncertainty", 0.5)),
        )

    def _marginal_score(
        self,
        item: SlateItem,
        selected: list[SlateItem],
        covered: set[str],
    ) -> float:
        # Relevance
        rel = item.score * self.W_RELEVANCE
        # Satisfaction
        sat = item.satisfaction_score * self.W_SATISFACTION
        # Diversity bonus for new genres
        new_genres = set(item.genres) - covered
        div = (len(new_genres) / max(1, len(item.genres))) * self.W_DIVERSITY
        # Exploration bonus proportional to uncertainty
        exp = item.uncertainty * self.W_EXPLORE
        # Business value (slight boost for non-repeated content types)
        seen_types = {s.content_type for s in selected}
        biz = (0.5 if item.content_type not in seen_types else 0.0) * self.W_BUSINESS
        return rel + sat + div + exp + biz

    def _diversity_score(self, items: list[SlateItem]) -> float:
        if len(items) < 2:
            return 0.0
        all_genres: set[str] = set()
        for item in items:
            all_genres.update(item.genres)
        unique_per_item = sum(len(set(i.genres)) for i in items)
        return len(all_genres) / max(1, unique_per_item / len(items))

    def _coherence_score(self, items: list[SlateItem]) -> float:
        """Page tells a coherent story if genres flow naturally."""
        if len(items) < 2:
            return 1.0
        transitions = 0
        for i in range(len(items) - 1):
            overlap = set(items[i].genres) & set(items[i+1].genres)
            if overlap:
                transitions += 1
        return transitions / (len(items) - 1)

    def _parse_genres(self, text: str) -> list[str]:
        import re
        m = re.search(r"Genres?:[\s]*([^.]+)", text, re.I)
        if m:
            return [g.strip().lower() for g in re.split(r"[,|]", m.group(1)) if g.strip()][:4]
        return ["unknown"]


# ── Intent-aware reranker ──────────────────────────────────────────────────────

class IntentAwareReranker:
    """
    Reranks candidates based on detected session intent.
    Different intents need different content mixes on the page.
    """

    INTENT_CONTENT_MIX = {
        "background_noise":   {"film": 0.3, "series": 0.5, "comedy": 0.8},
        "dedicated_viewing":  {"film": 0.6, "series": 0.4, "drama": 0.7},
        "discovery_mode":     {"film": 0.4, "documentary": 0.4, "new": 0.6},
        "binge_continuation": {"series": 0.9, "film": 0.1},
        "social_watching":    {"film": 0.5, "comedy": 0.6, "action": 0.5},
        "rewatch_comfort":    {"film": 0.5, "series": 0.5, "familiar": 0.8},
    }

    def rerank(
        self,
        candidates: list[dict[str, Any]],
        session_intent: str,
        alpha: float = 0.15,
    ) -> list[dict[str, Any]]:
        mix = self.INTENT_CONTENT_MIX.get(session_intent, {})
        for item in candidates:
            text = (item.get("text", "") + item.get("title", "")).lower()
            boost = sum(v for k, v in mix.items() if k in text) * alpha / max(len(mix), 1)
            item["score"] = round(float(item.get("score", 0)) + boost, 4)
            item["intent_boost"] = round(boost, 4)
        return sorted(candidates, key=lambda x: -x.get("score", 0))
