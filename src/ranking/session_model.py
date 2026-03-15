# src/ranking/session_model.py
"""
Netflix-grade Session & Temporal User State Modeling
=====================================================
Resolves Gap #4: "personalization is still shallow"

Closes:
  - recency decay on interaction history
  - sequence-aware session encoding
  - explicit negative feedback handling
  - intent drift detection across sessions
  - cold-start confidence estimation
  - household contamination scoring

Netflix 2026 standard: user state = f(history, recency, negatives, session_context)
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


HALF_LIFE_DAYS   = 14.0   # interaction weight halves every 14 days
MIN_CONFIDENCE   = 0.1    # cold-start confidence floor
COLD_START_N     = 5      # fewer than N interactions = cold start
NEG_WEIGHT       = -2.0   # negative feedback weight multiplier
SKIP_WEIGHT      = -0.5   # skip (< 10% watched) weight


@dataclass
class Interaction:
    doc_id: str
    timestamp: float           # unix epoch
    watch_pct: float           # 0.0 - 1.0
    explicit_rating: int | None = None   # 1-5 or None
    skipped: bool = False
    genres: list[str] = field(default_factory=list)
    language: str = ""


@dataclass
class UserState:
    user_id: str
    genre_affinities: dict[str, float]     # genre -> weighted score
    language_affinities: dict[str, float]
    recent_doc_ids: list[str]              # last 20, newest first
    interaction_count: int
    negative_doc_ids: set[str]             # explicitly disliked
    cold_start: bool
    confidence: float                      # 0-1, how reliable the profile is
    last_updated: float
    dominant_genres: list[str]             # top-3 genres
    session_intent_drift: float            # 0=stable, 1=drifting


@dataclass
class SessionContext:
    user_id: str
    session_start: float
    queries: list[str] = field(default_factory=list)
    viewed_doc_ids: list[str] = field(default_factory=list)
    intent_sequence: list[str] = field(default_factory=list)
    pivot_detected: bool = False   # user switched topic mid-session


class TemporalUserStateModel:
    """
    Maintains per-user state with recency decay and negative feedback.

    Recency decay: w(t) = 0.5 ^ (age_days / HALF_LIFE_DAYS)
    Negative signal: skips and low ratings get negative weight.
    """

    def __init__(self) -> None:
        self._interactions: dict[str, list[Interaction]] = defaultdict(list)
        self._negatives: dict[str, set[str]] = defaultdict(set)

    def record(self, user_id: str, interaction: Interaction) -> None:
        # Detect negative signal
        is_negative = (
            interaction.skipped or
            interaction.watch_pct < 0.1 or
            (interaction.explicit_rating is not None and interaction.explicit_rating <= 2)
        )
        if is_negative:
            self._negatives[user_id].add(interaction.doc_id)
        self._interactions[user_id].append(interaction)
        # Keep rolling window of 500
        if len(self._interactions[user_id]) > 500:
            self._interactions[user_id] = self._interactions[user_id][-500:]

    def get_state(self, user_id: str) -> UserState:
        interactions = self._interactions.get(user_id, [])
        negatives = self._negatives.get(user_id, set())
        now = time.time()

        genre_scores: dict[str, float] = defaultdict(float)
        lang_scores: dict[str, float] = defaultdict(float)
        recent_ids: list[str] = []

        for ix in sorted(interactions, key=lambda x: x.timestamp, reverse=True):
            age_days = (now - ix.timestamp) / 86400
            decay = 0.5 ** (age_days / HALF_LIFE_DAYS)

            if ix.doc_id in negatives:
                base_w = NEG_WEIGHT
            elif ix.skipped or ix.watch_pct < 0.1:
                base_w = SKIP_WEIGHT
            elif ix.watch_pct > 0.8:
                base_w = 1.5
            elif ix.watch_pct > 0.5:
                base_w = 1.0
            else:
                base_w = 0.6

            if ix.explicit_rating is not None:
                base_w *= (ix.explicit_rating - 3) * 0.5 + 1.0

            w = base_w * decay
            for g in ix.genres:
                genre_scores[g] += w
            if ix.language:
                lang_scores[ix.language] += decay

            if len(recent_ids) < 20:
                recent_ids.append(ix.doc_id)

        n = len(interactions)
        confidence = min(1.0, max(MIN_CONFIDENCE, math.log1p(n) / math.log1p(50)))
        cold_start = n < COLD_START_N

        dominant = sorted(genre_scores, key=lambda g: genre_scores[g], reverse=True)[:3]

        # Intent drift: compare last-5 genres vs first-5 genres in session
        drift = 0.0
        if n >= 10:
            early = set(g for ix in interactions[:5] for g in ix.genres)
            late  = set(g for ix in interactions[-5:] for g in ix.genres)
            overlap = len(early & late) / max(1, len(early | late))
            drift = 1.0 - overlap

        return UserState(
            user_id=user_id,
            genre_affinities=dict(genre_scores),
            language_affinities=dict(lang_scores),
            recent_doc_ids=recent_ids,
            interaction_count=n,
            negative_doc_ids=negatives,
            cold_start=cold_start,
            confidence=round(confidence, 3),
            last_updated=now,
            dominant_genres=dominant,
            session_intent_drift=round(drift, 3),
        )

    def score_candidate(self, user_id: str, doc: dict[str, Any]) -> float:
        """
        Personalisation re-score: returns delta to add to LTR score.
        Positive = user would likely enjoy. Negative = likely dislike.
        """
        state = self.get_state(user_id)
        doc_id = doc.get("doc_id", "")

        if doc_id in state.negative_doc_ids:
            return -10.0   # hard filter

        text = (doc.get("text", "") + " " + doc.get("title", "")).lower()
        score = 0.0

        for genre, affinity in state.genre_affinities.items():
            if genre in text:
                score += affinity * 0.1

        lang = doc.get("language", "")
        if lang and lang in state.language_affinities:
            score += state.language_affinities[lang] * 0.05

        if doc_id in state.recent_doc_ids:
            score -= 2.0   # penalise rewatch unless rewatch intent

        return round(score * state.confidence, 4)


class SessionEncoder:
    """
    Encodes the current session as a lightweight state vector.
    Detects mid-session topic pivots (user switches from thriller to comedy).
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionContext] = {}

    def start_session(self, user_id: str) -> SessionContext:
        ctx = SessionContext(user_id=user_id, session_start=time.time())
        self._sessions[user_id] = ctx
        return ctx

    def record_query(self, user_id: str, query: str, intent: str) -> None:
        ctx = self._sessions.setdefault(
            user_id, SessionContext(user_id=user_id, session_start=time.time())
        )
        ctx.queries.append(query)
        ctx.intent_sequence.append(intent)
        if len(ctx.intent_sequence) >= 3:
            recent = ctx.intent_sequence[-3:]
            if len(set(recent)) == 3:   # 3 different intents = pivot
                ctx.pivot_detected = True

    def get_session(self, user_id: str) -> SessionContext | None:
        return self._sessions.get(user_id)

    def session_summary(self, user_id: str) -> dict[str, Any]:
        ctx = self._sessions.get(user_id)
        if not ctx:
            return {"active": False}
        return {
            "active": True,
            "duration_min": round((time.time() - ctx.session_start) / 60, 1),
            "n_queries": len(ctx.queries),
            "pivot_detected": ctx.pivot_detected,
            "intent_sequence": ctx.intent_sequence[-5:],
        }


class HouseholdContaminationDetector:
    """
    Detects when a household profile has been contaminated by another
    viewer's watching habits using Jensen-Shannon divergence on genre
    distributions.

    High contamination score -> reduce personalisation weight, use
    session-level signals instead of long-term profile.
    """

    def score_contamination(
        self,
        recent_genres: dict[str, float],
        historical_genres: dict[str, float],
    ) -> float:
        """Returns 0.0 (clean) to 1.0 (heavily contaminated)."""
        if not recent_genres or not historical_genres:
            return 0.0

        all_genres = set(recent_genres) | set(historical_genres)
        r_total = max(sum(recent_genres.values()), 1e-9)
        h_total = max(sum(historical_genres.values()), 1e-9)

        p = {g: recent_genres.get(g, 0) / r_total for g in all_genres}
        q = {g: historical_genres.get(g, 0) / h_total for g in all_genres}

        # Jensen-Shannon divergence (0=identical, 1=completely different)
        def kl(a: dict, b: dict) -> float:
            s = 0.0
            for g in a:
                m = (a[g] + b.get(g, 1e-9)) / 2
                if a[g] > 0 and m > 0:
                    s += a[g] * math.log(a[g] / m)
            return s

        m = {g: (p[g] + q[g]) / 2 for g in all_genres}
        js = (kl(p, m) + kl(q, m)) / 2
        return round(min(1.0, max(0.0, js)), 3)
