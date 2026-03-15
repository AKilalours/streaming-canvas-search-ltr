# src/app/personalization_v2.py
"""
Phase 5 — Advanced Personalization Engine
==========================================
Replaces keyword_overlap_score() with:
  1. UserEmbeddingPersonalizer  — cosine sim between user history embedding
                                  and candidate doc embedding.
  2. ImplicitFeedbackCollector  — Redis-backed click/watch/skip signal store.
                                  Exports qrels for next LTR training cycle.
  3. HouseholdProfileMerger     — merges multi-profile feeds (anti-silo).
  4. explain_personalization()  — structured why-was-this-boosted payload.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass
class PersonalizationSignal:
    doc_id: str
    base_score: float
    personalization_boost: float
    final_score: float
    method: str        # "embedding" | "keyword" | "household" | "none"
    reason: str
    confidence: float  # 0.0 – 1.0


@dataclass
class FeedbackEvent:
    user_id: str
    doc_id: str
    event_type: str    # "click" | "watch_start" | "watch_complete" | "skip" | "dislike"
    timestamp: float
    query: str = ""
    rank: int = -1


class UserEmbeddingPersonalizer:
    """
    Cosine similarity between user profile embedding (mean of watch history)
    and candidate doc embedding. Falls back to keyword overlap if unavailable.
    """

    LABEL_MAP = {"watch_complete": 3, "watch_start": 2, "click": 1, "skip": 0, "dislike": -1}

    def __init__(self, boost_weight: float = 0.15, history_window: int = 20, min_history: int = 3) -> None:
        self.boost_weight = float(boost_weight)
        self.history_window = int(history_window)
        self.min_history = int(min_history)

    def _user_embedding(self, user_data: dict[str, Any], doc_embeddings: dict[str, np.ndarray] | None) -> np.ndarray | None:
        if doc_embeddings is None:
            return None
        history = user_data.get("watch_history", [])[-self.history_window:]
        if len(history) < self.min_history:
            return None
        vecs = [doc_embeddings[did] for did in history if did in doc_embeddings]
        if not vecs:
            return None
        mean_vec = np.stack(vecs).astype(np.float32).mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        return mean_vec / norm if norm > 0 else mean_vec

    def boost_scores(
        self,
        ranked: list[tuple[str, float]],
        corpus: dict[str, Any],
        user_id: str | None,
        users: dict[str, Any],
        doc_embeddings: dict[str, np.ndarray] | None = None,
    ) -> tuple[list[tuple[str, float]], list[PersonalizationSignal]]:
        if not user_id or user_id not in users:
            signals = [PersonalizationSignal(
                doc_id=did, base_score=s, personalization_boost=0.0,
                final_score=s, method="none", reason="No user profile", confidence=0.0
            ) for did, s in ranked]
            return ranked, signals

        user_data = users[user_id]
        user_emb = self._user_embedding(user_data, doc_embeddings)
        signals: list[PersonalizationSignal] = []
        rescored: list[tuple[str, float]] = []

        for did, base_s in ranked:
            boost, method, reason, confidence = 0.0, "none", "No signal", 0.0

            if user_emb is not None and doc_embeddings and did in doc_embeddings:
                doc_emb = doc_embeddings[did].astype(np.float32)
                sim = float(np.dot(user_emb, doc_emb))
                boost = self.boost_weight * max(0.0, sim)
                method, reason, confidence = "embedding", f"Cosine sim={sim:.3f}", min(1.0, sim)
            else:
                kws = user_data.get("keywords", [])
                row = corpus.get(str(did), {})
                text = (str(row.get("title", "")) + " " + str(row.get("text", ""))).lower()
                hits = sum(1 for kw in kws if str(kw).lower() in text)
                if kws:
                    overlap = hits / len(kws)
                    boost = self.boost_weight * overlap
                    method, reason, confidence = "keyword", f"Overlap={overlap:.2f} ({hits}/{len(kws)})", overlap

            final_s = base_s + boost
            rescored.append((did, final_s))
            signals.append(PersonalizationSignal(
                doc_id=did, base_score=float(base_s), personalization_boost=float(boost),
                final_score=float(final_s), method=method, reason=reason, confidence=float(confidence),
            ))

        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored, signals


class ImplicitFeedbackCollector:
    """
    Stores implicit signals (click/watch/skip) in Redis.
    Exports as qrels dict for LTR training.
    """

    LABEL_MAP = {"watch_complete": 3, "watch_start": 2, "click": 1, "skip": 0, "dislike": -1}

    def __init__(self, redis_client: Any | None = None) -> None:
        self.redis = redis_client

    def record(self, event: FeedbackEvent) -> bool:
        if self.redis is None:
            return False
        try:
            payload = json.dumps({
                "user_id": event.user_id, "doc_id": event.doc_id,
                "event_type": event.event_type,
                "label": self.LABEL_MAP.get(event.event_type, 0),
                "timestamp": event.timestamp, "query": event.query, "rank": event.rank,
            })
            key = f"feedback:{event.user_id}:{event.doc_id}"
            self.redis.setex(key, 86400 * 30, payload)
            self.redis.lpush("feedback:stream", payload)
            self.redis.ltrim("feedback:stream", 0, 99999)
            return True
        except Exception:
            return False

    def export_qrels(self, user_id: str | None = None, min_label: int = 1) -> dict[str, dict[str, int]]:
        if self.redis is None:
            return {}
        try:
            raw_events = self.redis.lrange("feedback:stream", 0, -1)
        except Exception:
            return {}
        qrels: dict[str, dict[str, int]] = {}
        for raw in raw_events:
            try:
                ev = json.loads(raw)
                label = int(ev.get("label", 0))
                if label < min_label:
                    continue
                if user_id and ev.get("user_id") != user_id:
                    continue
                query = str(ev.get("query", ""))
                if not query:
                    continue
                qhash = hashlib.md5(query.encode()).hexdigest()[:12]
                doc_id = str(ev.get("doc_id", ""))
                if not doc_id:
                    continue
                qrels.setdefault(qhash, {})[doc_id] = max(
                    label, qrels.get(qhash, {}).get(doc_id, 0)
                )
            except Exception:
                continue
        return qrels


class HouseholdProfileMerger:
    """
    Round-robin merge of multiple user profile feeds into one balanced household feed.
    Resolves Personalization Prison for shared-account households.
    """

    def merge_feeds(
        self,
        profile_feeds: dict[str, list[tuple[str, float]]],
        corpus: dict[str, Any],
        total_slots: int = 20,
        balance: dict[str, float] | None = None,
    ) -> list[tuple[str, float, str]]:
        if not profile_feeds:
            return []
        profiles = list(profile_feeds.keys())
        if balance is None:
            balance = {p: 1.0 / len(profiles) for p in profiles}
        total_w = sum(balance.values())
        norm_balance = {p: w / total_w for p, w in balance.items()}
        slots_per_profile = {p: max(1, round(norm_balance.get(p, 0) * total_slots)) for p in profiles}
        merged: list[tuple[str, float, str]] = []
        used_ids: set[str] = set()
        iterators = {p: iter(profile_feeds[p]) for p in profiles}
        remaining_slots = dict(slots_per_profile)
        while sum(remaining_slots.values()) > 0 and len(merged) < total_slots:
            made_progress = False
            for profile in profiles:
                if remaining_slots.get(profile, 0) <= 0:
                    continue
                for did, score in iterators[profile]:
                    if did not in used_ids:
                        merged.append((did, float(score), profile))
                        used_ids.add(did)
                        remaining_slots[profile] -= 1
                        made_progress = True
                        break
            if not made_progress:
                break
        return merged[:total_slots]


def explain_personalization(
    signal: PersonalizationSignal, user_id: str, corpus_row: dict[str, Any]
) -> dict[str, Any]:
    title = str(corpus_row.get("title", "Unknown"))
    return {
        "user_id": user_id, "doc_id": signal.doc_id, "title": title,
        "personalization": {
            "method": signal.method, "boost": round(signal.personalization_boost, 4),
            "confidence": round(signal.confidence, 3), "reason": signal.reason,
        },
        "scores": {
            "base": round(signal.base_score, 4),
            "personalized": round(signal.final_score, 4),
            "lift_pct": round(100 * signal.personalization_boost / max(abs(signal.base_score), 1e-9), 2),
        },
    }
