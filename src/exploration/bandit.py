# src/exploration/bandit.py
"""
Phase 3 — Serendipity & Anti-Silo Engine
==========================================
Breaks the "Personalization Prison" by injecting exploration into every feed row.

Components:
  ContextualBandit         — epsilon-greedy slot allocator. Reserves configurable
                             fraction of feed slots for high-novelty content.
  SerendipityScorer        — measures Discovery Breadth KPI (distinct genres / k).
  DiversityReranker        — MMR (Maximal Marginal Relevance) genre-aware reranker.
  MultiObjectiveReranker   — combines Relevance + Diversity + Business signal.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any


# ─── Helpers ──────────────────────────────────────────────────────────────────

_CATALOG_RE = re.compile(r"([A-Za-z ]+):\s*([^|]+)")

def _parse_genres(text: str, title: str = "") -> list[str]:
    genres: list[str] = []
    for m in _CATALOG_RE.finditer(text or ""):
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if key in {"genres", "genre", "tags", "tag"}:
            genres.extend([g.strip().lower() for g in re.split(r"[,|]", val) if g.strip()])
    if not genres and title:
        for kw in ["action", "comedy", "drama", "thriller", "romance", "sci-fi",
                   "horror", "animation", "documentary", "fantasy", "crime"]:
            if kw in title.lower():
                genres.append(kw)
    return genres or ["unknown"]

def _genre_set(text: str, title: str = "") -> set[str]:
    return {g.lower().strip() for g in _parse_genres(text, title) if g.strip()}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ScoredDoc:
    doc_id: str
    score: float
    title: str = ""
    text: str = ""
    genres: list[str] = field(default_factory=list)
    slot_type: str = "exploit"   # "exploit" | "explore"


@dataclass
class SerendipityReport:
    query: str
    total_hits: int
    distinct_genres: int
    discovery_breadth: float
    exploration_slots: int
    exploitation_slots: int
    genre_distribution: dict[str, int] = field(default_factory=dict)
    in_silo: bool = False


# ─── Contextual Bandit ────────────────────────────────────────────────────────

class ContextualBandit:
    """
    Epsilon-greedy bandit for feed row slot allocation.
    epsilon=0.15 means 15% of slots are exploration ("explore new genres").
    """

    def __init__(self, epsilon: float = 0.15, exploit_cutoff: int = 5, seed: int | None = None) -> None:
        self.epsilon = float(epsilon)
        self.exploit_cutoff = int(exploit_cutoff)
        self._rng = random.Random(seed)

    def select(self, candidates: list[tuple[str, float]], corpus: dict[str, Any], n_slots: int) -> list[ScoredDoc]:
        if not candidates:
            return []
        exploit_pool = candidates[: self.exploit_cutoff]
        explore_pool = candidates[self.exploit_cutoff :]
        selected: list[ScoredDoc] = []
        used: set[str] = set()

        for _ in range(min(n_slots, len(candidates))):
            is_explore = self._rng.random() < self.epsilon and len(explore_pool) > 0
            pool = explore_pool if is_explore else exploit_pool
            avail = [(d, s) for d, s in pool if d not in used]
            if not avail:
                fallback = explore_pool if not is_explore else exploit_pool
                avail = [(d, s) for d, s in fallback if d not in used]
            if not avail:
                break
            did, score = avail[0]
            used.add(did)
            row = corpus.get(str(did), {})
            genres = _parse_genres(str(row.get("text", "")), str(row.get("title", "")))
            selected.append(ScoredDoc(
                doc_id=did, score=float(score),
                title=str(row.get("title", "")), text=str(row.get("text", "")),
                genres=genres, slot_type="explore" if is_explore else "exploit",
            ))
        return selected


# ─── Serendipity Scorer ───────────────────────────────────────────────────────

class SerendipityScorer:
    """Measures Discovery Breadth KPI: distinct_genres / total_results. Target > 0.40."""

    def score(self, query: str, docs: list[ScoredDoc]) -> SerendipityReport:
        if not docs:
            return SerendipityReport(query=query, total_hits=0, distinct_genres=0,
                                     discovery_breadth=0.0, exploration_slots=0, exploitation_slots=0)
        genre_dist: dict[str, int] = {}
        all_genres: set[str] = set()
        for doc in docs:
            for g in _genre_set(doc.text, doc.title):
                genre_dist[g] = genre_dist.get(g, 0) + 1
                all_genres.add(g)
        n = len(docs)
        return SerendipityReport(
            query=query, total_hits=n, distinct_genres=len(all_genres),
            discovery_breadth=round(len(all_genres) / n, 4) if n else 0.0,
            exploration_slots=sum(1 for d in docs if d.slot_type == "explore"),
            exploitation_slots=sum(1 for d in docs if d.slot_type == "exploit"),
            genre_distribution=genre_dist,
        )

    def is_in_silo(self, genre_history: list[str], window: int = 10, threshold: float = 0.8) -> bool:
        if len(genre_history) < window:
            return False
        recent = genre_history[-window:]
        most_common = max(recent.count(g) for g in set(recent))
        return (most_common / window) >= threshold


# ─── MMR Diversity Reranker ───────────────────────────────────────────────────

class DiversityReranker:
    """
    Maximal Marginal Relevance reranker.
    lambda_param=1.0 → pure relevance | 0.0 → pure diversity. Default: 0.7
    """

    def __init__(self, lambda_param: float = 0.7) -> None:
        self.lambda_param = float(lambda_param)

    def rerank(self, candidates: list[tuple[str, float]], corpus: dict[str, Any], k: int) -> list[tuple[str, float]]:
        if not candidates:
            return []
        doc_genres: dict[str, set[str]] = {}
        scores: dict[str, float] = {}
        for did, s in candidates:
            row = corpus.get(str(did), {})
            doc_genres[did] = _genre_set(str(row.get("text", "")), str(row.get("title", "")))
            scores[did] = float(s)

        max_s = max(scores.values()) if scores else 1.0
        min_s = min(scores.values()) if scores else 0.0
        span = max(max_s - min_s, 1e-9)
        norm = {d: (s - min_s) / span for d, s in scores.items()}

        selected: list[tuple[str, float]] = []
        sel_genres: set[str] = set()
        remaining = list(candidates)

        while remaining and len(selected) < k:
            best_did, best_mmr = None, -float("inf")
            for did, _ in remaining:
                rel = norm[did]
                if not sel_genres:
                    div = 1.0
                else:
                    g = doc_genres[did]
                    inter = len(g & sel_genres)
                    union = len(g | sel_genres)
                    div = 1.0 - (inter / union if union > 0 else 0.0)
                mmr = self.lambda_param * rel + (1.0 - self.lambda_param) * div
                if mmr > best_mmr:
                    best_mmr, best_did = mmr, did
            if best_did is None:
                break
            selected.append((best_did, scores[best_did]))
            sel_genres |= doc_genres[best_did]
            remaining = [(d, s) for d, s in remaining if d != best_did]

        return selected


# ─── Multi-Objective Reranker ─────────────────────────────────────────────────

class MultiObjectiveReranker:
    """Combines Relevance (0.60) + Diversity (0.25) + Business value (0.15)."""

    def __init__(
        self,
        relevance_weight: float = 0.60,
        diversity_weight: float = 0.25,
        business_weight: float = 0.15,
        business_boost_tags: list[str] | None = None,
    ) -> None:
        total = relevance_weight + diversity_weight + business_weight
        self.w_rel = relevance_weight / total
        self.w_div = diversity_weight / total
        self.w_bus = business_weight / total
        self.boost_tags = set(t.lower() for t in (business_boost_tags or [
            "netflix original", "original", "exclusive", "award"]))

    def _business_score(self, text: str, title: str) -> float:
        combined = (text + " " + title).lower()
        return 1.0 if any(t in combined for t in self.boost_tags) else 0.0

    def rerank(
        self,
        candidates: list[tuple[str, float]],
        corpus: dict[str, Any],
        k: int,
        diversity_reranker: DiversityReranker | None = None,
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        if diversity_reranker is None:
            diversity_reranker = DiversityReranker(lambda_param=0.7)
        mmr_list = diversity_reranker.rerank(candidates, corpus, k=len(candidates))
        mmr_rank = {did: i for i, (did, _) in enumerate(mmr_list)}
        max_s = max(s for _, s in candidates) if candidates else 1.0
        min_s = min(s for _, s in candidates) if candidates else 0.0
        span = max(max_s - min_s, 1e-9)
        n = len(candidates)
        combined: list[tuple[str, float]] = []
        for did, rel_s in candidates:
            row = corpus.get(str(did), {})
            norm_rel = (rel_s - min_s) / span
            norm_div = 1.0 - (mmr_rank.get(did, n) / max(n, 1))
            norm_bus = self._business_score(str(row.get("text", "")), str(row.get("title", "")))
            final = self.w_rel * norm_rel + self.w_div * norm_div + self.w_bus * norm_bus
            combined.append((did, float(final)))
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:k]
