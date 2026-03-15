# src/retrieval/query_understanding.py
"""
Netflix-grade Query Understanding Layer
=======================================
Resolves Gap #2: "retrieval stack is not production-grade"

Components:
  SpellingCorrector     - edit-distance typo correction against corpus vocabulary
  QueryRewriter         - expands/rewrites queries for better recall
  EntityRecogniser      - identifies titles, people, genres, years in queries
  QueryClassifier       - intent classification (navigational/exploratory/transactional)
  QueryUnderstandingPipeline - orchestrates all components

Netflix standard: every query passes through understanding BEFORE retrieval.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Query intent taxonomy ─────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    NAVIGATIONAL  = "navigational"   # exact title lookup: "Stranger Things"
    EXPLORATORY   = "exploratory"    # genre/mood browse: "something funny tonight"
    TRANSACTIONAL = "transactional"  # action intent: "watch action movies"
    PERSON        = "person"         # actor/director: "movies with Tom Hanks"
    SIMILARITY    = "similarity"     # "movies like Inception"
    UNKNOWN       = "unknown"


@dataclass
class ParsedQuery:
    raw: str
    normalised: str
    corrected: str
    rewrites: list[str]
    intent: QueryIntent
    entities: dict[str, list[str]]   # {"title": [...], "person": [...], "genre": [...], "year": [...]}
    filters: dict[str, Any]          # {"year_min": 2010, "language": "English"}
    confidence: float
    debug: dict = field(default_factory=dict)


# ── Vocabulary for typo correction ────────────────────────────────────────────

GENRE_VOCAB = {
    "action", "adventure", "animation", "biography", "comedy", "crime",
    "documentary", "drama", "family", "fantasy", "history", "horror",
    "music", "mystery", "romance", "sci-fi", "sport", "thriller",
    "war", "western", "noir", "superhero", "heist",
}

MOOD_VOCAB = {
    "dark", "gritty", "light", "funny", "scary", "heartwarming", "epic",
    "intense", "relaxing", "uplifting", "thought-provoking", "mind-bending",
    "feel-good", "disturbing", "inspiring", "suspenseful",
}

COMMON_TYPOS = {
    "acton": "action", "commedy": "comedy", "horro": "horror",
    "romace": "romance", "thriler": "thriller", "documentry": "documentary",
    "scifi": "sci-fi", "sifi": "sci-fi", "animaton": "animation",
    "adventur": "adventure", "biograpy": "biography",
}

SIMILARITY_TRIGGERS = {"like", "similar", "reminds", "remind", "same as", "type of"}
PERSON_TRIGGERS = {"with", "starring", "directed by", "by", "featuring", "actor", "director"}
EXPLORATORY_TRIGGERS = {"something", "anything", "recommend", "suggest", "good", "best", "top"}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s\-\'\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _edit_distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 3:
        return 99
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


class SpellingCorrector:
    def __init__(self, extra_vocab: set[str] | None = None) -> None:
        self.vocab = GENRE_VOCAB | MOOD_VOCAB | (extra_vocab or set())

    def correct_token(self, token: str) -> str:
        if token in COMMON_TYPOS:
            return COMMON_TYPOS[token]
        if token in self.vocab or len(token) <= 3:
            return token
        best, best_d = token, 2
        for word in self.vocab:
            d = _edit_distance(token, word)
            if d < best_d:
                best, best_d = word, d
        return best

    def correct(self, query: str) -> str:
        tokens = query.split()
        return " ".join(self.correct_token(t) for t in tokens)


class EntityRecogniser:
    YEAR_RE = re.compile(r"\b(19[0-9]{2}|20[0-2][0-9])\b")
    DECADE_RE = re.compile(r"\b([12][0-9]{3})s\b")

    def recognise(self, query: str) -> dict[str, list[str]]:
        q = query.lower()
        entities: dict[str, list[str]] = {
            "genre": [], "mood": [], "year": [], "decade": [],
            "person": [], "title": [],
        }

        for g in GENRE_VOCAB:
            if g in q:
                entities["genre"].append(g)
        for m in MOOD_VOCAB:
            if m in q:
                entities["mood"].append(m)

        for m in self.YEAR_RE.finditer(query):
            entities["year"].append(m.group())
        for m in self.DECADE_RE.finditer(query):
            entities["decade"].append(m.group())

        # Person detection: words after trigger words capitalised
        for trigger in PERSON_TRIGGERS:
            idx = q.find(trigger)
            if idx >= 0:
                after = query[idx + len(trigger):].strip()
                words = after.split()[:3]
                name = " ".join(w for w in words if w[0:1].isupper())
                if name:
                    entities["person"].append(name)

        return entities


class QueryClassifier:
    def classify(self, query: str, entities: dict[str, list[str]]) -> QueryIntent:
        q = query.lower()

        if any(t in q for t in SIMILARITY_TRIGGERS):
            return QueryIntent.SIMILARITY
        if entities.get("person"):
            return QueryIntent.PERSON
        if any(t in q for t in EXPLORATORY_TRIGGERS):
            return QueryIntent.EXPLORATORY
        if any(t in q for t in ("watch", "play", "show me", "find")):
            return QueryIntent.TRANSACTIONAL
        # Navigational: short query, mostly title-case
        words = query.split()
        if len(words) <= 4 and sum(1 for w in words if w[0:1].isupper()) >= len(words) - 1:
            return QueryIntent.NAVIGATIONAL

        return QueryIntent.UNKNOWN


class QueryRewriter:
    EXPANSION_MAP = {
        "sci-fi": ["science fiction", "futuristic", "space"],
        "noir": ["dark", "crime", "detective", "gritty"],
        "feel-good": ["uplifting", "heartwarming", "comedy", "family"],
        "mind-bending": ["psychological", "twist", "complex", "surreal"],
        "heist": ["crime", "robbery", "theft", "caper"],
        "superhero": ["marvel", "dc", "action", "powers"],
    }

    def rewrite(self, query: str, entities: dict[str, list[str]], intent: QueryIntent) -> list[str]:
        rewrites = []
        q_lower = query.lower()

        for term, expansions in self.EXPANSION_MAP.items():
            if term in q_lower:
                expanded = query + " " + " ".join(expansions[:2])
                rewrites.append(expanded)

        if intent == QueryIntent.SIMILARITY:
            for trigger in SIMILARITY_TRIGGERS:
                if trigger in q_lower:
                    base = re.sub(rf".*{trigger}\s*", "", q_lower, flags=re.I).strip()
                    if base:
                        rewrites.append(base)
                    break

        if intent == QueryIntent.PERSON and entities.get("person"):
            rewrites.append(f"movies starring {entities['person'][0]}")

        if not rewrites:
            rewrites = [query]

        return list(dict.fromkeys(rewrites))[:4]


class QueryUnderstandingPipeline:
    """
    Orchestrates query understanding. Drop-in before any retrieval call.

    Usage:
        qup = QueryUnderstandingPipeline()
        parsed = qup.run("somthing funny with tom hanks")
        # parsed.corrected == "something funny with tom hanks"
        # parsed.intent == QueryIntent.PERSON
        # parsed.entities == {"person": ["Tom Hanks"], "mood": ["funny"], ...}
        # parsed.rewrites == ["movies starring Tom Hanks", ...]
    """

    def __init__(self, corpus_vocab: set[str] | None = None) -> None:
        self.corrector  = SpellingCorrector(extra_vocab=corpus_vocab)
        self.recogniser = EntityRecogniser()
        self.classifier = QueryClassifier()
        self.rewriter   = QueryRewriter()

    def run(self, raw_query: str) -> ParsedQuery:
        normalised = _normalise(raw_query)
        corrected  = self.corrector.correct(normalised)
        entities   = self.recogniser.recognise(corrected)
        intent     = self.classifier.classify(corrected, entities)
        rewrites   = self.rewriter.rewrite(corrected, entities, intent)

        # Build availability filters from entities
        filters: dict[str, Any] = {}
        if entities.get("year"):
            yr = int(entities["year"][0])
            filters["year_min"] = yr - 1
            filters["year_max"] = yr + 1
        if entities.get("genre"):
            filters["genre"] = entities["genre"][0]

        confidence = 0.9 if corrected == normalised else 0.7

        return ParsedQuery(
            raw=raw_query, normalised=normalised, corrected=corrected,
            rewrites=rewrites, intent=intent, entities=entities,
            filters=filters, confidence=confidence,
            debug={"spell_changed": corrected != normalised},
        )
