"""
StreamLens — NER Query Understanding
======================================
Extracts named entities from queries to enable entity-aware retrieval.

Examples:
  "Tom Hanks movies"      → {actors: ["tom hanks"]}
  "Christopher Nolan"     → {directors: ["christopher nolan"]}
  "sci-fi action"         → {genres: ["sci-fi", "action"]}
  "90s crime thrillers"   → {decades: ["1990s"], genres: ["crime", "thriller"]}
  "movies like Inception" → {similar_to: ["Inception"]}

Why this matters:
  - BM25 on "movies with Tom Hanks" → matches docs with "movies", "with", "tom", "hanks"
  - NER-aware: identify "Tom Hanks" as ACTOR entity → direct lookup in actor index
  - Closes vocabulary gap between natural language and structured metadata
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

# Known genres in MovieLens
GENRES = {
    "action", "adventure", "animation", "children", "comedy", "crime",
    "documentary", "drama", "fantasy", "film-noir", "horror", "musical",
    "mystery", "romance", "sci-fi", "thriller", "war", "western",
    # Common aliases
    "animated", "romantic", "scary", "funny", "dark", "light",
}

# Decade patterns
DECADE_PATTERNS = {
    r"\b(19)?[0-9]0s\b": lambda m: f"{m.group(0)[:2]}0s",
    r"\b(20)?[0-2][0-9]0s\b": lambda m: m.group(0),
}

# Intent patterns
INTENT_PATTERNS = {
    "similar_to":  [r"(?:like|similar to|reminds me of|same as)\s+([A-Z][\w\s]+?)(?:\s*$|\s*[,;.])"],
    "by_director": [r"(?:by|directed by|from director)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"],
    "with_actor":  [r"(?:with|starring|featuring)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"],
    "decade":      [r"\b([12][90][0-9]0)s\b"],
}


@dataclass
class QueryEntities:
    """Structured representation of extracted query entities."""
    genres:     list[str] = field(default_factory=list)
    actors:     list[str] = field(default_factory=list)
    directors:  list[str] = field(default_factory=list)
    decades:    list[str] = field(default_factory=list)
    similar_to: list[str] = field(default_factory=list)
    keywords:   list[str] = field(default_factory=list)
    raw_query:  str = ""

    def has_entities(self) -> bool:
        return any([self.genres, self.actors, self.directors,
                    self.decades, self.similar_to])


def extract_entities(query: str) -> QueryEntities:
    """
    Extract named entities from a query string.
    
    Args:
        query: Raw user query
    
    Returns:
        QueryEntities with extracted actors, genres, decades, etc.
    """
    entities = QueryEntities(raw_query=query)
    q_lower  = query.lower()
    q_words  = set(q_lower.split())

    # Extract genres
    entities.genres = [g for g in GENRES if g in q_lower]

    # Extract decades (e.g., "90s", "1980s", "2000s")
    decade_matches = re.findall(r"\b(?:19|20)?([0-9][0-9])0s\b", q_lower)
    for dec in decade_matches:
        if dec.startswith(("19", "20")):
            entities.decades.append(f"{dec}0s")
        elif int(dec) >= 50:
            entities.decades.append(f"19{dec}0s")
        else:
            entities.decades.append(f"20{dec}0s")

    # Extract "similar to X"
    for pattern in INTENT_PATTERNS["similar_to"]:
        matches = re.findall(pattern, query)
        entities.similar_to.extend(matches)

    # Extract director mentions
    for pattern in INTENT_PATTERNS["by_director"]:
        matches = re.findall(pattern, query)
        entities.directors.extend([m.strip() for m in matches])

    # Extract actor mentions
    for pattern in INTENT_PATTERNS["with_actor"]:
        matches = re.findall(pattern, query)
        entities.actors.extend([m.strip() for m in matches])

    # Remaining words as keywords
    stop_words = {"movies", "films", "show", "watch", "like", "want",
                  "good", "best", "top", "similar", "with", "by", "the", "a", "an"}
    entity_words = set(" ".join(entities.genres + entities.decades).split())
    entities.keywords = [w for w in q_lower.split()
                         if w not in stop_words and w not in entity_words
                         and len(w) > 2]

    return entities


def entity_boost_scores(
    entities: QueryEntities,
    candidates: list[dict],
    genres_index: dict,
    actors_index: dict,
) -> list[dict]:
    """
    Boost candidate scores based on extracted entities.
    
    Entity matches provide a relevance signal that BM25 misses
    when the query uses natural language (e.g., "Tom Hanks movies"
    vs the metadata key "cast: Tom Hanks").
    """
    # Build boost set
    boosted_docs = set()

    for genre in entities.genres:
        boosted_docs.update(genres_index.get(genre, []))

    for actor in entities.actors + entities.keywords:
        boosted_docs.update(actors_index.get(actor.lower(), []))

    # Apply boost to candidates
    for item in candidates:
        doc_id = item.get("doc_id", "")
        if doc_id in boosted_docs:
            item["ner_boost"] = 0.15  # +15% score boost
            item["score"] = item.get("score", 0) * 1.15
        else:
            item["ner_boost"] = 0.0

    return candidates
