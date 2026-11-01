"""
StreamLens — HyDE (Hypothetical Document Embeddings) Query Rewriting
Improves dense retrieval by generating a hypothetical relevant document,
then using its embedding as the query vector instead of the raw query.
Run: integrated into retrieval pipeline via /search?hyde=true
"""
from __future__ import annotations
import os, json, urllib.request, urllib.error, time

_URL = "https://api.openai.com/v1/chat/completions"

def hyde_rewrite(query: str, genre_hint: str = "") -> str:
    """
    Generate a hypothetical movie description that would be relevant to the query.
    This embedding is then used for dense retrieval instead of the raw query.
    
    Why HyDE works: 'crime thriller' as a query has a different embedding than
    a movie description containing crime thriller elements. HyDE bridges that gap.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key.startswith("sk-"):
        return query  # fallback to original query

    genre_context = f" The film is likely a {genre_hint}." if genre_hint else ""
    
    system = (
        "You write brief, realistic movie corpus descriptions. "
        "Write a 2-sentence hypothetical movie description that would be "
        "HIGHLY RELEVANT to the user's search query. "
        "Format: 'Title: [Title] | Genres: [genres] | Tags: [tags]' "
        "Make it specific and realistic. No preamble."
    )
    user = f"Search query: {query}{genre_context}\n\nWrite a hypothetical relevant movie description."

    payload = json.dumps({
        "model": "gpt-4o-mini",
        "max_tokens": 80,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    }).encode()

    for attempt in range(3):
        req = urllib.request.Request(_URL, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        })
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                result = json.loads(r.read())
                return result["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep((2 ** attempt) * 1.5)
                continue
            return query
        except Exception:
            return query
    return query


def should_use_hyde(query: str) -> bool:
    """
    HyDE is most useful for semantic/mood queries, not navigational ones.
    'crime thriller' → YES (semantic)
    'Toy Story 1995' → NO (navigational — exact match is better)
    """
    query_lower = query.lower()
    # Navigational signals — exact title lookup
    nav_signals = ["(19", "(20", "part 1", "part 2", "season", "episode"]
    if any(s in query_lower for s in nav_signals):
        return False
    # Very short queries (1-2 words) — BM25 handles these well
    if len(query.split()) <= 2:
        return False
    # Mood/semantic queries — HyDE helps
    mood_signals = ["feel", "mood", "like", "similar", "something", "recommend", "good", "best", "want"]
    return any(s in query_lower for s in mood_signals) or len(query.split()) >= 4


if __name__ == "__main__":
    # Test HyDE
    test_queries = [
        "crime thriller with twists",
        "something heartwarming for family night",
        "mind-bending sci-fi like Interstellar",
        "Toy Story 1995",  # should NOT use HyDE
        "funny",           # should NOT use HyDE
    ]
    print("HyDE Query Rewriting Test")
    print("=" * 50)
    for q in test_queries:
        use = should_use_hyde(q)
        print(f"\nQuery: '{q}'")
        print(f"Use HyDE: {use}")
        if use:
            rewritten = hyde_rewrite(q)
            print(f"HyDE output: {rewritten[:120]}...")
        else:
            print("Skipping HyDE — navigational or short query")
