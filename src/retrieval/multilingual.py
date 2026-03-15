# src/retrieval/multilingual.py
"""
Multilingual Query Handling
============================
Language detection + query translation before retrieval.
Netflix serves 190+ countries — multilingual query handling is critical.

Pipeline:
  1. Detect query language
  2. If non-English, translate to English (or search in original)
  3. Pass translated query to retrieval stack
  4. Return results with language metadata
"""
from __future__ import annotations
import re
from typing import Any

# Simple language detection via character set heuristics
# Production would use langdetect or OpenAI
LANG_PATTERNS = {
    "japanese":  re.compile(r"[぀-ヿ一-鿿]"),
    "korean":    re.compile(r"[가-힯ᄀ-ᇿ]"),
    "arabic":    re.compile(r"[؀-ۿ]"),
    "hindi":     re.compile(r"[ऀ-ॿ]"),
    "chinese":   re.compile(r"[一-鿿]"),
    "russian":   re.compile(r"[Ѐ-ӿ]"),
    "greek":     re.compile(r"[Ͱ-Ͽ]"),
    "thai":      re.compile(r"[฀-๿]"),
}

# Common movie query translations (production uses NMT model or API)
QUERY_TRANSLATIONS = {
    "action": "action", "accione": "action", "aksyon": "action",
    "romantique": "romance", "romance": "romance",
    "horreur": "horror", "horror": "horror",
    "comedie": "comedy", "comedia": "comedy",
    "thriller": "thriller", "krimi": "crime",
    "documentaire": "documentary", "documental": "documentary",
    "anime": "animation", "animazione": "animation",
    "enfants": "children", "niños": "children",
    "guerre": "war", "guerra": "war",
    "science fiction": "sci-fi", "ciencia ficcion": "sci-fi",
}

def detect_language(query: str) -> str:
    """Detect query language from character set."""
    for lang, pattern in LANG_PATTERNS.items():
        if pattern.search(query):
            return lang
    # Check for common non-English Latin characters
    if re.search(r"[àáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ]", query.lower()):
        return "european_latin"
    return "english"

def normalize_query(query: str) -> dict[str, Any]:
    """
    Normalize a query for retrieval:
    - Detect language
    - Translate known terms
    - Return normalized query + metadata
    """
    detected_lang = detect_language(query)
    original = query.strip()
    normalized = original.lower()

    # Apply known translations
    translated_terms = []
    for foreign, english in QUERY_TRANSLATIONS.items():
        if foreign in normalized and foreign != english:
            normalized = normalized.replace(foreign, english)
            translated_terms.append(f"{foreign}→{english}")

    return {
        "original_query": original,
        "normalized_query": normalized,
        "detected_language": detected_lang,
        "translations_applied": translated_terms,
        "needs_translation": detected_lang not in ("english", "european_latin"),
        "retrieval_query": normalized,
    }

def multilingual_expand(query: str, openai_client=None) -> dict[str, Any]:
    """
    Full multilingual expansion using OpenAI if available,
    falling back to rule-based normalization.
    """
    base = normalize_query(query)

    if openai_client and base["needs_translation"]:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Translate this movie search query to English. "
                        f"Return only the translation, nothing else: '{query}'"
                    )
                }],
                max_tokens=50,
                temperature=0,
            )
            translated = resp.choices[0].message.content.strip()
            base["retrieval_query"] = translated
            base["llm_translation"] = translated
        except Exception:
            pass  # Fall back to rule-based

    return base
