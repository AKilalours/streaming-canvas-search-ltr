from __future__ import annotations

import re
from typing import Optional

import langid


def _clean_lang_tag(tag: str) -> str:
    # Accept "en", "en-US", "pt-BR", etc.
    tag = tag.strip()
    tag = re.split(r"[;, ]+", tag)[0]
    tag = tag.replace("_", "-")
    return tag


def normalize_lang(tag: Optional[str]) -> str:
    """
    Return a BCP-47-ish tag like 'en' or 'es' or 'hi' (we keep region if provided).
    """
    if not tag:
        return "en"
    tag = _clean_lang_tag(tag)
    if not tag:
        return "en"
    return tag


def detect_lang(text: str) -> str:
    """
    langid returns ISO639-1 like 'en', 'es', 'hi', etc.
    """
    if not text or not text.strip():
        return "en"
    code, _score = langid.classify(text)
    return normalize_lang(code)
