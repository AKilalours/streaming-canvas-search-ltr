# src/ranking/features.py
from __future__ import annotations

import math
import re
from typing import Any

# IMPORTANT: this order must be used at TRAINING and INFERENCE
FEATURE_NAMES: list[str] = [
    "bm25_score",
    "dense_score",
    "hybrid_score",
    "bm25_log1p",
    "dense_log1p",
    "hybrid_log1p",
    "query_len",
    "query_unique",
    "doc_len",
    "doc_unique",
    "title_len",
    "title_overlap",
    "text_overlap",
    "title_jaccard",
    "text_jaccard",
]

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _overlap_and_jaccard(qt: set[str], dt: set[str]) -> tuple[float, float]:
    if not qt or not dt:
        return 0.0, 0.0
    inter = len(qt & dt)
    union = len(qt | dt)
    return float(inter), float(inter / union) if union else 0.0


def build_features(
    *,
    query: str,
    doc: dict[str, Any],
    bm25_score: float,
    dense_score: float,
    hybrid_score: float,
) -> dict[str, float]:
    title = str(doc.get("title", "") or "")
    text = str(doc.get("text", "") or "")

    q_toks = _tokens(query)
    t_toks = _tokens(title)
    d_toks = _tokens(text)

    q_set = set(q_toks)
    title_set = set(t_toks)
    doc_set = set(d_toks)

    title_overlap, title_jaccard = _overlap_and_jaccard(q_set, title_set)
    text_overlap, text_jaccard = _overlap_and_jaccard(q_set, doc_set)

    feats: dict[str, float] = {
        "bm25_score": float(bm25_score),
        "dense_score": float(dense_score),
        "hybrid_score": float(hybrid_score),
        "bm25_log1p": float(math.log1p(max(bm25_score, 0.0))),
        "dense_log1p": float(math.log1p(max(dense_score, 0.0))),
        "hybrid_log1p": float(math.log1p(max(hybrid_score, 0.0))),
        "query_len": float(len(q_toks)),
        "query_unique": float(len(q_set)),
        "doc_len": float(len(d_toks)),
        "doc_unique": float(len(doc_set)),
        "title_len": float(len(t_toks)),
        "title_overlap": float(title_overlap),
        "text_overlap": float(text_overlap),
        "title_jaccard": float(title_jaccard),
        "text_jaccard": float(text_jaccard),
    }

    # Ensure all expected features exist (defensive)
    for name in FEATURE_NAMES:
        feats.setdefault(name, 0.0)

    return feats
