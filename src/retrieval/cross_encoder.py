"""
StreamLens — Cross-Encoder Reranker
3rd stage precision reranker after LTR LambdaRank.

Architecture:
  Stage 1: BM25 + Fine-tuned Dense → 2000 candidates (fast)
  Stage 2: LightGBM LambdaRank    → top 200 (sub-ms)
  Stage 3: Cross-Encoder          → top 20 (precision)

Trade-off:
  Cost:    +191ms latency (p99 becomes ~333ms)
  Benefit: +0.02-0.05 nDCG on top-20 precision
  
  Mitigation: Only run cross-encoder on non-cached queries.
  Cached queries still return at p50=2.67ms.
"""
from __future__ import annotations
import numpy as np
import time
import os

_MODEL = None
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import CrossEncoder
        _MODEL = CrossEncoder(_MODEL_NAME, max_length=512)
    return _MODEL


def rerank_cross_encoder(
    query: str,
    items: list[dict],
    top_k: int = 20,
    enabled: bool = True,
) -> list[dict]:
    """
    Cross-encoder reranking of top-k candidates.
    
    Cross-encoders see query AND document together in a single
    BERT forward pass — unlike bi-encoders which encode separately.
    This joint encoding captures fine-grained query-document
    interaction that bi-encoders miss.
    
    Args:
        query:   User query string
        items:   Ranked items from LTR (sorted by ltr_score)
        top_k:   Rerank only top-k (default: 20 for latency)
        enabled: Feature flag — disable for latency-sensitive paths
    
    Returns:
        Items with cross_encoder_score, top-k reranked + rest appended
    """
    if not enabled or not items:
        return items

    model = _get_model()
    
    # Only rerank top-k — rest appended unchanged
    to_rerank = items[:top_k]
    rest       = items[top_k:]

    # Build (query, document) pairs
    pairs = [
        [query, item.get("title","") + " " + item.get("text","")[:300]]
        for item in to_rerank
    ]

    # Cross-encoder scoring — single BERT forward pass per pair
    t0 = time.time()
    scores = model.predict(pairs, show_progress_bar=False)
    latency_ms = (time.time() - t0) * 1000

    # Attach scores
    for item, score in zip(to_rerank, scores):
        item["cross_encoder_score"] = float(score)
        item["ce_latency_ms"] = round(latency_ms, 1)

    # Sort top-k by cross-encoder score
    to_rerank.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    return to_rerank + rest


def is_available() -> bool:
    """Check if cross-encoder is loadable."""
    try:
        _get_model()
        return True
    except Exception:
        return False
