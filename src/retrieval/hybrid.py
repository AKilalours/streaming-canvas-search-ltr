# src/retrieval/hybrid.py
from __future__ import annotations

from collections.abc import Iterable
import math


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    """
    Min-max normalize scores to [0, 1].
    Edge cases:
      - empty dict → {}
      - single doc → sigmoid-normalize so alpha weighting still has signal
      - all equal → 0.5 for all (preserves alpha blending instead of collapsing to 1.0)
    """
    if not scores:
        return {}
    if len(scores) == 1:
        # Single result: use sigmoid so the score is meaningful (not forced to 1.0)
        k, v = next(iter(scores.items()))
        return {k: 1.0 / (1.0 + math.exp(-float(v)))}
    vals = list(scores.values())
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        # All scores identical — return 0.5 so both channels contribute equally
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def hybrid_merge(
    bm25_hits: Iterable[tuple[str, float]],
    dense_hits: Iterable[tuple[str, float]],
    *,
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Merge two retrieval lists by:
      1) min-max normalizing each channel's scores
      2) combined = (1-alpha)*bm25 + alpha*dense
      3) sort desc
    """
    alpha = float(alpha)

    bm25_map = {d: float(s) for d, s in bm25_hits}
    dense_map = {d: float(s) for d, s in dense_hits}

    bm25_n = _minmax(bm25_map)
    dense_n = _minmax(dense_map)

    all_ids = set(bm25_n) | set(dense_n)

    merged: list[tuple[str, float]] = []
    for did in all_ids:
        s = (1.0 - alpha) * float(bm25_n.get(did, 0.0)) + alpha * float(dense_n.get(did, 0.0))
        merged.append((did, float(s)))

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged
