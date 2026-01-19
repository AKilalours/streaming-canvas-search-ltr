# Commit 4: implement hybrid candidate merge
# - union candidates from bm25 + faiss
# - normalize scores
# - weighted combine + dedupe
from __future__ import annotations

from typing import Dict, List, Tuple


def _minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def hybrid_merge(
    bm25: List[Tuple[str, float]],
    faiss: List[Tuple[str, float]],
    alpha: float = 0.5,
) -> List[Tuple[str, float]]:
    """
    Merge two ranked lists by:
      - min-max normalizing scores within each list
      - combined_score = alpha * bm25_norm + (1-alpha) * faiss_norm
    """
    bm25_map = {d: float(s) for d, s in bm25}
    faiss_map = {d: float(s) for d, s in faiss}

    bm25_n = _minmax_norm(bm25_map)
    faiss_n = _minmax_norm(faiss_map)

    keys = set(bm25_n) | set(faiss_n)
    merged = []
    for d in keys:
        s = alpha * bm25_n.get(d, 0.0) + (1.0 - alpha) * faiss_n.get(d, 0.0)
        merged.append((d, s))

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged

