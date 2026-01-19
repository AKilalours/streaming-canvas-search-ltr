from __future__ import annotations


def _minmax_norm(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def hybrid_merge(
    bm25: list[tuple[str, float]],
    dense: list[tuple[str, float]],
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Merge two ranked lists by:
      - min-max normalizing scores within each list
      - combined_score = alpha * bm25_norm + (1-alpha) * dense_norm
    """
    bm25_map = {d: float(s) for d, s in bm25}
    dense_map = {d: float(s) for d, s in dense}

    bm25_n = _minmax_norm(bm25_map)
    dense_n = _minmax_norm(dense_map)

    keys = set(bm25_n) | set(dense_n)
    merged: list[tuple[str, float]] = []
    for d in keys:
        s = alpha * bm25_n.get(d, 0.0) + (1.0 - alpha) * dense_n.get(d, 0.0)
        merged.append((d, s))

    merged.sort(key=lambda x: x[1], reverse=True)
    return merged

