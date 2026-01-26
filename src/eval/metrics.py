from __future__ import annotations

import math


def dcg(rels: list[int]) -> float:
    s = 0.0
    for i, r in enumerate(rels, start=1):
        s += (2**r - 1) / math.log2(i + 1)
    return s


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """NDCG@k for a single query.
    ranked: ranked doc_ids (best first)
    qrels:  {doc_id: relevance_int}
    """
    rels = [int(qrels.get(d, 0)) for d in ranked[:k]]
    ideal = sorted((int(v) for v in qrels.values()), reverse=True)[:k]
    denom = dcg(ideal)
    return 0.0 if denom == 0.0 else dcg(rels) / denom


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int, *, min_rel: int = 1) -> float:
    relevant = {d for d, r in qrels.items() if int(r) >= min_rel}
    if not relevant:
        return 0.0
    hit = sum(1 for d in ranked[:k] if d in relevant)
    return hit / len(relevant)


def average_precision_at_k(
    ranked: list[str],
    qrels: dict[str, int],
    k: int,
    *,
    min_rel: int = 1,
) -> float:
    relevant = {d for d, r in qrels.items() if int(r) >= min_rel}
    if not relevant:
        return 0.0
    hits = 0
    s = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            hits += 1
            s += hits / i
    return s / len(relevant)


def aggregate_methods_list(
    results: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    *,
    k: int,
    min_rel: int = 1,
    recall_k: int = 100,
) -> dict[str, float]:
    """Aggregate per-query ranked lists -> metrics for one method."""
    ndcgs: list[float] = []
    maps: list[float] = []
    recalls_k: list[float] = []
    recalls_100: list[float] = []

    for qid, ranked in results.items():
        qr = qrels.get(qid, {})
        ndcgs.append(ndcg_at_k(ranked, qr, k))
        maps.append(average_precision_at_k(ranked, qr, k, min_rel=min_rel))
        recalls_k.append(recall_at_k(ranked, qr, k, min_rel=min_rel))
        recalls_100.append(recall_at_k(ranked, qr, recall_k, min_rel=min_rel))

    n = max(1, len(ndcgs))
    return {
        f"ndcg@{k}": float(sum(ndcgs) / n),
        f"map@{k}": float(sum(maps) / n),
        f"recall@{k}": float(sum(recalls_k) / n),
        f"recall@{recall_k}": float(sum(recalls_100) / n),
        "num_queries": float(len(ndcgs)),
    }
