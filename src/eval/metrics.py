# src/eval/metrics.py
from __future__ import annotations

import math
from typing import Dict, List


def dcg(rels: List[int]) -> float:
    s = 0.0
    for i, r in enumerate(rels, start=1):
        s += (2**r - 1) / math.log2(i + 1)
    return s


def ndcg_at_k(qrels: Dict[str, int], ranked: List[str], k: int) -> float:
    rels = [int(qrels.get(d, 0)) for d in ranked[:k]]
    ideal = sorted([int(v) for v in qrels.values()], reverse=True)[:k]
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(rels) / denom


def recall_at_k(qrels: Dict[str, int], ranked: List[str], k: int, min_rel: int = 1) -> float:
    relevant = {d for d, r in qrels.items() if int(r) >= min_rel}
    if not relevant:
        return 0.0
    hit = sum(1 for d in ranked[:k] if d in relevant)
    return hit / len(relevant)


def average_precision_at_k(qrels: Dict[str, int], ranked: List[str], k: int, min_rel: int = 1) -> float:
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
    results: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    *,
    k: int,
    min_rel: int = 1,
    recall_k: int = 100,
) -> Dict[str, float]:
    """
    Aggregate per-query ranked lists -> metrics, shaped for your metrics.json "methods" list.

    results: {qid: [doc_id1, doc_id2, ...]}
    qrels:   {qid: {doc_id: relevance_int}}
    """
    ndcgs: List[float] = []
    maps: List[float] = []
    recalls_k: List[float] = []
    recalls_100: List[float] = []

    for qid, ranked in results.items():
        qr = qrels.get(qid, {})
        ndcgs.append(ndcg_at_k(qr, ranked, k))
        maps.append(average_precision_at_k(qr, ranked, k, min_rel=min_rel))
        recalls_k.append(recall_at_k(qr, ranked, k, min_rel=min_rel))
        recalls_100.append(recall_at_k(qr, ranked, recall_k, min_rel=min_rel))

    n = max(1, len(ndcgs))
    return {
        f"ndcg@{k}": float(sum(ndcgs) / n),
        f"map@{k}": float(sum(maps) / n),
        f"recall@{k}": float(sum(recalls_k) / n),
        f"recall@{recall_k}": float(sum(recalls_100) / n),
        "num_queries": float(len(ndcgs)),
    }
