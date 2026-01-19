from __future__ import annotations


def _dcg(rels: list[float]) -> float:
    # DCG = sum (2^rel - 1) / log2(i+2)
    import math

    s = 0.0
    for i, rel in enumerate(rels):
        s += (2.0**rel - 1.0) / math.log2(i + 2)
    return s


def ndcg_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    gains = [float(qrels.get(d, 0)) for d in ranked_doc_ids[:k]]
    dcg = _dcg(gains)
    ideal = sorted([float(v) for v in qrels.values()], reverse=True)[:k]
    idcg = _dcg(ideal)
    return 0.0 if idcg == 0 else dcg / idcg


def average_precision_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    # Binary AP@k: relevance > 0 counts as relevant
    if k <= 0:
        return 0.0
    num_rel = 0
    sum_prec = 0.0
    for i, d in enumerate(ranked_doc_ids[:k], start=1):
        if qrels.get(d, 0) > 0:
            num_rel += 1
            sum_prec += num_rel / i
    total_relevant = sum(1 for v in qrels.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    return sum_prec / min(total_relevant, k)


def recall_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(relevant & retrieved) / len(relevant)
