# scripts/eval_offline_movielens.py
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from app.deps import load_state
from retrieval.hybrid import hybrid_merge
from ranking.ltr_infer import LTRReranker


def _dcg(rels: List[int], k: int) -> float:
    s = 0.0
    for i, r in enumerate(rels[:k], start=1):
        if r <= 0:
            continue
        s += 1.0 / math.log2(i + 1)
    return s


def _ndcg_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int) -> float:
    rels = [int(qrels.get(d, 0)) for d in ranked_doc_ids[:k]]
    dcg = _dcg(rels, k)
    ideal_rels = sorted([int(v) for v in qrels.values() if int(v) > 0], reverse=True)
    idcg = _dcg(ideal_rels, k)
    return 0.0 if idcg <= 0 else (dcg / idcg)


def _ap_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int) -> float:
    hits = 0
    s = 0.0
    for i, d in enumerate(ranked_doc_ids[:k], start=1):
        if int(qrels.get(d, 0)) > 0:
            hits += 1
            s += hits / i
    denom = max(1, sum(1 for v in qrels.values() if int(v) > 0))
    return s / denom


def _recall_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int) -> float:
    relevant = {d for d, r in qrels.items() if int(r) > 0}
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def _load_queries(p: Path) -> List[Tuple[str, str]]:
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out.append((str(row["query_id"]), str(row["text"])))
    return out


def _load_qrels(p: Path) -> Dict[str, Dict[str, int]]:
    q = defaultdict(dict)
    for line in p.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        qid = str(row["query_id"])
        did = str(row["doc_id"])
        rel = int(row.get("relevance", 1))
        q[qid][did] = rel
    return dict(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k10", type=int, default=10)
    ap.add_argument("--k100", type=int, default=100)
    ap.add_argument("--candidate_k", type=int, default=None)
    ap.add_argument("--rerank_k", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    args = ap.parse_args()

    st = load_state()

    # Pull defaults from config if not overridden
    cfg = getattr(st, "cfg", None)  # if you stored cfg on AppState; safe if missing
    candidate_k = args.candidate_k or 1000
    rerank_k = args.rerank_k or 200
    alpha = args.alpha if args.alpha is not None else 0.5

    has_ltr = getattr(st, "ltr_path", None) is not None and Path(str(st.ltr_path)).exists()
    reranker = None
    if has_ltr:
        reranker = getattr(st, "reranker", None)
        if reranker is None:
            reranker = LTRReranker.load(str(st.ltr_path))

    queries = _load_queries(Path(args.queries))
    qrels_all = _load_qrels(Path(args.qrels))

    methods = ["bm25", "dense", "hybrid", "hybrid_ltr"]
    metrics = {m: {"ndcg@10": [], "map@10": [], "recall@10": [], "recall@100": [], "oracle_ndcg@10": []} for m in methods}

    for qid, qtext in queries:
        qrels = qrels_all.get(qid, {})
        if not qrels:
            continue

        bm25 = list(st.bm25_query(qtext, k=candidate_k))
        dense = list(st.dense.search(qtext, k=candidate_k)) if st.dense else []
        merged = hybrid_merge(bm25, dense, alpha=alpha)

        bm25_ids = [d for d, _ in bm25]
        dense_ids = [d for d, _ in dense]
        hybrid_ids = [d for d, _ in merged]

        # Oracle upper bound given the merged candidate set:
        cand_ids = hybrid_ids[: max(1, rerank_k)]
        oracle_sorted = sorted(cand_ids, key=lambda d: int(qrels.get(str(d), 0)), reverse=True)
        oracle = _ndcg_at_k([str(d) for d in oracle_sorted], qrels, args.k10)

        def add(m: str, ranked: List[str]) -> None:
            metrics[m]["ndcg@10"].append(_ndcg_at_k(ranked, qrels, args.k10))
            metrics[m]["map@10"].append(_ap_at_k(ranked, qrels, args.k10))
            metrics[m]["recall@10"].append(_recall_at_k(ranked, qrels, args.k10))
            metrics[m]["recall@100"].append(_recall_at_k(ranked, qrels, args.k100))
            metrics[m]["oracle_ndcg@10"].append(oracle)

        add("bm25", [str(x) for x in bm25_ids])
        add("dense", [str(x) for x in dense_ids])
        add("hybrid", [str(x) for x in hybrid_ids])

        if reranker is not None:
            bm25_map = {str(d): float(s) for d, s in bm25}
            dense_map = {str(d): float(s) for d, s in dense}
            to_rerank = [(str(d), float(s)) for d, s in merged[:rerank_k]]
            reranked = reranker.rerank(
                query=qtext,
                corpus=st.corpus,
                candidates=to_rerank,
                bm25_scores=bm25_map,
                dense_scores=dense_map,
            )
            reranked_ids = [d for d, _ in reranked]
            tail = [str(d) for d in hybrid_ids if str(d) not in set(reranked_ids)]
            hybrid_ltr_ids = reranked_ids + tail
            add("hybrid_ltr", [str(x) for x in hybrid_ltr_ids])
        else:
            add("hybrid_ltr", [str(x) for x in hybrid_ids])

    def mean(xs: List[float]) -> float:
        return float(sum(xs) / max(1, len(xs)))

    out = {
        "split": "test",
        "k": args.k10,
        "diagnostics": {
            "num_docs": len(getattr(st, "corpus", {})),
            "num_queries": len(metrics["hybrid"]["ndcg@10"]),
            "has_ltr": bool(reranker is not None),
            "candidate_k": candidate_k,
            "rerank_k": rerank_k,
            "alpha": alpha,
        },
        "methods": [],
    }

    for m in methods:
        out["methods"].append(
            {
                "method": m,
                "ndcg@10": mean(metrics[m]["ndcg@10"]),
                "map@10": mean(metrics[m]["map@10"]),
                "recall@10": mean(metrics[m]["recall@10"]),
                "recall@100": mean(metrics[m]["recall@100"]),
                "num_queries": float(len(metrics[m]["ndcg@10"])),
                "oracle_ndcg@10": mean(metrics[m]["oracle_ndcg@10"]),
            }
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] wrote {args.out}")
    # Print lift for convenience
    ms = {row["method"]: row for row in out["methods"]}
    lift = ms["hybrid_ltr"]["ndcg@10"] - ms["hybrid"]["ndcg@10"]
    print(f"[OK] lift_ndcg@10 = {lift:.6f} (hybrid_ltr - hybrid)")


if __name__ == "__main__":
    main()
