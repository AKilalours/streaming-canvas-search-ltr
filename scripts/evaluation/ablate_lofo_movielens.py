#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.deps import load_state
from ranking.ltr_infer import LTRReranker
from retrieval.hybrid import hybrid_merge


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _ndcg_at_k(ranked_ids: list[str], relset: set[str], k: int) -> float:
    if not relset:
        return 0.0

    def dcg(ids: list[str]) -> float:
        s = 0.0
        for i, did in enumerate(ids[:k], start=1):
            if did in relset:
                s += 1.0 / math.log2(i + 1)
        return s

    dcg_val = dcg(ranked_ids)
    ideal_hits = min(k, len(relset))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg_val / idcg if idcg > 0 else 0.0


def _ablate_corpus_field(corpus: dict[str, Any], mode: str) -> dict[str, Any]:
    """
    Input-level ablation (robust to any LTR implementation):
      - drop_title: title=""
      - drop_genres: remove 'Genres: ...' segment from text
      - drop_tags:   remove 'Tags: ...' segment from text
      - drop_title_year: strip '(YYYY)' suffix from title (best-effort)
    """
    if mode == "__none__":
        return corpus

    out: dict[str, Any] = {}
    for did, row in corpus.items():
        if not isinstance(row, dict):
            out[did] = row
            continue

        r = dict(row)
        title = str(r.get("title") or "")
        text = str(r.get("text") or "")

        if mode == "drop_title":
            r["title"] = ""
        elif mode == "drop_title_year":
            # strip "(1997)" at end
            if title.endswith(")") and "(" in title:
                base = title.rsplit("(", 1)[0].rstrip()
                r["title"] = base
        elif mode == "drop_genres":
            # keep everything except "Genres: ..."
            parts = [p.strip() for p in text.split("|")]
            parts2 = [p for p in parts if not p.lower().startswith("genres:")]
            r["text"] = " | ".join(parts2)
        elif mode == "drop_tags":
            parts = [p.strip() for p in text.split("|")]
            parts2 = [p for p in parts if not p.lower().startswith("tags:")]
            r["text"] = " | ".join(parts2)

        out[did] = r
    return out


def _score_query(
    *,
    st: Any,
    reranker: Any,
    query: str,
    relset: set[str],
    k_eval: int,
    candidate_k: int,
    rerank_k: int,
    alpha: float,
    ablation: str,
) -> float:
    # retrieval
    bm25 = list(st.bm25_query(query, k=int(candidate_k)))
    dense = list(st.dense.search(query, k=int(candidate_k))) if getattr(st, "dense", None) is not None else []
    merged = hybrid_merge(bm25, dense, alpha=float(alpha))

    # cap rerank pool
    to_rerank = merged[: min(int(rerank_k), len(merged))]

    bm25_scores = {d: float(s) for d, s in bm25}
    dense_scores = {d: float(s) for d, s in dense}

    # score ablations (feature-group ablations)
    corpus = st.corpus
    if ablation in {"drop_title", "drop_title_year", "drop_genres", "drop_tags"}:
        corpus = _ablate_corpus_field(st.corpus, ablation)
    if ablation == "drop_bm25":
        bm25_scores = {}
    if ablation == "drop_dense":
        dense_scores = {}

    # LTR rerank
    try:
        reranked = reranker.rerank(
            query=query,
            corpus=corpus,
            candidates=to_rerank,
            bm25_scores=bm25_scores,
            dense_scores=dense_scores,
        )
    except TypeError:
        # older signature fallback
        reranked = reranker.rerank(query=query, corpus=corpus, candidates=to_rerank)

    reranked_ids = [str(d) for d, _ in reranked]
    # append tail (non-reranked) to form a full list
    reranked_set = set(reranked_ids)
    tail = [str(d) for d, _ in merged if str(d) not in reranked_set]
    ranked_ids = reranked_ids + tail

    return _ndcg_at_k(ranked_ids, relset, int(k_eval))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--candidate_k", type=int, default=1000)
    ap.add_argument("--rerank_k", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--ltr", default="artifacts/ltr/movielens_ltr.pkl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    st = load_state()
    reranker = getattr(st, "reranker", None)
    if reranker is None:
        reranker = LTRReranker.load(str(args.ltr))

    qrows = _read_jsonl(args.queries)
    rrows = _read_jsonl(args.qrels)

    qid2q = {str(q["query_id"]): str(q["text"]) for q in qrows}
    qid2rel: dict[str, set[str]] = defaultdict(set)

    for rr in rrows:
        qid = str(rr.get("query_id"))
        did = str(rr.get("doc_id"))
        # treat any presence as relevant (binary). If you have graded rel, extend later.
        qid2rel[qid].add(did)

    ablations = [
        "__none__",        # baseline
        "drop_bm25",
        "drop_dense",
        "drop_title",
        "drop_title_year",
        "drop_genres",
        "drop_tags",
    ]

    results: list[dict[str, Any]] = []
    baseline_vals: list[float] = []

    # baseline
    for qid, query in qid2q.items():
        relset = qid2rel.get(qid, set())
        if not relset:
            continue
        baseline_vals.append(
            _score_query(
                st=st, reranker=reranker, query=query, relset=relset,
                k_eval=args.k, candidate_k=args.candidate_k, rerank_k=args.rerank_k, alpha=args.alpha,
                ablation="__none__",
            )
        )
    baseline = sum(baseline_vals) / max(1, len(baseline_vals))

    results.append({"ablation": "__none__", f"ndcg@{args.k}": baseline, "delta": 0.0})

    # ablations
    for ab in ablations[1:]:
        vals: list[float] = []
        for qid, query in qid2q.items():
            relset = qid2rel.get(qid, set())
            if not relset:
                continue
            vals.append(
                _score_query(
                    st=st, reranker=reranker, query=query, relset=relset,
                    k_eval=args.k, candidate_k=args.candidate_k, rerank_k=args.rerank_k, alpha=args.alpha,
                    ablation=ab,
                )
            )
        m = sum(vals) / max(1, len(vals))
        results.append({"ablation": ab, f"ndcg@{args.k}": m, "delta": m - baseline})

    # sort: most negative delta = most important signal
    results_sorted = [results[0]] + sorted(results[1:], key=lambda x: float(x["delta"]))

    out = {
        "diagnostics": {
            "num_docs": len(getattr(st, "corpus", {}) or {}),
            "num_queries": len(qid2q),
            "k": args.k,
            "candidate_k": args.candidate_k,
            "rerank_k": args.rerank_k,
            "alpha": args.alpha,
        },
        "results": results_sorted,
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] wrote {p}")


if __name__ == "__main__":
    main()
