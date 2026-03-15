# scripts/shadow_ab_eval.py
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import requests
import ir_datasets
import pytrec_eval


def load_dataset():
    for did in ["beir/nfcorpus/test", "nfcorpus/test"]:
        try:
            return ir_datasets.load(did), did
        except Exception:
            pass
    raise RuntimeError("Could not load nfcorpus test dataset via ir_datasets.")


def build_qrels(ds):
    qrels = {}
    for r in ds.qrels_iter():
        qrels.setdefault(str(r.query_id), {})[str(r.doc_id)] = int(r.relevance)
    return qrels


def fetch_run(
    api_base: str,
    queries,
    *,
    method: str,
    k: int,
    candidate_k: int,
    rerank_k: int,
    alpha: float,
    sleep_s: float,
    timeout_s: float,
):
    """
    Calls /search?q=... and builds a TREC run dict: run[qid][docid] = score
    """
    run = {}
    for q in queries:
        qid = str(q.query_id)
        qtext = str(q.text)

        params = [
            ("q", qtext),
            ("method", method),
            ("k", str(k)),
            ("candidate_k", str(candidate_k)),
            ("rerank_k", str(rerank_k)),
            ("alpha", str(alpha)),
        ]

        resp = requests.get(f"{api_base}/search", params=params, timeout=timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(
                f"HTTP {resp.status_code} for qid={qid} method={method}\n"
                f"URL={resp.url}\n"
                f"Body={resp.text[:800]}"
            )

        j = resp.json()
        hits = j.get("hits") or j.get("results") or []
        qrun = {}

        for rank, h in enumerate(hits, 1):
            doc_id = h.get("doc_id") or h.get("id") or h.get("document_id")
            score = h.get("score")
            if score is None:
                score = 1.0 / rank
            if doc_id is not None:
                qrun[str(doc_id)] = float(score)

        run[qid] = qrun

        if sleep_s:
            time.sleep(sleep_s)

    return run


def eval_per_query(qrels, run, *, k: int):
    # compute recall@min(100,k) so it's meaningful even if k<100
    r_cut = min(100, int(k))
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {f"ndcg_cut.10", f"map_cut.10", f"recall.10", f"recall.{r_cut}"},
    )
    res = evaluator.evaluate(run)
    ndcg10 = {qid: float(v.get("ndcg_cut_10", 0.0)) for qid, v in res.items()}
    map10 = {qid: float(v.get("map_cut_10", 0.0)) for qid, v in res.items()}
    r10 = {qid: float(v.get("recall_10", 0.0)) for qid, v in res.items()}
    rK = {qid: float(v.get(f"recall_{r_cut}", 0.0)) for qid, v in res.items()}
    return {"per_query": res, "ndcg10": ndcg10, "map10": map10, "recall10": r10, f"recall@{r_cut}": rK}


def mean(d: dict[str, float]) -> float:
    return float(np.mean(list(d.values()))) if d else 0.0


def bootstrap_pvalue(a: dict[str, float], b: dict[str, float], iters=3000, seed=42):
    """
    One-sided bootstrap: p = P(mean(b-a) <= 0)
    """
    rng = random.Random(seed)
    keys = sorted(set(a) & set(b))
    diffs = np.array([b[k] - a[k] for k in keys], dtype=np.float64)
    if len(diffs) == 0:
        return None, 0.0

    observed = float(diffs.mean())
    n = len(diffs)
    cnt = 0
    for _ in range(iters):
        samp = np.array([diffs[rng.randrange(n)] for _ in range(n)], dtype=np.float64)
        if float(samp.mean()) <= 0.0:
            cnt += 1
    return cnt / iters, observed


def top_misses(qrels, run, queries_map, cutoff=50, topn=10):
    misses = []
    for qid, rels in qrels.items():
        if not rels:
            continue

        scored = run.get(qid, {})
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        pos = {doc: i for i, (doc, _) in enumerate(ranked, 1)}

        best = math.inf
        for rel_doc in rels.keys():
            best = min(best, pos.get(rel_doc, math.inf))

        if best > cutoff:
            misses.append(
                {
                    "qid": qid,
                    "query": queries_map.get(qid, ""),
                    "best_relevant_rank": None if best == math.inf else int(best),
                }
            )

    misses.sort(key=lambda x: (x["best_relevant_rank"] is None, x["best_relevant_rank"] or 10**9), reverse=True)
    return misses[:topn]


def top_deltas(a: dict[str, float], b: dict[str, float], queries_map, topn=10):
    keys = sorted(set(a) & set(b))
    deltas = [(qid, float(b[qid] - a[qid])) for qid in keys]
    deltas.sort(key=lambda x: x[1], reverse=True)
    best = [{"qid": q, "query": queries_map.get(q, ""), "delta": d} for q, d in deltas[:topn]]
    worst = [{"qid": q, "query": queries_map.get(q, ""), "delta": d} for q, d in deltas[-topn:]][::-1]
    return {"top_improvements": best, "top_regressions": worst}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--candidate-k", type=int, default=1000)
    ap.add_argument("--rerank-k", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--bootstrap", type=int, default=3000)
    ap.add_argument("--limit", type=int, default=0, help="0 = full dataset; else sample N queries")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/latest_eval/shadow_ab.json")
    args = ap.parse_args()

    ds, did = load_dataset()
    queries_all = list(ds.queries_iter())

    rng = random.Random(args.seed)
    if args.limit and args.limit > 0 and args.limit < len(queries_all):
        queries = rng.sample(queries_all, args.limit)
    else:
        queries = queries_all

    queries_map = {str(q.query_id): str(q.text) for q in queries_all}
    qrels = build_qrels(ds)

    # only evaluate on queries present in qrels to avoid “null relevance” noise
    qids_with_qrels = {qid for qid, rels in qrels.items() if rels}
    queries = [q for q in queries if str(q.query_id) in qids_with_qrels]

    methods = ["bm25", "dense", "hybrid", "hybrid_ltr"]
    runs = {}
    metrics = {}
    perq = {}

    print(f"[shadow] dataset={did} queries_total={len(queries_all)} queries_eval={len(queries)}")
    print(f"[shadow] params: k={args.k} candidate_k={args.candidate_k} rerank_k={args.rerank_k} alpha={args.alpha}")

    for m in methods:
        print(f"[shadow] fetching method={m} queries={len(queries)}")
        run = fetch_run(
            args.api,
            queries,
            method=m,
            k=args.k,
            candidate_k=args.candidate_k,
            rerank_k=args.rerank_k,
            alpha=args.alpha,
            sleep_s=args.sleep,
            timeout_s=args.timeout,
        )
        ev = eval_per_query(qrels, run, k=args.k)
        runs[m] = run
        perq[m] = ev["ndcg10"]
        r_cut = min(100, int(args.k))
        metrics[m] = {
            "ndcg@10": mean(ev["ndcg10"]),
            "map@10": mean(ev["map10"]),
            "recall@10": mean(ev["recall10"]),
            f"recall@{r_cut}": mean(ev[f"recall@{r_cut}"]),
        }
        print(f"[shadow] {m} ndcg@10={metrics[m]['ndcg@10']:.4f}")

    p, lift = bootstrap_pvalue(perq["hybrid"], perq["hybrid_ltr"], iters=args.bootstrap, seed=args.seed)

    out = {
        "dataset_id": did,
        "num_queries_total": len(queries_all),
        "num_queries_eval": len(queries),
        "params": {
            "k": args.k,
            "candidate_k": args.candidate_k,
            "rerank_k": args.rerank_k,
            "alpha": args.alpha,
        },
        "metrics": metrics,
        "bootstrap": {
            "compare": "hybrid_ltr - hybrid (ndcg@10)",
            "iters": args.bootstrap,
            "p_value_one_sided": p,
            "observed_mean_lift": lift,
        },
        "qrels_coverage": {
            "qids_with_qrels": len(qids_with_qrels),
            "qids_total": len(qrels),
        },
        "deltas": top_deltas(perq["hybrid"], perq["hybrid_ltr"], queries_map, topn=10),
        "top_misses": {
            "hybrid": top_misses(qrels, runs["hybrid"], queries_map, cutoff=50, topn=10),
            "hybrid_ltr": top_misses(qrels, runs["hybrid_ltr"], queries_map, cutoff=50, topn=10),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n wrote {args.out}")


if __name__ == "__main__":
    main()
