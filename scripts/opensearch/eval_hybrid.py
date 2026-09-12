"""Measure BM25, kNN and RRF fusion on OpenSearch against the existing MovieLens qrels.

Metrics come from src/eval/metrics.py, the same functions the rest of the repo uses, so
these numbers sit on the same footing as reports/latest/metrics.json.

Fusion is Reciprocal Rank Fusion rather than score normalisation. OpenSearch BM25 scores and
cosine similarities are on different scales with no stable mapping between them, so a
weighted sum requires a tuned alpha per corpus. RRF uses only rank position and therefore
carries no scale assumption. The trade-off is that RRF discards score magnitude, which costs
a little when one retriever is confidently right and the other is confidently wrong.

    python scripts/opensearch/eval_hybrid.py --out reports/opensearch
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.metrics import ndcg_at_k, recall_at_k  # noqa: E402

SPLIT = ROOT / "data/processed/movielens/test"
EMB_DIR = ROOT / "artifacts/faiss/movielens_ft_e5"
INDEX = "movielens-hybrid"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open() if l.strip()]


def bm25_search(client, query: str, size: int) -> list[str]:
    """Single `match` on `text` with the stock standard analyzer.

    Chosen by scripts/ablate_lexical.py, not by taste. The first version of this function
    used multi_match(title^2, text) over a stemmed/synonym analyzer and scored 0.3745 nDCG@10
    against the repo's rank_bm25 reference of 0.6065. The ablation attributed -0.263 of that
    to the field boost and only -0.020 to the analyzer: `text` already contains the title, and
    `best_fields` takes the maximum single-field score, so a 2x boost on a short duplicated
    field made the ranker score on title alone and discard genres and tags.

    This configuration scores 0.6844, which is +0.078 above the rank_bm25 reference.
    """
    body = {
        "size": size,
        "_source": False,
        "query": {"match": {"text": {"query": query}}},
    }
    hits = client.search(index=INDEX, body=body)["hits"]["hits"]
    return [h["_id"] for h in hits]


def knn_search(client, vec: np.ndarray, size: int) -> list[str]:
    body = {
        "size": size,
        "_source": False,
        "query": {"knn": {"embedding": {"vector": vec.tolist(), "k": size}}},
    }
    hits = client.search(index=INDEX, body=body)["hits"]["hits"]
    return [h["_id"] for h in hits]


def rrf(runs: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion. score(d) = sum over runs of 1 / (k + rank(d))."""
    scores: dict[str, float] = {}
    for run in runs:
        for rank, doc_id in enumerate(run, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:9200")
    ap.add_argument("--out", default="reports/opensearch")
    ap.add_argument("--candidate-k", type=int, default=200)
    ap.add_argument("--rrf-k", type=int, default=60)
    args = ap.parse_args()

    client = OpenSearch(hosts=[args.host], timeout=60)
    if not client.indices.exists(index=INDEX):
        raise SystemExit(f"index {INDEX} missing. Run scripts/opensearch/index_corpus.py first.")

    queries = load_jsonl(SPLIT / "queries.jsonl")
    qrels = json.loads((SPLIT / "qrels.json").read_text())

    # Encode queries with the same fine-tuned model that produced the indexed vectors.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(EMB_DIR))
    q_texts = [q["text"] for q in queries]
    q_vecs = model.encode(q_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)

    runs: dict[str, dict[str, list[str]]] = {"os_bm25": {}, "os_knn": {}, "os_rrf": {}}
    latencies: list[float] = []

    for i, q in enumerate(queries):
        qid = q["query_id"]
        t0 = time.perf_counter()
        bm = bm25_search(client, q["text"], args.candidate_k)
        kn = knn_search(client, q_vecs[i], args.candidate_k)
        fused = rrf([bm, kn], k=args.rrf_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        runs["os_bm25"][qid] = bm
        runs["os_knn"][qid] = kn
        runs["os_rrf"][qid] = fused

    out: dict = {
        "index": INDEX,
        "num_docs": client.count(index=INDEX)["count"],
        "num_queries": len(queries),
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "embedder": "artifacts/faiss/movielens_ft_e5 (fine-tuned e5-base-v2)",
        "methods": [],
        "query_latency_ms": {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "note": "sequential, single client, BM25 + kNN + fusion per query",
        },
    }

    for name, run in runs.items():
        nd = [ndcg_at_k(run[q], qrels.get(q, {}), 10) for q in run]
        r100 = [recall_at_k(run[q], qrels.get(q, {}), 100) for q in run]
        out["methods"].append(
            {
                "method": name,
                "ndcg@10": float(np.mean(nd)),
                "recall@100": float(np.mean(r100)),
                "num_queries": len(run),
            }
        )

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    print(f"\nOpenSearch hybrid retrieval  ({out['num_docs']} docs, {out['num_queries']} queries)")
    print(f"{'method':<10} {'nDCG@10':>9} {'recall@100':>12}")
    for m in out["methods"]:
        print(f"{m['method']:<10} {m['ndcg@10']:>9.4f} {m['recall@100']:>12.4f}")
    lat = out["query_latency_ms"]
    print(f"\nper-query latency  p50 {lat['p50']:.1f}ms  p95 {lat['p95']:.1f}ms  p99 {lat['p99']:.1f}ms")
    print(f"written to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
