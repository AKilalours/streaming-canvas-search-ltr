"""Isolate why OpenSearch BM25 scored 0.3745 when the repo's rank_bm25 baseline scores 0.6065.

The first OpenSearch run used `multi_match(title^2, text)` over a `film_text` analyzer
(stemming + english stopwords + film synonyms). That is two changes at once against the
baseline, so the regression could not be attributed. This script holds the corpus, the
qrels, the metric code and the index fixed, and varies only the lexical query.

Two axes:
    analyzer : standard | film_text        (indexed as multi-fields on the same document)
    query    : text only | title+text cross_fields | title^2+text best_fields

The kNN run is the control. It was 0.5413 against the FAISS path's 0.5428 on identical
vectors, so indexing and query encoding are known good; anything that moves here is lexical.

    python scripts/opensearch/ablate_lexical.py --out reports/opensearch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.eval.metrics import ndcg_at_k, recall_at_k  # noqa: E402

SPLIT = ROOT / "data/processed/movielens/test"
EMB_DIR = ROOT / "artifacts/faiss/movielens_ft_e5"
INDEX = "movielens-hybrid"

# Reference line from reports/latest/metrics.json, produced by the rank_bm25 path.
BASELINE_BM25_NDCG = 0.6064944718128008
BASELINE_DENSE_NDCG = 0.542812817109545

# name -> (fields, match_type).  A single field uses `match`, several use `multi_match`.
VARIANTS: dict[str, tuple[list[str], str]] = {
    "text_standard":            (["text"], "match"),
    "text_film":                (["text.film"], "match"),
    "title_text_standard_xf":   (["title", "text"], "cross_fields"),
    "title_text_film_xf":       (["title.film", "text.film"], "cross_fields"),
    "title2_text_standard_bf":  (["title^2", "text"], "best_fields"),
    "title2_text_film_bf":      (["title.film^2", "text.film"], "best_fields"),  # original config
}


def lexical_search(client, query: str, fields: list[str], match_type: str, size: int) -> list[str]:
    if match_type == "match":
        q = {"match": {fields[0]: {"query": query}}}
    else:
        q = {"multi_match": {"query": query, "fields": fields, "type": match_type}}
    hits = client.search(index=INDEX, body={"size": size, "_source": False, "query": q})
    return [h["_id"] for h in hits["hits"]["hits"]]


def knn_search(client, vec: np.ndarray, size: int) -> list[str]:
    body = {"size": size, "_source": False,
            "query": {"knn": {"embedding": {"vector": vec.tolist(), "k": size}}}}
    hits = client.search(index=INDEX, body=body)
    return [h["_id"] for h in hits["hits"]["hits"]]


def rrf(runs: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for run in runs:
        for rank, doc_id in enumerate(run, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def score(run: dict[str, list[str]], qrels: dict) -> tuple[float, float]:
    nd = float(np.mean([ndcg_at_k(run[q], qrels.get(q, {}), 10) for q in run]))
    rc = float(np.mean([recall_at_k(run[q], qrels.get(q, {}), 100) for q in run]))
    return nd, rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:9200")
    ap.add_argument("--out", default="reports/opensearch")
    ap.add_argument("--candidate-k", type=int, default=200)
    ap.add_argument("--rrf-k", type=int, default=60)
    args = ap.parse_args()

    client = OpenSearch(hosts=[args.host], timeout=60)
    if not client.indices.exists(index=INDEX):
        raise SystemExit(f"index {INDEX} missing. Run index_corpus.py --recreate first.")

    mapping = client.indices.get_mapping(index=INDEX)[INDEX]["mappings"]["properties"]
    if "film" not in mapping.get("text", {}).get("fields", {}):
        raise SystemExit(
            "index lacks the multi-field analyzers. Re-run:\n"
            "  .venv/bin/python scripts/opensearch/index_corpus.py --recreate"
        )

    queries = [json.loads(l) for l in (SPLIT / "queries.jsonl").open() if l.strip()]
    qrels = json.loads((SPLIT / "qrels.json").read_text())

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(EMB_DIR))
    q_vecs = model.encode([q["text"] for q in queries], normalize_embeddings=True,
                          batch_size=64, show_progress_bar=False)

    knn_run = {q["query_id"]: knn_search(client, q_vecs[i], args.candidate_k)
               for i, q in enumerate(queries)}
    knn_nd, knn_rc = score(knn_run, qrels)

    rows = []
    for name, (fields, mtype) in VARIANTS.items():
        lex_run = {q["query_id"]: lexical_search(client, q["text"], fields, mtype, args.candidate_k)
                   for q in queries}
        lex_nd, lex_rc = score(lex_run, qrels)

        fused = {qid: rrf([lex_run[qid], knn_run[qid]], k=args.rrf_k) for qid in lex_run}
        f_nd, f_rc = score(fused, qrels)

        rows.append({
            "variant": name,
            "fields": fields,
            "match_type": mtype,
            "lexical_ndcg@10": lex_nd,
            "lexical_recall@100": lex_rc,
            "rrf_ndcg@10": f_nd,
            "rrf_recall@100": f_rc,
            "lexical_delta_vs_rank_bm25": lex_nd - BASELINE_BM25_NDCG,
        })

    rows.sort(key=lambda r: r["lexical_ndcg@10"], reverse=True)
    out = {
        "index": INDEX,
        "num_queries": len(queries),
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "control_knn": {"ndcg@10": knn_nd, "recall@100": knn_rc,
                        "faiss_path_reference": BASELINE_DENSE_NDCG},
        "reference_rank_bm25_ndcg@10": BASELINE_BM25_NDCG,
        "variants": rows,
    }

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lexical_ablation.json").write_text(json.dumps(out, indent=2))

    print(f"\nLexical ablation  ({len(queries)} queries, candidate_k={args.candidate_k})")
    print(f"control kNN      nDCG@10 {knn_nd:.4f}   (FAISS path {BASELINE_DENSE_NDCG:.4f})")
    print(f"reference BM25   nDCG@10 {BASELINE_BM25_NDCG:.4f}   (rank_bm25, reports/latest)\n")
    print(f"{'variant':<26}{'lex nDCG':>10}{'vs ref':>9}{'rrf nDCG':>10}{'rrf r@100':>11}")
    for r in rows:
        print(f"{r['variant']:<26}{r['lexical_ndcg@10']:>10.4f}"
              f"{r['lexical_delta_vs_rank_bm25']:>+9.4f}"
              f"{r['rrf_ndcg@10']:>10.4f}{r['rrf_recall@100']:>11.4f}")
    print(f"\nwritten to {out_dir / 'lexical_ablation.json'}")


if __name__ == "__main__":
    main()
