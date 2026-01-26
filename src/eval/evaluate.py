# src/eval/evaluate.py
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from eval.metrics import aggregate_methods_list
from ranking.ltr_infer import LTRReranker
from retrieval.hybrid import hybrid_merge
from utils.io import read_json, read_jsonl, write_json
from utils.logging import get_logger

log = get_logger("eval.evaluate")


def _pick_paths(cfg: dict[str, Any]) -> dict[str, str]:
    """
    Supports BOTH:
      A) cfg["paths"] = {corpus_path, queries_path, qrels_path, ...}
      B) cfg["dataset"] / cfg["artifacts"] / cfg["eval"] style (your current)
    """
    # Style A: explicit paths
    if "paths" in cfg:
        p = cfg["paths"]
        return {
            "corpus_path": p["corpus_path"],
            "queries_path": p["queries_path"],
            "qrels_path": p["qrels_path"],
            "bm25_artifact": p.get("bm25_artifact") or cfg["artifacts"]["bm25_artifact"],
            "emb_dir": p.get("emb_dir") or cfg["artifacts"]["emb_dir"],
            "ltr_path": p.get("ltr_path")
            or cfg["artifacts"].get("ltr_path", "artifacts/ltr/ltr.pkl"),
        }

    # Style B: dataset/artifacts/eval
    dataset = cfg.get("dataset", {})
    artifacts = cfg.get("artifacts", {})

    # Explicit file paths preferred
    corpus_path = dataset.get("corpus_path")
    queries_path = dataset.get("queries_path")
    qrels_path = dataset.get("qrels_path")

    # Or infer from processed_dir + split
    if not (corpus_path and queries_path and qrels_path):
        processed_dir = Path(dataset.get("processed_dir", "data/processed"))
        split = dataset.get("split", "scifact/test")
        corpus_path = str(processed_dir / split / "corpus.jsonl")
        queries_path = str(processed_dir / split / "queries.jsonl")
        qrels_path = str(processed_dir / split / "qrels.json")

    return {
        "corpus_path": corpus_path,
        "queries_path": queries_path,
        "qrels_path": qrels_path,
        "bm25_artifact": artifacts.get("bm25_artifact", "artifacts/bm25/scifact_bm25.pkl"),
        "emb_dir": artifacts.get(
            "emb_dir", "artifacts/faiss/scifact_sentence-transformers_all-MiniLM-L6-v2"
        ),
        "ltr_path": artifacts.get("ltr_path", "artifacts/ltr/ltr.pkl"),
    }


def _dense_topk(
    model: SentenceTransformer,
    query: str,
    doc_embs: np.ndarray,
    doc_ids: list[str],
    k: int,
) -> list[tuple[str, float]]:
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[
        0
    ]
    scores = doc_embs @ q
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(doc_ids[int(i)], float(scores[int(i)])) for i in idx]


def _oracle_ndcg_at_k(
    qrels: dict[str, int],
    candidates: list[str],
    k: int,
) -> float:
    # Best possible ranking restricted to the candidate set
    rels = [(d, int(qrels.get(d, 0))) for d in candidates]
    rels.sort(key=lambda x: x[1], reverse=True)
    oracle_ranked = [d for d, _ in rels[:k]]

    # reuse metric from aggregate_methods_list via a small inline ndcg
    # (keeps evaluate.py self-contained)
    import math

    def dcg(vals: list[int]) -> float:
        s = 0.0
        for i, r in enumerate(vals, start=1):
            s += (2**r - 1) / math.log2(i + 1)
        return s

    got = [int(qrels.get(d, 0)) for d in oracle_ranked]
    ideal = sorted([int(v) for v in qrels.values()], reverse=True)[:k]
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(got) / denom


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = _pick_paths(cfg)

    eval_cfg = cfg.get("eval", {})
    k = int(eval_cfg.get("k", 10))
    cand_k = int(eval_cfg.get("candidates_k", 200))
    rerank_k = int(eval_cfg.get("rerank_k", 50))
    split_name = str(eval_cfg.get("split", cfg.get("dataset", {}).get("split", "unknown")))

    corpus_rows = read_jsonl(paths["corpus_path"])
    queries_rows = read_jsonl(paths["queries_path"])
    qrels_all = read_json(paths["qrels_path"])

    corpus = {r["doc_id"]: r for r in corpus_rows}
    queries = {r["query_id"]: r["text"] for r in queries_rows}

    # BM25 artifact (your object uses bm25.query(query, k=...))
    with Path(paths["bm25_artifact"]).open("rb") as f:
        bm25 = pickle.load(f)

    # Dense embeddings
    emb_dir = Path(paths["emb_dir"])
    doc_embs = np.load(emb_dir / "embeddings.npy").astype(np.float32)
    doc_ids = json.loads((emb_dir / "doc_ids.json").read_text(encoding="utf-8"))

    # Dense embed model name
    embed_model_name = cfg.get("artifacts", {}).get("embed_model")
    if not embed_model_name:
        # heuristic from directory name: scifact_sentence-transformers_all-MiniLM-L6-v2
        name = emb_dir.name
        if "sentence-transformers_" in name:
            embed_model_name = name.split("sentence-transformers_", 1)[1].replace("_", "/")
        else:
            embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embed_model = SentenceTransformer(embed_model_name)

    # Optional reranker
    ltr_path = Path(paths["ltr_path"])
    reranker: LTRReranker | None = LTRReranker.load(str(ltr_path)) if ltr_path.exists() else None
    if reranker:
        log.info("Loaded reranker: %s", ltr_path)

    runs_ranked: dict[str, dict[str, list[str]]] = {
        "bm25": {},
        "dense": {},
        "hybrid": {},
    }
    if reranker:
        runs_ranked["hybrid_ltr"] = {}

    # Diagnostics
    oracle: dict[str, list[float]] = {m: [] for m in runs_ranked.keys()}

    for qid, qtext in queries.items():
        # BM25
        if hasattr(bm25, "query"):
            bm25_hits = bm25.query(qtext, k=cand_k)
        else:
            bm25_hits = bm25.search(qtext, k=cand_k)  # fallback
        bm25_ids = [d for d, _ in bm25_hits]

        # Dense
        dense_hits = _dense_topk(embed_model, qtext, doc_embs, doc_ids, cand_k)
        dense_ids = [d for d, _ in dense_hits]

        # Hybrid
        hy_hits = hybrid_merge(bm25_hits, dense_hits, alpha=float(eval_cfg.get("alpha", 0.5)))
        hy_ids = [d for d, _ in hy_hits]

        runs_ranked["bm25"][qid] = bm25_ids[:k]
        runs_ranked["dense"][qid] = dense_ids[:k]
        runs_ranked["hybrid"][qid] = hy_ids[:k]

        qr = qrels_all.get(qid, {})

        oracle["bm25"].append(_oracle_ndcg_at_k(qr, bm25_ids[:cand_k], k))
        oracle["dense"].append(_oracle_ndcg_at_k(qr, dense_ids[:cand_k], k))
        oracle["hybrid"].append(_oracle_ndcg_at_k(qr, hy_ids[:cand_k], k))

        if reranker:
            # rerank top rerank_k hybrid candidates
            to_rerank = hy_hits[: min(rerank_k, len(hy_hits))]
            bm25_map = {d: float(s) for d, s in bm25_hits}
            dense_map = {d: float(s) for d, s in dense_hits}

            reranked = reranker.rerank(
                query=qtext,
                corpus=corpus,
                candidates=to_rerank,
                bm25_scores=bm25_map,
                dense_scores=dense_map,
            )
            reranked_ids = [d for d, _ in reranked]
            runs_ranked["hybrid_ltr"][qid] = reranked_ids[:k]  # type: ignore[index]

            oracle["hybrid_ltr"].append(_oracle_ndcg_at_k(qr, reranked_ids[:rerank_k], k))  # type: ignore[index]

    # Aggregate into list-based "methods" format
    methods_out: list[dict[str, Any]] = []
    for method_name, res in runs_ranked.items():
        agg = aggregate_methods_list(res, qrels_all, k=k, min_rel=1, recall_k=100)
        methods_out.append(
            {
                "method": method_name,
                **agg,
                "oracle_ndcg@10": float(
                    sum(oracle[method_name]) / max(1, len(oracle[method_name]))
                ),
                "split": split_name,
            }
        )

    out = {
        "k": k,
        "split": split_name,
        "diagnostics": {
            "num_docs": len(corpus),
            "num_queries": len(queries),
            "has_ltr": bool(reranker),
            "embed_model": embed_model_name,
            "candidate_k": cand_k,
            "rerank_k": rerank_k,
        },
        "methods": methods_out,
    }

    write_json("reports/latest/metrics.json", out)
    log.info("Wrote reports/latest/metrics.json (methods=%s)", [m["method"] for m in methods_out])


if __name__ == "__main__":
    main()
