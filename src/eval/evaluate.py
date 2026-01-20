from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from eval.metrics import average_precision_at_k, ndcg_at_k, recall_at_k
from ranking.ltr_infer import LTRReranker
from retrieval.hybrid import hybrid_merge
from utils.io import ensure_dir, read_json, read_jsonl, write_json
from utils.logging import get_logger

log = get_logger("eval.evaluate")


def _load_split(processed_dir: Path, split: str) -> tuple[dict[str, str], dict]:
    split_dir = processed_dir / split
    queries_rows = read_jsonl(split_dir / "queries.jsonl")
    qrels_all = read_json(split_dir / "qrels.json")
    queries_map = {r["query_id"]: r["text"] for r in queries_rows}
    return queries_map, qrels_all


def _load_corpus(processed_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(processed_dir / split / "corpus.jsonl")
    return {r["doc_id"]: r for r in rows}


def _compute_metrics(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> dict[str, float]:
    return {
        f"ndcg@{k}": ndcg_at_k(ranked_doc_ids, qrels, k),
        f"map@{k}": average_precision_at_k(ranked_doc_ids, qrels, k),
        f"recall@{k}": recall_at_k(ranked_doc_ids, qrels, k),
    }


def _oracle_ndcg_at_k(scores_by_doc: dict[str, float], qrels: dict[str, int], k: int) -> float:
    """
    Upper bound NDCG@k obtainable if a perfect reranker could sort any candidate list.
    We approximate by sorting candidates by true relevance (qrels) then scoring with ndcg_at_k.
    """
    if not scores_by_doc:
        return 0.0

    cand_doc_ids = list(scores_by_doc.keys())
    cand_doc_ids.sort(
        key=lambda d: (int(qrels.get(d, 0)), float(scores_by_doc.get(d, 0.0))),
        reverse=True,
    )
    return ndcg_at_k(cand_doc_ids[:k], qrels, k)


def _bm25_retrieve(bm25_obj, queries_map: dict[str, str], k: int) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    for qid, qtext in queries_map.items():
        out[qid] = bm25_obj.query(qtext, k=k)
    return out


def _dense_load(emb_dir: Path) -> tuple[np.ndarray, list[str]]:
    emb_path = emb_dir / "embeddings.npy"
    ids_path = emb_dir / "doc_ids.json"
    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Missing dense artifacts in {emb_dir}. Run `make index-faiss`.")
    embs = np.load(emb_path).astype(np.float32)
    doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    return embs, doc_ids


def _topk_dot(q_emb: np.ndarray, doc_embs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = q_emb @ doc_embs.T  # [Q, N]
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    s = scores[row, idx]
    order = np.argsort(-s, axis=1)
    idx = idx[row, order]
    s = s[row, order]
    return s, idx


def _dense_retrieve(
    queries_map: dict[str, str],
    model_name: str,
    emb_dir: Path,
    k: int,
) -> dict[str, list[tuple[str, float]]]:
    doc_embs, doc_ids = _dense_load(emb_dir)

    model = SentenceTransformer(model_name)
    qids = list(queries_map.keys())
    qtexts = [queries_map[qid] for qid in qids]

    q_emb = model.encode(
        qtexts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, idxs = _topk_dot(q_emb, doc_embs, k)
    out: dict[str, list[tuple[str, float]]] = {}
    for i, qid in enumerate(qids):
        hits = [(doc_ids[int(idxs[i, j])], float(scores[i, j])) for j in range(scores.shape[1])]
        out[qid] = hits
    return out


def _dense_single_query(
    model: SentenceTransformer,
    query: str,
    doc_embs: np.ndarray,
    doc_ids: list[str],
    k: int,
) -> list[tuple[str, float]]:
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
    scores = doc_embs @ q_emb
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(doc_ids[int(i)], float(scores[int(i)])) for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/eval.yaml")
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Optional override for output dir. If set, writes to <out_dir>/metrics.json etc.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    processed_dir = Path(cfg["eval"]["dataset_processed_dir"])
    split = str(cfg["eval"]["split"])
    k = int(cfg["eval"]["k"])

    diagnostics = cfg["eval"].get("diagnostics", {}) if isinstance(cfg["eval"], dict) else {}
    recall_k = int(diagnostics.get("recall_k", 100))
    oracle_k = int(diagnostics.get("oracle_k", k))

    need_k = max(k, recall_k)  # CRITICAL: must have >= recall_k hits to compute recall@recall_k

    queries_map, qrels_all = _load_split(processed_dir, split)
    log.info("Loaded %d queries for split=%s", len(queries_map), split)
    log.info("Eval k=%d | recall_k=%d | need_k=%d | oracle_k=%d", k, recall_k, need_k, oracle_k)

    if args.out_dir:
        out_dir = ensure_dir(Path(args.out_dir))
    else:
        out_dir = ensure_dir(Path(cfg["reporting"]["out_dir"]) / "latest_eval")

    # Optionally load corpus for LTR branch (only if needed)
    corpus_cache: dict[str, dict[str, Any]] | None = None

    # BM25 artifact (needed for bm25/hybrid/hybrid_ltr)
    bm25_obj = None
    bm25_methods = [m for m in cfg["methods"] if m["type"] in ("bm25", "hybrid", "hybrid_ltr")]
    if bm25_methods:
        bm25_path = Path(bm25_methods[0]["bm25_artifact"])
        with bm25_path.open("rb") as f:
            bm25_obj = pickle.load(f)

    # Dense caches
    dense_cache: dict[str, dict[str, list[tuple[str, float]]]] = {}
    dense_model_cache: dict[str, SentenceTransformer] = {}
    dense_doc_cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    method_summaries: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []

    for method in cfg["methods"]:
        name = method["name"]
        mtype = method["type"]

        # CANDIDATES must be >= need_k for recall@recall_k to be meaningful.
        if mtype == "bm25":
            if bm25_obj is None:
                raise RuntimeError("BM25 artifact not loaded but bm25 method requested.")
            cand = _bm25_retrieve(bm25_obj, queries_map, k=need_k)

        elif mtype == "dense":
            model_name = method["model_name"]
            emb_dir = Path(method["emb_dir"])
            cand_k = max(int(method.get("candidate_k", k)), need_k)
            cache_key = f"{model_name}::{emb_dir}::{cand_k}"
            if cache_key not in dense_cache:
                dense_cache[cache_key] = _dense_retrieve(queries_map, model_name, emb_dir, cand_k)
            cand = dense_cache[cache_key]

        elif mtype == "hybrid":
            if bm25_obj is None:
                raise RuntimeError("BM25 artifact not loaded but hybrid method requested.")
            alpha = float(method.get("alpha", 0.5))
            bm25_k = max(int(method.get("bm25_candidate_k", 50)), need_k)
            dense_k = max(int(method.get("dense_candidate_k", 50)), need_k)
            model_name = method["model_name"]
            emb_dir = Path(method["emb_dir"])

            bm25_c = _bm25_retrieve(bm25_obj, queries_map, k=bm25_k)

            cache_key = f"{model_name}::{emb_dir}::{dense_k}"
            if cache_key not in dense_cache:
                dense_cache[cache_key] = _dense_retrieve(queries_map, model_name, emb_dir, dense_k)
            dense_c = dense_cache[cache_key]

            cand = {}
            for qid in queries_map:
                merged = hybrid_merge(bm25_c[qid], dense_c[qid], alpha=alpha)
                cand[qid] = merged[:need_k]

        elif mtype == "hybrid_ltr":
            if bm25_obj is None:
                raise RuntimeError("BM25 artifact not loaded but hybrid_ltr method requested.")

            alpha = float(method.get("alpha", 0.5))
            rerank_k = int(method.get("rerank_k", 50))
            model_name = method["model_name"]
            emb_dir = Path(method["emb_dir"])
            ltr_model_path = Path(method["ltr_model_path"])
            corpus_split_for_features = str(method.get("corpus_split_for_features", "train"))

            # Must retrieve enough to compute recall@recall_k and provide a stable tail
            bm25_k = max(int(method.get("bm25_candidate_k", 200)), need_k, rerank_k)
            dense_k = max(int(method.get("dense_candidate_k", 200)), need_k, rerank_k)

            # Load corpus once
            if corpus_cache is None:
                corpus_cache = _load_corpus(processed_dir, corpus_split_for_features)

            # Load dense docs once
            dd_key = f"{emb_dir}"
            if dd_key not in dense_doc_cache:
                dense_doc_cache[dd_key] = _dense_load(emb_dir)
            doc_embs, doc_ids = dense_doc_cache[dd_key]

            # Load dense model once
            if model_name not in dense_model_cache:
                dense_model_cache[model_name] = SentenceTransformer(model_name)
            dense_model = dense_model_cache[model_name]

            # Load LTR once
            reranker = LTRReranker.load(str(ltr_model_path))

            cand = {}
            for qid, qtext in queries_map.items():
                bm25_hits = bm25_obj.query(qtext, k=bm25_k)
                dense_hits = _dense_single_query(dense_model, qtext, doc_embs, doc_ids, dense_k)
                merged = hybrid_merge(bm25_hits, dense_hits, alpha=alpha)

                # candidate set for reranking
                to_rerank = merged[:rerank_k]
                bm25_scores = {d: s for d, s in bm25_hits}
                dense_scores = {d: s for d, s in dense_hits}

                reranked = reranker.rerank(
                    query=qtext,
                    corpus=corpus_cache,
                    candidates=to_rerank,
                    bm25_scores=bm25_scores,
                    dense_scores=dense_scores,
                )

                # Fill remaining with the non-reranked tail to keep output length stable
                reranked_ids = {d for d, _ in reranked}
                tail = [(d, s) for d, s in merged if d not in reranked_ids]
                final = reranked + tail
                cand[qid] = final[:need_k]

        else:
            raise ValueError(f"Unknown method type: {mtype}")

        # Aggregate metrics + per-query rows
        ndcgs: list[float] = []
        maps: list[float] = []
        recalls_k: list[float] = []
        recalls_r: list[float] = []
        oracles: list[float] = []

        for qid, hits in cand.items():
            qrels = qrels_all.get(qid, {})

            ranked_ids_k = [d for d, _ in hits[:k]]
            ms = _compute_metrics(ranked_ids_k, qrels, k)

            ranked_ids_r = [d for d, _ in hits[:recall_k]]
            recall_r = recall_at_k(ranked_ids_r, qrels, recall_k)

            cand_score_map = {d: s for d, s in hits[:recall_k]}
            oracle = _oracle_ndcg_at_k(cand_score_map, qrels, oracle_k)

            ndcgs.append(ms[f"ndcg@{k}"])
            maps.append(ms[f"map@{k}"])
            recalls_k.append(ms[f"recall@{k}"])
            recalls_r.append(recall_r)
            oracles.append(oracle)

            per_query_rows.append(
                {
                    "method": name,
                    "query_id": qid,
                    f"ndcg@{k}": ms[f"ndcg@{k}"],
                    f"map@{k}": ms[f"map@{k}"],
                    f"recall@{k}": ms[f"recall@{k}"],
                    f"recall@{recall_k}": recall_r,
                    f"oracle_ndcg@{oracle_k}": oracle,
                }
            )

        summary: dict[str, Any] = {
            "method": name,
            f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
            f"map@{k}": float(np.mean(maps)) if maps else 0.0,
            f"recall@{k}": float(np.mean(recalls_k)) if recalls_k else 0.0,
            f"recall@{recall_k}": float(np.mean(recalls_r)) if recalls_r else 0.0,
            f"oracle_ndcg@{oracle_k}": float(np.mean(oracles)) if oracles else 0.0,
            "num_queries": len(queries_map),
            "split": split,
        }
        method_summaries.append(summary)

    metrics = {
        "k": k,
        "split": split,
        "diagnostics": {"recall_k": recall_k, "oracle_k": oracle_k},
        "methods": method_summaries,
    }
    write_json(out_dir / "metrics.json", metrics)

    csv_path = out_dir / "ablations.csv"
    cols = [
        "method",
        f"ndcg@{k}",
        f"map@{k}",
        f"recall@{k}",
        f"recall@{recall_k}",
        f"oracle_ndcg@{oracle_k}",
        "num_queries",
        "split",
    ]
    lines = [",".join(cols)]
    for row in method_summaries:
        lines.append(",".join(str(row.get(c, "")) for c in cols))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for r in per_query_rows:
            f.write(json.dumps(r) + "\n")

    log.info("Wrote metrics to %s/metrics.json", out_dir)
    log.info("Wrote ablations to %s", csv_path)
    log.info("Wrote per-query results to %s", results_path)


if __name__ == "__main__":
    main()
