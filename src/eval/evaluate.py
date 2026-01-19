import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from eval.metrics import average_precision_at_k, ndcg_at_k, recall_at_k
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


def _compute_metrics(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> dict[str, float]:
    return {
        f"ndcg@{k}": ndcg_at_k(ranked_doc_ids, qrels, k),
        f"map@{k}": average_precision_at_k(ranked_doc_ids, qrels, k),
        f"recall@{k}": recall_at_k(ranked_doc_ids, qrels, k),
    }


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
    # q_emb: [Q, D], doc_embs: [N, D]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/eval.yaml")
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Optional output directory override (e.g., reports/<run_id>/). If omitted, writes to reports/latest_eval/.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    processed_dir = Path(cfg["eval"]["dataset_processed_dir"])
    split = cfg["eval"]["split"]
    k = int(cfg["eval"]["k"])

    queries_map, qrels_all = _load_split(processed_dir, split)
    log.info("Loaded %d queries for split=%s", len(queries_map), split)

    # Output directory:
    # - if --out_dir provided: use it
    # - else: default to reports/latest_eval
    if args.out_dir:
        out_dir = ensure_dir(Path(args.out_dir))
    else:
        out_dir = ensure_dir(Path(cfg["reporting"]["out_dir"]) / "latest_eval")

    bm25_obj = None
    bm25_methods = [m for m in cfg["methods"] if m["type"] in ("bm25", "hybrid")]
    if bm25_methods:
        bm25_path = Path(bm25_methods[0]["bm25_artifact"])
        with bm25_path.open("rb") as f:
            bm25_obj = pickle.load(f)

    dense_cache: dict[str, dict[str, list[tuple[str, float]]]] = {}

    method_summaries: list[dict] = []
    per_query_rows: list[dict] = []

    for method in cfg["methods"]:
        name = method["name"]
        mtype = method["type"]

        if mtype == "bm25":
            cand = _bm25_retrieve(bm25_obj, queries_map, k=k)

        elif mtype == "dense":
            model_name = method["model_name"]
            emb_dir = Path(method["emb_dir"])
            cand_k = int(method.get("candidate_k", k))
            cache_key = f"{model_name}::{emb_dir}::{cand_k}"
            if cache_key not in dense_cache:
                dense_cache[cache_key] = _dense_retrieve(queries_map, model_name, emb_dir, cand_k)
            cand = dense_cache[cache_key]

        elif mtype == "hybrid":
            alpha = float(method.get("alpha", 0.5))
            bm25_k = int(method.get("bm25_candidate_k", 50))
            dense_k = int(method.get("dense_candidate_k", 50))
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
                cand[qid] = merged[:k]

        else:
            raise ValueError(f"Unknown method type: {mtype}")

        ndcgs: list[float] = []
        maps: list[float] = []
        recalls: list[float] = []

        for qid, hits in cand.items():
            qrels = qrels_all.get(qid, {})
            ranked_ids = [d for d, _ in hits[:k]]
            ms = _compute_metrics(ranked_ids, qrels, k)

            ndcgs.append(ms[f"ndcg@{k}"])
            maps.append(ms[f"map@{k}"])
            recalls.append(ms[f"recall@{k}"])

            per_query_rows.append(
                {
                    "method": name,
                    "query_id": qid,
                    "ndcg": ms[f"ndcg@{k}"],
                    "map": ms[f"map@{k}"],
                    "recall": ms[f"recall@{k}"],
                }
            )

        summary = {
            "method": name,
            f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
            f"map@{k}": float(np.mean(maps)) if maps else 0.0,
            f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
            "num_queries": len(queries_map),
            "split": split,
        }
        method_summaries.append(summary)

    metrics = {"k": k, "split": split, "methods": method_summaries}
    write_json(out_dir / "metrics.json", metrics)

    csv_path = out_dir / "ablations.csv"
    cols = ["method", f"ndcg@{k}", f"map@{k}", f"recall@{k}", "num_queries", "split"]
    lines = [",".join(cols)]
    for row in method_summaries:
        lines.append(",".join(str(row[c]) for c in cols))
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

