import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

from eval.metrics import average_precision_at_k, ndcg_at_k, recall_at_k
from retrieval.hybrid import hybrid_merge
from utils.io import ensure_dir, read_json, read_jsonl, write_json
from utils.logging import get_logger

log = get_logger("eval.evaluate")


def _load_split(processed_dir: Path, split: str):
    split_dir = processed_dir / split
    corpus = read_jsonl(split_dir / "corpus.jsonl")
    queries = read_jsonl(split_dir / "queries.jsonl")
    qrels = read_json(split_dir / "qrels.json")
    queries_map = [(r["query_id"], r["text"]) for r in queries]
    return corpus, queries, qrels, dict(queries)


def _bm25_retrieve(bm25_obj, queries_map: dict[str, str], k: int) -> dict[str, list[tuple[str, float]]]:
    out = {}
    for qid, qtext in queries_map.items():
        out[qid] = bm25_obj.query(qtext, k=k)
    return out


def _faiss_load(faiss_dir: Path):
    import faiss  # local import

    index_path = faiss_dir / "index.faiss"
    doc_ids_path = faiss_dir / "doc_ids.json"
    meta_path = faiss_dir / "meta.json"

    if not index_path.exists() or not doc_ids_path.exists():
        raise FileNotFoundError(f"Missing FAISS artifacts in {faiss_dir}. Run `make index-faiss`.")

    index = faiss.read_index(str(index_path))
    doc_ids = json.loads(doc_ids_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return index, doc_ids, meta


def _faiss_retrieve(
    queries_map: dict[str, str],
    model_name: str,
    faiss_dir: Path,
    k: int,
) -> dict[str, list[tuple[str, float]]]:
    from sentence_transformers import SentenceTransformer

    index, doc_ids, meta = _faiss_load(faiss_dir)

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

    scores, idxs = index.search(q_emb, k)
    out = {}
    for i, qid in enumerate(qids):
        hits = []
        for j in range(k):
            di = int(idxs[i, j])
            if di < 0:
                continue
            hits.append((doc_ids[di], float(scores[i, j])))
        out[qid] = hits
    return out


def _compute_metrics(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> dict[str, float]:
    return {
        f"ndcg@{k}": ndcg_at_k(ranked_doc_ids, qrels, k),
        f"map@{k}": average_precision_at_k(ranked_doc_ids, qrels, k),
        f"recall@{k}": recall_at_k(ranked_doc_ids, qrels, k),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/eval.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    processed_dir = Path(cfg["eval"]["dataset_processed_dir"])
    split = cfg["eval"]["split"]
    k = int(cfg["eval"]["k"])

    _corpus, _queries_rows, qrels_all, queries_map = _load_split(processed_dir, split)

    out_dir = ensure_dir(Path(cfg["reporting"]["out_dir"]) / "latest_eval")

    method_summaries = []
    per_query_rows = []

    # Pre-load BM25 if any method needs it
    bm25_obj = None
    if any(m["type"] in ("bm25", "hybrid") for m in cfg["methods"]):
        bm25_path = Path([m for m in cfg["methods"] if "bm25_artifact" in m][0]["bm25_artifact"])
        with bm25_path.open("rb") as f:
            bm25_obj = pickle.load(f)

    # Precompute FAISS retrieval if used (shared model/index per config entry)
    faiss_cache: dict[str, dict[str, list[tuple[str, float]]]] = {}

    for method in cfg["methods"]:
        name = method["name"]
        mtype = method["type"]

        if mtype == "bm25":
            cand = _bm25_retrieve(bm25_obj, queries_map, k=k)

        elif mtype == "faiss":
            model_name = method["model_name"]
            faiss_dir = Path(method["faiss_dir"])
            cand_k = int(method.get("candidate_k", k))
            cache_key = f"{model_name}::{faiss_dir}::{cand_k}"
            if cache_key not in faiss_cache:
                faiss_cache[cache_key] = _faiss_retrieve(
                    queries_map=queries_map,
                    model_name=model_name,
                    faiss_dir=faiss_dir,
                    k=cand_k,
                )
            cand = faiss_cache[cache_key]

        elif mtype == "hybrid":
            alpha = float(method.get("alpha", 0.5))
            bm25_k = int(method.get("bm25_candidate_k", 50))
            faiss_k = int(method.get("faiss_candidate_k", 50))
            model_name = method["model_name"]
            faiss_dir = Path(method["faiss_dir"])

            bm25_c = _bm25_retrieve(bm25_obj, queries_map, k=bm25_k)

            cache_key = f"{model_name}::{faiss_dir}::{faiss_k}"
            if cache_key not in faiss_cache:
                faiss_cache[cache_key] = _faiss_retrieve(
                    queries_map=queries_map,
                    model_name=model_name,
                    faiss_dir=faiss_dir,
                    k=faiss_k,
                )
            faiss_c = faiss_cache[cache_key]

            cand = {}
            for qid in queries_map:
                merged = hybrid_merge(bm25_c[qid], faiss_c[qid], alpha=alpha)
                cand[qid] = merged[:k]

        else:
            raise ValueError(f"Unknown method type: {mtype}")

        # Aggregate metrics
        ndcgs, maps, recalls = [], [], []

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

    # Write outputs
    metrics = {"k": k, "split": split, "methods": method_summaries}
    write_json(out_dir / "metrics.json", metrics)

    # Write ablations.csv
    csv_path = out_dir / "ablations.csv"
    cols = ["method", f"ndcg@{k}", f"map@{k}", f"recall@{k}", "num_queries", "split"]
    lines = [",".join(cols)]
    for row in method_summaries:
        lines.append(",".join(str(row[c]) for c in cols))
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    # Per-query results (jsonl)
    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for r in per_query_rows:
            f.write(json.dumps(r) + "\n")

    log.info("Wrote metrics to %s/metrics.json", out_dir)
    log.info("Wrote ablations to %s", csv_path)
    log.info("Wrote per-query results to %s", results_path)


if __name__ == "__main__":
    main()
