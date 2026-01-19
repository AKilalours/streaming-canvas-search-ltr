import argparse
import pickle
from pathlib import Path

import yaml

from eval.metrics import average_precision_at_k, ndcg_at_k, recall_at_k
from utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from utils.logging import get_logger

log = get_logger("eval.evaluate")


def _load_split(processed_dir: Path, split: str):
    split_dir = processed_dir / split
    corpus = read_jsonl(split_dir / "corpus.jsonl")
    queries = read_jsonl(split_dir / "queries.jsonl")
    qrels = read_json(split_dir / "qrels.json")
    corpus_map = {r["doc_id"]: r for r in corpus}
    queries_map = {r["query_id"]: r["text"] for r in queries}
    return corpus_map, queries_map, qrels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/eval.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    processed_dir = Path(cfg["eval"]["dataset_processed_dir"])
    split = cfg["eval"]["split"]
    k = int(cfg["eval"]["k"])

    retriever_type = cfg["retriever"]["type"]
    if retriever_type != "bm25":
        raise ValueError("Only bm25 is wired in MVP. Add faiss/hybrid later.")

    bm25_path = Path(cfg["retriever"]["bm25_artifact"])
    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 artifact not found: {bm25_path}. Run `make index-bm25`.")

    with bm25_path.open("rb") as f:
        bm25 = pickle.load(f)

    corpus_map, queries_map, qrels_all = _load_split(processed_dir, split)

    rows = []
    ndcgs = []
    maps = []
    recalls = []

    for qid, qtext in queries_map.items():
        qrels = qrels_all.get(qid, {})
        ranked = [doc_id for doc_id, _s in bm25.query(qtext, k=k)]
        nd = ndcg_at_k(ranked, qrels, k)
        apk = average_precision_at_k(ranked, qrels, k)
        rk = recall_at_k(ranked, qrels, k)

        ndcgs.append(nd)
        maps.append(apk)
        recalls.append(rk)

        rows.append(
            {
                "query_id": qid,
                "query": qtext,
                "ndcg@k": nd,
                "map@k": apk,
                "recall@k": rk,
                "top_doc_ids": ranked,
            }
        )

    metrics = {
        f"ndcg@{k}": float(sum(ndcgs) / max(1, len(ndcgs))),
        f"map@{k}": float(sum(maps) / max(1, len(maps))),
        f"recall@{k}": float(sum(recalls) / max(1, len(recalls))),
        "num_queries": len(rows),
        "split": split,
        "retriever": retriever_type,
    }

    out_dir = ensure_dir(Path(cfg["reporting"]["out_dir"]) / "latest_eval")
    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "results.jsonl", rows)

    log.info("Wrote metrics to %s/metrics.json", out_dir)
    log.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
