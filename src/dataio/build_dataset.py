import argparse
from pathlib import Path

import yaml

from dataio.beir_download import download_beir_dataset, load_beir_split
from utils.io import ensure_dir, write_json, write_jsonl
from utils.logging import get_logger

log = get_logger("dataio.build_dataset")


def _export_split(processed_dir: Path, split: str, corpus, queries, qrels) -> None:
    # corpus: dict[doc_id] -> {"title":..., "text":...}
    # queries: dict[qid] -> str
    # qrels: dict[qid] -> dict[doc_id] -> int relevance

    split_dir = ensure_dir(processed_dir / split)

    corpus_rows: list[dict] = []
    for doc_id, d in corpus.items():
        corpus_rows.append(
            {
                "doc_id": doc_id,
                "title": d.get("title", "") or "",
                "text": d.get("text", "") or "",
            }
        )

    query_rows: list[dict] = []
    for qid, text in queries.items():
        query_rows.append({"query_id": qid, "text": text})

    write_jsonl(split_dir / "corpus.jsonl", corpus_rows)
    write_jsonl(split_dir / "queries.jsonl", query_rows)
    write_json(split_dir / "qrels.json", qrels)

    log.info("Exported %s split to %s", split, split_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/dataset.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    dataset = cfg["dataset"]["name"]
    raw_dir = cfg["dataset"]["raw_dir"]
    processed_dir = Path(cfg["dataset"]["processed_dir"])
    export_splits = cfg.get("build", {}).get("export_splits", ["train", "test"])

    dataset_dir = download_beir_dataset(dataset=dataset, raw_root=raw_dir)

    ensure_dir(processed_dir)

    for split in export_splits:
        try:
            corpus, queries, qrels = load_beir_split(dataset_dir, split=split)
        except ValueError as e:
            # Some BEIR datasets (including SciFact) may not have all splits (e.g., dev)
            log.info("Skipping split=%s (not present): %s", split, e)
            continue

        _export_split(processed_dir, split, corpus, queries, qrels)

    log.info("Done. Processed dataset at: %s", processed_dir)


if __name__ == "__main__":
    main()
