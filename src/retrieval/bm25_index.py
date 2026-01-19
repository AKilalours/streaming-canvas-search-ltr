import argparse
import pickle
from pathlib import Path

import yaml
from rank_bm25 import BM25Okapi

from retrieval.bm25_artifact import BM25Artifact, tokenize
from utils.io import read_jsonl
from utils.logging import get_logger

log = get_logger("retrieval.bm25_index")


def build_bm25(processed_split_dir: Path) -> BM25Artifact:
    corpus = read_jsonl(processed_split_dir / "corpus.jsonl")

    doc_ids: list[str] = []
    tokenized_docs: list[list[str]] = []

    for row in corpus:
        doc_id = row["doc_id"]
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        doc_ids.append(doc_id)
        tokenized_docs.append(tokenize(f"{title} {text}"))

    bm25 = BM25Okapi(tokenized_docs)
    return BM25Artifact(doc_ids=doc_ids, bm25=bm25)


def pick_split_dir(processed_dir: Path, preferred: str | None = None) -> Path:
    # Allow override but also work robustly if a split doesn't exist.
    if preferred:
        p = processed_dir / preferred
        if not p.exists():
            raise FileNotFoundError(
                f"Requested split '{preferred}' not found under {processed_dir}"
            )
        return p

    for split in ["train", "test", "dev"]:
        p = processed_dir / split
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No processed splits found under {processed_dir} (expected train/test/dev)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/dataset.yaml")
    ap.add_argument(
        "--split",
        default=None,
        help="Optional split to index from (train/test/dev). If omitted, picks first existing.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    processed_dir = Path(cfg["dataset"]["processed_dir"])

    split_dir = pick_split_dir(processed_dir, preferred=args.split)
    log.info("Indexing split: %s", split_dir)

    artifact = build_bm25(split_dir)

    out_path = Path("artifacts/bm25") / f"{cfg['dataset']['name']}_bm25.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        pickle.dump(artifact, f)

    log.info("Saved BM25 artifact: %s", out_path)


if __name__ == "__main__":
    main()
