# Commit 4: implement FAISS semantic index build + query
# - encode corpus with sentence-transformers
# - build faiss index
# - save artifacts/faiss/{index.faiss, doc_ids.json, embeddings.npy}
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from utils.io import ensure_dir, read_jsonl, write_json
from utils.logging import get_logger

log = get_logger("retrieval.embed_index")


def _slug(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/dataset.yaml")
    ap.add_argument(
        "--split",
        default=None,
        help="Split corpus to index from (train/test). If omitted, uses train if present else test.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = cfg["dataset"]["name"]
    processed_dir = Path(cfg["dataset"]["processed_dir"])

    emb_cfg = cfg.get("embeddings", {})
    model_name = emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = int(emb_cfg.get("batch_size", 64))
    normalize = bool(emb_cfg.get("normalize", True))

    # Pick corpus split
    if args.split:
        split_dir = processed_dir / args.split
    else:
        split_dir = processed_dir / "train"
        if not split_dir.exists():
            split_dir = processed_dir / "test"
    if not split_dir.exists():
        raise FileNotFoundError(f"Processed split not found: {split_dir}. Run `make data` first.")

    corpus_path = split_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus.jsonl at: {corpus_path}")

    corpus = read_jsonl(corpus_path)
    doc_ids = [r["doc_id"] for r in corpus]
    texts = [(r.get("title", "") or "") + " " + (r.get("text", "") or "") for r in corpus]

    log.info("Loading embed model: %s", model_name)
    model = SentenceTransformer(model_name)

    # Encode
    log.info("Encoding %d docs (batch_size=%d, normalize=%s)", len(texts), batch_size, normalize)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    ).astype(np.float32)

    dim = embs.shape[1]

    # Build FAISS index
    import faiss  # local import

    log.info("Building FAISS IndexFlatIP (dim=%d)", dim)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    model_slug = _slug(model_name)
    out_dir = ensure_dir(Path("artifacts/faiss") / f"{dataset}_{model_slug}")
    index_path = out_dir / "index.faiss"
    doc_ids_path = out_dir / "doc_ids.json"
    meta_path = out_dir / "meta.json"

    faiss.write_index(index, str(index_path))
    doc_ids_path.write_text(json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8")
    write_json(
        meta_path,
        {
            "dataset": dataset,
            "model_name": model_name,
            "model_slug": model_slug,
            "normalize_embeddings": normalize,
            "num_docs": len(doc_ids),
            "dim": dim,
            "index_type": "IndexFlatIP",
        },
    )

    log.info("Saved FAISS index: %s", index_path)
    log.info("Saved doc_ids: %s", doc_ids_path)
    log.info("Saved meta: %s", meta_path)


if __name__ == "__main__":
    main()

