import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from utils.io import ensure_dir, write_json
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

    # Load corpus
    corpus = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))

    doc_ids = [r["doc_id"] for r in corpus]
    texts = [(r.get("title", "") or "") + " " + (r.get("text", "") or "") for r in corpus]

    log.info("Loading embed model: %s", model_name)
    model = SentenceTransformer(model_name)

    log.info("Encoding %d docs (batch_size=%d, normalize=%s)", len(texts), batch_size, normalize)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    ).astype(np.float32)

    dim = int(embs.shape[1])

    model_slug = _slug(model_name)
    out_dir = ensure_dir(Path("artifacts/faiss") / f"{dataset}_{model_slug}")

    # Save dense artifacts for stable eval
    np.save(out_dir / "embeddings.npy", embs)
    (out_dir / "doc_ids.json").write_text(json.dumps(doc_ids, ensure_ascii=False), encoding="utf-8")
    write_json(
        out_dir / "meta.json",
        {
            "dataset": dataset,
            "model_name": model_name,
            "model_slug": model_slug,
            "normalize_embeddings": normalize,
            "num_docs": len(doc_ids),
            "dim": dim,
            "retrieval": "dense_dot",
        },
    )
    log.info("Saved dense embeddings: %s", out_dir / "embeddings.npy")

    # Keep FAISS build too (optional) — but eval will not depend on it
    try:
        import faiss

        log.info("Building FAISS IndexFlatIP (dim=%d)", dim)
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        faiss.write_index(index, str(out_dir / "index.faiss"))
        log.info("Saved FAISS index: %s", out_dir / "index.faiss")
    except Exception as e:
        log.info("FAISS build skipped (non-fatal): %s", e)


if __name__ == "__main__":
    main()
