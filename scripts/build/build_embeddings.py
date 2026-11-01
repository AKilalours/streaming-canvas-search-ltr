from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    ap.add_argument("--out_dir", required=True, help="Output dir (embeddings.npy + doc_ids.json)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(corpus_path)
    doc_ids = [str(r["doc_id"]) for r in rows]
    texts = []
    for r in rows:
        title = str(r.get("title") or "")
        text = str(r.get("text") or "")
        texts.append(f"{title}. {text}".strip())

    model = SentenceTransformer(args.model)
    embs = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(out_dir / "embeddings.npy", embs)
    (out_dir / "doc_ids.json").write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps({"model": args.model, "num_docs": len(doc_ids), "dim": int(embs.shape[1])}, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] wrote embeddings -> {out_dir} (docs={len(doc_ids)} dim={embs.shape[1]})")


if __name__ == "__main__":
    main()
