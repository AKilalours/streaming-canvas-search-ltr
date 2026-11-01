from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from retrieval.bm25 import build_bm25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    bm25 = build_bm25(corpus)
    with out.open("wb") as f:
        pickle.dump(bm25, f)

    print(f"[OK] wrote BM25 -> {out} (docs={len(bm25.doc_ids)})")


if __name__ == "__main__":
    main()
