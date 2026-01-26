# src/app/deps.py
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.logging import get_logger

log = get_logger("app.deps")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _first_existing(patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(Path(".").glob(pat))
        if hits:
            return hits[0]
    return None


def _best_corpus_path() -> Path | None:
    root = Path("data/processed")
    if not root.exists():
        return None
    hits = sorted(root.glob("**/corpus.jsonl"))
    if not hits:
        return None
    # choose the most recently modified (usually what you last built)
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


@dataclass
class DenseIndex:
    model_name: str
    doc_embs: np.ndarray
    doc_ids: list[str]
    model: SentenceTransformer

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = (
            self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            .astype(np.float32)[0]
        )
        scores = self.doc_embs @ q
        k = min(k, scores.shape[0])
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in idx]


class AppState:
    def __init__(self) -> None:
        self.ready = False
        self.corpus: dict[str, dict[str, Any]] = {}
        self.bm25_obj: Any | None = None
        self.dense: DenseIndex | None = None
        self.ltr_path: Path | None = None

    def bm25_query(self, query: str, k: int) -> list[tuple[str, float]]:
        if self.bm25_obj is None:
            return []
        if hasattr(self.bm25_obj, "query"):
            return list(self.bm25_obj.query(query, k=k))
        if hasattr(self.bm25_obj, "search"):
            return list(self.bm25_obj.search(query, top_k=k))  # type: ignore[attr-defined]
        raise RuntimeError("BM25 object does not support .query() or .search().")


def load_state() -> AppState:
    st = AppState()

    # ---- corpus (best-effort) ----
    corpus_path = _best_corpus_path()
    if corpus_path and corpus_path.exists():
        corpus_rows = _load_jsonl(corpus_path)
        st.corpus = {r["doc_id"]: r for r in corpus_rows}
        log.info("Loaded corpus: %s (docs=%d)", corpus_path, len(st.corpus))
    else:
        log.error("No corpus.jsonl found under data/processed/**. Serving will be degraded.")

    # ---- BM25 ----
    bm25_path = _first_existing(["artifacts/bm25/*.pkl"])
    if bm25_path is None or not bm25_path.exists():
        raise FileNotFoundError("Could not find BM25 .pkl under artifacts/bm25/.")
    with bm25_path.open("rb") as f:
        st.bm25_obj = pickle.load(f)
    log.info("Loaded BM25 artifact: %s", bm25_path)

    # ---- Dense (optional) ----
    emb_path = _first_existing(["artifacts/faiss/**/embeddings.npy"])
    ids_path = _first_existing(["artifacts/faiss/**/doc_ids.json"])
    if emb_path and ids_path and emb_path.exists() and ids_path.exists():
        doc_embs = np.load(emb_path).astype(np.float32)
        doc_ids = list(_load_json(ids_path))

        # normalize defensively
        norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        doc_embs = doc_embs / norms

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        st.dense = DenseIndex(model_name=model_name, doc_embs=doc_embs, doc_ids=doc_ids, model=model)
        log.info("Loaded dense artifacts: %s + %s", emb_path, ids_path)
    else:
        log.info("Dense artifacts not found; dense/hybrid methods will be unavailable.")

    # ---- LTR ----
    ltr_path = _first_existing(["artifacts/ltr/ltr.pkl", "artifacts/ltr/*.pkl"])
    if ltr_path and ltr_path.exists():
        st.ltr_path = ltr_path
        log.info("Found LTR artifact: %s", ltr_path)
    else:
        log.info("LTR artifact not found; hybrid_ltr will behave like hybrid.")

    st.ready = True
    return st
 
