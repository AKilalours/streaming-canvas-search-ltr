from __future__ import annotations

import json
import os
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


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to load config. Install it in the api image/env: pip install pyyaml"
        ) from e
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _first_existing(patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = sorted(Path(".").glob(pat))
        if hits:
            return hits[0]
    return None


def _require(path: Path | None, msg: str) -> Path:
    if path is None or not path.exists():
        raise FileNotFoundError(msg)
    return path


def _select_config_path() -> Path:
    # Allows easy switching:
    #   APP_CONFIG=config/app.nfcorpus.yaml docker compose up ...
    p = Path(os.environ.get("APP_CONFIG", "config/app.yaml"))
    if p.exists():
        return p
    p2 = Path("configs/app.yaml")
    if p2.exists():
        return p2
    raise FileNotFoundError("No config found. Expected config/app.yaml (or set APP_CONFIG=...).")


def _dataset_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    ds = cfg.get("dataset") or {}
    return ds if isinstance(ds, dict) else {}


def _resolve_corpus_path(ds: dict[str, Any]) -> Path:
    p = (ds.get("paths") or {}).get("corpus")
    if p:
        return Path(p)
    dsid = str(ds.get("id") or "nfcorpus")
    split = str(ds.get("split") or "test")
    return Path(f"data/processed/{dsid}/{split}/corpus.jsonl")


def _resolve_bm25_path(ds: dict[str, Any]) -> Path:
    a = ds.get("artifacts") or {}

    # explicit path wins
    p = a.get("bm25")
    if p:
        pp = Path(p)
        return _require(pp, f"BM25 not found at configured path: {pp}")

    dsid = str(ds.get("id") or "nfcorpus")
    cand = _first_existing([f"artifacts/bm25/{dsid}*.pkl"])
    return _require(cand, f"Could not find BM25 artifact for dataset='{dsid}' under artifacts/bm25/.")


def _resolve_dense_dir(ds: dict[str, Any]) -> Path | None:
    a = ds.get("artifacts") or {}

    # explicit directory wins
    dense_dir = a.get("dense_dir")
    if dense_dir:
        d = Path(dense_dir)
        return d if d.exists() and d.is_dir() else None

    # optional glob
    dense_dir_glob = a.get("dense_dir_glob")
    if dense_dir_glob:
        d = _first_existing([str(dense_dir_glob)])
        if d and d.exists() and d.is_dir():
            return d

    # fallback: dataset-specific directory that contains embeddings.npy + doc_ids.json
    dsid = str(ds.get("id") or "nfcorpus")
    for emb in Path(".").glob(f"artifacts/faiss/**/{dsid}*/embeddings.npy"):
        if emb.is_file() and (emb.parent / "doc_ids.json").exists():
            return emb.parent

    # final fallback: any directory that looks valid
    for emb in Path(".").glob("artifacts/faiss/**/embeddings.npy"):
        if emb.is_file() and (emb.parent / "doc_ids.json").exists():
            return emb.parent

    return None


def _resolve_ltr_path(ds: dict[str, Any]) -> Path | None:
    a = ds.get("artifacts") or {}

    # CRITICAL: if config explicitly sets ltr (even null), respect it.
    if "ltr" in a:
        p = a.get("ltr")
        if not p:
            return None
        pp = Path(p)
        return pp if pp.exists() else None

    dsid = str(ds.get("id") or "nfcorpus")
    return _first_existing([f"artifacts/ltr/{dsid}*.pkl", "artifacts/ltr/ltr.pkl", "artifacts/ltr/*.pkl"])


@dataclass
class DenseIndex:
    model_name: str
    doc_embs: np.ndarray
    doc_ids: list[str]
    model: SentenceTransformer

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(
            np.float32
        )[0]
        scores = self.doc_embs @ q
        k = min(k, scores.shape[0])
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.doc_ids[int(i)], float(scores[int(i)])) for i in idx]


class AppState:
    def __init__(self) -> None:
        self.ready = False
        self.dataset_id: str = "unknown"
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

    cfg_path = _select_config_path()
    cfg = _load_yaml(cfg_path)
    ds = _dataset_cfg(cfg)

    st.dataset_id = str(ds.get("id") or "nfcorpus")
    log.info("Loading AppState from config=%s dataset=%s", cfg_path, st.dataset_id)

    # ---- corpus (explicit) ----
    corpus_path = _resolve_corpus_path(ds)
    _require(corpus_path, f"Corpus not found: {corpus_path} (dataset={st.dataset_id})")
    corpus_rows = _load_jsonl(corpus_path)

    if corpus_rows and "doc_id" not in corpus_rows[0]:
        raise KeyError(f"Corpus rows must contain 'doc_id'. First row keys: {list(corpus_rows[0].keys())}")

    st.corpus = {r["doc_id"]: r for r in corpus_rows}
    log.info("Loaded corpus: %s (docs=%d)", corpus_path, len(st.corpus))

    # ---- BM25 ----
    bm25_path = _resolve_bm25_path(ds)
    with bm25_path.open("rb") as f:
        st.bm25_obj = pickle.load(f)
    log.info("Loaded BM25 artifact: %s", bm25_path)

    # ---- Dense (optional) ----
    dense_dir = _resolve_dense_dir(ds)
    if dense_dir:
        emb_path = dense_dir / "embeddings.npy"
        ids_path = dense_dir / "doc_ids.json"
        meta_path = dense_dir / "meta.json"

        doc_embs = np.load(emb_path).astype(np.float32)
        doc_ids = list(_load_json(ids_path))

        if doc_embs.shape[0] != len(doc_ids):
            raise RuntimeError(
                f"Dense artifact mismatch: embeddings rows={doc_embs.shape[0]} != doc_ids={len(doc_ids)} in {dense_dir}"
            )

        # normalize defensively
        norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        doc_embs = doc_embs / norms

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        if meta_path.exists():
            meta = _load_json(meta_path)
            model_name = str(meta.get("model_name") or meta.get("encoder") or model_name)

        model = SentenceTransformer(model_name)
        st.dense = DenseIndex(model_name=model_name, doc_embs=doc_embs, doc_ids=doc_ids, model=model)
        log.info("Loaded dense artifacts: %s (model=%s)", dense_dir, model_name)
    else:
        log.info("Dense artifacts not found for dataset=%s; dense/hybrid may be degraded.", st.dataset_id)

    # ---- LTR (optional) ----
    ltr_path = _resolve_ltr_path(ds)
    if ltr_path and ltr_path.exists():
        st.ltr_path = ltr_path
        log.info("Found LTR artifact: %s", ltr_path)
    else:
        log.info("LTR artifact not found for dataset=%s; hybrid_ltr behaves like hybrid.", st.dataset_id)

    st.ready = True
    return st
