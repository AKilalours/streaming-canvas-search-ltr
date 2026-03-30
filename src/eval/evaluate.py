# src/eval/evaluate.py
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from eval.metrics import aggregate_methods_list
from ranking.ltr_infer import LTRReranker
from retrieval.hybrid import hybrid_merge
from utils.io import read_json, read_jsonl, write_json
from utils.logging import get_logger

log = get_logger("eval.evaluate")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _infer_embed_model_name_from_emb_dir(emb_dir: Path) -> str:
    """Infer model name from meta.json first, then directory name."""
    import json as _json
    meta = emb_dir / "meta.json"
    if meta.exists():
        try:
            m = _json.loads(meta.read_text())
            if m.get("model_name"):
                return str(m["model_name"])
        except Exception:
            pass
    d = emb_dir.name
    if "sentence-transformers_" in d:
        tail = d.split("sentence-transformers_", 1)[1]
        return "sentence-transformers/" + tail.replace("_", "/")
    if "intfloat_" in d:
        return "intfloat/" + d.split("intfloat_", 1)[1]
    return "sentence-transformers/all-MiniLM-L6-v2"


def _dense_topk(
    model: SentenceTransformer,
    query: str,
    doc_embs: np.ndarray,
    doc_ids: list[str],
    k: int,
) -> list[tuple[str, float]]:
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
    scores = doc_embs @ q
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(doc_ids[int(i)], float(scores[int(i)])) for i in idx]


def _oracle_ndcg_at_k(qrels: dict[str, int], candidates: list[str], k: int) -> float:
    def dcg(vals: list[int]) -> float:
        s = 0.0
        for i, r in enumerate(vals, start=1):
            s += (2**r - 1) / math.log2(i + 1)
        return s

    rels = [(d, int(qrels.get(d, 0))) for d in candidates]
    rels.sort(key=lambda x: x[1], reverse=True)
    oracle_ranked = [d for d, _ in rels[:k]]

    got = [int(qrels.get(d, 0)) for d in oracle_ranked]
    ideal = sorted([int(v) for v in qrels.values()], reverse=True)[:k]
    denom = dcg(ideal)
    return 0.0 if denom == 0 else dcg(got) / denom


def _load_bm25(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _bm25_query(bm25: Any, qtext: str, k: int) -> list[tuple[str, float]]:
    if hasattr(bm25, "query"):
        return bm25.query(qtext, k=k)
    if hasattr(bm25, "search"):
        return bm25.search(qtext, k=k)
    raise RuntimeError("BM25 object has neither query() nor search()")


def _find_method(cfg: dict[str, Any], name: str) -> dict[str, Any] | None:
    for m in (cfg.get("methods") or []):
        if isinstance(m, dict) and m.get("name") == name:
            return m
    return None


def _method_type(m: dict[str, Any]) -> str:
    return str(m.get("type") or m.get("name") or "").strip().lower()


def _best_ltr_path(cfg: dict[str, Any]) -> Path | None:
    """
    Your YAML currently shows:
      ltr_model_path: artifacts/ltr/ltr.pkl   (stale / wrong)
    But your training produced:
      artifacts/ltr/nfcorpus_ltr.pkl
    We fix this by:
      1) try method key 'ltr_model_path'
      2) if missing/nonexistent, infer dataset name from dataset_processed_dir and use artifacts/ltr/{dataset}_ltr.pkl
    """
    m = _find_method(cfg, "hybrid_ltr")
    cand = None
    if m and isinstance(m, dict):
        cand = m.get("ltr_model_path") or m.get("ltr_path") or m.get("model_path")

    if cand:
        p = Path(str(cand))
        if p.exists():
            return p

    # infer dataset name from processed dir, e.g. data/processed/nfcorpus
    eval_cfg = cfg.get("eval", {}) or {}
    processed = Path(str(eval_cfg.get("dataset_processed_dir", "data/processed/nfcorpus")))
    dataset_name = processed.name  # nfcorpus
    fallback = Path(f"artifacts/ltr/{dataset_name}_ltr.pkl")
    if fallback.exists():
        return fallback

    return None


# ---------------------------------------------------------------------
# Load inputs from your current YAML layout
# ---------------------------------------------------------------------
def load_eval_inputs_from_cfg(cfg: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, Any]]:
    eval_cfg = cfg.get("eval", {}) or {}
    processed_dir = Path(str(eval_cfg.get("dataset_processed_dir", "data/processed/nfcorpus")))
    split = str(eval_cfg.get("split", "test"))

    corpus_path = processed_dir / split / "corpus.jsonl"
    queries_path = processed_dir / split / "queries.jsonl"
    qrels_path = processed_dir / split / "qrels.json"

    corpus_rows = read_jsonl(str(corpus_path))
    queries_rows = read_jsonl(str(queries_path))
    qrels_all = read_json(str(qrels_path))

    corpus = {r["doc_id"]: r for r in corpus_rows}
    queries = {r["query_id"]: r["text"] for r in queries_rows}
    return corpus, queries, qrels_all


# ---------------------------------------------------------------------
# Main evaluation runner (supports your methods: bm25, dense, hybrid, hybrid_ltr)
# ---------------------------------------------------------------------
def run_eval(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = cfg or {}
    eval_cfg = cfg.get("eval", {}) or {}

    k = int(eval_cfg.get("k", 10))
    diagnostics_cfg = eval_cfg.get("diagnostics", {}) or {}
    recall_k = int(diagnostics_cfg.get("recall_k", 100))
    oracle_k = int(diagnostics_cfg.get("oracle_k", k))

    # Load data
    corpus, queries, qrels_all = load_eval_inputs_from_cfg(cfg)

    # Locate artifacts from methods blocks
    bm25_m = _find_method(cfg, "bm25")
    dense_m = _find_method(cfg, "dense")
    hy_m = _find_method(cfg, "hybrid")
    hyltr_m = _find_method(cfg, "hybrid_ltr")

    if not bm25_m:
        raise FileNotFoundError("methods: missing bm25 block")
    bm25_artifact = Path(str(bm25_m.get("bm25_artifact")))
    if not bm25_artifact.exists():
        raise FileNotFoundError(f"bm25_artifact not found: {bm25_artifact}")

    # dense emb_dir may be defined in dense/hybrid/hybrid_ltr; prefer dense if present
    emb_dir_str = None
    for mm in (dense_m, hy_m, hyltr_m):
        if mm and isinstance(mm, dict) and mm.get("emb_dir"):
            emb_dir_str = str(mm.get("emb_dir"))
            break
    if not emb_dir_str:
        raise FileNotFoundError("methods: missing emb_dir (dense/hybrid/hybrid_ltr)")

    emb_dir = Path(emb_dir_str)
    if not emb_dir.exists():
        raise FileNotFoundError(f"emb_dir not found: {emb_dir}")

    # Load artifacts
    bm25 = _load_bm25(bm25_artifact)

    doc_embs = np.load(emb_dir / "embeddings.npy").astype(np.float32)
    doc_ids = json.loads((emb_dir / "doc_ids.json").read_text(encoding="utf-8"))

    # CRITICAL: use model inferred from embeddings directory, not YAML's stale model_name
    embed_model_name = _infer_embed_model_name_from_emb_dir(emb_dir)
    embed_model = SentenceTransformer(embed_model_name)

    # Load reranker (if available)
    ltr_path = _best_ltr_path(cfg)
    reranker: LTRReranker | None = None
    if ltr_path is not None and ltr_path.exists():
        reranker = LTRReranker.load(str(ltr_path))
        log.info("Loaded LTR reranker: %s", ltr_path)
    else:
        log.warning("LTR reranker not found (will report has_ltr=false)")

    # Determine max candidate ks so we don’t recompute multiple times
    def _int(m: dict[str, Any] | None, key: str, default: int) -> int:
        if not m:
            return default
        v = m.get(key)
        return int(v) if v is not None else default

    bm25_k_max = max(
        _int(hy_m, "bm25_candidate_k", 200),
        _int(hyltr_m, "bm25_candidate_k", 200),
        _int(bm25_m, "k", 200),
        200,
    )
    dense_k_max = max(
        _int(hy_m, "dense_candidate_k", 200),
        _int(hyltr_m, "dense_candidate_k", 200),
        _int(dense_m, "candidate_k", 200),
        200,
    )

    # Per-method knobs
    alpha_hybrid = float((hy_m or {}).get("alpha", 0.5)) if hy_m else 0.5
    alpha_hyltr = float((hyltr_m or {}).get("alpha", 0.5)) if hyltr_m else 0.5
    rerank_k = int((hyltr_m or {}).get("rerank_k", 50)) if hyltr_m else 50

    bm25_k_hy = _int(hy_m, "bm25_candidate_k", bm25_k_max)
    dense_k_hy = _int(hy_m, "dense_candidate_k", dense_k_max)
    bm25_k_hyltr = _int(hyltr_m, "bm25_candidate_k", bm25_k_max)
    dense_k_hyltr = _int(hyltr_m, "dense_candidate_k", dense_k_max)

    # Ranked lists output
    ranked: dict[str, dict[str, list[str]]] = {"bm25": {}, "dense": {}, "hybrid": {}, "hybrid_ltr": {}}

    # Oracle trackers
    oracle: dict[str, list[float]] = {m: [] for m in ranked.keys()}

    for qid, qtext in queries.items():
        qr = qrels_all.get(qid, {})

        bm25_hits_full = _bm25_query(bm25, qtext, k=bm25_k_max)
        dense_hits_full = _dense_topk(embed_model, qtext, doc_embs, doc_ids, dense_k_max)

        # For metrics/oracle
        bm25_ids_full = [d for d, _ in bm25_hits_full]
        dense_ids_full = [d for d, _ in dense_hits_full]

        # bm25
        ranked["bm25"][qid] = bm25_ids_full[:k]
        oracle["bm25"].append(_oracle_ndcg_at_k(qr, bm25_ids_full[:bm25_k_max], oracle_k))

        # dense
        ranked["dense"][qid] = dense_ids_full[:k]
        oracle["dense"].append(_oracle_ndcg_at_k(qr, dense_ids_full[:dense_k_max], oracle_k))

        # hybrid (use method-specific candidate sizes)
        hy_hits = hybrid_merge(bm25_hits_full[:bm25_k_hy], dense_hits_full[:dense_k_hy], alpha=alpha_hybrid)
        hy_ids = [d for d, _ in hy_hits]
        ranked["hybrid"][qid] = hy_ids[:k]
        oracle["hybrid"].append(_oracle_ndcg_at_k(qr, hy_ids[: len(hy_ids)], oracle_k))

        # hybrid_ltr
        hy2_hits = hybrid_merge(bm25_hits_full[:bm25_k_hyltr], dense_hits_full[:dense_k_hyltr], alpha=alpha_hyltr)
        hy2_ids = [d for d, _ in hy2_hits]

        if reranker is None:
            ranked["hybrid_ltr"][qid] = hy2_ids[:k]
            oracle["hybrid_ltr"].append(_oracle_ndcg_at_k(qr, hy2_ids, oracle_k))
        else:
            to_rerank = hy2_hits[: min(rerank_k, len(hy2_hits))]
            bm25_map = {d: float(s) for d, s in bm25_hits_full}
            dense_map = {d: float(s) for d, s in dense_hits_full}

            reranked = reranker.rerank(
                query=qtext,
                corpus=corpus,
                candidates=to_rerank,
                bm25_scores=bm25_map,
                dense_scores=dense_map,
            )
            reranked_ids = [d for d, _ in reranked]
            reranked_set = set(reranked_ids)

            tail = [d for d in hy2_ids if d not in reranked_set]
            final_ids = (reranked_ids + tail)[:k]
            ranked["hybrid_ltr"][qid] = final_ids

            # oracle over reranked candidate set
            oracle["hybrid_ltr"].append(_oracle_ndcg_at_k(qr, (reranked_ids + tail), oracle_k))

    # Aggregate metrics
    methods_out: list[dict[str, Any]] = []
    for method_name, res in ranked.items():
        agg = aggregate_methods_list(res, qrels_all, k=k, min_rel=1, recall_k=recall_k)
        methods_out.append(
            {
                "method": method_name,
                **agg,
                "oracle_ndcg@10": float(sum(oracle[method_name]) / max(1, len(oracle[method_name]))),
            }
        )

    out = {
        "k": k,
        "split": str(eval_cfg.get("split", "test")),
        "diagnostics": {
            "num_docs": len(corpus),
            "num_queries": len(queries),
            "has_ltr": bool(reranker is not None),
            "ltr_path": str(ltr_path) if ltr_path else None,
            "embed_model": embed_model_name,
            "bm25_candidate_k_max": bm25_k_max,
            "dense_candidate_k_max": dense_k_max,
            "hybrid_alpha": alpha_hybrid,
            "hybrid_ltr_alpha": alpha_hyltr,
            "rerank_k": rerank_k,
        },
        "methods": methods_out,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    out = run_eval(cfg)

    out_dir = Path("reports/latest")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", out)
    log.info("Wrote reports/latest/metrics.json (methods=%s)", [m["method"] for m in out["methods"]])


if __name__ == "__main__":
    main()
