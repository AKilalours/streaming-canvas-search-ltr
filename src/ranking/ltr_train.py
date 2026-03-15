# src/ranking/ltr_train.py
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import yaml
from lightgbm import LGBMRanker
from sentence_transformers import SentenceTransformer

from ranking.features import FEATURE_NAMES, build_features
from retrieval.hybrid import hybrid_merge
from utils.io import ensure_dir, read_json, read_jsonl, write_json
from utils.logging import get_logger

log = get_logger("ranking.ltr_train")


def _load_corpus(processed_dir: Path, split: str) -> dict[str, dict]:
    rows = read_jsonl(processed_dir / split / "corpus.jsonl")
    return {r["doc_id"]: r for r in rows}


def _load_queries(processed_dir: Path, split: str) -> dict[str, str]:
    rows = read_jsonl(processed_dir / split / "queries.jsonl")
    return {r["query_id"]: r["text"] for r in rows}


def _load_qrels(processed_dir: Path, split: str) -> dict:
    return read_json(processed_dir / split / "qrels.json")


def _dense_topk_single(
    model: SentenceTransformer,
    query: str,
    doc_embs: np.ndarray,
    doc_ids: list[str],
    k: int,
) -> list[tuple[str, float]]:
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(
        np.float32
    )[0]
    scores = doc_embs @ q_emb
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(doc_ids[int(i)], float(scores[int(i)])) for i in idx]


def _to_float_feature_list(feats: object) -> list[float]:
    """
    Your build_features() sometimes returns:
      - list[float] / tuple[float]
      - dict[str, float] (feature_name -> value)

    We normalize into list[float] in FEATURE_NAMES order.
    """
    if isinstance(feats, dict):
        out: list[float] = []
        for name in FEATURE_NAMES:
            v = feats.get(name, 0.0)
            out.append(float(v))
        return out

    if isinstance(feats, (list, tuple)):
        return [float(x) for x in feats]

    raise TypeError(f"Unsupported features type: {type(feats)}")


def _split_params_for_fit(params: dict) -> tuple[dict, dict]:
    """
    Prevent LightGBM warning:
      'Found eval_at in params. Will use it instead of eval_at argument'

    We remove eval_at / ndcg_eval_at from model params and pass them to fit().
    """
    params = dict(params)
    fit_kwargs: dict = {}

    eval_at = None
    if "eval_at" in params:
        eval_at = params.pop("eval_at")
    elif "ndcg_eval_at" in params:
        eval_at = params.pop("ndcg_eval_at")

    if eval_at is not None:
        fit_kwargs["eval_at"] = eval_at

    if "early_stopping_rounds" in params:
        fit_kwargs["early_stopping_rounds"] = params.pop("early_stopping_rounds")

    return params, fit_kwargs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/train.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tcfg = cfg["train"]

    processed_dir = Path(tcfg["dataset_processed_dir"])
    corpus_split = tcfg["corpus_split"]
    q_split = tcfg["query_split"]
    qrels_split = tcfg["qrels_split"]

    corpus = _load_corpus(processed_dir, corpus_split)
    queries = _load_queries(processed_dir, q_split)
    qrels_all = _load_qrels(processed_dir, qrels_split)

    # Load BM25 artifact
    with Path(tcfg["bm25_artifact"]).open("rb") as f:
        bm25 = pickle.load(f)

    # Load dense embeddings
    emb_dir = Path(tcfg["emb_dir"])
    doc_embs = np.load(emb_dir / "embeddings.npy").astype(np.float32)
    doc_ids = json.loads((emb_dir / "doc_ids.json").read_text(encoding="utf-8"))

    embed_model = SentenceTransformer(tcfg["embed_model"])

    bm25_k = int(tcfg["bm25_candidate_k"])
    dense_k = int(tcfg["dense_candidate_k"])
    alpha = float(tcfg["alpha"])
    feat_k = int(tcfg["train_feature_k"])
    min_rel = int(tcfg["min_relevance"])

    seed = int(tcfg["random_seed"])
    rnd = random.Random(seed)
    qids = list(queries.keys())
    rnd.shuffle(qids)
    val_n = int(len(qids) * float(tcfg["val_frac"]))
    val_set = set(qids[:val_n])

    X_train: list[list[float]] = []
    y_train: list[int] = []
    group_train: list[int] = []

    X_val: list[list[float]] = []
    y_val: list[int] = []
    group_val: list[int] = []

    for qid, qtext in queries.items():
        bm25_hits = bm25.query(qtext, k=bm25_k)
        dense_hits = _dense_topk_single(embed_model, qtext, doc_embs, doc_ids, dense_k)
        merged = hybrid_merge(bm25_hits, dense_hits, alpha=alpha)

        cand = merged[:feat_k]
        if not cand:
            log.warning("Skipping qid=%s: zero candidates after hybrid merge (bm25=%d dense=%d)",
                        qid, len(bm25_hits), len(dense_hits))
            continue

        bm25_map = {d: float(s) for d, s in bm25_hits}
        dense_map = {d: float(s) for d, s in dense_hits}
        hybrid_map = {d: float(s) for d, s in cand}

        X_rows: list[list[float]] = []
        y_rows: list[int] = []
        qr = qrels_all.get(qid, {})

        for did, _hs in cand:
            doc = corpus.get(did, {"title": "", "text": ""})
            label_raw = int(qr.get(did, 0))
            label = label_raw if label_raw >= min_rel else 0

            feats_obj = build_features(
                query=qtext,
                doc=doc,
                bm25_score=float(bm25_map.get(did, 0.0)),
                dense_score=float(dense_map.get(did, 0.0)),
                hybrid_score=float(hybrid_map.get(did, 0.0)),
            )
            feats = _to_float_feature_list(feats_obj)

            X_rows.append(feats)
            y_rows.append(label)

        if not X_rows:
            continue

        if qid in val_set:
            group_val.append(len(X_rows))
            X_val.extend(X_rows)
            y_val.extend(y_rows)
        else:
            group_train.append(len(X_rows))
            X_train.extend(X_rows)
            y_train.extend(y_rows)

    X_train_np = np.asarray(X_train, dtype=np.float32)
    y_train_np = np.asarray(y_train, dtype=np.int32)
    X_val_np = np.asarray(X_val, dtype=np.float32)
    y_val_np = np.asarray(y_val, dtype=np.int32)

    log.info("Train rows=%d queries=%d", X_train_np.shape[0], len(group_train))
    log.info("Val rows=%d queries=%d", X_val_np.shape[0], len(group_val))

    raw_params = dict(cfg["lightgbm"])
    params, fit_kwargs = _split_params_for_fit(raw_params)

    model = LGBMRanker(**params)

    if X_val_np.shape[0] > 0 and len(group_val) > 0:
        model.fit(
            X_train_np,
            y_train_np,
            group=group_train,
            eval_set=[(X_val_np, y_val_np)],
            eval_group=[group_val],
            **fit_kwargs,
        )
    else:
        model.fit(X_train_np, y_train_np, group=group_train)

    out_dir = ensure_dir(cfg["artifacts"]["ltr_dir"])
    model_path = out_dir / cfg["artifacts"]["model_name"]
    meta_path = out_dir / cfg["artifacts"]["meta_name"]

    with model_path.open("wb") as f:
        pickle.dump(model, f)

    write_json(
        meta_path,
        {
            "feature_names": FEATURE_NAMES,
            "train_config": tcfg,
            "lightgbm_params": params,
            "fit_kwargs": fit_kwargs,
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        },
    )

    log.info("Saved LTR model -> %s", model_path)
    log.info("Saved LTR meta  -> %s", meta_path)


if __name__ == "__main__":
    main()
