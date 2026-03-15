# flows/train_ltr.py
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import joblib
import ir_datasets
import pytrec_eval

from app.deps import load_state
from retrieval.hybrid import hybrid_merge


_TOKEN = re.compile(r"[a-z0-9]+")


def tok(s: str) -> list[str]:
    return _TOKEN.findall((s or "").lower())


def overlap_ratio(q: str, d: str) -> float:
    qt = tok(q)
    if not qt:
        return 0.0
    dt = set(tok(d))
    hits = sum(1 for t in qt if t in dt)
    return hits / max(1, len(qt))


def safe_len(s: str) -> int:
    return len(tok(s))


def load_ir_dataset(did: str):
    return ir_datasets.load(did)


def build_qrels(ds) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for r in ds.qrels_iter():
        qrels.setdefault(str(r.query_id), {})[str(r.doc_id)] = int(r.relevance)
    return qrels


@dataclass
class LTRReranker:
    model: Any
    feature_names: list[str]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    @staticmethod
    def load(path: str) -> "LTRReranker":
        obj = joblib.load(path)
        if isinstance(obj, dict) and "model" in obj:
            return LTRReranker(model=obj["model"], feature_names=list(obj.get("feature_names", [])))
        raise RuntimeError(f"Unexpected LTR artifact format at {path}")

    def featurize(
        self,
        *,
        query: str,
        doc_id: str,
        corpus: dict[str, Any],
        bm25_scores: dict[str, float],
        dense_scores: dict[str, float],
        merged_score: float,
    ) -> list[float]:
        row = corpus.get(str(doc_id), {})
        title = str(row.get("title") or "")
        text = str(row.get("text") or "")

        f_bm25 = float(bm25_scores.get(str(doc_id), 0.0))
        f_dense = float(dense_scores.get(str(doc_id), 0.0))
        f_merge = float(merged_score)

        qlen = float(safe_len(query))
        tlen = float(safe_len(title))
        dlen = float(safe_len(text))

        o_text = float(overlap_ratio(query, text))
        o_title = float(overlap_ratio(query, title))

        return [f_bm25, f_dense, f_merge, qlen, tlen, dlen, o_title, o_text]

    def rerank(
        self,
        *,
        query: str,
        corpus: dict[str, Any],
        candidates: list[tuple[str, float]],
        bm25_scores: dict[str, float] | None = None,
        dense_scores: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        bm25_scores = bm25_scores or {}
        dense_scores = dense_scores or {}

        X = []
        doc_ids = []
        for did, merged_score in candidates:
            doc_ids.append(str(did))
            X.append(
                self.featurize(
                    query=query,
                    doc_id=str(did),
                    corpus=corpus,
                    bm25_scores=bm25_scores,
                    dense_scores=dense_scores,
                    merged_score=float(merged_score),
                )
            )
        X = np.asarray(X, dtype=np.float32)
        scores = self.model.predict(X)
        pairs = list(zip(doc_ids, [float(s) for s in scores]))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs


def build_candidates(state, query: str, *, candidate_k: int, alpha: float):
    bm25 = list(state.bm25_query(query, k=candidate_k))
    dense = list(state.dense.search(query, k=candidate_k)) if state.dense is not None else []
    merged = hybrid_merge(bm25, dense, alpha=alpha)

    bm25_scores = {d: float(s) for d, s in bm25}
    dense_scores = {d: float(s) for d, s in dense}
    return merged, bm25_scores, dense_scores


def trec_eval(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]]):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "map_cut.10", "recall.10", "recall.100"})
    res = evaluator.evaluate(run)
    ndcg10 = float(np.mean([v.get("ndcg_cut_10", 0.0) for v in res.values()])) if res else 0.0
    map10 = float(np.mean([v.get("map_cut_10", 0.0) for v in res.values()])) if res else 0.0
    r10 = float(np.mean([v.get("recall_10", 0.0) for v in res.values()])) if res else 0.0
    r100 = float(np.mean([v.get("recall_100", 0.0) for v in res.values()])) if res else 0.0
    return {"ndcg@10": ndcg10, "map@10": map10, "recall@10": r10, "recall@100": r100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--candidate-k", type=int, default=1000)
    ap.add_argument("--rerank-k", type=int, default=200)
    ap.add_argument("--limit-train", type=int, default=2000)
    ap.add_argument("--limit-dev", type=int, default=500)
    ap.add_argument("--out", default="artifacts/ltr/ltr.pkl")
    ap.add_argument("--meta", default="artifacts/ltr/nfcorpus_ltr_meta.json")
    ap.add_argument("--report", default="reports/latest_eval/ltr_train_eval.json")
    args = ap.parse_args()

    # Load retrieval state (bm25 + dense + corpus)
    state = load_state()
    if state is None or not getattr(state, "ready", False):
        raise RuntimeError("State not ready. Ensure artifacts exist and load_state() works.")
    if state.dense is None:
        raise RuntimeError("Dense index not loaded. Build embeddings first (faiss artifacts).")

    # datasets
    ds_train = load_ir_dataset("beir/nfcorpus/train")
    ds_dev = load_ir_dataset("beir/nfcorpus/dev")

    qrels_train = build_qrels(ds_train)
    qrels_dev = build_qrels(ds_dev)

    train_q = list(ds_train.queries_iter())
    dev_q = list(ds_dev.queries_iter())

    if args.limit_train and args.limit_train < len(train_q):
        train_q = train_q[: args.limit_train]
    if args.limit_dev and args.limit_dev < len(dev_q):
        dev_q = dev_q[: args.limit_dev]

    # Build training data
    X_rows: list[list[float]] = []
    y: list[int] = []
    group: list[int] = []

    feature_names = ["bm25", "dense", "merged", "q_len", "title_len", "doc_len", "overlap_title", "overlap_text"]

    for q in train_q:
        qid = str(q.query_id)
        text = str(q.text)
        rels = qrels_train.get(qid, {})

        merged, bm25_scores, dense_scores = build_candidates(state, text, candidate_k=args.candidate_k, alpha=args.alpha)

        # Train on top candidate_k merged
        rows_this_q = 0
        for did, merged_score in merged[: args.candidate_k]:
            label = int(rels.get(str(did), 0) > 0)

            row = state.corpus.get(str(did), {})
            title = str(row.get("title") or "")
            dtext = str(row.get("text") or "")

            feats = [
                float(bm25_scores.get(str(did), 0.0)),
                float(dense_scores.get(str(did), 0.0)),
                float(merged_score),
                float(safe_len(text)),
                float(safe_len(title)),
                float(safe_len(dtext)),
                float(overlap_ratio(text, title)),
                float(overlap_ratio(text, dtext)),
            ]
            X_rows.append(feats)
            y.append(label)
            rows_this_q += 1

        if rows_this_q > 0:
            group.append(rows_this_q)

    X = np.asarray(X_rows, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int32)

    # Train LambdaMART
    import lightgbm as lgb

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[10],
        learning_rate=0.05,
        n_estimators=600,
        num_leaves=31,
        min_data_in_leaf=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    ranker.fit(X, y_arr, group=group)

    reranker = LTRReranker(model=ranker, feature_names=feature_names)
    reranker.save(args.out)

    meta = {
        "dataset": "beir/nfcorpus",
        "alpha": args.alpha,
        "candidate_k": args.candidate_k,
        "rerank_k": args.rerank_k,
        "features": feature_names,
        "train_queries": len(train_q),
        "train_rows": int(X.shape[0]),
        "out": args.out,
    }
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta).write_text(json.dumps(meta, indent=2))

    # Evaluate on dev: baseline hybrid vs hybrid+ltr
    run_hybrid: dict[str, dict[str, float]] = {}
    run_ltr: dict[str, dict[str, float]] = {}

    for q in dev_q:
        qid = str(q.query_id)
        qtext = str(q.text)

        merged, bm25_scores, dense_scores = build_candidates(state, qtext, candidate_k=args.candidate_k, alpha=args.alpha)

        # baseline hybrid run
        run_hybrid[qid] = {str(d): float(s) for d, s in merged[:100]}

        # ltr rerank on top rerank_k
        head = merged[: min(args.rerank_k, len(merged))]
        tail = merged[min(args.rerank_k, len(merged)) :]

        reranked = reranker.rerank(
            query=qtext,
            corpus=state.corpus,
            candidates=head,
            bm25_scores=bm25_scores,
            dense_scores=dense_scores,
        )
        reranked_ids = {d for d, _ in reranked}
        tail2 = [(d, s) for d, s in merged if d not in reranked_ids]

        final = reranked + [(d, float(s)) for d, s in tail2]
        run_ltr[qid] = {str(d): float(s) for d, s in final[:100]}

    ev_h = trec_eval(qrels_dev, run_hybrid)
    ev_l = trec_eval(qrels_dev, run_ltr)

    report = {
        "dev_metrics": {"hybrid": ev_h, "hybrid_ltr": ev_l, "lift_ndcg@10": float(ev_l["ndcg@10"] - ev_h["ndcg@10"])},
        "meta": meta,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(" saved:", args.out)
    print(" meta :", args.meta)
    print(" report:", args.report)
    print("DEV hybrid ndcg@10:", ev_h["ndcg@10"])
    print("DEV ltr    ndcg@10:", ev_l["ndcg@10"])
    print("DEV lift   ndcg@10:", report["dev_metrics"]["lift_ndcg@10"])


if __name__ == "__main__":
    main()
