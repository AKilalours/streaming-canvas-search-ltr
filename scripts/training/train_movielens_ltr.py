from __future__ import annotations
import argparse, json, math, pickle, re
from pathlib import Path
from typing import Any

import numpy as np
import lightgbm as lgb
from sentence_transformers import SentenceTransformer

# ---- small utils ----
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def parse_catalog(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    parts = [p.strip() for p in (text or "").split("|")]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        out[k.strip().lower()] = v.strip()
    return out

def dense_search(model: SentenceTransformer, doc_embs: np.ndarray, doc_ids: list[str], query: str, k: int):
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
    scores = doc_embs @ q
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(doc_ids[int(i)], float(scores[int(i)])) for i in idx]

def hybrid_merge(bm25: list[tuple[str,float]], dense: list[tuple[str,float]], alpha: float):
    # alpha*bm25 + (1-alpha)*dense after min-max normalization per list
    def norm(xs):
        if not xs:
            return {}
        vals = np.array([s for _, s in xs], dtype=np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-9:
            return {d: 1.0 for d, _ in xs}
        return {d: float((s - lo) / (hi - lo)) for d, s in xs}

    nb = norm(bm25)
    nd = norm(dense)
    keys = set(nb) | set(nd)
    out = []
    for d in keys:
        out.append((d, alpha * nb.get(d, 0.0) + (1.0 - alpha) * nd.get(d, 0.0)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--bm25_pkl", required=True)
    ap.add_argument("--dense_dir", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidate_k", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    corpus_rows = load_jsonl(Path(args.corpus))
    corpus = {str(r["doc_id"]): r for r in corpus_rows}

    bm25 = pickle.load(open(args.bm25_pkl, "rb"))

    dense_dir = Path(args.dense_dir)
    doc_embs = np.load(dense_dir / "embeddings.npy").astype(np.float32)
    doc_ids = list(load_json(dense_dir / "doc_ids.json"))

    # normalize embeddings defensively
    norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    doc_embs = doc_embs / norms

    enc = SentenceTransformer(args.model_name)

    # load queries + qrels
    qs = load_jsonl(Path(args.queries))
    rel = load_jsonl(Path(args.qrels))
    qid2pos: dict[str, set[str]] = {}
    for r in rel:
        if int(r.get("relevance", 0)) > 0:
            qid2pos.setdefault(str(r["query_id"]), set()).add(str(r["doc_id"]))

    feature_names = [
        "bm25", "dense",
        "title_overlap", "text_overlap",
        "title_len_log", "text_len_log",
        "tag_match", "genre_match",
    ]

    X = []
    y = []
    group = []

    for q in qs:
        qid = str(q["query_id"])
        qtext = str(q["text"])
        pos = qid2pos.get(qid, set())

        # candidates from bm25 + dense -> hybrid
        if hasattr(bm25, "query"):
            bm25_hits = list(bm25.query(qtext, k=args.candidate_k))
        else:
            bm25_hits = list(bm25.search(qtext, top_k=args.candidate_k))
        dense_hits = dense_search(enc, doc_embs, doc_ids, qtext, k=args.candidate_k)

        merged = hybrid_merge(bm25_hits, dense_hits, alpha=args.alpha)[: args.candidate_k]

        bm25_map = {d: float(s) for d, s in bm25_hits}
        dense_map = {d: float(s) for d, s in dense_hits}

        qtok = set(tokenize(qtext))

        n = 0
        for did, _ in merged:
            row = corpus.get(str(did), {})
            title = str(row.get("title") or "")
            text = str(row.get("text") or "")

            meta = parse_catalog(text)
            tags = (meta.get("tags") or "").lower()
            genres = (meta.get("genres") or "").lower()

            ttok = set(tokenize(title))
            xtok = set(tokenize(text))

            title_ov = len(qtok & ttok) / max(1, len(qtok))
            text_ov  = len(qtok & xtok) / max(1, len(qtok))

            tag_match = 1.0 if any(t in tags for t in qtok) else 0.0
            genre_match = 1.0 if any(t in genres for t in qtok) else 0.0

            feats = [
                bm25_map.get(str(did), 0.0),
                dense_map.get(str(did), 0.0),
                float(title_ov),
                float(text_ov),
                float(math.log1p(len(title))),
                float(math.log1p(len(text))),
                float(tag_match),
                float(genre_match),
            ]
            X.append(feats)
            y.append(1 if str(did) in pos else 0)
            n += 1

        if n > 0:
            group.append(n)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    train = lgb.Dataset(X, label=y, group=group, feature_name=feature_names)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "verbose": -1,
    }

    booster = lgb.train(params, train, num_boost_round=400)

    out = {"feature_names": feature_names, "model": booster}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)

    print(f"[OK] wrote LTR -> {args.out}")
    print(f"[OK] trained rows={X.shape[0]} groups={len(group)} positives={int(y.sum())}")

if __name__ == "__main__":
    main()
