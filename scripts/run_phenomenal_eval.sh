#!/bin/bash
set -e
echo "=========================================="
echo "STREAMLENS PHENOMENAL EVAL PIPELINE"
echo "=========================================="

cd ~/streaming-canvas-search-ltr

# Step 1: Rebuild BM25 with tuned params
echo ""
echo "[1/4] Rebuilding BM25 (k1=1.5, b=0.65, title_boost=3x)..."
docker compose exec api /app/.venv/bin/python3.11 -c "
import sys, json, pickle, pathlib
sys.path.insert(0, '/app/src')
from retrieval.bm25 import BM25Index

corpus = []
with open('data/processed/movielens/train/corpus.jsonl') as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            title = d.get('title','')
            text = d.get('text','')
            # Title boost: repeat 3x for higher lexical weight
            d['text'] = f'{title} {title} {title} {text}'
            corpus.append(d)

print(f'Building BM25 on {len(corpus)} docs...')
idx = BM25Index(k1=1.5, b=0.65)
idx.build(corpus)
out = 'artifacts/bm25/movielens_bm25_v2.pkl'
pickle.dump(idx, open(out,'wb'))
print(f'BM25 v2 saved: k1=1.5, b=0.65, title_boost=3x')
"

# Step 2: Retrain LTR with candidate_k=2000
echo ""
echo "[2/4] Retraining LTR (candidate_k=2000, n_estimators=500)..."
docker compose exec api /app/.venv/bin/python3.11 -c "
import sys, json, pickle, pathlib, numpy as np
from collections import Counter
sys.path.insert(0, '/app/src')

print('Loading data...')
corpus = {}
with open('data/processed/movielens/train/corpus.jsonl') as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            corpus[str(d['doc_id'])] = d

queries = {}
with open('data/processed/movielens/train/queries.jsonl') as f:
    for line in f:
        if line.strip():
            q = json.loads(line)
            queries[q['query_id']] = q['text']

qrels = json.load(open('data/processed/movielens/train/qrels.json'))
print(f'Corpus={len(corpus)}, Queries={len(queries)}, Qrels={len(qrels)}')

bm25 = pickle.load(open('artifacts/bm25/movielens_bm25.pkl','rb'))

from retrieval.embed_index import EmbedIndex
dense = EmbedIndex.load('artifacts/faiss/movielens_sentence-transformers_all-MiniLM-L6-v2')
from retrieval.hybrid import hybrid_merge
from ranking.features import build_features

CANDIDATE_K = 2000
X_rows, y_rows, qids = [], [], []

for i, (qid, qtext) in enumerate(list(queries.items())):
    if qid not in qrels: continue
    rel = qrels[qid]
    bm25_hits = dict(bm25.query(qtext, k=CANDIDATE_K))
    dense_hits = dict(dense.query(qtext, k=CANDIDATE_K))
    merged = hybrid_merge(bm25_hits, dense_hits, alpha=0.5, k=CANDIDATE_K)
    for doc_id, score in merged[:500]:
        doc = corpus.get(str(doc_id), {})
        label = int(rel.get(str(doc_id), rel.get(doc_id, 0)) or 0)
        feats = build_features(query=qtext, doc_id=str(doc_id), doc=doc,
            bm25_score=bm25_hits.get(doc_id,0.0),
            dense_score=dense_hits.get(doc_id,0.0),
            hybrid_score=score, rank=len(X_rows))
        X_rows.append(feats); y_rows.append(label); qids.append(qid)
    if (i+1) % 100 == 0: print(f'  {i+1}/{len(queries)} queries')

print(f'Training rows: {len(X_rows)}, positive: {sum(1 for y in y_rows if y>0)}')
X = np.array(X_rows); y = np.array(y_rows)
qid_counts = Counter(qids)
groups = [qid_counts[q] for q in dict.fromkeys(qids)]

import lightgbm as lgb
model = lgb.LGBMRanker(objective='lambdarank', n_estimators=500, num_leaves=63,
    learning_rate=0.05, min_child_samples=10, subsample=0.8,
    colsample_bytree=0.8, random_state=42, n_jobs=-1)
model.fit(X, y, group=groups)
pickle.dump(model, open('artifacts/ltr/movielens_ltr_v2.pkl','wb'))
print('LTR v2 saved!')
"

# Step 3: Run full eval with new artifacts
echo ""
echo "[3/4] Running full eval with new artifacts..."
# Update config temporarily
cp configs/eval_movielens.yaml configs/eval_movielens.yaml.bak
sed -i.tmp 's/movielens_bm25.pkl/movielens_bm25_v2.pkl/' configs/eval_movielens.yaml
sed -i.tmp 's/movielens_ltr.pkl/movielens_ltr_v2.pkl/' configs/eval_movielens.yaml
sed -i.tmp 's/bm25_candidate_k: 1000/bm25_candidate_k: 2000/' configs/eval_movielens.yaml
sed -i.tmp 's/dense_candidate_k: 1000/dense_candidate_k: 1000/' configs/eval_movielens.yaml
sed -i.tmp 's/rerank_k: 200/rerank_k: 300/' configs/eval_movielens.yaml

make eval_full_v2

# Step 4: Show results
echo ""
echo "[4/4] Results:"
python3 -c "
import json
m = json.load(open('reports/latest/metrics.json'))
print(f'num_queries: {m[\"diagnostics\"][\"num_queries\"]}')
print(f'bm25_candidate_k: {m[\"diagnostics\"][\"bm25_candidate_k_max\"]}')
print()
for method in m['methods']:
    name = method['method']
    ndcg = method['ndcg@10']
    r100 = method['recall@100']
    mrr  = method.get('mrr@10', 'N/A')
    print(f'{name:15} nDCG@10={ndcg:.4f}  recall@100={r100:.4f}')
"

echo ""
echo "DONE. Compare against phenomenal targets:"
echo "  BM25  nDCG@10 >= 0.65"
echo "  LTR   nDCG@10 >= 0.80"
echo "  LTR   recall@100 >= 0.92 (dense-qrel definition)"
