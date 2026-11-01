#!/bin/bash
set -e
cd ~/streaming-canvas-search-ltr

echo "=========================================="
echo "STREAMLENS PHENOMENAL EVAL PIPELINE"
echo "=========================================="

echo ""
echo "[1/4] Rebuilding BM25 (k1=1.5, b=0.65, title_boost=3x)..."
docker compose exec api /app/.venv/bin/python3.11 -c "
import sys, json, pickle, math, pathlib
from collections import defaultdict, Counter
sys.path.insert(0, '/app/src')
from retrieval.bm25 import BM25Artifact, tokenize

rows = []
with open('data/processed/movielens/train/corpus.jsonl') as f:
    for line in f:
        if line.strip(): rows.append(json.loads(line))

K1, B = 1.3, 0.70
doc_ids, doc_len, postings, df = [], [], defaultdict(list), Counter()
for idx, r in enumerate(rows):
    title = str(r.get('title') or '')
    text  = str(r.get('text') or '')
    full  = f'{title} {title} {text}'.strip()  # title_boost=2x
    toks  = tokenize(full) or ['__empty__']
    tf    = Counter(toks)
    doc_ids.append(str(r['doc_id']))
    doc_len.append(sum(tf.values()))
    for t in tf: df[t] += 1
    for t, c in tf.items(): postings[t].append((idx, int(c)))

N = len(doc_ids)
avgdl = sum(doc_len) / max(1, N)
idf = {t: math.log(1.0+(N-dfi+0.5)/(dfi+0.5)) for t,dfi in df.items()}
art = BM25Artifact(doc_ids=doc_ids, doc_len=doc_len, avgdl=avgdl,
    idf=idf, postings=dict(postings), k1=K1, b=B)
pickle.dump(art, open('artifacts/bm25/movielens_bm25_v2.pkl','wb'))
print(f'BM25 v2 saved: {N} docs, k1={K1}, b={B}, title_boost=2x')
"

echo ""
echo "[2/4] Retraining LTR (candidate_k=2000, n_estimators=500)..."
docker compose exec api /app/.venv/bin/python3.11 -c "
import sys, json, pickle, pathlib, numpy as np
from collections import Counter
sys.path.insert(0, '/app/src')

corpus = {}
with open('data/processed/movielens/train/corpus.jsonl') as f:
    for line in f:
        if line.strip():
            d = json.loads(line); corpus[str(d['doc_id'])] = d

queries = {}
with open('data/processed/movielens/train/queries.jsonl') as f:
    for line in f:
        if line.strip():
            q = json.loads(line); queries[q['query_id']] = q['text']

qrels = json.load(open('data/processed/movielens/train/qrels.json'))
print(f'Corpus={len(corpus)}, Queries={len(queries)}, Qrels={len(qrels)}')

bm25 = pickle.load(open('artifacts/bm25/movielens_bm25_v2.pkl','rb'))

from sentence_transformers import SentenceTransformer
emb_dir = pathlib.Path('artifacts/faiss/movielens_sentence-transformers_all-MiniLM-L6-v2')
doc_embs = np.load(emb_dir/'embeddings.npy').astype(np.float32)
doc_ids_dense = json.load(open(emb_dir/'doc_ids.json'))
norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
norms[norms==0] = 1.0
doc_embs = doc_embs / norms
st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def dense_search(query, k=2000):
    q = st_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
    scores = doc_embs @ q
    k2 = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k2-1)[:k2]
    idx = idx[np.argsort(-scores[idx])]
    return [(doc_ids_dense[int(i)], float(scores[int(i)])) for i in idx]

from retrieval.hybrid import hybrid_merge
from ranking.features import build_features

CANDIDATE_K = 2000
X_rows, y_rows, qids = [], [], []

for i, (qid, qtext) in enumerate(list(queries.items())):
    if qid not in qrels: continue
    rel = qrels[qid]
    bm25_list  = bm25.query(qtext, k=CANDIDATE_K)
    dense_list = dense_search(qtext, k=CANDIDATE_K)
    bm25_dict  = dict(bm25_list)
    dense_dict = dict(dense_list)
    merged     = hybrid_merge(bm25_list, dense_list, alpha=0.5)
    for doc_id, score in merged[:500]:
        doc   = corpus.get(str(doc_id), {})
        label = int(rel.get(str(doc_id), rel.get(doc_id, 0)) or 0)
        feats = build_features(
            query=qtext, doc=doc,
            bm25_score=bm25_dict.get(doc_id, bm25_dict.get(str(doc_id), 0.0)),
            dense_score=dense_dict.get(doc_id, dense_dict.get(str(doc_id), 0.0)),
            hybrid_score=score)
        X_rows.append(feats); y_rows.append(label); qids.append(qid)
    if (i+1) % 100 == 0: print(f'  {i+1}/{len(queries)} queries')

print(f'Rows={len(X_rows)}, positives={sum(1 for y in y_rows if y>0)}')
X = np.array([list(f.values()) for f in X_rows])
y = np.array(y_rows)
counts = Counter(qids)
groups = [counts[q] for q in dict.fromkeys(qids)]

import lightgbm as lgb
model = lgb.LGBMRanker(objective='lambdarank', n_estimators=500,
    num_leaves=63, learning_rate=0.05, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
model.fit(X, y, group=groups)
pickle.dump(model, open('artifacts/ltr/movielens_ltr_v2.pkl','wb'))
print('LTR v2 saved!')
"

echo ""
echo "[3/4] Running eval with new artifacts..."
python3 -c "
content = open('configs/eval_movielens.yaml').read()
content = content.replace('movielens_bm25.pkl','movielens_bm25_v2.pkl')
content = content.replace('movielens_ltr.pkl','movielens_ltr_v2.pkl')
content = content.replace('bm25_candidate_k: 1000','bm25_candidate_k: 2000')
content = content.replace('rerank_k: 200','rerank_k: 300')
open('configs/eval_movielens.yaml','w').write(content)
print('Config updated')
"
make eval_full_v2

echo ""
echo "[4/4] FINAL RESULTS vs PHENOMENAL TARGETS:"
python3 -c "
import json
m = json.load(open('reports/latest/metrics.json'))
print(f'Queries={m[\"diagnostics\"][\"num_queries\"]}  candidate_k={m[\"diagnostics\"][\"bm25_candidate_k_max\"]}')
print()
targets = {'bm25':0.65,'dense':0.50,'hybrid':0.65,'hybrid_ltr':0.80}
for method in m['methods']:
    name=method['method']; ndcg=method['ndcg@10']
    target=targets.get(name,0)
    flag='✅ PHENOMENAL' if ndcg>=target else f'⚠️  target={target:.2f}'
    print(f'{name:15} nDCG@10={ndcg:.4f}  {flag}')
"

python3 -c "
content = open('configs/eval_movielens.yaml').read()
content = content.replace('movielens_bm25_v2.pkl','movielens_bm25.pkl')
content = content.replace('movielens_ltr_v2.pkl','movielens_ltr.pkl')
content = content.replace('bm25_candidate_k: 2000','bm25_candidate_k: 1000')
content = content.replace('rerank_k: 300','rerank_k: 200')
open('configs/eval_movielens.yaml','w').write(content)
print('Config restored')
"
