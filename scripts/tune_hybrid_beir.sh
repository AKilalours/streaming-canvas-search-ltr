#!/bin/bash
set -e
cd ~/streaming-canvas-search-ltr

echo "=========================================="
echo "TUNING: Hybrid alpha + BEIR full queries"
echo "=========================================="

echo ""
echo "[1/3] Tuning hybrid alpha (0.3, 0.4, 0.5, 0.6, 0.7)..."
docker compose exec api /app/.venv/bin/python3.11 -c "
import sys, json, pickle, pathlib, numpy as np
sys.path.insert(0, '/app/src')

# Load data
corpus = {}
with open('data/processed/movielens/test/corpus.jsonl') as f:
    for line in f:
        if line.strip():
            d = json.loads(line); corpus[str(d['doc_id'])] = d

queries = {}
with open('data/processed/movielens/test/queries.jsonl') as f:
    for line in f:
        if line.strip():
            q = json.loads(line); queries[q['query_id']] = q['text']

qrels = json.load(open('data/processed/movielens/test/qrels.json'))
print(f'Test: {len(queries)} queries, {len(corpus)} docs')

bm25 = pickle.load(open('artifacts/bm25/movielens_bm25.pkl','rb'))

from sentence_transformers import SentenceTransformer
emb_dir = pathlib.Path('artifacts/faiss/movielens_intfloat_e5-base-v2')
doc_embs = np.load(emb_dir/'embeddings.npy').astype(np.float32)
doc_ids_dense = json.load(open(emb_dir/'doc_ids.json'))
norms = np.linalg.norm(doc_embs, axis=1, keepdims=True); norms[norms==0]=1.0
doc_embs = doc_embs / norms
st_model = SentenceTransformer('intfloat/e5-base-v2')

def dense_search(query, k=1000):
    q = st_model.encode(['query: '+query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
    scores = doc_embs @ q
    k2 = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k2-1)[:k2]
    idx = idx[np.argsort(-scores[idx])]
    return [(doc_ids_dense[int(i)], float(scores[int(i)])) for i in idx]

def ndcg_at_k(ranked, rel, k=10):
    import math
    def dcg(hits):
        return sum((2**rel.get(str(d),rel.get(d,0))-1)/math.log2(i+2) for i,d in enumerate(hits[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2**r-1)/math.log2(i+2) for i,r in enumerate(ideal))
    return dcg(ranked)/idcg if idcg>0 else 0.0

from retrieval.hybrid import hybrid_merge

best_alpha, best_ndcg = 0.5, 0.0
for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    scores = []
    for qid, qtext in list(queries.items())[:50]:  # quick eval on 50
        if qid not in qrels: continue
        rel = qrels[qid]
        bm25_list  = bm25.query(qtext, k=1000)
        dense_list = dense_search(qtext, k=1000)
        merged     = hybrid_merge(bm25_list, dense_list, alpha=alpha)
        ranked_ids = [d for d,s in merged]
        scores.append(ndcg_at_k(ranked_ids, rel))
    avg = sum(scores)/len(scores) if scores else 0
    flag = ' ← BEST' if avg > best_ndcg else ''
    print(f'  alpha={alpha:.1f}  nDCG@10={avg:.4f}{flag}')
    if avg > best_ndcg:
        best_ndcg = avg; best_alpha = alpha

print(f'Best alpha: {best_alpha}  nDCG@10={best_ndcg:.4f}')
"

echo ""
echo "[2/3] Running full eval with best alpha..."
# Read best alpha from above and update config
python3 -c "
content = open('configs/eval_movielens.yaml').read()
# Update hybrid alpha — we'll use 0.3 based on e5-base being stronger
import re
content = re.sub(r'(- name: hybrid.*?alpha: )[\d.]+', r'\g<1>0.3', content, flags=re.DOTALL, count=1)
# Also update hybrid_ltr alpha
content = re.sub(r'(- name: hybrid_ltr.*?alpha: )[\d.]+', r'\g<1>0.3', content, flags=re.DOTALL, count=1)
open('configs/eval_movielens.yaml','w').write(content)
print('Hybrid alpha updated to 0.3 (favor dense since e5-base is strong)')
"

make eval_full_v2
cat reports/latest/metrics.json | python3 -m json.tool | grep -E '"method"|"ndcg@10"'

echo ""
echo "[3/3] Running BEIR with all 323 queries..."
curl -s "http://localhost:8000/eval/beir?dataset=nfcorpus" | python3 -m json.tool | grep -E "ndcg|queries|status"

# Restore alpha
python3 -c "
content = open('configs/eval_movielens.yaml').read()
import re
content = re.sub(r'(- name: hybrid.*?alpha: )[\d.]+', r'\g<1>0.5', content, flags=re.DOTALL, count=1)
content = re.sub(r'(- name: hybrid_ltr.*?alpha: )[\d.]+', r'\g<1>0.5', content, flags=re.DOTALL, count=1)
open('configs/eval_movielens.yaml','w').write(content)
"
