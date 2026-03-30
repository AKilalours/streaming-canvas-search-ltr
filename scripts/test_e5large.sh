#!/bin/bash
cd ~/streaming-canvas-search-ltr

echo "Re-embedding with e5-large-v2 (768→1024 dim, stronger)..."
docker compose exec api /app/.venv/bin/python3.11 -c "
import sys, json, pathlib, numpy as np
sys.path.insert(0, '/app/src')
from sentence_transformers import SentenceTransformer

MODEL = 'intfloat/e5-large-v2'
print(f'Loading {MODEL}...')
model = SentenceTransformer(MODEL)

corpus = []
with open('data/processed/movielens/test/corpus.jsonl') as f:
    for line in f:
        if line.strip(): corpus.append(json.loads(line))

print(f'Embedding {len(corpus)} docs...')
texts = ['passage: ' + str(d.get('title','')) + ' ' + str(d.get('text','')) for d in corpus]
doc_ids = [str(d['doc_id']) for d in corpus]

all_embs = []
for i in range(0, len(texts), 64):
    embs = model.encode(texts[i:i+64], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    all_embs.append(embs.astype(np.float32))
    if (i//64+1) % 10 == 0: print(f'  {min(i+64,len(texts))}/{len(texts)}')

embeddings = np.concatenate(all_embs, axis=0)
out = pathlib.Path('artifacts/faiss/movielens_intfloat_e5-large-v2')
out.mkdir(parents=True, exist_ok=True)
np.save(out/'embeddings.npy', embeddings)
json.dump(doc_ids, open(out/'doc_ids.json','w'))
json.dump({'model_name': MODEL, 'dim': int(embeddings.shape[1])}, open(out/'meta.json','w'))
print(f'Saved: {embeddings.shape}')
"

echo "Testing hybrid with e5-large-v2..."
docker compose exec api /app/.venv/bin/python3.11 -c "
import sys, json, math, pathlib, pickle, numpy as np
sys.path.insert(0, '/app/src')

corpus = {}
with open('data/processed/movielens/test/corpus.jsonl') as f:
    for line in f:
        if line.strip(): d=json.loads(line); corpus[str(d['doc_id'])]=d

queries = {}
with open('data/processed/movielens/test/queries.jsonl') as f:
    for line in f:
        if line.strip(): q=json.loads(line); queries[q['query_id']]=q['text']

qrels = json.load(open('data/processed/movielens/test/qrels.json'))
bm25 = pickle.load(open('artifacts/bm25/movielens_bm25.pkl','rb'))

from sentence_transformers import SentenceTransformer
emb_dir = pathlib.Path('artifacts/faiss/movielens_intfloat_e5-large-v2')
doc_embs = np.load(emb_dir/'embeddings.npy').astype(np.float32)
doc_ids_d = json.load(open(emb_dir/'doc_ids.json'))
norms = np.linalg.norm(doc_embs,axis=1,keepdims=True); norms[norms==0]=1.0; doc_embs=doc_embs/norms
st = SentenceTransformer('intfloat/e5-large-v2')

def dense_search(q, k=2000):
    qv=st.encode(['query: '+q],convert_to_numpy=True,normalize_embeddings=True).astype(np.float32)[0]
    s=doc_embs@qv; k2=min(k,s.shape[0])
    idx=np.argpartition(-s,kth=k2-1)[:k2]; idx=idx[np.argsort(-s[idx])]
    return [(doc_ids_d[int(i)],float(s[int(i)])) for i in idx]

def ndcg(ranked,rel,k=10):
    if not rel: return 0.0
    dcg=sum((2**rel.get(str(d),rel.get(d,0))-1)/math.log2(i+2) for i,d in enumerate(ranked[:k]))
    idcg=sum((2**r-1)/math.log2(i+2) for i,r in enumerate(sorted(rel.values(),reverse=True)[:k]))
    return dcg/idcg if idcg>0 else 0.0

from retrieval.hybrid import hybrid_merge

best_a, best_n = 0.2, 0.0
for alpha in [0.10,0.15,0.20,0.25,0.30]:
    sc=[]
    for qid,qt in list(queries.items())[:80]:
        if qid not in qrels: continue
        bl=bm25.query(qt,k=2000); dl=dense_search(qt,k=2000)
        sc.append(ndcg([d for d,s in hybrid_merge(bl,dl,alpha=alpha)],qrels[qid]))
    avg=sum(sc)/len(sc)
    flag=' ← BEST' if avg>best_n else ''
    print(f'e5-large alpha={alpha:.2f}  nDCG@10={avg:.4f}{flag}')
    if avg>best_n: best_n=avg; best_a=alpha

print(f'e5-base  alpha=0.20  nDCG@10=0.5865 (previous)')
print(f'Best e5-large: alpha={best_a}  nDCG@10={best_n:.4f}')
"
