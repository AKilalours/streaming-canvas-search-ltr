#!/bin/bash
cd ~/streaming-canvas-search-ltr
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
emb_dir = pathlib.Path('artifacts/faiss/movielens_intfloat_e5-base-v2')
doc_embs = np.load(emb_dir/'embeddings.npy').astype(np.float32)
doc_ids_d = json.load(open(emb_dir/'doc_ids.json'))
norms = np.linalg.norm(doc_embs,axis=1,keepdims=True); norms[norms==0]=1.0; doc_embs=doc_embs/norms
st = SentenceTransformer('intfloat/e5-base-v2')

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
from retrieval.bm25 import tokenize

def prf_expand(query, bm25_top, corpus, n_terms=5):
    from collections import Counter
    stop={'the','a','an','in','of','to','and','for','is','with','its','as','by','on','at','film','movie'}
    tc=Counter()
    for doc_id,_ in bm25_top[:3]:
        doc=corpus.get(str(doc_id),{})
        text=str(doc.get('title',''))+' '+str(doc.get('text',''))
        toks=[t for t in tokenize(text) if len(t)>3 and t not in stop]
        for t in toks[:50]: tc[t]+=1
    qtoks=set(tokenize(query))
    return query+' '+' '.join(t for t,_ in tc.most_common(20) if t not in qtoks)[:n_terms*8]

results={'baseline_02':[],'prf_02':[],'prf_015':[]}
qlist=[(qid,qt) for qid,qt in list(queries.items())[:80] if qid in qrels]

for qid,qtext in qlist:
    rel=qrels[qid]
    bl=bm25.query(qtext,k=2000)
    dl=dense_search(qtext,k=2000)
    results['baseline_02'].append(ndcg([d for d,s in hybrid_merge(bl,dl,alpha=0.2)],rel))
    exp=prf_expand(qtext,bl,corpus)
    bl2=bm25.query(exp,k=2000)
    results['prf_02'].append(ndcg([d for d,s in hybrid_merge(bl2,dl,alpha=0.2)],rel))
    results['prf_015'].append(ndcg([d for d,s in hybrid_merge(bl2,dl,alpha=0.15)],rel))

for name,sc in results.items():
    print(f'{name:20} nDCG@10={sum(sc)/len(sc):.4f}')
"
