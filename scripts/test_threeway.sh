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

# Build title-only BM25
from collections import defaultdict, Counter
from retrieval.bm25 import BM25Artifact, tokenize
import math as _math

doc_ids2, doc_len2, postings2, df2 = [], [], defaultdict(list), Counter()
for idx, (did, doc) in enumerate(corpus.items()):
    title = str(doc.get('title',''))
    toks = tokenize(title) or ['__empty__']
    tf = Counter(toks)
    doc_ids2.append(did); doc_len2.append(sum(tf.values()))
    for t in tf: df2[t] += 1
    for t, c in tf.items(): postings2[t].append((idx, int(c)))

N2 = len(doc_ids2); avgdl2 = sum(doc_len2)/max(1,N2)
idf2 = {t: _math.log(1.0+(N2-dfi+0.5)/(dfi+0.5)) for t,dfi in df2.items()}
title_bm25 = BM25Artifact(doc_ids=doc_ids2,doc_len=doc_len2,avgdl=avgdl2,idf=idf2,postings=dict(postings2))

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

def minmax(d):
    if not d: return d
    mn,mx=min(d.values()),max(d.values())
    r=mx-mn
    return {k:(v-mn)/(r+1e-9) for k,v in d.items()}

def threeway_merge(bl, tbl, dl, w_bm25=0.4, w_title=0.3, w_dense=0.3):
    bm25_d=minmax(dict(bl)); title_d=minmax(dict(tbl)); dense_d=minmax(dict(dl))
    all_ids=set(bm25_d)|set(title_d)|set(dense_d)
    scores={did: w_bm25*bm25_d.get(did,0)+w_title*title_d.get(did,0)+w_dense*dense_d.get(did,0) for did in all_ids}
    return sorted(scores.items(),key=lambda x:-x[1])

from retrieval.hybrid import hybrid_merge

results={'baseline_02':[],'3way_433':[],'3way_442':[],'3way_352':[]}
qlist=[(qid,qt) for qid,qt in list(queries.items())[:80] if qid in qrels]

for qid,qtext in qlist:
    rel=qrels[qid]
    bl=bm25.query(qtext,k=2000)
    tbl=title_bm25.query(qtext,k=2000)
    dl=dense_search(qtext,k=2000)
    results['baseline_02'].append(ndcg([d for d,s in hybrid_merge(bl,dl,alpha=0.2)],rel))
    results['3way_433'].append(ndcg([d for d,s in threeway_merge(bl,tbl,dl,0.4,0.3,0.3)],rel))
    results['3way_442'].append(ndcg([d for d,s in threeway_merge(bl,tbl,dl,0.4,0.4,0.2)],rel))
    results['3way_352'].append(ndcg([d for d,s in threeway_merge(bl,tbl,dl,0.3,0.5,0.2)],rel))

for name,sc in results.items():
    print(f'{name:20} nDCG@10={sum(sc)/len(sc):.4f}')
"
