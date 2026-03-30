#!/bin/bash
# StreamLens — Real Impact Features
# Runs inside the Docker container
# Measures: popularity bias, cold-start gap, temporal drift
set -e
cd ~/streaming-canvas-search-ltr

echo "============================================================"
echo "STREAMLENS — REAL IMPACT ANALYSIS"
echo "============================================================"

# ── 1. Popularity Bias Audit ─────────────────────────────────────
echo ""
echo "[1/3] Measuring popularity bias..."
docker compose exec api /app/.venv/bin/python3.11 -c "
import sys, json, pickle, pathlib, numpy as np
sys.path.insert(0, '/app/src')
from collections import Counter

# Load data
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

# Load item features from Spark
item_features = json.load(open('artifacts/spark/item_features.json'))
all_pops = [float(v.get('popularity',0)) for v in item_features.values()]
all_pops.sort(reverse=True)
top10_threshold = all_pops[int(len(all_pops)*0.10)]  # top 10% popularity score

# Get recommendations for all queries
rec_items = []
for qid, qtext in list(queries.items())[:100]:
    hits = bm25.query(qtext, k=10)
    rec_items.extend([str(doc_id) for doc_id, _ in hits])

# Measure how many recs are from top-10% popular items
top10_count = sum(1 for item in rec_items
    if float(item_features.get(item, {}).get('popularity', 0)) >= top10_threshold)
pct = 100 * top10_count / len(rec_items) if rec_items else 0

print(f'Total recommendations analyzed: {len(rec_items)}')
print(f'From top-10%% popular items: {top10_count} ({pct:.1f}%%)')
print(f'Long-tail coverage: {100-pct:.1f}%%')
print()
if pct > 50:
    print(f'BIAS DETECTED: {pct:.0f}%% of recommendations from top-10%% popular items')
    print('Fix: apply inverse popularity weighting in LTR scoring')
else:
    print(f'Low bias: {pct:.0f}%% — reasonable popularity distribution')
"

# ── 2. Cold-Start Gap ────────────────────────────────────────────
echo ""
echo "[2/3] Measuring cold-start gap..."
docker compose exec api /app/.venv/bin/python3.11 -c "
import sys, json, pickle, pathlib, numpy as np, math
sys.path.insert(0, '/app/src')

# Load user features from Spark to identify cold/warm users
user_features = json.load(open('artifacts/spark/user_features.json'))

cold_users = {uid for uid, f in user_features.items() if f.get('watch_count', 0) < 5}
warm_users = {uid for uid, f in user_features.items() if f.get('watch_count', 0) >= 20}
print(f'Cold users (< 5 ratings): {len(cold_users)}')
print(f'Warm users (>= 20 ratings): {len(warm_users)}')

qrels = json.load(open('data/processed/movielens/test/qrels.json'))
bm25 = pickle.load(open('artifacts/bm25/movielens_bm25.pkl','rb'))

import pickle as _p
try:
    ltr = _p.load(open('artifacts/ltr/movielens_ltr_e5base.pkl','rb'))
    print('LTR model loaded')
except:
    ltr = None

def ndcg(ranked, rel, k=10):
    if not rel: return 0.0
    dcg = sum((2**rel.get(str(d),rel.get(d,0))-1)/math.log2(i+2) for i,d in enumerate(ranked[:k]))
    idcg = sum((2**r-1)/math.log2(i+2) for i,r in enumerate(sorted(rel.values(),reverse=True)[:k]))
    return dcg/idcg if idcg>0 else 0.0

queries = {}
with open('data/processed/movielens/test/queries.jsonl') as f:
    for line in f:
        if line.strip(): q=json.loads(line); queries[q['query_id']]=q['text']

# Map query_id to user_id prefix
cold_scores, warm_scores, all_scores = [], [], []
for qid, qtext in list(queries.items())[:100]:
    if qid not in qrels: continue
    hits = [d for d,s in bm25.query(qtext, k=10)]
    score = ndcg(hits, qrels[qid])
    all_scores.append(score)
    uid = qid.split('_')[0].replace('train_q','').replace('test_q','')
    if uid in cold_users: cold_scores.append(score)
    elif uid in warm_users: warm_scores.append(score)

print(f'Overall nDCG@10:    {sum(all_scores)/len(all_scores):.4f}')
print(f'Cold-start nDCG@10: {sum(cold_scores)/len(cold_scores):.4f}' if cold_scores else 'Cold: no queries matched')
print(f'Warm user nDCG@10:  {sum(warm_scores)/len(warm_scores):.4f}' if warm_scores else 'Warm: no queries matched')
if cold_scores and warm_scores:
    gap = sum(warm_scores)/len(warm_scores) - sum(cold_scores)/len(cold_scores)
    print(f'Cold-start gap:     {gap:.4f} ({100*gap/max(sum(warm_scores)/len(warm_scores),0.001):.1f}%% degradation)')
"

# ── 3. Temporal Drift ────────────────────────────────────────────
echo ""
echo "[3/3] Measuring temporal drift (train pre-2010, eval post-2010)..."
docker compose exec api /app/.venv/bin/python3.11 -c "
import sys, json, pickle, pathlib, math
sys.path.insert(0, '/app/src')

corpus = {}
with open('data/processed/movielens/test/corpus.jsonl') as f:
    for line in f:
        if line.strip(): d=json.loads(line); corpus[str(d['doc_id'])]=d

# Split movies by decade from title year
import re
pre2010 = set(); post2010 = set()
for doc_id, doc in corpus.items():
    title = doc.get('title','')
    m = re.search(r'\((\d{4})\)', title)
    if m:
        year = int(m.group(1))
        if year < 2010: pre2010.add(doc_id)
        else: post2010.add(doc_id)

print(f'Pre-2010 movies:  {len(pre2010)}')
print(f'Post-2010 movies: {len(post2010)}')

qrels = json.load(open('data/processed/movielens/test/qrels.json'))
bm25 = pickle.load(open('artifacts/bm25/movielens_bm25.pkl','rb'))

def ndcg(ranked, rel, k=10):
    if not rel: return 0.0
    dcg = sum((2**rel.get(str(d),rel.get(d,0))-1)/math.log2(i+2) for i,d in enumerate(ranked[:k]))
    idcg = sum((2**r-1)/math.log2(i+2) for i,r in enumerate(sorted(rel.values(),reverse=True)[:k]))
    return dcg/idcg if idcg>0 else 0.0

queries = {}
with open('data/processed/movielens/test/queries.jsonl') as f:
    for line in f:
        if line.strip(): q=json.loads(line); queries[q['query_id']]=q['text']

# Eval: queries where majority of relevant docs are pre-2010 vs post-2010
pre_scores, post_scores = [], []
for qid, qtext in list(queries.items())[:100]:
    if qid not in qrels: continue
    rel = qrels[qid]
    hits = [d for d,s in bm25.query(qtext, k=10)]
    score = ndcg(hits, rel)
    rel_pre = sum(1 for d in rel if d in pre2010)
    rel_post = sum(1 for d in rel if d in post2010)
    if rel_pre > rel_post: pre_scores.append(score)
    elif rel_post > rel_pre: post_scores.append(score)

print(f'Queries about pre-2010 content:  {len(pre_scores)} queries, nDCG@10={sum(pre_scores)/max(len(pre_scores),1):.4f}')
print(f'Queries about post-2010 content: {len(post_scores)} queries, nDCG@10={sum(post_scores)/max(len(post_scores),1):.4f}')
if pre_scores and post_scores:
    drift = sum(pre_scores)/len(pre_scores) - sum(post_scores)/len(post_scores)
    print(f'Temporal drift (pre vs post 2010): {drift:+.4f}')
    print('Interpretation: positive = model favors older content (trained on pre-2010 patterns)')
"

echo ""
echo "============================================================"
echo "IMPACT ANALYSIS COMPLETE"
echo "Add these numbers to your portfolio writeup."
echo "============================================================"
