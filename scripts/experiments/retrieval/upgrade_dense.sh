#!/bin/bash
set -e
cd ~/streaming-canvas-search-ltr

echo "=========================================="
echo "DENSE MODEL UPGRADE: e5-small-v2"
echo "Expected: Dense 0.30 → 0.38-0.45 (Strong)"
echo "=========================================="

echo ""
echo "[1/3] Re-embedding corpus with intfloat/e5-small-v2..."
docker compose exec api /app/.venv/bin/python3.11 -c "
import sys, json, pathlib, numpy as np
sys.path.insert(0, '/app/src')
from sentence_transformers import SentenceTransformer

MODEL = 'intfloat/e5-small-v2'
print(f'Loading {MODEL}...')
model = SentenceTransformer(MODEL)

corpus = []
with open('data/processed/movielens/test/corpus.jsonl') as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            corpus.append(d)

print(f'Embedding {len(corpus)} docs...')
# e5 models need prefix: passage: for docs, query: for queries
texts = ['passage: ' + str(d.get('title','')) + ' ' + str(d.get('text','')) for d in corpus]
doc_ids = [str(d['doc_id']) for d in corpus]

BATCH = 256
all_embs = []
for i in range(0, len(texts), BATCH):
    batch = texts[i:i+BATCH]
    embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    all_embs.append(embs.astype(np.float32))
    if (i//BATCH+1) % 5 == 0:
        print(f'  {min(i+BATCH, len(texts))}/{len(texts)} docs embedded')

embeddings = np.concatenate(all_embs, axis=0)
print(f'Embeddings shape: {embeddings.shape}')

# Save
slug = MODEL.replace('/','_').replace(':','_')
out_dir = pathlib.Path(f'artifacts/faiss/movielens_{slug}')
out_dir.mkdir(parents=True, exist_ok=True)
np.save(out_dir / 'embeddings.npy', embeddings)
json.dump(doc_ids, open(out_dir / 'doc_ids.json', 'w'))
json.dump({'model_name': MODEL, 'n_docs': len(doc_ids), 'dim': embeddings.shape[1]},
          open(out_dir / 'meta.json', 'w'), indent=2)
print(f'Saved to {out_dir}')
print(f'Model: {MODEL}, dim={embeddings.shape[1]}')
"

echo ""
echo "[2/3] Running eval with e5-small-v2..."
# Update eval config to use new model
python3 -c "
import re
content = open('configs/eval_movielens.yaml').read()
# Update dense and hybrid methods to use new model
content = content.replace(
    'model_name: sentence-transformers/all-MiniLM-L6-v2',
    'model_name: intfloat/e5-small-v2'
)
content = content.replace(
    'emb_dir: artifacts/faiss/movielens_sentence-transformers_all-MiniLM-L6-v2',
    'emb_dir: artifacts/faiss/movielens_intfloat_e5-small-v2'
)
content = content.replace('movielens_ltr_v2.pkl', 'movielens_ltr_v2.pkl')
open('configs/eval_movielens.yaml', 'w').write(content)
print('Config updated to e5-small-v2')
"

make eval_full_v2

echo ""
echo "[3/3] RESULTS:"
python3 -c "
import json
m = json.load(open('reports/latest/metrics.json'))
print(f'Model: intfloat/e5-small-v2')
print(f'Queries={m["diagnostics"]["num_queries"]}')
print()
baselines = {'bm25':0.6065,'dense':0.3031,'hybrid':0.4753,'hybrid_ltr':0.7753}
strong    = {'bm25':0.60,  'dense':0.35,  'hybrid':0.55,  'hybrid_ltr':0.75}
phenomenal= {'bm25':0.65,  'dense':0.50,  'hybrid':0.65,  'hybrid_ltr':0.80}
for method in m['methods']:
    name=method['method']; ndcg=method['ndcg@10']
    base=baselines.get(name,0); s=strong.get(name,0); p=phenomenal.get(name,0)
    if ndcg >= p:   flag='✅ PHENOMENAL'
    elif ndcg >= s: flag='💪 STRONG'
    else:           flag=f'⚠️  below strong ({s})'
    delta = ndcg - base
    print(f'{name:15} nDCG@10={ndcg:.4f}  delta={delta:+.4f}  {flag}')
"

# Restore config
python3 -c "
content = open('configs/eval_movielens.yaml').read()
content = content.replace('intfloat/e5-small-v2','sentence-transformers/all-MiniLM-L6-v2')
content = content.replace('movielens_intfloat_e5-small-v2','movielens_sentence-transformers_all-MiniLM-L6-v2')
open('configs/eval_movielens.yaml','w').write(content)
print('Config restored')
"
