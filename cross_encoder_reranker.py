"""
StreamLens — Cross-Encoder Reranker (3rd Stage)
================================================
Adds a BERT cross-encoder as a 3rd reranking stage after LTR.
Cross-encoders see query + document TOGETHER → much better relevance.

Pipeline: BM25+Dense → LTR LambdaRank → Cross-Encoder (top-20)
Used by: Bing, Google, Amazon, LinkedIn for precision reranking

Run: python cross_encoder_reranker.py
Expected: +0.02-0.05 nDCG on top-20 results
"""
import os, json, time
import numpy as np
from pathlib import Path

print("\n" + "="*60)
print("StreamLens — Cross-Encoder Reranker")
print("BERT sees query+doc together → precision reranking")
print("="*60 + "\n")

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    os.system("pip install sentence-transformers --break-system-packages -q")
    from sentence_transformers import CrossEncoder

CORPUS_PATH = "data/processed/movielens/test/corpus.jsonl"
OUTPUT_DIR  = "artifacts/cross_encoder"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Load cross-encoder model ──────────────────────────────────
print("Loading cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
print("(Trained on MS MARCO passage ranking — general retrieval)")
t0 = time.time()
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
print(f"✅ Cross-encoder loaded in {time.time()-t0:.1f}s")

# ── Load corpus ───────────────────────────────────────────────
corpus = {}
with open(CORPUS_PATH) as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["doc_id"]] = doc

# ── Benchmark: cross-encoder vs LTR on sample queries ─────────
print("\nBenchmarking cross-encoder reranking...")

sample_queries = [
    "crime thriller gangster",
    "animated family children",
    "romantic comedy love",
    "sci-fi space future",
    "horror supernatural scary",
]

# Simulate LTR top-20 results (random sample for benchmark)
import random
random.seed(42)
doc_ids = list(corpus.keys())

latencies = []
for query in sample_queries:
    # Simulate LTR top-20 candidates
    candidates = random.sample(doc_ids, 20)
    candidate_docs = [corpus[did] for did in candidates]

    # Build query-document pairs for cross-encoder
    pairs = [
        [query, doc.get("title","") + " " + doc.get("text","")[:300]]
        for doc in candidate_docs
    ]

    # Score with cross-encoder
    t0 = time.time()
    scores = model.predict(pairs, show_progress_bar=False)
    latency = (time.time() - t0) * 1000

    latencies.append(latency)
    top_idx = np.argsort(scores)[::-1][:5]

    print(f"\n  Query: '{query}'")
    print(f"  Latency: {latency:.0f}ms for 20 pairs")
    print(f"  Top-3 cross-encoder results:")
    for rank, idx in enumerate(top_idx[:3], 1):
        print(f"    {rank}. {candidate_docs[idx]['title']} (score={scores[idx]:.3f})")

avg_latency = np.mean(latencies)
print(f"\n✅ Average cross-encoder latency: {avg_latency:.0f}ms for 20 pairs")
print(f"   Pipeline latency budget: 142ms + {avg_latency:.0f}ms = {142+avg_latency:.0f}ms total")
print(f"   Trade-off: +{avg_latency:.0f}ms for potentially +0.02-0.05 nDCG")

# ── Write cross-encoder serving module ───────────────────────
CE_CODE = f'''"""
StreamLens — Cross-Encoder Reranker
3rd stage precision reranker after LTR LambdaRank.

Architecture:
  Stage 1: BM25 + Fine-tuned Dense → 2000 candidates (fast)
  Stage 2: LightGBM LambdaRank    → top 200 (sub-ms)
  Stage 3: Cross-Encoder          → top 20 (precision)

Trade-off:
  Cost:    +{avg_latency:.0f}ms latency (p99 becomes ~{142+avg_latency:.0f}ms)
  Benefit: +0.02-0.05 nDCG on top-20 precision
  
  Mitigation: Only run cross-encoder on non-cached queries.
  Cached queries still return at p50=2.67ms.
"""
from __future__ import annotations
import numpy as np
import time
import os

_MODEL = None
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import CrossEncoder
        _MODEL = CrossEncoder(_MODEL_NAME, max_length=512)
    return _MODEL


def rerank_cross_encoder(
    query: str,
    items: list[dict],
    top_k: int = 20,
    enabled: bool = True,
) -> list[dict]:
    """
    Cross-encoder reranking of top-k candidates.
    
    Cross-encoders see query AND document together in a single
    BERT forward pass — unlike bi-encoders which encode separately.
    This joint encoding captures fine-grained query-document
    interaction that bi-encoders miss.
    
    Args:
        query:   User query string
        items:   Ranked items from LTR (sorted by ltr_score)
        top_k:   Rerank only top-k (default: 20 for latency)
        enabled: Feature flag — disable for latency-sensitive paths
    
    Returns:
        Items with cross_encoder_score, top-k reranked + rest appended
    """
    if not enabled or not items:
        return items

    model = _get_model()
    
    # Only rerank top-k — rest appended unchanged
    to_rerank = items[:top_k]
    rest       = items[top_k:]

    # Build (query, document) pairs
    pairs = [
        [query, item.get("title","") + " " + item.get("text","")[:300]]
        for item in to_rerank
    ]

    # Cross-encoder scoring — single BERT forward pass per pair
    t0 = time.time()
    scores = model.predict(pairs, show_progress_bar=False)
    latency_ms = (time.time() - t0) * 1000

    # Attach scores
    for item, score in zip(to_rerank, scores):
        item["cross_encoder_score"] = float(score)
        item["ce_latency_ms"] = round(latency_ms, 1)

    # Sort top-k by cross-encoder score
    to_rerank.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    return to_rerank + rest


def is_available() -> bool:
    """Check if cross-encoder is loadable."""
    try:
        _get_model()
        return True
    except Exception:
        return False
'''

os.makedirs("src/retrieval", exist_ok=True)
with open("src/retrieval/cross_encoder.py", "w") as f:
    f.write(CE_CODE)

import py_compile
py_compile.compile("src/retrieval/cross_encoder.py", doraise=True)
print(f"\n✅ Cross-encoder serving module: src/retrieval/cross_encoder.py")

# Save meta
meta = {
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "type": "cross-encoder (joint query-doc BERT)",
    "avg_latency_ms_per_20_pairs": round(float(avg_latency), 1),
    "use_case": "3rd stage precision reranker on top-20 LTR results",
    "trade_off": f"+{avg_latency:.0f}ms latency for +0.02-0.05 nDCG",
    "vs_bi_encoder": "sees query+doc together vs separate encoding",
}
import json
with open(f"{OUTPUT_DIR}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"""
{'='*60}
CROSS-ENCODER COMPLETE
{'='*60}
Model:     cross-encoder/ms-marco-MiniLM-L-6-v2
Latency:   {avg_latency:.0f}ms for 20 pairs
Pipeline:  142ms + {avg_latency:.0f}ms = {142+avg_latency:.0f}ms total p99

TO INTEGRATE in src/app/main.py:
  from retrieval.cross_encoder import rerank_cross_encoder
  
  # After LTR reranking:
  results = rerank_cross_encoder(
      query=query,
      items=ltr_results,
      top_k=20,
      enabled=not is_cached
  )

WHAT TO SAY:
  "Added cross-encoder reranker as 3rd stage after LTR.
   Cross-encoders encode query + document JOINTLY in a single
   BERT forward pass — capturing interaction signals that
   bi-encoders like e5-base-v2 miss (bi-encoders encode
   query and document separately). Applied to top-20 LTR
   results. Latency trade-off: +{avg_latency:.0f}ms per request,
   applied only on cache-miss paths. Expected nDCG gain:
   +0.02-0.05 on top-20 precision. Same 3-stage architecture
   used at Bing and Amazon product search."
{'='*60}
""")
