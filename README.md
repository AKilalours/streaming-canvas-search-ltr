# streaming-canvas-search-ltr

A search relevance workbench (Netflix-aligned pattern):
- **Dataset**: BEIR SciFact (queries + qrels)
- **Retrieval**: BM25 baseline (working now), FAISS embeddings (next), Hybrid (next)
- **Rerank**: LightGBM Learning-to-Rank (next)
- **Eval**: nDCG@10, MAP@10, Recall@10
- **Service**: FastAPI `/search` with timing breakdown
- **Repro**: `reports/<run_id>/` artifacts + metrics snapshot
- **Gates**: metric/latency regression thresholds

## Quickstart

### 1) Setup
```bash
make setup

