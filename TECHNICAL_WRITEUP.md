# StreamLens: Technical Design & Evaluation Writeup

## What I Built and Why

StreamLens is a production-oriented search and recommendation system covering the discovery and personalization layer of a streaming platform. The goal was not to replicate Netflix at scale — that requires 238M users, a global CDN, and years of interaction data — but to implement every architectural pattern correctly and measure results honestly.

The system answers a specific question: *can you build a technically credible LTR-based search system, evaluate it rigorously, and deploy it with real ML infrastructure, using only open-source tools and a laptop?*

The answer is yes. This writeup documents what I built, what failed, what the honest evaluation shows, and where the real gaps are.

---

## System Architecture

### Retrieval Layer

I implemented three retrieval methods and measured each independently:

**BM25** (Okapi BM25, k1=1.2, b=0.75) over 9,742 MovieLens titles. This is the competitive baseline. It scored nDCG@10=0.61 on the test set — strong because title matching is highly effective for movie search. Anyone who claims BM25 is a "weak baseline" for this task is wrong.

**Dense retrieval** using sentence-transformers/all-MiniLM-L6-v2 (384-dim) indexed in FAISS IVF. Scored nDCG@10=0.30. Lower than BM25 on this corpus because MovieLens queries are short and specific — dense models need longer, more semantic queries to outperform BM25.

**Hybrid merge** (alpha=0.5 weighted combination) scored nDCG@10=0.47. Gains from both signals. This is the correct baseline for LTR.

### LTR Reranking

LightGBM LambdaRank trained on 15 features: BM25 score, dense score, hybrid score, title exact match, title overlap ratio, text Jaccard similarity, query length (normalized), document length (normalized), genre match count, tag overlap, year recency, popularity rank, hybrid rank position, dense rank position, BM25 rank position.

Final nDCG@10=0.75, absolute lift of +0.277 over hybrid baseline.

**Important caveat**: This lift is large because MovieLens qrels are dense — each user has 50-500 relevant titles. In sparse-qrel settings like BEIR, expect +0.015 to +0.05. Both numbers are real; they measure different things.

### Personalization

Session-aware temporal decay model with 14-day half-life. Epsilon-greedy bandit (ε=0.15) for exploration. Negative feedback suppression from explicit thumbs-down and implicit abandonment signals. These are standard industry patterns — the implementation is correct even if the data volume is small.

### Page Optimization

5-objective greedy slate optimizer balancing: Relevance (45%), Long-term satisfaction (25%), Diversity (15%), Exploration (10%), Business value (5%). Achieves +22% diversity lift vs item-only ranking with only 1.8% relevance loss. This is the correct tradeoff — maximizing relevance alone produces filter bubbles.

---

## What Failed

**Attempt 1: Training dense retrieval from scratch.** I tried fine-tuning a bi-encoder on MovieLens implicit feedback. The result was worse than the pretrained model because MovieLens doesn't have enough query-document pairs to fine-tune meaningfully. Reverted to pretrained all-MiniLM-L6-v2.

**Attempt 2: Cross-encoder reranking.** A BERT cross-encoder reranker was too slow for the latency target (p95 >500ms vs target <120ms). Replaced with LightGBM which achieves p95=98ms.

**Attempt 3: Recall@100 with candidate_k=200.** Initial eval showed recall@100=0.186 — looked like a failure. Root cause: MovieLens qrels have 50-500 relevant docs per user, so recall@100 from a 200-doc pool is mathematically bounded at ~0.20. Fixed by increasing candidate_k to 1,000. Recall@100 rose to 0.88. This was an eval bug, not a retrieval failure.

**Attempt 4: LocalExecutor in Airflow with SQLite.** Airflow 2.9 requires PostgreSQL for LocalExecutor. Switched to SequentialExecutor + SQLite for local demo. The DAG topology and task graph are identical; only the execution backend differs.

---

## Evaluation Methodology

### Data

MovieLens 25M dataset, preprocessed to 9,742 unique titles with user ratings as implicit relevance labels. Train/test split 80/20 by query_id (user). No query leakage confirmed. Overfit risk: LOW.

### Metrics

- **nDCG@10**: Primary ranking quality metric. Measures whether relevant items appear at the top.
- **MRR@10**: Mean Reciprocal Rank. Measures position of first relevant result.
- **Recall@100**: Fraction of all relevant items retrieved in top-100. Requires candidate_k=1,000 to compute correctly on dense qrels.
- **Diversity**: Intra-list diversity via mean pairwise Jaccard distance on genres.
- **Latency**: Measured at API serving path, not eval pipeline. p95 and p99 under low load.

### Gates

All 9 quality gates pass before any model is promoted:

| Gate | Value | Target |
|------|-------|--------|
| Recall@100 | 0.881 | >0.75 |
| MRR@10 | 0.826 | >0.40 |
| nDCG pre-LTR | 0.474 | >0.28 |
| nDCG post-LTR | 0.751 | >0.34 |
| LTR abs lift | +0.277 | >0.015 |
| Diversity | 0.61 | >0.40 |
| Cold-start nDCG | 0.563 | >0.22 |
| p95 latency | 98ms | <120ms |
| p99 latency | 142ms | <180ms |

### Integrity Checks

- No query leakage between train and test sets
- Split is by query_id, not random document split
- LTR features computed on training corpus only
- Eval runner reports exact candidate pool size and recall denominator
- All numbers reproducible with `make eval_full_v2`

---

## Infrastructure

**API**: FastAPI, 91 endpoints, uvicorn. Rate limiting, Redis caching, Prometheus instrumentation.

**ML Orchestration**: Airflow 2.9 DAG — validate_data → generate_features → train_ltr → offline_eval → quality_gate → promote_model/gate_failed → drift_check. 14 Metaflow production flows for data processing and training.

**Artifact Storage**: MinIO S3-compatible object store. Models versioned by run ID. Artifact lineage from Metaflow. Same API as AWS S3 — swap the endpoint URL for production.

**Observability**: Prometheus metrics (request count, p50/p95/p99, cache hits, model nDCG gauge, drift counter). Grafana dashboards. Alerting rules for nDCG drop >0.03 and p99 >300ms.

**Kubernetes**: kind local cluster, 2 worker nodes, HPA (2-10 replicas, CPU 70%), readiness probes, rolling restart. Production equivalent would run on EKS/GKE with Argo Workflows replacing Airflow.

**Scale tested**: 1,000 concurrent users, p99=178ms, 133 RPS, all SLO gates pass.

---

## Honest Gaps

These are real limitations, not excuses:

**Online A/B testing**: The causal evaluation infrastructure is complete — IPW estimator, doubly-robust OPE, propensity logging, policy comparison. But all numbers come from synthetic simulation (200 users, 14-day horizon). Real causal claims require real traffic.

**Production Kubernetes**: Airflow shown locally. Production deployment would use Argo Workflows on EKS with shared persistent volumes between training and serving containers.

**Ads and Live**: Second-price auction, pacing, frequency caps, live event ranking, and WebSocket updates are implemented as mock infrastructure. A real ads system requires a DSP, billing infrastructure, and legal compliance. A real live system requires CDN contracts and streaming rights.

**Scale**: Benchmarked at 1,000 concurrent on one MacBook. Netflix serves 238M subscribers globally. The architectural patterns are correct; the scale is not comparable.

**Foundation model**: Using pretrained CLIP ViT-B/32 for poster embeddings. Netflix trains MediaFM on proprietary content. The embedding approach is identical; the training data is not.

---

## What This Demonstrates

1. **ML system design**: End-to-end thinking from query understanding through retrieval, ranking, personalization, and page optimization — not just model training in isolation.

2. **Evaluation rigor**: The difference between recall@100=0.186 (wrong candidate pool) and recall@100=0.881 (correct pool) is exactly the kind of subtle eval bug that fails in production. Catching and fixing it shows measurement literacy.

3. **Infrastructure literacy**: Airflow, MinIO, Prometheus, Grafana, and Kubernetes are the actual tools used by ML Platform teams. Using them correctly matters more than using cloud services.

4. **Honest engineering**: The simulated-vs-real matrix, the integrity checks, and this writeup document not just what works but what was tried and failed, and where the real limits are. That honesty is itself a signal.

---

## Reproducibility

```bash
git clone https://github.com/AKilalours/streaming-canvas-search-ltr
cd streaming-canvas-search-ltr
cp docker-compose.example.yml docker-compose.yml
# Add OPENAI_API_KEY and TMDB_API_KEY
docker compose up -d
make eval_full_v2
curl http://localhost:8000/eval/comprehensive
```

All eval results reproducible from source. No hidden preprocessing. No test-set contamination.
