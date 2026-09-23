# StreamLens MLOps Reference

**Akila Lourdes Miriyala Francis · MS in Artificial Intelligence**

Complete MLOps documentation for the StreamLens production ML pipeline — training, evaluation, deployment, monitoring, and rollback.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Airflow DAG](#airflow-dag)
3. [Metaflow Flows](#metaflow-flows)
4. [Quality Gates](#quality-gates)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation Pipeline](#evaluation-pipeline)
7. [Deployment Strategy](#deployment-strategy)
8. [Monitoring & Observability](#monitoring--observability)
9. [Drift Detection](#drift-detection)
10. [Rollback Strategy](#rollback-strategy)
11. [Feature Store](#feature-store)
12. [Infrastructure](#infrastructure)
13. [Cost Management](#cost-management)
14. [Experiment Tracking](#experiment-tracking)

---

## Pipeline Overview

```
                         STREAMLENS ML PIPELINE
                         ─────────────────────

  Data Ingestion          Training              Evaluation
  ──────────────         ──────────            ────────────
  MovieLens CSV           BM25 Build            nDCG@10
  PySpark ETL    ──►      Dense Embed   ──►     BEIR NFCorpus
  Feature Eng             LTR Train             9 Measured Gates
  Redis Store             Fine-tune e5          Latency SLO

                              │
                              ▼ gate pass only
                         Deployment
                         ──────────
                         MinIO artifact push
                         Docker image tag
                         K8s rolling update
                         Shadow mode 24h
                              │
                              ▼
                         Monitoring
                         ──────────
                         Prometheus metrics
                         Grafana dashboards
                         Drift alerts
                         Auto-rollback
```

---

## Airflow DAG

File: `flows/streamlens_airflow_dag.py`. Each task runs `python -m pipelines.promotion <step>`
(`src/pipelines/promotion.py`); no step is simulated.

```
validate_data -> train_candidate -> evaluate -+-> run_gates -> decide -+-> promote_model
                                              |                        +-> block_promotion (fails the run)
                                              +-> drift_check
```

| Task | What it does |
|------|--------------|
| `validate_data` | 11 checks on corpus/queries/qrels: files, schema, unique ids, qrels reference known docs and queries, same corpus in every split, query-text overlap with train (`src/pipelines/data_quality.py`) |
| `train_candidate` | LightGBM LambdaRank (`ranking/ltr_train.py`) on the **train** split, written to `artifacts/ltr/candidates/<run_id>/`. Production model untouched |
| `evaluate` | Candidate on val and test, current production model on val, same code and split, strict model path (no silent fallback) |
| `run_gates` | 9 gates below, from this run's measured metrics, on **val** (test is reported only) |
| `promote_model` | Archive current model, atomic rename, sha256 check, append to `artifacts/ltr/registry.jsonl`. Idempotent |
| `block_promotion` | Fails the DAG run with the failing gate names, production model unchanged |
| `drift_check` | bm25/dense/hybrid scores vs previous run, and the same production model vs its last evaluation; fails if they move > 0.005 |

Gates (`src/pipelines/promotion_gates.py`, thresholds in `configs/promotion.yaml`):

| # | Gate | Rule |
|---|------|------|
| 1 | data_validation_passed | all 11 data checks pass |
| 2 | candidate_model_was_evaluated | eval loaded exactly the candidate pickle |
| 3 | feature_schema_matches_serving | candidate features == `ranking.features.FEATURE_NAMES` (names, order, count) |
| 4 | full_query_coverage | every val query scored |
| 5 | ndcg10_above_floor | val nDCG@10 >= 0.70 |
| 6 | beats_best_first_stage | LTR beats best of bm25/dense/hybrid by > 0.05 |
| 7 | ndcg10_no_regression | drop <= 0.01 vs max(production model, best promoted) |
| 8 | recall100_no_regression | drop <= 0.01 vs production model |
| 9 | map10_no_regression | drop <= 0.01 vs production model |

Evidence: three `airflow dags test` runs (Airflow 2.10.5) are committed in
`reports/pipeline_evidence/2026-09-23/`: two runs promoted (9/9), one run with a stricter demo
threshold was blocked (8/9) and failed, leaving production unchanged.
Retrained candidates scored val nDCG@10 0.9465 to 0.9481 and test 0.9497 to 0.9517.

### Running

```bash
# one-off local run of the whole DAG (no scheduler needed)
export STREAMLENS_HOME=$PWD STREAMLENS_PYTHON=$PWD/.venv/bin/python
airflow dags test streamlens_ml_pipeline 2026-09-23T02:00:00+00:00

# or one step at a time, without Airflow
PYTHONPATH=src python -m pipelines.promotion validate --run-id r1   # then train, evaluate, gate, promote, drift
```

Schedule: daily 02:00 UTC, `max_active_runs=1`, `catchup=False`. The docker-compose service
starts the DAG paused.

---

## Metaflow Flows

14 production Metaflow flows covering the complete ML lifecycle.

| Flow | File | Steps | Purpose |
|------|------|-------|---------|
| `StreamLensTrainFlow` | `flows/train_flow.py` | 15 | Full end-to-end training |
| `MultimodalPipelineFlow` | `flows/multimodal_pipeline_flow.py` | 7 | CLIP + VLM features |
| `EvalFlow` | `flows/eval_flow.py` | 6 | Metrics + gate validation |
| `DriftMonitorFlow` | `flows/drift_monitor_flow.py` | 4 | Temporal drift detection |
| `CausalValidationFlow` | `flows/causal_validation_flow.py` | 5 | IPW + OPE |
| `BM25BuildFlow` | `flows/bm25_build_flow.py` | 3 | BM25 index build |
| `DenseEmbedFlow` | `flows/dense_embed_flow.py` | 4 | FAISS index build |
| `FineTuneFlow` | `flows/fine_tune_flow.py` | 5 | e5-base-v2 fine-tuning |
| `LTRTrainFlow` | `flows/ltr_train_flow.py` | 6 | LightGBM LambdaRank |
| `SparkFeatureFlow` | `flows/spark_feature_flow.py` | 4 | PySpark feature engineering |
| `ShadowEvalFlow` | `flows/shadow_eval_flow.py` | 3 | Shadow A/B comparison |
| `ArtifactPromoteFlow` | `flows/promote_flow.py` | 2 | MinIO artifact promotion |
| `LoadTestFlow` | `flows/load_test_flow.py` | 3 | Locust benchmark |
| `ColdStartFlow` | `flows/cold_start_flow.py` | 4 | New item bootstrapping |

### Running Flows

```bash
# Full training pipeline
python flows/train_flow.py run

# Evaluation only
python flows/eval_flow.py run --candidate_k 2000

# Drift monitor
python flows/drift_monitor_flow.py run

# View artifact lineage
python flows/train_flow.py dump --include-artifacts
```

### Artifact Storage (MinIO)

```
s3://artifacts/
  ├── bm25/
  │   └── movielens_bm25.pkl          ← BM25 index
  ├── faiss/
  │   └── movielens_ft_e5/            ← FAISS + model
  ├── ltr/
  │   ├── movielens_ltr.pkl           ← base LTR
  │   └── movielens_ltr_tuned.pkl     ← tuned LTR (active)
  ├── multimodal/
  │   └── lineage.json                ← CLIP pipeline lineage
  └── reports/
      ├── latest/metrics.json         ← current eval results
      ├── latest/latency.json         ← latency benchmarks
      └── reference/                  ← rollback baseline
```

---

## Quality Gates

See [Airflow DAG](#airflow-dag): 9 gates computed per run from measured metrics, thresholds in
`configs/promotion.yaml`. On failure the `block_promotion` task fails the run, the gate names
and values are in `reports/runs/<run_id>/gates.json`, and the production model is not touched.
Not yet wired: Slack/on-call alerting (use Airflow `on_failure_callback`).

---

## Training Pipeline

### 1. Data Preparation

```bash
# PySpark feature engineering
python spark/feature_engineering.py \
  --input data/raw/movielens \
  --output data/processed/movielens \
  --co_watch_min 3

# Generates:
# - data/processed/movielens/train/corpus.jsonl  (9,742 items)
# - data/processed/movielens/train/qrels.json    (150 queries)
# - data/processed/movielens/spark_features.parquet
```

### 2. BM25 Index

```bash
python scripts/build_bm25.py \
  --corpus data/processed/movielens/test/corpus.jsonl \
  --output artifacts/bm25/movielens_bm25.pkl \
  --k1 1.2 --b 0.75
```

### 3. Dense Embeddings + Fine-tuning

```bash
# Fine-tune e5-base-v2 on MovieLens domain data
python fine_tune_retrieval.py \
  --model intfloat/e5-base-v2 \
  --epochs 2 \
  --output artifacts/faiss/movielens_ft_e5

# Results: Spearman 0.6809 → 0.8066 (+18.4% dense nDCG)
```

### 4. LTR Training

```bash
# Build 15-feature training set
python scripts/build_ltr_features.py \
  --corpus data/processed/movielens/test/corpus.jsonl \
  --qrels data/processed/movielens/test/qrels.json \
  --output data/ltr_features.jsonl

# Train LightGBM LambdaRank
python scripts/train_ltr.py \
  --features data/ltr_features.jsonl \
  --output artifacts/ltr/movielens_ltr_tuned.pkl \
  --trees 500 --leaves 31
```

### 5. LTR Features

| # | Feature | Type | Importance |
|---|---------|------|-----------|
| 1 | BM25 score | Retrieval | ~25% |
| 2 | Dense score | Retrieval | ~30% |
| 3 | Hybrid score | Retrieval | ~20% |
| 4 | Title overlap | Text | ~10% |
| 5 | Text Jaccard | Text | ~8% |
| 6 | Length ratio | Text | ~3% |
| 7 | Coverage | Text | ~2% |
| 8 | Genre match | Content | ~5% |
| 9 | Tag overlap | Content | ~4% |
| 10 | Recency | Content | ~3% |
| 11 | Popularity | Content | ~4% |
| 12 | User watch_count | Spark | ~6% |
| 13 | Taste breadth | Spark | ~4% |
| 14 | Co-watch score | Spark | ~8% |
| 15 | Item popularity | Spark | ~5% |

---

## Evaluation Pipeline

```bash
# Full evaluation — reproduces all published metrics
make eval_full_v2

# Individual evaluations
python eval/evaluate_retrieval.py --method bm25
python eval/evaluate_retrieval.py --method dense
python eval/evaluate_retrieval.py --method hybrid
python eval/evaluate_retrieval.py --method hybrid_ltr

# BEIR benchmark
python eval/beir_eval.py --dataset nfcorpus --max_queries 323

# Latency benchmark
python eval/latency_benchmark.py --concurrency 1000

# Comprehensive (all gates)
curl http://localhost:8000/eval/comprehensive | python3 -m json.tool
```

### Eval Split

- **80/20 split by query_id** (not by document — prevents leakage through co-watch pairs)
- 150 MovieLens queries, 323 BEIR NFCorpus queries
- BM25 and dense scores computed fresh on test set only

---

## Deployment Strategy

### Zero-Downtime Rolling Update

```bash
# Tag new artifact
ARTIFACT_VERSION=$(date +%Y%m%d_%H%M%S)
mc cp artifacts/ltr/movielens_ltr_tuned.pkl \
   local/artifacts/ltr/${ARTIFACT_VERSION}.pkl

# Update active symlink
mc cp local/artifacts/ltr/${ARTIFACT_VERSION}.pkl \
   local/artifacts/ltr/active.pkl

# Rolling restart (K8s — 0 downtime)
kubectl rollout restart deployment/streamlens-api

# Verify
kubectl rollout status deployment/streamlens-api
curl http://localhost:8000/health | jq .ltr_loaded
```

### Shadow Mode (24h before promotion)

New model runs parallel to production for 24h. Users get production results. Both models score every query. Shadow model must beat production by ≥2% nDCG to promote.

```bash
# Check shadow status
curl http://localhost:8000/agent/shadow_status | python3 -m json.tool

# Shadow report
curl http://localhost:8000/shadow/report | python3 -m json.tool
```

### Kubernetes HPA

```yaml
minReplicas: 2
maxReplicas: 10
targetCPUUtilizationPercentage: 70
# Scales up at 1,000 concurrent → stays within 178ms p99
```

---

## Monitoring & Observability

### Prometheus Metrics

```
# Key metrics exposed at /metrics
streamlens_request_total{path, method, status}
streamlens_request_latency_ms{path, method}
streamlens_cache_hits_total{kind}
streamlens_cache_misses_total{kind}
streamlens_ltr_ndcg_latest
streamlens_openai_calls_total
streamlens_openai_errors_total{error_type}
```

### Grafana Dashboards

Access: http://localhost:3000 (admin / searchltr2026)

| Dashboard | Key Panels |
|-----------|-----------|
| Search Quality | nDCG@10, MRR, Recall@100 |
| Latency | p50/p95/p99 by method |
| Cache | Hit rate, Redis memory |
| GenAI | OpenAI call latency, error rate, cost |
| Infrastructure | CPU, memory, replica count |

### Alerts

| Alert | Threshold | Action |
|-------|-----------|--------|
| nDCG drops >5% | vs last 24h | Page on-call |
| p99 > 200ms | 5-min window | Scale up |
| Cache hit rate < 60% | 10-min window | Investigate |
| OpenAI error rate > 10% | 5-min window | Switch to fallback |
| Redis memory > 80% | — | Evict + alert |

---

## Drift Detection

### Temporal Drift (24.6% measured gap)

Pre-2010 content scores 24.6% lower than post-2010 — caused by sparse metadata in older films.

```bash
# Run drift detection
python scripts/drift_monitor.py \
  --corpus data/processed/movielens/test/corpus.jsonl \
  --threshold 0.05

# Check drift report
curl http://localhost:8000/reports/drift | python3 -m json.tool
```

### Drift Report Fields

```json
{
  "temporal_drift_pct": 24.6,
  "pre_2010_ndcg": 0.71,
  "post_2010_ndcg": 0.95,
  "genre_drift": {"Animation": +0.02, "Documentary": -0.08},
  "alert": "Pre-2010 content underperforming — LLM enrichment recommended"
}
```

### Continuous Drift Monitor (Airflow)

`drift_check` runs in every DAG run. It fails the run when first-stage retrieval scores
(bm25/dense/hybrid) or the unchanged production model's score move more than 0.005 nDCG@10
from the previous run, and records whether the data fingerprint changed.

---

## Rollback Strategy

### Automatic Rollback

```bash
# Admin endpoint — restores reference baseline
curl -X POST http://localhost:8000/admin/rollback \
  -H "X-Admin-Token: ${ADMIN_TOKEN}"

# Returns: {"ok": true, "rolled_back_files": ["metrics.json", "latency.json"]}
```

### Manual Rollback

```bash
# List available artifact versions
mc ls local/artifacts/ltr/

# Restore specific version
mc cp local/artifacts/ltr/20260401_120000.pkl \
   local/artifacts/ltr/movielens_ltr_tuned.pkl

# Restart API (picks up restored artifact)
OPENAI_API_KEY=$(grep "^OPENAI_API_KEY" .env | head -1 | cut -d= -f2-) \
  docker compose up -d --force-recreate api
```

### Reference Baseline

`reports/reference/` always contains the last known-good metrics. Updated manually after each successful production deployment.

---

## Feature Store

### Architecture

```
Offline (Airflow nightly)                Online (API serving)
─────────────────────                    ──────────────────
PySpark computes:                        Redis lookup:
  - user_watch_count                       <1ms per feature
  - user_taste_breadth                     Pre-joined at read time
  - item_co_watch_score                    No database join at serve time
  - item_popularity
       │
       ▼
  Redis feature store
  (key: user:{id}:features)
  (key: item:{id}:features)
```

### Accessing Feature Store

```bash
# User features
curl http://localhost:8000/feature_store/user/u1

# Stats
curl http://localhost:8000/feature_store/stats
```

---

## Infrastructure

### Docker Services

```bash
docker compose up -d          # All services
docker compose up -d api      # API only

# Services:
# api          :8000  FastAPI + ML models
# redis        :6379  Cache + feature store
# minio        :9001  Artifact storage
# prometheus   :9090  Metrics
# grafana      :3000  Dashboards
# airflow      :8080  Pipeline orchestration
# kafka        :9092  Event streaming
# zookeeper    :2181  Kafka coordination
```

### Kubernetes (Local kind cluster)

```bash
# Apply manifests
kubectl apply -f k8s/

# Check HPA
kubectl get hpa streamlens-api

# Rolling restart
kubectl rollout restart deployment/streamlens-api

# Scale manually
kubectl scale deployment/streamlens-api --replicas=5
```

---

## Cost Management

### OpenAI Cost Breakdown

| Feature | Model | Cost/call | Daily calls | Daily cost |
|---------|-------|-----------|-------------|------------|
| Why This | gpt-4o-mini | $0.0003 | ~500 | $0.15 |
| RAG | gpt-4o-mini | $0.0005 | ~200 | $0.10 |
| VLM poster | gpt-4o | $0.003 | ~50 | $0.15 |
| TTS | tts-1 | $0.015/1K chars | ~100 | $0.05 |
| **Total** | | | | **~$0.45/day** |

### Redis Cache Savings

7-day explanation cache + 30-day VLM cache reduces OpenAI calls by ~85% after warm-up.

```
Without cache: 1,000 explain calls/day × $0.0003 = $0.30/day
With cache:    150 unique films × $0.0003 = $0.045/day  (85% saving)
```

### FinOps Gate

```bash
# Check if nDCG lift justifies cost
curl "http://localhost:8000/finops/cost_gate?ndcg_lift=0.02"

# Artifact tiering (archive old models to cold storage)
curl -X POST "http://localhost:8000/finops/artifact_tier?dry_run=true"
```

---

## Experiment Tracking

### Metaflow Lineage

Every training run creates an artifact record with full lineage.

```python
from metaflow import Flow, Run

# List all runs
for run in Flow("StreamLensTrainFlow"):
    print(run.id, run.data.ndcg_at_10)

# Access specific artifact
run = Run("StreamLensTrainFlow/latest")
model = run.data.ltr_model
metrics = run.data.eval_metrics
```

### A/B Experiment Tracking

```bash
# Statistical significance test
curl "http://localhost:8000/causal/ab_test?n_samples=500"

# Shadow comparison report
curl "http://localhost:8000/shadow/report"

# Off-policy evaluation
curl "http://localhost:8000/uplift/ope?q=action+movies&k=20"
```

### Current Experiment Status

| Experiment | Control | Treatment | p-value | Status |
|------------|---------|-----------|---------|--------|
| hybrid vs hybrid_ltr | 0.5891 | 0.9300 | 0.065 | ⚠️ Underpowered |
| base vs fine-tuned e5 | 0.4640 | 0.5496 | < 0.01 | ✅ Significant |
| α=0.5 vs α=0.2 | 0.5641 | 0.5891 | < 0.05 | ✅ Significant |

---

<div align="center">

**Akila Lourdes Miriyala Francis · MS in Artificial Intelligence**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/akila-lourdes-miriyala-francis-5b047019a/)
[![GitHub](https://img.shields.io/badge/GitHub-View%20Project-181717?style=flat&logo=github)](https://github.com/AKilalours/streaming-canvas-search-ltr)

</div>
