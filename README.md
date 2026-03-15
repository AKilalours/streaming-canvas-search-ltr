# StreamLens — Netflix-Standard LTR Search & Recommendation

Production-oriented streaming discovery and personalization platform.

## Architecture
- **Retrieval**: BM25 + FAISS dense + hybrid merge
- **Ranking**: LightGBM LambdaRank (15 features, nDCG@10: 0.75)
- **Personalization**: Temporal decay + contextual bandits
- **Page optimization**: 5-objective slate optimizer
- **Multimodal**: CLIP ViT-B/32 pretrained embeddings
- **Causal**: Doubly-robust IPW OPE, 200-user simulation
- **Infra**: Airflow DAG, MinIO S3, Prometheus+Grafana, kind K8s

## Eval Results (All Gates Passing)
| Metric | Value | Target |
|--------|-------|--------|
| nDCG@10 (LTR) | 0.7506 | >0.34 |
| MRR@10 | 0.8256 | >0.40 |
| Recall@100 | 0.881 | >0.75 |
| p95 latency | 98ms | <120ms |
| p99 latency | 142ms | <180ms |

## Stack
FastAPI · LightGBM · FAISS · Redis · MinIO · Airflow · Prometheus · Grafana · Docker · Kubernetes

## Run locally
```bash
docker compose up -d
open http://localhost:8000/demo
```
