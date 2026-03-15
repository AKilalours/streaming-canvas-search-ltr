# StreamLens Model Card

## Model Details

| Field | Value |
|-------|-------|
| Model name | StreamLens LambdaRank LTR v1 |
| Model type | LightGBM LambdaRank (gradient-boosted trees, pairwise ranking) |
| Training date | 2026-01-19 |
| Version | movielens_ltr.pkl |
| Framework | LightGBM 4.x |
| Input | 15 handcrafted retrieval + query features |
| Output | Relevance score for reranking top-200 candidates |

---

## Intended Use

**Primary use case**: Reranking top-200 hybrid retrieval candidates for movie/TV title search queries.

**Intended users**: Search and recommendation engineers evaluating LTR architectures.

**Out-of-scope uses**:
- Production deployment without online A/B validation
- Domains other than entertainment content discovery
- Languages without sufficient training representation
- Real-time personalization without user history

---

## Training Data

| Field | Value |
|-------|-------|
| Dataset | MovieLens 25M (grouplens.org) |
| Corpus size | 9,742 unique titles |
| Query source | User rating histories treated as implicit relevance |
| Relevance labels | Binarized ratings (≥4.0 = relevant) |
| Train/test split | 80/20 by user_id (no leakage) |
| Training queries | ~800 users |
| Test queries | ~200 users |

**Known biases in training data**:
- MovieLens skews toward English-language Western titles
- Rating behavior reflects a specific user population (academic dataset)
- Sparse coverage of non-English, non-Hollywood content
- No coverage of TV series released after 2019 cutoff

---

## Evaluation Results

### Offline Metrics (test set, candidate_k=1000)

| Metric | Value | Notes |
|--------|-------|-------|
| nDCG@10 | 0.7506 | Dense qrels inflate this vs sparse-qrel benchmarks |
| MRR@10 | 0.8256 | Strong first-result placement |
| Recall@100 | 0.881 | Requires candidate_k=1000 to compute correctly |
| Diversity (ILD) | 0.61 | Mean pairwise Jaccard distance |
| p95 latency | 98ms | API serving path, low load |
| p99 latency | 142ms | API serving path, low load |

### Ablation

| System | nDCG@10 | Delta |
|--------|---------|-------|
| BM25 only | 0.6065 | baseline |
| Dense only | 0.3031 | -0.303 |
| Hybrid (BM25+Dense) | 0.4740 | -0.133 |
| Hybrid + LTR (this model) | 0.7506 | +0.277 |

### Slice Performance

| Slice | nDCG@10 |
|-------|---------|
| Short queries (1-2 words) | 0.713 |
| Long queries (5+ words) | 0.788 |
| Typo queries | 0.616 |
| Cold-start titles | 0.563 |
| Sparse users (<5 interactions) | 0.585 |
| Heavy users (50+ interactions) | 0.811 |
| Multilingual queries | 0.661 |

---

## Features

| Feature | Type | Description |
|---------|------|-------------|
| bm25_score | float | BM25 retrieval score |
| dense_score | float | Cosine similarity (FAISS) |
| hybrid_score | float | Weighted BM25+dense merge |
| title_exact_match | binary | Exact query-title match |
| title_overlap | float | Token overlap ratio |
| text_jaccard | float | Jaccard similarity on text |
| query_len_norm | float | Normalized query length |
| doc_len_norm | float | Normalized document length |
| genre_match | int | Number of matching genres |
| tag_overlap | float | Tag set overlap |
| year_recency | float | Decay by release year |
| popularity_rank | int | Corpus popularity rank |
| hybrid_rank | int | Position in hybrid list |
| dense_rank | int | Position in dense list |
| bm25_rank | int | Position in BM25 list |

---

## Limitations

**What this model does well**:
- Reranking when candidates contain many relevant documents (dense qrels)
- Title-centric queries with exact or near-exact matches
- English-language content with rich metadata

**What this model does poorly**:
- Cold-start titles with no interaction history (mitigated by CLIP embeddings)
- Queries in underrepresented languages
- Abstract mood queries ("something comforting") — dense retrieval handles these better
- Long-tail titles with sparse training signal

**Known failure modes**:
- Popularity bias: highly-rated popular titles score higher even for niche queries
- Recency bias: recent titles get a small boost that may not reflect user intent
- Genre label noise: MovieLens genre labels are user-contributed and inconsistent

---

## Ethical Considerations

**Filter bubble risk**: Relevance-only ranking concentrates results around popular, well-rated titles. Mitigated by the diversity slot injector (MMR, every 5th position) and the slate optimizer's 15% exploration weight.

**Representation**: The training data underrepresents non-English and non-Western content. A production system serving global audiences would require separate models or domain adaptation per market.

**Feedback loops**: Ranking models trained on click data can amplify existing popularity bias. The epsilon-greedy bandit and diversity injection partially counteract this offline; real mitigation requires online A/B validation.

**Data provenance**: MovieLens data is publicly available for research use. No personally identifiable information is used in training.

---

## Deployment Notes

**Promotion criteria**: Model is promoted only when all 9 quality gates pass (see eval/comprehensive endpoint).

**Drift monitoring**: Airflow drift_check task alerts when nDCG drops >0.03 vs reference baseline.

**Rollback**: Previous model artifact retained in MinIO. Rollback by updating `artifacts/ltr/movielens_ltr.pkl` symlink.

**Retraining trigger**: Drift alert OR scheduled weekly retraining via Airflow DAG.

---

## Citation

```
StreamLens LTR Search System
Trained on MovieLens 25M (Harper & Konstan, 2015)
https://github.com/AKilalours/streaming-canvas-search-ltr
```
