# StreamLens — Definitive Numbers

**Single source of truth.** Every figure here comes from a committed artifact, named
per row. If a number appears in `README.md`, `MODEL_CARD.md` or a resume and is not in
this file, it is not supported.

Last reconciled after fixing a recall-truncation defect in `src/eval/evaluate.py`
(see *Corrections* at the bottom).

---

## 1. Which dataset produced which number

Both MovieLens distributions are in this repo and they are **not interchangeable**:

| Path | Ratings | Used for |
|---|---|---|
| `data/raw/movielens/ml-latest/` | **33,832,162** | PySpark feature-engineering job only |
| `data/raw/movielens/ml-latest-small/` | **100,836** | The corpus, queries, qrels and **every retrieval metric below** |

The evaluated corpus is **9,742 titles, 610 users, 150 test queries**.

> **Do not write "nDCG@10 = 0.93 over 33.8M interactions".** The Spark job processes
> 33.8M ratings, but `load_spark_features()` is never called anywhere in the codebase,
> so that output does not reach the ranking path. The retrieval numbers are measured on
> ml-latest-small.

---

## 2. Retrieval quality — MovieLens

**Reference configuration** (`reports/reference/metrics.json`)
150 queries · 9,742 docs · candidate_k = 200 · `all-MiniLM-L6-v2`

| Method | nDCG@10 |
|---|---|
| BM25 | 0.6065 |
| Dense | 0.3031 |
| Hybrid | 0.4756 |
| **Hybrid + LTR** | **0.7506** |

**Latest configuration** (`reports/latest/metrics.json`)
150 queries · 9,742 docs · BM25 candidate_k = 2,000 · dense candidate_k = 200 ·
contrastively fine-tuned `e5-base-v2` · rerank_k = 200

| Method | nDCG@10 | recall@10 | recall@100 |
|---|---|---|---|
| BM25 | 0.6065 | 0.1618 | 0.4106 |
| Dense (fine-tuned e5) | 0.5428 | 0.1848 | 0.4164 |
| Hybrid (α = 0.2) | 0.5902 | 0.1645 | 0.4171 |
| **Hybrid + LTR** | **0.9491** | **0.2327** | **0.4340** |

**The defensible headline:** nDCG@10 improved from **0.7506 to 0.9491** by contrastively
fine-tuning the dense retriever and widening BM25 candidate generation from 200 to 2,000.

### How to read recall@100 here

The qrels are extremely dense: **mean 831 relevant titles per query, median 502**, out
of 9,742 documents. With 502 relevant items and a 100-document cut-off, the maximum
achievable recall@100 for the median query is **0.199**. So 0.4340 is close to
structurally capped and must be read as a relative lift over BM25, never as an absolute
quality score.

---

## 3. Cross-corpus validation — BEIR SciFact

**These runs are SciFact, not MovieLens.** The files
`reports/latest/metrics_test.json`, `reports/latest/metrics_scifact_test.json` and
`reports/latest_eval/ablations.csv` all report **5,183 documents and 300 queries**, which
is SciFact's shape. MovieLens is 9,742 documents and 150 queries.

| Method | nDCG@10 | recall@100 |
|---|---|---|
| BM25 | 0.6523 | 0.7757 |
| Dense | 0.6451 | 0.7833 |
| Hybrid | 0.7070 | 0.8270 |
| **Hybrid + LTR** | **0.7684** | **0.8576** |

Reporting both corpora is the point: the LTR lift is far larger on MovieLens (+0.199)
than on SciFact (+0.061) because MovieLens qrels are dense. Quoting only the MovieLens
lift overstates how well the reranker generalises.

**BEIR NFCorpus** (`reports/latest_eval/ltr_train_eval.json`, `shadow_ab.json`):
hybrid 0.2767 → hybrid+LTR 0.2943, a lift of **+0.0177**. That is the realistic
sparse-qrel range.

---

## 4. Latency

| Source | p50 | p95 | p99 | What it measures |
|---|---|---|---|---|
| `reports/latest/latency.json` (n = 200, concurrency 20) | 34.50 ms | 143.86 ms | 165.32 ms | `/search`, 100% success |

Quote the measurement conditions with the number. A latency figure without its
concurrency level and cache state is not comparable to anything.

---

## 5. OpenSearch hybrid retrieval (platform migration)

The same corpus, the same fine-tuned e5 vectors and the same metric functions, moved from a
FAISS library call plus a separate rank_bm25 index onto one OpenSearch index carrying a BM25
analyzed field and a `knn_vector` on every document.

**Final configuration** (`reports/opensearch/metrics.json`)
9,742 docs · 150 queries · candidate_k = 200 · RRF k = 60 · lexical: `match` on `text`,
standard analyzer

| Method | nDCG@10 | recall@100 |
|---|---|---|
| OpenSearch BM25 | **0.6844** | 0.4170 |
| OpenSearch kNN | 0.5428 | 0.4160 |
| OpenSearch RRF (BM25 + kNN) | 0.5850 | **0.4311** |

Per-query latency, sequential and single-client: p50 77.0ms, p95 130.9ms, p99 250.1ms.
**Not comparable** to `reports/latest/latency.json`, which measures `/search` at concurrency 20.

### The control arm

OpenSearch kNN scores **0.5428** against the FAISS path's **0.5428** on byte-identical
vectors. Indexing, HNSW recall and query encoding are therefore correct, and any difference
elsewhere is attributable to the lexical path alone.

### Lexical ablation (`reports/opensearch/lexical_ablation.json`)

The first OpenSearch run scored 0.3745, well below the rank_bm25 reference of 0.6065. That
run changed two things at once, so the cause could not be attributed. Holding corpus, qrels,
metric code and index fixed and varying only the lexical query:

| Variant | nDCG@10 | vs rank_bm25 (0.6065) |
|---|---|---|
| `text`, standard analyzer | **0.6844** | +0.0779 |
| `title` + `text`, cross_fields, standard | 0.6653 | +0.0588 |
| `text.film`, film analyzer | 0.6646 | +0.0581 |
| `title.film` + `text.film`, cross_fields | 0.6602 | +0.0538 |
| `title^2` + `text`, best_fields, standard | 0.4218 | -0.1847 |
| `title.film^2` + `text.film`, best_fields | 0.3745 | -0.2320 |

**Attribution:** the field boost costs **-0.263** (0.6844 to 0.4218); the stemming and
synonym analyzer costs **-0.020**. The regression was almost entirely the boost, not the
analyzer and not the platform.

**Mechanism:** `text` already contains the title. `best_fields` takes the maximum single
field score rather than combining fields, so a 2x boost on a short duplicated field wins
nearly every comparison, and the ranker effectively scored on title alone while discarding
genres and tags.

**Conclusion:** the platform migration did not cost retrieval quality. Correctly configured,
OpenSearch BM25 is **+0.078 above** the rank_bm25 baseline on the same corpus. The initial
regression was a query-configuration defect in this repository.

### Fusion does not help on this corpus

RRF over BM25 and kNN scores 0.5850, **below BM25 alone at 0.6844**. The FAISS path shows
the same shape: weighted hybrid at alpha = 0.2 scores 0.5902, below its BM25 at 0.6065. Two
different fusion methods, same direction.

The cause is the qrel density documented in section 2: with a median of 502 relevant titles
per query, the weaker retriever contributes many plausible lower-ranked relevant documents
that displace the stronger retriever's top hits. Only the learned reranker recovers the loss
(0.9491). Fusion should not be assumed to be free.

---

## 6. Corrections made

| Previously stated | Corrected to | Why |
|---|---|---|
| nDCG@10 = 0.9300 | **0.9491** | 0.9300 appears in no artifact. The measured latest-run value is 0.9491 |
| nDCG@10 = 0.8589 (conservative) | **0.7506** (reference run) | 0.8589 appears in no artifact |
| recall@100: 0.76 → 0.88 | **0.4106 → 0.4340** | The old pair came from this document, not from a run. The real values needed a code fix first (below) |
| BM25 recall@100 = 0.7612 | **0.4106** | Same |
| Dense nDCG@10 = 0.5496 | **0.5428** | Measured value in `reports/latest/metrics.json` |
| "33.8M ratings" attached to retrieval metrics | ml-latest-small, **100,836 ratings, 9,742 titles, 610 users** | The 33.8M set feeds only the Spark job, whose output is unused by ranking |
| MODEL_CARD "MovieLens 25M" | ml-latest-small (and ml-latest for Spark) | Neither 25M file is in the repo |
| SciFact runs presented as MovieLens | Labelled as SciFact | 5,183 docs / 300 queries is SciFact's shape |

### The recall bug

`src/eval/evaluate.py` truncated every ranked list to `k` (10) **before** passing it to
`aggregate_methods_list(..., recall_k=100)`. `recall_at_k` then sliced `ranked[:100]` on a
list that only ever held 10 documents, so recall@100 was mathematically incapable of
exceeding recall@10 — and every row in `reports/latest` and `reports/reference` showed
them identical.

Fixed by keeping `max(k, recall_k)` documents per query; nDCG@10 and MAP@10 are unaffected
because those functions truncate internally. Covered by
`tests/test_eval_recall_truncation.py`, which fails against the original code.

### Known reproducibility issue

The fine-tuned embedder was saved with `sentence-transformers` 5.3.0 while the container
runs 5.2.0, which is why hybrid_ltr came out 0.9499 on one run and 0.9491 on another. Pin
the version in `docker/Dockerfile.api` before quoting a figure to more than three decimals.
