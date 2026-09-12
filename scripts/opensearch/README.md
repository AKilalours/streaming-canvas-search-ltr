# OpenSearch hybrid retrieval

One index, two retrieval paths, one fusion step.

```
                    movielens-hybrid  (OpenSearch 2.17, 1 shard)
 ┌──────────────────────────────────────────────────────────────────┐
 │ doc_id   keyword                                                 │
 │ title    text   → film_text analyzer                             │
 │ text     text   → lowercase, english stop, film synonyms, stemmer│
 │ embedding knn_vector(768) → HNSW, cosinesimil, ef_c=256, m=16    │
 └──────────────────────────────────────────────────────────────────┘
        │                                   │
     BM25 multi_match                   kNN query
     title^2, text                      k = candidate_k
        │                                   │
        └──────────────┬────────────────────┘
                Reciprocal Rank Fusion (k=60)
```

Both fields live on the **same document**, so a hybrid query is one round trip to one
platform. That is the difference from the FAISS path: FAISS is a library the service calls
and keeps in step with a separate lexical index; OpenSearch is a platform with analyzers,
mappings, shards and aliases, and the index is the contract.

## Why RRF rather than a weighted score sum

BM25 scores are unbounded and corpus dependent. Cosine similarity is bounded in [-1, 1].
There is no stable mapping between them, so a weighted sum needs an alpha tuned per corpus,
which is what the FAISS hybrid path does (alpha = 0.2, measured). RRF uses rank position
only and carries no scale assumption, so it transfers across corpora without retuning.

The cost is real: RRF throws away score magnitude, so it gives up ground when one retriever
is confidently right and the other is confidently wrong. Both are measured below rather than
argued about.

## Run it

```bash
# 1. dependency
uv add opensearch-py          # or: pip install opensearch-py

# 2. cluster
docker compose -f docker/opensearch.compose.yml up -d
curl -s localhost:9200/_cluster/health | jq .status     # wait for yellow or green

# 3. index the corpus and the existing fine-tuned e5 vectors
python scripts/opensearch/index_corpus.py --recreate

# 4. measure
python scripts/opensearch/eval_hybrid.py --out reports/opensearch
```

Results land in `reports/opensearch/metrics.json` and are reconciled into
`DEFINITIVE_NUMBERS.md` like every other figure in this repo.

## Results

```
method                nDCG@10   recall@100
OpenSearch BM25        0.6844       0.4170
OpenSearch kNN         0.5428       0.4160
OpenSearch RRF         0.5850       0.4311
```

**Control.** kNN scores 0.5428 against the FAISS path's 0.5428 on identical vectors. The
vector half of the migration is provably neutral, so any difference is lexical.

**The migration did not cost quality. My first query did.** The initial run scored 0.3745
against the rank_bm25 reference of 0.6065. `ablate_lexical.py` holds everything fixed and
varies only the lexical query:

| Variant | nDCG@10 | vs rank_bm25 |
|---|---|---|
| `text`, standard analyzer | **0.6844** | +0.0779 |
| `title` + `text`, cross_fields | 0.6653 | +0.0588 |
| `text.film`, film analyzer | 0.6646 | +0.0581 |
| `title.film` + `text.film`, cross_fields | 0.6602 | +0.0538 |
| `title^2` + `text`, best_fields | 0.4218 | -0.1847 |
| `title.film^2` + `text.film`, best_fields | 0.3745 | -0.2320 |

The field boost costs -0.263. The stemming and synonym analyzer costs -0.020. `text` already
contains the title, and `best_fields` takes the maximum single-field score rather than
combining, so boosting a short duplicated field by 2x made the ranker score on title alone
and discard genres and tags. Correctly configured, OpenSearch BM25 beats the rank_bm25
baseline by 0.078 on the same corpus.

**Fusion is not free.** RRF scores 0.5850, below BM25 alone at 0.6844. The FAISS path shows
the same shape with a different fusion method (weighted alpha = 0.2: 0.5902, below BM25 at
0.6065). With a median of 502 relevant titles per query, the weaker retriever contributes
plausible lower-ranked relevant documents that displace the stronger retriever's top hits.
Only the learned reranker recovers it.

Per-query latency, sequential single-client: p50 77.0ms, p95 130.9ms, p99 250.1ms. Not
comparable to the `/search` figures in `reports/latest/latency.json`, which are at
concurrency 20.

## Notes

- Security plugin is disabled. Local evaluation only, never a deployment template.
- The index reuses `artifacts/faiss/movielens_ft_e5/embeddings.npy`, so the vectors are
  identical to the FAISS path. Any quality difference is attributable to the retrieval
  platform and the fusion method, not to a different embedder.
- Latency reported by `eval_hybrid.py` is sequential and single-client. It is not comparable
  to the `/search` figures in `reports/latest/latency.json`, which are measured at
  concurrency 20.
