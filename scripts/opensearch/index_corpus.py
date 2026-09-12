"""Build a single OpenSearch index that carries BM25 text and a knn_vector on the same document.

The point of putting both on one document is that a hybrid query is then one round trip to
one platform, rather than a library call plus a separate lexical index that must be kept in
step. That is the difference between FAISS-as-a-library and a search platform.

    python scripts/opensearch/index_corpus.py --recreate

Reads:
    data/processed/movielens/test/corpus.jsonl      doc_id, title, text
    artifacts/faiss/movielens_ft_e5/embeddings.npy  (9742, 768) float32, L2-normalised
    artifacts/faiss/movielens_ft_e5/doc_ids.json    row order for the matrix above
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from opensearchpy import OpenSearch, helpers

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/processed/movielens/test/corpus.jsonl"
EMB_DIR = ROOT / "artifacts/faiss/movielens_ft_e5"
INDEX = "movielens-hybrid"

# Film vocabulary that a plain english analyzer splits badly. Synonyms live in the index,
# not in the query, so they are applied consistently at index and search time.
SYNONYMS = [
    "sci-fi, scifi, science fiction",
    "rom-com, romcom, romantic comedy",
    "wwii, ww2, world war two, world war ii",
    "animated, animation, cartoon",
    "docu, documentary",
]

MAPPING = {
    "settings": {
        # ef_search is an INDEX setting in OpenSearch 2.x, not a node setting.
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 256,
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "analysis": {
            "filter": {
                "film_synonyms": {"type": "synonym_graph", "synonyms": SYNONYMS},
                "english_stop": {"type": "stop", "stopwords": "_english_"},
                "english_stemmer": {"type": "stemmer", "language": "english"},
            },
            "analyzer": {
                "film_text": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "english_stop", "film_synonyms", "english_stemmer"],
                }
            },
        },
    },
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            # Each text field is indexed twice: once with the stock `standard` analyzer and
            # once with `film_text` (stemming, english stopwords, film synonyms). Multi-fields
            # mean one document, one build, and an analyzer ablation that is exact rather than
            # approximate: every variant scores the identical corpus.
            "title": {
                "type": "text",
                "analyzer": "standard",
                "fields": {"film": {"type": "text", "analyzer": "film_text"}},
            },
            "text": {
                "type": "text",
                "analyzer": "standard",
                "fields": {"film": {"type": "text", "analyzer": "film_text"}},
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                    "parameters": {"ef_construction": 256, "m": 16},
                },
            },
        }
    },
}


def load_corpus() -> dict[str, dict]:
    docs: dict[str, dict] = {}
    with CORPUS.open() as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                docs[str(d["doc_id"])] = d
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:9200")
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()

    client = OpenSearch(hosts=[args.host], timeout=60)

    docs = load_corpus()
    emb = np.load(EMB_DIR / "embeddings.npy")
    ids = [str(x) for x in json.loads((EMB_DIR / "doc_ids.json").read_text())]

    if emb.shape[0] != len(ids):
        raise SystemExit(f"embeddings rows {emb.shape[0]} != doc_ids {len(ids)}")
    missing = [i for i in ids if i not in docs]
    if missing:
        raise SystemExit(f"{len(missing)} embedded ids absent from corpus, first: {missing[:5]}")

    if args.recreate and client.indices.exists(index=INDEX):
        client.indices.delete(index=INDEX)
    if not client.indices.exists(index=INDEX):
        client.indices.create(index=INDEX, body=MAPPING)
        print(f"created index {INDEX} (dim={emb.shape[1]}, docs={len(ids)})")

    def actions():
        for row, doc_id in enumerate(ids):
            d = docs[doc_id]
            yield {
                "_index": INDEX,
                "_id": doc_id,
                "_source": {
                    "doc_id": doc_id,
                    "title": d.get("title", ""),
                    "text": d.get("text", ""),
                    "embedding": emb[row].tolist(),
                },
            }

    ok, errors = helpers.bulk(client, actions(), chunk_size=500, request_timeout=120)
    client.indices.refresh(index=INDEX)
    count = client.count(index=INDEX)["count"]
    print(f"indexed ok={ok} errors={len(errors) if errors else 0} count={count}")


if __name__ == "__main__":
    main()
