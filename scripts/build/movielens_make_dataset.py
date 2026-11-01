from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def _norm(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw/movielens/ml-latest-small")
    ap.add_argument("--out_dir", default="data/processed/movielens")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_train", type=int, default=600)
    ap.add_argument("--n_val", type=int, default=150)
    ap.add_argument("--n_test", type=int, default=150)
    ap.add_argument("--min_rels_per_query", type=int, default=5)
    args = ap.parse_args()

    random.seed(args.seed)

    raw = Path(args.raw_dir)
    if not (raw / "movies.csv").exists():
        raise FileNotFoundError(f"missing movies.csv at {raw}")
    if not (raw / "tags.csv").exists():
        raise FileNotFoundError(f"missing tags.csv at {raw}")

    movies = pd.read_csv(raw / "movies.csv")  # movieId,title,genres
    tags = pd.read_csv(raw / "tags.csv")      # userId,movieId,tag,timestamp

    # tags aggregation
    tag_counts = Counter(_norm(t) for t in tags["tag"].astype(str).tolist())
    # keep reasonably frequent tags to form queries
    frequent_tags = [t for t, c in tag_counts.items() if c >= 8 and len(t) >= 3]
    frequent_tags = sorted(frequent_tags, key=lambda x: (-tag_counts[x], x))[:400]

    # movie -> tags (top few)
    movie_to_tags: dict[int, list[str]] = defaultdict(list)
    for mid, t in zip(tags["movieId"].tolist(), tags["tag"].astype(str).tolist(), strict=False):
        movie_to_tags[int(mid)].append(_norm(t))

    movie_tags_top: dict[int, list[str]] = {}
    for mid, ts in movie_to_tags.items():
        c = Counter(ts)
        movie_tags_top[mid] = [t for t, _ in c.most_common(8)]

    # corpus
    corpus_rows: list[dict] = []
    movie_meta: dict[str, dict[str, set[str]]] = {}
    all_genres: set[str] = set()

    for _, r in movies.iterrows():
        mid = int(r["movieId"])
        title = str(r["title"])
        genres = str(r["genres"])
        gs = [g for g in genres.split("|") if g and g != "(no genres listed)"]
        for g in gs:
            all_genres.add(_norm(g))

        tags_top = movie_tags_top.get(mid, [])

        # create "catalog text" like Netflix metadata
        text = " | ".join(
            [
                f"Title: {title}",
                f"Genres: {', '.join(gs) if gs else 'unknown'}",
                f"Tags: {', '.join(tags_top) if tags_top else 'none'}",
            ]
        )

        doc_id = str(mid)
        corpus_rows.append({"doc_id": doc_id, "title": title, "text": text})

        movie_meta[doc_id] = {
            "genres": set(_norm(g) for g in gs),
            "tags": set(tags_top),
        }

    all_genres_list = sorted(all_genres) if all_genres else ["drama"]

    templates = [
        "{tag}",
        "{genre}",
        "{tag} {genre}",
        "feel good {genre}",
        "gritty {genre}",
        "{genre} with {tag}",
        "{tag} vibes",
    ]

    def make_query() -> str:
        tag = random.choice(frequent_tags) if frequent_tags else random.choice(all_genres_list)
        genre = random.choice(all_genres_list)
        t = random.choice(templates)
        return _norm(t.format(tag=tag, genre=genre))

    def label_for(movie_id: str, q: str) -> int:
        meta = movie_meta[movie_id]
        qn = _norm(q)
        # 2: tag match, 1: genre match, 0: none
        for tg in meta["tags"]:
            if tg and tg in qn:
                return 2
        for g in meta["genres"]:
            if g and g in qn:
                return 1
        return 0

    def build_split(nq: int, split_name: str) -> tuple[list[dict], dict[str, dict[str, int]]]:
        queries_rows: list[dict] = []
        qrels: dict[str, dict[str, int]] = {}
        tries = 0
        made = 0

        # guard: avoid infinite loops if tags are sparse
        max_tries = nq * 60

        while made < nq and tries < max_tries:
            tries += 1
            qtext = make_query()

            rels: dict[str, int] = {}
            for doc_id in movie_meta.keys():
                rel = label_for(doc_id, qtext)
                if rel > 0:
                    rels[doc_id] = rel

            if len(rels) < args.min_rels_per_query:
                continue

            qid = f"{split_name}_q{made:05d}"
            queries_rows.append({"query_id": qid, "text": qtext})
            qrels[qid] = rels
            made += 1

        if made < nq:
            raise RuntimeError(
                f"Could only create {made}/{nq} queries for split={split_name}. "
                f"Try lowering --min_rels_per_query (current={args.min_rels_per_query})."
            )

        return queries_rows, qrels

    train_q, train_qrels = build_split(args.n_train, "train")
    val_q, val_qrels = build_split(args.n_val, "val")
    test_q, test_qrels = build_split(args.n_test, "test")

    out_root = Path(args.out_dir)
    for split, qrows, qrels in [
        ("train", train_q, train_qrels),
        ("val", val_q, val_qrels),
        ("test", test_q, test_qrels),
    ]:
        split_dir = out_root / split
        _write_jsonl(split_dir / "corpus.jsonl", corpus_rows)
        _write_jsonl(split_dir / "queries.jsonl", qrows)
        _write_json(split_dir / "qrels.json", qrels)

    card = {
        "dataset": "MovieLens (ml-latest-small)",
        "purpose": "Proxy streaming catalog dataset for search relevance demos (titles + genres + user tags).",
        "note": "Queries/qrels are synthetic from tags/genres; used to validate ranking/eval infra and LTR lift.",
        "num_docs": len(corpus_rows),
        "splits": {"train": len(train_q), "val": len(val_q), "test": len(test_q)},
    }
    _write_json(out_root / "dataset_card.json", card)

    print("[OK] wrote MovieLens demo dataset to:", out_root)
    print("[OK] docs:", len(corpus_rows), "train/val/test queries:", len(train_q), len(val_q), len(test_q))


if __name__ == "__main__":
    main()
