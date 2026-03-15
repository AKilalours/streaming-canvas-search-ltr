from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class BM25Artifact:
    """
    Pickle-safe BM25 artifact (lives in retrieval.bm25, NOT __main__).
    """
    doc_ids: list[str]
    doc_len: list[int]
    avgdl: float
    idf: dict[str, float]
    postings: dict[str, list[tuple[int, int]]]  # term -> [(doc_idx, tf)]
    k1: float = 1.2
    b: float = 0.75

    def query(self, q: str, k: int = 10) -> list[tuple[str, float]]:
        terms = tokenize(q)
        if not terms:
            return []

        scores: dict[int, float] = defaultdict(float)

        for t in terms:
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = self.idf.get(t, 0.0)
            for doc_idx, tf in plist:
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[doc_idx] += idf * (tf * (self.k1 + 1.0)) / (denom + 1e-12)

        if not scores:
            return []

        items = [(self.doc_ids[i], s) for i, s in scores.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return [(d, float(s)) for d, s in items[:k]]


def build_bm25(corpus_jsonl: Path) -> BM25Artifact:
    rows = read_jsonl(corpus_jsonl)

    doc_ids: list[str] = []
    doc_len: list[int] = []

    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    df: Counter[str] = Counter()

    for idx, r in enumerate(rows):
        did = str(r["doc_id"])
        title = str(r.get("title") or "")
        text = str(r.get("text") or "")
        full = f"{title} {text}".strip()

        toks = tokenize(full) or ["__empty__"]
        tf = Counter(toks)

        doc_ids.append(did)
        doc_len.append(sum(tf.values()))

        for t in tf.keys():
            df[t] += 1
        for t, c in tf.items():
            postings[t].append((idx, int(c)))

    N = len(doc_ids)
    avgdl = float(sum(doc_len) / max(1, N))

    idf: dict[str, float] = {}
    for t, dfi in df.items():
        idf[t] = math.log(1.0 + (N - dfi + 0.5) / (dfi + 0.5))

    return BM25Artifact(
        doc_ids=doc_ids,
        doc_len=doc_len,
        avgdl=avgdl,
        idf=idf,
        postings=dict(postings),
    )
