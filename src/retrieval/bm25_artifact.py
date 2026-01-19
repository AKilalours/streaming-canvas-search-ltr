import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


@dataclass
class BM25Artifact:
    doc_ids: list[str]
    bm25: BM25Okapi

    def query(self, q: str, k: int) -> list[tuple[str, float]]:
        toks = tokenize(q)
        scores = np.asarray(self.bm25.get_scores(toks), dtype=np.float32)
        if k <= 0:
            return []
        k = min(k, len(scores))
        top_idx = np.argpartition(-scores, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]
