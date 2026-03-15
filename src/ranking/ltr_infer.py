from __future__ import annotations

import math
import pickle
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ranking.features import FEATURE_NAMES, build_features


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _parse_catalog(text: str) -> dict[str, str]:
    """
    Parses MovieLens-style text like:
      "Title: X | Genres: A,B | Tags: t1, t2 | Language: English"
    Returns dict with lower keys: title/genres/tags/language when present.
    """
    out: dict[str, str] = {}
    parts = [p.strip() for p in (text or "").split("|")]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        out[k.strip().lower()] = v.strip()
    return out


class LTRReranker:
    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names or FEATURE_NAMES
        self.meta = meta or {}

    @staticmethod
    def load(path: str) -> "LTRReranker":
        p = Path(path)

        if p.is_dir():
            cand = p / "ltr.pkl"
            if cand.exists():
                p = cand
            else:
                pkls = sorted(p.glob("*.pkl"))
                if not pkls:
                    raise FileNotFoundError(f"No .pkl found in directory: {path}")
                p = pkls[0]

        with p.open("rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict) and "model" in obj:
            return LTRReranker(
                model=obj["model"],
                feature_names=list(obj.get("feature_names") or FEATURE_NAMES),
                meta=dict(obj.get("meta") or {}),
            )

        return LTRReranker(model=obj, feature_names=FEATURE_NAMES, meta={})

    def _coerce_feature_row(self, feats: Any) -> dict[str, float]:
        """
        build_features may return:
          - dict[name->val]
          - list/tuple aligned to FEATURE_NAMES
        We always return dict[name->val].
        """
        if isinstance(feats, dict):
            return {k: float(v) for k, v in feats.items()}

        if isinstance(feats, (list, tuple)):
            if len(feats) != len(self.feature_names):
                raise RuntimeError(
                    f"Feature length mismatch: got {len(feats)} expected {len(self.feature_names)}"
                )
            return {name: float(val) for name, val in zip(self.feature_names, feats, strict=True)}

        raise TypeError(f"Unsupported feature type from build_features: {type(feats).__name__}")

    def _align_to_model_features(
        self,
        raw: dict[str, float],
        *,
        query: str,
        doc: dict[str, Any],
        bm25_score: float,
        dense_score: float,
    ) -> dict[str, float]:
        """
        Make sure the feature dict contains EXACT keys required by the model artifact.
        This fixes schema drift between training and inference.
        """
        row = dict(raw)

        # Common aliases (your features.py uses *_score; older models expect bm25/dense)
        if "bm25" in self.feature_names and "bm25" not in row:
            row["bm25"] = float(row.get("bm25_score", bm25_score))
        if "dense" in self.feature_names and "dense" not in row:
            row["dense"] = float(row.get("dense_score", dense_score))

        # Logs expected by older models
        if "title_len_log" in self.feature_names and "title_len_log" not in row:
            tl = float(row.get("title_len", 0.0))
            row["title_len_log"] = float(math.log1p(max(0.0, tl)))
        if "text_len_log" in self.feature_names and "text_len_log" not in row:
            # doc_len is what your current features.py computes
            dl = float(row.get("doc_len", row.get("text_len", 0.0)))
            row["text_len_log"] = float(math.log1p(max(0.0, dl)))

        # Tag/genre match (MovieLens)
        if ("tag_match" in self.feature_names and "tag_match" not in row) or (
            "genre_match" in self.feature_names and "genre_match" not in row
        ):
            meta = _parse_catalog(str(doc.get("text") or ""))
            qtok = _tokenize(query)

            tags = meta.get("tags", "")
            genres = meta.get("genres", "")

            tag_tok = _tokenize(tags)
            genre_tok = _tokenize(genres)

            if "tag_match" in self.feature_names and "tag_match" not in row:
                row["tag_match"] = 1.0 if (qtok & tag_tok) else 0.0
            if "genre_match" in self.feature_names and "genre_match" not in row:
                row["genre_match"] = 1.0 if (qtok & genre_tok) else 0.0

        # Final: restrict to model schema, fill missing with 0.0 to avoid NaNs
        return {name: float(row.get(name, 0.0)) for name in self.feature_names}

    def rerank(
        self,
        *,
        query: str,
        corpus: dict[str, dict[str, Any]],
        candidates: list[tuple[str, float]],
        bm25_scores: dict[str, float],
        dense_scores: dict[str, float],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        rows: list[dict[str, float]] = []
        doc_ids: list[str] = []

        for did, hscore in candidates:
            doc = corpus.get(did, {"title": "", "text": ""})
            b = float(bm25_scores.get(did, 0.0))
            d = float(dense_scores.get(did, 0.0))

            feats = build_features(
                query=query,
                doc=doc,
                bm25_score=b,
                dense_score=d,
                hybrid_score=float(hscore),
            )
            raw = self._coerce_feature_row(feats)
            aligned = self._align_to_model_features(raw, query=query, doc=doc, bm25_score=b, dense_score=d)

            rows.append(aligned)
            doc_ids.append(did)

        X_df = pd.DataFrame(rows, columns=self.feature_names).fillna(0.0)
        preds = self.model.predict(X_df)

        scored = list(zip(doc_ids, preds, strict=False))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(d, float(s)) for d, s in scored]
