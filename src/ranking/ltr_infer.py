from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ranking.features import FEATURE_NAMES, build_features


class LTRReranker:
    """
    Loads either:
      (A) raw LightGBM model pickle
      (B) dict artifact: {"model":..., "feature_names":[...], "meta":{...}}

    Always predicts using a pandas DataFrame with the correct feature column order
    to avoid sklearn/lightgbm warnings and keep inference stable.
    """

    def __init__(self, model: Any, feature_names: list[str] | None = None, meta: dict | None = None) -> None:
        self.model = model
        self.feature_names = list(feature_names) if feature_names else list(FEATURE_NAMES)
        self.meta = dict(meta) if meta else {}

    @staticmethod
    def load(path: str) -> "LTRReranker":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)

        # Support artifact dict
        if isinstance(obj, dict) and "model" in obj:
            return LTRReranker(
                model=obj["model"],
                feature_names=list(obj.get("feature_names") or FEATURE_NAMES),
                meta=dict(obj.get("meta") or {}),
            )

        # Support raw model pickle
        return LTRReranker(model=obj, feature_names=list(FEATURE_NAMES), meta={})

    def _feats_to_vector(self, feats: Any) -> list[float]:
        """
        build_features may return either:
          - list[float] / np.ndarray
          - dict[str, float]
        We convert everything into an ordered list[float] aligned to self.feature_names.
        """
        if isinstance(feats, dict):
            return [float(feats.get(name, 0.0)) for name in self.feature_names]
        if isinstance(feats, np.ndarray):
            return [float(x) for x in feats.tolist()]
        return [float(x) for x in feats]

    def rerank(
        self,
        query: str,
        corpus: dict[str, dict],
        candidates: list[tuple[str, float]],
        bm25_scores: dict[str, float],
        dense_scores: dict[str, float],
    ) -> list[tuple[str, float]]:
        """
        candidates: list of (doc_id, hybrid_score) or any prior score.
        Returns: list of (doc_id, ltr_score) sorted desc by ltr_score.
        """
        if not candidates:
            return []

        X_rows: list[list[float]] = []
        doc_ids: list[str] = []

        for did, hscore in candidates:
            doc = corpus.get(did, {"title": "", "text": ""})
            feats = build_features(
                query=query,
                doc=doc,
                bm25_score=float(bm25_scores.get(did, 0.0)),
                dense_score=float(dense_scores.get(did, 0.0)),
                hybrid_score=float(hscore),
            )
            X_rows.append(self._feats_to_vector(feats))
            doc_ids.append(did)

        X_df = pd.DataFrame(X_rows, columns=self.feature_names)
        preds = self.model.predict(X_df)

        out = list(zip(doc_ids, preds, strict=False))
        out.sort(key=lambda x: x[1], reverse=True)
        return [(d, float(s)) for d, s in out]
