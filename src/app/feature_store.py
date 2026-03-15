# src/app/feature_store.py
"""
Redis-backed Feature Store
============================
Serves pre-computed user and item features for online serving.
Separates offline feature computation from online serving — 
the same pattern used by Netflix, Uber, and Airbnb ML platforms.

Offline: features computed by Metaflow/Airflow pipeline, stored in Redis
Online: API reads features at serving time with <1ms lookup
"""
from __future__ import annotations
import json, time
from typing import Any


class FeatureStore:
    """
    Two-tier feature store:
    - User features: personalization signals (precomputed per user)
    - Item features: content signals (precomputed per title)
    """

    def __init__(self, redis_client=None) -> None:
        self.redis = redis_client
        self.user_prefix = "fs:user:"
        self.item_prefix = "fs:item:"
        self.ttl_user = 86400      # 24h — user features refresh daily
        self.ttl_item = 604800     # 7d  — item features refresh weekly

    def put_user_features(self, user_id: str, features: dict) -> bool:
        if not self.redis:
            return False
        try:
            self.redis.setex(
                f"{self.user_prefix}{user_id}",
                self.ttl_user,
                json.dumps({**features, "_ts": time.time()}),
            )
            return True
        except Exception:
            return False

    def get_user_features(self, user_id: str) -> dict[str, Any]:
        if not self.redis:
            return self._default_user_features(user_id)
        try:
            raw = self.redis.get(f"{self.user_prefix}{user_id}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return self._default_user_features(user_id)

    def put_item_features(self, doc_id: str, features: dict) -> bool:
        if not self.redis:
            return False
        try:
            self.redis.setex(
                f"{self.item_prefix}{doc_id}",
                self.ttl_item,
                json.dumps({**features, "_ts": time.time()}),
            )
            return True
        except Exception:
            return False

    def get_item_features(self, doc_id: str) -> dict[str, Any]:
        if not self.redis:
            return self._default_item_features(doc_id)
        try:
            raw = self.redis.get(f"{self.item_prefix}{doc_id}")
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return self._default_item_features(doc_id)

    def batch_get_item_features(self, doc_ids: list[str]) -> dict[str, dict]:
        return {doc_id: self.get_item_features(doc_id) for doc_id in doc_ids}

    def get_stats(self) -> dict[str, Any]:
        if not self.redis:
            return {"available": False}
        try:
            user_keys = len(self.redis.keys(f"{self.user_prefix}*"))
            item_keys = len(self.redis.keys(f"{self.item_prefix}*"))
            return {
                "available": True,
                "user_features_cached": user_keys,
                "item_features_cached": item_keys,
                "user_ttl_hours": self.ttl_user // 3600,
                "item_ttl_days": self.ttl_item // 86400,
                "pattern": "offline_precompute → redis → online_serve",
            }
        except Exception:
            return {"available": False}

    def _default_user_features(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "avg_rating": 3.5,
            "n_ratings": 0,
            "preferred_genres": [],
            "recency_weight": 0.5,
            "exploration_rate": 0.15,
            "_source": "default",
        }

    def _default_item_features(self, doc_id: str) -> dict:
        return {
            "doc_id": doc_id,
            "popularity_score": 0.5,
            "avg_rating": 3.5,
            "n_ratings": 0,
            "release_year": 2000,
            "_source": "default",
        }
