"""
StreamLens — Spark Feature Store loader
========================================
Loads Spark-computed batch features into memory at startup.
These features augment the LTR scoring with user and item signals
computed offline by spark/feature_engineering.py.

At Netflix scale: these would come from a Redis Cluster or DynamoDB,
populated by a Flink streaming job reading from the Spark Parquet output.
Here: loaded from JSON files at startup (batch refresh pattern).
"""
from __future__ import annotations
import json
import pathlib
from typing import Any

_USER_FEATURES: dict[str, dict] = {}
_ITEM_FEATURES: dict[str, dict] = {}
_SPARK_LOADED = False


def load_spark_features(spark_dir: str = "artifacts/spark") -> bool:
    """Load Spark-computed features from JSON exports."""
    global _USER_FEATURES, _ITEM_FEATURES, _SPARK_LOADED
    base = pathlib.Path(spark_dir)

    user_path = base / "user_features.json"
    item_path = base / "item_features.json"

    if user_path.exists():
        _USER_FEATURES = json.load(open(user_path))

    if item_path.exists():
        _ITEM_FEATURES = json.load(open(item_path))

    _SPARK_LOADED = user_path.exists() or item_path.exists()
    return _SPARK_LOADED


def get_user_features(user_id: str) -> dict[str, float]:
    """Get Spark-computed features for a user. Returns defaults if not found."""
    return _USER_FEATURES.get(str(user_id), {
        "watch_count": 0,
        "avg_rating": 3.5,
        "high_rating_ratio": 0.5,
        "taste_breadth": 0.0,
        "rating_stddev": 0.5,
    })


def get_item_features(movie_id: str) -> dict[str, float]:
    """Get Spark-computed features for an item. Returns defaults if not found."""
    return _ITEM_FEATURES.get(str(movie_id), {
        "popularity": 0.0,
        "avg_rating": 3.5,
        "sentiment": 0.5,
        "recency": 0.5,
        "controversy": 0.5,
    })


def spark_feature_status() -> dict[str, Any]:
    return {
        "spark_features_loaded": _SPARK_LOADED,
        "n_users": len(_USER_FEATURES),
        "n_items": len(_ITEM_FEATURES),
        "source": "artifacts/spark/",
        "note": "Run: python spark/feature_engineering.py to populate"
    }
