"""Persistent contextual Thompson sampling for recommendation slates.

This is a conservative online-learning layer over an existing relevance ranker.
It learns a Bayesian linear reward model from displayed-item feedback and uses
posterior samples to make small, logged, stochastic ranking adjustments.

The SQLite journal stores each impression and its conditional selection
probability. Reward-model updates use the observed outcome directly; they are
not inverse-propensity weighted. This is appropriate for the contextual reward
model and avoids amplifying low-probability, noisy observations. The logged
probability is conditional on a sampled Thompson parameter and is not, by
itself, a marginal propensity suitable for off-policy evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

GENRES = (
    "action",
    "adventure",
    "animation",
    "children",
    "comedy",
    "crime",
    "documentary",
    "drama",
    "fantasy",
    "horror",
    "mystery",
    "romance",
    "sci-fi",
    "thriller",
    "war",
    "western",
)
LEGACY_FEATURE_COUNT = 2 + len(GENRES)
DECADES = tuple(range(1920, 2030, 10))
TAG_BUCKETS = 32
PREVIOUS_FEATURE_COUNT = LEGACY_FEATURE_COUNT + len(DECADES) + TAG_BUCKETS
FEATURE_NAMES = (
    "bias",
    "relevance",
    *GENRES,
    *(f"year_{decade}s" for decade in DECADES),
    *(f"tag_hash_{bucket:02d}" for bucket in range(TAG_BUCKETS)),
    "position",
)
N_FEATURES = len(FEATURE_NAMES)
_GENRES_RE = re.compile(r"genres?\s*:\s*([^|\n]+)", re.IGNORECASE)
_TAGS_RE = re.compile(r"tags?\s*:\s*([^|\n]+)", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_REWARDS = {
    "click": 0.25,
    "watch_start": 0.45,
    "watch_progress": 0.45,
    "watch_complete": 1.0,
    "skip": 0.0,
    "dislike": 0.0,
}


class ContextualSlateBandit:
    """Bayesian linear Thompson sampler with per-user and shared posteriors."""

    def __init__(
        self,
        db_path: str | Path = "artifacts/bandit/online_contextual.sqlite3",
        *,
        prior_precision: float = 2.0,
        reward_noise: float = 0.25,
        exploration_scale: float = 0.35,
        temperature: float = 0.30,
        relevance_weight: float = 2.8,
        shared_weight: float = 0.25,
        max_impressions: int = 250_000,
        seed: int | None = None,
    ) -> None:
        values = {
            "prior_precision": prior_precision,
            "reward_noise": reward_noise,
            "exploration_scale": exploration_scale,
            "temperature": temperature,
            "relevance_weight": relevance_weight,
            "shared_weight": shared_weight,
        }
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("bandit configuration values must all be finite")
        if prior_precision <= 0 or reward_noise <= 0 or temperature <= 0:
            raise ValueError("prior_precision, reward_noise, and temperature must be positive")
        if not 0 <= exploration_scale <= 2 or not 0 <= relevance_weight <= 100:
            raise ValueError("exploration_scale must be in [0, 2] and relevance_weight in [0, 100]")
        if not 0 <= shared_weight <= 1:
            raise ValueError("shared_weight must be in [0, 1]")
        if not isinstance(max_impressions, int) or not 1 <= max_impressions <= 10_000_000:
            raise ValueError("max_impressions must be an integer in [1, 10000000]")
        if not 0.01 <= temperature <= 10 or prior_precision > 1e6 or reward_noise > 1e3:
            raise ValueError("temperature, prior_precision, or reward_noise exceeds safe bounds")
        self.db_path = Path(db_path)
        self.prior_precision = float(prior_precision)
        self.reward_noise = float(reward_noise)
        self.exploration_scale = float(exploration_scale)
        self.temperature = float(temperature)
        self.relevance_weight = float(relevance_weight)
        self.shared_weight = float(shared_weight)
        self.max_impressions = max_impressions
        self._rng = np.random.default_rng(seed)
        self._rng_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        return conn

    def _initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS bandit_models (
                    user_id TEXT PRIMARY KEY,
                    precision TEXT NOT NULL,
                    response TEXT NOT NULL,
                    observations INTEGER NOT NULL DEFAULT 0,
                    reward_sum REAL NOT NULL DEFAULT 0
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS bandit_impressions (
                    impression_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    features TEXT NOT NULL,
                    propensity REAL NOT NULL,
                    created_at REAL NOT NULL,
                    event_type TEXT,
                    reward REAL,
                    updated_at REAL
                )"""
            )
            # Additive migrations keep local journals created by earlier demo
            # versions readable without resetting learned model state.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(bandit_impressions)")}
            for name, declaration in (
                ("slate_id", "TEXT NOT NULL DEFAULT ''"),
                ("position", "INTEGER NOT NULL DEFAULT 0"),
                ("candidate_count", "INTEGER NOT NULL DEFAULT 0"),
                ("baseline_rank", "INTEGER NOT NULL DEFAULT 0"),
                ("base_score", "REAL NOT NULL DEFAULT 0.0"),
                ("candidate_context", "TEXT NOT NULL DEFAULT '{}'"),
                ("exposed_at", "REAL"),
                ("causal_logged", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if name not in columns:
                    conn.execute(f"ALTER TABLE bandit_impressions ADD COLUMN {name} {declaration}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS bandit_impressions_user_time "
                "ON bandit_impressions(user_id, created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS bandit_impressions_slate "
                "ON bandit_impressions(slate_id, position)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS bandit_impressions_created "
                "ON bandit_impressions(created_at DESC)"
            )
            conn.commit()

    @staticmethod
    def _genres(row: dict[str, Any]) -> set[str]:
        text = str(row.get("text") or "")
        match = _GENRES_RE.search(text)
        raw = match.group(1) if match else str(row.get("genres") or "")
        values = {part.strip().lower() for part in re.split(r"[,;/]", raw)}
        return values & set(GENRES)

    def _features(
        self,
        row: dict[str, Any],
        relevance: float,
    ) -> np.ndarray:
        x = np.zeros(N_FEATURES, dtype=np.float64)
        x[0] = 1.0
        x[1] = float(np.clip(relevance, 0.0, 1.0))
        for genre in self._genres(row):
            x[2 + GENRES.index(genre)] = 1.0
        text = str(row.get("text") or "")
        title = str(row.get("title") or "")
        year_match = _YEAR_RE.search(f"{title} {text}")
        if year_match:
            decade = (int(year_match.group(1)) // 10) * 10
            if decade in DECADES:
                x[2 + len(GENRES) + DECADES.index(decade)] = 1.0
        tag_match = _TAGS_RE.search(text)
        raw_tags = tag_match.group(1) if tag_match else row.get("tags", "")
        tag_values = re.split(r"[,;/]", str(raw_tags))
        tag_offset = 2 + len(GENRES) + len(DECADES)
        for tag in tag_values:
            normalized = " ".join(tag.lower().split())
            if normalized:
                digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=4).digest()
                bucket = int.from_bytes(digest, "big") % TAG_BUCKETS
                x[tag_offset + bucket] = 1.0
        return x

    def _model(self, conn: sqlite3.Connection, user_id: str) -> tuple[np.ndarray, np.ndarray, int]:
        row = conn.execute(
            "SELECT precision, response, observations FROM bandit_models WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return self.prior_precision * np.eye(N_FEATURES), np.zeros(N_FEATURES), 0
        precision = np.asarray(json.loads(row["precision"]), dtype=np.float64)
        response = np.asarray(json.loads(row["response"]), dtype=np.float64)
        stored_size = response.shape[0] if response.ndim == 1 else -1
        if stored_size in {LEGACY_FEATURE_COUNT, PREVIOUS_FEATURE_COUNT} and precision.shape == (
            stored_size,
            stored_size,
        ):
            expanded_precision = self.prior_precision * np.eye(N_FEATURES)
            expanded_precision[:stored_size, :stored_size] = precision
            expanded_response = np.zeros(N_FEATURES, dtype=np.float64)
            expanded_response[:stored_size] = response
            return expanded_precision, expanded_response, int(row["observations"])
        if precision.shape != (N_FEATURES, N_FEATURES) or response.shape != (N_FEATURES,):
            raise ValueError("Stored bandit model has an incompatible feature schema")
        return precision, response, int(row["observations"])

    def _posterior(
        self, conn: sqlite3.Connection, user_id: str
    ) -> tuple[np.ndarray, np.ndarray, int]:
        user_a, user_b, user_n = self._model(conn, user_id)
        global_a, global_b, global_n = self._model(conn, "__global__")
        prior = self.prior_precision * np.eye(N_FEATURES)
        precision = prior + (user_a - prior) + self.shared_weight * (global_a - prior)
        response = user_b + self.shared_weight * global_b
        # Numerical protection for long-running or corrupted local state.
        precision = (precision + precision.T) * 0.5
        return precision, response, user_n + global_n

    def _sample_theta(self, precision: np.ndarray, response: np.ndarray) -> np.ndarray:
        mean = np.linalg.solve(precision, response)
        covariance = np.linalg.solve(precision, np.eye(N_FEATURES))
        covariance = (covariance + covariance.T) * 0.5
        # Cholesky may fail only for damaged state; return the posterior mean then.
        try:
            chol = np.linalg.cholesky(covariance + 1e-9 * np.eye(N_FEATURES))
        except np.linalg.LinAlgError:
            return mean
        noise = self._rng.standard_normal(N_FEATURES)
        return mean + self.exploration_scale * math.sqrt(self.reward_noise) * (chol @ noise)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        values = np.exp(np.clip(shifted, -60.0, 0.0))
        total = float(values.sum())
        return values / total if total > 0 else np.full(len(logits), 1.0 / len(logits))

    def recommend(
        self,
        user_id: str,
        candidates: list[tuple[str, float]],
        corpus: dict[str, dict[str, Any]],
        *,
        k: int,
    ) -> list[dict[str, Any]]:
        """Sample an ordered slate and persist each action's conditional propensity."""
        if not user_id or len(user_id) > 128 or user_id == "__global__":
            raise ValueError("user_id must be 1–128 characters and cannot be reserved")
        if k < 1:
            return []
        # Keep the best retrieved occurrence of each item and preserve its order.
        unique: list[tuple[str, float]] = []
        seen: set[str] = set()
        for doc_id, score in candidates:
            doc_id = str(doc_id)
            if doc_id not in seen and math.isfinite(float(score)):
                unique.append((doc_id, float(score)))
                seen.add(doc_id)
        if not unique:
            return []

        vals = np.asarray([score for _, score in unique], dtype=np.float64)
        lo, hi = float(vals.min()), float(vals.max())
        norm_scores = (vals - lo) / max(hi - lo, 1e-12)
        rows = [corpus.get(doc_id, {}) for doc_id, _ in unique]
        features = [self._features(row, float(norm_scores[i])) for i, row in enumerate(rows)]

        with self._rng_lock, closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            precision, response, _ = self._posterior(conn, user_id)
            theta = self._sample_theta(precision, response)
            remaining = list(range(len(unique)))
            output: list[dict[str, Any]] = []
            now = time.time()
            slate_id = secrets.token_urlsafe(18)
            slate_size = min(k, len(unique))
            for _position in range(min(k, len(remaining))):
                logits = (
                    np.asarray(
                        [
                            self.relevance_weight * norm_scores[i]
                            + float(np.clip(features[i] @ theta, -1.0, 1.0))
                            for i in remaining
                        ]
                    )
                    / self.temperature
                )
                probs = self._softmax(logits)
                selected_offset = int(self._rng.choice(len(remaining), p=probs))
                candidate_context = {
                    "candidate_scores": {unique[i][0]: float(unique[i][1]) for i in remaining},
                    "behavior_probabilities": {
                        unique[i][0]: float(probs[offset]) for offset, i in enumerate(remaining)
                    },
                    "policy": "contextual_thompson_sampling_v1",
                    "slate_id": slate_id,
                    "position": _position + 1,
                }
                selected_index = remaining.pop(selected_offset)
                doc_id, original_score = unique[selected_index]
                # Keep the actual conditional probability; flooring would
                # silently misreport the policy's logged randomization.
                propensity = float(probs[selected_offset])
                impression_id = secrets.token_urlsafe(18)
                logged_features = features[selected_index].copy()
                logged_features[FEATURE_NAMES.index("position")] = _position / max(
                    slate_size - 1, 1
                )
                conn.execute(
                    """INSERT INTO bandit_impressions
                       (impression_id, user_id, doc_id, features, propensity, created_at,
                        slate_id, position, candidate_count, baseline_rank, base_score,
                        candidate_context)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        impression_id,
                        user_id,
                        doc_id,
                        json.dumps(logged_features.tolist()),
                        propensity,
                        now,
                        slate_id,
                        _position + 1,
                        len(unique),
                        selected_index + 1,
                        original_score,
                        json.dumps(candidate_context),
                    ),
                )
                output.append(
                    {
                        "doc_id": doc_id,
                        "score": original_score,
                        "impression_id": impression_id,
                        "propensity": propensity,
                        "slate_id": slate_id,
                        "position": _position + 1,
                        "candidate_count": len(unique),
                        "baseline_rank": selected_index + 1,
                        "policy": "contextual_thompson_sampling_v1",
                        "candidate_context": candidate_context,
                    }
                )
            # Bound the journal while retaining enough data for delayed feedback.
            conn.execute("DELETE FROM bandit_impressions WHERE created_at < ?", (now - 90 * 86400,))
            exposure_count = int(
                conn.execute("SELECT COUNT(*) FROM bandit_impressions").fetchone()[0]
            )
            if exposure_count > self.max_impressions:
                conn.execute(
                    """DELETE FROM bandit_impressions WHERE impression_id IN (
                           SELECT impression_id FROM bandit_impressions
                           ORDER BY created_at DESC, impression_id DESC
                           LIMIT -1 OFFSET ?
                       )""",
                    (self.max_impressions,),
                )
            conn.commit()
        return output

    @staticmethod
    def _reward(event_type: str, watch_pct: float | None) -> float:
        if event_type not in _REWARDS:
            raise ValueError(f"event_type must be one of: {', '.join(_REWARDS)}")
        if event_type == "watch_progress" and watch_pct is None:
            raise ValueError("watch_progress requires watch_pct")
        if watch_pct is not None and not math.isfinite(float(watch_pct)):
            raise ValueError("watch_pct must be finite")
        if event_type in {"watch_progress", "watch_complete"} and watch_pct is not None:
            return 0.45 + 0.55 * float(np.clip(watch_pct, 0.0, 1.0))
        return _REWARDS[event_type]

    def mark_exposure(self, impression_id: str, *, user_id: str, doc_id: str) -> dict[str, Any]:
        """Confirm that a recommendation entered the user's visible viewport."""
        if not impression_id or not user_id or not doc_id:
            return {"accepted": False, "reason": "missing_identity"}
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM bandit_impressions WHERE impression_id=?",
                (impression_id,),
            ).fetchone()
            if row is None or row["user_id"] != user_id or row["doc_id"] != doc_id:
                conn.rollback()
                return {"accepted": False, "reason": "unknown_impression"}
            is_new = row["exposed_at"] is None
            if is_new:
                conn.execute(
                    "UPDATE bandit_impressions SET exposed_at=? WHERE impression_id=?",
                    (time.time(), impression_id),
                )
            conn.commit()
        return {
            "accepted": True,
            "new_exposure": is_new,
            "impression_id": impression_id,
            "propensity": float(row["propensity"]),
            "position": int(row["position"]),
            "slate_id": row["slate_id"],
            "candidate_context": json.loads(row["candidate_context"] or "{}"),
            "base_score": float(row["base_score"]),
            "causal_logged": bool(row["causal_logged"]),
        }

    def mark_causal_logged(self, impression_id: str, *, user_id: str) -> bool:
        """Mark successful persistence in the causal event journal for retry safety."""
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "UPDATE bandit_impressions SET causal_logged=1 WHERE impression_id=? AND user_id=? AND exposed_at IS NOT NULL",
                (impression_id, user_id),
            )
            conn.commit()
            return cursor.rowcount == 1

    def record_feedback(
        self,
        impression_id: str,
        event_type: str,
        *,
        user_id: str,
        doc_id: str,
        watch_pct: float | None = None,
        require_exposure: bool = False,
    ) -> dict[str, Any]:
        """Apply one bounded outcome to a previously served action."""
        reward = self._reward(event_type, watch_pct)
        if not impression_id:
            raise ValueError("impression_id is required")
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            impression = conn.execute(
                "SELECT * FROM bandit_impressions WHERE impression_id = ?",
                (impression_id,),
            ).fetchone()
            if (
                impression is None
                or not user_id
                or impression["user_id"] != user_id
                or not doc_id
                or impression["doc_id"] != doc_id
            ):
                conn.rollback()
                return {"accepted": False, "reason": "unknown_impression"}
            if require_exposure and impression["exposed_at"] is None:
                conn.rollback()
                return {"accepted": False, "reason": "exposure_not_confirmed"}
            previous_reward = impression["reward"]
            previous_event = impression["event_type"]
            terminal_events = {"watch_complete", "skip", "dislike"}
            if previous_reward is not None and (
                event_type == previous_event
                or previous_event in terminal_events
                or event_type == "click"
                or (event_type == "watch_progress" and reward <= float(previous_reward))
            ):
                conn.rollback()
                return {"accepted": False, "reason": "already_labeled"}

            x = np.asarray(json.loads(impression["features"]), dtype=np.float64)
            if x.shape in {(LEGACY_FEATURE_COUNT,), (PREVIOUS_FEATURE_COUNT,)}:
                x = np.pad(x, (0, N_FEATURES - len(x)))
            if x.shape != (N_FEATURES,):
                conn.rollback()
                raise ValueError("Stored impression has an incompatible feature schema")
            propensity = float(impression["propensity"])
            observation_weight = 1.0 / self.reward_noise
            reward_delta = reward - float(previous_reward or 0.0)
            for target_user in ("__global__", impression["user_id"]):
                matrix, vector, count = self._model(conn, target_user)
                if previous_reward is None:
                    matrix += observation_weight * np.outer(x, x)
                    count += 1
                vector += observation_weight * reward_delta * x
                conn.execute(
                    """INSERT INTO bandit_models
                       (user_id, precision, response, observations, reward_sum)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(user_id) DO UPDATE SET
                         precision=excluded.precision,
                         response=excluded.response,
                         observations=excluded.observations,
                         reward_sum=bandit_models.reward_sum + excluded.reward_sum""",
                    (
                        target_user,
                        json.dumps(matrix.tolist()),
                        json.dumps(vector.tolist()),
                        count,
                        reward_delta,
                    ),
                )
            conn.execute(
                "UPDATE bandit_impressions SET event_type=?, reward=?, updated_at=? WHERE impression_id=?",
                (event_type, reward, time.time(), impression_id),
            )
            conn.commit()
        return {
            "accepted": True,
            "reward": round(reward, 4),
            "previous_reward": round(float(previous_reward), 4)
            if previous_reward is not None
            else None,
            "propensity": round(propensity, 6),
        }

    def status(self) -> dict[str, Any]:
        now = time.time()
        with closing(self._connect()) as conn:
            recommendations = int(
                conn.execute("SELECT COUNT(*) FROM bandit_impressions").fetchone()[0]
            )
            exposures = int(
                conn.execute(
                    "SELECT COUNT(*) FROM bandit_impressions WHERE exposed_at IS NOT NULL"
                ).fetchone()[0]
            )
            labeled = int(
                conn.execute(
                    "SELECT COUNT(*) FROM bandit_impressions WHERE exposed_at IS NOT NULL AND reward IS NOT NULL"
                ).fetchone()[0]
            )
            users = int(
                conn.execute(
                    "SELECT COUNT(*) FROM bandit_models WHERE user_id != '__global__'"
                ).fetchone()[0]
            )
            model = conn.execute(
                "SELECT observations, reward_sum FROM bandit_models WHERE user_id='__global__'"
            ).fetchone()
            recent = conn.execute(
                """SELECT COUNT(exposed_at) AS impressions,
                          COUNT(DISTINCT slate_id) AS slates,
                          SUM(CASE WHEN reward IS NOT NULL THEN 1 ELSE 0 END) AS labeled,
                          AVG(reward) AS mean_reward,
                          AVG(propensity) AS mean_propensity,
                          MIN(propensity) AS min_propensity,
                          AVG(position) AS mean_position
                   FROM bandit_impressions WHERE created_at >= ? AND exposed_at IS NOT NULL""",
                (now - 7 * 86400,),
            ).fetchone()
            pending = int(
                conn.execute(
                    "SELECT COUNT(*) FROM bandit_impressions WHERE exposed_at IS NOT NULL AND reward IS NULL"
                ).fetchone()[0]
            )
            outcome_counts = {
                str(row["event_type"]): int(row["count"])
                for row in conn.execute(
                    """SELECT event_type, COUNT(*) AS count FROM bandit_impressions
                       WHERE created_at >= ? AND event_type IS NOT NULL
                       GROUP BY event_type""",
                    (now - 7 * 86400,),
                )
            }
        observations = int(model["observations"]) if model else 0
        reward_sum = float(model["reward_sum"]) if model else 0.0
        return {
            "enabled": True,
            "algorithm": "Bayesian linear contextual Thompson sampling",
            "feature_count": N_FEATURES,
            "features": list(FEATURE_NAMES),
            "max_impressions": self.max_impressions,
            "exposures": exposures,
            "recommendations": recommendations,
            "labeled_impressions": labeled,
            "users_with_feedback": users,
            "global_observations": observations,
            "mean_reward": round(reward_sum / observations, 4) if observations else None,
            "propensity_logged": True,
            "propensity_semantics": "exact conditional action probability given sampled Thompson parameter and remaining slate candidates",
            "off_policy_evaluation": "per-decision SNIPS diagnostic from eligible mature randomized outcomes",
            "pending_feedback": pending,
            "last_7_days": {
                "impressions": int(recent["impressions"] or 0),
                "slates": int(recent["slates"] or 0),
                "labeled_impressions": int(recent["labeled"] or 0),
                "feedback_coverage": round(
                    int(recent["labeled"] or 0) / int(recent["impressions"] or 1), 4
                ),
                "outcome_counts": outcome_counts,
                "mean_reward": round(float(recent["mean_reward"]), 4)
                if recent["mean_reward"] is not None
                else None,
                "mean_propensity": round(float(recent["mean_propensity"]), 6)
                if recent["mean_propensity"] is not None
                else None,
                "min_propensity": round(float(recent["min_propensity"]), 6)
                if recent["min_propensity"] is not None
                else None,
                "mean_position": round(float(recent["mean_position"]), 2)
                if recent["mean_position"] is not None
                else None,
            },
        }

    def predict_reward(
        self,
        user_id: str,
        row: dict[str, Any],
        *,
        relevance: float = 0.5,
    ) -> float:
        """Return a clipped posterior mean, useful for diagnostics and tests."""
        with closing(self._connect()) as conn:
            precision, response, _ = self._posterior(conn, user_id)
        mean = np.linalg.solve(precision, response)
        return float(np.clip(self._features(row, relevance) @ mean, 0.0, 1.0))
