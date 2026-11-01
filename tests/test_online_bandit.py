from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from exploration.online_bandit import (
    FEATURE_NAMES,
    LEGACY_FEATURE_COUNT,
    N_FEATURES,
    PREVIOUS_FEATURE_COUNT,
    ContextualSlateBandit,
)


def _fixtures():
    corpus = {
        "a": {"title": "Action A", "text": "Genres: Action, Adventure |"},
        "b": {"title": "Drama B", "text": "Genres: Drama |"},
        "c": {"title": "Romance C", "text": "Genres: Romance, Comedy |"},
        "d": {"title": "Sci-Fi D", "text": "Genres: Sci-Fi, Adventure |"},
    }
    candidates = [
        (doc_id, score) for doc_id, score in zip(corpus, [0.9, 0.8, 0.7, 0.6], strict=True)
    ]
    return corpus, candidates


def test_recommend_logs_unique_actions_and_valid_propensities(tmp_path):
    bandit = ContextualSlateBandit(tmp_path / "bandit.sqlite3", seed=7)
    corpus, candidates = _fixtures()

    slate = bandit.recommend("profile-1", candidates, corpus, k=3)

    assert len(slate) == 3
    assert len({item["doc_id"] for item in slate}) == 3
    assert all(0 < item["propensity"] <= 1 for item in slate)
    assert all(item["impression_id"] for item in slate)
    assert len({item["slate_id"] for item in slate}) == 1
    assert [item["position"] for item in slate] == [1, 2, 3]
    assert all(item["candidate_count"] == 4 for item in slate)
    assert all(1 <= item["baseline_rank"] <= 4 for item in slate)
    with bandit._connect() as conn:
        positions = conn.execute(
            "SELECT position, features FROM bandit_impressions ORDER BY position"
        ).fetchall()
    position_feature = FEATURE_NAMES.index("position")
    assert [json.loads(row["features"])[position_feature] for row in positions] == [0, 0.5, 1]
    assert bandit.status()["exposures"] == 0
    assert bandit.status()["recommendations"] == 3
    assert bandit.status()["labeled_impressions"] == 0


def test_features_cover_genre_release_decade_and_hash_stable_tags(tmp_path):
    bandit = ContextualSlateBandit(tmp_path / "bandit.sqlite3", seed=7)
    first = bandit._features(
        {
            "title": "Space Drama (1995)",
            "text": "Genres: Drama | Tags: space opera, spaceship",
        },
        0.7,
    )
    repeated = bandit._features(
        {
            "title": "Different title (1997)",
            "text": "Genres: Drama | Tags: space opera, spaceship",
        },
        0.7,
    )
    different_tags = bandit._features(
        {
            "title": "Other Drama (1995)",
            "text": "Genres: Drama | Tags: romantic comedy",
        },
        0.7,
    )

    assert len(first) == bandit.status()["feature_count"]
    assert first[FEATURE_NAMES.index("year_1990s")] == 1
    assert np.array_equal(first, repeated)
    tag_features = [
        FEATURE_NAMES.index(name) for name in FEATURE_NAMES if name.startswith("tag_hash_")
    ]
    assert first[tag_features].sum() >= 1
    assert not np.array_equal(first[tag_features], different_tags[tag_features])


def test_feedback_is_bounded_idempotent_and_updates_global_and_user_models(tmp_path):
    bandit = ContextualSlateBandit(tmp_path / "bandit.sqlite3", seed=7)
    corpus, candidates = _fixtures()
    impression = bandit.recommend("profile-1", candidates, corpus, k=1)[0]

    result = bandit.record_feedback(
        impression["impression_id"],
        "watch_complete",
        watch_pct=0.88,
        user_id="profile-1",
        doc_id=impression["doc_id"],
    )

    assert result["accepted"] is True
    assert result["reward"] == pytest.approx(0.934)
    assert 0 < result["propensity"] <= 1
    assert bandit.status()["global_observations"] == 1
    assert bandit.status()["users_with_feedback"] == 1
    duplicate = bandit.record_feedback(
        impression["impression_id"],
        "watch_complete",
        user_id="profile-1",
        doc_id=impression["doc_id"],
    )
    assert duplicate == {"accepted": False, "reason": "already_labeled"}
    assert bandit.status()["global_observations"] == 1


def test_exposure_confirmation_is_identity_checked_and_idempotent(tmp_path):
    bandit = ContextualSlateBandit(tmp_path / "bandit.sqlite3", seed=8)
    corpus, candidates = _fixtures()
    impression = bandit.recommend("profile-1", candidates, corpus, k=1)[0]

    wrong_user = bandit.mark_exposure(
        impression["impression_id"], user_id="other", doc_id=impression["doc_id"]
    )
    first = bandit.mark_exposure(
        impression["impression_id"], user_id="profile-1", doc_id=impression["doc_id"]
    )
    repeated = bandit.mark_exposure(
        impression["impression_id"], user_id="profile-1", doc_id=impression["doc_id"]
    )

    assert wrong_user["accepted"] is False
    assert first["accepted"] is True and first["new_exposure"] is True
    assert first["candidate_context"]["candidate_scores"]
    assert repeated["accepted"] is True and repeated["new_exposure"] is False
    assert bandit.status()["exposures"] == 1


def test_watch_progress_is_monotonic_and_completion_is_terminal(tmp_path):
    bandit = ContextualSlateBandit(tmp_path / "bandit.sqlite3", seed=9)
    corpus, candidates = _fixtures()
    impression = bandit.recommend("profile-1", candidates, corpus, k=1)[0]
    args = {
        "user_id": "profile-1",
        "doc_id": impression["doc_id"],
    }

    start = bandit.record_feedback(impression["impression_id"], "watch_start", **args)
    progress = bandit.record_feedback(
        impression["impression_id"], "watch_progress", watch_pct=0.2, **args
    )
    regression = bandit.record_feedback(
        impression["impression_id"], "watch_progress", watch_pct=0.1, **args
    )
    complete = bandit.record_feedback(
        impression["impression_id"], "watch_complete", watch_pct=1.0, **args
    )

    assert start["accepted"] is True
    assert progress["reward"] > start["reward"]
    assert regression["accepted"] is False
    assert complete["reward"] == 1.0
    assert bandit.record_feedback(impression["impression_id"], "skip", **args)["accepted"] is False


def test_model_learning_uses_direct_reward_likelihood_not_ips_weight(tmp_path):
    bandit = ContextualSlateBandit(tmp_path / "bandit.sqlite3", seed=2)
    corpus, candidates = _fixtures()
    impression = bandit.recommend("profile-1", candidates, corpus, k=1)[0]
    bandit.record_feedback(
        impression["impression_id"],
        "watch_complete",
        user_id="profile-1",
        doc_id=impression["doc_id"],
    )

    with bandit._connect() as conn:
        model_precision, _, _ = bandit._model(conn, "profile-1")
        stored = conn.execute(
            "SELECT features FROM bandit_impressions WHERE impression_id = ?",
            (impression["impression_id"],),
        ).fetchone()
    x = np.asarray(json.loads(stored["features"]))
    expected = bandit.prior_precision * np.eye(len(x))
    expected += np.outer(x, x) / bandit.reward_noise
    assert np.allclose(model_precision, expected)


def test_click_reward_can_be_refined_by_delayed_watch_outcome(tmp_path):
    bandit = ContextualSlateBandit(tmp_path / "bandit.sqlite3", seed=7)
    corpus, candidates = _fixtures()
    impression = bandit.recommend("profile-1", candidates, corpus, k=1)[0]
    impression_id = impression["impression_id"]

    clicked = bandit.record_feedback(
        impression_id, "click", user_id="profile-1", doc_id=impression["doc_id"]
    )
    completed = bandit.record_feedback(
        impression_id,
        "watch_complete",
        watch_pct=0.91,
        user_id="profile-1",
        doc_id=impression["doc_id"],
    )

    assert clicked["accepted"] is True
    assert completed["accepted"] is True
    assert completed["previous_reward"] == 0.25
    assert bandit.status()["global_observations"] == 1
    assert bandit.status()["mean_reward"] == pytest.approx(0.9505)


def test_policy_learns_genre_reward_and_state_survives_restart(tmp_path):
    db_path = tmp_path / "bandit.sqlite3"
    bandit = ContextualSlateBandit(db_path, seed=11)
    corpus, candidates = _fixtures()

    for _ in range(12):
        impression = bandit.recommend("profile-1", candidates, corpus, k=1)[0]
        event_type = "watch_complete" if impression["doc_id"] == "a" else "dislike"
        bandit.record_feedback(
            impression["impression_id"],
            event_type,
            user_id="profile-1",
            doc_id=impression["doc_id"],
        )

    restarted = ContextualSlateBandit(db_path, seed=11)
    action = restarted.predict_reward("profile-1", corpus["a"])
    romance = restarted.predict_reward("profile-1", corpus["c"])

    assert restarted.status()["global_observations"] == 12
    assert action > romance


def test_unknown_or_mismatched_impressions_are_not_learned(tmp_path):
    bandit = ContextualSlateBandit(tmp_path / "bandit.sqlite3", seed=7)
    corpus, candidates = _fixtures()
    impression = bandit.recommend("profile-1", candidates, corpus, k=1)[0]

    result = bandit.record_feedback(
        impression["impression_id"],
        "click",
        user_id="different-profile",
        doc_id=impression["doc_id"],
    )

    assert result == {"accepted": False, "reason": "unknown_impression"}
    assert bandit.status()["global_observations"] == 0

    wrong_item = bandit.record_feedback(
        impression["impression_id"], "click", user_id="profile-1", doc_id="different-item"
    )
    assert wrong_item == {"accepted": False, "reason": "unknown_impression"}
    assert bandit.status()["global_observations"] == 0


def test_invalid_reward_and_empty_candidate_list(tmp_path):
    bandit = ContextualSlateBandit(tmp_path / "bandit.sqlite3", seed=7)
    corpus, candidates = _fixtures()

    assert bandit.recommend("profile-1", [], corpus, k=3) == []
    impression = bandit.recommend("profile-1", candidates, corpus, k=1)[0]
    with pytest.raises(ValueError, match="event_type"):
        bandit.record_feedback(
            impression["impression_id"], "unknown", user_id="profile-1", doc_id=impression["doc_id"]
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": float("nan")},
        {"exploration_scale": float("inf")},
        {"exploration_scale": 100},
        {"temperature": 20},
    ],
)
def test_unsafe_model_configuration_is_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ContextualSlateBandit(tmp_path / "bandit.sqlite3", **kwargs)


def test_status_exposes_recent_feedback_health(tmp_path):
    bandit = ContextualSlateBandit(tmp_path / "bandit.sqlite3", seed=4)
    corpus, candidates = _fixtures()
    impression = bandit.recommend("profile-1", candidates, corpus, k=1)[0]
    exposure = bandit.mark_exposure(
        impression["impression_id"], user_id="profile-1", doc_id=impression["doc_id"]
    )
    assert exposure["accepted"] is True
    assert exposure["new_exposure"] is True
    bandit.record_feedback(
        impression["impression_id"], "click", user_id="profile-1", doc_id=impression["doc_id"]
    )

    status = bandit.status()
    assert status["pending_feedback"] == 0
    assert status["last_7_days"]["impressions"] == 1
    assert status["last_7_days"]["labeled_impressions"] == 1
    assert status["last_7_days"]["mean_reward"] == 0.25
    assert status["last_7_days"]["slates"] == 1
    assert "sampled Thompson parameter" in status["propensity_semantics"]
    assert status["off_policy_evaluation"] != "unavailable"


def test_old_impression_journal_gets_additive_slate_migration(tmp_path):
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE bandit_impressions (
            impression_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, doc_id TEXT NOT NULL,
            features TEXT NOT NULL, propensity REAL NOT NULL, created_at REAL NOT NULL,
            event_type TEXT, reward REAL, updated_at REAL
        )""")
        conn.execute("""INSERT INTO bandit_impressions
            VALUES ('legacy-id', 'profile-1', 'a', '[0]', 0.5, 1.0, NULL, NULL, NULL)""")

    ContextualSlateBandit(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(bandit_impressions)")}
        legacy = conn.execute(
            "SELECT impression_id, slate_id, position FROM bandit_impressions"
        ).fetchone()
    assert {"slate_id", "position", "candidate_count", "baseline_rank", "base_score"} <= columns
    assert legacy == ("legacy-id", "", 0)


def test_legacy_genre_only_model_is_expanded_without_losing_learned_weights(tmp_path):
    db_path = tmp_path / "legacy-model.sqlite3"
    precision = (2.0 * np.eye(LEGACY_FEATURE_COUNT)).tolist()
    response = np.zeros(LEGACY_FEATURE_COUNT)
    response[2] = 1.25
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE bandit_models (
            user_id TEXT PRIMARY KEY, precision TEXT NOT NULL, response TEXT NOT NULL,
            observations INTEGER NOT NULL DEFAULT 0, reward_sum REAL NOT NULL DEFAULT 0
        )""")
        conn.execute(
            "INSERT INTO bandit_models VALUES (?, ?, ?, ?, ?)",
            ("profile-1", json.dumps(precision), json.dumps(response.tolist()), 4, 2.0),
        )

    bandit = ContextualSlateBandit(db_path)
    with bandit._connect() as conn:
        migrated_precision, migrated_response, observations = bandit._model(conn, "profile-1")

    assert migrated_precision.shape == (N_FEATURES, N_FEATURES)
    assert migrated_response.shape == (N_FEATURES,)
    assert migrated_response[2] == 1.25
    assert observations == 4


def test_previous_tag_model_schema_is_expanded_without_losing_learned_weights(tmp_path):
    db_path = tmp_path / "previous-model.sqlite3"
    precision = (2.0 * np.eye(PREVIOUS_FEATURE_COUNT)).tolist()
    response = np.zeros(PREVIOUS_FEATURE_COUNT)
    response[-1] = 0.75
    with sqlite3.connect(db_path) as conn:
        conn.execute("""CREATE TABLE bandit_models (
            user_id TEXT PRIMARY KEY, precision TEXT NOT NULL, response TEXT NOT NULL,
            observations INTEGER NOT NULL DEFAULT 0, reward_sum REAL NOT NULL DEFAULT 0
        )""")
        conn.execute(
            "INSERT INTO bandit_models VALUES (?, ?, ?, ?, ?)",
            ("profile-1", json.dumps(precision), json.dumps(response.tolist()), 2, 1.0),
        )

    bandit = ContextualSlateBandit(db_path)
    with bandit._connect() as conn:
        migrated_precision, migrated_response, _ = bandit._model(conn, "profile-1")

    assert migrated_precision.shape == (N_FEATURES, N_FEATURES)
    assert migrated_response[-2] == 0.75


def test_impression_journal_respects_configured_capacity(tmp_path):
    bandit = ContextualSlateBandit(tmp_path / "bandit.sqlite3", seed=7, max_impressions=3)
    corpus, candidates = _fixtures()
    bandit.recommend("profile-1", candidates, corpus, k=3)
    bandit.recommend("profile-1", candidates, corpus, k=2)

    assert bandit.status()["recommendations"] == 3
    assert bandit.status()["exposures"] == 0
