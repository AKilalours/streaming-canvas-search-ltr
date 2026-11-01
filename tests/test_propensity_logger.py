from __future__ import annotations

import pytest

from causal import propensity_logger


def _logger(tmp_path, monkeypatch):
    monkeypatch.setattr(propensity_logger, "LOG_DIR", tmp_path)
    monkeypatch.setattr(propensity_logger.PropensityLogger, "_connect_redis", lambda _self: None)
    return propensity_logger.PropensityLogger()


def test_deterministic_rank_scores_do_not_create_fake_propensities(tmp_path, monkeypatch):
    logger = _logger(tmp_path, monkeypatch)

    event = logger.log_impression(
        user_id="u1", doc_id="d1", ltr_score=0.9, all_scores=[0.9], position=0
    )

    assert event.propensity is None
    assert event.ope_eligible is False
    assert logger.load_events_for_ope() == []


def test_ope_log_requires_exact_randomized_propensity_and_mature_outcome(tmp_path, monkeypatch):
    logger = _logger(tmp_path, monkeypatch)
    event = logger.log_impression(
        user_id="u1", doc_id="d1", ltr_score=0.9, propensity=0.25,
        randomized=True, serving_verified=True, position=0, event_id="bandit-impression-1",
        context={
            "target_scores": {"d1": 0.9, "d2": 0.5},
            "behavior_probabilities": {"d1": 0.25, "d2": 0.75},
        },
    )

    assert event.ope_eligible is True
    assert logger.load_events_for_ope() == []
    logger.record_outcome(event.event_id, watch_pct=0.75, clicked=True)

    rows = logger.load_events_for_ope()
    assert len(rows) == 1
    assert rows[0]["propensity"] == 0.25
    assert rows[0]["watched"] == 0.75
    assert rows[0]["context"]["target_scores"]["d1"] == 0.9
    assert rows[0]["event_id"] == "bandit-impression-1"
    assert logger.stats()["n_ope_eligible_impressions"] == 1


def test_non_randomized_probability_is_not_ope_eligible(tmp_path, monkeypatch):
    logger = _logger(tmp_path, monkeypatch)
    event = logger.log_impression(
        user_id="u1", doc_id="d1", ltr_score=0.9, propensity=1.0,
        randomized=False,
    )
    logger.record_outcome(event.event_id, watch_pct=0.75)

    assert event.ope_eligible is False
    assert logger.load_events_for_ope() == []


def test_caller_supplied_random_flag_cannot_make_event_ope_eligible(tmp_path, monkeypatch):
    logger = _logger(tmp_path, monkeypatch)
    event = logger.log_impression(
        user_id="u1", doc_id="d1", ltr_score=0.9, propensity=0.25,
        randomized=True, context={"target_scores": {"d1": 1.0},
                                  "behavior_probabilities": {"d1": 0.25}},
    )
    logger.record_outcome(event.event_id, watch_pct=0.75)

    assert event.ope_eligible is False
    assert logger.load_events_for_ope() == []


def test_invalid_propensity_and_outcome_are_rejected(tmp_path, monkeypatch):
    logger = _logger(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="selection_probability"):
        logger.log_impression(
            user_id="u1", doc_id="d1", ltr_score=0.9, propensity=0.0
        )
    with pytest.raises(ValueError, match="watch_pct"):
        logger.record_outcome("unknown", watch_pct=1.5)
