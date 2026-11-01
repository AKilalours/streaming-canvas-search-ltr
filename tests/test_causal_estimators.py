from __future__ import annotations

import random

import pytest

from causal.traffic_simulator import TrafficSimulator, ipw_estimate
from causal.uplift import IncrementalityScorer, LoggedEvent, OffPolicyEvaluator


def test_uplift_cold_start_is_deterministic_zero_and_unconfident():
    scorer = IncrementalityScorer()

    first = scorer.score("movie")
    second = scorer.score("movie")

    assert first == second
    assert first.incrementality_score == 0
    assert first.confident is False
    assert first.method == "clipped_ipw_hajek"


def test_uplift_uses_clipped_hajek_arm_estimates_and_effective_sample_size():
    scorer = IncrementalityScorer()
    for _ in range(10):
        scorer.update(LoggedEvent("movie", True, True, 0.5))
        scorer.update(LoggedEvent("movie", False, False, 0.5))

    result = scorer.score("movie")

    assert result.n_treated == result.n_control == 10
    assert result.effective_sample_size_treated == 10
    assert result.effective_sample_size_control == 10
    assert result.incrementality_score == pytest.approx(0.8)
    assert result.confident is True


@pytest.mark.parametrize("propensity", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_uplift_rejects_invalid_propensities(propensity):
    scorer = IncrementalityScorer()
    with pytest.raises(ValueError, match="propensity"):
        scorer.update(LoggedEvent("movie", True, True, propensity))


def test_uplift_marks_extreme_propensity_overlap_as_unconfident():
    scorer = IncrementalityScorer()
    for _ in range(12):
        scorer.update(LoggedEvent("movie", True, True, 0.01))
        scorer.update(LoggedEvent("movie", False, False, 0.5))

    result = scorer.score("movie")
    assert result.confident is False
    assert result.method == "clipped_ipw_hajek"


def test_snips_evaluator_reports_effective_sample_size_and_gate():
    events = [LoggedEvent("a", True, True, 0.5) for _ in range(10)] + [
        LoggedEvent("b", True, False, 0.5) for _ in range(10)
    ]
    evaluator = OffPolicyEvaluator(min_lift=0.0, min_effective_sample_size=10)

    result = evaluator.evaluate({"a": 0.0, "b": 0.0}, events)

    assert result.estimator == "snips"
    assert result.estimated_reward == pytest.approx(0.5)
    assert result.effective_sample_size == 20
    assert result.weight_clipping_fraction == 0
    assert result.passed is True


def test_snips_rejects_invalid_logged_probability():
    evaluator = OffPolicyEvaluator()
    with pytest.raises(ValueError, match="propensities"):
        evaluator.evaluate({"a": 1.0}, [LoggedEvent("a", True, True, 0.0)])


def test_legacy_ipw_helper_uses_snips_and_requires_real_propensity():
    result = ipw_estimate(
        [
            {"doc_id": "a", "shown": True, "watched": True, "propensity": 0.5},
            {"doc_id": "b", "shown": True, "watched": False, "propensity": 0.5},
        ],
        {"a": 0.0, "b": 0.0},
    )

    assert result["method"] == "snips"
    assert result["ipw_estimate"] == pytest.approx(0.5)
    assert result["deployable"] is False
    with pytest.raises(ValueError, match="exact action propensity"):
        ipw_estimate([{"doc_id": "a", "reward": 1}], {"a": 1.0})


def test_synthetic_traffic_simulation_is_reproducible_without_mutating_global_rng():
    random.seed(19)
    expected_next = random.random()
    random.seed(19)
    first = TrafficSimulator(n_users=10, seed=7)
    first_events = first.simulate_policy("test", [{"doc_id": "1", "genres": "action"}], n_days=2)
    actual_next = random.random()
    second = TrafficSimulator(n_users=10, seed=7)
    second_events = second.simulate_policy("test", [{"doc_id": "1", "genres": "action"}], n_days=2)

    assert actual_next == expected_next
    assert first_events == second_events
    assert all(0 <= user.completion_rate <= 1 for user in first.users)
