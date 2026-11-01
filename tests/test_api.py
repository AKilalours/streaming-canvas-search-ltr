import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
import causal.propensity_logger as propensity_module
from app.main import app
from exploration.online_bandit import ContextualSlateBandit


def test_health(tmp_path, monkeypatch):
    monkeypatch.setenv("STREAMLENS_BANDIT_DB", str(tmp_path / "bandit.sqlite3"))
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert "ok" in r.json()


def test_search_returns_debug_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("STREAMLENS_BANDIT_DB", str(tmp_path / "bandit.sqlite3"))
    with TestClient(app) as c:
        r = c.post("/search", json={"query": "test query", "k": 5, "debug": True})
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "test query"
        assert data["k"] == 5
        assert "hits" in data
        assert data["hits"]
        assert "score_breakdown" in data["hits"][0]


def test_contextual_bandit_feed_connects_exposure_outcome_and_live_ope(tmp_path, monkeypatch):
    monkeypatch.setenv("STREAMLENS_BANDIT_DB", str(tmp_path / "bandit.sqlite3"))
    monkeypatch.setattr(propensity_module, "LOG_DIR", tmp_path / "propensity")
    monkeypatch.setattr(propensity_module.PropensityLogger, "_connect_redis", lambda _self: None)
    monkeypatch.setattr(main_module, "_PROP_LOGGER", propensity_module.PropensityLogger())
    monkeypatch.setattr(main_module, "PROP_LOGGER_AVAILABLE", True)
    with TestClient(app) as c:
        status = c.get("/rl/status")
        assert status.status_code == 200
        assert status.json()["enabled"] is True

        response = c.get("/feed", params={"profile": "chrisen", "rows": 1, "k": 6})
        assert response.status_code == 200
        items = response.json()["rows"][0]["items"]
        assert items
        impression = items[0]["impression_id"]
        assert 0 < items[0]["propensity"] <= 1
        assert items[0]["slate_id"]
        assert items[0]["position"] == 1

        premature = c.post(
            "/rl/feedback",
            json={
                "impression_id": impression,
                "doc_id": items[0]["doc_id"],
                "user_id": "chrisen",
                "event_type": "click",
            },
        )
        assert premature.status_code == 409
        assert premature.json()["detail"] == "exposure_not_confirmed"

        exposure = c.post(
            "/rl/exposure",
            json={
                "impression_id": impression,
                "doc_id": items[0]["doc_id"],
                "user_id": "chrisen",
            },
        )
        assert exposure.status_code == 200
        assert exposure.json()["ope_eligible"] is True
        assert exposure.json()["propensity"] == items[0]["propensity"]

        outcome = c.post(
            "/rl/feedback",
            json={
                "impression_id": impression,
                "doc_id": items[0]["doc_id"],
                "user_id": "chrisen",
                "event_type": "watch_complete",
                "watch_pct": 0.92,
            },
        )
        assert outcome.status_code == 200
        assert outcome.json()["reward"] == pytest.approx(0.956)
        status = c.get("/rl/status").json()
        assert status["global_observations"] == 1
        assert status["last_7_days"]["labeled_impressions"] == 1
        ope = c.get("/uplift/ope", params={"n": 100}).json()
        assert ope["data_source"] == "live_randomized_feed_impressions"
        assert ope["synthetic_only"] is False
        assert ope["deployment_eligible"] is False
        assert ope["n_samples"] == 1
        duplicate = c.post(
            "/rl/feedback",
            json={
                "impression_id": impression,
                "doc_id": items[0]["doc_id"],
                "user_id": "chrisen",
                "event_type": "watch_complete",
            },
        )
        assert duplicate.status_code == 409

        forged = c.post(
            "/rl/feedback",
            json={
                "impression_id": items[1]["impression_id"],
                "doc_id": "not-the-served-item",
                "user_id": "chrisen",
                "event_type": "dislike",
            },
        )
        assert forged.status_code == 409

        missing_doc = c.post(
            "/rl/feedback",
            json={
                "impression_id": items[2]["impression_id"],
                "user_id": "chrisen",
                "event_type": "dislike",
            },
        )
        assert missing_doc.status_code == 422


def test_uplift_ope_route_never_fabricates_live_outcomes(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "CAUSAL_AVAILABLE", True)
    monkeypatch.setattr(propensity_module, "LOG_DIR", tmp_path / "empty")
    monkeypatch.setattr(propensity_module.PropensityLogger, "_connect_redis", lambda _self: None)
    monkeypatch.setattr(main_module, "_PROP_LOGGER", propensity_module.PropensityLogger())
    monkeypatch.setattr(main_module, "PROP_LOGGER_AVAILABLE", True)
    result = main_module.ope_evaluate(n=10)

    assert result["synthetic_only"] is False
    assert result["data_source"] == "live_randomized_feed_impressions"
    assert result["status"] == "insufficient_data"
    assert result["n_samples"] == 0
    assert result["deployment_eligible"] is False


def test_feed_falls_back_to_relevance_ranker_if_bandit_storage_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("STREAMLENS_BANDIT_DB", str(tmp_path / "bandit.sqlite3"))

    def fail_recommend(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ContextualSlateBandit, "recommend", fail_recommend)
    with TestClient(app) as client:
        response = client.get("/feed", params={"profile": "chrisen", "rows": 1, "k": 6})

    assert response.status_code == 200
    assert response.json()["policy"] == "relevance_ranker_fallback"
    row = response.json()["rows"][0]
    assert row["policy"] == "relevance_ranker_fallback"
    assert row["items"]
    assert "impression_id" not in row["items"][0]


def test_impression_endpoint_does_not_invent_propensity_from_rank_score(tmp_path, monkeypatch):
    monkeypatch.setattr(propensity_module, "LOG_DIR", tmp_path)
    monkeypatch.setattr(propensity_module.PropensityLogger, "_connect_redis", lambda _self: None)
    logger = propensity_module.PropensityLogger()
    monkeypatch.setattr(main_module, "_PROP_LOGGER", logger)
    monkeypatch.setattr(main_module, "PROP_LOGGER_AVAILABLE", True)
    with TestClient(app) as client:
        response = client.post(
            "/impression/log",
            params={
                "user_id": "u1",
                "doc_id": "d1",
                "ltr_score": 0.9,
            },
        )
        event = response.json()
        assert response.status_code == 200
        assert event["propensity"] is None
        assert event["ope_eligible"] is False

        outcome = client.post(
            "/impression/outcome",
            params={
                "event_id": event["event_id"],
                "watch_pct": 0.8,
            },
        )
        stats = client.get("/impression/stats").json()

    assert outcome.status_code == 200
    assert stats["ope_available"] is False
    assert stats["n_ope_eligible_outcomes"] == 0


def test_uplift_score_api_uses_live_events_without_claiming_missing_control_lift(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(main_module, "CAUSAL_AVAILABLE", True)
    monkeypatch.setattr(propensity_module, "LOG_DIR", tmp_path / "propensity")
    monkeypatch.setattr(propensity_module.PropensityLogger, "_connect_redis", lambda _self: None)
    logger = propensity_module.PropensityLogger()
    event = logger.log_impression(
        user_id="u1",
        doc_id="movie-1",
        ltr_score=0.9,
        propensity=0.5,
        randomized=True,
        serving_verified=True,
        context={
            "target_scores": {"movie-1": 0.9, "movie-2": 0.5},
            "behavior_probabilities": {"movie-1": 0.5, "movie-2": 0.5},
        },
    )
    logger.record_outcome(event.event_id, watch_pct=0.8, clicked=True)
    monkeypatch.setattr(main_module, "_PROP_LOGGER", logger)
    monkeypatch.setattr(main_module, "PROP_LOGGER_AVAILABLE", True)

    result = main_module.uplift_score(doc_id="movie-1")

    assert result["data_source"] == "live_randomized_feed_logs"
    assert result["status"] == "observed_outcomes_only"
    assert result["confident"] is False
    assert result["incrementality_score"] is None
    assert result["p_watch_shown"] == 0.8
    assert result["n_control"] == 0
    assert "control outcomes are unavailable" in result["note"]


def test_project_summary_uses_saved_reports_and_does_not_claim_gates_passed(tmp_path, monkeypatch):
    latest = tmp_path / "reports" / "latest"
    latest.mkdir(parents=True)
    (latest / "metrics.json").write_text(
        json.dumps({"methods": [{"method": "hybrid_ltr", "ndcg@10": 0.81, "recall@100": 0.92}]})
    )
    (latest / "latency.json").write_text(json.dumps({"p95_ms": 80, "p99_ms": 120}))
    monkeypatch.chdir(tmp_path)

    result = main_module.project_summary()

    assert result["all_eval_gates_passing"] is None
    assert result["key_metrics"]["ndcg_at_10_ltr"] == 0.81
    assert result["key_metrics"]["recall_at_100"] == 0.92
    assert result["key_metrics"]["p99_latency_ms"] == 120
    assert result["key_metrics"]["mrr_at_10"] is None


def test_comprehensive_eval_reports_artifact_values_without_synthesizing_metrics(
    tmp_path, monkeypatch
):
    latest = tmp_path / "reports" / "latest"
    latest.mkdir(parents=True)
    (latest / "metrics.json").write_text(
        json.dumps(
            {
                "k": 10,
                "split": "test",
                "diagnostics": {"num_queries": 12, "has_ltr": True},
                "methods": [
                    {"method": "hybrid", "ndcg@10": 0.30, "num_queries": 12},
                    {
                        "method": "hybrid_ltr",
                        "ndcg@10": 0.31,
                        "recall@100": 0.42,
                        "num_queries": 12,
                    },
                ],
            }
        )
    )
    monkeypatch.chdir(tmp_path)

    result = main_module.eval_comprehensive()

    assert result["status"] == "measured_artifact"
    assert result["ablation"]["ltr_final"] == 0.31
    assert result["retrieval"]["recall@100"] == 0.42
    assert result["gates"]["recall@100"]["passed"] is False
    assert result["latency"]["status"] == "unavailable"
    assert result["page_quality"]["status"] == "not_measured_in_artifact"
