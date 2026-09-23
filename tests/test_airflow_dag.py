"""
Structural tests for the Airflow DAG. Skipped when Airflow is not installed
(the project venv does not include it; run with the Airflow environment):

    AIRFLOW_HOME=$(mktemp -d) <airflow-venv>/bin/python -m pytest tests/test_airflow_dag.py
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

pytest.importorskip("airflow")

ROOT = Path(__file__).resolve().parents[1]
DAG_DIR = ROOT / "flows"


@pytest.fixture(scope="module")
def dagbag():
    from airflow.models import DagBag

    return DagBag(dag_folder=str(DAG_DIR / "streamlens_airflow_dag.py"), include_examples=False)


def test_dag_imports_without_errors(dagbag):
    assert dagbag.import_errors == {}
    assert "streamlens_ml_pipeline" in dagbag.dags


def test_dag_structure(dagbag):
    dag = dagbag.dags["streamlens_ml_pipeline"]
    assert dag.max_active_runs == 1
    assert not dag.catchup
    t = dag.task_dict
    assert set(t) == {"validate_data", "train_candidate", "evaluate", "run_gates", "decide",
                      "promote_model", "block_promotion", "drift_check"}
    assert t["validate_data"].downstream_task_ids == {"train_candidate"}
    assert t["train_candidate"].downstream_task_ids == {"evaluate"}
    assert t["evaluate"].downstream_task_ids == {"run_gates", "drift_check"}
    assert t["decide"].downstream_task_ids == {"promote_model", "block_promotion"}
    # promotion can only be reached through the gate decision
    assert t["promote_model"].upstream_task_ids == {"decide"}


def _dag_module(monkeypatch, home: Path):
    monkeypatch.setenv("STREAMLENS_HOME", str(home))
    import streamlens_airflow_dag as mod  # noqa: PLC0415

    return importlib.reload(mod)


def _repo_with_gates(tmp_path: Path, all_passed: bool) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "promotion.yaml").write_text((ROOT / "configs" / "promotion.yaml").read_text())
    (tmp_path / "src").symlink_to(ROOT / "src")
    rd = tmp_path / "reports" / "runs" / "manual__x"
    rd.mkdir(parents=True)
    gates = [{"name": "ndcg10_no_regression", "passed": all_passed, "detail": "d"}]
    (rd / "gates.json").write_text(json.dumps(
        {"all_passed": all_passed, "passed": int(all_passed), "total": 1, "gates": gates}))
    return tmp_path


def test_decide_branches_on_measured_gates(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(DAG_DIR))
    mod = _dag_module(monkeypatch, _repo_with_gates(tmp_path, all_passed=True))
    assert mod.decide("manual__x") == "promote_model"


def test_decide_blocks_and_block_task_fails(tmp_path, monkeypatch):
    from airflow.exceptions import AirflowFailException

    monkeypatch.syspath_prepend(str(DAG_DIR))
    mod = _dag_module(monkeypatch, _repo_with_gates(tmp_path, all_passed=False))
    assert mod.decide("manual__x") == "block_promotion"
    with pytest.raises(AirflowFailException, match="ndcg10_no_regression"):
        mod.block("manual__x")
