"""
StreamLens model promotion pipeline, Airflow DAG.

Every task runs the real pipeline code in src/pipelines/promotion.py:

    validate_data ─> train_candidate ─> evaluate ─┬─> run_gates ─> decide ─┬─> promote_model
                                                  │                        └─> block_promotion (fails the run)
                                                  └─> drift_check

* validate_data   schema, id integrity, split consistency, query overlap (pipelines/data_quality.py)
* train_candidate LightGBM LambdaRank on the train split, written to artifacts/ltr/candidates/<run_id>/
                  (production model is not touched)
* evaluate        candidate on val + test, current production model on val, same code and split
* run_gates       9 gates computed from this run's measured metrics (pipelines/promotion_gates.py)
* decide          branches on reports/runs/<run_id>/gates.json
* promote_model   archive current model, atomic swap, append to artifacts/ltr/registry.jsonl
* block_promotion fails the DAG run so a blocked candidate is visible in the UI and alerts
* drift_check     same production model should score the same as last time; fails if it moved

Environment:
    STREAMLENS_HOME    repo root (default: parent of this file's directory)
    STREAMLENS_PYTHON  interpreter with the project's ML dependencies (default: python)
Airflow's own environment does not need torch/lightgbm; tasks shell out to STREAMLENS_PYTHON.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

STREAMLENS_HOME = os.environ.get("STREAMLENS_HOME", str(Path(__file__).resolve().parents[1]))
STREAMLENS_PYTHON = os.environ.get("STREAMLENS_PYTHON", "python")
PROMOTION_CONFIG = os.environ.get("STREAMLENS_PROMOTION_CONFIG", "configs/promotion.yaml")


def _step(step: str) -> str:
    # run_id is passed through an env var, not interpolated into the shell string
    return (
        f'cd "{STREAMLENS_HOME}" && PYTHONPATH=src "{STREAMLENS_PYTHON}" '
        f'-m pipelines.promotion {step} --run-id "$RUN_ID" --config "{PROMOTION_CONFIG}"'
    )


def _gates_path(run_id: str) -> Path:
    src = str(Path(STREAMLENS_HOME) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from pipelines.promotion import load_cfg, run_dir, sanitize_run_id

    cfg = load_cfg(Path(STREAMLENS_HOME) / PROMOTION_CONFIG)
    return Path(STREAMLENS_HOME) / run_dir(cfg, sanitize_run_id(run_id)) / "gates.json"


def decide(run_id: str, **_: object) -> str:
    gates = json.loads(_gates_path(run_id).read_text(encoding="utf-8"))
    failed = [g["name"] for g in gates["gates"] if not g["passed"]]
    print(f"gates {gates['passed']}/{gates['total']} passed; failed={failed}")
    return "promote_model" if gates["all_passed"] and not failed else "block_promotion"


def block(run_id: str, **_: object) -> None:
    gates = json.loads(_gates_path(run_id).read_text(encoding="utf-8"))
    failed = [f"{g['name']} ({g['detail']})" for g in gates["gates"] if not g["passed"]]
    raise AirflowFailException("promotion blocked by quality gates: " + "; ".join(failed))


default_args = {
    "owner": "streamlens",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="streamlens_ml_pipeline",
    description="validate -> train candidate -> evaluate -> 9 gates -> promote or block; drift check",
    default_args=default_args,
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,  # two runs must never promote concurrently
    tags=["streamlens", "ltr", "promotion"],
) as dag:
    env = {"RUN_ID": "{{ run_id }}"}

    validate_data = BashOperator(
        task_id="validate_data", bash_command=_step("validate"), env=env, append_env=True,
        execution_timeout=timedelta(minutes=10), retries=0,
    )
    train_candidate = BashOperator(
        task_id="train_candidate", bash_command=_step("train"), env=env, append_env=True,
        execution_timeout=timedelta(hours=1),
    )
    evaluate = BashOperator(
        task_id="evaluate", bash_command=_step("evaluate"), env=env, append_env=True,
        execution_timeout=timedelta(hours=1),
    )
    run_gates = BashOperator(
        task_id="run_gates", bash_command=_step("gate"), env=env, append_env=True,
        execution_timeout=timedelta(minutes=10), retries=0,
    )
    decide_branch = BranchPythonOperator(
        task_id="decide", python_callable=decide, op_kwargs={"run_id": "{{ run_id }}"},
    )
    promote_model = BashOperator(
        task_id="promote_model", bash_command=_step("promote"), env=env, append_env=True,
        execution_timeout=timedelta(minutes=10), retries=0,
    )
    block_promotion = PythonOperator(
        task_id="block_promotion", python_callable=block, op_kwargs={"run_id": "{{ run_id }}"},
        retries=0,
    )
    drift_check = BashOperator(
        task_id="drift_check", bash_command=_step("drift"), env=env, append_env=True,
        execution_timeout=timedelta(minutes=10), retries=0,
    )

    validate_data >> train_candidate >> evaluate
    evaluate >> run_gates >> decide_branch >> [promote_model, block_promotion]
    evaluate >> drift_check
