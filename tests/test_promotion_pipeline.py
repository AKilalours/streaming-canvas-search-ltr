"""
Tests for the promotion pipeline (src/pipelines/*). Fast: no models are trained,
metrics are written as small JSON fixtures.

The first group pins the defects found in the original Airflow DAG:
  * validate_data imported a function that did not exist (ImportError at runtime)
  * train_ltr produced nDCG from random.uniform instead of training
  * the gate checked one hard-coded threshold while docs claimed 9 gates
"""
from __future__ import annotations

import ast
import json
import pickle
from pathlib import Path

import pytest
import yaml

from pipelines import promotion
from pipelines.data_quality import check_data_quality
from pipelines.promotion_gates import evaluate_gates, summarize

ROOT = Path(__file__).resolve().parents[1]
DAG_FILE = ROOT / "flows" / "streamlens_airflow_dag.py"

THRESHOLDS = {
    "min_ndcg10": 0.70,
    "min_lift_over_first_stage": 0.05,
    "max_ndcg10_drop": 0.01,
    "max_recall100_drop": 0.01,
    "max_map10_drop": 0.01,
}
FEATS = ["a", "b", "c"]


# ── regression tests for the original DAG defects ───────────────────────────
def test_dag_has_no_simulated_or_random_steps():
    src = DAG_FILE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "random" not in imported
    assert "random.uniform" not in src
    assert "simulate" not in src.lower()


def test_every_dag_step_calls_real_pipeline_code():
    src = DAG_FILE.read_text(encoding="utf-8")
    for step in ("validate", "train", "evaluate", "gate", "promote", "drift"):
        assert f'_step("{step}")' in src, step
        assert step in promotion.STEPS


def test_data_validation_function_exists_and_is_importable():
    # the original DAG did: from pipelines.gates import check_data_quality -> ImportError
    from pipelines.data_quality import check_data_quality as f  # noqa: F401

    assert callable(f)


def test_gate_count_matches_documentation():
    res = evaluate_gates(
        data_report={"passed": True, "errors": []},
        challenger=_metrics(0.95),
        champion=_metrics(0.95),
        candidate_model_path="cand.pkl",
        candidate_feature_names=FEATS,
        serving_feature_names=FEATS,
        expected_num_queries=150,
        thresholds=THRESHOLDS,
    )
    assert len(res) == 9
    assert summarize(res)["all_passed"]


# ── fixtures ────────────────────────────────────────────────────────────────
def _metrics(ltr: float, *, recall: float = 0.40, map10: float = 0.15, nq: int = 150,
             ltr_path: str = "cand.pkl", has_ltr: bool = True, first_stage: float = 0.57) -> dict:
    def m(name, v):
        return {"method": name, "ndcg@10": v, "recall@100": recall, "map@10": map10, "num_queries": nq}

    return {
        "diagnostics": {"has_ltr": has_ltr, "ltr_path": ltr_path},
        "methods": [m("bm25", first_stage), m("dense", first_stage - 0.05),
                    m("hybrid", first_stage), m("hybrid_ltr", ltr)],
    }


def _gates(**kw) -> dict:
    args = dict(
        data_report={"passed": True, "errors": []},
        challenger=_metrics(0.95),
        champion=_metrics(0.95),
        candidate_model_path="cand.pkl",
        candidate_feature_names=FEATS,
        serving_feature_names=FEATS,
        expected_num_queries=150,
        thresholds=THRESHOLDS,
    )
    args.update(kw)
    s = summarize(evaluate_gates(**args))
    return {g["name"]: g["passed"] for g in s["gates"]} | {"_all": s["all_passed"]}


def _write_dataset(root: Path, *, n_docs: int = 20, bad_doc_in_qrels: bool = False,
                   drop_doc_in: str | None = None) -> Path:
    for split in ("train", "val", "test"):
        d = root / split
        d.mkdir(parents=True)
        docs = [{"doc_id": str(i), "title": f"t{i}", "text": f"Title: t{i}"} for i in range(n_docs)]
        if drop_doc_in == split:
            docs = docs[:-1]
        (d / "corpus.jsonl").write_text("".join(json.dumps(x) + "\n" for x in docs))
        qs = [{"query_id": f"{split}_q{i}", "text": f"{split} query {i}"} for i in range(5)]
        (d / "queries.jsonl").write_text("".join(json.dumps(x) + "\n" for x in qs))
        qrels = {q["query_id"]: {"0": 1, "1": 2} for q in qs}
        if bad_doc_in_qrels and split == "test":
            qrels["test_q0"]["does_not_exist"] = 1
        (d / "qrels.json").write_text(json.dumps(qrels))
    return root


# ── data quality ────────────────────────────────────────────────────────────
def test_data_quality_passes_on_valid_dataset(tmp_path):
    rep = check_data_quality(_write_dataset(tmp_path), min_docs=10)
    assert rep.passed, rep.errors
    assert len(rep.fingerprint) == 64


def test_data_quality_fails_on_qrels_doc_not_in_corpus(tmp_path):
    rep = check_data_quality(_write_dataset(tmp_path, bad_doc_in_qrels=True), min_docs=10)
    assert not rep.passed
    assert rep.checks["qrels_reference_known_docs"] is False


def test_data_quality_fails_when_corpus_differs_between_splits(tmp_path):
    rep = check_data_quality(_write_dataset(tmp_path, drop_doc_in="val"), min_docs=10)
    assert rep.checks["corpus_consistent_across_splits"] is False


def test_data_quality_fails_on_missing_file(tmp_path):
    root = _write_dataset(tmp_path)
    (root / "val" / "qrels.json").unlink()
    rep = check_data_quality(root, min_docs=10)
    assert not rep.passed and rep.checks["files_present"] is False


def test_data_quality_fingerprint_changes_with_data(tmp_path):
    root = _write_dataset(tmp_path)
    a = check_data_quality(root, min_docs=10).fingerprint
    with (root / "test" / "queries.jsonl").open("a") as f:
        f.write(json.dumps({"query_id": "test_q9", "text": "new"}) + "\n")
    qr = json.loads((root / "test" / "qrels.json").read_text())
    qr["test_q9"] = {"0": 1}
    (root / "test" / "qrels.json").write_text(json.dumps(qr))
    assert check_data_quality(root, min_docs=10).fingerprint != a


# ── gates ───────────────────────────────────────────────────────────────────
def test_regression_vs_champion_blocks():
    g = _gates(challenger=_metrics(0.90), champion=_metrics(0.95))
    assert g["ndcg10_no_regression"] is False and g["_all"] is False


def test_small_noise_within_tolerance_passes():
    g = _gates(challenger=_metrics(0.945), champion=_metrics(0.95))
    assert g["_all"]


def test_small_drops_cannot_ratchet_quality_down():
    # each step is within 0.01 of the current champion, but far below the best ever promoted
    g = _gates(challenger=_metrics(0.935), champion=_metrics(0.94), best_promoted_ndcg10=0.95)
    assert g["ndcg10_no_regression"] is False
    assert _gates(challenger=_metrics(0.945), champion=_metrics(0.94), best_promoted_ndcg10=0.95)["_all"]


def test_eval_silently_using_another_model_blocks():
    # evaluate.py falls back to artifacts/ltr/<dataset>_ltr.pkl when the path is missing
    g = _gates(challenger=_metrics(0.95, ltr_path="artifacts/ltr/movielens_ltr.pkl"))
    assert g["candidate_model_was_evaluated"] is False


def test_missing_ltr_blocks():
    g = _gates(challenger=_metrics(0.95, has_ltr=False))
    assert g["candidate_model_was_evaluated"] is False


def test_dropped_queries_block():
    g = _gates(challenger=_metrics(0.95, nq=140))
    assert g["full_query_coverage"] is False


def test_feature_schema_mismatch_blocks():
    g = _gates(candidate_feature_names=["a", "c", "b"])
    assert g["feature_schema_matches_serving"] is False


def test_reranker_must_beat_first_stage():
    g = _gates(challenger=_metrics(0.60, first_stage=0.58), champion=None)
    assert g["beats_best_first_stage"] is False


def test_first_run_without_champion_uses_floor():
    assert _gates(champion=None)["_all"]
    assert _gates(challenger=_metrics(0.65, first_stage=0.50), champion=None)["ndcg10_above_floor"] is False


def test_failed_data_validation_blocks():
    g = _gates(data_report={"passed": False, "errors": ["x"]})
    assert g["data_validation_passed"] is False


# ── promote / drift (file-level, no ML) ─────────────────────────────────────
@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("configs").mkdir()
    Path("configs/train.yaml").write_text(yaml.safe_dump(
        {"artifacts": {"model_name": "m.pkl", "meta_name": "m_meta.json", "ltr_dir": "x"}}))
    return {
        "train_config": "configs/train.yaml",
        "production_model": "artifacts/ltr/m.pkl",
        "production_meta": "artifacts/ltr/m_meta.json",
        "candidates_dir": "artifacts/ltr/candidates",
        "archive_dir": "artifacts/ltr/archive",
        "registry": "artifacts/ltr/registry.jsonl",
        "runs_dir": "reports/runs",
        "gate_split": "val",
        "report_split": "test",
        "drift": {"max_champion_ndcg10_change": 0.005},
    }


def _stage_run(cfg, run_id, *, all_passed=True, payload=b"new-model"):
    rd = promotion.run_dir(cfg, run_id)
    cd = promotion.candidate_dir(cfg, run_id)
    cd.mkdir(parents=True)
    rd.mkdir(parents=True)
    cand = cd / "m.pkl"
    cand.write_bytes(pickle.dumps(payload))
    (cd / "m_meta.json").write_text("{}")
    (rd / "train.json").write_text(json.dumps({"model_path": str(cand), "sha256": promotion.sha256_file(cand)}))
    (rd / "gates.json").write_text(json.dumps({"all_passed": all_passed, "passed": 9 if all_passed else 8, "total": 9}))
    (rd / "eval_summary.json").write_text(json.dumps({
        "ndcg@10": {"challenger_val": {"hybrid_ltr": 0.95}, "challenger_test": {"hybrid_ltr": 0.95}}}))
    return cand


def test_promote_refuses_when_gates_failed(cfg):
    _stage_run(cfg, "r1", all_passed=False)
    with pytest.raises(promotion.PromotionError, match="refusing"):
        promotion.step_promote(cfg, "r1")
    assert not Path(cfg["production_model"]).exists()


def test_promote_archives_swaps_and_is_idempotent(cfg):
    prod = Path(cfg["production_model"])
    prod.parent.mkdir(parents=True)
    prod.write_bytes(b"old-model")
    cand = _stage_run(cfg, "r1")

    entry = promotion.step_promote(cfg, "r1")
    assert prod.read_bytes() == cand.read_bytes()
    archived = list(Path(cfg["archive_dir"]).glob("*_m.pkl"))
    assert len(archived) == 1 and archived[0].read_bytes() == b"old-model"
    assert entry["previous_sha256"] is not None

    promotion.step_promote(cfg, "r1")  # re-run of the same task
    lines = Path(cfg["registry"]).read_text().strip().splitlines()
    assert len(lines) == 1
    assert not list(prod.parent.glob("*.tmp"))


def test_promote_is_noop_when_candidate_identical(cfg):
    cand = _stage_run(cfg, "r1")
    prod = Path(cfg["production_model"])
    prod.parent.mkdir(parents=True, exist_ok=True)
    prod.write_bytes(cand.read_bytes())
    assert promotion.step_promote(cfg, "r1")["status"] == "unchanged"
    assert not Path(cfg["registry"]).exists()
    assert not Path(cfg["archive_dir"]).exists() or not list(Path(cfg["archive_dir"]).iterdir())


def test_promote_rejects_candidate_modified_after_gating(cfg):
    cand = _stage_run(cfg, "r1")
    cand.write_bytes(b"tampered")
    with pytest.raises(promotion.PromotionError, match="sha256"):
        promotion.step_promote(cfg, "r1")


def _drift_run(cfg, run_id, created, champ_sha, ltr, fp="f1", bm25=0.57):
    rd = promotion.run_dir(cfg, run_id)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "eval_summary.json").write_text(json.dumps({
        "created": created, "champion_sha256": champ_sha,
        "ndcg@10": {"champion_val": {"bm25": bm25, "dense": 0.51, "hybrid": 0.57, "hybrid_ltr": ltr}}}))
    (rd / "data_validation.json").write_text(json.dumps({"fingerprint": fp}))


def test_drift_no_baseline_on_first_run(cfg):
    _drift_run(cfg, "a", "2026-01-01", "s1", 0.95)
    assert promotion.step_drift(cfg, "a")["status"] == "no_baseline"


def test_drift_stable_when_nothing_changed(cfg):
    _drift_run(cfg, "a", "2026-01-01", "s1", 0.95)
    _drift_run(cfg, "b", "2026-01-02", "s1", 0.9502)
    assert promotion.step_drift(cfg, "b")["status"] == "stable"


def test_drift_fails_when_same_model_scores_differently(cfg):
    _drift_run(cfg, "a", "2026-01-01", "s1", 0.95)
    _drift_run(cfg, "b", "2026-01-02", "s1", 0.93, fp="f2")
    with pytest.raises(promotion.PromotionError, match="without a model change"):
        promotion.step_drift(cfg, "b")
    out = json.loads((promotion.run_dir(cfg, "b") / "drift.json").read_text())
    assert out["data_fingerprint_changed"] is True


def test_drift_first_stage_detected_even_after_model_changed(cfg):
    # new LTR model (s2) so no same-model comparison, but bm25 moved -> data/index drift
    _drift_run(cfg, "a", "2026-01-01", "s1", 0.95, bm25=0.57)
    _drift_run(cfg, "b", "2026-01-02", "s2", 0.95, bm25=0.50)
    with pytest.raises(promotion.PromotionError, match="bm25"):
        promotion.step_drift(cfg, "b")


def test_drift_new_model_alone_is_not_drift(cfg):
    _drift_run(cfg, "a", "2026-01-01", "s1", 0.95)
    _drift_run(cfg, "b", "2026-01-02", "s2", 0.90)  # different model; gates own this
    assert promotion.step_drift(cfg, "b")["status"] == "stable"


def test_drift_uses_latest_earlier_run(cfg):
    _drift_run(cfg, "z_old", "2026-01-01", "s1", 0.80, bm25=0.40)
    _drift_run(cfg, "a_newer", "2026-01-05", "s1", 0.95)
    _drift_run(cfg, "cur", "2026-01-07", "s1", 0.95)
    out = promotion.step_drift(cfg, "cur")
    assert {c["baseline_run"] for c in out["comparisons"]} == {"a_newer"}
    assert out["status"] == "stable"


def test_sanitize_run_id():
    assert promotion.sanitize_run_id("manual__2026-09-23T22:40:00+00:00") == "manual__2026-09-23T22_40_00_00_00"
    with pytest.raises(promotion.PromotionError):
        promotion.sanitize_run_id("///")


def test_eval_strict_mode_refuses_missing_model(tmp_path):
    from eval.evaluate import _best_ltr_path

    cfg = {"eval": {"strict_ltr_path": True, "dataset_processed_dir": "data/processed/movielens"},
           "methods": [{"name": "hybrid_ltr", "type": "hybrid_ltr", "ltr_model_path": str(tmp_path / "nope.pkl")}]}
    with pytest.raises(FileNotFoundError):
        _best_ltr_path(cfg)
