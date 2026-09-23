# src/pipelines/promotion.py
"""
StreamLens model promotion pipeline: validate -> train -> evaluate -> gate -> promote,
plus a drift check. Each step is a CLI subcommand so Airflow (or a human) can run
one step at a time:

    PYTHONPATH=src python -m pipelines.promotion validate --run-id R
    PYTHONPATH=src python -m pipelines.promotion train    --run-id R
    PYTHONPATH=src python -m pipelines.promotion evaluate --run-id R
    PYTHONPATH=src python -m pipelines.promotion gate     --run-id R
    PYTHONPATH=src python -m pipelines.promotion promote  --run-id R
    PYTHONPATH=src python -m pipelines.promotion drift    --run-id R

Steps share state only through files in reports/runs/<run_id>/ and
artifacts/ltr/candidates/<run_id>/, so any step can be re-run after a failure.

Exit codes: 0 = step succeeded. `gate` exits 0 whether or not the gates pass
(blocking promotion is a valid outcome); the decision is in gates.json.
`promote` refuses to run unless gates.json says all gates passed.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from pipelines.data_quality import check_data_quality
from pipelines.promotion_gates import evaluate_gates, summarize

DEFAULT_CONFIG = "configs/promotion.yaml"


class PromotionError(RuntimeError):
    pass


# ── helpers ─────────────────────────────────────────────────────────────────
def sanitize_run_id(run_id: str) -> str:
    """Airflow run ids look like 'manual__2026-09-23T22:40:00+00:00'; make them path-safe."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id.strip()).strip("_.")
    if not s:
        raise PromotionError(f"invalid run id: {run_id!r}")
    return s


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cfg(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, p)


def _read_json(p: Path) -> Any:
    if not p.exists():
        raise PromotionError(f"required file missing: {p} (did the upstream step run?)")
    return json.loads(p.read_text(encoding="utf-8"))


def run_dir(cfg: dict[str, Any], run_id: str) -> Path:
    return Path(cfg["runs_dir"]) / run_id


def candidate_dir(cfg: dict[str, Any], run_id: str) -> Path:
    return Path(cfg["candidates_dir"]) / run_id


def _train_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(Path(cfg["train_config"]).read_text(encoding="utf-8"))


def candidate_model_path(cfg: dict[str, Any], run_id: str) -> Path:
    tcfg = _train_cfg(cfg)
    return candidate_dir(cfg, run_id) / tcfg["artifacts"]["model_name"]


def candidate_meta_path(cfg: dict[str, Any], run_id: str) -> Path:
    tcfg = _train_cfg(cfg)
    return candidate_dir(cfg, run_id) / tcfg["artifacts"]["meta_name"]


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


# ── steps ───────────────────────────────────────────────────────────────────
def step_validate(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    dq = cfg.get("data_quality", {}) or {}
    rep = check_data_quality(
        cfg["dataset_processed_dir"],
        min_docs=int(dq.get("min_docs", 1000)),
        min_queries={k: int(v) for k, v in (dq.get("min_queries") or {}).items()},
        max_cross_split_duplicate_frac=float(dq.get("max_cross_split_duplicate_frac", 1.0)),
    ).to_dict()
    rep["run_id"] = run_id
    rep["created"] = _now()
    _write_json(run_dir(cfg, run_id) / "data_validation.json", rep)
    for w in rep["warnings"]:
        print(f"[DATA WARN] {w}")
    if not rep["passed"]:
        raise PromotionError("data validation failed: " + "; ".join(rep["errors"]))
    print(f"[DATA OK] {sum(rep['checks'].values())}/{len(rep['checks'])} checks passed, "
          f"fingerprint={rep['fingerprint'][:12]}")
    return rep


def step_train(cfg: dict[str, Any], run_id: str) -> Path:
    from ranking.ltr_train import train  # heavy import (lightgbm, torch)

    _read_json(run_dir(cfg, run_id) / "data_validation.json")  # enforce ordering
    tcfg = _train_cfg(cfg)
    tcfg = copy.deepcopy(tcfg)
    tcfg["artifacts"]["ltr_dir"] = str(candidate_dir(cfg, run_id))
    model_path = train(tcfg)
    info = {
        "run_id": run_id,
        "created": _now(),
        "model_path": str(model_path),
        "sha256": sha256_file(model_path),
        "train_config": cfg["train_config"],
    }
    _write_json(run_dir(cfg, run_id) / "train.json", info)
    print(f"[TRAIN OK] candidate {model_path} sha256={info['sha256'][:12]}")
    return model_path


def _eval_cfg_for(cfg: dict[str, Any], split: str, ltr_path: Path) -> dict[str, Any]:
    ecfg = yaml.safe_load(Path(cfg["eval_config"]).read_text(encoding="utf-8"))
    ecfg = copy.deepcopy(ecfg)
    ecfg["eval"]["split"] = split
    ecfg["eval"]["dataset_processed_dir"] = cfg["dataset_processed_dir"]
    ecfg["eval"]["strict_ltr_path"] = True
    for m in ecfg["methods"]:
        if m.get("name") == "hybrid_ltr" or m.get("type") == "hybrid_ltr":
            m["ltr_model_path"] = str(ltr_path)
    return ecfg


def step_evaluate(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    from eval.evaluate import run_eval  # heavy import

    rd = run_dir(cfg, run_id)
    train_info = _read_json(rd / "train.json")
    cand = Path(train_info["model_path"])
    if sha256_file(cand) != train_info["sha256"]:
        raise PromotionError(f"candidate {cand} changed after training (sha256 mismatch)")

    gate_split, report_split = cfg["gate_split"], cfg["report_split"]
    out: dict[str, Any] = {}

    out[f"challenger_{gate_split}"] = run_eval(_eval_cfg_for(cfg, gate_split, cand))
    out[f"challenger_{report_split}"] = run_eval(_eval_cfg_for(cfg, report_split, cand))

    prod = Path(cfg["production_model"])
    if prod.exists():
        out[f"champion_{gate_split}"] = run_eval(_eval_cfg_for(cfg, gate_split, prod))
        champion_sha = sha256_file(prod)
    else:
        champion_sha = None

    for name, metrics in out.items():
        _write_json(rd / f"metrics_{name}.json", metrics)

    summary = {
        "run_id": run_id,
        "created": _now(),
        "candidate_sha256": train_info["sha256"],
        "champion_sha256": champion_sha,
        "ndcg@10": {
            name: {m["method"]: round(float(m["ndcg@10"]), 4) for m in metrics["methods"]}
            for name, metrics in out.items()
        },
    }
    _write_json(rd / "eval_summary.json", summary)
    print("[EVAL OK]", json.dumps(summary["ndcg@10"]))
    return summary


def _feature_names_of(model_pickle: Path, meta_path: Path) -> list[str]:
    import pickle

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    names = list(meta.get("feature_names") or [])
    with model_pickle.open("rb") as f:
        model = pickle.load(f)
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and int(n_in) != len(names):
        # the model was fit on a different number of columns than its meta claims
        return names + [f"<model expects {n_in} features>"]
    return names


def step_gate(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    from ranking.features import FEATURE_NAMES

    rd = run_dir(cfg, run_id)
    gs = cfg["gate_split"]
    data_report = _read_json(rd / "data_validation.json")
    challenger = _read_json(rd / f"metrics_challenger_{gs}.json")
    champ_p = rd / f"metrics_champion_{gs}.json"
    champion = _read_json(champ_p) if champ_p.exists() else None
    train_info = _read_json(rd / "train.json")

    expected_nq = data_report["stats"][gs]["num_queries"]
    best_promoted = None
    reg = Path(cfg["registry"])
    if reg.exists():
        vals = [json.loads(x).get(f"{gs}_ndcg@10") for x in reg.read_text(encoding="utf-8").splitlines() if x.strip()]
        vals = [float(v) for v in vals if v is not None]
        best_promoted = max(vals) if vals else None
    cand_path = Path(train_info["model_path"])
    results = evaluate_gates(
        data_report=data_report,
        challenger=challenger,
        champion=champion,
        candidate_model_path=str(cand_path),
        candidate_feature_names=_feature_names_of(cand_path, candidate_meta_path(cfg, run_id)),
        serving_feature_names=list(FEATURE_NAMES),
        expected_num_queries=int(expected_nq),
        thresholds={k: float(v) for k, v in cfg["gates"].items()},
        best_promoted_ndcg10=best_promoted,
    )
    summary = summarize(results)
    summary.update({"run_id": run_id, "created": _now(), "split": gs})
    _write_json(rd / "gates.json", summary)

    for g in summary["gates"]:
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['name']}: {g['detail']}")
    print(f"[GATES] {summary['passed']}/{summary['total']} passed -> "
          f"{'PROMOTE' if summary['all_passed'] else 'BLOCK'}")
    return summary


def step_promote(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    rd = run_dir(cfg, run_id)
    gates = _read_json(rd / "gates.json")
    if not gates.get("all_passed"):
        raise PromotionError(f"refusing to promote: gates {gates.get('passed')}/{gates.get('total')}")

    train_info = _read_json(rd / "train.json")
    cand = Path(train_info["model_path"])
    cand_sha = sha256_file(cand)
    if cand_sha != train_info["sha256"]:
        raise PromotionError("candidate changed between gate and promote (sha256 mismatch)")

    registry = Path(cfg["registry"])
    if registry.exists():
        for line in registry.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("run_id") == run_id:
                print(f"[PROMOTE] run {run_id} already promoted; nothing to do")
                return json.loads(line)

    prod = Path(cfg["production_model"])
    prod_meta = Path(cfg["production_meta"])
    archive = Path(cfg["archive_dir"])
    archive.mkdir(parents=True, exist_ok=True)

    previous_sha = None
    if prod.exists():
        previous_sha = sha256_file(prod)
        if previous_sha == cand_sha:
            entry = {"run_id": run_id, "promoted_at": _now(), "model_sha256": cand_sha,
                     "previous_sha256": previous_sha, "status": "unchanged"}
            _write_json(rd / "promotion.json", entry)
            print(f"[PROMOTE] candidate is byte-identical to production ({cand_sha[:12]}); no swap")
            return entry
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(prod, archive / f"{stamp}_{previous_sha[:12]}_{prod.name}")
        if prod_meta.exists():
            shutil.copy2(prod_meta, archive / f"{stamp}_{previous_sha[:12]}_{prod_meta.name}")

    # atomic swap: copy next to the target, then rename over it
    prod.parent.mkdir(parents=True, exist_ok=True)
    tmp = prod.with_name(prod.name + f".{run_id}.tmp")
    shutil.copy2(cand, tmp)
    os.replace(tmp, prod)
    cand_meta = candidate_meta_path(cfg, run_id)
    if cand_meta.exists():
        tmpm = prod_meta.with_name(prod_meta.name + f".{run_id}.tmp")
        shutil.copy2(cand_meta, tmpm)
        os.replace(tmpm, prod_meta)

    if sha256_file(prod) != cand_sha:
        raise PromotionError("post-promotion checksum mismatch")

    gs, rs = cfg["gate_split"], cfg["report_split"]
    ev = _read_json(rd / "eval_summary.json")
    entry = {
        "run_id": run_id,
        "promoted_at": _now(),
        "model_sha256": cand_sha,
        "previous_sha256": previous_sha,
        "gates": f"{gates['passed']}/{gates['total']}",
        f"{gs}_ndcg@10": ev["ndcg@10"][f"challenger_{gs}"]["hybrid_ltr"],
        f"{rs}_ndcg@10": ev["ndcg@10"][f"challenger_{rs}"]["hybrid_ltr"],
        "reports": str(rd),
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    _write_json(rd / "promotion.json", entry)
    print(f"[PROMOTE OK] {prod} <- {cand} (sha256 {cand_sha[:12]}, previous {str(previous_sha)[:12]})")
    return entry


def step_drift(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    """
    Detect changes that are NOT caused by a new model.

    * First-stage retrieval (bm25, dense, hybrid) does not depend on the LTR model,
      so on the same split its scores should be identical run to run. Compared with
      the most recent earlier run.
    * The production LTR model is compared only against the most recent earlier run
      that evaluated the SAME model (same sha256); retraining is not byte-deterministic.
    A change beyond the limit means data, index, encoder, code or dependencies moved.
    """
    rd = run_dir(cfg, run_id)
    gs = cfg["gate_split"]
    key = f"champion_{gs}" if f"champion_{gs}" in _read_json(rd / "eval_summary.json")["ndcg@10"] else f"challenger_{gs}"
    cur = _read_json(rd / "eval_summary.json")
    cur_dq = _read_json(rd / "data_validation.json")
    limit = float(cfg["drift"]["max_champion_ndcg10_change"])

    earlier: list[tuple[Path, dict[str, Any]]] = []
    for d in Path(cfg["runs_dir"]).iterdir():
        f = d / "eval_summary.json"
        if d.name == run_id or not f.exists():
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        if s.get("created", "") <= cur.get("created", ""):
            earlier.append((d, s))
    earlier.sort(key=lambda x: x[1].get("created", ""))

    comparisons: list[dict[str, Any]] = []

    def _score(summary: dict[str, Any], method: str) -> float | None:
        for k in (f"champion_{gs}", f"challenger_{gs}"):
            v = summary.get("ndcg@10", {}).get(k, {}).get(method)
            if v is not None:
                return float(v)
        return None

    if earlier:
        d, s = earlier[-1]
        for m in ("bm25", "dense", "hybrid"):
            a, b = _score(s, m), _score(cur, m)
            if a is not None and b is not None:
                comparisons.append({"what": f"{m} (first stage)", "baseline_run": d.name,
                                    "baseline": a, "current": b, "change": round(b - a, 4)})
        prev_dq_p = d / "data_validation.json"
        fp_changed = (json.loads(prev_dq_p.read_text(encoding="utf-8")).get("fingerprint")
                      != cur_dq.get("fingerprint")) if prev_dq_p.exists() else None
    else:
        fp_changed = None

    champ_sha = cur.get("champion_sha256")
    same_model = [x for x in earlier if champ_sha and x[1].get("champion_sha256") == champ_sha]
    if same_model and key.startswith("champion"):
        d, s = same_model[-1]
        a = float(s["ndcg@10"][f"champion_{gs}"]["hybrid_ltr"])
        b = float(cur["ndcg@10"][key]["hybrid_ltr"])
        comparisons.append({"what": f"hybrid_ltr (same model {champ_sha[:12]})", "baseline_run": d.name,
                            "baseline": a, "current": b, "change": round(b - a, 4)})

    drifted = [c for c in comparisons if abs(c["change"]) > limit]
    result = {
        "run_id": run_id,
        "created": _now(),
        "split": gs,
        "limit": limit,
        "data_fingerprint_changed": fp_changed,
        "comparisons": comparisons,
        "status": "no_baseline" if not comparisons else ("drift" if drifted else "stable"),
    }
    _write_json(rd / "drift.json", result)
    for c in comparisons:
        print(f"  {c['what']}: {c['baseline']:.4f} -> {c['current']:.4f} ({c['change']:+.4f}) vs {c['baseline_run']}")
    print(f"[DRIFT] {result['status']} (limit {limit}, data fingerprint changed: {fp_changed})")
    if drifted:
        raise PromotionError("scores moved without a model change: "
                             + "; ".join(f"{c['what']} {c['change']:+.4f}" for c in drifted))
    return result


STEPS = {
    "validate": step_validate,
    "train": step_train,
    "evaluate": step_evaluate,
    "gate": step_gate,
    "promote": step_promote,
    "drift": step_drift,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=[*STEPS, "all"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    args = ap.parse_args(argv)

    cfg = load_cfg(args.config)
    run_id = sanitize_run_id(args.run_id)
    try:
        if args.step == "all":
            for name in ("validate", "train", "evaluate", "gate"):
                STEPS[name](cfg, run_id)
            if _read_json(run_dir(cfg, run_id) / "gates.json")["all_passed"]:
                step_promote(cfg, run_id)
            step_drift(cfg, run_id)
        else:
            STEPS[args.step](cfg, run_id)
    except PromotionError as e:
        print(f"[FAIL] {args.step}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
