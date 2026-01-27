from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from eval.evaluate import run_eval_for_dataset_config
from utils.io import ensure_dir, write_json
from utils.logging import get_logger

log = get_logger("pipelines.multi_eval")


def _get(cfg: dict[str, Any], *paths: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in paths:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)

    # Allow either:
    #  A) run: {..., datasets:[...]}
    #  B) datasets: [...] at top-level
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run", {}), dict) else {}

    out_dir = ensure_dir(
        _get(cfg, "run", "out_dir", default=_get(cfg, "out_dir", default="reports/multi_eval"))
    )

    # Params can live in run.*, eval.*, or top-level
    k = int(_get(cfg, "run", "k", default=_get(cfg, "eval", "k", default=10)))
    candidate_k = int(
        _get(cfg, "run", "candidate_k", default=_get(cfg, "run", "candidates_k", default=_get(cfg, "eval", "candidates_k", default=200)))
    )
    rerank_k = int(_get(cfg, "run", "rerank_k", default=_get(cfg, "eval", "rerank_k", default=50)))
    alpha = float(_get(cfg, "run", "alpha", default=_get(cfg, "eval", "alpha", default=0.5)))

    datasets = cfg.get("datasets")
    if datasets is None:
        datasets = run_cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise KeyError("No datasets found. Provide either top-level `datasets:` or `run.datasets:` in the YAML.")

    results: list[dict[str, Any]] = []

    for ds in datasets:
        # ds can be:
        # - {"name": "...", "config": "path.yaml"}
        # - "path.yaml"
        if isinstance(ds, str):
            name = Path(ds).stem
            ds_path = ds
        elif isinstance(ds, dict):
            name = str(ds.get("name") or Path(str(ds["config"])).stem)
            ds_path = str(ds["config"])
        else:
            raise TypeError(f"Invalid dataset entry: {ds}")

        dataset_cfg = _load_yaml(ds_path)

        # Apply overrides if user gave eval: section (optional)
        if isinstance(cfg.get("eval"), dict):
            dataset_cfg = dict(dataset_cfg)
            dataset_cfg["eval"] = dict(dataset_cfg.get("eval", {}))
            dataset_cfg["eval"].update(cfg["eval"])

        log.info("Running multi-eval dataset=%s config=%s", name, ds_path)
        out = run_eval_for_dataset_config(
            dataset_cfg=dataset_cfg,
            k=k,
            candidate_k=candidate_k,
            rerank_k=rerank_k,
            alpha=alpha,
            out_dir=Path(out_dir),
        )
        out["dataset_name"] = name
        out["dataset_config"] = ds_path
        results.append(out)

    write_json(Path(out_dir) / "summary.json", {"runs": results})
    log.info("Wrote %s", str(Path(out_dir) / "summary.json"))


if __name__ == "__main__":
    main()
