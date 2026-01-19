import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from utils.io import ensure_dir, write_json
from utils.logging import get_logger

log = get_logger("pipelines.repro")


def run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def sh(cmd: list[str]) -> None:
    log.info("RUN: %s", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to configs/dataset.yaml")
    ap.add_argument("--eval", required=True, help="Path to configs/eval.yaml")
    ap.add_argument("--reports_dir", default="reports", help="Reports root (default: reports)")
    args = ap.parse_args()

    rid = run_id()
    root = Path(args.reports_dir)
    run_dir = ensure_dir(root / rid)
    latest_dir = ensure_dir(root / "latest")

    sha = git_sha()
    py = sys.executable

    # Snapshot configs
    shutil.copy2(args.dataset, run_dir / "dataset.yaml")
    shutil.copy2(args.eval, run_dir / "eval.yaml")

    # Full snapshot (merged)
    dataset_cfg = yaml.safe_load(Path(args.dataset).read_text(encoding="utf-8"))
    eval_cfg = yaml.safe_load(Path(args.eval).read_text(encoding="utf-8"))
    snapshot = {
        "run_id": rid,
        "git_sha": sha,
        "python": sys.version,
        "dataset_config": dataset_cfg,
        "eval_config": eval_cfg,
    }
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "git_sha.txt").write_text(sha + "\n", encoding="utf-8")

    # Pipeline timings
    timings_ms: dict[str, float] = {}

    def timed_step(name: str, fn):
        t0 = time.perf_counter()
        fn()
        timings_ms[name] = (time.perf_counter() - t0) * 1000.0

    # End-to-end steps
    timed_step("data", lambda: sh([py, "-m", "dataio.build_dataset", "--config", args.dataset]))
    timed_step("bm25", lambda: sh([py, "-m", "retrieval.bm25_index", "--config", args.dataset]))
    timed_step("dense", lambda: sh([py, "-m", "retrieval.embed_index", "--config", args.dataset]))

    # Evaluate directly into run_dir
    timed_step(
        "eval",
        lambda: sh([py, "-m", "eval.evaluate", "--config", args.eval, "--out_dir", str(run_dir)]),
    )

    write_json(run_dir / "latency.json", {"pipeline_timings_ms": timings_ms})

    # Update reports/latest (clear old files but keep .gitkeep)
    for p in latest_dir.glob("*"):
        if p.name == ".gitkeep":
            continue
        if p.is_file():
            p.unlink()

    # Copy run outputs into latest
    for fname in ["metrics.json", "ablations.csv", "results.jsonl", "latency.json", "config_snapshot.yaml", "git_sha.txt"]:
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, latest_dir / fname)

    write_json(
        run_dir / "run_meta.json",
        {"run_id": rid, "git_sha": sha, "run_dir": str(run_dir), "latest_dir": str(latest_dir)},
    )

    print(f"[repro] done -> {run_dir}")
    print(f"[repro] latest -> {latest_dir}")


if __name__ == "__main__":
    main()

