# src/pipelines/gates.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _fail(msg: str) -> None:
    raise SystemExit(f"[GATE FAIL] {msg}")


def _norm(s: str) -> str:
    # normalize aggressively: keep only [a-z0-9], collapse to underscores
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _methods_list(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    methods = metrics.get("methods")
    if isinstance(methods, list):
        return [x for x in methods if isinstance(x, dict)]
    _fail(f"metrics.json has no list field 'methods'. top_keys={list(metrics.keys())}")
    raise AssertionError("unreachable")


def _available_method_names(methods: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for x in methods:
        name = x.get("method")
        if isinstance(name, str):
            out.append(name)
    return out


def _pick_method_entry(methods: list[dict[str, Any]], want_run: str) -> dict[str, Any]:
    want = _norm(want_run)
    names = _available_method_names(methods)

    # 1) exact normalized match
    for x in methods:
        name = x.get("method")
        if isinstance(name, str) and _norm(name) == want:
            return x

    # 2) fuzzy match (prefix/containment) pick closest
    cands: list[tuple[int, dict[str, Any]]] = []
    for x in methods:
        name = x.get("method")
        if not isinstance(name, str):
            continue
        n = _norm(name)
        if n.startswith(want) or want.startswith(n) or (want in n) or (n in want):
            # score = distance in length (smaller is better)
            cands.append((abs(len(n) - len(want)), x))
    if cands:
        cands.sort(key=lambda t: t[0])
        return cands[0][1]

    # 3) last resort: if want mentions ltr, pick any method containing ltr
    if "ltr" in want:
        for x in methods:
            name = x.get("method")
            if isinstance(name, str) and "ltr" in _norm(name):
                return x

    _fail(f"Metric run '{want_run}' not found in metrics.json methods. Available methods: {names}")
    raise AssertionError("unreachable")


def _pick_metric(entry: dict[str, Any], metric_key: str) -> float:
    mk = _norm(metric_key)

    # direct keys at top-level (your file has ndcg@10 directly)
    for k, v in entry.items():
        if _norm(k) == mk:
            return float(v)

    # tolerate nested
    for container in ("metrics", "scores"):
        inner = entry.get(container)
        if isinstance(inner, dict):
            for k, v in inner.items():
                if _norm(k) == mk:
                    return float(v)

    _fail(f"Metric '{metric_key}' not found in method entry keys={list(entry.keys())[:30]}")
    raise AssertionError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", required=True)
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.gates).read_text(encoding="utf-8")) or {}
    run_dir = Path(args.run_dir)

    gate_cfg = cfg.get("gate", {}) or {}
    gate_run = gate_cfg.get("run", "hybrid_ltr")
    gate_metric = gate_cfg.get("metric", "ndcg@10")

    thr = cfg.get("thresholds", {}) or {}
    min_metric = float(thr.get("min_ndcg10", 0.0))
    max_drop_pct = float(thr.get("max_ndcg10_drop_pct", 100.0))
    require_latency = bool(thr.get("require_latency", False))
    min_success_rate = float(thr.get("min_success_rate", 0.99))

    max_p95_ms = thr.get("max_p95_ms", 1e9)

    cur_metrics_path = run_dir / "metrics.json"
    if not cur_metrics_path.exists():
        _fail(f"current metrics not found: {cur_metrics_path}")
    cur_metrics = _read_json(cur_metrics_path)
    methods = _methods_list(cur_metrics)

    cur_entry = _pick_method_entry(methods, gate_run)
    cur_val = _pick_metric(cur_entry, gate_metric)

    if cur_val < min_metric:
        _fail(f"{gate_run}.{gate_metric} {cur_val:.4f} < min {min_metric:.4f}")

    # regression vs reference (optional)
    ref_cfg = cfg.get("reference", {}) or {}
    ref_metrics_path = ref_cfg.get("metrics_path")
    if ref_metrics_path and Path(ref_metrics_path).exists():
        ref_metrics = _read_json(Path(ref_metrics_path))
        ref_methods = _methods_list(ref_metrics)
        ref_entry = _pick_method_entry(ref_methods, gate_run)
        ref_val = _pick_metric(ref_entry, gate_metric)

        drop_pct = 100.0 * (ref_val - cur_val) / max(1e-9, ref_val)
        if drop_pct > max_drop_pct:
            _fail(
                f"{gate_run}.{gate_metric} regressed: ref={ref_val:.4f} cur={cur_val:.4f} "
                f"drop={drop_pct:.2f}% > {max_drop_pct:.2f}%"
            )
    else:
        print("[GATE WARN] no reference metrics found; skipping regression gate")

    # latency gate (optional)
    cur_latency_path = run_dir / "latency.json"
    if cur_latency_path.exists():
        lat = _read_json(cur_latency_path)
        endpoint = str(lat.get("endpoint", "/search"))
        p95 = float(lat.get("p95_ms", 1e9))
        sr = float(lat.get("success_rate", 1.0))

        if isinstance(max_p95_ms, dict):
            p95_max = float(max_p95_ms.get(endpoint, max_p95_ms.get("*", 1e9)))
        else:
            p95_max = float(max_p95_ms)

        if sr < min_success_rate:
            _fail(f"{endpoint} success_rate {sr:.3f} < min {min_success_rate:.3f}")
        if p95 > p95_max:
            _fail(f"{endpoint} p95 {p95:.1f}ms > max {p95_max:.1f}ms")
    else:
        if require_latency:
            _fail(f"latency.json missing at {cur_latency_path} but require_latency=true")
        print("[GATE WARN] latency.json not found; skipping latency gate")

    print("[GATE PASS] all gates OK")


if __name__ == "__main__":
    main()
