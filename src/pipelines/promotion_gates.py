# src/pipelines/promotion_gates.py
"""
Promotion gates: decide whether a candidate LTR model may replace production.

Design rules
------------
* Every gate reads a value that this run actually measured (files under
  reports/runs/<run_id>/). No gate has a hard-coded outcome.
* Regression gates compare challenger vs champion evaluated in the SAME run, with
  the SAME code, on the SAME split. That avoids comparing against a stale
  reference file produced by an older eval.
* Gates are evaluated on the validation split. The test split is only reported,
  so repeated promotion decisions do not tune the system to the test set.
* This module only needs the standard library, so the Airflow scheduler process
  can import it without the ML stack.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

FIRST_STAGE_METHODS = ("bm25", "dense", "hybrid")
LTR_METHOD = "hybrid_ltr"


@dataclass
class GateResult:
    name: str
    passed: bool
    value: Any
    threshold: Any
    detail: str


def _method(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    for m in metrics.get("methods", []):
        if isinstance(m, dict) and m.get("method") == name:
            return m
    raise KeyError(f"method '{name}' not in metrics (have {[m.get('method') for m in metrics.get('methods', [])]})")


def evaluate_gates(
    *,
    data_report: dict[str, Any],
    challenger: dict[str, Any],
    champion: dict[str, Any] | None,
    candidate_model_path: str,
    candidate_feature_names: list[str],
    serving_feature_names: list[str],
    expected_num_queries: int,
    thresholds: dict[str, float],
    best_promoted_ndcg10: float | None = None,
) -> list[GateResult]:
    """
    best_promoted_ndcg10: highest val nDCG@10 recorded in the promotion registry.
    The nDCG regression gate compares against max(champion, best_promoted) so that
    many small within-tolerance drops cannot ratchet quality down over time.
    """
    t = thresholds
    out: list[GateResult] = []

    # 1. data validation
    out.append(GateResult(
        "data_validation_passed", bool(data_report.get("passed")), data_report.get("passed"), True,
        "; ".join(data_report.get("errors", [])) or "all data checks passed",
    ))

    # 2. the eval really scored the candidate (evaluate.py can fall back to another pickle)
    diag = challenger.get("diagnostics", {})
    loaded = str(diag.get("ltr_path") or "")
    ok = bool(diag.get("has_ltr")) and loaded == str(candidate_model_path)
    out.append(GateResult(
        "candidate_model_was_evaluated", ok, loaded, str(candidate_model_path),
        "eval loaded the candidate pickle" if ok else "eval did not load the candidate model",
    ))

    # 3. training and serving use the same feature schema, in the same order
    ok = list(candidate_feature_names) == list(serving_feature_names)
    out.append(GateResult(
        "feature_schema_matches_serving", ok, len(candidate_feature_names), len(serving_feature_names),
        "feature names and order match ranking.features.FEATURE_NAMES" if ok
        else f"mismatch: candidate={candidate_feature_names} serving={serving_feature_names}",
    ))

    ltr = _method(challenger, LTR_METHOD)

    # 4. every query was scored
    nq = int(ltr.get("num_queries", 0))
    out.append(GateResult(
        "full_query_coverage", nq == expected_num_queries, nq, expected_num_queries,
        "all validation queries scored" if nq == expected_num_queries else "queries were dropped",
    ))

    # 5. absolute floor (safety net for the first run, before a champion exists)
    v = float(ltr["ndcg@10"])
    out.append(GateResult(
        "ndcg10_above_floor", v >= t["min_ndcg10"], round(v, 4), t["min_ndcg10"], "val nDCG@10",
    ))

    # 6. the reranker must add value over the retrieval it reranks
    best_name, best_v = max(
        ((m, float(_method(challenger, m)["ndcg@10"])) for m in FIRST_STAGE_METHODS),
        key=lambda x: x[1],
    )
    lift = v - best_v
    out.append(GateResult(
        "beats_best_first_stage", lift > t["min_lift_over_first_stage"], round(lift, 4),
        t["min_lift_over_first_stage"], f"LTR {v:.4f} vs best first stage {best_name} {best_v:.4f}",
    ))

    # 7-9. non-regression vs the current production model on the same split
    for gate, metric, key in (
        ("ndcg10_no_regression", "ndcg@10", "max_ndcg10_drop"),
        ("recall100_no_regression", "recall@100", "max_recall100_drop"),
        ("map10_no_regression", "map@10", "max_map10_drop"),
    ):
        if champion is None:
            out.append(GateResult(gate, True, None, t[key], "no production model yet; skipped"))
            continue
        cur = float(ltr[metric])
        ref = float(_method(champion, LTR_METHOD)[metric])
        label = "champion"
        if metric == "ndcg@10" and best_promoted_ndcg10 is not None and best_promoted_ndcg10 > ref:
            ref, label = float(best_promoted_ndcg10), "best promoted"
        drop = ref - cur
        out.append(GateResult(
            gate, drop <= t[key], round(drop, 4), t[key],
            f"{label} {ref:.4f} -> challenger {cur:.4f} (drop {drop:+.4f})",
        ))

    return out


def summarize(results: list[GateResult]) -> dict[str, Any]:
    return {
        "all_passed": all(r.passed for r in results),
        "passed": sum(r.passed for r in results),
        "total": len(results),
        "gates": [asdict(r) for r in results],
    }
