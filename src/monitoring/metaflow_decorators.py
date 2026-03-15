# src/monitoring/metaflow_decorators.py
"""
Phase 4 — @netflix_standard Metaflow Decorator
================================================
Wraps every @step with: structured logging, auto-retry, resource hints,
OpenTelemetry heartbeat, artifact pruning.

Usage:
    from monitoring.metaflow_decorators import netflix_standard

    @netflix_standard(retry=2, cpu=2, memory_mb=4096)
    @step
    def train(self): ...
"""
from __future__ import annotations

import functools
import json
import sys
import time
from typing import Any, Callable


def _log(event: str, flow: str, step: str, **kw: Any) -> None:
    payload = {"event": event, "flow": flow, "step": step, **kw}
    print(f"[metaflow] {json.dumps(payload)}", file=sys.stdout, flush=True)


def netflix_standard(
    retry: int = 2,
    cpu: int = 1,
    memory_mb: int = 2048,
    timeout_s: int = 3600,
    emit_heartbeat: bool = True,
    prune_large_attrs: list[str] | None = None,
) -> Callable:
    """
    Production step decorator. All params are advisory/logging — they do not
    override Metaflow's own @resources decorator, but are emitted in structured
    logs for capacity planning and alerting.

    Args:
        retry:             automatic retries on failure
        cpu:               CPU hint (logged for capacity planning)
        memory_mb:         memory hint in MB
        timeout_s:         logged timeout target (seconds)
        emit_heartbeat:    structured start/end log lines for Grafana Loki
        prune_large_attrs: self attrs to del before Metaflow persists artifacts
                           (reduces artifact store size → faster resume)
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            step_name = fn.__name__
            flow_name = type(self).__name__
            last_exc: Exception | None = None

            for attempt in range(retry + 1):
                t0 = time.perf_counter()
                try:
                    if emit_heartbeat:
                        _log("START", flow_name, step_name, attempt=attempt,
                             cpu=cpu, memory_mb=memory_mb, timeout_s=timeout_s)

                    result = fn(self, *args, **kwargs)
                    elapsed = time.perf_counter() - t0

                    if prune_large_attrs:
                        for attr in prune_large_attrs:
                            try:
                                if hasattr(self, attr):
                                    delattr(self, attr)
                            except Exception:
                                pass

                    if emit_heartbeat:
                        _log("END", flow_name, step_name, attempt=attempt,
                             elapsed_s=round(elapsed, 3), status="OK")
                    return result

                except Exception as exc:
                    elapsed = time.perf_counter() - t0
                    last_exc = exc
                    _log("ERROR", flow_name, step_name, attempt=attempt,
                         elapsed_s=round(elapsed, 3), error=str(exc))
                    if attempt < retry:
                        _log("RETRY", flow_name, step_name, attempt=attempt + 1, max_retry=retry)
                    else:
                        _log("GIVE_UP", flow_name, step_name, max_retry=retry, error=str(exc))
                        raise last_exc from last_exc

            return None  # unreachable
        return wrapper
    return decorator


# ── FinOps Cost Gate integration ──────────────────────────────────────────────

def netflix_standard_with_finops(
    retry: int = 2,
    cpu: int = 1,
    memory_mb: int = 2048,
    timeout_s: int = 3600,
    emit_heartbeat: bool = True,
    prune_large_attrs: list[str] | None = None,
    roi_threshold: float = 1.5,
    run_finops_gate: bool = False,
    ndcg_lift_attr: str = "ndcg_lift",
) -> Callable:
    """
    Extended @netflix_standard with integrated FinOps cost gate.

    If run_finops_gate=True, the decorator reads self.<ndcg_lift_attr>
    after the step completes and runs a pre-deployment cost/revenue check.
    The flow is halted if ROI < roi_threshold.
    """
    def decorator(fn: Callable) -> Callable:
        base = netflix_standard(
            retry=retry, cpu=cpu, memory_mb=memory_mb,
            timeout_s=timeout_s, emit_heartbeat=emit_heartbeat,
            prune_large_attrs=prune_large_attrs,
        )(fn)

        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = base(self, *args, **kwargs)

            if run_finops_gate:
                try:
                    from finops.cost_gates import FinOpsGate
                    ndcg_lift = float(getattr(self, ndcg_lift_attr, 0.0))
                    gate = FinOpsGate(roi_threshold=roi_threshold)
                    decision = gate.evaluate(
                        ndcg_lift=ndcg_lift,
                        latency_report_path="reports/latest/latency.json",
                    )
                    _log("FINOPS_GATE", type(self).__name__, fn.__name__,
                         approved=decision.approved, roi=decision.roi,
                         roi_threshold=roi_threshold,
                         recommendation=decision.recommendation)
                    if not decision.approved:
                        raise RuntimeError(
                            f"[FinOps Gate] Deployment REJECTED. ROI={decision.roi:.2f}x < "
                            f"threshold {roi_threshold}x. {decision.recommendation}"
                        )
                except ImportError:
                    _log("FINOPS_GATE_SKIP", type(self).__name__, fn.__name__,
                         reason="finops module not available")

            return result
        return wrapper
    return decorator


# ── Resource Advisor integration ──────────────────────────────────────────────

def check_memory_and_advise(peak_memory_gb: float, step_name: str = "") -> None:
    """
    Call at end of memory-intensive steps. Logs upgrade recommendation
    if peak RAM exceeds threshold. Integrates with ResourceAdvisor.
    """
    try:
        from agents.self_healing import ResourceAdvisor
        advisor = ResourceAdvisor()
        advice = advisor.check(peak_memory_gb)
        if advice["action"] == "upgrade_instance":
            _log("RESOURCE_UPGRADE_RECOMMENDED", "Metaflow", step_name,
                 current=advice["current"], recommended=advice["recommended"],
                 peak_gb=advice["peak_gb"])
    except ImportError:
        pass
