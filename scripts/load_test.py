#!/usr/bin/env python3
"""
Phase 6 — 100 RPS Load Test
==============================
Runs a concurrent load test against the Search API and reports:
  - p50 / p95 / p99 latency (ms)
  - Error rate
  - Requests per second achieved
  - Gate check: p95 < 300ms, error_rate < 2%

Usage:
  python scripts/load_test.py --url http://localhost:8000 --rps 100 --duration 60
  python scripts/load_test.py --url http://localhost:8000 --rps 50 --duration 30 --out reports/latest/load_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass


SAMPLE_QUERIES = [
    "gritty action thriller",
    "feel good romantic comedy",
    "mind-bending sci-fi",
    "dark crime drama",
    "family animation",
    "coming of age story",
    "supernatural horror",
    "political documentary",
    "space adventure",
    "historical epic",
    "heist movie",
    "psychological thriller",
    "sports drama",
    "musical comedy",
    "war film",
]


def _search(base_url: str, query: str, method: str = "hybrid", k: int = 10, timeout: float = 10.0) -> tuple[bool, float]:
    """Single search request. Returns (success, latency_ms)."""
    import urllib.request
    import urllib.error
    url = f"{base_url}/search?q={urllib.parse.quote(query)}&method={method}&k={k}"
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            _ = resp.read()
            ok = 200 <= resp.status < 300
    except Exception:
        ok = False
    return ok, (time.perf_counter() - t0) * 1000.0


def run_load_test(
    base_url: str,
    rps: int,
    duration_s: int,
    method: str = "hybrid",
    k: int = 10,
    max_workers: int = 50,
) -> dict[str, Any]:
    """
    Run load test at target RPS for duration_s seconds.
    Uses a thread pool with sleep-based pacing.
    """
    import urllib.parse  # noqa: PLC0415

    interval = 1.0 / rps  # seconds between requests
    deadline = time.perf_counter() + duration_s

    latencies: list[float] = []
    successes = 0
    errors = 0
    total = 0

    queries = SAMPLE_QUERIES * 100  # enough to cycle

    def send(q: str) -> tuple[bool, float]:
        return _search(base_url, q, method=method, k=k)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        qi = 0
        while time.perf_counter() < deadline:
            q = queries[qi % len(queries)]
            qi += 1
            futures.append(pool.submit(send, q))
            total += 1
            # Pace requests
            time.sleep(max(0.0, interval - 0.001))

        # Collect results
        for fut in as_completed(futures):
            try:
                ok, lat = fut.result()
                latencies.append(lat)
                if ok:
                    successes += 1
                else:
                    errors += 1
            except Exception:
                errors += 1

    if not latencies:
        return {"error": "No requests completed"}

    latencies.sort()
    n = len(latencies)

    def pct(p: float) -> float:
        idx = min(int(p / 100 * n), n - 1)
        return round(latencies[idx], 2)

    actual_rps = total / duration_s
    error_rate = errors / max(1, total)

    result: dict[str, Any] = {
        "target_rps": rps,
        "actual_rps": round(actual_rps, 2),
        "duration_s": duration_s,
        "total_requests": total,
        "successes": successes,
        "errors": errors,
        "error_rate": round(error_rate, 4),
        "latency_ms": {
            "p50": pct(50), "p75": pct(75), "p90": pct(90),
            "p95": pct(95), "p99": pct(99), "p100": pct(100),
            "mean": round(sum(latencies) / n, 2),
        },
        "gates": {
            "p95_under_300ms": pct(95) < 300,
            "error_rate_under_2pct": error_rate < 0.02,
        },
        "passed": pct(95) < 300 and error_rate < 0.02,
    }

    print(f"\n{'='*60}")
    print(f"Load Test Results — {base_url}")
    print(f"{'='*60}")
    print(f"Target RPS:       {rps}")
    print(f"Actual RPS:       {actual_rps:.1f}")
    print(f"Total Requests:   {total}")
    print(f"Errors:           {errors} ({error_rate*100:.1f}%)")
    print(f"Latency p50:      {pct(50):.0f}ms")
    print(f"Latency p95:      {pct(95):.0f}ms")
    print(f"Latency p99:      {pct(99):.0f}ms")
    print(f"\nGate: p95 < 300ms  → {'✅ PASS' if pct(95) < 300 else '❌ FAIL'}")
    print(f"Gate: error < 2%   → {'✅ PASS' if error_rate < 0.02 else '❌ FAIL'}")
    print(f"\nOverall: {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
    print(f"{'='*60}\n")

    return result


def main() -> None:
    import urllib.parse  # noqa: PLC0415
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",      default="http://localhost:8000")
    ap.add_argument("--rps",      type=int,   default=100)
    ap.add_argument("--duration", type=int,   default=60,   dest="duration")
    ap.add_argument("--method",   default="hybrid")
    ap.add_argument("--k",        type=int,   default=10)
    ap.add_argument("--workers",  type=int,   default=50)
    ap.add_argument("--out",      default="reports/latest/load_test.json")
    args = ap.parse_args()

    print(f"Starting load test: {args.rps} RPS for {args.duration}s → {args.url}")
    result = run_load_test(
        base_url=args.url, rps=args.rps, duration_s=args.duration,
        method=args.method, k=args.k, max_workers=args.workers,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"Results written → {args.out}")

    if not result.get("passed", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
