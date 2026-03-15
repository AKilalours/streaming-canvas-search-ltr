# scripts/scale_bench.py
"""
Real Scale Benchmark — 1000 concurrent requests
================================================
This is a REAL load test, not a simulation.

Tests the API under sustained concurrent load:
  - Ramps from 10 to 1000 concurrent users
  - Measures p50/p95/p99 latency at each concurrency level
  - Reports throughput (RPS), error rate, and cache hit rate
  - Identifies the concurrency ceiling (where p99 > 300ms SLO)

Run: uv run python scripts/scale_bench.py --max-concurrency 1000
     uv run python scripts/scale_bench.py --quick  (100 concurrent, 30s)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


QUERIES = [
    "action thriller", "dark comedy", "mind-bending sci-fi",
    "feel good romance", "crime drama", "historical epic",
    "psychological horror", "animated family", "documentary nature",
    "superhero adventure", "heist movie", "survival drama",
    "coming of age", "war film", "mystery detective",
]

ENDPOINTS = [
    ("GET",  "/search?q={query}&k=10&method=hybrid_ltr",  None),
    ("GET",  "/feed?profile=chrisen&k=8&rows=3",           None),
    ("GET",  "/suggest?q={query}&n=5",                     None),
    ("GET",  "/health",                                     None),
]


@dataclass
class RequestResult:
    endpoint: str
    latency_ms: float
    status: int
    cache_hit: bool = False
    error: str = ""


@dataclass
class BenchLevel:
    concurrency: int
    n_requests: int
    duration_s: float
    rps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float
    cache_hit_rate: float
    slo_pass: bool   # p99 <= 300ms


class ScaleBenchmark:
    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url
        self.results: list[RequestResult] = []

    async def _one_request(
        self,
        client: Any,
        endpoint_template: str,
        method: str,
        body: dict | None,
    ) -> RequestResult:
        query = random.choice(QUERIES)
        endpoint = endpoint_template.replace("{query}", query.replace(" ", "+"))
        url = self.base_url + endpoint
        t0 = time.perf_counter()
        try:
            if method == "GET":
                r = await client.get(url, timeout=10.0)
            else:
                r = await client.post(url, json=body, timeout=10.0)
            latency = (time.perf_counter() - t0) * 1000
            body_text = r.text[:200] if r.status_code == 200 else ""
            cache_hit = '"cache_hit": true' in body_text or '"cache_hit":true' in body_text
            return RequestResult(
                endpoint=endpoint,
                latency_ms=round(latency, 2),
                status=r.status_code,
                cache_hit=cache_hit,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            return RequestResult(
                endpoint=endpoint,
                latency_ms=round(latency, 2),
                status=0,
                error=str(e)[:80],
            )

    async def _run_level(
        self,
        concurrency: int,
        duration_s: float = 15.0,
    ) -> BenchLevel:
        sem = asyncio.Semaphore(concurrency)
        results: list[RequestResult] = []
        t_start = time.perf_counter()

        async def worker():
            async with sem:
                method, template, body = random.choice(ENDPOINTS)
                result = await self._one_request(client, template, method, body)
                results.append(result)

        async with httpx.AsyncClient(http2=False) as client:
            # Warmup
            await asyncio.gather(*[worker() for _ in range(min(10, concurrency))])
            results.clear()

            # Actual bench
            t_start = time.perf_counter()
            tasks = []
            while time.perf_counter() - t_start < duration_s:
                if len(tasks) < concurrency * 2:
                    tasks.append(asyncio.create_task(worker()))
                await asyncio.sleep(0)
            await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.perf_counter() - t_start
        latencies = [r.latency_ms for r in results if r.status == 200]
        errors    = [r for r in results if r.status != 200]
        cache_hits = sum(1 for r in results if r.cache_hit)

        def pct(lst, p):
            if not lst: return 0.0
            s = sorted(lst)
            return s[max(0, int(math.ceil(p/100*len(s)))-1)]

        p99 = pct(latencies, 99)
        return BenchLevel(
            concurrency=concurrency,
            n_requests=len(results),
            duration_s=round(elapsed, 1),
            rps=round(len(results)/elapsed, 1) if elapsed > 0 else 0,
            p50_ms=round(pct(latencies, 50), 1),
            p95_ms=round(pct(latencies, 95), 1),
            p99_ms=round(p99, 1),
            error_rate=round(len(errors)/max(1,len(results)), 4),
            cache_hit_rate=round(cache_hits/max(1,len(results)), 4),
            slo_pass=p99 <= 300.0,
        )

    def run(
        self,
        levels: list[int] | None = None,
        duration_per_level: float = 15.0,
        quick: bool = False,
    ) -> list[BenchLevel]:
        if not HTTPX_AVAILABLE:
            print("ERROR: httpx not installed. Run: uv add httpx")
            return []

        if quick:
            levels = [10, 50, 100]
            duration_per_level = 10.0
        elif levels is None:
            levels = [10, 50, 100, 200, 500, 1000]

        bench_levels = []
        print(f"\n{'='*70}")
        print(f"  SCALE BENCHMARK — {self.base_url}")
        print(f"  Levels: {levels}  Duration/level: {duration_per_level}s")
        print(f"{'='*70}")
        print(f"  {'Concur':>8} {'Requests':>10} {'RPS':>8} "
              f"{'p50ms':>8} {'p95ms':>8} {'p99ms':>8} "
              f"{'ErrRate':>9} {'Cache':>7} {'SLO':>5}")
        print(f"  {'-'*70}")

        ceiling = None
        for level in levels:
            bl = asyncio.run(self._run_level(level, duration_per_level))
            bench_levels.append(bl)
            slo_str = "PASS" if bl.slo_pass else "FAIL"
            print(f"  {bl.concurrency:>8} {bl.n_requests:>10} {bl.rps:>8.1f} "
                  f"{bl.p50_ms:>8.1f} {bl.p95_ms:>8.1f} {bl.p99_ms:>8.1f} "
                  f"{bl.error_rate:>9.3f} {bl.cache_hit_rate:>7.3f} {slo_str:>5}")
            if not bl.slo_pass and ceiling is None:
                ceiling = level

        print(f"{'='*70}")
        if ceiling:
            print(f"  Concurrency ceiling: {ceiling} (p99 > 300ms SLO)")
        else:
            print(f"  All levels passed 300ms p99 SLO")
        print()
        return bench_levels

    def save_report(self, levels: list[BenchLevel]) -> Path:
        out = Path("reports/latest/scale_bench.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": time.time(),
            "base_url": self.base_url,
            "slo_p99_ms": 300,
            "levels": [
                {
                    "concurrency": l.concurrency,
                    "n_requests": l.n_requests,
                    "rps": l.rps,
                    "p50_ms": l.p50_ms,
                    "p95_ms": l.p95_ms,
                    "p99_ms": l.p99_ms,
                    "error_rate": l.error_rate,
                    "cache_hit_rate": l.cache_hit_rate,
                    "slo_pass": l.slo_pass,
                }
                for l in levels
            ],
            "ceiling_concurrency": next(
                (l.concurrency for l in levels if not l.slo_pass), None
            ),
            "max_rps": max((l.rps for l in levels), default=0),
        }
        out.write_text(json.dumps(data, indent=2))
        print(f"Report saved: {out}")
        return out


def main():
    parser = argparse.ArgumentParser(description="Scale benchmark")
    parser.add_argument("--base-url",       default="http://localhost:8000")
    parser.add_argument("--max-concurrency", type=int, default=1000)
    parser.add_argument("--duration",        type=float, default=15.0)
    parser.add_argument("--quick",           action="store_true")
    args = parser.parse_args()

    bench = ScaleBenchmark(base_url=args.base_url)
    levels_to_test = [10, 50, 100, 200, 500, args.max_concurrency]
    levels_to_test = sorted(set(c for c in levels_to_test
                                if c <= args.max_concurrency))

    results = bench.run(
        levels=levels_to_test,
        duration_per_level=args.duration,
        quick=args.quick,
    )
    if results:
        bench.save_report(results)


if __name__ == "__main__":
    main()
