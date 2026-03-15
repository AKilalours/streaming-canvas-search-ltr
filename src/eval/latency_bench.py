# src/eval/latency_bench.py
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = int(round((p / 100.0) * (len(ys) - 1)))
    return float(ys[max(0, min(k, len(ys) - 1))])


def _preflight(base: str) -> None:
    base = base.rstrip("/")
    try:
        r = httpx.get(f"{base}/health", timeout=3.0)
        if r.status_code != 200:
            raise RuntimeError(f"/health returned {r.status_code}")
    except Exception as e:
        raise SystemExit(
            f"[latency] Cannot reach API at {base}. Start the server first.\n"
            f"Run in another terminal:\n"
            f"  PYTHONPATH=src uv run uvicorn app.main:app --host 127.0.0.1 --port 8000\n"
            f"Original error: {e}"
        ) from e


async def _one(
    client: httpx.AsyncClient, url: str, payload: dict[str, Any], timeout_s: float
) -> float | None:
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
    except (httpx.ReadTimeout, httpx.ConnectTimeout):
        return None
    return (time.perf_counter() - t0) * 1000.0


async def _run(
    url: str, payload: dict[str, Any], n: int, conc: int, timeout_s: float
) -> dict[str, Any]:
    limits = httpx.Limits(max_connections=conc, max_keepalive_connections=conc)
    async with httpx.AsyncClient(limits=limits) as client:
        sem = asyncio.Semaphore(conc)
        lat: list[float] = []
        fails = 0

        async def task():
            nonlocal fails
            async with sem:
                v = await _one(client, url, payload, timeout_s=timeout_s)
                if v is None:
                    fails += 1
                else:
                    lat.append(v)

        await asyncio.gather(*[task() for _ in range(n)])

    ok = n - fails
    success_rate = ok / max(1, n)

    return {
        "n": n,
        "concurrency": conc,
        "timeout_s": float(timeout_s),
        "ok": ok,
        "fail": fails,
        "success_rate": float(success_rate),
        "p50_ms": _pct(lat, 50),
        "p95_ms": _pct(lat, 95),
        "p99_ms": _pct(lat, 99),
        "mean_ms": float(sum(lat) / max(1, len(lat))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--endpoint", default="/search", choices=["/search", "/answer", "/agent_answer"]
    )
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=None, help="Per-request timeout seconds")
    ap.add_argument("--out", default="reports/latest/latency.json")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    _preflight(base)

    url = f"{base}{args.endpoint}"

    # sensible defaults
    if args.timeout is None:
        if args.endpoint == "/search":
            timeout_s = 10.0
        elif args.endpoint == "/answer":
            timeout_s = 180.0
        else:
            timeout_s = 240.0
    else:
        timeout_s = float(args.timeout)

    if args.endpoint == "/search":
        payload = {
            "query": "covid transmission evidence",
            "method": "hybrid_ltr",
            "k": 10,
            "debug": False,
        }
    else:
        payload = {
            "query": "What is SciFact used for?",
            "method": "hybrid_ltr",
            "k": 8,
            "context_k": 6,
            "debug": False,
        }

    res = asyncio.run(_run(url, payload, n=args.n, conc=args.concurrency, timeout_s=timeout_s))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"endpoint": args.endpoint, **res}, indent=2), encoding="utf-8")
    print(f"[latency] wrote {outp}")


if __name__ == "__main__":
    main()
