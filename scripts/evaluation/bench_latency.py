from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import requests
import ir_datasets

def load_queries(limit=200):
    ds = ir_datasets.load("beir/nfcorpus/test")
    qs = list(ds.queries_iter())
    return qs[:limit]

def timed_get(url, params):
    t0 = time.perf_counter()
    r = requests.get(url, params=params, timeout=120)
    dt = (time.perf_counter() - t0) * 1000.0
    return r.status_code, dt

def summary(arr):
    arr = np.array(arr, dtype=np.float64)
    return {
        "count": int(arr.size),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="reports/latest/latency.json")
    args = ap.parse_args()

    queries = load_queries(args.limit)
    methods = ["bm25", "dense", "hybrid", "hybrid_ltr"]

    out = {"num_queries": len(queries), "k": args.k, "methods": {}}

    for m in methods:
        times = []
        for q in queries:
            code, dt = timed_get(f"{args.api}/search", {"q": q.text, "method": m, "k": args.k})
            if code != 200:
                raise RuntimeError(f"HTTP {code} on method={m}")
            times.append(dt)
        out["methods"][m] = summary(times)
        print(m, out["methods"][m])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"✅ wrote {args.out}")

if __name__ == "__main__":
    main()
