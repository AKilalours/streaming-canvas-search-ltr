from __future__ import annotations
import argparse, json
from pathlib import Path

def load_json(p: Path):
    return json.loads(p.read_text())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", default="reports/latest/metrics.json")
    ap.add_argument("--reference", default="reports/reference/metrics.json")
    ap.add_argument("--out", default="reports/latest/drift_report.json")
    ap.add_argument("--ndcg_drop", type=float, default=0.03)   # 3% absolute drop triggers
    ap.add_argument("--p99_max_ms", type=float, default=200.0) # latency guardrail
    args = ap.parse_args()

    latest_p = Path(args.latest)
    ref_p = Path(args.reference)
    out_p = Path(args.out)

    if not ref_p.exists():
        ref_p.parent.mkdir(parents=True, exist_ok=True)
        ref_p.write_text(latest_p.read_text())
        print(f"⚠️ reference missing -> created {ref_p}. Re-run monitor next time.")
        return

    latest = load_json(latest_p)
    ref = load_json(ref_p)

    def get(m, key, default=None):
        return (m.get(key) if isinstance(m, dict) else default)

    # expects metrics.json contains methods list with ndcg@10 etc (your format may differ)
    # We'll be robust: look for "methods" table or top-level.
    def ndcg10(obj, method="hybrid_ltr"):
        if "methods" in obj:
            for row in obj["methods"]:
                if row.get("method") == method:
                    return float(row.get("ndcg@10", 0.0))
        # fallback
        return float(obj.get("ndcg@10", 0.0))

    def p99(obj, method="hybrid_ltr"):
        if "methods" in obj:
            for row in obj["methods"]:
                if row.get("method") == method:
                    v = row.get("p99_ms")
                    return None if v in (None, "None") else float(v)
        return None

    ref_nd = ndcg10(ref, "hybrid_ltr")
    lat_nd = ndcg10(latest, "hybrid_ltr")
    drop = ref_nd - lat_nd

    ref_p99 = p99(ref, "hybrid_ltr")
    lat_p99 = p99(latest, "hybrid_ltr")

    drift = {
        "reference": {"ndcg10": ref_nd, "p99_ms": ref_p99},
        "latest": {"ndcg10": lat_nd, "p99_ms": lat_p99},
        "ndcg10_drop": drop,
        "trigger": {
            "ndcg_drop_threshold": args.ndcg_drop,
            "p99_max_ms": args.p99_max_ms,
        },
    }

    trigger = False
    reasons = []

    if drop >= args.ndcg_drop:
        trigger = True
        reasons.append(f"ndcg@10 drop {drop:.4f} >= {args.ndcg_drop:.4f}")

    if lat_p99 is not None and lat_p99 > args.p99_max_ms:
        trigger = True
        reasons.append(f"p99 {lat_p99:.1f}ms > {args.p99_max_ms:.1f}ms")

    drift["should_retrain"] = trigger
    drift["reasons"] = reasons
    drift["retrain_command"] = "python flows/train_ltr.py run"  # update to your flow name

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(drift, indent=2))
    print(f"✅ wrote {out_p}")
    print("RETRAIN?" , trigger, reasons)

if __name__ == "__main__":
    main()
