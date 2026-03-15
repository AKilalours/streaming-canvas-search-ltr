from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Metaflow Run Dashboard", layout="wide")
st.title("Metaflow Run Comparison Dashboard")

REPORTS = Path("reports")
LATEST = REPORTS / "latest" / "metrics.json"


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_metrics_files():
    out = []
    if not REPORTS.exists():
        return out
    for p in REPORTS.rglob("metrics.json"):
        out.append(p)
    out.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return out


def metaflow_runs_cli(flow: str):
    try:
        s = subprocess.check_output(["metaflow", "show", flow], text=True)
        return s
    except Exception as e:
        return f"(metaflow CLI not available or flow not found) {e}"


left, right = st.columns([2, 1])

with right:
    st.subheader("Latest Metrics")
    m = read_json(LATEST)
    if m is None:
        st.warning("reports/latest/metrics.json not found")
    else:
        st.json(m.get("diagnostics", {}))
        st.write("Methods:")
        st.dataframe(m.get("methods", []), use_container_width=True)

with left:
    st.subheader("Compare Two Runs (metrics.json)")
    files = list_metrics_files()
    if not files:
        st.info("No reports/**/metrics.json files found yet.")
    else:
        labels = [str(p) for p in files[:50]]
        a = st.selectbox("Run A", labels, index=0)
        b = st.selectbox("Run B", labels, index=1 if len(labels) > 1 else 0)

        ma = read_json(Path(a)) or {}
        mb = read_json(Path(b)) or {}

        def to_table(m):
            rows = m.get("methods", [])
            out = {}
            for r in rows:
                meth = r.get("method")
                if meth:
                    out[meth] = r
            return out

        ta = to_table(ma)
        tb = to_table(mb)
        all_methods = sorted(set(ta.keys()) | set(tb.keys()))
        rows = []
        for meth in all_methods:
            ra = ta.get(meth, {})
            rb = tb.get(meth, {})
            rows.append({
                "method": meth,
                "A_ndcg@10": ra.get("ndcg@10"),
                "B_ndcg@10": rb.get("ndcg@10"),
                "Δ_ndcg@10": (rb.get("ndcg@10") - ra.get("ndcg@10")) if (isinstance(ra.get("ndcg@10"), (int,float)) and isinstance(rb.get("ndcg@10"), (int,float))) else None,
                "A_p99_ms": ra.get("p99_ms"),
                "B_p99_ms": rb.get("p99_ms"),
                "Δ_p99_ms": (rb.get("p99_ms") - ra.get("p99_ms")) if (isinstance(ra.get("p99_ms"), (int,float)) and isinstance(rb.get("p99_ms"), (int,float))) else None,
            })
        st.dataframe(rows, use_container_width=True)

st.divider()
st.subheader("Metaflow CLI quick view (optional)")
flow_name = st.text_input("Flow name (example: MyFlow)", value=os.environ.get("MF_FLOW_NAME", ""))
if flow_name.strip():
    st.code(metaflow_runs_cli(flow_name.strip()))
else:
    st.caption("Set MF_FLOW_NAME env var or type your flow name to view CLI output.")
