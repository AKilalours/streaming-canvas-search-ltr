from __future__ import annotations
import sys
from metaflow import Flow

FLOW = sys.argv[1] if len(sys.argv) > 1 else "PhenomenalLTRFlow"
f = Flow(FLOW)

runs = list(f.runs())
print(f"Flow: {FLOW}")
print("Runs (newest first):")
for r in runs[:10]:
    print(" ", r.id, "finished=", r.finished)

if not runs:
    raise SystemExit(0)

latest = runs[0]
print("\nLatest run:", latest.id)
print("Steps:", [s.id for s in latest.steps()])

if "train" in [s.id for s in latest.steps()]:
    step = latest["train"]
    tasks = list(step.tasks())
    print("\nTrain tasks:", [t.id for t in tasks])
    for t in tasks[:3]:
        print(f"\n--- train/{t.id} stdout ---")
        try:
            print(t.stdout)
        except Exception as e:
            print("no stdout:", e)
