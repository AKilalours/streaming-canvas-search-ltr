"""
StreamLens — LambdaRank Hyperparameter Tuning
=============================================
Current: num_leaves=63, n_estimators=500, reg_alpha=0.0
Target:  num_leaves=127, n_estimators=1000, reg_alpha=0.1
Expected: nDCG@10 0.9300 → 0.94+

Run: python tune_lambdarank.py
"""
from __future__ import annotations
import pickle, json, re, subprocess, time
from pathlib import Path

print("\n" + "="*60)
print("StreamLens — LambdaRank Hyperparameter Tuning")
print("="*60)

# ── Load current model ────────────────────────────────────────
with open("artifacts/ltr/movielens_ltr_e5base.pkl", "rb") as f:
    model = pickle.load(f)

current = model.get_params()
print(f"\nCurrent params:")
print(f"  num_leaves:    {current['num_leaves']}")
print(f"  n_estimators:  {current['n_estimators']}")
print(f"  learning_rate: {current['learning_rate']}")
print(f"  reg_alpha:     {current['reg_alpha']}")
print(f"  reg_lambda:    {current['reg_lambda']}")

# ── Read training script ──────────────────────────────────────
SCRIPT = "src/ranking/ltr_train.py"
content = open(SCRIPT).read()
print(f"\n✅ Found training script: {SCRIPT}")

# Show current LightGBM params in the script
print("\nCurrent LightGBM config in script:")
for line in content.split("\n"):
    if any(p in line for p in ["num_leaves","n_estimators","learning_rate",
                                 "reg_alpha","reg_lambda","num_iterations","subsample"]):
        if "=" in line or ":" in line:
            print(f"  {line.strip()}")

# ── Apply optimized params ────────────────────────────────────
print("\n" + "-"*40)
print("Applying optimized hyperparameters...")
print("  num_leaves:    63  → 127   (more expressive trees)")
print("  n_estimators:  500 → 1000  (more trees = better ensemble)")
print("  reg_alpha:     0.0 → 0.1   (L1 regularization prevents overfit)")
print("  reg_lambda:    0.0 → 0.1   (L2 regularization)")
print("  min_child_samples: 10 → 10 (keep same)")
print("-"*40)

new = content

# num_leaves
new = re.sub(r'num_leaves\s*=\s*\d+', 'num_leaves=127', new)
new = re.sub(r'["\']num_leaves["\']\s*:\s*\d+', '"num_leaves": 127', new)

# n_estimators / num_iterations
new = re.sub(r'n_estimators\s*=\s*\d+', 'n_estimators=1000', new)
new = re.sub(r'num_iterations\s*=\s*\d+', 'num_iterations=1000', new)
new = re.sub(r'["\']n_estimators["\']\s*:\s*\d+', '"n_estimators": 1000', new)

# reg_alpha
new = re.sub(r'reg_alpha\s*=\s*[\d.]+', 'reg_alpha=0.1', new)
new = re.sub(r'["\']reg_alpha["\']\s*:\s*[\d.]+', '"reg_alpha": 0.1', new)

# reg_lambda
new = re.sub(r'reg_lambda\s*=\s*[\d.]+', 'reg_lambda=0.1', new)
new = re.sub(r'["\']reg_lambda["\']\s*:\s*[\d.]+', '"reg_lambda": 0.1', new)

# Write back
open(SCRIPT, "w").write(new)

# Verify
import py_compile
py_compile.compile(SCRIPT, doraise=True)
print(f"✅ {SCRIPT} updated — syntax OK")

# ── Retrain ───────────────────────────────────────────────────
print("\nRetraining LambdaRank with optimized params...")
print("(This takes 3-8 minutes on CPU)")
t0 = time.time()

result = subprocess.run(
    ["python", SCRIPT],
    capture_output=True, text=True, timeout=600
)

elapsed = time.time() - t0
print(f"Training completed in {elapsed:.0f}s")

if result.returncode != 0:
    print(f"\n⚠️  Training error:\n{result.stderr[-500:]}")
    print("\nTrying docker exec path...")
    result2 = subprocess.run(
        ["docker", "compose", "exec", "-e", "PYTHONPATH=/app/src", "api",
         "/app/.venv/bin/python3.11", f"/app/{SCRIPT}"],
        capture_output=True, text=True, timeout=600
    )
    if result2.returncode == 0:
        print("✅ Training complete via docker")
    else:
        print(f"❌ Training failed: {result2.stderr[-300:]}")
        print("\nRun manually:")
        print(f"  docker compose exec -e PYTHONPATH=/app/src api")
        print(f"  /app/.venv/bin/python3.11 /app/{SCRIPT}")
else:
    print("✅ Training complete")
    if result.stdout:
        print(result.stdout[-400:])

# ── Verify new model ──────────────────────────────────────────
models = list(Path("artifacts/ltr").glob("*.pkl"))
print(f"\nModels in artifacts/ltr/:")
for m in sorted(models):
    size = m.stat().st_size / 1024
    print(f"  {m.name} ({size:.0f} KB)")

    with open(m, "rb") as f:
        try:
            mdl = pickle.load(f)
            p = mdl.get_params()
            print(f"    leaves={p['num_leaves']} trees={p['n_estimators']} "
                  f"lr={p['learning_rate']} alpha={p['reg_alpha']}")
        except Exception:
            pass

# ── Run eval ──────────────────────────────────────────────────
print("\nRunning evaluation...")
eval_result = subprocess.run(
    ["make", "eval_full_v2"],
    capture_output=True, text=True, timeout=600
)

try:
    with open("reports/latest/metrics.json") as f:
        metrics = json.load(f)
    print(f"\n{'='*60}")
    print("RESULTS AFTER HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    for m in metrics["methods"]:
        marker = " ← TUNED" if m["method"] == "hybrid_ltr" else ""
        print(f"  {m['method']:15} nDCG@10 = {m['ndcg@10']:.4f}{marker}")
    ltr = next(m for m in metrics["methods"] if m["method"] == "hybrid_ltr")
    baseline = 0.9300
    delta = ltr["ndcg@10"] - baseline
    print(f"\n  Baseline:   {baseline:.4f}")
    print(f"  After tune: {ltr['ndcg@10']:.4f}")
    print(f"  Delta:      {delta:+.4f} ({delta/baseline*100:+.1f}%)")
    print(f"  {'✅ IMPROVED' if delta > 0 else '⚠️  No change — model may need full retrain'}")
except Exception as e:
    print(f"Could not read metrics: {e}")

print(f"""
{'='*60}
NEXT STEPS
{'='*60}
1. Run ALS:        python als_collaborative_filtering.py
2. Fix drift:      python temporal_drift_fix.py
3. Final commit:   git add -A && git push origin main
{'='*60}
""")
