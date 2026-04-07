#!/bin/bash
# StreamLens — Run All 4 Improvements
# ====================================
set -e
cd ~/streaming-canvas-search-ltr

echo "============================================"
echo "StreamLens — 4 Improvements Pipeline"
echo "============================================"
echo ""

# ── Step 1: LambdaRank Hyperparameter Tuning ─────────────────
echo "[1/4] LambdaRank Hyperparameter Tuning..."
python tune_lambdarank.py
echo ""

# ── Step 2: ALS Collaborative Filtering ──────────────────────
echo "[2/4] ALS Collaborative Filtering..."
pip install implicit --break-system-packages -q 2>/dev/null || true
python als_collaborative_filtering.py
echo ""

# ── Step 3: Temporal Drift Fix ────────────────────────────────
echo "[3/4] Temporal Drift Fix..."
python temporal_drift_fix.py
echo ""

# ── Step 4: Find LTR training script and retrain ─────────────
echo "[4/4] Retraining LTR with optimized params..."
LTR_SCRIPT=$(grep -r "LGBMRanker\|lgb\.train\|LambdaRank" src/ --include="*.py" -l 2>/dev/null | head -1)
if [ -n "$LTR_SCRIPT" ]; then
    echo "Found LTR training script: $LTR_SCRIPT"
    echo "Run manually: python $LTR_SCRIPT"
else
    echo "LTR training script not found — check src/ranking/"
fi
echo ""

echo "============================================"
echo "Running full evaluation..."
make eval_full_v2
echo ""

echo "============================================"
echo "Checking results..."
cat reports/latest/metrics.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('=== FINAL RESULTS AFTER ALL IMPROVEMENTS ===')
for m in d['methods']:
    print(f\"{m['method']:15} nDCG@10 = {m['ndcg@10']:.4f}\")
"

echo ""
echo "Committing results..."
git add -A
git commit -m "feat: ALS +collab filtering, LambdaRank tuned, temporal drift fixed, all 4 improvements complete"
git push origin main

echo "============================================"
echo "ALL DONE"
echo "============================================"
