#!/bin/bash
# StreamLens — Fix AI Stack HTML values + push everything
# Run from project root: bash fix_html_and_push.sh

echo "=== Fixing AI Stack values in index.html ==="

# Fix 1: Old LTR value 0.8589 → 0.9300 in the top metrics section (line 444)
sed -i '' 's|<div class="num" style="color:#3ddc84">0.8589</div><div class="lbl">LTR nDCG@10</div><div class="desc">Phenomenal · e5-base-v2 + LambdaRank</div>|<div class="num" style="color:#3ddc84">0.9300</div><div class="lbl">LTR nDCG@10</div><div class="desc">Extraordinary · e5-base-v2 + LambdaRank</div>|g' src/app/demo_ui/index.html

# Fix 2: Old Dense value 0.4640 → 0.5496 in top metrics (line 445)
sed -i '' 's|<div class="num" style="color:#76b4ff">0.4640</div><div class="lbl">Dense nDCG@10</div><div class="desc">Strong · intfloat/e5-base-v2 768-dim</div>|<div class="num" style="color:#76b4ff">0.5496</div><div class="lbl">Dense nDCG@10</div><div class="desc">Fine-tuned +18.4% · e5-base-v2 768-dim</div>|g' src/app/demo_ui/index.html

# Fix 3: Old Hybrid value 0.5848 → 0.5891
sed -i '' 's|0.5848</div><div class="lbl">Hybrid nDCG@10</div>|0.5891</div><div class="lbl">Hybrid nDCG@10</div>|g' src/app/demo_ui/index.html

# Fix 4: Old LTR 0.9499 → 0.9300 everywhere
sed -i '' 's/0\.9499/0.9300/g' src/app/demo_ui/index.html

# Fix 5: Old ablation text with 0.8589
sed -i '' 's/BM25 → 0\.61 · Dense → 0\.46 · Hybrid → 0\.58 → <strong style="color:#3ddc84">LTR → 0\.8589<\/strong>/BM25 → 0.61 · Dense → 0.55 · Hybrid → 0.59 → <strong style="color:#3ddc84">LTR → 0.9300<\/strong>/g' src/app/demo_ui/index.html

# Fix 6: Footer LTR value
sed -i '' 's/LTR nDCG@10 = 0\.8589/LTR nDCG@10 = 0.9300/g' src/app/demo_ui/index.html

# Fix 7: Fine-tune comparison text  
sed -i '' 's/0\.85 → 0\.9300/0.8589 → 0.9300/g' src/app/demo_ui/index.html

# Fix 8: LTR label in table
sed -i '' 's/LTR nDCG@10 (tuned 1000 trees)/LTR nDCG@10 (LambdaRank 500 trees)/g' src/app/demo_ui/index.html

# Fix 9: sidebar badge
sed -i '' 's/<strong id="lval">0\.9300<\/strong>/<strong id="lval">0.9300<\/strong>/g' src/app/demo_ui/index.html

# Fix 10: Spearman fine-tune line
sed -i '' 's/Spearman: <strong>0\.55 → 0\.81<\/strong> · Dense nDCG: <strong>0\.46 → 0\.55<\/strong> · LTR: <strong style="color:#3ddc84">0\.85 → 0\.9300<\/strong>/Spearman: <strong>0.68 → 0.79<\/strong> · Dense nDCG: <strong>0.46 → 0.55<\/strong> · LTR: <strong style="color:#3ddc84">0.8589 → 0.9300<\/strong>/g' src/app/demo_ui/index.html

echo "=== Verifying fixes ==="
echo "--- Checking for old 0.8589 value ---"
grep -c "0\.8589" src/app/demo_ui/index.html && echo "Still has 0.8589 occurrences" || echo "✅ 0.8589 cleaned"
echo "--- Checking for old 0.9499 value ---"
grep -c "0\.9499" src/app/demo_ui/index.html && echo "Still has 0.9499 occurrences" || echo "✅ 0.9499 cleaned"
echo "--- Checking 0.9300 is present ---"
grep -c "0\.9300" src/app/demo_ui/index.html
echo "--- Key metric lines ---"
grep -n "0\.9300\|0\.5496\|0\.5891" src/app/demo_ui/index.html | head -10

echo ""
echo "=== Pushing everything to GitHub ==="
git add src/app/demo_ui/index.html
git add src/genai/openai_explain.py
git add README.md
git add MLOPS.md
git add LICENSE
git add project_description.md

git status

git commit -m "feat: fix AI Stack metrics (0.9300 LTR, 0.5496 dense, 0.5891 hybrid), phenomenal explanations, TMDB posters, full docs"

git push origin main

echo ""
echo "=== DONE ==="
echo "✅ All files pushed to GitHub"
echo "✅ AI Stack now shows: LTR=0.9300, Dense=0.5496, Hybrid=0.5891"
echo ""
echo "Rebuild Docker to see UI changes:"
echo "OPENAI_API_KEY=\$(grep '^OPENAI_API_KEY' .env | head -1 | cut -d= -f2-) docker compose up -d --force-recreate api"
