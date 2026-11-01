#!/bin/bash
# Test Why This, RAG, and VLM across 10 languages
cd ~/streaming-canvas-search-ltr

echo "============================================"
echo "TESTING ALL EXPLANATION ENDPOINTS"
echo "============================================"

LANGUAGES=("English" "Arabic" "Japanese" "French" "Hindi" "Korean" "Spanish" "Chinese" "German" "Portuguese")
DOC_ID="296"  # Pulp Fiction
PROFILE="chrisen"

for lang in "${LANGUAGES[@]}"; do
    echo ""
    echo "─── $lang ───"
    
    # Why This
    WHY=$(curl -s "http://localhost:8000/explain?doc_id=$DOC_ID&profile=$PROFILE&language=$lang" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print('WHY:', d.get('answer','ERROR')[:120])")
    echo "$WHY"
    
    # VLM
    VLM=$(curl -s "http://localhost:8000/vlm/describe_poster?doc_id=$DOC_ID&title=Pulp+Fiction&language=$lang" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print('VLM:', d.get('text','ERROR')[:100], '| src:', d.get('source','?')[:20])")
    echo "$VLM"
done

echo ""
echo "─── Arabic RAG (agentic) ───"
curl -s "http://localhost:8000/explain?doc_id=$DOC_ID&profile=$PROFILE&language=Arabic&agentic=true" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('answer','ERROR')[:300])"

echo ""
echo "─── Japanese RAG (agentic) ───"
curl -s "http://localhost:8000/explain?doc_id=$DOC_ID&profile=$PROFILE&language=Japanese&agentic=true" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('answer','ERROR')[:300])"

echo "============================================"
echo "DONE"
echo "============================================"
