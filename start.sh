#!/bin/bash
echo "Starting StreamLens..."
colima start
sleep 10
cd ~/streaming-canvas-search-ltr
OPENAI_API_KEY=$(grep "^OPENAI_API_KEY" .env | head -1 | cut -d= -f2-) docker compose up -d
until curl -s http://localhost:8000/health | python3 -c \
  "import sys,json; d=json.load(sys.stdin); exit(0 if d['ready'] else 1)" \
  2>/dev/null; do echo "loading..."; sleep 8; done
echo "✅ READY"
open http://localhost:8000/demo
