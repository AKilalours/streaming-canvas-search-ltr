#!/bin/bash
set -e
cd ~/streaming-canvas-search-ltr

echo "=== Fixing _STREAMING_ENABLED ==="
python3 - << 'PY'
content = open('src/app/main.py').read()

# Clean any broken partial blocks
import re
content = re.sub(r'\n# ── (Real-time|Kafka).*?_STREAMING_ENABLED = False\n+',
                 '\n', content, flags=re.DOTALL)
content = re.sub(r'\nimport asyncio as _asyncio.*?_STREAMING_ENABLED = False\n+',
                 '\n', content, flags=re.DOTALL)

TARGET = 'from app.cache import CacheClient'
BLOCK = """
from app.cache import CacheClient

# ── Kafka + WebSocket ─────────────────────────────────────────────────────────
import asyncio as _asyncio
import time as _time_mod
try:
    from streaming.kafka_events import get_producer, InteractionEvent
    from streaming.websocket_feed import get_manager, make_interaction_ack
    from fastapi import WebSocket, WebSocketDisconnect
    _STREAMING_ENABLED = True
except Exception:
    _STREAMING_ENABLED = False
# ─────────────────────────────────────────────────────────────────────────────
"""

content = content.replace(TARGET, BLOCK.strip())
open('src/app/main.py', 'w').write(content)
import py_compile; py_compile.compile('src/app/main.py', doraise=True)

# Verify at module level
idx = content.find('_STREAMING_ENABLED = True')
before = content[max(0,idx-100):idx]
indent = len(before.split('\n')[-1]) - len(before.split('\n')[-1].lstrip())
print(f"_STREAMING_ENABLED at module level: indent={indent} ({'OK' if indent==0 else 'WRONG'})")
PY

echo "=== Restarting API ==="
docker compose up -d --force-recreate api
sleep 20

echo "=== Testing ==="
docker compose logs api 2>&1 | grep -E "Streaming|STREAMING|kafka|WebSocket" | head -5
curl -s http://localhost:8000/ws/stats | python3 -m json.tool
curl -s -X POST "http://localhost:8000/events/interaction?user_id=chrisen&doc_id=296&event_type=watch_complete&watch_pct=0.95&position=3" | python3 -m json.tool
