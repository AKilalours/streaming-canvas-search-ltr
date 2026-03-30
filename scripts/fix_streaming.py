"""Run this on your Mac to fix _STREAMING_ENABLED scope issue."""
import re

content = open('src/app/main.py').read()

# Remove any broken partial streaming blocks first
content = re.sub(
    r'\n# ── Real-time streaming.*?_STREAMING_ENABLED = False\n',
    '\n', content, flags=re.DOTALL
)
content = re.sub(
    r'\nimport asyncio as _asyncio.*?_STREAMING_ENABLED = False\n\n',
    '\n', content, flags=re.DOTALL
)

# Find line 17 area — after 'from app.cache import CacheClient'
TARGET = 'from app.cache import CacheClient'

STREAMING_BLOCK = """
# ── Kafka + WebSocket (real-time streaming layer) ─────────────────────────────
import asyncio as _asyncio
import time as _time_mod

try:
    from streaming.kafka_events import get_producer, InteractionEvent
    from streaming.websocket_feed import get_manager, make_interaction_ack
    from fastapi import WebSocket, WebSocketDisconnect
    _STREAMING_ENABLED = True
except Exception as _stream_err:
    _STREAMING_ENABLED = False
# ─────────────────────────────────────────────────────────────────────────────
"""

if TARGET in content and '_STREAMING_ENABLED' not in content:
    content = content.replace(TARGET, TARGET + STREAMING_BLOCK)
    open('src/app/main.py', 'w').write(content)
    import py_compile
    py_compile.compile('src/app/main.py', doraise=True)
    print("FIXED — _STREAMING_ENABLED added at module level, syntax OK")
elif '_STREAMING_ENABLED' in content:
    # Verify it's at module level (not inside a function)
    idx = content.find('_STREAMING_ENABLED = True')
    before = content[max(0,idx-500):idx]
    # Count indent level
    last_newline = before.rfind('\n')
    line_start = before[last_newline+1:]
    indent = len(line_start) - len(line_start.lstrip())
    if indent == 0:
        print("Already at module level — OK")
    else:
        print(f"PROBLEM: indented by {indent} spaces — inside a block")
else:
    print(f"Target '{TARGET}' not found in main.py")
    # Show first 30 lines
    for i, line in enumerate(content.split('\n')[:30]):
        print(f"{i:3}: {line}")
