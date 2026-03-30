#!/bin/bash
set -e
cd ~/streaming-canvas-search-ltr

echo "[1/5] Setting up streaming modules..."
mkdir -p src/streaming
cp src/streaming/kafka_events.py src/streaming/kafka_events.py
cp src/streaming/websocket_feed.py src/streaming/websocket_feed.py
touch src/streaming/__init__.py
echo "  src/streaming/ created"

echo "[2/5] Adding Kafka to docker-compose.yml..."
python3 - << 'PY'
content = open('docker-compose.yml').read()
if 'zookeeper' in content:
    print("  Kafka already in docker-compose.yml")
else:
    kafka_block = """
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    ports: ["2181:2181"]
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    depends_on: [zookeeper]
    ports:
      - "9092:9092"
      - "29092:29092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092,PLAINTEXT_HOST://localhost:29092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
    healthcheck:
      test: ["CMD-SHELL", "kafka-topics --bootstrap-server localhost:9092 --list"]
      interval: 15s
      timeout: 10s
      retries: 5

"""
    # Insert before volumes: section
    content = content.replace('\nvolumes:', kafka_block + '\nvolumes:', 1)
    open('docker-compose.yml', 'w').write(content)
    print("  Kafka + Zookeeper added")
PY

echo "[3/5] Adding WebSocket + Kafka endpoints to main.py..."
python3 - << 'PY'
content = open('src/app/main.py').read()

if 'ws/feed' in content:
    print("  WebSocket already wired in main.py")
else:
    streaming_import = '''
# ── Real-time streaming: Kafka + WebSocket ────────────────────────────────────
import asyncio, time as _time
try:
    from streaming.kafka_events import get_producer, InteractionEvent
    from streaming.websocket_feed import get_manager, make_interaction_ack
    from fastapi import WebSocket, WebSocketDisconnect
    _STREAMING_ENABLED = True
    print("[Streaming] Kafka + WebSocket enabled")
except ImportError as _se:
    _STREAMING_ENABLED = False
    print(f"[Streaming] Disabled: {_se}")
'''

    ws_endpoints = '''

# ── WebSocket: real-time feed ─────────────────────────────────────────────────

@app.websocket("/ws/feed/{user_id}")
async def ws_feed(websocket: WebSocket, user_id: str):
    """
    Real-time feed updates. Client connects once; server pushes updated
    rankings when interactions arrive or model updates.
    Netflix equivalent: feed rows update without page refresh.
    """
    if not _STREAMING_ENABLED:
        await websocket.close(code=1011, reason="Streaming not enabled")
        return
    manager = get_manager()
    await manager.connect(websocket, user_id)
    try:
        await websocket.send_json({
            "type": "connected",
            "user_id": user_id,
            "message": "StreamLens real-time feed active",
            "active_users": manager.active_users,
        })
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                if data.get("type") == "interaction" and _STREAMING_ENABLED:
                    import uuid as _uuid
                    producer = get_producer()
                    evt = InteractionEvent(
                        event_id=str(_uuid.uuid4()),
                        user_id=user_id,
                        doc_id=data.get("doc_id", ""),
                        title=data.get("title", ""),
                        event_type=data.get("event_type", "click"),
                        watch_pct=float(data.get("watch_pct", 0)),
                        position=int(data.get("position", 0)),
                        query=data.get("query", ""),
                        language=data.get("language", "English"),
                        session_id=data.get("session_id", ""),
                    )
                    producer.publish_interaction(evt)
                    await websocket.send_json(
                        make_interaction_ack(evt.event_id, evt.doc_id, evt.event_type)
                    )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "ts": _time.time()})
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(websocket, user_id)


@app.get("/ws/stats")
def ws_stats() -> dict:
    """WebSocket + Kafka connection stats — shown in Grafana dashboard."""
    if not _STREAMING_ENABLED:
        return {"enabled": False}
    manager = get_manager()
    producer = get_producer()
    return {
        "enabled": True,
        "active_users": manager.active_users,
        "active_connections": manager.active_connections,
        "kafka_mode": producer._mode,
    }


@app.post("/events/interaction")
async def log_interaction(
    user_id: str, doc_id: str,
    event_type: str = "click",
    watch_pct: float = 0.0,
    position: int = 0,
    query: str = "",
    language: str = "English",
) -> dict:
    """
    Log interaction to Kafka/Redis Streams.
    Powers: real-time popularity, propensity logging for IPW,
    retraining trigger at 10K interactions.
    """
    if not _STREAMING_ENABLED:
        return {"status": "streaming_disabled"}
    import uuid as _uuid
    producer = get_producer()
    evt = InteractionEvent(
        event_id=str(_uuid.uuid4()),
        user_id=user_id, doc_id=doc_id,
        title=_STATE.corpus.get(doc_id, {}).get("title", ""),
        event_type=event_type, watch_pct=watch_pct,
        position=position, query=query, language=language,
    )
    ok = producer.publish_interaction(evt)
    manager = get_manager()
    if manager.active_users > 0:
        await manager.send_to_user(user_id,
            make_interaction_ack(evt.event_id, evt.doc_id, evt.event_type))
    return {"status": "ok" if ok else "queued", "event_id": evt.event_id,
            "kafka_mode": producer._mode}
'''

    # Add import after spark_features import
    if 'from app.spark_features import' in content:
        content = content.replace(
            'from app.spark_features import',
            streaming_import + 'from app.spark_features import'
        )
    else:
        # Add near top after other imports
        content = streaming_import + content

    # Add endpoints at end of file
    content = content.rstrip() + '\n' + ws_endpoints + '\n'
    open('src/app/main.py', 'w').write(content)

    import py_compile
    py_compile.compile('src/app/main.py', doraise=True)
    print("  WebSocket + Kafka endpoints added — syntax OK")
PY

echo "[4/5] Installing kafka-python in container..."
docker compose exec api uv pip install kafka-python --quiet 2>/dev/null && echo "  kafka-python installed" || echo "  kafka-python install skipped (will use Redis fallback)"

echo "[5/5] Restarting API..."
docker compose up -d --force-recreate api
echo "  Waiting for startup..."
sleep 15

echo ""
echo "============================================"
echo "Testing WebSocket stats endpoint..."
curl -s http://localhost:8000/ws/stats | python3 -m json.tool

echo ""
echo "Testing interaction logging..."
curl -s -X POST "http://localhost:8000/events/interaction?user_id=chrisen&doc_id=296&event_type=watch_complete&watch_pct=0.95&position=3&query=crime+thriller" \
  | python3 -m json.tool

echo ""
echo "============================================"
echo "DONE. To start Kafka broker:"
echo "  docker compose up -d zookeeper kafka"
echo ""
echo "To test WebSocket (needs: npm install -g wscat):"
echo "  wscat -c ws://localhost:8000/ws/feed/chrisen"
echo "============================================"
