"""
StreamLens — Kafka Event Streaming
Real-time user interaction events published to Kafka topics.
Falls back to Redis Streams if Kafka unavailable.
"""
from __future__ import annotations
import json, time, os
from dataclasses import dataclass, asdict


@dataclass
class InteractionEvent:
    event_id: str
    user_id: str
    doc_id: str
    title: str
    event_type: str      # click | watch_start | watch_complete | skip | thumbs_up
    watch_pct: float = 0.0
    position: int = 0
    query: str = ""
    language: str = "English"
    session_id: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.event_id:
            import uuid; self.event_id = str(uuid.uuid4())


@dataclass
class ImpressionEvent:
    user_id: str
    query: str
    shown_doc_ids: list
    scores: list
    model_version: str
    alpha: float = 0.2
    candidate_k: int = 2000
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class ModelUpdateEvent:
    trigger: str
    current_ndcg: float
    baseline_ndcg: float
    drift_pct: float
    n_new_interactions: int
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class StreamLensProducer:
    TOPICS = {
        "interactions":  "streamlens.interactions",
        "impressions":   "streamlens.impressions",
        "feedback":      "streamlens.feedback",
        "model_updates": "streamlens.model_updates",
    }

    def __init__(self, bootstrap_servers="localhost:9092", redis_fallback=True):
        self._kafka = None
        self._redis = None
        self._mode = "noop"

        try:
            from kafka import KafkaProducer
            self._kafka = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all", retries=3, max_block_ms=1000,
            )
            self._mode = "kafka"
            print(f"[Kafka] Connected to {bootstrap_servers}")
        except Exception as e:
            print(f"[Kafka] Not available ({e})")

        if self._mode == "noop" and redis_fallback:
            try:
                import redis
                self._redis = redis.Redis.from_url(
                    os.environ.get("REDIS_URL", "redis://localhost:6379"),
                    decode_responses=True)
                self._redis.ping()
                self._mode = "redis_streams"
                print("[Kafka] Using Redis Streams fallback")
            except Exception as e:
                print(f"[Kafka] Redis also unavailable — events dropped")

    def publish_interaction(self, event: InteractionEvent) -> bool:
        return self._publish("interactions", event.user_id, asdict(event))

    def publish_impression(self, event: ImpressionEvent) -> bool:
        return self._publish("impressions", event.user_id, asdict(event))

    def publish_model_update(self, event: ModelUpdateEvent) -> bool:
        return self._publish("model_updates", "system", asdict(event))

    def _publish(self, topic_key: str, key: str, payload: dict) -> bool:
        topic = self.TOPICS[topic_key]
        if self._mode == "kafka":
            try:
                self._kafka.send(topic, key=key, value=payload).get(timeout=1.0)
                return True
            except Exception as e:
                print(f"[Kafka] Publish failed: {e}"); return False
        elif self._mode == "redis_streams":
            try:
                self._redis.xadd(f"streamlens:{topic_key}",
                    {"key": key, "payload": json.dumps(payload)}, maxlen=10000)
                return True
            except Exception as e:
                print(f"[Redis Streams] Publish failed: {e}"); return False
        return False

    def close(self):
        if self._kafka:
            self._kafka.flush(); self._kafka.close()


class StreamLensConsumer:
    def __init__(self, redis_url="redis://localhost:6379"):
        try:
            import redis
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
        except Exception:
            self._redis = None

    def process_interaction(self, event: dict) -> None:
        if not self._redis: return
        doc_id = event.get("doc_id", "")
        user_id = event.get("user_id", "")
        evt_type = event.get("event_type", "")
        watch_pct = float(event.get("watch_pct", 0))

        if evt_type in ("click", "watch_start"):
            self._redis.incr(f"rt:popularity:{doc_id}")
            self._redis.expire(f"rt:popularity:{doc_id}", 86400)

        if evt_type == "watch_complete" and watch_pct > 0.8:
            self._redis.incr(f"rt:completions:{doc_id}")
            self._redis.expire(f"rt:completions:{doc_id}", 604800)

        if evt_type in ("watch_complete", "thumbs_up"):
            self._redis.incr(f"rt:user_affinity:{user_id}:{doc_id}")
            self._redis.expire(f"rt:user_affinity:{user_id}:{doc_id}", 2592000)

        self._redis.incr("rt:interaction_count")

    def get_realtime_popularity(self, doc_id: str) -> int:
        if not self._redis: return 0
        try: return int(self._redis.get(f"rt:popularity:{doc_id}") or 0)
        except Exception: return 0

    def should_retrain(self, threshold=10000) -> bool:
        if not self._redis: return False
        try: return int(self._redis.get("rt:interaction_count") or 0) >= threshold
        except Exception: return False


_PRODUCER = None

def get_producer() -> StreamLensProducer:
    global _PRODUCER
    if _PRODUCER is None:
        _PRODUCER = StreamLensProducer(
            bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092"),
            redis_fallback=True)
    return _PRODUCER
