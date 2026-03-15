# src/app/observability.py
"""
Production-grade observability for StreamLens.
Exposes all key metrics for Prometheus scraping + Grafana dashboards.
"""
from __future__ import annotations
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response

# ── Request metrics ──────────────────────────────────────────────────────────
REQ_COUNT = Counter(
    "api_requests_total", "Total API requests", ["path", "method", "status"]
)
REQ_LAT_MS = Histogram(
    "api_request_latency_ms", "Request latency ms", ["path", "method"],
    buckets=[5, 10, 25, 50, 100, 200, 300, 500, 1000, 2000]
)

# ── Retrieval metrics ────────────────────────────────────────────────────────
RETRIEVAL_LAT_MS = Histogram(
    "retrieval_latency_ms", "BM25+FAISS retrieval latency ms", ["method"],
    buckets=[1, 5, 10, 25, 50, 100, 200]
)
RERANKER_LAT_MS = Histogram(
    "reranker_latency_ms", "LTR reranker latency ms", ["ranker"],
    buckets=[1, 5, 10, 25, 50, 100]
)
EXPLAIN_LAT_MS = Histogram(
    "explain_latency_ms", "GPT explanation latency ms", ["type"],
    buckets=[100, 500, 1000, 2000, 5000, 10000]
)

# ── Cache metrics ────────────────────────────────────────────────────────────
CACHE_HITS = Counter("cache_hits_total", "Redis cache hits", ["cache_type"])
CACHE_MISSES = Counter("cache_misses_total", "Redis cache misses", ["cache_type"])

# ── Model quality metrics ────────────────────────────────────────────────────
MODEL_NDCG = Gauge("model_ndcg_at_10", "Latest model nDCG@10")
MODEL_MRR = Gauge("model_mrr", "Latest model MRR")
MODEL_DRIFT = Gauge("model_ndcg_drift", "nDCG drift vs reference (negative = degradation)")
LTR_LIFT = Gauge("ltr_lift_ndcg", "LTR lift over hybrid baseline")

# ── Business metrics ─────────────────────────────────────────────────────────
SEARCHES_TOTAL = Counter("searches_total", "Total search requests", ["method", "profile"])
FEED_REQUESTS = Counter("feed_requests_total", "Total feed requests", ["profile"])
EXPLAINS_TOTAL = Counter("explains_total", "Total explain requests", ["type", "language"])
VOICE_REQUESTS = Counter("voice_requests_total", "TTS/transcription requests", ["type"])
AD_IMPRESSIONS = Counter("ad_impressions_total", "Ad impressions served", ["slot"])
LIVE_EVENTS_ACTIVE = Gauge("live_events_active", "Currently active live events")

# ── Infrastructure metrics ───────────────────────────────────────────────────
CORPUS_SIZE = Gauge("corpus_size_docs", "Number of documents in corpus")
KG_EDGES = Gauge("knowledge_graph_edges", "Number of knowledge graph edges")
CLIP_CACHE_SIZE = Gauge("clip_cache_embeddings", "Number of cached CLIP embeddings")
RATE_LIMITED_TOTAL = Counter("rate_limited_requests_total", "Requests rejected by rate limiter")


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_model_metrics(ndcg: float, mrr: float, lift: float, drift: float = 0.0) -> None:
    MODEL_NDCG.set(ndcg)
    MODEL_MRR.set(mrr)
    LTR_LIFT.set(lift)
    MODEL_DRIFT.set(drift)
