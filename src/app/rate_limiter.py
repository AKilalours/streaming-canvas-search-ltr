# src/app/rate_limiter.py
"""
Phase 6 — Production Rate Limiter
===================================
Redis sliding-window token-bucket rate limiter.
Gracefully degrades to allow-all if Redis is unavailable.

Env vars (configurable per deployment):
  RATE_LIMIT_SEARCH=60    requests/60s per IP (default)
  RATE_LIMIT_ANSWER=10    requests/60s per IP (expensive LLM calls)
  RATE_LIMIT_GLOBAL=1000  requests/60s total (circuit breaker)
"""
from __future__ import annotations

import os
import time
from typing import Any

from fastapi import HTTPException, Request


class TokenBucketRateLimiter:
    """Sliding-window counter backed by Redis sorted sets."""

    def __init__(self, redis_client: Any | None, limit: int, window_s: int = 60, key_prefix: str = "rl") -> None:
        self.redis = redis_client
        self.limit = int(limit)
        self.window_s = int(window_s)
        self.key_prefix = key_prefix

    def check(self, identifier: str) -> tuple[bool, dict[str, str]]:
        if self.redis is None:
            return True, {}
        key = f"{self.key_prefix}:{identifier}"
        now = int(time.time())
        window_start = now - self.window_s
        try:
            pipe = self.redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {f"{now}:{time.time_ns()}": now})
            pipe.expire(key, self.window_s + 5)
            results = pipe.execute()
            current_count = int(results[1])
            remaining = max(0, self.limit - current_count - 1)
            return current_count < self.limit, {
                "X-RateLimit-Limit": str(self.limit),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(now + self.window_s),
                "X-RateLimit-Window": str(self.window_s),
            }
        except Exception:
            return True, {}  # graceful degradation

    def require(self, identifier: str) -> dict[str, str]:
        allowed, headers = self.check(identifier)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={"error": "rate_limit_exceeded",
                        "message": f"Too many requests. Limit: {self.limit} per {self.window_s}s.",
                        "retry_after_s": self.window_s},
                headers={**headers, "Retry-After": str(self.window_s)},
            )
        return headers


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def build_rate_limiters(redis_client: Any | None) -> dict[str, TokenBucketRateLimiter]:
    return {
        "search": TokenBucketRateLimiter(redis_client, limit=int(os.environ.get("RATE_LIMIT_SEARCH", "60")), window_s=60, key_prefix="rl:search"),
        "answer": TokenBucketRateLimiter(redis_client, limit=int(os.environ.get("RATE_LIMIT_ANSWER", "10")), window_s=60, key_prefix="rl:answer"),
        "global": TokenBucketRateLimiter(redis_client, limit=int(os.environ.get("RATE_LIMIT_GLOBAL", "1000")), window_s=60, key_prefix="rl:global"),
    }
