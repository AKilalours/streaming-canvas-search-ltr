from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    search_hits: int = 0
    search_misses: int = 0
    answer_hits: int = 0
    answer_misses: int = 0


class RedisCache:
    """
    Two-tier cache:
      - Tier 1: search results
      - Tier 2: answers (RAG)
    """

    def __init__(self, url: str):
        if redis is None:
            raise RuntimeError("redis package not installed")
        self.url = url
        self.enabled = True
        self.r = redis.Redis.from_url(url, decode_responses=True)
        self._stats = CacheStats()

    @classmethod
    def from_env(cls) -> "RedisCache | None":
        url = os.environ.get("REDIS_URL", "").strip()
        if not url:
            return None
        try:
            c = cls(url)
            # sanity ping
            c.r.ping()
            return c
        except Exception:
            return None

    def _key(self, prefix: str, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return f"{prefix}:{_sha(raw)}"

    def get_json(self, prefix: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, float]:
        t0 = _now_ms()
        if not self.enabled:
            return None, 0.0
        k = self._key(prefix, payload)
        v = self.r.get(k)
        ms = _now_ms() - t0
        if not v:
            return None, ms
        try:
            return json.loads(v), ms
        except Exception:
            return None, ms

    def set_json(self, prefix: str, payload: dict[str, Any], value: dict[str, Any], ttl_s: int = 60) -> None:
        if not self.enabled:
            return
        k = self._key(prefix, payload)
        self.r.setex(k, ttl_s, json.dumps(value, ensure_ascii=False))

    def bump(self, which: str, hit: bool) -> None:
        # local counters
        if which == "search":
            if hit:
                self._stats.search_hits += 1
            else:
                self._stats.search_misses += 1
        elif which == "answer":
            if hit:
                self._stats.answer_hits += 1
            else:
                self._stats.answer_misses += 1

        # redis counters (best effort)
        try:
            name = f"stats:{which}:{'hit' if hit else 'miss'}"
            self.r.incr(name, 1)
        except Exception:
            pass

    def stats(self) -> dict[str, Any]:
        out = {
            "redis_url": self.url,
            "enabled": bool(self.enabled),
            "local": {
                "search_hits": self._stats.search_hits,
                "search_misses": self._stats.search_misses,
                "answer_hits": self._stats.answer_hits,
                "answer_misses": self._stats.answer_misses,
            },
            "redis": {},
        }
        try:
            keys = [
                "stats:search:hit",
                "stats:search:miss",
                "stats:answer:hit",
                "stats:answer:miss",
            ]
            vals = self.r.mget(keys)
            out["redis"] = dict(zip(keys, [int(v or 0) for v in vals]))
        except Exception:
            pass
        return out
