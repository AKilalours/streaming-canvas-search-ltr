from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

try:
    import redis  # type: ignore
except Exception:
    redis = None  # type: ignore


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "sets": self.sets, "errors": self.errors}


class CacheClient:
    """
    Two-tier cache:
      - search results cache (fast)
      - answer cache (slower, expensive)
    Also provides per-process "singleflight" locks to reduce thundering herds.
    """

    def __init__(self, r: Any | None):
        self.r = r
        self.stats_search = CacheStats()
        self.stats_answer = CacheStats()
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    @classmethod
    def from_env(cls) -> "CacheClient":
        url = os.environ.get("REDIS_URL", "").strip()
        if not url or redis is None:
            return cls(None)
        try:
            r = redis.Redis.from_url(url, decode_responses=False, socket_timeout=0.25)
            r.ping()
            return cls(r)
        except Exception:
            return cls(None)

    def ok(self) -> bool:
        return self.r is not None

    def _lock_for(self, key: str) -> threading.Lock:
        with self._locks_guard:
            lk = self._locks.get(key)
            if lk is None:
                lk = threading.Lock()
                self._locks[key] = lk
            return lk

    def make_key(self, kind: str, payload: dict[str, Any]) -> str:
        # stable, small key
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return f"{kind}:{_sha(blob)}"

    def get_json(self, key: str, *, kind: str) -> dict[str, Any] | None:
        stats = self.stats_search if kind == "search" else self.stats_answer
        if self.r is None:
            stats.misses += 1
            return None
        try:
            raw = self.r.get(key)
            if raw is None:
                stats.misses += 1
                return None
            stats.hits += 1
            return json.loads(raw.decode("utf-8"))
        except Exception:
            stats.errors += 1
            return None

    def set_json(self, key: str, obj: dict[str, Any], *, ttl_s: int, kind: str) -> None:
        stats = self.stats_search if kind == "search" else self.stats_answer
        if self.r is None:
            return
        try:
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.r.setex(key, ttl_s, raw)
            stats.sets += 1
        except Exception:
            stats.errors += 1

    def singleflight(self, key: str, fn):
        """
        Per-process herd control: only one thread computes per key.
        Others wait and then read from cache (or proceed if still missing).
        """
        lk = self._lock_for(key)
        with lk:
            return fn()

    def stats(self) -> dict[str, Any]:
        return {
            "redis_enabled": self.ok(),
            "search": self.stats_search.to_dict(),
            "answer": self.stats_answer.to_dict(),
        }
