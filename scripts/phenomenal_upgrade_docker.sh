#!/usr/bin/env bash
set -euo pipefail

echo "===> [1/7] Fix HF dependency conflict (transformers requires huggingface_hub < 1.0)"
python -m pip install -U "huggingface_hub>=0.34,<1.0" >/dev/null

echo "===> [2/7] Ensure Redis client dependency exists"
python -m pip install -U redis >/dev/null

echo "===> [3/7] Write Redis cache helper"
mkdir -p src/utils
cat > src/utils/cache.py <<'PY'
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
PY

echo "===> [4/7] Patch schemas.py to add cache_hit + context knobs (safe overwrite)"
cat > src/app/schemas.py <<'PY'
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    method: str = Field("bm25", description="bm25 | dense | hybrid | hybrid_ltr")
    k: int = Field(10, ge=1, le=100)
    candidate_k: int = Field(200, ge=10, le=5000)
    rerank_k: int = Field(50, ge=1, le=2000)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    debug: bool = False

    # phenomenal knobs (optional)
    device_type: str | None = None
    network_speed: str | None = None
    profile: str | None = None


class SearchHit(BaseModel):
    doc_id: str
    score: float
    title: str | None = None
    text: str | None = None
    score_breakdown: dict[str, float] | None = None


class SearchResponse(BaseModel):
    query: str
    method: str
    k: int
    hits: list[SearchHit]
    timings_ms: dict[str, float] | None = None
    cache_hit: bool | None = None


class Source(BaseModel):
    doc_id: str
    title: str | None = None
    snippet: str | None = None
    score: float | None = None
    score_breakdown: dict[str, float] | None = None


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    method: str = Field("hybrid_ltr", description="bm25 | dense | hybrid | hybrid_ltr")
    k: int = Field(10, ge=1, le=100)
    candidate_k: int = Field(200, ge=10, le=5000)
    rerank_k: int = Field(50, ge=1, le=2000)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    context_k: int = Field(6, ge=1, le=20)

    temperature: float = Field(0.2, ge=0.0, le=1.5)
    max_tokens: int = Field(400, ge=64, le=4096)
    debug: bool = False

    # phenomenal knobs (optional)
    device_type: str | None = None
    network_speed: str | None = None
    profile: str | None = None


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    timings_ms: dict[str, float] | None = None
    warning: str | None = None
    raw: dict[str, Any] | None = None
    cache_hit: bool | None = None
PY

echo "===> [5/7] Patch FastAPI main.py for /cache/stats + two-tier caching (surgical patch)"
python - <<'PY'
from __future__ import annotations
from pathlib import Path

p = Path("src/app/main.py")
txt = p.read_text(encoding="utf-8")

if "from utils.cache import RedisCache" not in txt:
    # insert import near utils.logging
    marker = "from utils.logging import get_logger\n"
    if marker in txt:
        txt = txt.replace(marker, marker + "from utils.cache import RedisCache\n")
    else:
        raise SystemExit("Can't find get_logger import marker to patch main.py")

# add global cache handle
if "_CACHE:" not in txt:
    marker = "_OLLAMA: OllamaClient | None = None\n"
    if marker in txt:
        txt = txt.replace(marker, marker + "_CACHE: RedisCache | None = None\n")
    else:
        raise SystemExit("Can't find _OLLAMA global marker to patch main.py")

# init cache in lifespan
if "RedisCache.from_env()" not in txt:
    marker = "STATE = load_state()\n"
    if marker in txt:
        txt = txt.replace(marker, marker + "\n        # Redis cache (two-tier)\n        _CACHE = RedisCache.from_env()\n")
    else:
        raise SystemExit("Can't find STATE = load_state() marker to patch main.py")

# ensure redis status in startup log line (optional) + health
if '"redis_enabled"' not in txt:
    # patch health() return dict
    marker = '"ltr_path": str(getattr(st, "ltr_path", "")) if st else "",\n'
    if marker in txt:
        txt = txt.replace(marker, marker + '        "redis_enabled": bool(_CACHE is not None),\n')
    else:
        # fallback: do nothing
        pass

# add /cache/stats endpoint if missing
if "@app.get(\"/cache/stats\")" not in txt:
    insert_after = "@app.get(\"/metrics/lift\")\n"
    idx = txt.find(insert_after)
    if idx == -1:
        raise SystemExit("Can't find /metrics/lift endpoint marker to insert /cache/stats")
    # insert after metrics_lift() block end by searching next blank line after that function
    # crude but works: append at end of metrics section by inserting before suggestions section header
    needle = "# -----------------------------\n# Suggestions (typeahead + intent)\n# -----------------------------\n"
    j = txt.find(needle)
    if j == -1:
        raise SystemExit("Can't find Suggestions section marker")
    block = """
@app.get("/cache/stats")
def cache_stats() -> dict[str, object]:
    if _CACHE is None:
        return {"enabled": False, "error": "REDIS_URL not set or Redis unreachable"}
    return _CACHE.stats()

"""
    txt = txt[:j] + block + txt[j:]

# cache in /search endpoint (only if not already)
if "prefix = \"search\"" not in txt:
    needle = "return _search_core(\n"
    if needle not in txt:
        raise SystemExit("Can't find _search_core call in /search endpoint to patch caching")

    # Insert caching right before calling _search_core by patching within search() endpoint
    # We find the line "return _search_core(" and inject code above it.
    txt = txt.replace(
        "    return _search_core(\n",
        "    # ---- Redis Tier-1 cache (search) ----\n"
        "    cache_hit = False\n"
        "    if _CACHE is not None:\n"
        "        cache_payload = {\n"
        "            \"query\": sreq.query,\n"
        "            \"method\": sreq.method,\n"
        "            \"k\": sreq.k,\n"
        "            \"candidate_k\": sreq.candidate_k,\n"
        "            \"rerank_k\": sreq.rerank_k,\n"
        "            \"alpha\": sreq.alpha,\n"
        "            \"x_language\": x_language,\n"
        "        }\n"
        "        cached, _ms = _CACHE.get_json(prefix=\"search\", payload=cache_payload)\n"
        "        if cached is not None:\n"
        "            _CACHE.bump(\"search\", hit=True)\n"
        "            cached[\"cache_hit\"] = True\n"
        "            return SearchResponse(**cached)\n"
        "        _CACHE.bump(\"search\", hit=False)\n"
        "\n"
        "    res = _search_core(\n"
    )

    # and after _search_core returns, set + store cache
    # convert the first "return SearchResponse(" from end? Safer: patch the end of function by replacing "return _search_core(" only.
    # Now we need to patch the later "return res" - find the line that closes call and returns.
    # We look for the end of that _search_core call in /search and replace the final ")\n" + dedent return.
    # Simpler: append cache-set right before "return res" by injecting at the end of search().
    # We'll insert just before the final 'return res' if present.
    if "\n    return res\n" in txt:
        txt = txt.replace(
            "\n    return res\n",
            "\n    if _CACHE is not None:\n"
            "        try:\n"
            "            _CACHE.set_json(prefix=\"search\", payload=cache_payload, value=res.model_dump(), ttl_s=60)\n"
            "        except Exception:\n"
            "            pass\n"
            "    res.cache_hit = False\n"
            "    return res\n"
        )
    else:
        # If code returns directly, leave it.
        pass

# cache in /answer endpoint
if "prefix=\"answer\"" not in txt:
    # find start of answer() after req parsing; inject before retrieval
    marker = "    t0 = _now_ms()\n"
    if marker in txt and "def answer(" in txt:
        txt = txt.replace(
            marker,
            "    # ---- Redis Tier-2 cache (answer) ----\n"
            "    cache_payload = {\n"
            "        \"query\": req.query,\n"
            "        \"method\": req.method,\n"
            "        \"k\": req.k,\n"
            "        \"candidate_k\": req.candidate_k,\n"
            "        \"rerank_k\": req.rerank_k,\n"
            "        \"alpha\": req.alpha,\n"
            "        \"context_k\": req.context_k,\n"
            "        \"x_language\": x_language,\n"
            "    }\n"
            "    if _CACHE is not None:\n"
            "        cached, _ms = _CACHE.get_json(prefix=\"answer\", payload=cache_payload)\n"
            "        if cached is not None:\n"
            "            _CACHE.bump(\"answer\", hit=True)\n"
            "            cached[\"cache_hit\"] = True\n"
            "            return AnswerResponse(**cached)\n"
            "        _CACHE.bump(\"answer\", hit=False)\n"
            "\n"
            + marker
        )
    else:
        pass

    # set cache before return AnswerResponse in answer()
    # We patch the first "return AnswerResponse(" occurrence in answer() by caching the built response.
    # Safer: wrap just before final return in answer(). We'll find "return AnswerResponse(" and replace with building obj then caching.
    token = "    return AnswerResponse(\n"
    if token in txt:
        txt = txt.replace(
            token,
            "    resp = AnswerResponse(\n"
        )
        # then replace the corresponding closing "    )\n" for that resp with caching + return.
        # This is brittle; instead patch the end: find "raw=raw,\n    )" pattern.
        txt = txt.replace(
            "        raw=raw,\n    )\n",
            "        raw=raw,\n        cache_hit=False,\n    )\n"
            "    if _CACHE is not None:\n"
            "        try:\n"
            "            _CACHE.set_json(prefix=\"answer\", payload=cache_payload, value=resp.model_dump(), ttl_s=120)\n"
            "        except Exception:\n"
            "            pass\n"
            "    return resp\n"
        )

p.write_text(txt, encoding="utf-8")
print("✅ Patched src/app/main.py")
PY

echo "===> [6/7] Docker base-image layering + compose stack"
mkdir -p docker

cat > docker/Dockerfile.base <<'DOCKER'
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl build-essential \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Heavy hitters in base layer (the whole point)
RUN python -m pip install --upgrade pip && \
    pip install \
      torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install \
      fastapi uvicorn[standard] redis \
      metaflow streamlit \
      numpy scipy pandas scikit-learn lightgbm \
      sentence-transformers faiss-cpu

WORKDIR /app
DOCKER

cat > docker/Dockerfile.api <<'DOCKER'
FROM streaming-search-base:latest
WORKDIR /app
COPY . /app
ENV REDIS_URL="redis://redis:6379/0"
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKER

cat > docker/Dockerfile.dashboard <<'DOCKER'
FROM streaming-search-base:latest
WORKDIR /app
COPY . /app
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/metaflow_dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
DOCKER

cat > docker-compose.yml <<'YML'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    environment:
      REDIS_URL: "redis://redis:6379/0"
      # If you run Ollama on your host:
      OLLAMA_URL: "http://host.docker.internal:11434"
      OLLAMA_MODEL: "llama3:latest"
    ports:
      - "8000:8000"
    depends_on:
      - redis

  dashboard:
    build:
      context: .
      dockerfile: docker/Dockerfile.dashboard
    ports:
      - "8501:8501"
    depends_on:
      - api
YML

echo "===> [7/7] Add run-inspection commands that actually work (Metaflow Client API)"
cat > scripts/metaflow_inspect.py <<'PY'
from __future__ import annotations

import sys
from metaflow import Flow

FLOW = sys.argv[1] if len(sys.argv) > 1 else "PhenomenalLTRFlow"
f = Flow(FLOW)

print(f"Flow: {FLOW}")
runs = list(f.runs())
print("Runs (newest first):")
for r in runs[:10]:
    print(" ", r.id, "finished=", r.finished)

if not runs:
    sys.exit(0)

latest = runs[0]
print("\nLatest run:", latest.id)
print("Steps:", [s.id for s in latest.steps()])

# Show train step tasks output if exists
if "train" in [s.id for s in latest.steps()]:
    step = latest["train"]
    tasks = list(step.tasks())
    print("\nTrain tasks:", [t.id for t in tasks])
    for t in tasks[:3]:
        try:
            print(f"\n--- train/{t.id} stdout ---")
            print(t.stdout)
        except Exception:
            pass
PY

echo "✅ Phenomenal Docker upgrade files written."
echo "Next:"
echo "  1) Build base image: docker build -t streaming-search-base:latest -f docker/Dockerfile.base ."
echo "  2) Start stack:      docker compose up --build"
echo "  3) Cache stats:      curl -s http://127.0.0.1:8000/cache/stats | python -m json.tool"
echo "  4) Inspect Metaflow: python scripts/metaflow_inspect.py PhenomenalLTRFlow"
