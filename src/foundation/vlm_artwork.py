# src/foundation/vlm_artwork.py
"""
Real VLM Artwork Analysis — wired to OpenAI GPT-4V API
========================================================
This is a REAL implementation, not a simulation.

Fetches movie poster images from TMDB, sends them to GPT-4V,
and extracts structured visual features for ranking.

Requirements:
  - OPENAI_API_KEY env var
  - TMDB_API_KEY env var (free at themoviedb.org)

Without keys: falls back to the rule-based proxy in foundation/multimodal.py
With keys: real visual analysis of actual movie artwork.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TMDB_BASE      = "https://api.themoviedb.org/3"
TMDB_IMG_BASE  = "https://image.tmdb.org/t/p/w500"
OPENAI_BASE    = "https://api.openai.com/v1"
CACHE_DIR      = Path("artifacts/vlm_cache")

VLM_PROMPT = """Analyse this movie poster/thumbnail and respond ONLY with valid JSON, no markdown.

Required fields:
{
  "mood": one of [dark_moody, bright_action, warm_romantic, cold_thriller, playful_comedy, epic_drama, documentary, unknown],
  "brightness": float 0.0-1.0 (0=very dark, 1=very bright),
  "contrast": float 0.0-1.0,
  "dominant_colors": list of 3 color names (e.g. ["deep red", "black", "gold"]),
  "face_count": integer 0-5+,
  "text_overlay": boolean (is the title/text visible),
  "mood_confidence": float 0.0-1.0,
  "visual_tags": list of up to 5 descriptive tags (e.g. ["silhouette", "rain", "urban"]),
  "predicted_audience": one of [family, adult, teen, all_ages],
  "atmosphere": one of [tense, warm, cold, energetic, melancholic, uplifting, mysterious]
}"""


@dataclass
class RealArtworkFeatures:
    doc_id: str
    title: str
    mood: str
    brightness: float
    contrast: float
    dominant_colors: list[str]
    face_count: int
    text_overlay: bool
    mood_confidence: float
    visual_tags: list[str]
    predicted_audience: str
    atmosphere: str
    poster_url: str = ""
    analysis_source: str = "gpt4v"
    cached: bool = False
    error: str = ""


class TMDBClient:
    """Fetches movie posters from The Movie Database (free API)."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search_movie(self, title: str, year: str = "") -> dict[str, Any] | None:
        query = urllib.request.quote(title)
        url = f"{TMDB_BASE}/search/movie?api_key={self.api_key}&query={query}"
        if year:
            url += f"&year={year}"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
            results = data.get("results", [])
            return results[0] if results else None
        except Exception:
            return None

    def get_poster_url(self, title: str, year: str = "") -> str | None:
        movie = self.search_movie(title, year)
        if not movie:
            return None
        path = movie.get("poster_path")
        return f"{TMDB_IMG_BASE}{path}" if path else None

    def fetch_poster_b64(self, poster_url: str) -> str | None:
        """Download poster and return base64-encoded JPEG."""
        try:
            with urllib.request.urlopen(poster_url, timeout=8) as r:
                data = r.read()
            return base64.b64encode(data).decode("utf-8")
        except Exception:
            return None


class GPT4VClient:
    """Calls OpenAI GPT-4V for real visual artwork analysis."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self.api_key = api_key
        self.model = model

    def analyse_image_b64(self, image_b64: str, title: str = "") -> dict[str, Any]:
        """Send base64 image to GPT-4V and parse structured response."""
        payload = json.dumps({
            "model": self.model,
            "max_tokens": 500,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Movie: {title}\n\n{VLM_PROMPT}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "low"
                    }}
                ]
            }]
        }).encode()

        req = urllib.request.Request(
            f"{OPENAI_BASE}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                resp = json.loads(r.read())
            text = resp["choices"][0]["message"]["content"].strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception as e:
            return {"error": str(e)}


class RealArtworkAnalyser:
    """
    Real VLM artwork analyser.
    - Fetches poster from TMDB
    - Sends to GPT-4V
    - Caches results in artifacts/vlm_cache/ to avoid redundant API calls
    - Falls back to rule-based proxy if API keys missing
    """

    def __init__(self) -> None:
        self.openai_key = os.environ.get("OPENAI_API_KEY", "")
        self.tmdb_key   = os.environ.get("TMDB_API_KEY", "")
        self.cache_dir  = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tmdb   = TMDBClient(self.tmdb_key) if self.tmdb_key else None
        self.gpt4v  = GPT4VClient(self.openai_key) if self.openai_key else None
        self.available = bool(self.openai_key and self.tmdb_key)

    def _cache_key(self, doc_id: str) -> Path:
        h = hashlib.md5(doc_id.encode()).hexdigest()[:12]
        return self.cache_dir / f"{h}.json"

    def _load_cache(self, doc_id: str) -> dict | None:
        p = self._cache_key(doc_id)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return None
        return None

    def _save_cache(self, doc_id: str, data: dict) -> None:
        self._cache_key(doc_id).write_text(json.dumps(data, indent=2))

    def analyse(self, doc_id: str, title: str = "", year: str = "") -> RealArtworkFeatures:
        # Check cache first
        cached = self._load_cache(doc_id)
        if cached and "mood" in cached:
            return RealArtworkFeatures(
                doc_id=doc_id, title=title, cached=True,
                analysis_source="gpt4v_cached",
                **{k: cached[k] for k in [
                    "mood","brightness","contrast","dominant_colors",
                    "face_count","text_overlay","mood_confidence",
                    "visual_tags","predicted_audience","atmosphere",
                ] if k in cached},
                poster_url=cached.get("poster_url",""),
            )

        # Always try TMDB poster first (free, no credits needed)
        poster_url = None
        if self.tmdb:
            poster_url = self.tmdb.get_poster_url(title, year)

        if not self.available:
            fb = self._fallback(doc_id, title)
            fb.poster_url = poster_url or ""
            return fb

        if not poster_url:
            return self._fallback(doc_id, title, error="TMDB: poster not found")

        # Download poster
        image_b64 = self.tmdb.fetch_poster_b64(poster_url)
        if not image_b64:
            fb = self._fallback(doc_id, title, error="Could not download poster")
            fb.poster_url = poster_url
            return fb

        # Call GPT-4V — if rate limited, return fallback WITH poster_url
        result = self.gpt4v.analyse_image_b64(image_b64, title)
        if "error" in result:
            fb = self._fallback(doc_id, title, error=result["error"])
            fb.poster_url = poster_url
            return fb

        # Cache and return
        result["poster_url"] = poster_url
        self._save_cache(doc_id, result)

        return RealArtworkFeatures(
            doc_id=doc_id, title=title,
            mood=result.get("mood","unknown"),
            brightness=float(result.get("brightness", 0.5)),
            contrast=float(result.get("contrast", 0.5)),
            dominant_colors=result.get("dominant_colors", []),
            face_count=int(result.get("face_count", 0)),
            text_overlay=bool(result.get("text_overlay", False)),
            mood_confidence=float(result.get("mood_confidence", 0.7)),
            visual_tags=result.get("visual_tags", []),
            predicted_audience=result.get("predicted_audience", "all_ages"),
            atmosphere=result.get("atmosphere", "mysterious"),
            poster_url=poster_url,
            analysis_source="gpt4v",
        )

    def _fallback(self, doc_id: str, title: str, error: str = "") -> RealArtworkFeatures:
        """Rule-based fallback when API keys not set."""
        from foundation.multimodal import ArtworkAnalyser
        proxy = ArtworkAnalyser().analyse(doc_id=doc_id, title=title)
        return RealArtworkFeatures(
            doc_id=doc_id, title=title,
            mood=proxy.mood.value,
            brightness=proxy.brightness,
            contrast=proxy.contrast,
            dominant_colors=[],
            face_count=proxy.face_count,
            text_overlay=proxy.text_overlay,
            mood_confidence=proxy.mood_confidence,
            visual_tags=[],
            predicted_audience="all_ages",
            atmosphere="mysterious",
            analysis_source="rule_based_fallback",
            error=error,
        )

    def batch_analyse(
        self,
        items: list[dict],
        delay_s: float = 0.5,
    ) -> list[RealArtworkFeatures]:
        """
        Batch analyse with rate limiting.
        GPT-4V tier-1: ~500 RPM — delay_s=0.5 keeps us safe.
        """
        results = []
        for item in items:
            doc_id = item.get("doc_id","")
            title  = item.get("title","")
            year   = str(item.get("year",""))
            result = self.analyse(doc_id, title, year)
            results.append(result)
            if not result.cached and self.available:
                time.sleep(delay_s)
        return results
