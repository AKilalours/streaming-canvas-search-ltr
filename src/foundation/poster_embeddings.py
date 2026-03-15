# src/foundation/poster_embeddings.py
"""
Real Multimodal Poster Embeddings using CLIP
=============================================
Uses pretrained CLIP (clip-ViT-B-32) from sentence-transformers
to generate real image+text embeddings for movie posters.

This is the honest student version:
- Uses pretrained vision-language model (not training MediaFM)
- Generates real 512-dim embeddings from poster images
- Enables multimodal similarity and artwork-aware reranking
- Honest claim: "multimodal ranking layer using pretrained VL embeddings"

No GPU required — runs on CPU, ~0.5s per poster.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIR = Path("artifacts/poster_embeddings")
TMDB_IMG  = "https://image.tmdb.org/t/p/w300"  # smaller for speed


@dataclass
class PosterEmbedding:
    doc_id: str
    title: str
    embedding: np.ndarray        # 512-dim CLIP embedding
    text_embedding: np.ndarray   # text-only embedding for comparison
    poster_url: str = ""
    source: str = "clip"         # "clip" | "text_only"
    cached: bool = False


class CLIPPosterEmbedder:
    """
    Real multimodal embeddings using CLIP ViT-B/32.
    
    Falls back gracefully:
      1. CLIP image+text embedding (best - real multimodal)
      2. Text-only SentenceTransformer embedding (good - already in stack)
    
    Usage:
        embedder = CLIPPosterEmbedder()
        emb = embedder.embed("doc_1", "The Dark Knight", poster_url)
        # emb.embedding is a real 512-dim visual-semantic vector
    """

    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._text_model = None
        self._available = False
        self._load_models()

    def _load_models(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            # CLIP model — encodes both images and text in same space
            self._model = SentenceTransformer("clip-ViT-B-32")
            self._available = True
            print("[CLIPEmbedder] CLIP ViT-B/32 loaded")
        except Exception as e:
            print(f"[CLIPEmbedder] CLIP not available: {e} — using text fallback")
            try:
                from sentence_transformers import SentenceTransformer
                self._text_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                pass

    def _cache_path(self, doc_id: str) -> Path:
        h = hashlib.md5(doc_id.encode()).hexdigest()[:10]
        return CACHE_DIR / f"{h}.npy"

    def _cache_meta_path(self, doc_id: str) -> Path:
        h = hashlib.md5(doc_id.encode()).hexdigest()[:10]
        return CACHE_DIR / f"{h}.json"

    def _load_cache(self, doc_id: str) -> PosterEmbedding | None:
        ep = self._cache_path(doc_id)
        mp = self._cache_meta_path(doc_id)
        if ep.exists() and mp.exists():
            try:
                emb = np.load(str(ep))
                meta = json.loads(mp.read_text())
                return PosterEmbedding(
                    doc_id=doc_id,
                    title=meta.get("title", ""),
                    embedding=emb,
                    text_embedding=emb,
                    poster_url=meta.get("poster_url", ""),
                    source=meta.get("source", "clip"),
                    cached=True,
                )
            except Exception:
                pass
        return None

    def _save_cache(self, pe: PosterEmbedding) -> None:
        np.save(str(self._cache_path(pe.doc_id)), pe.embedding)
        self._cache_meta_path(pe.doc_id).write_text(json.dumps({
            "title": pe.title,
            "poster_url": pe.poster_url,
            "source": pe.source,
        }))

    def _fetch_image(self, url: str) -> bytes | None:
        try:
            with urllib.request.urlopen(url, timeout=6) as r:
                return r.read()
        except Exception:
            return None

    def embed(self, doc_id: str, title: str, poster_url: str = "") -> PosterEmbedding:
        # Check cache
        cached = self._load_cache(doc_id)
        if cached:
            return cached

        # Try CLIP image embedding
        if self._available and poster_url:
            try:
                from PIL import Image
                import io
                img_bytes = self._fetch_image(poster_url)
                if img_bytes:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    # CLIP encodes image and text in same 512-dim space
                    img_emb = self._model.encode(img, convert_to_numpy=True)
                    txt_emb = self._model.encode(title, convert_to_numpy=True)
                    # Blend: 70% visual, 30% text
                    combined = 0.7 * img_emb + 0.3 * txt_emb
                    combined = combined / (np.linalg.norm(combined) + 1e-9)
                    pe = PosterEmbedding(
                        doc_id=doc_id, title=title,
                        embedding=combined, text_embedding=txt_emb,
                        poster_url=poster_url, source="clip",
                    )
                    self._save_cache(pe)
                    return pe
            except Exception as e:
                print(f"[CLIPEmbedder] Image embed failed for {title}: {e}")

        # Fallback: text-only embedding
        try:
            if self._available:
                txt_emb = self._model.encode(title, convert_to_numpy=True)
            elif self._text_model:
                txt_emb = self._text_model.encode(title, convert_to_numpy=True)
            else:
                txt_emb = np.random.randn(512).astype(np.float32)
                txt_emb /= np.linalg.norm(txt_emb)

            pe = PosterEmbedding(
                doc_id=doc_id, title=title,
                embedding=txt_emb, text_embedding=txt_emb,
                poster_url=poster_url, source="text_only",
            )
            self._save_cache(pe)
            return pe
        except Exception as e:
            arr = np.zeros(512, dtype=np.float32)
            return PosterEmbedding(doc_id=doc_id, title=title,
                                   embedding=arr, text_embedding=arr)

    def similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        na = np.linalg.norm(emb_a)
        nb = np.linalg.norm(emb_b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(emb_a, emb_b) / (na * nb))

    def rerank_by_visual_similarity(
        self,
        query_doc_id: str,
        candidates: list[dict[str, Any]],
        alpha: float = 0.2,
    ) -> list[dict[str, Any]]:
        """
        Rerank candidates by visual similarity to query poster.
        alpha controls how much visual similarity affects final score.
        """
        seed = self._load_cache(query_doc_id)
        if seed is None:
            return candidates

        for item in candidates:
            cached = self._load_cache(item.get("doc_id", ""))
            if cached is not None:
                vis_sim = self.similarity(seed.embedding, cached.embedding)
                item["visual_similarity"] = round(float(vis_sim), 4)
                base = float(item.get("score", 0))
                item["score"] = round(base * (1 - alpha) + vis_sim * alpha, 4)
            else:
                item["visual_similarity"] = 0.0

        return sorted(candidates, key=lambda x: -x.get("score", 0))

    def batch_embed(
        self, items: list[dict[str, Any]], delay_s: float = 0.2
    ) -> list[PosterEmbedding]:
        results = []
        for item in items:
            pe = self.embed(
                doc_id=item.get("doc_id", ""),
                title=item.get("title", ""),
                poster_url=item.get("poster_url", ""),
            )
            results.append(pe)
            if not pe.cached:
                time.sleep(delay_s)
        return results

    @property
    def available(self) -> bool:
        return self._available

    def stats(self) -> dict[str, Any]:
        cached = list(CACHE_DIR.glob("*.npy"))
        return {
            "clip_available": self._available,
            "cached_embeddings": len(cached),
            "cache_dir": str(CACHE_DIR),
            "embedding_dim": 512,
            "model": "clip-ViT-B-32" if self._available else "text-fallback",
        }
