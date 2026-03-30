# src/genai/local_llm.py
"""
Local LLM + VLM Layer
======================
LLM  → Ollama (llama3, mistral, phi3) running locally — no API cost
VLM  → CLIP ViT-B/32 for visual embeddings (already loaded)
       + GPT-4o vision for image captioning (when API key available)
       + LLaVA via Ollama for fully local image understanding

Architecture:
  Query → Local LLM (intent extraction, query rewrite)
  Poster → VLM (image captioning, mood description)
  Title + Caption → LLM (grounded explanation)

Honest claim:
  "Local LLM via Ollama for inference. CLIP ViT-B/32 as the VLM
   embedding backbone. GPT-4o-mini as cloud LLM fallback.
   No training. No fine-tuning."
"""
from __future__ import annotations
import json
import time
from typing import Any


class LocalLLM:
    """
    Local LLM via Ollama.
    Falls back to GPT-4o-mini if Ollama is not running.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3:latest",
        openai_client=None,
    ) -> None:
        self.ollama_url = ollama_url
        self.model = model
        self.openai = openai_client
        self._ollama_available: bool | None = None

    def _check_ollama(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available
        try:
            import urllib.request
            urllib.request.urlopen(f"{self.ollama_url}/api/tags", timeout=2)
            self._ollama_available = True
        except Exception:
            self._ollama_available = False
        return self._ollama_available

    def complete(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Generate completion using local Ollama or cloud fallback.
        Returns text + metadata about which model was used.
        """
        if self._check_ollama():
            return self._ollama_complete(prompt, system, max_tokens, temperature)
        elif self.openai:
            return self._openai_complete(prompt, system, max_tokens, temperature)
        else:
            return {
                "text": "LLM not available — install Ollama or provide OpenAI key",
                "model": "none",
                "source": "fallback",
                "latency_ms": 0,
            }

    def _ollama_complete(
        self, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        import urllib.request
        t0 = time.time()
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode()
        try:
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            return {
                "text": result.get("response", "").strip(),
                "model": self.model,
                "source": "ollama_local",
                "latency_ms": round((time.time() - t0) * 1000),
            }
        except Exception as e:
            self._ollama_available = False
            return {"text": "", "model": self.model, "source": "ollama_error", "error": str(e)}

    def _openai_complete(
        self, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> dict[str, Any]:
        t0 = time.time()
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return {
                "text": resp.choices[0].message.content.strip(),
                "model": "gpt-4o-mini",
                "source": "openai_cloud",
                "latency_ms": round((time.time() - t0) * 1000),
            }
        except Exception as e:
            return {"text": "", "model": "gpt-4o-mini", "source": "openai_error", "error": str(e)}

    def status(self) -> dict[str, Any]:
        ollama_ok = self._check_ollama()
        models = []
        if ollama_ok:
            try:
                import urllib.request
                with urllib.request.urlopen(f"{self.ollama_url}/api/tags", timeout=2) as r:
                    d = json.loads(r.read())
                    models = [m["name"] for m in d.get("models", [])]
            except Exception:
                pass
        return {
            "ollama_running": ollama_ok,
            "ollama_url": self.ollama_url,
            "ollama_models_loaded": models,
            "cloud_fallback": "gpt-4o-mini" if self.openai else "none",
            "active_model": self.model if ollama_ok else ("gpt-4o-mini" if self.openai else "none"),
            "source": "local" if ollama_ok else ("cloud" if self.openai else "none"),
        }


class VLMImageDescriber:
    """
    Vision-Language Model for poster image description.

    Tier 1: LLaVA via Ollama (fully local, free)
    Tier 2: GPT-4o vision (cloud, costs API)
    Tier 3: CLIP zero-shot tags (already working, no description)

    Generates natural language description of a poster image,
    which then feeds into LLM-based ranking explanations.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        openai_client=None,
        clip_analyzer=None,
    ) -> None:
        self.ollama_url = ollama_url
        self.openai = openai_client
        self.clip = clip_analyzer
        self._llava_available: bool | None = None

    def _check_llava(self) -> bool:
        if self._llava_available is not None:
            return self._llava_available
        try:
            import urllib.request
            with urllib.request.urlopen(f"{self.ollama_url}/api/tags", timeout=2) as r:
                d = json.loads(r.read())
                models = [m["name"] for m in d.get("models", [])]
                self._llava_available = any("llava" in m.lower() for m in models)
        except Exception:
            self._llava_available = False
        return self._llava_available

    def describe_poster(
        self,
        image_url: str | None,
        title: str = "",
        genres: list[str] | None = None,
        mood_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate natural language description of a poster.
        Falls through tiers: LLaVA → GPT-4o vision → CLIP tags → genre inference
        """
        # Tier 1: LLaVA (fully local)
        if image_url and self._check_llava():
            result = self._llava_describe(image_url, title)
            if result.get("text"):
                return result

        # Tier 2: GPT-4o vision (cloud)
        if image_url and self.openai:
            result = self._gpt4v_describe(image_url, title)
            if result.get("text"):
                return result

        # Tier 3: CLIP mood tags → synthesized description
        if self.clip and mood_tags:
            desc = self._synthesize_from_tags(title, genres or [], mood_tags)
            return {
                "text": desc,
                "model": "clip_tag_synthesis",
                "source": "clip_zero_shot",
                "has_image": image_url is not None,
            }

        # Tier 4: Pure genre inference
        desc = self._genre_description(title, genres or [])
        return {
            "text": desc,
            "model": "genre_inference",
            "source": "text_only",
            "has_image": False,
        }

    def _llava_describe(self, image_url: str, title: str) -> dict[str, Any]:
        """Use LLaVA via Ollama for image description."""
        import urllib.request, base64
        try:
            with urllib.request.urlopen(image_url, timeout=10) as r:
                img_data = base64.b64encode(r.read()).decode()
            payload = json.dumps({
                "model": "llava",
                "prompt": f"Describe the mood, visual style, and atmosphere of this movie poster for '{title}'. Be concise (2-3 sentences).",
                "images": [img_data],
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                f"{self.ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            return {
                "text": result.get("response", "").strip(),
                "model": "llava",
                "source": "ollama_local_vlm",
                "has_image": True,
            }
        except Exception as e:
            return {"text": "", "error": str(e)}

    def _gpt4v_describe(self, image_url: str, title: str) -> dict[str, Any]:
        """Use GPT-4o vision for image description."""
        try:
            resp = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": f"Describe the mood, visual style, and atmosphere of this movie poster for '{title}'. Be concise (2-3 sentences)."},
                    ],
                }],
                max_tokens=150,
            )
            return {
                "text": resp.choices[0].message.content.strip(),
                "model": "gpt-4o-mini-vision",
                "source": "openai_cloud_vlm",
                "has_image": True,
            }
        except Exception as e:
            return {"text": "", "error": str(e)}

    def _synthesize_from_tags(self, title: str, genres: list[str], mood_tags: list[str]) -> str:
        mood_desc = {
            "dark_and_gritty": "dark, gritty atmosphere",
            "scary_horror": "terrifying and suspenseful visual style",
            "romantic": "warm, romantic aesthetic",
            "comedic": "light-hearted and playful tone",
            "action_intense": "high-energy, intense action aesthetic",
            "heartwarming": "warm, emotionally resonant imagery",
            "epic_grand": "grand, epic scale and scope",
            "melancholic": "melancholic, introspective mood",
            "mysterious": "mysterious, atmospheric quality",
            "light_and_fun": "bright, colorful, fun visual style",
        }
        desc_parts = [mood_desc[t] for t in mood_tags if t in mood_desc]
        if desc_parts:
            return f"The poster conveys a {' and '.join(desc_parts[:2])}. Visual signals suggest {', '.join(genres[:2]) if genres else 'genre content'}."
        return f"A {', '.join(genres[:2]) if genres else 'film'} with visual elements consistent with the title."

    def _genre_description(self, title: str, genres: list[str]) -> str:
        if not genres:
            return f"A film titled '{title}'."
        return f"A {'/'.join(genres[:2])} film. Visual style inferred from genre metadata."

    def status(self) -> dict[str, Any]:
        return {
            "llava_available": self._check_llava(),
            "gpt4v_available": self.openai is not None,
            "clip_available": self.clip is not None,
            "active_vlm": (
                "llava_local" if self._check_llava() else
                "gpt4o_mini_vision" if self.openai else
                "clip_zero_shot"
            ),
            "tier_1_local_vlm": "LLaVA via Ollama (install: ollama pull llava)",
            "tier_2_cloud_vlm": "GPT-4o-mini vision (needs OPENAI_API_KEY)",
            "tier_3_clip": "CLIP ViT-B/32 zero-shot (always available)",
        }
