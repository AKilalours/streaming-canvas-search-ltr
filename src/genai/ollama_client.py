# src/genai/ollama_client.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from utils.logging import get_logger

log = get_logger("genai.ollama_client")


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3:latest"  # MUST match `ollama list`
    timeout_s: float = 120.0


class OllamaClient:
    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg
        self._client = httpx.Client(timeout=cfg.timeout_s)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        format: str | dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        url = f"{self.cfg.base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if format is not None:
            payload["format"] = (
                format  # "json" OR a JSON schema dict :contentReference[oaicite:3]{index=3}
            )

        # options are optional; only send when set
        opts: dict[str, Any] = {}
        if temperature is not None:
            opts["temperature"] = float(temperature)
        if top_p is not None:
            opts["top_p"] = float(top_p)
        if opts:
            payload["options"] = opts

        r = self._client.post(url, json=payload)
        if r.status_code != 200:
            raise RuntimeError(
                f"Ollama HTTP {r.status_code}: {r.text[:500]} | "
                f"Check `ollama list` and set model to an installed tag."
            )

        data = r.json()
        return str(data.get("response", ""))

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> dict[str, Any]:
        # Strongest mode: schema-enforced JSON if schema provided.
        fmt: str | dict[str, Any] = schema if schema is not None else "json"

        raw = self.generate(
            prompt,
            format=fmt,
            temperature=temperature,
            top_p=top_p,
        ).strip()

        try:
            return json.loads(raw)
        except Exception as e:
            # If this happens with format="json", the model violated contract.
            raise RuntimeError(f"Model did not return valid JSON. Raw={raw[:500]}") from e
