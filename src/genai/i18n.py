from __future__ import annotations

from typing import Any

_EN_ALIASES = {"en", "en-us", "en-gb", "english"}


def should_translate(lang: str) -> bool:
    x = (lang or "en").strip().lower()
    return x not in _EN_ALIASES


def translate_with_ollama(ollama: Any, text: str, target_lang: str, *, max_tokens: int = 512) -> str:
    """
    Best-effort translation that works with varying OllamaClient method names.
    """
    if not text or not text.strip():
        return text

    system = (
        "You are a professional translator. Translate faithfully, preserve names, titles, and bullet formatting."
    )
    prompt = f"Target language: {target_lang}\n\nTEXT:\n{text}\n\nTRANSLATION:"

    # Prefer text generation methods if present.
    for meth in ("generate", "chat"):
        fn = getattr(ollama, meth, None)
        if fn is None:
            continue
        try:
            out = fn(prompt=prompt, system=system, max_tokens=max_tokens)  # type: ignore[misc]
        except TypeError:
            out = fn(prompt=prompt)  # type: ignore[misc]
        return out.strip() if isinstance(out, str) else str(out).strip()

    # Fallback: use JSON helpers if that's all we have.
    schema = {"type": "object", "properties": {"translation": {"type": "string"}}, "required": ["translation"]}
    for meth in ("generate_json", "chat_json"):
        fn = getattr(ollama, meth, None)
        if fn is None:
            continue
        try:
            j = fn(prompt=prompt, schema=schema)  # type: ignore[misc]
        except TypeError:
            j = fn(prompt=prompt, schema=schema)  # type: ignore[misc]
        tr = (j or {}).get("translation", "")
        return tr.strip() if isinstance(tr, str) else str(tr).strip()

    return text
