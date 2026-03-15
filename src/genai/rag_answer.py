# src/genai/rag_answer.py
from __future__ import annotations

from typing import Any


def output_schema() -> dict[str, Any]:
    """
    JSON schema contract the LLM must obey.
    We keep it strict to reduce garbage outputs.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answer": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "description": "List of source indices like [1,2] matching the context blocks.",
            },
            "warning": {"type": ["string", "null"]},
        },
        "required": ["answer", "citations"],
    }


def rag_prompt(query: str, context: str | None = None, *, ctx: str | None = None) -> str:
    """
    Compatible prompt builder.
    Accepts `context=` (new) or `ctx=` (legacy) or positional second arg.
    """
    use_ctx = context if context is not None else (ctx or "")

    schema = output_schema()

    return f"""You are a retrieval-grounded assistant.

RULES (non-negotiable):
- Use ONLY the provided CONTEXT to answer.
- If the answer is not in CONTEXT, say you don't know and set warning.
- Do NOT hallucinate.
- Output MUST be valid JSON matching this schema: {schema}

CITATIONS:
- Cite sources by index: [1], [2], etc.
- Return citations as a list of integers in `citations`.

QUERY:
{query}

CONTEXT:
{use_ctx}

Return JSON only (no markdown, no extra text).
"""


def build_sources(hits: list[Any]) -> list[dict[str, Any]]:
    """
    Convert top SearchHits into a stable Source payload.
    Works with pydantic objects or dict-like rows.
    """
    out: list[dict[str, Any]] = []
    for h in hits:
        doc_id = getattr(h, "doc_id", None) or (h.get("doc_id") if isinstance(h, dict) else None)
        title = getattr(h, "title", None) or (h.get("title") if isinstance(h, dict) else None)
        snippet = getattr(h, "text", None) or (h.get("text") if isinstance(h, dict) else None)
        score = getattr(h, "score", None) or (h.get("score") if isinstance(h, dict) else None)
        score_breakdown = getattr(h, "score_breakdown", None) or (
            h.get("score_breakdown") if isinstance(h, dict) else None
        )

        out.append(
            {
                "doc_id": str(doc_id) if doc_id is not None else "",
                "title": title,
                "snippet": snippet,
                "score": float(score) if score is not None else 0.0,
                "score_breakdown": score_breakdown,
            }
        )
    return out
