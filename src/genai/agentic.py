# src/genai/agentic.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from app.schemas import SearchHit
from genai.rag_answer import output_schema, rag_prompt


@dataclass
class AgentStep:
    step: int
    candidate_k: int
    context_k: int
    supported: bool
    reason: str | None
    citations: list[int]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def _ollama_json_call(
    ollama: Any,
    *,
    prompt: str,
    schema: dict[str, Any],
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """
    Minimal compatibility for agentic.py (avoid importing app.main).
    """
    # Prefer generate_json
    try:
        fn = ollama.generate_json
        try:
            return fn(prompt=prompt, schema=schema, temperature=temperature, top_p=top_p)
        except TypeError:
            return fn(prompt=prompt, schema=schema)
    except AttributeError:
        pass

    # Fallback chat_json
    try:
        fn2 = ollama.chat_json
        try:
            return fn2(prompt=prompt, schema=schema, temperature=temperature, top_p=top_p)
        except TypeError:
            return fn2(prompt=prompt, schema=schema)
    except AttributeError as e:
        raise RuntimeError("Ollama client missing generate_json/chat_json") from e


def _extract_citations(llm_out: dict[str, Any]) -> list[int]:
    raw = llm_out.get("citations", [])
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for x in raw:
        if isinstance(x, int):
            out.append(x)
    return out


def run_agentic_rag(
    *,
    ollama: Any,
    query: str,
    method: str,
    k: int,
    initial_candidate_k: int,
    initial_context_k: int,
    alpha: float,
    rerank_k: int,
    max_steps: int,
    search_fn: Callable[[dict[str, Any]], dict[str, Any]],
    build_context_fn: Callable[[list[SearchHit]], str],
    sources_fn: Callable[[list[SearchHit]], list[dict[str, Any]]],
    temperature: float,
    top_p: float,
) -> tuple[dict[str, Any], list[AgentStep]]:
    """
    Agentic loop (simple but robust):
    - Step N: retrieve -> build context -> ask LLM for strict JSON (answer+citations)
    - If citations are missing/invalid and answer is not an abstention -> expand retrieval and retry
    """
    trace: list[AgentStep] = []

    candidate_k = int(initial_candidate_k)
    context_k = int(initial_context_k)

    last_llm: dict[str, Any] | None = None
    last_hits: list[SearchHit] = []

    for step in range(1, max_steps + 1):
        # 1) Retrieval
        sr_payload = {
            "query": query,
            "method": method,
            "k": k,
            "candidate_k": candidate_k,
            "rerank_k": rerank_k,
            "alpha": alpha,
            "debug": True,
        }
        sres = search_fn(sr_payload)
        hits_raw = sres.get("hits", []) or []
        hits: list[SearchHit] = [SearchHit.model_validate(h) for h in hits_raw]

        last_hits = hits[:context_k]

        # 2) Context
        context = build_context_fn(last_hits)

        # 3) Prompt
        prompt = rag_prompt(query, context=context)

        # 4) LLM JSON
        llm_out = _ollama_json_call(
            ollama,
            prompt=prompt,
            schema=output_schema(),
            temperature=float(temperature),
            top_p=float(top_p),
        )
        last_llm = llm_out

        answer = str(llm_out.get("answer", "")).strip()
        warning = llm_out.get("warning")
        citations = _extract_citations(llm_out)

        # 5) Determine support
        # If abstention or explicit warning: treat as "supported" in the sense that it’s not hallucinating.
        abstains = answer.lower() in {"i don't know", "i do not know", "unknown"}
        if abstains:
            trace.append(
                AgentStep(
                    step=step,
                    candidate_k=candidate_k,
                    context_k=context_k,
                    supported=True,
                    reason="abstained",
                    citations=[],
                )
            )
            break

        # Build sources now (aligned with context) and validate citation range
        sources = sources_fn(last_hits)
        num_sources = len(sources)

        invalid = [c for c in citations if c < 1 or c > num_sources]
        missing = len(citations) == 0

        if warning is not None:
            # Model says it’s not confident -> stop, do not hallucinate
            trace.append(
                AgentStep(
                    step=step,
                    candidate_k=candidate_k,
                    context_k=context_k,
                    supported=True,
                    reason=str(warning),
                    citations=[],
                )
            )
            break

        if missing or invalid:
            trace.append(
                AgentStep(
                    step=step,
                    candidate_k=candidate_k,
                    context_k=context_k,
                    supported=False,
                    reason=("missing citations" if missing else f"invalid citations: {invalid}"),
                    citations=citations,
                )
            )
            # Expand and retry
            candidate_k = min(candidate_k * 2, 2000)
            context_k = min(context_k + 4, 20)
            continue

        # Otherwise: accept
        trace.append(
            AgentStep(
                step=step,
                candidate_k=candidate_k,
                context_k=context_k,
                supported=True,
                reason=None,
                citations=citations,
            )
        )
        break

    # Final payload
    final_sources = sources_fn(last_hits)
    final_answer = "" if last_llm is None else str(last_llm.get("answer", "")).strip()

    # If we never got a model output, hard fail safely
    if last_llm is None:
        return (
            {
                "answer": "I don't know.",
                "sources": final_sources,
                "warning": "LLM call failed.",
                "raw": None,
            },
            trace,
        )

    # If we ended unsupported, return warning
    final_warning = None
    if trace and trace[-1].supported is False:
        final_warning = trace[-1].reason or "Unsupported answer; citations invalid."

    return (
        {
            "answer": final_answer,
            "sources": final_sources,
            "warning": final_warning,
            "raw": last_llm,
        },
        trace,
    )
