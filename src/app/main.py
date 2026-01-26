# src/app/main.py
from __future__ import annotations

import os
import time
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from app.deps import AppState, load_state
from app.schemas import (
    AnswerRequest,
    AnswerResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    Source,
)
from genai.agentic import run_agentic_rag
from genai.ollama_client import OllamaClient, OllamaConfig
from genai.rag_answer import build_sources, output_schema, rag_prompt
from ranking.ltr_infer import LTRReranker
from retrieval.hybrid import hybrid_merge
from utils.logging import get_logger

log = get_logger("app.main")

STATE: AppState | None = None
_OLLAMA: OllamaClient | None = None


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _snippet(txt: str | None, n: int = 300) -> str | None:
    if not txt:
        return None
    t = " ".join(str(txt).split())
    return t[:n] + ("…" if len(t) > n else "")


def _ensure_ready() -> AppState:
    if not STATE or not getattr(STATE, "ready", False):
        raise HTTPException(status_code=503, detail="Server not ready. Check startup logs.")
    return STATE


def _ensure_ollama() -> OllamaClient:
    global _OLLAMA
    if _OLLAMA is not None:
        return _OLLAMA

    base_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3:latest")

    cfg = OllamaConfig(base_url=base_url, model=model)
    _OLLAMA = OllamaClient(cfg)
    log.info("Ollama configured: url=%s model=%s", base_url, model)
    return _OLLAMA


def _ollama_json(
    ollama: OllamaClient,
    *,
    prompt: str,
    schema: dict[str, Any],
    temperature: float | None = None,
    top_p: float | None = None,
) -> dict[str, Any]:
    """
    Compatibility wrapper:
    - supports OllamaClient.generate_json OR OllamaClient.chat_json
    - supports clients that do NOT accept temperature/top_p
    """
    temp = 0.2 if temperature is None else float(temperature)
    tp = 0.9 if top_p is None else float(top_p)

    # Prefer generate_json if present
    try:
        fn = ollama.generate_json  # type: ignore[attr-defined]
        try:
            return fn(prompt=prompt, schema=schema, temperature=temp, top_p=tp)
        except TypeError:
            return fn(prompt=prompt, schema=schema)
    except AttributeError:
        pass

    # Fallback to chat_json if present
    try:
        fn2 = ollama.chat_json  # type: ignore[attr-defined]
        try:
            return fn2(prompt=prompt, schema=schema, temperature=temp, top_p=tp)
        except TypeError:
            return fn2(prompt=prompt, schema=schema)
    except AttributeError as e:
        raise RuntimeError("OllamaClient has no generate_json() or chat_json().") from e


def _maybe_dataset_card(query: str) -> tuple[str, Source] | None:
    """
    Prevents ‘SciFact used for?’ producing nonsense from random abstracts.
    We inject a dataset card when the query is clearly meta.
    """
    q = query.lower()
    meta_triggers = ("what is", "used for", "dataset", "beir", "benchmark", "scifact")
    if "scifact" in q and any(t in q for t in meta_triggers):
        text = (
            "SciFact is a scientific claim verification benchmark (often used within BEIR-style evaluation). "
            "Given a claim/query, the system retrieves scientific abstracts and determines whether evidence "
            "supports or refutes the claim, typically with sentence-level evidence annotations."
        )
        src = Source(
            doc_id="__dataset_card_scifact__",
            title="SciFact Dataset Card",
            snippet=text,
            score=None,
            score_breakdown=None,
        )
        return text, src
    return None


def _as_hit_dicts(hits: Iterable[Any]) -> list[dict[str, Any]]:
    """
    Normalizes hits to dicts.
    Accepts SearchHit models or dict-like objects.
    """
    out: list[dict[str, Any]] = []
    for h in hits:
        if hasattr(h, "model_dump"):
            out.append(h.model_dump())  # pydantic v2
        elif isinstance(h, dict):
            out.append(dict(h))
        else:
            out.append(
                {
                    "doc_id": getattr(h, "doc_id", ""),
                    "title": getattr(h, "title", None),
                    "text": getattr(h, "text", None),
                    "score": getattr(h, "score", 0.0),
                    "score_breakdown": getattr(h, "score_breakdown", None),
                }
            )
    return out


def _context_and_sources(query: str, hits: list[Any], context_k: int) -> tuple[str, list[Source]]:
    """
    Builds context + sources with OPTIONAL dataset card prepended.
    Ensures citation indices match returned sources ordering.
    """
    hit_dicts = _as_hit_dicts(hits)

    card = _maybe_dataset_card(query)
    if card is not None:
        _, card_src = card
        doc_hits = hit_dicts[: max(0, context_k - 1)]
        srcs = [card_src] + [Source(**d) for d in build_sources(doc_hits)]
    else:
        doc_hits = hit_dicts[:context_k]
        srcs = [Source(**d) for d in build_sources(doc_hits)]

    blocks: list[str] = []
    for i, s in enumerate(srcs, start=1):
        title = (s.title or "").strip()
        snippet = (s.snippet or "").strip()
        blocks.append(f"[{i}] doc_id={s.doc_id}\nTITLE: {title}\nTEXT: {snippet}\n")

    return "\n".join(blocks), srcs


def _validate_llm_output(
    llm_out: dict[str, Any], *, num_sources: int
) -> tuple[str, list[int], str | None]:
    """
    Enforces: factual answer => in-range citations.
    If model violates, we keep answer but return warning and empty citations.
    """
    answer = str(llm_out.get("answer", "")).strip()
    warning = llm_out.get("warning")

    citations_raw = llm_out.get("citations", [])
    citations: list[int] = []
    if isinstance(citations_raw, list):
        for x in citations_raw:
            if isinstance(x, int):
                citations.append(x)

    abstain = answer.lower() in {"i don't know", "i do not know", "unknown"}
    if abstain:
        return answer, [], warning or "Insufficient evidence in retrieved context."

    if not citations:
        return answer, [], warning or "Model returned no citations for a factual answer."

    bad = [c for c in citations if c < 1 or c > num_sources]
    if bad:
        return (
            answer,
            [],
            warning or f"Model returned out-of-range citations: {bad} (sources={num_sources}).",
        )

    return answer, citations, warning


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Replaces deprecated startup/shutdown events.
    Removes FastAPI DeprecationWarning.
    """
    global STATE, _OLLAMA
    try:
        STATE = load_state()

        if STATE and getattr(STATE, "ltr_path", None) is not None:
            try:
                STATE.reranker = LTRReranker.load(str(STATE.ltr_path))  # type: ignore[attr-defined]
                log.info("Loaded LTR reranker: %s", STATE.ltr_path)
            except Exception:
                log.exception("Failed to load LTR reranker at startup.")

        log.info("Startup complete. ready=%s", bool(STATE and STATE.ready))
        yield
    finally:
        _OLLAMA = None
        STATE = None


app = FastAPI(title="streaming-canvas-search-ltr", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "ready": bool(STATE and STATE.ready),
        "bm25_loaded": bool(STATE and getattr(STATE, "bm25_obj", None) is not None),
        "dense_loaded": bool(STATE and getattr(STATE, "dense", None) is not None),
        "ltr_loaded": bool(STATE and getattr(STATE, "ltr_path", None) is not None),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    st = _ensure_ready()

    method = req.method.strip().lower()
    if method not in {"bm25", "dense", "hybrid", "hybrid_ltr"}:
        raise HTTPException(
            status_code=400, detail="method must be one of: bm25, dense, hybrid, hybrid_ltr"
        )

    t0 = _now_ms()
    timings: dict[str, float] = {}

    bm25_hits: list[tuple[str, float]] = []
    dense_hits: list[tuple[str, float]] = []
    merged: list[tuple[str, float]] = []

    if method in {"bm25", "hybrid", "hybrid_ltr"}:
        a = _now_ms()
        bm25_hits = st.bm25_query(req.query, k=req.candidate_k)
        timings["bm25_ms"] = _now_ms() - a

    if method in {"dense", "hybrid", "hybrid_ltr"}:
        if st.dense is None:
            raise HTTPException(
                status_code=503, detail="Dense artifacts not loaded. Build embeddings first."
            )
        a = _now_ms()
        dense_hits = st.dense.search(req.query, k=req.candidate_k)
        timings["dense_ms"] = _now_ms() - a

    if method == "bm25":
        merged = bm25_hits
    elif method == "dense":
        merged = dense_hits
    else:
        a = _now_ms()
        merged = hybrid_merge(bm25_hits, dense_hits, alpha=req.alpha)
        timings["merge_ms"] = _now_ms() - a

    score_break_bm25 = {d: float(s) for d, s in bm25_hits}
    score_break_dense = {d: float(s) for d, s in dense_hits}
    score_break_hybrid = {d: float(s) for d, s in merged}

    final: list[tuple[str, float]] = merged[: req.k]

    if method == "hybrid_ltr" and getattr(st, "ltr_path", None) is not None:
        rerank_k = min(req.rerank_k, len(merged))
        to_rerank = merged[:rerank_k]

        reranker = getattr(st, "reranker", None)
        if reranker is None:
            a = _now_ms()
            reranker = LTRReranker.load(str(st.ltr_path))
            timings["ltr_load_ms"] = _now_ms() - a

        a = _now_ms()
        reranked = reranker.rerank(
            query=req.query,
            corpus=st.corpus,
            candidates=to_rerank,
            bm25_scores=score_break_bm25,
            dense_scores=score_break_dense,
        )
        timings["ltr_ms"] = _now_ms() - a

        reranked_ids = {d for d, _ in reranked}
        tail = [(d, s) for d, s in merged if d not in reranked_ids]
        final = (reranked + tail)[: req.k]

    timings["total_ms"] = _now_ms() - t0

    hits: list[SearchHit] = []
    for did, score in final:
        row = st.corpus.get(did, {})
        breakdown = None
        if req.debug:
            breakdown = {
                "bm25": float(score_break_bm25.get(did, 0.0)),
                "dense": float(score_break_dense.get(did, 0.0)),
                "hybrid": float(score_break_hybrid.get(did, 0.0)),
            }
        hits.append(
            SearchHit(
                doc_id=str(did),
                score=float(score),
                title=row.get("title"),
                text=_snippet(row.get("text")),
                score_breakdown=breakdown,
            )
        )

    return SearchResponse(
        query=req.query,
        method=method,
        k=req.k,
        hits=hits,
        timings_ms=timings if req.debug else None,
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    _ensure_ready()
    ollama = _ensure_ollama()

    t0 = _now_ms()
    timings: dict[str, float] = {}

    a = _now_ms()
    sres = search(
        SearchRequest(
            query=req.query,
            method=req.method,
            k=req.k,
            candidate_k=req.candidate_k,
            rerank_k=req.rerank_k,
            alpha=req.alpha,
            debug=True,
        )
    )
    timings["retrieval_ms"] = _now_ms() - a

    a = _now_ms()
    context, sources = _context_and_sources(req.query, sres.hits, req.context_k)
    timings["context_ms"] = _now_ms() - a

    a = _now_ms()
    prompt = rag_prompt(req.query, context=context)
    timings["prompt_ms"] = _now_ms() - a

    a = _now_ms()
    llm_out = _ollama_json(
        ollama,
        prompt=prompt,
        schema=output_schema(),
        temperature=req.temperature,
        top_p=0.9,
    )
    timings["llm_ms"] = _now_ms() - a
    timings["total_ms"] = _now_ms() - t0

    answer_text, citations, warn = _validate_llm_output(llm_out, num_sources=len(sources))

    raw = None
    if req.debug:
        raw = dict(llm_out)
        raw["citations_validated"] = citations

    return AnswerResponse(
        query=req.query,
        answer=answer_text,
        sources=sources,
        timings_ms=timings if req.debug else None,
        warning=warn,
        raw=raw,
    )


@app.post("/agent_answer", response_model=AnswerResponse)
def agent_answer(req: AnswerRequest) -> AnswerResponse:
    _ensure_ready()
    ollama = _ensure_ollama()

    def search_fn(payload: dict[str, Any]) -> dict[str, Any]:
        sreq = SearchRequest(**payload)
        sres = search(sreq)
        return sres.model_dump()

    def build_context_fn(hits: list[Any]) -> str:
        ctx, _ = _context_and_sources(req.query, hits, req.context_k)
        return ctx

    def sources_fn(hits: list[Any]) -> list[dict[str, Any]]:
        _, srcs = _context_and_sources(req.query, hits, req.context_k)
        return [s.model_dump() for s in srcs]

    payload, trace = run_agentic_rag(
        ollama=ollama,
        query=req.query,
        method=req.method,
        k=req.k,
        initial_candidate_k=req.candidate_k,
        initial_context_k=req.context_k,
        alpha=req.alpha,
        rerank_k=req.rerank_k,
        max_steps=2,
        search_fn=search_fn,
        build_context_fn=build_context_fn,
        sources_fn=sources_fn,
        temperature=0.0,
        top_p=0.9,
    )

    sources = [Source(**s) for s in payload["sources"]]
    warning = payload.get("warning")
    if trace and (trace[-1].supported is False) and (trace[-1].reason is not None):
        warning = warning or trace[-1].reason

    return AnswerResponse(
        query=req.query,
        answer=payload["answer"],
        sources=sources,
        timings_ms={
            "agent_steps": float(len(trace)),
            "final_candidate_k": float(trace[-1].candidate_k) if trace else float(req.candidate_k),
            "final_context_k": float(trace[-1].context_k) if trace else float(req.context_k),
        },
        warning=warning,
        raw={"agent_trace": [t.model_dump() for t in trace], "llm": payload.get("raw")}
        if req.debug
        else None,
    )
