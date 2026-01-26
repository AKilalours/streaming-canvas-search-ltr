# src/app/schemas.py
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# -----------------------
# Search API
# -----------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    method: str = Field("bm25", description="bm25 | dense | hybrid | hybrid_ltr")
    k: int = Field(10, ge=1, le=50)
    candidate_k: int = Field(200, ge=10, le=5000)
    rerank_k: int = Field(50, ge=1, le=2000)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    debug: bool = False


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


# -----------------------
# RAG / Answer API
# -----------------------
class Source(BaseModel):
    doc_id: str
    title: str | None = None
    snippet: str | None = None
    score: float | None = None
    score_breakdown: dict[str, float] | None = None


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    method: str = Field("hybrid_ltr", description="bm25 | dense | hybrid | hybrid_ltr")
    k: int = Field(10, ge=1, le=50)
    candidate_k: int = Field(200, ge=10, le=5000)
    rerank_k: int = Field(50, ge=1, le=2000)
    alpha: float = Field(0.5, ge=0.0, le=1.0)

    context_k: int = Field(6, ge=1, le=20)

    # LLM settings
    temperature: float = Field(0.2, ge=0.0, le=1.5)
    max_tokens: int = Field(400, ge=64, le=4096)

    debug: bool = False


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[Source]
    timings_ms: dict[str, float] | None = None
    warning: str | None = None
    raw: dict[str, Any] | None = None
