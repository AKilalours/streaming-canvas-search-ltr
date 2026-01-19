from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int = Field(10, ge=1, le=50)
    debug: bool = False


class SearchHit(BaseModel):
    doc_id: str
    score: float


class SearchResponse(BaseModel):
    query: str
    k: int
    hits: list[SearchHit]
    timings_ms: dict[str, float] | None = None
