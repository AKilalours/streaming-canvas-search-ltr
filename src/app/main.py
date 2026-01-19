from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.deps import STORE, load_artifacts
from app.schemas import SearchHit, SearchRequest, SearchResponse
from utils.timing import timed


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup, once at shutdown
    load_artifacts("configs/serve.yaml")
    yield
    # Optional: free references on shutdown
    STORE.bm25 = None


app = FastAPI(title="streaming-canvas-search-ltr", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "bm25_loaded": STORE.bm25 is not None}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    timings = {}
    if STORE.bm25 is None:
        return SearchResponse(
            query=req.query,
            k=req.k,
            hits=[],
            timings_ms=timings if req.debug else None,
        )

    with timed("bm25_query", timings):
        results = STORE.bm25.query(req.query, k=req.k)

    hits = [SearchHit(doc_id=doc_id, score=score) for doc_id, score in results]
    return SearchResponse(
        query=req.query, k=req.k, hits=hits, timings_ms=timings if req.debug else None
    )
