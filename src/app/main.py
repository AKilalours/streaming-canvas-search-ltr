from fastapi import FastAPI

from app.deps import STORE, load_artifacts
from app.schemas import SearchRequest, SearchResponse, SearchHit
from utils.timing import timed

app = FastAPI(title="streaming-canvas-search-ltr", version="0.1.0")


@app.on_event("startup")
def _startup():
    # default path; override by setting SERVE_CONFIG env if you want later
    load_artifacts("configs/serve.yaml")


@app.get("/health")
def health():
    return {"ok": True, "bm25_loaded": STORE.bm25 is not None}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    timings = {}
    hits = []

    if STORE.bm25 is None:
        return SearchResponse(query=req.query, k=req.k, hits=[], timings_ms=timings if req.debug else None)

    with timed("bm25_query", timings):
        results = STORE.bm25.query(req.query, k=req.k)

    hits = [SearchHit(doc_id=doc_id, score=score) for doc_id, score in results]
    return SearchResponse(query=req.query, k=req.k, hits=hits, timings_ms=timings if req.debug else None)

