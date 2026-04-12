"""
StreamLens — Edge Voice Pipeline (Faster-Whisper + local LLM)
Runs entirely on CPU / Apple Silicon with zero API cost.
Replaces the cloud OpenAI Whisper endpoint with local Faster-Whisper.

Install:
    pip install faster-whisper
    ollama pull llama3  (already done)

Usage:
    python faster_whisper_edge.py --audio query.webm
    
Architecture (edge pipeline):
    Audio → Faster-Whisper ASR → Query → BM25+FAISS retrieval → 
    Cross-Encoder rerank → Llama3 (Ollama) generation → Response
"""
from __future__ import annotations
import argparse, time, json, urllib.request
from pathlib import Path

def transcribe_local(audio_path: str, model_size: str = "base") -> dict:
    """
    Local ASR using Faster-Whisper.
    model_size: tiny (fastest), base, small, medium, large
    tiny  → ~39MB,  ~32x realtime on CPU
    base  → ~74MB,  ~16x realtime on CPU  ← good default
    small → ~244MB,  ~6x realtime on CPU
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {
            "error": "faster-whisper not installed. Run: pip install faster-whisper",
            "fallback": "Using OpenAI Whisper API instead"
        }

    t0 = time.perf_counter()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)
    text = " ".join(seg.text for seg in segments).strip()
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "transcription": text,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "latency_ms": latency_ms,
        "model": f"faster-whisper-{model_size}",
        "device": "cpu",
        "cost": "$0.00  (local inference)"
    }


def search_streamlens(query: str, k: int = 5) -> dict:
    """Call StreamLens search API."""
    try:
        url = f"http://localhost:8000/search?q={urllib.parse.quote(query)}&method=hybrid_ltr&k={k}"
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e), "hits": []}


def generate_local(query: str, context: str) -> dict:
    """Generate answer using local Llama3 via Ollama."""
    t0 = time.perf_counter()
    try:
        payload = json.dumps({
            "model": "llama3:latest",
            "prompt": f"Movie search query: {query}\n\nTop results:\n{context}\n\nIn 2 sentences, recommend the best match:",
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 80}
        }).encode()
        req = urllib.request.Request(
            "http://host.docker.internal:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            result = json.loads(r.read())
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "response": result.get("response", ""),
            "latency_ms": latency_ms,
            "model": "llama3:latest",
            "cost": "$0.00  (local Ollama)"
        }
    except Exception as e:
        return {"error": str(e), "response": ""}


def run_edge_pipeline(audio_path: str) -> dict:
    """
    Full edge pipeline:
    Audio → Faster-Whisper → BM25+FAISS → Cross-Encoder → Llama3
    Zero cloud API cost. Runs on Apple Silicon / CPU.
    """
    print(f"Edge Pipeline: {audio_path}")
    print("-" * 40)

    # Stage 1: ASR
    print("Stage 1: Faster-Whisper transcription...")
    asr = transcribe_local(audio_path)
    if "error" in asr:
        print(f"  ⚠️  {asr['error']}")
        query = "crime thriller recommendation"  # demo fallback
    else:
        query = asr["transcription"]
        print(f"  ✅ '{query}' ({asr['latency_ms']}ms, {asr['model']})")

    # Stage 2: Retrieval
    print("Stage 2: StreamLens hybrid_ltr retrieval...")
    import urllib.parse
    search = search_streamlens(query, k=5)
    hits = search.get("hits", [])
    print(f"  ✅ {len(hits)} results retrieved")

    # Stage 3: Context building
    context = "\n".join([
        f"- {h.get('title','?')} (score: {h.get('score',0):.3f})"
        for h in hits[:3]
    ])

    # Stage 4: Local LLM generation
    print("Stage 4: Llama3 (Ollama) generation...")
    gen = generate_local(query, context)
    if "error" in gen:
        print(f"  ⚠️  Ollama not running: {gen['error']}")
    else:
        print(f"  ✅ Generated ({gen['latency_ms']}ms)")

    return {
        "pipeline": "edge (zero API cost)",
        "asr": asr,
        "query": query,
        "n_results": len(hits),
        "top_results": [h.get("title") for h in hits[:3]],
        "generation": gen,
        "total_cost": "$0.00",
        "runs_on": "CPU / Apple Silicon (no GPU required)"
    }


if __name__ == "__main__":
    import urllib.parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="test.webm", help="Audio file path")
    parser.add_argument("--model", default="base", help="Whisper model size")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"Audio file not found: {args.audio}")
        print("Testing pipeline with demo query instead...")
        # Demo mode — skip ASR, test retrieval + generation
        result = {
            "pipeline": "edge demo (no audio file)",
            "query": "gritty crime thriller",
            "note": "Provide --audio path.webm to test full ASR pipeline"
        }
        search = search_streamlens("gritty crime thriller", k=5)
        result["n_results"] = len(search.get("hits", []))
        result["top_results"] = [h.get("title") for h in search.get("hits", [])[:3]]
        print(json.dumps(result, indent=2))
    else:
        result = run_edge_pipeline(args.audio)
        print(json.dumps(result, indent=2))
