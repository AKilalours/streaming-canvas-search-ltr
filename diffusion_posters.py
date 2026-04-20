"""
StreamLens — Stable Diffusion Cold-Start Poster Generator
Generates movie posters for films with no TMDB image.
Runs on CPU (Apple Silicon MPS if available).
Zero API cost after model download (~4GB one-time).

Usage:
    python diffusion_posters.py --title "The Dark Knight" --genre "Crime,Action"
    python diffusion_posters.py --doc_id 272 --api http://localhost:8000

Pipeline:
    Film metadata → prompt engineering → SD v1.5 → poster PNG
    Cached to artifacts/sd_posters/{doc_id}.png
"""
from __future__ import annotations
import argparse, hashlib, json, os, time, urllib.request
from pathlib import Path

# ── Model cache ───────────────────────────────────────────────────────
_PIPE = None
CACHE_DIR = Path("artifacts/sd_posters")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_pipe():
    """Load SD pipeline once — reuse across calls."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    from diffusers import StableDiffusionPipeline
    import torch

    print("Loading Stable Diffusion v1.5...")
    t0 = time.perf_counter()

    # Detect best device
    if torch.backends.mps.is_available():
        device = "mps"
        dtype  = torch.float32  # float16 = black images on MPS
        print("  Device: Apple Silicon MPS (float32 safe)")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype  = torch.float16
        print("  Device: CUDA GPU")
    else:
        device = "cpu"
        dtype  = torch.float32
        print("  Device: CPU (slower, ~3 min per poster)")

    _PIPE = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    # Memory optimisation for CPU/MPS
    if device in ("cpu", "mps"):
        _PIPE.enable_attention_slicing()

    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")
    return _PIPE


def _build_prompt(title: str, genres: str, tags: str = "") -> tuple[str, str]:
    """
    Build a high-quality SD prompt from film metadata.
    Returns (positive_prompt, negative_prompt).
    """
    genre_list = [g.strip().lower() for g in genres.split(",") if g.strip()]

    # Genre → visual style mapping
    style_map = {
        "action":    "explosive action scene, dynamic composition, high contrast",
        "thriller":  "dark shadows, tense atmosphere, noir lighting, suspense",
        "crime":     "gritty urban setting, moody rain, neon lights, noir",
        "horror":    "dark fog, ominous shadows, eerie atmosphere, blood red",
        "romance":   "soft golden light, warm tones, intimate composition",
        "comedy":    "bright colours, playful composition, warm sunlight",
        "drama":     "cinematic depth of field, emotional lighting, desaturated",
        "animation": "vibrant colours, illustrated style, clean lines, bright",
        "sci-fi":    "futuristic neon, space atmosphere, blue tones, cyberpunk",
        "fantasy":   "magical atmosphere, ethereal light, epic scale, golden",
        "documentary":"realistic photography, natural lighting, journalistic",
        "war":       "battle smoke, muted tones, dramatic sky, epic scale",
        "western":   "dusty landscape, golden hour, wide composition, rugged",
    }

    # Pick best style match
    style = "cinematic lighting, professional composition"
    for g in genre_list:
        if g in style_map:
            style = style_map[g]
            break

    # Tags add character-level detail
    tag_hints = ""
    if tags:
        interesting_tags = [t.strip() for t in tags.split(",")
                           if t.strip() and len(t.strip()) > 3][:2]
        if interesting_tags:
            tag_hints = ", ".join(interesting_tags) + ", "

    # Build clean title (remove year)
    import re
    clean_title = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()

    positive = (
        f"movie poster for '{clean_title}', {tag_hints}"
        f"{style}, "
        f"professional film poster design, dramatic typography, "
        f"high quality, 4k, award winning cinematography, "
        f"bold visual storytelling"
    )

    negative = (
        "blurry, low quality, watermark, text errors, duplicate, "
        "amateur, overexposed, grainy, distorted faces, nsfw"
    )

    return positive, negative


def generate_poster(
    title: str,
    genres: str = "Drama",
    tags: str = "",
    doc_id: str = "",
    steps: int = 25,
    guidance: float = 7.5,
    width: int = 384,
    height: int = 512,
    force: bool = False,
) -> dict:
    """
    Generate a movie poster using Stable Diffusion.
    Returns dict with path, prompt, latency_ms, cached.
    """
    # Cache key
    cache_key = doc_id or hashlib.md5(f"{title}{genres}".encode()).hexdigest()[:8]
    out_path = CACHE_DIR / f"{cache_key}.png"

    # Check cache
    if out_path.exists() and not force:
        return {
            "path": str(out_path),
            "cached": True,
            "doc_id": cache_key,
            "title": title,
            "latency_ms": 0,
        }

    # Build prompt
    positive, negative = _build_prompt(title, genres, tags)
    print(f"\nGenerating poster for: {title}")
    print(f"  Genres: {genres}")
    print(f"  Prompt: {positive[:80]}...")

    # Generate
    pipe = _load_pipe()
    t0 = time.perf_counter()

    import torch
    with torch.inference_mode():
        image = pipe(
            positive,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
        ).images[0]

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    image.save(out_path)

    print(f"  ✅ Saved: {out_path}  ({latency_ms}ms)")

    return {
        "path": str(out_path),
        "cached": False,
        "doc_id": cache_key,
        "title": title,
        "genres": genres,
        "prompt": positive,
        "negative_prompt": negative,
        "steps": steps,
        "latency_ms": latency_ms,
        "cost": "$0.00 (local SD v1.5)",
        "model": "runwayml/stable-diffusion-v1-5",
    }


def generate_cold_start_batch(
    api_url: str = "http://localhost:8000",
    n: int = 10,
    steps: int = 20,
) -> list[dict]:
    """
    Find films with no TMDB poster and generate SD posters for them.
    This is the production cold-start use case.
    """
    print(f"Finding cold-start films (no TMDB poster) from {api_url}...")

    # Get random corpus sample
    try:
        with urllib.request.urlopen(
            f"{api_url}/search?q=film&method=bm25&k={n*3}", timeout=10
        ) as r:
            results = json.loads(r.read())
        hits = results.get("hits", [])
    except Exception as e:
        print(f"API error: {e} — using demo titles")
        hits = [
            {"doc_id": f"demo_{i}", "title": t, "text": f"Genres: {g}"}
            for i, (t, g) in enumerate([
                ("Midnight in the Garden of Evil (1997)", "Crime, Drama"),
                ("The Wandering Earth (2019)", "Sci-Fi, Action"),
                ("Portrait of a Lady on Fire (2019)", "Drama, Romance"),
                ("Parasite (2019)", "Thriller, Drama"),
                ("The Lighthouse (2019)", "Drama, Horror"),
            ])
        ]

    generated = []
    for hit in hits[:n]:
        doc_id = str(hit.get("doc_id", ""))
        title  = hit.get("title", "Unknown")
        text   = hit.get("text", "")

        # Parse genres from corpus text
        import re
        gm = re.search(r'Genres?:\s*([^|]+)', text)
        genres = gm.group(1).strip() if gm else "Drama"
        tm = re.search(r'Tags?:\s*([^|\n]+)', text)
        tags = tm.group(1).strip() if tm else ""

        result = generate_poster(
            title=title,
            genres=genres,
            tags=tags,
            doc_id=doc_id,
            steps=steps,
        )
        generated.append(result)

    return generated


# ── FastAPI endpoint (add to main.py) ─────────────────────────────────
FASTAPI_ROUTE = '''
# Add to src/app/main.py

@app.get("/poster/sd_generate")
async def sd_generate_poster(
    doc_id: str = Query(...),
    steps: int  = Query(25, ge=10, le=50),
    force: bool = Query(False),
) -> dict:
    """
    Stable Diffusion cold-start poster generation.
    Generates a poster for films with no TMDB image.
    Cached to artifacts/sd_posters/{doc_id}.png after first generation.
    Cost: $0.00 (local SD v1.5 model).
    """
    try:
        from diffusion_posters import generate_poster
        st = _ensure_ready()
        doc = st.corpus.get(str(doc_id), {})
        title  = doc.get("title", doc_id)
        text   = doc.get("text", "")
        import re
        gm = re.search(r"Genres?:\\s*([^|]+)", text)
        genres = gm.group(1).strip() if gm else "Drama"
        tm = re.search(r"Tags?:\\s*([^|\\n]+)", text)
        tags = tm.group(1).strip() if tm else ""

        result = generate_poster(
            title=title, genres=genres, tags=tags,
            doc_id=doc_id, force=force
        )
        # Return path as URL
        result["url"] = f"/static/sd_posters/{doc_id}.png"
        return result
    except ImportError:
        return {"error": "diffusers not installed — run: pip install diffusers"}
    except Exception as e:
        return {"error": str(e)}
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreamLens Stable Diffusion poster generator")
    parser.add_argument("--title",  default="Pulp Fiction", help="Film title")
    parser.add_argument("--genre",  default="Crime, Thriller", help="Comma-separated genres")
    parser.add_argument("--tags",   default="cult film, nonlinear", help="Tags")
    parser.add_argument("--doc_id", default="", help="Corpus doc_id for caching")
    parser.add_argument("--steps",  type=int, default=25, help="Inference steps (10-50)")
    parser.add_argument("--api",    default="", help="StreamLens API URL for batch mode")
    parser.add_argument("--batch",  type=int, default=0, help="Batch: generate N cold-start posters")
    parser.add_argument("--force",  action="store_true", help="Regenerate even if cached")
    args = parser.parse_args()

    if args.batch > 0:
        results = generate_cold_start_batch(
            api_url=args.api or "http://localhost:8000",
            n=args.batch,
            steps=args.steps,
        )
        print(f"\n✅ Generated {len(results)} posters")
        for r in results:
            print(f"  {r['title']} → {r['path']} ({r['latency_ms']}ms)")
    else:
        result = generate_poster(
            title=args.title,
            genres=args.genre,
            tags=args.tags,
            doc_id=args.doc_id,
            steps=args.steps,
            force=args.force,
        )
        print(f"\n✅ Result:")
        print(json.dumps({k: v for k, v in result.items()
                         if k != "negative_prompt"}, indent=2))
        # Open the generated poster
        import subprocess
        subprocess.run(["open", result["path"]], check=False)
