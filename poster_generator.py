"""
StreamLens — Phenomenal Poster Generator
3-tier system: Flux.1 API (best) → DALL-E 3 (great) → SDXL local (free)
All three produce phenomenal quality vs SD v1.5 garbage.

Setup:
    pip install huggingface_hub --break-system-packages
    # Get free HF token: https://huggingface.co/settings/tokens
    export HF_TOKEN=hf_your_token_here

Usage:
    python poster_generator.py --title "Pulp Fiction" --genre "Crime,Thriller"
    python poster_generator.py --title "Toy Story" --genre "Animation,Comedy" --tier dalle3
    python poster_generator.py --title "Inception" --genre "Sci-Fi,Thriller" --tier sdxl
"""
from __future__ import annotations
import argparse, base64, hashlib, json, os, time, urllib.request, urllib.error
from pathlib import Path

CACHE_DIR = Path("artifacts/posters_hq")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Prompt Engineering ────────────────────────────────────────────────
def build_prompt(title: str, genres: str, tags: str = "") -> str:
    """Genre-aware cinematic prompt. This is where quality lives."""
    import re
    clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
    gl = [g.strip().lower() for g in genres.split(",") if g.strip()]

    style_map = {
        "crime":     "neo-noir atmosphere, rain-slicked streets, dramatic shadows, neon reflections",
        "thriller":  "tense dark atmosphere, high contrast lighting, psychological intensity",
        "action":    "explosive dynamic composition, motion blur, dramatic lighting, cinematic scale",
        "horror":    "dark fog, blood red accents, ominous shadows, moonlit gothic atmosphere",
        "romance":   "golden hour warmth, soft bokeh, intimate composition, pastel tones",
        "comedy":    "vibrant saturated colours, playful composition, warm lighting, joyful",
        "drama":     "desaturated realism, shallow depth of field, emotional natural lighting",
        "animation": "bold vibrant illustration, clean graphic style, expressive characters",
        "sci-fi":    "neon blue cyberpunk atmosphere, futuristic skyline, holographic elements",
        "fantasy":   "magical golden light, epic scale, ethereal atmosphere, mystical fog",
        "war":       "desaturated battle atmosphere, smoke and ash, dramatic sky, epic scale",
        "western":   "dusty golden hour, wide open landscape, weathered textures, warm tones",
        "documentary":"photorealistic journalistic photography, natural lighting, raw authenticity",
        "adventure": "sweeping landscape, vibrant colours, heroic composition, sense of wonder",
    }

    style = "cinematic atmosphere, dramatic lighting, professional composition"
    for g in gl:
        if g in style_map:
            style = style_map[g]
            break

    tag_str = ""
    if tags:
        good_tags = [t.strip() for t in tags.split(",")
                     if t.strip() and len(t.strip()) > 3 and
                     t.strip().lower() not in {"film","movie","good","great","classic"}][:2]
        if good_tags:
            tag_str = ", " + ", ".join(good_tags)

    return (
        f"Professional movie poster for '{clean}'{tag_str}. "
        f"{style}. "
        f"Award-winning cinematography, 4K ultra detailed, "
        f"striking visual composition, bold typography placement, "
        f"Hollywood production quality. "
        f"Aspect ratio 2:3 portrait orientation."
    )


# ── TIER 1: Flux.1 via HuggingFace Inference API (FREE, BEST) ────────
def generate_flux(title: str, genres: str, tags: str = "",
                  doc_id: str = "", force: bool = False) -> dict:
    """
    Flux.1-dev via HuggingFace Inference API.
    Free with HF token. Best quality available in 2024.
    Get token: https://huggingface.co/settings/tokens (free account)
    """
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        return {"error": "HF_TOKEN not set. Get free token: https://huggingface.co/settings/tokens"}

    cache_key = doc_id or hashlib.md5(f"flux{title}{genres}".encode()).hexdigest()[:8]
    out_path = CACHE_DIR / f"flux_{cache_key}.png"
    if out_path.exists() and not force:
        return {"path": str(out_path), "cached": True, "tier": "flux1", "title": title}

    prompt = build_prompt(title, genres, tags)
    print(f"\n🎨 Flux.1 generating: {title}")
    print(f"   Prompt: {prompt[:100]}...")

    payload = json.dumps({
        "inputs": prompt,
        "parameters": {
            "width": 512, "height": 768,
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
        }
    }).encode()

    # Try flux-dev first, fallback to schnell (faster)
    for model in [
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-schnell",
    ]:
        req = urllib.request.Request(
            f"https://api-inference.huggingface.co/models/{model}",
            data=payload,
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json",
                "Accept": "image/png",
            }
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status == 200:
                    img_bytes = r.read()
                    out_path.write_bytes(img_bytes)
                    latency = round((time.perf_counter() - t0) * 1000, 1)
                    print(f"   ✅ Saved: {out_path} ({latency}ms) model={model.split('/')[-1]}")
                    return {
                        "path": str(out_path),
                        "cached": False,
                        "tier": "flux1",
                        "model": model,
                        "title": title,
                        "genres": genres,
                        "prompt": prompt,
                        "latency_ms": latency,
                        "cost": "$0.00 (HuggingFace free tier)",
                    }
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:200]
            print(f"   {model} → HTTP {e.code}: {body}")
            if e.code == 503:
                print("   Model loading, waiting 20s...")
                time.sleep(20)
            continue
        except Exception as ex:
            print(f"   {model} → {ex}")
            continue

    return {"error": "Flux.1 unavailable — try --tier dalle3 or --tier sdxl"}


# ── TIER 2: DALL-E 3 via OpenAI (GREAT, ~$0.04/image) ───────────────
def generate_dalle3(title: str, genres: str, tags: str = "",
                    doc_id: str = "", force: bool = False) -> dict:
    """
    DALL-E 3 — best text understanding, photorealistic.
    Uses your existing OPENAI_API_KEY. ~$0.04 per 1024x1792 image.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key.startswith("sk-"):
        return {"error": "OPENAI_API_KEY not set"}

    cache_key = doc_id or hashlib.md5(f"dalle3{title}{genres}".encode()).hexdigest()[:8]
    out_path = CACHE_DIR / f"dalle3_{cache_key}.png"
    if out_path.exists() and not force:
        return {"path": str(out_path), "cached": True, "tier": "dalle3", "title": title}

    prompt = build_prompt(title, genres, tags)
    print(f"\n🎨 DALL-E 3 generating: {title}")
    print(f"   Prompt: {prompt[:100]}...")

    payload = json.dumps({
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1792",   # portrait — perfect for movie poster
        "quality": "hd",
        "response_format": "b64_json",
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/images/generations",
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read())
        img_b64 = result["data"][0]["b64_json"]
        revised_prompt = result["data"][0].get("revised_prompt", prompt)
        img_bytes = base64.b64decode(img_b64)
        out_path.write_bytes(img_bytes)
        latency = round((time.perf_counter() - t0) * 1000, 1)
        print(f"   ✅ Saved: {out_path} ({latency}ms)")
        return {
            "path": str(out_path),
            "cached": False,
            "tier": "dalle3",
            "model": "dall-e-3",
            "title": title,
            "genres": genres,
            "prompt": prompt,
            "revised_prompt": revised_prompt[:200],
            "latency_ms": latency,
            "cost": "~$0.04 (DALL-E 3 HD 1024x1792)",
        }
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        return {"error": f"DALL-E 3 HTTP {e.code}: {body}"}
    except Exception as ex:
        return {"error": str(ex)}


# ── TIER 3: SDXL Local (FREE, good quality, slow on CPU) ─────────────
def generate_sdxl(title: str, genres: str, tags: str = "",
                  doc_id: str = "", steps: int = 30,
                  force: bool = False) -> dict:
    """
    Stable Diffusion XL — much better than SD v1.5.
    Free, local, no API needed. ~45s on Apple Silicon MPS.
    """
    try:
        from diffusers import StableDiffusionXLPipeline
        import torch
    except ImportError:
        return {"error": "pip install diffusers transformers accelerate"}

    cache_key = doc_id or hashlib.md5(f"sdxl{title}{genres}".encode()).hexdigest()[:8]
    out_path = CACHE_DIR / f"sdxl_{cache_key}.png"
    if out_path.exists() and not force:
        return {"path": str(out_path), "cached": True, "tier": "sdxl", "title": title}

    prompt = build_prompt(title, genres, tags)
    neg = "blurry, low quality, watermark, text errors, distorted, nsfw, amateur"

    print(f"\n🎨 SDXL generating: {title}")
    print(f"   Loading SDXL (downloading ~6GB first time)...")

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)
    pipe.enable_attention_slicing()

    t0 = time.perf_counter()
    with torch.inference_mode():
        image = pipe(
            prompt,
            negative_prompt=neg,
            num_inference_steps=steps,
            guidance_scale=7.0,
            width=768, height=1024,
        ).images[0]

    latency = round((time.perf_counter() - t0) * 1000, 1)
    image.save(out_path)
    print(f"   ✅ Saved: {out_path} ({latency}ms)")

    return {
        "path": str(out_path),
        "cached": False,
        "tier": "sdxl",
        "model": "stable-diffusion-xl-base-1.0",
        "title": title,
        "genres": genres,
        "prompt": prompt,
        "latency_ms": latency,
        "cost": "$0.00 (local SDXL)",
    }


# ── Auto-tier: try best available ────────────────────────────────────
def generate_poster(title: str, genres: str = "Drama", tags: str = "",
                    doc_id: str = "", tier: str = "auto",
                    force: bool = False) -> dict:
    """
    Auto-select best available tier:
    Flux.1 (HF_TOKEN set) → DALL-E 3 (OPENAI_API_KEY set) → SDXL local
    """
    if tier == "flux" or (tier == "auto" and os.environ.get("HF_TOKEN")):
        result = generate_flux(title, genres, tags, doc_id, force)
        if "error" not in result:
            return result
        print(f"   Flux failed: {result['error']} — trying DALL-E 3")

    if tier == "dalle3" or (tier == "auto" and os.environ.get("OPENAI_API_KEY","").startswith("sk-")):
        result = generate_dalle3(title, genres, tags, doc_id, force)
        if "error" not in result:
            return result
        print(f"   DALL-E 3 failed: {result['error']} — trying SDXL")

    return generate_sdxl(title, genres, tags, doc_id, force=force)


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--title",  default="Pulp Fiction")
    parser.add_argument("--genre",  default="Crime,Thriller")
    parser.add_argument("--tags",   default="cult film,nonlinear")
    parser.add_argument("--doc_id", default="")
    parser.add_argument("--tier",   default="auto",
                        choices=["auto","flux","dalle3","sdxl"])
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()

    result = generate_poster(
        title=args.title, genres=args.genre,
        tags=args.tags, doc_id=args.doc_id,
        tier=args.tier, force=args.force,
    )

    print(f"\n✅ Result:")
    print(json.dumps({k: v for k, v in result.items()
                     if k not in {"revised_prompt","negative_prompt"}}, indent=2))

    if "path" in result and not result.get("error"):
        import subprocess
        subprocess.run(["open", result["path"]], check=False)
