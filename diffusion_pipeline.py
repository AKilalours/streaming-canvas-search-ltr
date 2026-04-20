"""
StreamLens — Diffusion Model Poster Generation Pipeline
Demonstrates diffusion model architecture knowledge + production deployment.

Architecture:
  Text prompt → CLIP text encoder → Latent diffusion (UNet + DDPM scheduler)
  → VAE decoder → PNG image

Key insight: DALL-E 2 cannot render text reliably.
Solution: Generate pure visual cinematic stills — no text in prompt.
The StreamLens UI overlays the title on top of the generated image.

Usage:
    python diffusion_pipeline.py --demo
    python diffusion_pipeline.py --title "Pulp Fiction" --genre "Crime,Thriller"
    python diffusion_pipeline.py --schedule
"""
from __future__ import annotations
import argparse, base64, hashlib, json, os, re, time, urllib.request, urllib.error
from pathlib import Path

CACHE = Path("artifacts/sd_posters")
CACHE.mkdir(parents=True, exist_ok=True)

# ── Genre → Cinematic visual style mapping ────────────────────────────
# Key skill: understanding how CLIP text encoders map words to visual space
STYLES = {
    "crime":     "rain-drenched dark alley, neon signs reflecting on wet pavement, deep shadows, urban noir, low angle shot",
    "thriller":  "shadowy silhouette against harsh backlight, tense atmosphere, high contrast black and white, cinematic",
    "action":    "explosion with dramatic backlight, motion blur, hero silhouette, golden hour epic scale",
    "horror":    "moonlit foggy forest, red mist, shadowy figure, terrifying dark atmosphere, gothic architecture",
    "romance":   "couple silhouette at golden sunset, soft warm bokeh, intimate close-up, pastel dreamy colours",
    "comedy":    "bright sunny street scene, vibrant saturated colours, playful warm light, cheerful atmosphere",
    "drama":     "single person by rainy window, desaturated cool tones, emotional natural light, contemplative",
    "animation": "magical glowing portal in enchanted forest, vibrant saturated colours, whimsical fantastical",
    "sci-fi":    "futuristic neon-lit cityscape at night, holographic lights, vast space station, cyberpunk blue",
    "fantasy":   "magical castle on floating island, golden ethereal light, dragons, mystical epic landscape",
    "war":       "soldiers silhouette at sunrise, smoky battlefield, dramatic orange sky, epic historical scale",
    "western":   "lone rider on horseback at golden hour, vast desert canyon, dusty warm amber light",
    "adventure": "explorer at cliff edge overlooking vast jungle, epic sweeping landscape, sense of wonder",
    "documentary":"raw authentic street photography, natural light, photorealistic, journalistic composition",
    "musical":   "performer on dramatic stage, spotlight from above, theatrical smoke, vibrant coloured lights",
}

def build_prompt(title: str, genres: str, tags: str = "") -> str:
    """
    Diffusion-optimised prompt engineering.
    
    Key principle: DALL-E 2 cannot render text — never ask for words/titles.
    Instead: describe the VISUAL MOOD and CINEMATIC ATMOSPHERE only.
    The UI overlays film title on top of the generated image.
    
    Prompt structure (optimised for CLIP encoder):
      [specific scene] + [lighting] + [mood] + [quality boosters]
    """
    gl = [g.strip().lower() for g in genres.split(",") if g.strip()]

    # Get best style match
    style = "dramatic cinematic scene, moody lighting, rich atmospheric depth"
    for g in gl:
        if g in STYLES:
            style = STYLES[g]
            break

    # Only use tags that describe visual elements, not narrative
    visual_tags = {
        "pixar": "colourful animated world",
        "cult film": "stylised surreal atmosphere",
        "nonlinear": "fragmented mirror reflections",
        "superhero": "dramatic cape in wind",
        "musical": "dramatic stage spotlight",
        "biographical": "vintage film grain texture",
    }
    tag_extras = ""
    if tags:
        for t in tags.split(","):
            t = t.strip().lower()
            if t in visual_tags:
                tag_extras = ", " + visual_tags[t]
                break

    return (
        f"{style}{tag_extras}. "
        f"No text, no words, no letters, no titles, no writing. "
        f"Cinematic photography, ultra-detailed, 4K, "
        f"professional film still, dramatic composition, "
        f"shallow depth of field, award-winning cinematography, "
        f"rich deep colours, perfect lighting."
    )

def negative_prompt() -> str:
    return (
        "text, words, letters, title, watermark, blurry, "
        "low quality, duplicate, ugly, deformed, nsfw, "
        "overexposed, grainy, amateur, distorted"
    )

# ── DDPM Noise Schedule ────────────────────────────────────────────────
def ddpm_noise_schedule(timesteps: int = 1000) -> dict:
    """
    Linear DDPM beta schedule — mathematical foundation of all diffusion models.
    Forward: q(x_t|x_{t-1}) = N(x_t; sqrt(1-β_t)*x_{t-1}, β_t*I)
    Reverse: p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
    """
    import numpy as np
    betas = np.linspace(0.0001, 0.02, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    return {
        "timesteps": timesteps,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "betas_first5": betas[:5].tolist(),
        "alphas_cumprod_first5": alphas_cumprod[:5].tolist(),
        "signal_to_noise_t0":   float(alphas_cumprod[0]),
        "signal_to_noise_t999": float(alphas_cumprod[-1]),
        "interpretation": {
            "t=0":   "Clean image. SNR≈1.0. No noise added.",
            "t=500": f"Half noisy. SNR≈{float(alphas_cumprod[499]):.4f}.",
            "t=999": "Pure Gaussian noise. SNR≈0. Original image unrecoverable.",
        },
        "inference": (
            "Start from x_T ~ N(0,I). "
            "Run UNet to predict noise ε_θ(x_t, t). "
            "Subtract predicted noise: x_{t-1} = (x_t - sqrt(β_t)*ε_θ) / sqrt(α_t). "
            "Repeat T→0. Result is generated image."
        )
    }

# ── DALL-E 2 Generation ───────────────────────────────────────────────
def generate_dalle2(title: str, genres: str, tags: str = "",
                    doc_id: str = "", force: bool = False) -> dict:
    """
    DALL-E 2: CLIP text encoder → diffusion prior → DALL-E decoder.
    Cost: $0.002 per 512x512. No text in prompt — UI overlays title.
    """
    key = os.environ.get("OPENAI_API_KEY","").strip()
    if not key.startswith("sk-"):
        return {"error": "OPENAI_API_KEY not set in environment"}

    cache_key = doc_id or hashlib.md5(f"dalle2v2{title}{genres}".encode()).hexdigest()[:8]
    out_path = CACHE / f"dalle2_{cache_key}.png"
    if out_path.exists() and not force:
        print(f"   ✅ Cached: {out_path}")
        return {"path": str(out_path), "cached": True, "tier": "dalle2", "title": title}

    prompt = build_prompt(title, genres, tags)
    print(f"\n🎨 DALL-E 2 → {title}")
    print(f"   Genres: {genres}")
    print(f"   Prompt: {prompt[:100]}...")

    payload = json.dumps({
        "model": "dall-e-2",
        "prompt": prompt,
        "n": 1,
        "size": "512x512",
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
        with urllib.request.urlopen(req, timeout=30) as r:
            result = json.loads(r.read())
        img_bytes = base64.b64decode(result["data"][0]["b64_json"])
        out_path.write_bytes(img_bytes)
        ms = round((time.perf_counter()-t0)*1000, 1)
        print(f"   ✅ Saved: {out_path} ({ms}ms, {len(img_bytes)//1024}KB)")
        return {
            "path": str(out_path),
            "cached": False,
            "tier": "dalle2",
            "model": "dall-e-2",
            "architecture": "CLIP text encoder → latent diffusion prior → DALL-E 2 decoder",
            "title": title,
            "genres": genres,
            "prompt": prompt,
            "latency_ms": ms,
            "cost": "$0.002 per image",
            "note": "No text in prompt — DALL-E 2 cannot render text. UI overlays title.",
        }
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as ex:
        return {"error": str(ex)}


def generate(title: str, genres: str = "Drama", tags: str = "",
             doc_id: str = "", force: bool = False) -> dict:
    return generate_dalle2(title, genres, tags, doc_id, force)


def run_demo():
    films = [
        ("Pulp Fiction (1994)",      "Crime,Thriller",  "cult film,nonlinear"),
        ("Toy Story (1995)",         "Animation,Comedy","pixar,friendship"),
        ("Inception (2010)",         "Sci-Fi,Thriller", "mind-bending,dreams"),
        ("Braveheart (1995)",        "War,Drama",       "historical,battle"),
        ("Grand Budapest Hotel",     "Comedy,Drama",    "quirky,stylised"),
    ]
    print("StreamLens Diffusion Pipeline — Demo")
    print("=" * 55)
    print("DDPM Noise Schedule (mathematical foundation):")
    s = ddpm_noise_schedule()
    print(f"  β: {s['beta_start']} → {s['beta_end']} over {s['timesteps']} steps")
    print(f"  SNR at t=0:   {s['signal_to_noise_t0']:.4f} (clean)")
    print(f"  SNR at t=999: {s['signal_to_noise_t999']:.6f} (pure noise)")
    print(f"  Inference: {s['inference'][:80]}...")
    print("=" * 55)
    print(f"Prompt strategy: visual mood only — NO text/words")
    print(f"  DALL-E 2 cannot render readable text reliably.")
    print(f"  StreamLens UI overlays the film title on top.")
    print("=" * 55)

    results = []
    for title, genres, tags in films:
        r = generate(title, genres, tags)
        results.append((title, r))
        if "path" in r and "error" not in r:
            import subprocess
            subprocess.run(["open", r["path"]], check=False)
            time.sleep(1)

    print(f"\n{'='*55}")
    print("RESULTS")
    print(f"{'='*55}")
    ok = sum(1 for _, r in results if "error" not in r)
    print(f"Generated: {ok}/{len(films)}")
    for title, r in results:
        if "error" not in r:
            print(f"  ✅ {title[:35]} → {r['path']} ({r.get('latency_ms',0):.0f}ms)")
        else:
            print(f"  ❌ {title[:35]} → {r['error'][:60]}")
    if ok > 0:
        print(f"\nTotal cost: ${ok * 0.002:.3f}")
        print(f"Cached permanently: zero cost on subsequent runs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--title",    default="Pulp Fiction")
    ap.add_argument("--genre",    default="Crime,Thriller")
    ap.add_argument("--tags",     default="cult film,nonlinear")
    ap.add_argument("--doc_id",   default="")
    ap.add_argument("--force",    action="store_true")
    ap.add_argument("--demo",     action="store_true")
    ap.add_argument("--schedule", action="store_true")
    args = ap.parse_args()

    if args.schedule:
        print(json.dumps(ddpm_noise_schedule(), indent=2))
    elif args.demo:
        run_demo()
    else:
        result = generate(
            args.title, args.genre, args.tags, args.doc_id, args.force
        )
        print(json.dumps({k:v for k,v in result.items()
                         if k not in {"prompt"}}, indent=2))
        if "path" in result and "error" not in result:
            import subprocess
            subprocess.run(["open", result["path"]], check=False)
