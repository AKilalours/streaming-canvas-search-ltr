"""
StreamLens — Diffusion Model Pipeline
DDPM noise schedule (pure numpy) + DALL-E 3 HD poster generation.

Demonstrates:
1. DDPM forward/reverse diffusion mathematics from scratch
2. Production DALL-E 3 latent diffusion deployment
3. Genre-aware cinematic prompt engineering
4. Cold-start poster generation for films with no TMDB image

Architecture:
    Text → CLIP encoder → Latent diffusion (UNet + DDPM scheduler)
    → VAE decoder → 1024x1792 HD image

Usage:
    python diffusion_pipeline.py --demo              # generate 5 posters
    python diffusion_pipeline.py --title "Inception" --genre "Sci-Fi,Thriller"
    python diffusion_pipeline.py --schedule          # show DDPM math
    python diffusion_pipeline.py --schedule --plot   # visualise noise curve
"""
from __future__ import annotations
import argparse, base64, hashlib, json, os, re, time, urllib.request, urllib.error
from pathlib import Path

CACHE = Path("artifacts/sd_posters")
CACHE.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# PART 1 — DDPM NOISE SCHEDULE (pure numpy, no ML framework needed)
# The mathematical foundation of ALL diffusion models:
# Stable Diffusion, DALL-E, Midjourney, Imagen — all use this.
# ═══════════════════════════════════════════════════════════════════════

def ddpm_noise_schedule(timesteps: int = 1000,
                        beta_start: float = 0.0001,
                        beta_end: float = 0.02) -> dict:
    """
    Linear DDPM beta schedule from Ho et al. 2020 "Denoising Diffusion
    Probabilistic Models" — the paper that started the diffusion revolution.

    Forward process (training):
        q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) * x_{t-1}, β_t * I)
        q(x_t | x_0)     = N(x_t; sqrt(ᾱ_t) * x_0, (1-ᾱ_t) * I)

    Reverse process (inference / generation):
        p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))

    Where:
        β_t  = noise variance at step t (increases linearly)
        α_t  = 1 - β_t
        ᾱ_t  = ∏(α_1 ... α_t)  [cumulative product]
        SNR  = ᾱ_t / (1 - ᾱ_t)  [signal-to-noise ratio]
    """
    import numpy as np

    # Linear schedule: β increases from beta_start to beta_end
    betas         = np.linspace(beta_start, beta_end, timesteps)
    alphas        = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)  # ᾱ_t

    # Signal-to-noise ratio at each timestep
    snr = alphas_cumprod / (1.0 - alphas_cumprod)

    # Posterior mean coefficient (used in reverse step)
    # μ_θ(x_t, t) = (1/sqrt(α_t)) * (x_t - β_t/sqrt(1-ᾱ_t) * ε_θ)
    posterior_mean_coef1 = (
        np.sqrt(alphas_cumprod[:-1]) * betas[1:] /
        (1.0 - alphas_cumprod[1:])
    )
    posterior_mean_coef2 = (
        np.sqrt(alphas[1:]) * (1.0 - alphas_cumprod[:-1]) /
        (1.0 - alphas_cumprod[1:])
    )

    return {
        "paper": "Ho et al. 2020 — Denoising Diffusion Probabilistic Models",
        "timesteps": timesteps,
        "schedule": "linear",
        "beta_start": beta_start,
        "beta_end": beta_end,

        # Key values at selected timesteps
        "t=0   (clean)":  {
            "beta": float(betas[0]),
            "alpha_cumprod": float(alphas_cumprod[0]),
            "SNR": float(snr[0]),
            "interpretation": "Original image. Almost no noise added."
        },
        "t=250 (25%)": {
            "beta": float(betas[249]),
            "alpha_cumprod": float(alphas_cumprod[249]),
            "SNR": float(snr[249]),
            "interpretation": "Lightly noisy. Image still mostly recognisable."
        },
        "t=500 (50%)": {
            "beta": float(betas[499]),
            "alpha_cumprod": float(alphas_cumprod[499]),
            "SNR": float(snr[499]),
            "interpretation": "Half signal, half noise. Edges blurring."
        },
        "t=750 (75%)": {
            "beta": float(betas[749]),
            "alpha_cumprod": float(alphas_cumprod[749]),
            "SNR": float(snr[749]),
            "interpretation": "Mostly noise. Original barely detectable."
        },
        "t=999 (100%)": {
            "beta": float(betas[-1]),
            "alpha_cumprod": float(alphas_cumprod[-1]),
            "SNR": float(snr[-1]),
            "interpretation": "Pure Gaussian noise N(0,I). Original unrecoverable."
        },

        "inference_algorithm": {
            "step1": "Sample x_T ~ N(0, I)  [start from pure noise]",
            "step2": "For t = T, T-1, ..., 1:",
            "step3": "  Predict noise: ε_θ = UNet(x_t, t, text_embedding)",
            "step4": "  Compute mean: μ = (x_t - β_t/sqrt(1-ᾱ_t) * ε_θ) / sqrt(α_t)",
            "step5": "  Sample: x_{t-1} = μ + sqrt(β_t) * z  [z ~ N(0,I) if t>1 else 0]",
            "step6": "Return x_0  [the generated image]",
        },

        "why_it_works": (
            "The UNet learns to predict the noise added at each timestep. "
            "By iteratively subtracting predicted noise (T→0), "
            "it recovers a clean image from pure Gaussian noise. "
            "Text conditioning via CLIP embeddings guides WHAT to generate."
        ),

        "streamlens_usage": (
            "StreamLens uses DALL-E 3's latent diffusion (same DDPM principle "
            "but in compressed 64x64 latent space, not pixel space). "
            "VAE encodes image → latent → diffusion → VAE decodes → HD image."
        ),
    }


def plot_noise_schedule():
    """Visualise the DDPM noise curve — shows SNR decay over timesteps."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        T = 1000
        betas = np.linspace(0.0001, 0.02, T)
        alphas_cumprod = np.cumprod(1.0 - betas)
        snr = alphas_cumprod / (1.0 - alphas_cumprod)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor('#0d1117')
        for ax in axes:
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('#30363d')
            ax.spines['left'].set_color('#30363d')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        t = np.arange(T)

        axes[0].plot(t, betas, color='#f97316', linewidth=2)
        axes[0].set_title('β_t (noise variance)', color='white', fontsize=12)
        axes[0].set_xlabel('Timestep t', color='#8b949e')
        axes[0].set_ylabel('β_t', color='#8b949e')

        axes[1].plot(t, alphas_cumprod, color='#3b82f6', linewidth=2)
        axes[1].set_title('ᾱ_t (signal retention)', color='white', fontsize=12)
        axes[1].set_xlabel('Timestep t', color='#8b949e')
        axes[1].set_ylabel('ᾱ_t', color='#8b949e')
        axes[1].axhline(0.5, color='#ffffff', linestyle='--',
                        alpha=0.3, label='50% signal')
        axes[1].legend(labelcolor='white', facecolor='#161b22')

        axes[2].semilogy(t, snr, color='#00ff88', linewidth=2)
        axes[2].set_title('SNR = ᾱ_t / (1-ᾱ_t)', color='white', fontsize=12)
        axes[2].set_xlabel('Timestep t', color='#8b949e')
        axes[2].set_ylabel('Signal-to-Noise Ratio', color='#8b949e')

        plt.suptitle('DDPM Linear Noise Schedule — StreamLens Diffusion Pipeline',
                     color='white', fontsize=13, y=1.02)
        plt.tight_layout()

        out = Path("artifacts/sd_posters/ddpm_schedule.png")
        plt.savefig(out, dpi=150, bbox_inches='tight',
                    facecolor='#0d1117')
        plt.close()
        print(f"✅ Noise schedule plot saved: {out}")
        import subprocess
        subprocess.run(["open", str(out)], check=False)
        return str(out)
    except ImportError:
        print("matplotlib not installed — run: pip install matplotlib")
        return None


# ═══════════════════════════════════════════════════════════════════════
# PART 2 — DALL-E 3 HD POSTER GENERATION
# Production latent diffusion via OpenAI API
# ═══════════════════════════════════════════════════════════════════════

# Genre → cinematic visual prompt (no text — DALL-E 3 ignores text requests
# when they'd look bad; we overlay title in the StreamLens UI instead)
CINEMATIC_STYLES = {
    "crime":     "Two figures in sharp suits under rain-soaked neon street lights, dramatic shadows, puddles reflecting city glow, neo-noir film still",
    "thriller":  "Lone silhouette against harsh industrial backlight, tense atmosphere, desaturated high-contrast, cinematic psychological tension",
    "action":    "Hero silhouette leaping across rooftops at golden hour, city skyline below, dynamic motion, dramatic backlighting, epic scale",
    "horror":    "Dark figure at end of long corridor, single flickering light, blood red atmospheric glow, terrifying gothic architecture",
    "romance":   "Couple silhouette embracing at sunset over ocean, warm golden bokeh, soft pastel sky, intimate cinematic close-up",
    "comedy":    "Cheerful street scene with warm sunlight, vibrant saturated colours, playful joyful atmosphere, bright and inviting",
    "drama":     "Person alone by rain-streaked window, soft natural light, desaturated tones, contemplative emotional solitude",
    "animation": "Magical glowing portal in enchanted colourful forest, vibrant fantastical creatures, rich animated storybook world",
    "sci-fi":    "Lone astronaut on alien planet surface, vast starfield above, two moons rising, blue atmospheric haze, epic cosmic scale",
    "fantasy":   "Dragon soaring over medieval castle at dusk, magical golden light rays, epic fantasy landscape, clouds parting dramatically",
    "war":       "Soldiers silhouetted against burning horizon at dawn, smoky battlefield, dramatic orange-red sky, epic historical scale",
    "western":   "Lone rider on horseback at golden hour in vast canyon, long shadow, dusty warm amber light, wide cinematic composition",
    "adventure": "Explorer at cliff edge overlooking vast ancient jungle ruins, epic sweeping landscape, sense of discovery and wonder",
    "musical":   "Performer on dramatic stage, single spotlight from above, theatrical smoke, vibrant coloured lights, expressive movement",
    "biography": "Close portrait in warm vintage light, authentic period setting, film grain texture, intimate documentary realism",
    "mystery":   "Shadowy figure examining clues under lamplight, dark ornate study, candle flickering, atmospheric Victorian mystery",
}

def build_dalle3_prompt(title: str, genres: str, tags: str = "") -> str:
    """
    DALL-E 3 prompt engineering — specific, visual, cinematic.
    Key difference from DALL-E 2: DALL-E 3 understands composition
    and lighting instructions much better.
    """
    clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
    gl = [g.strip().lower() for g in genres.split(",") if g.strip()]

    style = "dramatic cinematic scene, professional film photography, rich atmospheric depth"
    for g in gl:
        if g in CINEMATIC_STYLES:
            style = CINEMATIC_STYLES[g]
            break

    # Visual tag enrichment
    visual_map = {
        "pixar": "Pixar animation style, vibrant 3D animated world, expressive characters",
        "cult film": "stylised surreal visual, bold graphic composition",
        "based on true story": "authentic realistic photography, documentary realism",
        "martial arts": "dynamic action pose, motion blur, dramatic fighting stance",
        "heist": "sleek sophisticated team, city lights, stylish composition",
        "time travel": "swirling temporal vortex, multiple eras layered, surreal",
        "dystopia": "oppressive grey cityscape, surveillance towers, dark future",
    }
    extras = ""
    if tags:
        for t in tags.split(","):
            t_lower = t.strip().lower()
            if t_lower in visual_map:
                extras = f", {visual_map[t_lower]}"
                break

    return (
        f"Cinematic movie poster artwork for '{clean}'. "
        f"{style}{extras}. "
        f"Professional Hollywood cinematography, "
        f"ultra-detailed 4K, dramatic lighting, rich deep colours, "
        f"award-winning visual composition, portrait orientation 2:3 ratio, "
        f"no text, no words, no letters, no titles anywhere in image."
    )


def generate_dalle3(title: str, genres: str = "Drama", tags: str = "",
                    doc_id: str = "", force: bool = False,
                    quality: str = "hd") -> dict:
    """
    DALL-E 3 HD — OpenAI's best latent diffusion model.

    Architecture: Enhanced CLIP text encoder → improved latent diffusion
    → better text-to-image alignment → 1024x1792 HD output.

    Cost: $0.040 per HD image (1024x1792 portrait)
          $0.020 per standard image (1024x1024)
    """
    key = os.environ.get("OPENAI_API_KEY","").strip()
    if not key.startswith("sk-"):
        return {"error": "OPENAI_API_KEY not set — run: export OPENAI_API_KEY=..."}

    cache_key = doc_id or hashlib.md5(
        f"dalle3hd{title}{genres}".encode()).hexdigest()[:8]
    out_path = CACHE / f"dalle3_{cache_key}.png"

    if out_path.exists() and not force:
        print(f"   ✅ Cached: {out_path}")
        return {"path": str(out_path), "cached": True,
                "tier": "dalle3", "title": title}

    prompt = build_dalle3_prompt(title, genres, tags)
    print(f"\n🎨 DALL-E 3 HD → {title}")
    print(f"   Genres: {genres}")
    print(f"   Prompt: {prompt[:110]}...")

    payload = json.dumps({
        "model": "dall-e-3",
        "prompt": prompt,
        "n": 1,
        "size": "1024x1792",   # portrait — perfect movie poster ratio
        "quality": quality,     # "hd" for best quality
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
        with urllib.request.urlopen(req, timeout=90) as r:
            result = json.loads(r.read())

        img_bytes = base64.b64decode(result["data"][0]["b64_json"])
        revised   = result["data"][0].get("revised_prompt", "")
        out_path.write_bytes(img_bytes)
        ms = round((time.perf_counter()-t0)*1000, 1)

        print(f"   ✅ Saved: {out_path}")
        print(f"   Size: {len(img_bytes)//1024}KB  |  Time: {ms:.0f}ms  |  Cost: $0.04")

        return {
            "path": str(out_path),
            "cached": False,
            "tier": "dalle3",
            "model": "dall-e-3",
            "architecture": (
                "CLIP text encoder → latent diffusion UNet "
                "(DDPM schedule, T=1000 steps) → VAE decoder → 1024x1792 PNG"
            ),
            "title": title,
            "genres": genres,
            "prompt": prompt,
            "revised_prompt": revised[:200] if revised else "",
            "latency_ms": ms,
            "size_kb": len(img_bytes)//1024,
            "cost": "$0.040 (DALL-E 3 HD 1024x1792)",
            "quality": quality,
        }

    except urllib.error.HTTPError as e:
        body = e.read().decode()[:300]
        return {"error": f"HTTP {e.code}: {body}"}
    except Exception as ex:
        return {"error": str(ex)}


# ═══════════════════════════════════════════════════════════════════════
# PART 3 — DEMO: Generate 5 phenomenal posters
# ═══════════════════════════════════════════════════════════════════════

DEMO_FILMS = [
    ("Pulp Fiction (1994)",         "Crime,Thriller",   "cult film,nonlinear"),
    ("Toy Story (1995)",            "Animation,Comedy", "pixar,friendship"),
    ("Inception (2010)",            "Sci-Fi,Thriller",  "mind-bending,dreams"),
    ("Schindler's List (1993)",     "Drama,War",        "historical,emotional"),
    ("Grand Budapest Hotel (2014)", "Comedy,Drama",     "quirky,stylised"),
]

def run_demo():
    print("StreamLens Diffusion Pipeline")
    print("=" * 60)
    print()
    print("PART 1 — DDPM Noise Schedule (mathematical foundation)")
    print("-" * 60)
    sched = ddpm_noise_schedule()
    for k in ["t=0   (clean)", "t=500 (50%)", "t=999 (100%)"]:
        v = sched[k]
        print(f"  {k}: ᾱ={v['alpha_cumprod']:.4f}  SNR={v['SNR']:.4f}  → {v['interpretation']}")
    print()
    print("  Inference:", sched["inference_algorithm"]["step1"])
    print("  →", sched["inference_algorithm"]["step3"])
    print("  →", sched["inference_algorithm"]["step6"])
    print()
    print("PART 2 — DALL-E 3 HD Poster Generation")
    print("-" * 60)
    print(f"  Model: DALL-E 3 (latent diffusion, DDPM-based)")
    print(f"  Resolution: 1024×1792 (HD portrait)")
    print(f"  Cost: $0.04 per image")
    print(f"  Prompt strategy: cinematic visual atmosphere, no text")
    print(f"  StreamLens UI overlays film title on top of image")
    print()

    results = []
    total_cost = 0.0

    for title, genres, tags in DEMO_FILMS:
        r = generate_dalle3(title, genres, tags)
        results.append((title, r))
        if "error" not in r:
            total_cost += 0.04
            import subprocess
            subprocess.run(["open", r["path"]], check=False)
            time.sleep(0.5)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    ok = sum(1 for _, r in results if "error" not in r)
    print(f"Generated: {ok}/{len(DEMO_FILMS)}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Cached: zero cost on subsequent runs")
    print()
    for title, r in results:
        if "error" not in r:
            print(f"  ✅ {title[:38]} → {Path(r['path']).name} ({r.get('latency_ms',0):.0f}ms)")
        else:
            print(f"  ❌ {title[:38]} → {r['error'][:60]}")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="StreamLens Diffusion Pipeline — DDPM + DALL-E 3"
    )
    ap.add_argument("--title",    default="Pulp Fiction")
    ap.add_argument("--genre",    default="Crime,Thriller")
    ap.add_argument("--tags",     default="cult film,nonlinear")
    ap.add_argument("--doc_id",   default="")
    ap.add_argument("--quality",  default="hd", choices=["standard","hd"])
    ap.add_argument("--force",    action="store_true")
    ap.add_argument("--demo",     action="store_true",
                    help="Generate 5 phenomenal demo posters")
    ap.add_argument("--schedule", action="store_true",
                    help="Show DDPM noise schedule mathematics")
    ap.add_argument("--plot",     action="store_true",
                    help="Plot DDPM noise curve (requires matplotlib)")
    args = ap.parse_args()

    if args.schedule:
        sched = ddpm_noise_schedule()
        print(json.dumps(
            {k: v for k, v in sched.items()
             if k not in {"inference_algorithm"}},
            indent=2
        ))
        print("\nInference algorithm:")
        for step, desc in sched["inference_algorithm"].items():
            print(f"  {step}: {desc}")
        if args.plot:
            plot_noise_schedule()

    elif args.demo:
        run_demo()

    else:
        export_key = os.environ.get("OPENAI_API_KEY","")
        if not export_key:
            print("❌ Set your key first:")
            print("   export OPENAI_API_KEY=$(grep '^OPENAI_API_KEY' .env | head -1 | cut -d= -f2-)")
        else:
            result = generate_dalle3(
                args.title, args.genre, args.tags,
                args.doc_id, args.force, args.quality
            )
            print(json.dumps(
                {k: v for k, v in result.items()
                 if k not in {"prompt","revised_prompt"}},
                indent=2
            ))
            if "path" in result and "error" not in result:
                import subprocess
                subprocess.run(["open", result["path"]], check=False)
