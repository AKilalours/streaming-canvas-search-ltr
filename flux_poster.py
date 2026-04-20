"""
StreamLens — Flux.1 Poster Generator (fixed API endpoint)
Flux.1-schnell: 4 steps, ~10 seconds, phenomenal quality, free.
Usage: python flux_poster.py --title "Pulp Fiction" --genre "Crime,Thriller"
"""
import argparse, hashlib, json, os, time, urllib.request, urllib.error, re
from pathlib import Path

CACHE = Path("artifacts/posters_hq")
CACHE.mkdir(parents=True, exist_ok=True)

STYLES = {
    "crime":     "neo-noir, rain-slicked streets, dramatic shadows, neon reflections",
    "thriller":  "dark tense atmosphere, high contrast, psychological intensity",
    "action":    "explosive dynamic composition, motion blur, cinematic scale",
    "horror":    "dark fog, blood red accents, ominous gothic atmosphere",
    "romance":   "golden hour warmth, soft bokeh, intimate warm composition",
    "comedy":    "vibrant saturated colours, playful warm lighting",
    "drama":     "desaturated realism, shallow depth of field, emotional lighting",
    "animation": "bold vibrant illustration, clean graphic style, expressive",
    "sci-fi":    "neon blue cyberpunk, futuristic skyline, holographic elements",
    "fantasy":   "magical golden light, epic scale, ethereal mystical fog",
    "war":       "desaturated smoke and ash, dramatic epic sky",
    "western":   "dusty golden hour, wide open landscape, warm tones",
    "adventure": "sweeping landscape, heroic composition, sense of wonder",
}

def prompt_for(title, genres, tags=""):
    clean = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
    gl = [g.strip().lower() for g in genres.split(",") if g.strip()]
    style = next((STYLES[g] for g in gl if g in STYLES),
                 "cinematic atmosphere, dramatic professional lighting")
    tag_str = ""
    if tags:
        good = [t.strip() for t in tags.split(",")
                if len(t.strip()) > 3 and t.strip().lower()
                not in {"film","movie","good","great","classic"}][:2]
        if good: tag_str = ", " + ", ".join(good)
    return (
        f"Professional cinematic movie poster for '{clean}'{tag_str}. "
        f"{style}. Award-winning composition, 4K ultra detailed, "
        f"bold visual storytelling, Hollywood production quality, portrait orientation."
    )

def generate(title, genres="Drama", tags="", doc_id="", force=False):
    token = os.environ.get("HF_TOKEN","").strip()
    if not token:
        return {"error": "Set HF_TOKEN env var — free at huggingface.co/settings/tokens"}

    key = doc_id or hashlib.md5(f"flux{title}{genres}".encode()).hexdigest()[:8]
    out = CACHE / f"flux_{key}.png"
    if out.exists() and not force:
        print(f"   ✅ Cached: {out}")
        return {"path": str(out), "cached": True, "title": title}

    prompt = prompt_for(title, genres, tags)
    print(f"\n🎨 Flux.1 → {title}")
    print(f"   {prompt[:90]}...")

    # Try multiple endpoints — HF changes these occasionally
    endpoints = [
        ("https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell/v1/images/generations",
         json.dumps({"prompt": prompt, "num_inference_steps": 4,
                     "width": 512, "height": 768, "response_format": "b64_json"}).encode()),
        ("https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
         json.dumps({"inputs": prompt,
                     "parameters": {"width": 512, "height": 768,
                                    "num_inference_steps": 4}}).encode()),
        ("https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
         json.dumps({"inputs": prompt}).encode()),
    ]

    for url, payload in endpoints:
        model_name = url.split("/models/")[-1].split("/")[0] if "/models/" in url else url.split("/")[-1]
        print(f"   Trying: {model_name}...")
        req = urllib.request.Request(url, data=payload, headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                raw = r.read()
                ct = r.headers.get("Content-Type","")

            # Parse response — either raw PNG or JSON with b64
            if "image" in ct or raw[:4] == b'\x89PNG' or raw[:3] == b'\xff\xd8\xff':
                img_bytes = raw
            else:
                data = json.loads(raw)
                if isinstance(data, list) and "generated_image" in data[0]:
                    import base64
                    img_bytes = base64.b64decode(data[0]["generated_image"])
                elif isinstance(data, dict) and "data" in data:
                    import base64
                    img_bytes = base64.b64decode(data["data"][0].get("b64_json",""))
                elif isinstance(data, list) and data[0].get("image"):
                    import base64
                    img_bytes = base64.b64decode(data[0]["image"])
                else:
                    # Try b64_json at top level
                    import base64
                    b64 = (data.get("b64_json") or
                           (data.get("images") or [""])[0] if isinstance(data, dict) else "")
                    if b64:
                        img_bytes = base64.b64decode(b64)
                    else:
                        print(f"   Unknown response format: {str(data)[:100]}")
                        continue

            if len(img_bytes) < 1000:
                print(f"   Response too small ({len(img_bytes)} bytes) — skipping")
                continue

            out.write_bytes(img_bytes)
            ms = round((time.perf_counter()-t0)*1000, 1)
            print(f"   ✅ Saved: {out} ({ms}ms, {len(img_bytes)//1024}KB)")
            return {
                "path": str(out), "cached": False,
                "model": model_name, "title": title,
                "genres": genres, "latency_ms": ms,
                "cost": "$0.00 (HuggingFace free)",
                "prompt": prompt,
            }

        except urllib.error.HTTPError as e:
            body = e.read().decode()[:150]
            print(f"   HTTP {e.code}: {body}")
            if e.code == 503:
                print("   Model loading — waiting 15s...")
                time.sleep(15)
        except Exception as ex:
            print(f"   Error: {ex}")

    return {"error": "All endpoints failed — check HF_TOKEN and try again"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--title",  default="Pulp Fiction")
    ap.add_argument("--genre",  default="Crime,Thriller")
    ap.add_argument("--tags",   default="cult film,nonlinear")
    ap.add_argument("--doc_id", default="")
    ap.add_argument("--force",  action="store_true")
    args = ap.parse_args()

    result = generate(args.title, args.genre, args.tags, args.doc_id, args.force)
    print(json.dumps({k:v for k,v in result.items() if k!="prompt"}, indent=2))
    if "path" in result and "error" not in result:
        import subprocess
        subprocess.run(["open", result["path"]], check=False)
