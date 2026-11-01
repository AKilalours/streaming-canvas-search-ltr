"""Capture a real local StreamLens search response as a compact README reel.

Usage: python scripts/experiments/media/build_readme_reel.py [--api http://127.0.0.1:8000]
Requires Pillow and ffmpeg. The output is a visual replay of API results, not a
screen recording of the browser UI.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "docs/site/assets"
DATA = ROOT / "docs/site/data/search-flow.json"
QUERY = "space travel and time loops"
W, H, FPS = 1280, 720, 12


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str,
            outline: str = "#263244", radius: int = 16, width: int = 1) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def make_backdrop() -> Image.Image:
    image = Image.new("RGB", (W, H))
    pix = image.load()
    for y in range(H):
        for x in range(W):
            glow = max(0.0, 1.0 - (((x - 990) / 760) ** 2 + ((y - 10) / 560) ** 2) ** 0.5)
            pix[x, y] = (int(9 + 14 * glow), int(12 + 16 * glow), int(19 + 24 * glow))
    d = ImageDraw.Draw(image, "RGBA")
    for x in range(40, W, 48):
        d.line((x, 0, x, H), fill=(255, 255, 255, 8), width=1)
    for y in range(24, H, 48):
        d.line((0, y, W, y), fill=(255, 255, 255, 8), width=1)
    return image


def centered(draw: ImageDraw.ImageDraw, text: str, y: int, f: ImageFont.FreeTypeFont,
             color: str = "#F4F7FC") -> None:
    box = draw.textbbox((0, 0), text, font=f)
    draw.text(((W - (box[2] - box[0])) // 2, y), text, font=f, fill=color)


def result_lines(payload: dict) -> list[dict]:
    return payload.get("hits", [])[:5]


def draw_frame(backdrop: Image.Image, payload: dict, frame: int) -> Image.Image:
    image = backdrop.copy()
    d = ImageDraw.Draw(image)
    # Header / source marker
    rounded(d, (56, 35, 1224, 90), "#111722", "#293447", 14)
    d.text((82, 50), "STREAM", font=font(22, True), fill="#F4F7FC")
    d.text((168, 50), "LENS", font=font(22, True), fill="#F04452")
    d.text((900, 56), "REAL LOCAL API RESPONSE", font=font(13, True), fill="#8EE0BD")
    d.ellipse((1175, 57, 1187, 69), fill="#8EE0BD")
    # Progress steps
    steps = [("QUERY", 99), ("RETRIEVE", 445), ("RANK", 800)]
    scene = frame / FPS
    active = 0 if scene < 4 else 1 if scene < 6 else 2
    for i, (label, x) in enumerate(steps):
        col = "#F04452" if i == active else "#526075"
        d.rounded_rectangle((x, 116, x + 145, 121), radius=3, fill=col)
        d.text((x, 132), label, font=font(12, True), fill="#D9E1EC" if i == active else "#78869B")
    if scene < 2:
        centered(d, "Search that finds its way.", 242, font(47, True))
        centered(d, "One natural-language query → a ranked streaming slate.", 313, font(21), "#AAB6C8")
        rounded(d, (350, 402, 930, 466), "#151D2A", "#34445B", 13)
        d.text((380, 420), "BM25", font=font(16, True), fill="#F7C866")
        d.text((482, 420), "+", font=font(16, True), fill="#8391A6")
        d.text((523, 420), "DENSE e5", font=font(16, True), fill="#78B8FF")
        d.text((690, 420), "→", font=font(16, True), fill="#8391A6")
        d.text((743, 420), "LAMBDA RANK", font=font(16, True), fill="#8EE0BD")
    elif scene < 4:
        d.text((82, 217), "Describe the mood. We'll find the films.", font=font(31, True), fill="#F4F7FC")
        rounded(d, (82, 285, 1198, 368), "#111722", "#3B4B64", 15)
        shown = min(len(QUERY), max(0, int((scene - 2) / 1.35 * len(QUERY))))
        d.text((110, 307), QUERY[:shown] + ("▌" if shown < len(QUERY) else ""), font=font(27), fill="#F5F7FB")
        d.text((84, 410), "Natural language", font=font(14, True), fill="#8EE0BD")
        d.text((84, 442), "GET /search", font=font(14), fill="#8190A5")
    elif scene < 6:
        d.text((82, 212), "Two retrieval signals.", font=font(34, True), fill="#F4F7FC")
        d.text((82, 260), "The API merges lexical matches with semantic candidates.", font=font(18), fill="#AAB6C8")
        for x, title, desc, color in [
            (82, "BM25", "Exact titles · genres · tags", "#F7C866"),
            (660, "Dense e5", "Meaning · themes · intent", "#78B8FF"),
        ]:
            rounded(d, (x, 337, x + 538, 476), "#131B27", "#2C3A4E", 15)
            d.text((x + 26, 363), title, font=font(24, True), fill=color)
            d.text((x + 26, 411), desc, font=font(17), fill="#AAB6C8")
        d.text((585, 387), "+", font=font(33, True), fill="#8EE0BD")
    else:
        d.text((82, 204), "Ranked for your search", font=font(34, True), fill="#F4F7FC")
        d.text((84, 251), f'“{payload.get("query", QUERY)}”  ·  {payload.get("method", "hybrid").upper()} retrieval',
               font=font(17), fill="#AAB6C8")
        hits = result_lines(payload)
        reveal = min(len(hits), max(1, int((scene - 6) / 0.65) + 1))
        for i, hit in enumerate(hits[:reveal]):
            y = 308 + i * 70
            rounded(d, (82, y, 1198, y + 56), "#121A26", "#263447", 12)
            d.text((104, y + 14), f"{i+1:02d}", font=font(17, True), fill="#F04452")
            title = hit.get("title", "Untitled")
            d.text((160, y + 14), title[:58], font=font(17, True), fill="#F4F7FC")
            score = float(hit.get("score", 0))
            d.text((1027, y + 16), f"{score:.3f}", font=font(16, True), fill="#8EE0BD")
            d.text((1112, y + 18), "score", font=font(12), fill="#8190A5")
        if scene >= 9.5 and hits:
            rounded(d, (82, 674, 1198, 690), "#263446", "#263446", 6)
            d.rounded_rectangle((82, 674, 82 + int(1116 * min(1, (scene - 9.5) / 2.4)), 690), radius=6, fill="#F04452")
    # Timestamp and evidence note
    d.text((82, 636), "MovieLens catalog · captured from the running local API · sample response in docs/site/data", font=font(12), fill="#748197")
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--query", default=QUERY)
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode({"q": args.query, "k": 5})
    with urllib.request.urlopen(f"{args.api.rstrip('/')}/search?{query}", timeout=30) as response:
        payload = json.load(response)
    payload["source"] = "GET /search from a running local StreamLens API"
    payload["captured_at_utc"] = datetime.now(timezone.utc).isoformat()
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(payload, indent=2) + "\n")
    backdrop = make_backdrop()
    frames = [draw_frame(backdrop, payload, i) for i in range(FPS * 12)]
    with tempfile.TemporaryDirectory(prefix="streamlens-reel-") as temp:
        frame_dir = Path(temp)
        for i, frame in enumerate(frames):
            frame.save(frame_dir / f"frame-{i:04d}.png", optimize=False)
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
            "-i", str(frame_dir / "frame-%04d.png"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-crf", "25", "-movflags", "+faststart",
            str(ASSETS / "search-flow.mp4"),
        ], check=True)
    gif_frames = frames[::2]
    gif_frames = [im.resize((960, 540), Image.Resampling.LANCZOS).convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for im in gif_frames]
    gif_frames[0].save(ASSETS / "search-flow.gif", save_all=True, append_images=gif_frames[1:],
                       duration=1000 / (FPS / 2), loop=0, optimize=True, disposal=2)
    print(f"Saved API replay: {len(payload.get('hits', []))} hits; mp4/gif in {ASSETS}")


if __name__ == "__main__":
    main()
