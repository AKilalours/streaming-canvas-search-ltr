#!/usr/bin/env python3
"""Build a short visual tour from the real StreamLens UI screenshots in docs/site/assets."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "docs/site/assets"
SIZE = (1280, 720)
FPS = 20
HOLD_SECONDS = 3.6
FADE_SECONDS = 0.5


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_frame(source: Path, step: str, caption: str) -> Image.Image:
    with Image.open(source) as original:
        screenshot = original.convert("RGB")

    backdrop = screenshot.resize(SIZE, Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(32))
    backdrop = Image.blend(backdrop, Image.new("RGB", SIZE, (5, 7, 11)), 0.68)
    frame = backdrop.convert("RGBA")
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rectangle((0, 0, SIZE[0], 78), fill=(7, 9, 13, 230))
    draw.text((48, 27), "STREAMLENS  /  PRODUCT TOUR", font=font(18), fill=(235, 239, 247, 255))

    max_width, max_height = SIZE[0] - 120, SIZE[1] - 170
    scale = min(max_width / screenshot.width, max_height / screenshot.height)
    display_size = (round(screenshot.width * scale), round(screenshot.height * scale))
    screenshot = screenshot.resize(display_size, Image.Resampling.LANCZOS)
    x = (SIZE[0] - display_size[0]) // 2
    y = 88 + (max_height - display_size[1]) // 2
    shadow = Image.new("RGBA", (display_size[0] + 24, display_size[1] + 24), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((8, 8, display_size[0] + 16, display_size[1] + 16), 14, fill=(0, 0, 0, 160))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    frame.alpha_composite(shadow, (x - 12, y - 12))
    frame.alpha_composite(screenshot.convert("RGBA"), (x, y))

    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rectangle((0, SIZE[1] - 62, SIZE[0], SIZE[1]), fill=(7, 9, 13, 235))
    draw.text((48, SIZE[1] - 48), step, font=font(15), fill=(142, 224, 189, 255))
    draw.text((260, SIZE[1] - 48), caption, font=font(16), fill=(235, 239, 247, 255))
    return frame.convert("RGB")


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required to build the README video and GIF")

    scenes = [
        make_frame(ASSETS / "ui-profile.webp", "01  ·  CHOOSE A PROFILE", "The feed uses the selected demo profile."),
        make_frame(ASSETS / "ui-home.webp", "02  ·  DISCOVER", "Browse personalized rows, search, open details, and save titles."),
    ]
    hold = round(HOLD_SECONDS * FPS)
    transition = round(FADE_SECONDS * FPS)
    frames: list[Image.Image] = [scenes[0]] * hold
    for i in range(transition):
        frames.append(Image.blend(scenes[0], scenes[1], (i + 1) / (transition + 1)))
    frames.extend([scenes[1]] * hold)

    with tempfile.TemporaryDirectory(prefix="streamlens-ui-tour-") as temp_dir:
        temp = Path(temp_dir)
        for index, frame in enumerate(frames):
            frame.save(temp / f"frame-{index:04d}.png", optimize=True)
        pattern = str(temp / "frame-%04d.png")
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pattern,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(ASSETS / "ui-tour.mp4")],
            check=True,
        )
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(ASSETS / "ui-tour.mp4"),
             "-filter_complex", "[0:v]fps=6,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer",
             str(ASSETS / "ui-tour.gif")],
            check=True,
        )

    print(f"Created {ASSETS / 'ui-tour.mp4'} and {ASSETS / 'ui-tour.gif'}")


if __name__ == "__main__":
    main()
