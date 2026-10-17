"""Hand-written example scene for smoke-testing the render path (no API calls)."""

from PIL import Image, ImageDraw, ImageFont

from easing import ease_out_back, progress

DURATION = 3.0


def frame(t: float, W: int, H: int) -> Image.Image:
    img = Image.new("RGB", (W, H), (20, 24, 40))
    d = ImageDraw.Draw(img)
    s = ease_out_back(progress(t, 0.2, 1.2))
    font = ImageFont.load_default(size=max(1, int(64 * s)))
    d.text((W / 2, H / 2), "vidloop", font=font, fill=(240, 240, 250), anchor="mm")
    return img
