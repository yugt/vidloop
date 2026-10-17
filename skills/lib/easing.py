"""Easing curves for animation timing. All take and return values in [0, 1]."""

import math

from PIL import Image, ImageDraw


def clamp01(x: float) -> float:
    """Clamp x into [0, 1]."""
    return max(0.0, min(1.0, x))


def progress(t: float, start: float, end: float) -> float:
    """Linear 0->1 progress of t through the window [start, end], clamped."""
    return clamp01((t - start) / (end - start))


def ease_in_out_cubic(x: float) -> float:
    """Slow start, fast middle, slow end."""
    return 4 * x**3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_back(x: float) -> float:
    """Overshoots slightly then settles; good for things popping in."""
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def demo(t: float, W: int, H: int) -> Image.Image:
    img = Image.new("RGB", (W, H), (18, 18, 28))
    d = ImageDraw.Draw(img)
    x = 40 + (W - 120) * ease_in_out_cubic(progress(t, 0.0, 1.0))
    d.ellipse([x, H / 2 - 20, x + 40, H / 2 + 20], fill=(90, 200, 250))
    return img
