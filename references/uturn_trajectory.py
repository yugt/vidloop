"""Choreography of the Roadster's escape U-turn over the Golden Gate Bridge median barrier, top-down.

Hand-made reference for the agents (storyboard writer, judge, Codex reviewer). North is up. The deck has
three northbound lanes (east side, right) and three southbound lanes (west side, left) divided by a
continuous concrete median barrier that a car cannot cross on the ground. Timeline of the 5-second shot:
    0.0-1.0 s  drive north in the middle northbound lane toward the landslide
    1.0 s      lift-off while still moving (cold-gas plumes, white vapor cloud)
    1.0-3.5 s  airborne arc over the barrier, yawing flat (like a compass needle) from north to south
    2.25 s     apex, about 6 m above the barrier, nose pointing west
    3.5 s      land in the middle southbound lane, nose pointing south
    3.5-5.0 s  drive south toward San Francisco
Rendered with the vidloop scene contract (DURATION, frame(t, W, H)); also drawable at any size.
"""

import math

from PIL import Image, ImageDraw, ImageFont

DURATION = 5.0
LIFT, LAND, H_MAX = 1.0, 3.5, 6.0  # seconds, seconds, metres
KEYFRAMES = [(0.0, "drive north"), (1.0, "lift-off, still moving"), (1.6, "over the NB lanes, turning"),
             (2.25, "apex ~6 m over barrier, nose west"), (2.9, "over the SB lanes, turning"),
             (3.5, "lands, nose south"), (5.0, "drives south to SF")]

BG, DECK, LANE, BARRIER = (24, 26, 32), (58, 60, 66), (200, 200, 190), (205, 205, 200)
CAR, SHADOW, PATH, SLIDE = (245, 245, 245), (10, 10, 12), (90, 200, 255), (150, 110, 70)


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _layout(W: int, H: int) -> dict:
    """Deck on the left 55% of the frame, elevation and beat panels on the right."""
    lane = W * 0.06
    mid = W * 0.31
    return {"lane": lane, "mid": mid, "nb": mid + lane * 1.5, "sb": mid - lane * 1.5,
            "top": H * 0.06, "bottom": H * 0.96, "start_y": H * 0.90, "turn_y": H * 0.62, "apex_y": H * 0.30}


def _bezier(p0, p1, p2, p3, u):
    a = (1 - u) ** 3
    b = 3 * (1 - u) ** 2 * u
    c = 3 * (1 - u) * u ** 2
    d = u ** 3
    return (a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1])


def state(t: float, W: int, H: int) -> tuple[float, float, float, float]:
    """Ground position (x, y), heading in degrees (0 = north, 90 = east, 180 = south, 270 = west), height in metres."""
    L = _layout(W, H)
    p0, p3 = (L["nb"], L["turn_y"]), (L["sb"], L["turn_y"])
    p1, p2 = (L["nb"], L["apex_y"]), (L["sb"], L["apex_y"])
    if t <= LIFT:
        y = L["start_y"] + (L["turn_y"] - L["start_y"]) * (t / LIFT)
        return L["nb"], y, 0.0, 0.0
    if t >= LAND:
        y = L["turn_y"] + (L["start_y"] - L["turn_y"]) * ((t - LAND) / (DURATION - LAND))
        return L["sb"], y, 180.0, 0.0
    u = (t - LIFT) / (LAND - LIFT)
    x, y = _bezier(p0, p1, p2, p3, u)
    # Heading turns smoothly north -> west -> south through the arc (flat yaw, never a roll).
    heading = (360.0 - 180.0 * (0.5 - 0.5 * math.cos(math.pi * u))) % 360.0
    return x, y, heading, H_MAX * math.sin(math.pi * u)


def _car(d: ImageDraw.ImageDraw, x: float, y: float, heading: float, size: float, fill, outline=None) -> None:
    """A car as a rotated rectangle with a nose wedge; heading 0 points up (north)."""
    w, l = size * 0.5, size
    a = math.radians(heading)
    fwd, right = (math.sin(a), -math.cos(a)), (math.cos(a), math.sin(a))

    def pt(f, r):
        return (x + fwd[0] * f + right[0] * r, y + fwd[1] * f + right[1] * r)

    body = [pt(l / 2 * 0.6, -w / 2), pt(l / 2 * 0.6, w / 2), pt(-l / 2, w / 2), pt(-l / 2, -w / 2)]
    nose = [pt(l / 2 * 0.6, -w / 2), pt(l / 2, 0), pt(l / 2 * 0.6, w / 2)]
    d.polygon(body, fill=fill, outline=outline)
    d.polygon(nose, fill=(90, 200, 255) if outline is None else fill, outline=outline)


def _deck(img: Image.Image, t: float) -> None:
    W, H = img.size
    L = _layout(W, H)
    d = ImageDraw.Draw(img)
    left, right = L["mid"] - L["lane"] * 3, L["mid"] + L["lane"] * 3
    d.rectangle([left, L["top"], right, L["bottom"]], fill=DECK)
    for k in (-2, -1, 1, 2):
        x = L["mid"] + k * L["lane"]
        for y in range(int(L["top"]), int(L["bottom"]), max(6, H // 30)):
            d.line([x, y, x, y + H / 60], fill=LANE, width=max(1, W // 400))
    d.rectangle([L["mid"] - W * 0.004, L["top"], L["mid"] + W * 0.004, L["bottom"]], fill=BARRIER)
    # Landslide at the north end, spreading over time.
    spread = min(1.0, 0.35 + t / DURATION)
    d.rectangle([left - W * 0.03, 0, right + W * 0.03, L["top"] + H * 0.05 * spread], fill=SLIDE)
    f = _font(max(10, H // 36))
    d.text((L["mid"], L["top"] * 0.5), "LANDSLIDE (Marin, north)", font=f, fill=(255, 235, 210), anchor="mm")
    d.text((L["mid"], H * 0.985), "San Francisco (south)", font=f, fill=(220, 220, 230), anchor="mm")
    d.text((left - W * 0.01, H * 0.5), "SB", font=f, fill=(200, 200, 210), anchor="rm")
    d.text((right + W * 0.01, H * 0.5), "NB", font=f, fill=(200, 200, 210), anchor="lm")
    d.text((L["mid"] + W * 0.012, H * 0.80), "median\nbarrier", font=_font(max(9, H // 45)), fill=BARRIER, anchor="lm")
    # Optional trapped ordinary cars: stopped in the NB lanes near the landslide, unable to turn across the barrier.
    for k, dy in ((-1, 0.16), (1, 0.19), (0, 0.24)):
        _car(d, L["nb"] + k * L["lane"], H * dy, 0.0, W * 0.04, (110, 112, 120))


def _vapor(img: Image.Image, t: float) -> None:
    """White vapor puffs left on the ground under the car from lift-off to landing, growing and fading."""
    W, H = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    birth = LIFT - 0.1
    while birth <= min(t, LAND + 0.1):
        age = t - birth
        x, y, _, _ = state(birth, W, H)
        r = W * (0.008 + 0.03 * min(age, 1.5))
        alpha = int(170 * max(0.0, 1 - age / 1.8))
        if alpha:
            d.ellipse([x - r, y - r, x + r, y + r], fill=(235, 240, 245, alpha))
        birth += 0.08
    img.alpha_composite(layer)


def _panels(img: Image.Image, t: float) -> None:
    W, H = img.size
    d = ImageDraw.Draw(img)
    f, fs = _font(max(10, H // 34)), _font(max(9, H // 42))
    x0, x1, y0, y1 = W * 0.60, W * 0.97, H * 0.12, H * 0.42
    d.text((x0, H * 0.05), "Height above the deck", font=f, fill=(230, 230, 235))
    d.line([x0, y1, x1, y1], fill=(150, 150, 160), width=2)
    d.line([x0, y0, x0, y1], fill=(150, 150, 160), width=2)
    curve = [(x0 + (x1 - x0) * (s / DURATION), y1 - (y1 - y0) * state(s, W, H)[3] / H_MAX)
             for s in [i * DURATION / 100 for i in range(101)]]
    d.line(curve, fill=PATH, width=max(2, W // 300))
    cx = x0 + (x1 - x0) * t / DURATION
    cy = y1 - (y1 - y0) * state(t, W, H)[3] / H_MAX
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=CAR)
    d.text((x0 - 4, y0), f"{H_MAX:g} m", font=fs, fill=(170, 170, 180), anchor="ra")
    d.text((x1, y1 + 4), f"{DURATION:g} s", font=fs, fill=(170, 170, 180), anchor="ra")
    y = H * 0.50
    for k, (kt, label) in enumerate(KEYFRAMES):
        active = kt <= t and (k + 1 == len(KEYFRAMES) or t < KEYFRAMES[k + 1][0])
        d.text((x0, y), f"{k + 1}. t={kt:.2f}s  {label}", font=fs, fill=(255, 255, 255) if active else (130, 130, 140))
        y += H * 0.065


def _draw_car_at(img: Image.Image, t: float, label: str | None = None, ghost: bool = False) -> None:
    W, H = img.size
    x, y, heading, h = state(t, W, H)
    d = ImageDraw.Draw(img)
    size = W * 0.045
    d.ellipse([x - size * 0.35, y - size * 0.35, x + size * 0.35, y + size * 0.35], fill=SHADOW)  # shadow on the deck
    lift = h / H_MAX * H * 0.07  # draw the airborne car offset up-left of its shadow, and a bit larger
    cx, cy = x - lift * 0.5, y - lift
    _car(d, cx, cy, heading, size * (1 + 0.25 * h / H_MAX), (170, 170, 175) if ghost else CAR)
    if label:
        d.text((cx + size * 0.7, cy), label, font=_font(max(10, H // 34)), fill=(255, 220, 120), anchor="lm")


def frame(t: float, W: int, H: int) -> Image.Image:
    img = Image.new("RGBA", (W, H), BG + (255,))
    _deck(img, t)
    d = ImageDraw.Draw(img)
    path = [state(s, W, H)[:2] for s in [i * DURATION / 120 for i in range(121)]]
    d.line(path, fill=PATH, width=max(2, W // 320))
    _vapor(img, t)
    _draw_car_at(img, t)
    _panels(img, t)
    return img.convert("RGB")


def diagram(W: int = 1600, H: int = 900) -> Image.Image:
    """Static keyframe diagram: the full path with numbered car positions and their headings."""
    img = Image.new("RGBA", (W, H), BG + (255,))
    _deck(img, DURATION)
    d = ImageDraw.Draw(img)
    path = [state(s, W, H)[:2] for s in [i * DURATION / 200 for i in range(201)]]
    d.line(path, fill=PATH, width=max(2, W // 320))
    for k, (kt, _) in enumerate(KEYFRAMES):
        _draw_car_at(img, kt, str(k + 1), ghost=k not in (0, 3, 5))
    _panels(img, LAND)
    return img.convert("RGB")


if __name__ == "__main__":
    import sys
    diagram().save(sys.argv[1] if len(sys.argv) > 1 else "uturn_trajectory.png")
