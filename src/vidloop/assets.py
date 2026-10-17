"""Generation tools scene code can call: generate_image (Qwen-Image) and animate_image (HappyHorse i2v).

Scenes call it at import time, never inside frame(), so each image is fetched once per render:

    from vidloop.assets import generate_image
    BG = generate_image("misty mountain lake at dawn, cinematic, no text", (640, 360))

    def frame(t, W, H):
        img = BG.copy()   # then draw crisp text and motion on top with Pillow

Images are cached on disk by (model, prompt, size, seed), so a revision that only changes text or
timing costs nothing. Each render may request at most MAX_IMAGES_PER_RENDER distinct images.
"""

import base64
import hashlib
import io
import json
import os
import threading
import time
from pathlib import Path

import httpx
import imageio_ffmpeg
from PIL import Image

from . import config, trace
from .files import atomic_write_bytes, atomic_write_text

MAX_IMAGES_PER_RENDER = 3
CACHE_DIR = config.ROOT / "assets_cache"

_requested: set[str] = set()
# Caps parallel Alibaba requests (parallel shots x parallel candidates) at the plan's concurrency.
_alibaba = threading.BoundedSemaphore(config.ALIBABA_CONCURRENCY)
# One lock per cache key: a request for a still or clip that is already being generated (e.g. speculatively)
# waits for that generation instead of paying for a second one.
_key_locks: dict[str, threading.Lock] = {}
_key_locks_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _key_locks_guard:
        return _key_locks.setdefault(key, threading.Lock())


def _key(prompt: str, size: tuple[int, int], seed: int) -> str:
    raw = json.dumps([config.IMAGE_MODEL, prompt, list(size), seed])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def generate_image(prompt: str, size: tuple[int, int] = (640, 360), seed: int = 0) -> Image.Image:
    """Return an RGB image of exactly `size` generated from `prompt`. Generated images cannot render
    reliable text: draw all text with Pillow on top."""
    key = _key(prompt, size, seed)
    if not (CACHE_DIR / f"{key}.png").exists():
        _requested.add(key)
        if len(_requested) > MAX_IMAGES_PER_RENDER:
            raise RuntimeError(f"a scene may generate at most {MAX_IMAGES_PER_RENDER} images per render")
    return image(prompt, size, seed)


def image(prompt: str, size: tuple[int, int] = (640, 360), seed: int = 0) -> Image.Image:
    """The cached generated image for `prompt`, generating it on first request (no per-render cap)."""
    key = _key(prompt, size, seed)
    path = CACHE_DIR / f"{key}.png"
    with _lock_for(key):
        if not path.exists():
            _generate_image(prompt, size, seed, key, path)
    return Image.open(path).convert("RGB")


def _generate_image(prompt: str, size: tuple[int, int], seed: int, key: str, path: Path) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    with _alibaba, trace.span("still", config.IMAGE_MODEL, prompt=prompt[:40], seed=seed):
        img = Image.open(io.BytesIO(_fetch(prompt, size, seed))).convert("RGB")
    out = io.BytesIO()
    img.resize(size, Image.LANCZOS).save(out, format="PNG")
    atomic_write_text(CACHE_DIR / f"{key}.json", json.dumps({"prompt": prompt, "size": size, "seed": seed}))
    atomic_write_bytes(path, out.getvalue())
    trace.record("still", config.IMAGE_MODEL, prompt, f"image {path.name}", seed=seed, size=list(size))


# Native generation sizes (width*height); the closest aspect ratio is requested, then resized.
NATIVE_SIZES = [(1664, 928), (1472, 1104), (1328, 1328), (1104, 1472), (928, 1664)]


def _post_alibaba(url: str, model: str, **kwargs) -> httpx.Response:
    """POST to Alibaba, backing off on throttling (HTTP 429/503 or a Throttling code) and recording each one."""
    for attempt in range(6):
        resp = httpx.post(url, **kwargs)
        throttled = resp.status_code in (429, 503) or "Throttling" in resp.text[:300]
        if not throttled:
            return resp
        trace.throttle("alibaba", model, resp.status_code, resp.text[:200])
        if attempt < 5:
            time.sleep(min(60, 2 ** attempt * 2))
    return resp


def _fetch(prompt: str, size: tuple[int, int], seed: int) -> bytes:
    w, h = min(NATIVE_SIZES, key=lambda s: abs(s[0] / s[1] - size[0] / size[1]))
    resp = _post_alibaba(
        f"{config.IMAGE_BASE_URL}/api/v1/services/aigc/multimodal-generation/generation", config.IMAGE_MODEL,
        headers={"Authorization": f"Bearer {os.environ['ALIBABA_API_KEY']}"},
        json={
            "model": config.IMAGE_MODEL,
            "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
            "parameters": {"size": f"{w}*{h}", "seed": seed, "watermark": False},
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"image generation failed ({resp.status_code}): {resp.text[:500]}")
    url = resp.json()["output"]["choices"][0]["message"]["content"][0]["image"]
    img = httpx.get(url, timeout=120)
    img.raise_for_status()
    return img.content


MAX_CLIPS_PER_RENDER = 1
CLIP_POLL_S = 10
CLIP_TIMEOUT_S = 480

_clips_requested: set[tuple] = set()


def animate_image(image: Image.Image, prompt: str, duration: int = 5, seed: int = 0) -> list[Image.Image]:
    """Animate `image` as the first frame of a `duration`-second clip (HappyHorse image-to-video).
    Returns duration * FPS RGB frames of size (WIDTH, HEIGHT). The first request takes 1-5 minutes."""
    png = _png(image)
    _clips_requested.add(_clip_key(png, prompt, duration, seed))
    if len(_clips_requested) > MAX_CLIPS_PER_RENDER:
        raise RuntimeError(f"a scene may animate at most {MAX_CLIPS_PER_RENDER} clip per render")
    return _decode(clip_path(image, prompt, duration, seed), duration)


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _clip_key(png: bytes, prompt: str, duration: int, seed: int) -> str:
    return hashlib.sha256(png + json.dumps([config.VIDEO_MODEL, prompt, duration, seed]).encode()).hexdigest()[:24]


def clip_path(image: Image.Image, prompt: str, duration: int = 5, seed: int = 0) -> Path:
    """The cached MP4 of `image` animated by `prompt`, generating it on first request."""
    png = _png(image)
    key = _clip_key(png, prompt, duration, seed)
    path = CACHE_DIR / f"{key}.mp4"
    with _lock_for(key):
        if not path.exists():
            CACHE_DIR.mkdir(exist_ok=True)
            with _alibaba, trace.span("clip", config.VIDEO_MODEL, secs=duration):
                clip = _fetch_clip(png, prompt, duration, seed)
            atomic_write_text(CACHE_DIR / f"{key}.json", json.dumps({"prompt": prompt, "duration": duration, "seed": seed}))
            atomic_write_bytes(path, clip)
            trace.record("clip", config.VIDEO_MODEL, f"[first frame image] {prompt}", f"clip {path.name}", duration=duration)
    return path


def _fetch_clip(png: bytes, prompt: str, duration: int, seed: int) -> bytes:
    headers = {"Authorization": f"Bearer {os.environ['ALIBABA_API_KEY']}"}
    resp = _post_alibaba(
        f"{config.IMAGE_BASE_URL}/api/v1/services/aigc/video-generation/video-synthesis", config.VIDEO_MODEL,
        headers={**headers, "X-DashScope-Async": "enable"},
        json={
            "model": config.VIDEO_MODEL,
            "input": {"prompt": prompt, "media": [
                {"type": "first_frame", "url": "data:image/png;base64," + base64.b64encode(png).decode()}]},
            "parameters": {"resolution": "480P", "duration": duration, "seed": seed, "watermark": False},
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"clip submission failed ({resp.status_code}): {resp.text[:500]}")
    task_id = resp.json()["output"]["task_id"]
    deadline = time.monotonic() + CLIP_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(CLIP_POLL_S)
        out = httpx.get(f"{config.IMAGE_BASE_URL}/api/v1/tasks/{task_id}", headers=headers, timeout=60).json()["output"]
        if out["task_status"] == "SUCCEEDED":
            video = httpx.get(out["video_url"], timeout=120)
            video.raise_for_status()
            return video.content
        if out["task_status"] in ("FAILED", "CANCELED", "UNKNOWN"):
            raise RuntimeError(f"clip task {task_id} {out['task_status']}: {out.get('message', '')}")
    raise RuntimeError(f"clip task {task_id} not finished after {CLIP_TIMEOUT_S} seconds")


def _decode(path, duration: int) -> list[Image.Image]:
    """Resample the clip to config.FPS and resize to the scene size by nearest source frame."""
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = reader.__next__()
    w, h = meta["size"]
    src = [Image.frombytes("RGB", (w, h), f).resize((config.WIDTH, config.HEIGHT), Image.LANCZOS) for f in reader]
    n = duration * config.FPS
    return [src[min(len(src) - 1, int(i * len(src) / n))] for i in range(n)]
