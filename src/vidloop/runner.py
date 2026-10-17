"""Subprocess entry point: import a scene file and encode it to MP4.

Scene contract (what the coder agent must write):
    DURATION: float                      # seconds, <= MAX_DURATION
    def frame(t: float, W: int, H: int) -> PIL.Image.Image   # RGB, size (W, H)

Usage: python -m vidloop.runner <scene.py> <out.mp4> [duration_override]
"""

import importlib.util
import os
import sys
import traceback

import imageio_ffmpeg
import numpy as np

from . import config


def main() -> None:
    scene_path, out_path = sys.argv[1], sys.argv[2]
    spec = importlib.util.spec_from_file_location("scene", scene_path)
    scene = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scene)

    duration = float(sys.argv[3]) if len(sys.argv) > 3 else float(scene.DURATION)
    if not 0 < duration <= config.MAX_DURATION:
        raise ValueError(f"DURATION must be in (0, {config.MAX_DURATION}], got {duration}")

    W, H, fps = config.WIDTH, config.HEIGHT, config.FPS
    n_frames = int(round(duration * fps))
    if n_frames < 1:
        raise ValueError(f"DURATION {duration} gives no frames at {fps} fps")
    writer = imageio_ffmpeg.write_frames(
        out_path, (W, H), fps=fps, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1,
        output_params=["-crf", "28", "-preset", "veryfast"],
    )
    writer.send(None)
    try:
        for i in range(n_frames):
            img = scene.frame(i / fps, W, H)
            if img.size != (W, H):
                raise ValueError(f"frame({i / fps:.2f}) returned size {img.size}, expected {(W, H)}")
            writer.send(np.ascontiguousarray(np.asarray(img.convert("RGB"))))
    except Exception:
        # Exit before the writer's cleanup so ffmpeg's messages don't bury the scene's traceback.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    writer.close()


if __name__ == "__main__":
    main()
