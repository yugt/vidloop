import io
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image

from vidloop.animate import starts_from_still
from vidloop.render import render

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "hello_scene.py"


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_clip_starting_from_its_still_passes_and_a_different_still_fails(tmp_path):
    clip = tmp_path / "clip.mp4"
    assert render(EXAMPLE, clip).ok
    first = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(clip), "-frames:v", "1",
                            "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    assert starts_from_still(clip, first)[1]
    assert not starts_from_still(clip, _png(Image.new("RGB", (640, 360), (230, 230, 230))))[1]
