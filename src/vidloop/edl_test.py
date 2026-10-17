from pathlib import Path

import imageio_ffmpeg
import pytest

from vidloop import edl
from vidloop.render import render

SIZE = (640, 360)
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "hello_scene.py"


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("edl") / "src.mp4"
    assert render(EXAMPLE, out).ok  # 3.0 seconds at 24 fps
    return out


def _secs(p: Path) -> float:
    return imageio_ffmpeg.count_frames_and_secs(str(p))[1]


def test_trim_hold_and_speed_change_duration_as_specified(clip, tmp_path):
    out = edl.apply(clip, tmp_path / "a.mp4", [{"op": "trim", "start": 0.5, "end": 2.5},
                                                {"op": "hold_end", "seconds": 1.0},
                                                {"op": "speed", "factor": 2.0}], SIZE)
    assert _secs(out) == pytest.approx(1.5, abs=0.1)  # (2.0 + 1.0) / 2


def test_mirror_zoom_grade_reverse_keep_duration(clip, tmp_path):
    out = edl.apply(clip, tmp_path / "b.mp4", [{"op": "mirror"}, {"op": "reverse"},
                                                {"op": "zoom", "x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8},
                                                {"op": "grade", "brightness": 0.05, "saturation": 1.2}], SIZE)
    assert _secs(out) == pytest.approx(3.0, abs=0.1)


def test_zoom_outside_frame_is_rejected(clip, tmp_path):
    with pytest.raises(ValueError, match="outside the frame"):
        edl.apply(clip, tmp_path / "c.mp4", [{"op": "zoom", "x": 0.5, "y": 0, "w": 0.8, "h": 0.5}], SIZE)


def test_trim_past_the_end_leaves_no_frames_and_is_rejected(clip, tmp_path):
    with pytest.raises(Exception):
        edl.apply(clip, tmp_path / "d.mp4", [{"op": "trim", "start": 5.0, "end": 6.0}], SIZE)


def test_unknown_op_is_rejected(clip, tmp_path):
    with pytest.raises(ValueError, match="unknown EDL op"):
        edl.apply(clip, tmp_path / "e.mp4", [{"op": "explode"}], SIZE)
