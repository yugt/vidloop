"""Edit decision lists: deterministic ffmpeg fixes for shots, decided by the judge loop.

Many film failures are editorial, not generative: a shot morphs for half a second (trim it), ends
abruptly (hold the last frame), faces the wrong way for the story (mirror it), is graded differently
from its neighbours (grade it), or joins with a fade where the action should be continuous (cut).
Regenerating costs minutes and can break what worked; an edit costs a second and touches nothing else.

An EDL is a list of operations (OPS_HELP) applied in order to a shot's best clip, producing
shot_N/cut.mp4. The join with the previous shot is not an op: it is the shot's "join" field ("fade" or "cut").
"""

import subprocess
from pathlib import Path

import imageio_ffmpeg

from . import config, trace

OPS_HELP = """
    {"op": "trim", "start": 0.4, "end": 3.0}                 keep [start, end) seconds of the shot
    {"op": "speed", "factor": 1.25}                           >1 faster, <1 slower (0.25-4)
    {"op": "reverse"}                                         play backwards
    {"op": "hold_start", "seconds": 0.5}                      freeze the first frame
    {"op": "hold_end", "seconds": 0.5}                        freeze the last frame
    {"op": "zoom", "x": 0.1, "y": 0.0, "w": 0.8, "h": 0.8}    crop to this window (fractions), scaled back up
    {"op": "mirror"}                                          flip left-right (reverses screen direction)
    {"op": "grade", "brightness": 0.0, "contrast": 1.0, "saturation": 1.0}"""

OPS = ("trim", "speed", "reverse", "hold_start", "hold_end", "zoom", "mirror", "grade")

EDL_SCHEMA = {"type": "array", "items": {"type": "object", "required": ["op"], "properties": {
    "op": {"type": "string", "enum": list(OPS)},
    "start": {"type": "number"}, "end": {"type": "number"}, "factor": {"type": "number"},
    "seconds": {"type": "number"}, "x": {"type": "number"}, "y": {"type": "number"},
    "w": {"type": "number"}, "h": {"type": "number"}, "brightness": {"type": "number"},
    "contrast": {"type": "number"}, "saturation": {"type": "number"}}}}


def _filter(op: dict, size: tuple[int, int]) -> str:
    name = op["op"]
    if name == "trim":
        return f"trim=start={op['start']}:end={op['end']},setpts=PTS-STARTPTS"
    if name == "speed":
        if not 0.25 <= op["factor"] <= 4:
            raise ValueError(f"speed factor {op['factor']} outside 0.25-4")
        return f"setpts=PTS/{op['factor']}"
    if name == "reverse":
        return "reverse"
    # tpad silently does nothing after trim unless its input has a constant frame rate.
    if name == "hold_start":
        return f"fps={config.FPS},tpad=start_mode=clone:start_duration={op['seconds']}"
    if name == "hold_end":
        return f"fps={config.FPS},tpad=stop_mode=clone:stop_duration={op['seconds']}"
    if name == "zoom":
        x, y, w, h = op["x"], op["y"], op["w"], op["h"]
        if not (0 <= x and 0 <= y and 0 < w and 0 < h and x + w <= 1 and y + h <= 1):
            raise ValueError(f"zoom window {op} is outside the frame")
        return f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y},scale={size[0]}:{size[1]}"
    if name == "mirror":
        return "hflip"
    if name == "grade":
        return (f"eq=brightness={op.get('brightness', 0.0)}:contrast={op.get('contrast', 1.0)}"
                f":saturation={op.get('saturation', 1.0)}")
    raise ValueError(f"unknown EDL op {name!r}; known: {', '.join(OPS)}")


def apply(src: Path, dst: Path, ops: list[dict], size: tuple[int, int]) -> Path:
    """Render `src` through the EDL into `dst` (re-encoded at the film size and fps)."""
    chain = [_filter(op, size) for op in ops] + [f"scale={size[0]}:{size[1]},fps={config.FPS},format=yuv420p,setsar=1"]
    with trace.span("edit", "ffmpeg (EDL)", ops=len(ops)):
        _run(src, dst, chain)
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(dst))
    if frames < 1:
        raise ValueError(f"EDL {ops} left no frames in {src}")
    return dst


def _run(src: Path, dst: Path, chain: list[str]) -> None:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
                    "-vf", ",".join(chain), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(dst)],
                   check=True)
