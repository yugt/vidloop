"""Render agent-written scene code in an isolated subprocess."""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class RenderResult:
    ok: bool
    video: Path | None
    error: str  # traceback tail when ok is False


def render(scene_file: Path, out_mp4: Path, duration: float | None = None,
           lib_dirs: list[Path] | None = None) -> RenderResult:
    # Scenes import agent-written helpers from the skill library by module name.
    paths = [str(d) for d in (lib_dirs or [config.SKILL_LIB_DIR])]
    # Scene code never calls Crusoe, so it does not inherit that key (the image key is needed for assets).
    env = {k: v for k, v in os.environ.items() if k != "CRUSOE_API_KEY"}
    env["PYTHONPATH"] = os.pathsep.join([*paths, os.environ.get("PYTHONPATH", "")])
    cmd = [sys.executable, "-m", "vidloop.runner", str(scene_file), str(out_mp4)]
    if duration is not None:
        cmd.append(str(duration))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=config.RENDER_TIMEOUT_S,
            cwd=scene_file.parent, env=env,
        )
    except subprocess.TimeoutExpired:
        return RenderResult(False, None, f"render timed out after {config.RENDER_TIMEOUT_S} seconds")
    if proc.returncode != 0:
        error = _python_error(proc.stderr).strip() or f"renderer exited with status {proc.returncode} and no output"
        return RenderResult(False, None, error[-4000:])
    return RenderResult(True, out_mp4, "")


def _python_error(stderr: str) -> str:
    """The last Python traceback in stderr, without the ffmpeg shutdown lines printed after it."""
    lines = stderr.splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("Traceback (most recent call last)")]
    if not starts:
        return stderr
    for end in range(starts[-1] + 1, len(lines)):
        if lines[end] and not lines[end].startswith(" "):  # the unindented "SomeError: message" line
            return "\n".join(lines[starts[-1] : end + 1])
    return "\n".join(lines[starts[-1] :])
