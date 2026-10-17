"""The coder agent: writes and revises scene code from a brief, critiques and the skill library."""

import json

from . import config, llm, skills

CONTRACT = f"""Write ONE self-contained Python file that renders a video, using only Pillow, numpy, math,
the standard library, and the helper modules listed below. The harness imports your file and calls frame().
It must define:
    DURATION = <float seconds, at most {config.MAX_DURATION}>
    def frame(t: float, W: int, H: int) -> PIL.Image.Image   # RGB, exactly (W, H) = ({config.WIDTH}, {config.HEIGHT}), {config.FPS} fps
frame() must be a pure function of t (no state carried between calls).
Reply with the full file in a single ```python block."""

IMAGE_TOOL = f"""You also have an image generation tool (Qwen-Image) for photographic or illustrated layers such as
backgrounds, textures and illustrations:
    from vidloop.assets import generate_image
    BG = generate_image("<detailed visual prompt>", ({config.WIDTH}, {config.HEIGHT}))   # -> RGB PIL image of that size
Call it ONLY at module level (never inside frame()), at most 3 times per file. Each new prompt takes about
30 seconds; identical prompts are cached, so keep a prompt unchanged unless the image itself must change.
Generated images cannot contain reliable text: never ask for text in the prompt, and draw all text, numbers
and precise shapes yourself with Pillow on top. Animate generated layers with crops, pans, zooms and fades."""


VIDEO_TOOL = f"""You also have an image-to-video tool (HappyHorse) for real camera motion and live action that 2D drawing
cannot fake, such as flying through a scene, moving water, or people and animals in motion:
    from vidloop.assets import generate_image, animate_image
    START = generate_image("<the first frame, no text>", ({config.WIDTH}, {config.HEIGHT}))
    CLIP = animate_image(START, "<camera move and action>", duration=5)   # -> list of duration*{config.FPS} RGB frames
    def frame(t, W, H):
        img = CLIP[min(len(CLIP) - 1, int(t * {config.FPS}))].copy()   # then draw text and graphics on top
Call it ONLY at module level, at most once per file, with an integer duration from 3 to 10 seconds. A new clip
takes 1-5 minutes, while an unchanged image and prompt are cached, so keep them unchanged unless the clip itself
must change. Describe only motion in the clip prompt (never text); draw all text yourself on top."""


def _system() -> str:
    tool = f"\n\n{IMAGE_TOOL}" if config.IMAGE_TOOL else ""
    tool += f"\n\n{VIDEO_TOOL}" if config.VIDEO_TOOL else ""
    return f"{skills.system_prompt()}\n\n{CONTRACT}{tool}\n\n{skills.render_context()}"


def write(brief: str) -> str:
    messages = [
        {"role": "system", "content": _system()},
        {"role": "user", "content": f"BRIEF:\n{brief}"},
    ]
    return llm.extract_python(llm.chat(config.CODER_MODEL, messages))


def revise(brief: str, code: str, feedback: dict) -> str:
    """feedback is either a judge critique or {"render_error": traceback}."""
    if "render_error" in feedback:
        ask = f"The file crashed while rendering. Fix it.\n\nTRACEBACK:\n{feedback['render_error']}"
    else:
        ask = ("A director watched the rendered video and wrote this critique. Fix every issue, "
               f"critical issues first, then major. Keep what already works.\n\nCRITIQUE:\n{json.dumps(feedback, indent=1)}")
    messages = [
        {"role": "system", "content": _system()},
        {"role": "user", "content": f"BRIEF:\n{brief}\n\nCURRENT FILE:\n```python\n{code}```\n\n{ask}"},
    ]
    return llm.extract_python(llm.chat(config.CODER_MODEL, messages))
