"""Image-to-video director: turn a still into a short clip that is checked to be meaningful.

1. Nemotron Omni looks at the still and writes a camera/motion prompt that suits it, plus the
   specific things that should visibly happen in the clip.
2. HappyHorse animates the still with that prompt.
3. Omni watches the clip and votes on those expectations plus fixed realism checks.
4. If it fails, the director rewrites the prompt from the failed checks and tries again.
"""

import base64
import io
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from . import assets, config, llm, notebook, panel
from .judge import Unjudgeable, answer_checks
from .loop import log

REALISM_CHECKS = [
    "Solid structures (buildings, bridges, towers, furniture) keep their shape: no warping, melting, bending or flickering geometry.",
    "Objects keep a consistent size, color and identity throughout; nothing morphs into something else.",
    "The camera motion is smooth and continuous, with no jumps, cuts or sudden changes of direction.",
    "The motion is physically plausible: nothing passes through solid objects and nothing appears or vanishes from nowhere.",
    "There is clearly visible, meaningful motion (not a nearly static image).",
]

PLAN_FORMAT = {"type": "json_schema", "json_schema": {"name": "plan", "schema": {
    "type": "object", "required": ["subject", "motion_prompt", "expectations"], "properties": {
        "subject": {"type": "string"},
        "motion_prompt": {"type": "string"},
        "expectations": {"type": "array", "items": {"type": "string"}}}}}}

DIRECT = """You are a cinematographer planning a {duration}-second image-to-video shot. The image is the FIRST frame.
Write a motion prompt for an image-to-video model: one clear camera move (dolly, push-in, orbit, pan, crane,
fly-through) plus any natural motion in the scene (water, fog, clouds, light). Keep it achievable in
{duration} seconds and consistent with what is actually in the image; never ask for text.
Also list 3-5 specific yes/no expectations a viewer could verify in the finished clip (phrase each so YES = good),
for example "The camera moves closer to the south tower" or "Fog drifts across the water".
{feedback}Reply as JSON: subject (what the image shows), motion_prompt, expectations."""


START_MATCH_MAX = 0.05  # mean abs grayscale difference; matching first frames measured 0.006, mismatches 0.26+


def starts_from_still(clip: Path, png: bytes) -> tuple[str, bool, str]:
    """Measured check: the clip's first frame is the still. Asking the judge to compare a still with the video
    made it reason past its token cap (2 of 3 samples), so this is measured instead."""
    frame = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(clip), "-frames:v", "1",
                            "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    a = np.asarray(Image.open(io.BytesIO(frame)).convert("L").resize((64, 36)), float)
    b = np.asarray(Image.open(io.BytesIO(png)).convert("L").resize((64, 36)), float)
    diff = float(np.mean(np.abs(a - b)) / 255)
    return ("The clip starts from the same scene as the still image.", diff <= START_MATCH_MAX,
            f"first frame differs from the still by {diff:.3f} (limit {START_MATCH_MAX})")


def _plan(png: bytes, duration: int, feedback: str) -> dict:
    messages = [{"role": "user", "content": [
        {"type": "text", "text": notebook.text() + DIRECT.format(duration=duration, feedback=feedback)},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
    ]}]
    for attempt in range(3):
        try:
            return json.loads(llm.chat(config.JUDGE_MODEL, messages, max_tokens=12000, temperature=0.7,
                                       extra_body={"reasoning_effort": "low"}, response_format=PLAN_FORMAT))
        except llm.TruncatedError:
            log(f"    director plan cut off, attempt {attempt + 1}/3")
    raise RuntimeError("director could not produce a plan")


def animate_still(still: Path, out_dir: Path, duration: int = 4, attempts: int = 2) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(still, out_dir / "still.png")
    summary = make_shot(Image.open(still).convert("RGB"), out_dir, duration, attempts=attempts)
    summary["still"] = str(still)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def make_shot(image: Image.Image, out_dir: Path, duration: int, plan: dict | None = None,
              attempts: int = 2, intent: str = "") -> dict:
    """Animate `image` into a checked clip. Each attempt generates config.CLIP_CANDIDATES clips in parallel
    (different seeds); a judge panel reviews them all at once (Omni checks, Super-VL critique, Codex pick).
    If none is acceptable, the director re-plans from the panel's combined, detailed feedback.
    `plan` (subject, motion_prompt, expectations, must) is used for the first attempt if given;
    `intent` is what the shot must achieve (kept across re-plans)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / "still.png")
    png = (out_dir / "still.png").read_bytes()
    must = f"The shot must achieve: {intent}\n" if intent else ""
    feedback, tries = must, []
    must_pass = (plan or {}).get("must", [])  # hard facts kept across re-planned attempts
    k_n = config.CLIP_CANDIDATES
    for k in range(1, attempts + 1):
        if plan is None or k > 1:
            plan = _plan(png, duration, feedback)
        log(f"  [{out_dir.name}] attempt {k}: {k_n} clips of: {plan['motion_prompt'][:100]}")
        att_dir = out_dir / f"attempt_{k}"
        att_dir.mkdir(exist_ok=True)
        checks = plan["expectations"] + must_pass + REALISM_CHECKS
        context = (f"This is a {duration}-second clip generated from a still image of: {plan['subject']}. "
                   f"Intended motion: {plan['motion_prompt']}")

        def generate(j: int) -> Path:
            clip = att_dir / f"clip_{j}.mp4"
            shutil.copy(assets.clip_path(image, plan["motion_prompt"], duration, seed=(k - 1) * k_n + j), clip)
            panel.sheet(clip)
            return clip

        def omni(clip: Path) -> dict:
            try:
                return answer_checks(checks, context, clip, measured=[starts_from_still(clip, png)])
            except Unjudgeable as e:
                return _unjudgeable(e)

        with ThreadPoolExecutor(3 * k_n + 1) as pool:
            # Streaming: each clip goes to Omni and Super-VL the moment HappyHorse returns it.
            generating = {pool.submit(generate, j): j for j in range(k_n)}
            judging, criticising = {}, {}
            for done in as_completed(generating):
                j, clip = generating[done], done.result()
                judging[j] = pool.submit(omni, clip)
                criticising[j] = pool.submit(panel.critique, clip, intent, plan["motion_prompt"], must_pass)
            # Overlap: Codex compares all contact sheets while Omni and Super-VL are still working.
            sheets = [att_dir / f"clip_{j}.png" for j in range(k_n)]
            codex = (pool.submit(panel.pick, sheets, None, None, None, intent, plan["motion_prompt"], must_pass)
                     if config.CODEX_REVIEW else None)
            verdicts = [judging[j].result() for j in range(k_n)]
            critiques = [criticising[j].result() for j in range(k_n)]
            codex_choice = codex.result() if codex else None
        broken = [[c for c in must_pass if any(i["problem"] == f"FAILED CHECK: {c}" for i in v["issues"])] for v in verdicts]
        ok = [v["overall"] >= config.PASS_SCORE and not b for v, b in zip(verdicts, broken)]
        choice = codex_choice or panel.pick(sheets, verdicts, critiques, ok, intent, plan["motion_prompt"], must_pass)
        chosen = choice["best"] if choice["best"] >= 0 else max(range(k_n), key=lambda j: verdicts[j]["overall"])
        shutil.copy(att_dir / f"clip_{chosen}.mp4", att_dir / "clip.mp4")
        (att_dir / "result.json").write_text(json.dumps({"plan": plan, "verdict": verdicts[chosen], "candidate": chosen,
                                                         "panel": {"verdicts": verdicts, "critiques": critiques,
                                                                   "pick": choice}}, indent=1))
        accepted = choice["best"] >= 0
        log(f"  [{out_dir.name}] attempt {k}: Omni scores {[v['overall'] for v in verdicts]}, "
            + (f"{choice['by']} picked clip {chosen}" if accepted else f"none accepted ({'; '.join(choice['problems'])[:120]})"))
        tries.append({"attempt": k, "score": verdicts[chosen]["overall"] if accepted else 0.0, "plan": plan,
                      "verdict": verdicts[chosen]})
        if accepted:
            break
        feedback = (f"{must}A previous attempt with the motion prompt \"{plan['motion_prompt']}\" produced "
                    f"{k_n} clips, none acceptable. The judge panel found:\n{panel.findings(verdicts, critiques)}\n"
                    f"Codex's combined diagnosis: {'; '.join(choice['problems'])}\nCodex's advice: {choice['advice']}\n"
                    "Write a new, much more specific motion prompt: the exact camera position and that it stays fixed, "
                    "the subject's path, any rotation's axis, degrees and start/end orientation, timing in seconds, "
                    "and what must NOT happen (e.g. no rolling, no duplicate vehicles).\n")
    best = max(tries, key=lambda t: t["score"])
    shutil.copy(out_dir / f"attempt_{best['attempt']}" / "clip.mp4", out_dir / "best.mp4")
    return {"scores": [t["score"] for t in tries], "best_attempt": best["attempt"],
            "best_prompt": best["plan"]["motion_prompt"], "meaningful": best["score"] >= config.PASS_SCORE}


def _unjudgeable(e: Exception) -> dict:
    """A clip no judge sample can make sense of counts as a failed candidate, not a crash of the whole film."""
    return {"overall": 0.0, "sample_scores": [], "description": "", "summary": f"Unjudgeable clip: {e}",
            "issues": [{"t": "", "severity": "major", "problem": "FAILED CHECK: the judge can reach a verdict on the clip",
                        "fix": "Every judge sample got lost reasoning about this clip; it is likely chaotic "
                               "(duplicated or morphing objects, tumbling). Use a simpler, more controlled motion."}]}
