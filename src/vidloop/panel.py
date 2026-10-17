"""Judge panel for clip candidates: several models look at every candidate at once, and their feedback is merged.

- Nemotron Omni: the check-by-check verdict and score (judge.answer_checks).
- Nemotron 3.5 Super-VL: a second video model writing a free-form critique with concrete motion-prompt edits.
- Codex: compares all candidates' contact sheets side by side with the other judges' findings, picks one or
  writes a combined diagnosis.
The candidates are generated in parallel, so the panel's feedback per round is far richer than one verdict,
and a failed round gives the director detailed material for a more specific motion prompt.
"""

import base64
import json
import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

from . import codex_review, config, llm

CRITIC_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL"

CRITIQUE_FORMAT = {"type": "json_schema", "json_schema": {"name": "critique", "schema": {
    "type": "object", "required": ["problems", "prompt_edits"], "properties": {
        "problems": {"type": "array", "items": {"type": "object", "required": ["t", "problem"], "properties": {
            "t": {"type": "string"}, "problem": {"type": "string"}}}},
        "prompt_edits": {"type": "array", "items": {"type": "string"}}}}}}

CRITIQUE = """You are reviewing a clip from an image-to-video model against what it was asked to do.
INTENT: {intent}
MOTION PROMPT IT WAS GIVEN: {motion}
HARD FACTS IT MUST SHOW: {must}
Watch the whole clip. List every problem with a timestamp (wrong motion, wrong direction or rotation axis, objects
that duplicate, morph or vanish, camera moves that were not asked for, physics that look wrong, effects that look
wrong). Then give concrete edits to the motion prompt that would prevent each problem (exact wording to add or
remove, camera position, timing in seconds, rotation axis and start/end orientation, what must NOT happen)."""

PICK_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["best", "problems", "advice"], "properties": {
    "best": {"type": "integer"}, "problems": {"type": "array", "items": {"type": "string"}}, "advice": {"type": "string"}}}

PICK = """The attached images are contact sheets of {k} candidate clips for the same shot (candidate 0, 1, ... in order;
frames left to right every 0.5 seconds). Other judges' findings for each candidate are below. Check every candidate
yourself against the intent and the hard facts (follow the motion frame by frame: direction, rotation, whether the
vehicle stays one object, camera steadiness), then:
- best: the index of the best candidate that achieves the intent and every hard fact, or -1 if none does;
- problems: what is wrong, saying which candidates share each problem;
- advice: if best is -1, a concrete, detailed rewrite of the motion prompt that would fix them ("" otherwise).
INTENT: {intent}
HARD FACTS: {must}
MOTION PROMPT USED: {motion}
OTHER JUDGES PER CANDIDATE:
{findings}"""


def critique(clip: Path, intent: str, motion: str, must: list[str]) -> dict:
    b64 = base64.b64encode(clip.read_bytes()).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": CRITIQUE.format(intent=intent, motion=motion, must=json.dumps(must))},
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}]}]
    for _ in range(2):
        try:
            return json.loads(llm.chat(CRITIC_MODEL, messages, max_tokens=8000, temperature=0.3,
                                       response_format=CRITIQUE_FORMAT))
        except llm.TruncatedError:
            continue
    return {"problems": [{"t": "", "problem": "the critic could not finish reviewing this clip"}], "prompt_edits": []}


def sheet(clip: Path) -> Path:
    out = clip.with_suffix(".png")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(clip),
                    "-vf", "fps=2,scale=320:-1,tile=10x1", "-frames:v", "1", str(out)], check=True)
    return out


def findings(verdicts: list[dict], critiques: list[dict]) -> str:
    lines = []
    for j, (v, c) in enumerate(zip(verdicts, critiques)):
        failed = "; ".join(i["problem"].removeprefix("FAILED CHECK: ") for i in v["issues"]) or "none"
        problems = "; ".join(f"{p['t']} {p['problem']}" for p in c["problems"]) or "none"
        lines.append(f"candidate {j}: Omni score {v['overall']}/10, failed checks: {failed}\n"
                     f"  Super-VL critic: {problems}\n  Super-VL prompt edits: {'; '.join(c['prompt_edits']) or 'none'}")
    return "\n".join(lines)


def pick(sheets: list[Path], verdicts: list[dict] | None, critiques: list[dict] | None, ok: list[bool] | None,
         intent: str, motion: str, must: list[str]) -> dict:
    """Codex compares all candidates (when enabled); otherwise the best candidate that passed Omni is taken.
    verdicts=None runs Codex independently, in parallel with the other judges."""
    if not config.CODEX_REVIEW:
        passing = [j for j, good in enumerate(ok) if good]
        best = max(passing, key=lambda j: verdicts[j]["overall"]) if passing else -1
        return {"best": best, "problems": [], "advice": "", "by": "nemotron"}
    found = ("not available: Omni and Super-VL are reviewing the candidates in parallel; judge them yourself"
             if verdicts is None else findings(verdicts, critiques))
    answer = codex_review.run(sheets, PICK.format(k=len(sheets), intent=intent, must=json.dumps(must), motion=motion,
                                                  findings=found), PICK_SCHEMA, "pick clip",
                              effort="medium")
    if not -1 <= answer["best"] < len(sheets):
        answer["best"] = -1
    return {**answer, "by": "codex"}
