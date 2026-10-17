"""Nemotron Omni watches the rendered MP4 and answers a fixed yes/no checklist for the brief.

Why a checklist (each step was measured on real renders and failed):
- A free-form 0-10 score gave 9/10 while listing missing brief requirements.
- A severity-weighted issue list scored the SAME video anywhere from 0 to 10 across samples, and
  its median could not separate a draft from a clearly better revision (5.25 vs 5.75).
- Side-by-side pairwise comparison was a coin flip with a bias toward the right-hand video.
- With the checks in view, the judge confirmed elements that were not in the video (see describe()).
Specific perceptual yes/no questions, grounded in an unprimed description, are what video models
answer consistently. The checklist is
generated once per brief and stored in checklists/, so every iteration, generation and run is
scored against the same questions. Score = 10 * fraction of checks passed by majority vote.
"""

import base64
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio_ffmpeg

from . import config, llm
from .files import atomic_write_text

CHECKLIST_DIR = config.ROOT / "checklists"
MIN_SAMPLES = 3


class Unjudgeable(RuntimeError):
    """Every judge sample on this video reasoned past the token cap without answering. On a chaotic clip
    (a car that tumbled and split in two) all 10 samples did; the clip itself is the problem."""
MIN_CHECKS = 8

CHECKLIST_PROMPT = """Turn this motion-graphics brief into 10-14 concrete yes/no checks that a viewer can verify
by watching the video. Phrase every check so YES = good. Only test what the brief states: never invent a screen
direction, color, count or detail it does not specify, and skip anything it marks as optional. Do NOT include checks about total duration,
resolution or frame rate (those are measured exactly elsewhere). Include:
- one check per requirement in the brief (each element, its content, its order and its animation);
- craft checks that a polished professional piece passes but a typical first draft fails, for example:
  main text is large (at least about a tenth of the frame height) and high-contrast; nothing touches or
  crosses the frame edge; elements are aligned to a clear layout and evenly spaced; animations ease in
  and out instead of moving at constant speed; something meaningful moves or changes in every second
  (no dead air) and nothing appears before the first half second is over (no empty opening); the final
  composition holds still for at least half a second before the end; colors are harmonious; there are
  no glitches, flicker, overlapping text or rendering artifacts; the picture fills the whole frame with
  no black bars, borders, masks or hard-edged vignettes.
BRIEF: """

CHECKLIST_FORMAT = {"type": "json_schema", "json_schema": {"name": "checklist", "schema": {
    "type": "object", "required": ["checks"],
    "properties": {"checks": {"type": "array", "items": {"type": "string"}, "minItems": MIN_CHECKS}}}}}

ANSWER_FORMAT = {"type": "json_schema", "json_schema": {"name": "answers", "schema": {
    "type": "object", "required": ["answers", "summary"], "properties": {
        "answers": {"type": "array", "items": {"type": "object", "required": ["check", "evidence", "pass"],
            "properties": {"check": {"type": "integer"}, "evidence": {"type": "string"},
                           "pass": {"type": "boolean"}}}},
        "summary": {"type": "string"}}}}}


def checklist(brief: str) -> list[str]:
    """The stored checklist for a brief, generated on first use."""
    path = CHECKLIST_DIR / f"{hashlib.sha256(brief.encode()).hexdigest()[:16]}.json"
    if path.exists():
        checks = json.loads(path.read_text())["checks"]
    else:
        checks = json.loads(llm.chat(config.LIBRARIAN_MODEL, [{"role": "user", "content": CHECKLIST_PROMPT + brief}],
                                     max_tokens=8000, temperature=0.2, response_format=CHECKLIST_FORMAT))["checks"]
        if len(checks) >= MIN_CHECKS:
            CHECKLIST_DIR.mkdir(exist_ok=True)
            atomic_write_text(path, json.dumps({"brief": brief, "checks": checks}, indent=1))
    if len(checks) < MIN_CHECKS:
        raise ValueError(f"checklist for brief has {len(checks)} checks, need at least {MIN_CHECKS}: {path}")
    return checks


def _sample(messages: list[dict], n_checks: int) -> dict | None:
    """One answer sheet, or None if the model reasoned past the token cap without answering."""
    try:
        text = llm.chat(config.JUDGE_MODEL, messages, max_tokens=12000, temperature=0.6,
                        extra_body={"reasoning_effort": "low"}, response_format=ANSWER_FORMAT)
    except llm.TruncatedError as e:
        print(f"    judge sample dropped ({e})", flush=True)
        return None
    result = json.loads(text)
    answers = {a["check"]: a for a in result["answers"] if 0 <= a["check"] < n_checks}
    if len(answers) != n_checks:
        print(f"    judge sample dropped (answered {len(answers)} of {n_checks} checks)", flush=True)
        return None
    return {"answers": answers, "summary": result["summary"]}


def critique(brief: str, video: Path, extra_checks: list[str] = ()) -> dict:
    context = f"BRIEF: {brief}\n\nRendered at {config.WIDTH}x{config.HEIGHT}, {config.FPS} fps."
    # Duration is measured, not judged (the model misreads it), so it is a fixed pass/fail check.
    measured = []
    asked = re.search(r"(\d+(?:\.\d+)?)-second", brief)
    if asked:
        _, secs = imageio_ffmpeg.count_frames_and_secs(str(video))
        want = float(asked.group(1))
        measured.append((f"The video is {want:g} seconds long, as the brief asks.", abs(secs - want) <= 0.5,
                         f"measured duration {secs:.2f} seconds"))
    return answer_checks(checklist(brief) + list(extra_checks), context, video, measured=measured)


DESCRIBE_FORMAT = {"type": "json_schema", "json_schema": {"name": "description", "schema": {
    "type": "object", "required": ["description"], "properties": {"description": {"type": "string"}}}}}


def describe(video: Path) -> str:
    """A literal second-by-second description of the video, written WITHOUT the brief or checks.
    Asked with the checks in view, the judge confirmed elements that were not there: a 'vidloop'
    title card scored 7.1 against a podcast brief whose correct video scored 7.9. Grounding the
    checks in an unprimed description dropped wrong videos to 1.4-2.9 and kept the right one at 8.6."""
    b64 = base64.b64encode(video.read_bytes()).decode()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe this video precisely and literally, second by second: every piece of text "
         "exactly as written, every shape and line, the background, what moves and how. Do not guess intent; "
         "only what is visible."},
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
    ]}]
    for attempt in range(3):
        try:
            return json.loads(llm.chat(config.JUDGE_MODEL, messages, max_tokens=12000, temperature=0.2,
                                       extra_body={"reasoning_effort": "low"},
                                       response_format=DESCRIBE_FORMAT))["description"]
        except llm.TruncatedError as e:
            print(f"    description dropped ({e}), attempt {attempt + 1}/3", flush=True)
    raise RuntimeError("judge could not describe the video")


def answer_checks(checks: list[str], context: str, video: Path, measured: list[tuple[str, bool, str]] = ()) -> dict:
    """Have the judge watch `video` and vote on each yes/no check; score = 10 * fraction passed.
    `measured` checks are (check, passed, evidence) decided by measurement, not by the model."""
    frames, secs = imageio_ffmpeg.count_frames_and_secs(str(video))
    b64 = base64.b64encode(video.read_bytes()).decode()
    description = describe(video)
    context += (f"\n\nAN INDEPENDENT OBSERVER (who did not see the brief or the checks) DESCRIBED THIS VIDEO AS:\n"
                f"{description}\n\nWhere a check mentions an element this description does not contain, that "
                "element is absent: fail the check.")
    prompt = (f"{context}\n\nMEASURED FACTS (exact): duration {secs:.2f} seconds, {frames} frames.\n\n"
              "Watch the whole video and answer every check strictly: pass only if it is clearly true. For each, "
              "give short visual evidence (with a timestamp) and then pass true/false. Number the checks as given.\n"
              + "\n".join(f"{i}. {c}" for i, c in enumerate(checks)))
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
    ]}]
    # Runaway samples cluster on some videos (3 of 5 at once), so draw a second round before giving up.
    # Crusoe takes 16+ concurrent calls without slowing, so draw up to three rounds before settling.
    samples, drawn = [], 0
    with ThreadPoolExecutor(config.JUDGE_SAMPLES) as pool:
        for _ in range(3):
            samples += [s for s in pool.map(lambda _: _sample(messages, len(checks)), range(config.JUDGE_SAMPLES)) if s]
            drawn += config.JUDGE_SAMPLES
            if len(samples) >= MIN_SAMPLES:
                break
    if len(samples) < 2:
        # A clip almost no sample can make sense of is a failed candidate, not a crash of the run.
        raise Unjudgeable(f"{len(samples)} of {drawn} judge samples answered (the rest got lost reasoning)")

    issues, passed = [], 0
    for i, check in enumerate(checks):
        votes = [s["answers"][i]["pass"] for s in samples]
        if sum(votes) * 2 > len(votes):  # strict majority; a tie fails
            passed += 1
            continue
        evidence = Counter(s["answers"][i]["evidence"] for s in samples if not s["answers"][i]["pass"])
        issues.append({"t": "", "severity": "major", "problem": f"FAILED CHECK: {check}",
                       "fix": f"Judge evidence: {evidence.most_common(1)[0][0]}"})
    for check, ok, evidence in measured:
        if ok:
            passed += 1
        else:
            issues.append({"t": "", "severity": "major", "problem": f"FAILED CHECK: {check}", "fix": f"Measured: {evidence}"})
    total = len(checks) + len(measured)
    measured_passed = sum(ok for _, ok, _ in measured)
    per_sample = [round(10 * (sum(s["answers"][i]["pass"] for i in range(len(checks))) + measured_passed) / total, 1)
                  for s in samples]
    return {
        "overall": round(10 * passed / total, 1),
        "issues": issues,
        "summary": f"{passed} of {total} checks passed. " + samples[0]["summary"],
        "sample_scores": sorted(per_sample),
        "description": description,
    }
