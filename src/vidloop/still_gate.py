"""Still gate: check a generated still against the shot's hard facts BEFORE it is animated.

Stills used to go from Qwen-Image straight to HappyHorse; only the finished clips were judged, so an
open-top car or two median barriers baked into a still were caught by a human, not the agents. Now
Omni (and Codex when enabled) check each still: Omni first describes the image literally without seeing
the checks (the same grounding the video judge needs), then votes on each check. A failed still is
regenerated from a minimally revised prompt with a new seed; the failures are logged as notes for the
director's notebook. Motion-only facts ("lifts off while moving") are answered not_applicable.
"""

import base64
import io
import json
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from . import codex_review, config, llm

SAMPLES = 3
DESCRIBE_FORMAT = {"type": "json_schema", "json_schema": {"name": "d", "schema": {
    "type": "object", "required": ["description"], "properties": {"description": {"type": "string"}}}}}
ANSWER_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answers"], "properties": {
    "answers": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["check", "evidence", "verdict"], "properties": {
                    "check": {"type": "integer"}, "evidence": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["yes", "no", "not_applicable"]}}}}}}

ASK = """Look at this single still image (the first frame of a video shot) and answer each check strictly.
verdict: "yes" only if it is clearly true in the image, "no" if it is false or not visible when it should be,
"not_applicable" only if the check is about motion or events that a single still cannot show, including anything
about how the shot ENDS or what happens over time ("by the end", "after", "while", "then"): this image is the FIRST frame.
Count things carefully (barriers, lanes, cars) and check body details (roof, windows) closely.
Give short visual evidence for each. Number the checks as given.
{context}CHECKS:
{checks}"""


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _image_msg(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


def describe(png: bytes) -> str:
    """Literal description of the still, written without seeing the checks."""
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image precisely and literally: the camera position, every vehicle and its "
         "body style (roof, windows, doors), every person or robot and where they are, the road layout (count the lanes, "
         "barriers, railings and markings), the background and the light. Only what is visible."},
        _image_msg(png)]}]
    for _ in range(3):
        try:
            return json.loads(llm.chat(config.JUDGE_MODEL, messages, max_tokens=8000, temperature=0.2,
                                       extra_body={"reasoning_effort": "low"}, response_format=DESCRIBE_FORMAT))["description"]
        except llm.TruncatedError:
            continue
    raise RuntimeError("could not describe the still")


def _omni_sample(png: bytes, prompt: str, n: int) -> dict | None:
    fmt = {"type": "json_schema", "json_schema": {"name": "answers", "schema": ANSWER_SCHEMA}}
    try:
        text = llm.chat(config.JUDGE_MODEL, [{"role": "user", "content": [{"type": "text", "text": prompt}, _image_msg(png)]}],
                        max_tokens=8000, temperature=0.6, extra_body={"reasoning_effort": "low"}, response_format=fmt)
    except llm.TruncatedError:
        return None
    answers = {a["check"]: a for a in json.loads(text)["answers"] if 0 <= a["check"] < n}
    return answers if len(answers) == n else None


def check(img: Image.Image, checks: list[str], context: str = "", use_codex: bool | None = None) -> dict:
    """Vote on each check; a check fails if a majority of Omni samples, or Codex (when enabled), says no."""
    png = _png(img)
    description = describe(png)
    listed = "\n".join(f"{i}. {c}" for i, c in enumerate(checks))
    grounded = (f"{context}AN INDEPENDENT OBSERVER (who did not see the checks) DESCRIBED THIS IMAGE AS:\n{description}\n"
                "If a check mentions something this description does not contain, treat it as absent.\n\n")
    prompt = ASK.format(context=grounded, checks=listed)
    with ThreadPoolExecutor(SAMPLES) as pool:
        samples = [s for s in pool.map(lambda _: _omni_sample(png, prompt, len(checks)), range(SAMPLES)) if s]
    if len(samples) < 2:
        raise RuntimeError(f"only {len(samples)} of {SAMPLES} still-check samples answered")
    failed = []
    for i, c in enumerate(checks):
        votes = Counter(s[i]["verdict"] for s in samples)
        if votes["no"] * 2 > len(samples):
            failed.append({"check": c, "by": "omni", "evidence": next(s[i]["evidence"] for s in samples if s[i]["verdict"] == "no")})
    if config.CODEX_REVIEW if use_codex is None else use_codex:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "still.png"
            path.write_bytes(png)
            answer = codex_review.run([path], ASK.format(context=context, checks=listed), ANSWER_SCHEMA, "still check")
        for a in answer["answers"]:
            if a["verdict"] == "no" and 0 <= a["check"] < len(checks) and not any(f["check"] == checks[a["check"]] for f in failed):
                failed.append({"check": checks[a["check"]], "by": "codex", "evidence": a["evidence"]})
    return {"ok": not failed, "failed": failed, "description": description}


PICK_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["best", "problems", "advice"], "properties": {
    "best": {"type": "integer"}, "problems": {"type": "array", "items": {"type": "string"}}, "advice": {"type": "string"}}}

PICK = """The attached images are {k} candidate stills for the same shot (candidate 0, 1, ... in order). Several Nemotron
judges already checked each one; their failed checks are below. Verify the checks yourself on every candidate (count
barriers and lanes, look at roofs and windows, which way each vehicle faces). Ignore checks about motion or events
over time (taking off, turning, landing, driving on) and checks about how the shot ENDS ("by the end", "after",
"then"): each image is the FIRST frame, so they never disqualify a candidate; judge only what the first frame shows. Then:
- best: the index of the best candidate that satisfies EVERY check, or -1 if none does;
- problems: the problems you see, saying which candidates share each one;
- advice: if best is -1, how to change the still prompt so the next candidates pass ("" otherwise).
{context}CHECKS:
{checks}
NEMOTRON VERDICTS PER CANDIDATE:
{verdicts}"""


def pick(images: list[Image.Image], verdicts: list[dict] | None, checks: list[str], context: str = "") -> dict:
    """Aggregate a round of candidates: Codex (when enabled) compares them all and picks one or explains what they
    share. verdicts=None runs Codex independently, in parallel with the Nemotron judges (no waiting for them);
    without Codex, the first candidate Nemotron passed is picked."""
    listed = "\n".join(f"{i}. {c}" for i, c in enumerate(checks))
    per = ("not available: the Nemotron judges are checking the candidates in parallel; judge them yourself"
           if verdicts is None else "\n".join(f"candidate {j}: " + ("passed all checks" if v["ok"] else "; ".join(
               f"FAILED {f['check']} ({f['evidence'][:120]})" for f in v["failed"])) for j, v in enumerate(verdicts)))
    if not config.CODEX_REVIEW:
        best = next((j for j, v in enumerate(verdicts) if v["ok"]), -1)
        problems = sorted({f["check"] for v in verdicts for f in v["failed"]})
        return {"best": best, "problems": problems, "advice": "", "by": "nemotron"}
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for j, img in enumerate(images):
            paths.append(Path(tmp) / f"candidate_{j}.png")
            img.save(paths[-1])
        answer = codex_review.run(paths, PICK.format(k=len(images), context=context, checks=listed, verdicts=per),
                                  PICK_SCHEMA, "pick still", effort="medium")
    if not -1 <= answer["best"] < len(images):
        answer["best"] = -1
    return {**answer, "by": "codex"}
