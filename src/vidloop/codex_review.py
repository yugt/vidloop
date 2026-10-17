"""A second, independent film reviewer: Codex looks at every shot's contact sheet side by side.

Nemotron Omni judges one shot (or the whole film) as video, and has missed problems that are obvious
when the shots are compared: a car that changed model between shots, a barrel roll accepted as a
turn-around, a tower that ghosted. Codex (the user's configured OpenAI model, via the codex CLI) sees
the contact sheets of all shots at once, so it reviews story beats, continuity across shots,
physical realism and the brief's hard facts, and reports problems per shot.
"""

import json
import subprocess
import tempfile
from pathlib import Path

from . import trace

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["blocking", "issues", "summary"],
          "properties": {
              "blocking": {"type": "boolean"},
              "summary": {"type": "string"},
              "issues": {"type": "array", "items": {
                  "type": "object", "additionalProperties": False,
                  "required": ["shot", "kind", "blocking", "problem", "fix"],
                  "properties": {
                      "shot": {"type": "integer"},
                      "kind": {"type": "string", "enum": ["story", "continuity", "realism", "brief"]},
                      "blocking": {"type": "boolean"},
                      "problem": {"type": "string"},
                      "fix": {"type": "string"}}}}}}

PROMPT = """You are the second reviewer of a short AI-generated film. The attached images are contact sheets:
one per shot in order (shot 0, shot 1, ...; frames left to right every 0.5 seconds), then the whole film
(one frame per second, left to right, top to bottom). Do not run any commands; just look and answer.

Check, comparing the shots with each other:
- story: does each shot visibly show its intended beat, in order?
- continuity: does every recurring subject (vehicle model, shape, color; characters) look the same in every shot?
- realism: physically plausible motion, no warping, morphing, ghosting or impossible paths?
- brief: is every hard fact of the brief and each shot's must-pass checks visibly true?
For each problem give the shot index (-1 for the whole film), kind, whether it is blocking (the story or a hard
fact is broken) or minor, the problem as you see it in the frames, and a concrete fix.
blocking = true if any issue is blocking.

{references}BRIEF: {brief}
SHOTS: {shots}"""


def review(film_dir: Path, board: dict, brief: str, timeout: int = 600) -> dict:
    sheets = [film_dir / f"shot_{i}" / "sheet.png" for i in range(len(board["shots"]))] + [film_dir / "film_sheet.png"]
    # Reference images (e.g. a choreography diagram) come after the film sheet, in the storyboard's order.
    refs = board.get("references") or []
    sheets += [Path(r["path"]) for r in refs]
    references = "".join(f"REFERENCE IMAGE {k + 1} (attached after the film sheet): {r['caption']} Compare the shots against it.\n"
                         for k, r in enumerate(refs))
    shots = [{"shot": i, "intent": s["intent"], "must": s.get("must", [])} for i, s in enumerate(board["shots"])]
    prompt = PROMPT.format(brief=brief, shots=json.dumps(shots, indent=1), references=references)
    return run(sheets, prompt, SCHEMA, "film review", timeout)


def run(images: list[Path], prompt: str, schema: dict, stage: str, timeout: int = 600, effort: str | None = None) -> dict:
    """One read-only codex exec call with attached images and a JSON-schema answer. `effort` overrides the
    configured reasoning effort (picks among candidates use "medium"; the film review keeps the default)."""
    with tempfile.TemporaryDirectory() as tmp:
        schema_file, out = Path(tmp) / "schema.json", Path(tmp) / "answer.json"
        schema_file.write_text(json.dumps(schema))
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only",
               "--output-schema", str(schema_file), "-o", str(out)]
        if effort:
            cmd += ["-c", f'model_reasoning_effort="{effort}"']
        for image in images:
            cmd += ["-i", str(image)]
        # -i takes any number of files and would swallow a positional prompt, so send the prompt on stdin.
        with trace.span(stage, "codex (configured OpenAI model)", images=len(images)):
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, cwd=tmp)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"codex {stage} failed ({proc.returncode}): {proc.stderr[-1500:]}")
        answer = out.read_text()
        trace.record(stage, "codex (configured OpenAI model)", prompt, answer, images=[Path(i).name for i in images])
        return json.loads(answer)
