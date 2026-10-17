"""Outer loop agent: distills a generation's trajectories into the skill library.

It rewrites the lessons, writes new or improved helper modules, and may rewrite the coder's
system prompt. Every helper module must ship a demo(t, W, H) function; a module is only
accepted if its demo renders successfully, so the library never contains broken code.
"""

import json
import shutil
import tempfile
from pathlib import Path

from . import config, llm, skills
from .render import render

PROMPT = """You maintain the skill library of a coding agent that writes Python (Pillow + numpy) programs
that render short motion-graphics videos. A video-watching judge scored its work. Your job: make the
NEXT generation score higher on the FIRST attempt, for briefs it has never seen.

You receive this generation's trajectories (brief, score per iteration, the judge's issues, and the
best final code) plus the current library. Return ONLY a JSON object:
{
 "lessons": ["..."],          // the COMPLETE new lessons list (max 25). Each a concrete, general rule
                               // (e.g. "Body text must be >= 28px with a dark outline on light backgrounds").
                               // Merge duplicates, drop rules that did not help, keep ones that did.
 "helpers": [{"module": "snake_case_name", "code": "..."}],
                               // New or rewritten helper modules (max 3 per generation). Extract code that
                               // recurring fixes needed: easing, text with outline/shadow, layouts, transitions.
                               // Each module: Pillow/numpy/stdlib only, typed functions with one-line docstrings,
                               // AND a `def demo(t: float, W: int, H: int) -> Image.Image` showing it off.
 "system_prompt": "",          // "" to keep the current one, or a full replacement if it clearly needs changing
 "rationale": "2-3 sentences on what you changed and why"
}
Generalize. Never write rules or helpers that only apply to one specific brief."""


# Constrained decoding: a free-form proposal with one JSON syntax error crashed an evolve run.
PROPOSAL_FORMAT = {"type": "json_schema", "json_schema": {"name": "proposal", "schema": {
    "type": "object", "required": ["lessons", "helpers", "system_prompt", "rationale"], "properties": {
        "lessons": {"type": "array", "items": {"type": "string"}},
        "helpers": {"type": "array", "items": {"type": "object", "required": ["module", "code"],
            "properties": {"module": {"type": "string"}, "code": {"type": "string"}}}},
        "system_prompt": {"type": "string"},
        "rationale": {"type": "string"}}}}}


def _validate(module: str, code: str) -> str | None:
    """Return None if the module's demo renders, else the error."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lib = tmp / "lib"
        lib.mkdir()
        (lib / f"{module}.py").write_text(code)
        scene = tmp / "scene.py"
        scene.write_text(f"from {module} import demo as frame\nDURATION = 1.0\n")
        result = render(scene, tmp / "demo.mp4", lib_dirs=[lib, config.SKILL_LIB_DIR])
        return None if result.ok else result.error


def distill(trajectories: list[dict]) -> dict:
    compact = [{
        "brief": t["brief"], "scores": t["scores"], "passed": t["passed"],
        "issues": [i for f in t["feedback"] for i in f.get("issues", [])][:20],
        "render_errors": [f["render_error"][-600:] for f in t["feedback"] if "render_error" in f],
        "best_code": t["best_code"],
    } for t in trajectories]
    current = {
        "system_prompt": skills.system_prompt(),
        "lessons": skills.lessons(),
        "helpers": {p.stem: p.read_text() for p in sorted(config.SKILL_LIB_DIR.glob("*.py"))},
    }
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"CURRENT LIBRARY:\n{json.dumps(current, indent=1)}\n\n"
                                    f"TRAJECTORIES:\n{json.dumps(compact, indent=1)}"},
    ]
    proposal = json.loads(llm.chat(config.LIBRARIAN_MODEL, messages, max_tokens=24000,
                                   response_format=PROPOSAL_FORMAT))
    return apply(proposal)


def apply(proposal: dict) -> dict:
    """Apply a librarian proposal; returns a change log."""
    changes = {"rationale": proposal.get("rationale", ""), "accepted": [], "rejected": {}}
    for h in proposal.get("helpers", []):
        module = h["module"]
        if not module.isidentifier():
            changes["rejected"][module] = "invalid module name"
            continue
        error = _validate(module, h["code"])
        if error:
            changes["rejected"][module] = error[-600:]
            continue
        (config.SKILL_LIB_DIR / f"{module}.py").write_text(h["code"])
        changes["accepted"].append(module)

    skills.write_lessons(proposal["lessons"][:25])
    changes["lessons"] = len(proposal["lessons"][:25])
    if proposal.get("system_prompt"):
        (config.SKILLS_DIR / "system_prompt.md").write_text(proposal["system_prompt"])
        changes["system_prompt_rewritten"] = True
    return changes


def snapshot(dest: Path) -> None:
    """Copy the current library so each generation's skills can be diffed."""
    shutil.copytree(config.SKILLS_DIR, dest, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__"))
