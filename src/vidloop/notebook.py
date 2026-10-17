"""The director's notebook: lessons about prompting the image and video models, learned across films.

After each film, the librarian reads every shot attempt (prompts, the judge's verdicts, failed
must-pass checks), the film-level verdict and the reviewers' notes, and rewrites the notebook. The
storyboard writer, the shot director and the reviser all read it, so the next film starts from what
earlier films learned, the way `evolve` does for the code agent's skill library.
"""

import json
from pathlib import Path

from . import config, llm
from .files import atomic_write_text

PATH = config.SKILLS_DIR / "director_lessons.md"
MAX_LESSONS = 20

FORMAT = {"type": "json_schema", "json_schema": {"name": "notebook", "schema": {
    "type": "object", "required": ["lessons", "rationale"], "properties": {
        "lessons": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"}}}}}

PROMPT = f"""You keep the notebook of a film director who makes short films by generating one still per shot
(Qwen-Image) and animating it (HappyHorse image-to-video), checked by a video-watching judge and by reviewers.
From the record below, rewrite the COMPLETE notebook (at most {MAX_LESSONS} lessons). Each lesson is a concrete,
general rule about writing storyboards, still prompts, motion prompts or checks that makes a shot pass on the
first attempt: which phrasings produced the wanted result, which produced failures, what the models cannot do.
Prefer lessons backed by a pass/fail contrast or a reviewer note. Keep earlier lessons that still hold, merge
duplicates, drop ones the record contradicts. Never write rules that only apply to this one film's subject.
Reply as JSON: lessons, rationale (2-3 sentences on what changed)."""


def lessons() -> list[str]:
    if not PATH.exists():
        return []
    return [l[2:].strip() for l in PATH.read_text().splitlines() if l.startswith("- ")]


def text() -> str:
    """The notebook as prompt context ("" when empty)."""
    items = lessons()
    return "DIRECTOR'S NOTEBOOK (lessons from earlier films; follow them):\n" + "\n".join(f"- {l}" for l in items) + "\n\n" if items else ""


def record(film_dir: Path) -> dict:
    """Everything one film taught: each shot attempt with its prompts and verdict, reviewers' notes, film verdict."""
    attempts = []
    for result in sorted(film_dir.glob("shot_*/attempt_*/result.json")):
        r = json.loads(result.read_text())
        attempts.append({
            "shot": result.parts[-3], "attempt": result.parts[-2], "score": r["verdict"]["overall"],
            "motion_prompt": r["plan"]["motion_prompt"],
            "failed": [i["problem"].removeprefix("FAILED CHECK: ") for i in r["verdict"]["issues"]]})
    board = json.loads((film_dir / "storyboard.json").read_text())
    notes = film_dir / "notes.jsonl"
    film = film_dir / "result.json"
    return {
        "storyboard": {"style": board.get("style"), "cast": board.get("cast"),
                       "shots": [{k: s.get(k) for k in ("intent", "still_prompt", "motion_prompt", "must", "continues")}
                                 for s in board["shots"]]},
        "attempts": attempts,
        "reviewer_notes": [json.loads(l) for l in notes.read_text().splitlines()] if notes.exists() else [],
        "film_failed_checks": json.loads(film.read_text())["failed"] if film.exists() else [],
    }


def distill(film_dir: Path) -> dict:
    """Rewrite the notebook from one film's record; returns the change log."""
    messages = [{"role": "user", "content": f"{PROMPT}\n\nCURRENT NOTEBOOK:\n{json.dumps(lessons(), indent=1)}\n\n"
                                            f"RECORD OF THE LATEST FILM:\n{json.dumps(record(film_dir), indent=1)}"}]
    before = len(lessons())
    out = json.loads(llm.chat(config.LIBRARIAN_MODEL, messages, max_tokens=12000, temperature=0.3, response_format=FORMAT))
    new = out["lessons"][:MAX_LESSONS]
    PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(PATH, "".join(f"- {l}\n" for l in new))
    return {"before": before, "after": len(new), "rationale": out["rationale"]}
