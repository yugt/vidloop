"""Inner loop: write -> render -> judge -> revise, until the judge passes it or iterations run out."""

import json
from pathlib import Path

from . import coder, config, judge
from .files import atomic_write_text
from .render import render


def log(msg: str) -> None:
    print(msg, flush=True)


def run_brief(brief: str, out_dir: Path, max_iters: int = config.MAX_ITERS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    iters = []
    code = coder.write(brief)
    for k in range(1, max_iters + 1):
        it_dir = out_dir / f"iter_{k}"
        it_dir.mkdir(exist_ok=True)
        scene = it_dir / "scene.py"
        scene.write_text(code)

        result = render(scene, it_dir / "video.mp4")
        if result.ok:
            feedback = judge.critique(brief, result.video)
            score = feedback["overall"]
            log(f"  iter {k}: score {score:.1f} - {feedback.get('summary', '')}")
        else:
            feedback = {"render_error": result.error}
            score = 0.0
            log(f"  iter {k}: render failed - {result.error.strip().splitlines()[-1]}")
        (it_dir / "critique.json").write_text(json.dumps(feedback, indent=1))
        iters.append({"iter": k, "score": score, "feedback": feedback, "code": code})

        if score >= config.PASS_SCORE or k == max_iters:
            break
        # Hill-climb: revise the best version so far, so one bad revision can't drag later ones down.
        # On ties prefer the latest, so repeated crashes are fixed from the newest attempt.
        best = max(reversed(iters), key=lambda i: i["score"])
        if best["iter"] != k:
            log(f"  iter {k} regressed; revising iter {best['iter']} instead")
        code = coder.revise(brief, best["code"], best["feedback"])

    best = max(iters, key=lambda i: i["score"])
    trajectory = {
        "brief": brief,
        "scores": [i["score"] for i in iters],
        "first_score": iters[0]["score"],
        "best_score": best["score"],
        "best_iter": best["iter"],
        "passed": best["score"] >= config.PASS_SCORE,
        "iterations": len(iters),
        "render_failures": sum("render_error" in i["feedback"] for i in iters),
        "feedback": [i["feedback"] for i in iters],
        "best_code": best["code"],
    }
    atomic_write_text(out_dir / "trajectory.json", json.dumps(trajectory, indent=1))
    return trajectory
