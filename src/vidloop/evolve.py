"""Outer loop: each generation runs every brief with the current library, then the librarian
rewrites the library from the TRAIN trajectories only. TEST briefs are never shown to the
librarian, so their scores measure whether the learned skills generalize."""

import json
import os
import statistics
import time
from pathlib import Path

from . import config, librarian
from .loop import log, run_brief


def _metrics(trajs: list[dict]) -> dict:
    return {
        "first_score": round(statistics.mean(t["first_score"] for t in trajs), 2),
        "best_score": round(statistics.mean(t["best_score"] for t in trajs), 2),
        "iterations": round(statistics.mean(t["iterations"] for t in trajs), 2),
        "pass_rate": round(sum(t["passed"] for t in trajs) / len(trajs), 2),
        "render_failures": sum(t["render_failures"] for t in trajs),
    }


def evolve(briefs_file: Path, gens: int, resume: Path | None = None) -> Path:
    """With `resume`, continue an interrupted run: generations already in history.jsonl are skipped
    and briefs that already have a trajectory.json are loaded instead of re-run."""
    briefs = json.loads(briefs_file.read_text())
    if resume:
        run_dir = resume
    else:
        run_dir = config.RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{'image' if config.IMAGE_TOOL else 'code'}-{os.getpid()}"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({
            "coder": config.CODER_MODEL, "judge": config.JUDGE_MODEL, "librarian": config.LIBRARIAN_MODEL,
            "image_tool": config.IMAGE_TOOL, "image_model": config.IMAGE_MODEL if config.IMAGE_TOOL else None,
            "skills_dir": str(config.SKILLS_DIR), "max_iters": config.MAX_ITERS, "pass_score": config.PASS_SCORE,
            "judge_samples": config.JUDGE_SAMPLES, "gens": gens, "briefs": str(briefs_file)}, indent=1))
    history = run_dir / "history.jsonl"
    done = {json.loads(l)["gen"] for l in history.read_text().splitlines()} if history.exists() else set()

    for gen in range(gens):
        if gen in done:
            log(f"[gen {gen}] already complete, skipping")
            continue
        gen_dir = run_dir / f"gen_{gen}"
        if not (gen_dir / "skills").exists():
            librarian.snapshot(gen_dir / "skills")
        record = {"gen": gen}
        trajs = {}
        for split in ("train", "test"):
            trajs[split] = []
            for i, brief in enumerate(briefs[split]):
                brief_dir = gen_dir / split / f"brief_{i}"
                if (brief_dir / "trajectory.json").exists():
                    saved = json.loads((brief_dir / "trajectory.json").read_text())
                    if saved["brief"] != brief:
                        raise ValueError(f"{brief_dir} was run on a different brief; resume with the run's own briefs file")
                    log(f"[gen {gen}] {split} {i}: loaded from disk")
                    trajs[split].append(saved)
                    continue
                log(f"[gen {gen}] {split} {i}: {brief[:70]}")
                trajs[split].append(run_brief(brief, brief_dir))
            record[split] = _metrics(trajs[split])
        log(f"[gen {gen}] train {record['train']}  test {record['test']}")

        if gen < gens - 1:
            log(f"[gen {gen}] librarian distilling skills...")
            record["library_changes"] = librarian.distill(trajs["train"])
            log(f"[gen {gen}] {record['library_changes']}")
        with history.open("a") as f:
            f.write(json.dumps(record) + "\n")
    return run_dir
