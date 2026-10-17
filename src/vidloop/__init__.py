"""vidloop: a self-improving agent that writes code to render videos, watches them, and learns."""

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="vidloop")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="run the inner loop on one brief")
    m.add_argument("brief")
    m.add_argument("--iters", type=int)

    e = sub.add_parser("evolve", help="run the outer self-improvement loop over a brief set")
    e.add_argument("--briefs", type=Path, default=Path("briefs.json"))
    e.add_argument("--gens", type=int, default=3)
    e.add_argument("--resume", type=Path, help="continue an interrupted run directory")

    r = sub.add_parser("render", help="render a scene file to MP4 (no API calls)")
    r.add_argument("scene", type=Path)
    r.add_argument("out", type=Path)

    j = sub.add_parser("judge", help="ask the judge to critique an MP4")
    j.add_argument("video", type=Path)
    j.add_argument("brief")

    a = sub.add_parser("animate", help="turn still images into short checked clips (image-to-video director)")
    a.add_argument("stills", type=Path, nargs="+")
    a.add_argument("--duration", type=int, default=4)
    a.add_argument("--parallel", type=int, default=2, help="clips generated at once")

    fm = sub.add_parser("film", help="make a 10-20 second film from checked 3-5 second shots")
    fm.add_argument("brief", nargs="?", help="the film brief (omit with --redo)")
    fm.add_argument("--redo", type=int, help="regenerate this shot index of an existing film (needs --dir)")
    fm.add_argument("--note", default="", help="reviewer note for --redo: what was wrong with the shot")
    fm.add_argument("--dir", type=Path, help="existing film directory: resume it, or redo a shot with --redo")
    fm.add_argument("--rounds", type=int, default=2,
                    help="closed-loop rounds: judge the film, diagnose a shot, regenerate it (0 = off)")
    fm.add_argument("--codex", action="store_true",
                    help="add Codex as a second film reviewer (compares all shots' contact sheets)")
    fm.add_argument("--distill", action="store_true",
                    help="only update the director's notebook from the film in --dir")

    st = sub.add_parser("status", help="what each model is doing now in a film or run directory")
    st.add_argument("run_dir", type=Path)

    tl = sub.add_parser("timeline", help="Gantt chart of which model was busy when, with idle gaps")
    tl.add_argument("run_dir", type=Path)
    tl.add_argument("--last", type=float, help="only the last N minutes")

    lg = sub.add_parser("log", help="chronological story of a film run: who did what, what failed, what changed")
    lg.add_argument("run_dir", type=Path)
    lg.add_argument("--raw", action="store_true", help="show the raw prompts and replies exchanged with each model")
    lg.add_argument("--last", type=int, help="with --raw: only the last N exchanges")

    pr = sub.add_parser("probe", help="measure how many concurrent requests a provider accepts")
    pr.add_argument("provider", choices=["crusoe", "image", "video"])
    pr.add_argument("--levels", default="1,2,4", help="comma-separated concurrency levels")

    rp = sub.add_parser("report", help="write a self-contained report.html into a run directory")
    rp.add_argument("run_dir", type=Path)

    args = p.parse_args()
    from . import config

    if args.cmd == "make":
        from .loop import run_brief
        out = config.RUNS_DIR / f"make-{time.strftime('%Y%m%d-%H%M%S')}"
        t = run_brief(args.brief, out, args.iters or config.MAX_ITERS)
        best = out / f"iter_{t['best_iter']}" / "video.mp4"
        print(f"scores {t['scores']} -> best {best}")
    elif args.cmd == "evolve":
        from .evolve import evolve
        print(f"run written to {evolve(args.briefs, args.gens, args.resume.resolve() if args.resume else None)}")
    elif args.cmd == "animate":
        from concurrent.futures import ThreadPoolExecutor
        from .animate import animate_still
        out = config.RUNS_DIR / f"animate-{time.strftime('%Y%m%d-%H%M%S')}"
        with ThreadPoolExecutor(args.parallel) as pool:
            results = list(pool.map(lambda a: animate_still(a[1].resolve(), out / f"{a[0]}_{a[1].stem}", args.duration),
                                    enumerate(args.stills)))
        (out / "summary.json").write_text(json.dumps(results, indent=1))
        for r in results:
            print(f"{'MEANINGFUL' if r['meaningful'] else 'not passed'}  scores {r['scores']}  {Path(r['still']).name}")
        print(f"clips in {out}")
    elif args.cmd == "film":
        from . import notebook
        if args.codex:
            config.CODEX_REVIEW = True
        from .film import improve, make_film, redo, resume
        if args.distill:
            print(json.dumps(notebook.distill(args.dir.resolve()), indent=1))
        elif args.redo is not None:
            print(json.dumps(redo(args.dir.resolve(), args.redo, args.note), indent=1))
        else:
            film_dir = args.dir.resolve() if args.dir else config.RUNS_DIR / f"film-{time.strftime('%Y%m%d-%H%M%S')}"
            if args.dir:
                resume(film_dir)
            else:
                make_film(args.brief, film_dir)
            print(json.dumps(improve(film_dir, args.rounds), indent=1))
    elif args.cmd == "status":
        from .trace import status
        print(status(args.run_dir.resolve()))
    elif args.cmd == "timeline":
        from .trace import timeline
        out = args.run_dir.resolve() / "timeline.png"
        since = time.time() - args.last * 60 if args.last else None
        print(timeline(args.run_dir.resolve(), out, since))
        print(f"chart: {out}")
    elif args.cmd == "log":
        from .runlog import raw, story
        print(raw(args.run_dir.resolve(), args.last) if args.raw else story(args.run_dir.resolve()))
    elif args.cmd == "probe":
        from .probe import run
        config.RUNS_DIR.mkdir(exist_ok=True)
        print(run(args.provider, [int(x) for x in args.levels.split(",")], config.RUNS_DIR))
    elif args.cmd == "report":
        from .report import report
        print(f"report written to {report(args.run_dir.resolve())}")
    elif args.cmd == "render":
        from .render import render
        res = render(args.scene.resolve(), args.out.resolve())
        print(res.video if res.ok else res.error)
        raise SystemExit(0 if res.ok else 1)
    elif args.cmd == "judge":
        from .judge import critique
        print(json.dumps(critique(args.brief, args.video), indent=1))
