"""`vidloop report <run_dir>`: write one self-contained static report.html into the run directory.

Understands two on-disk layouts (see evolve.py and loop.py):
    evolve run:  history.jsonl, gen_N/skills/ (library snapshot), gen_N/{train,test}/brief_I/iter_K/...
    make run:    trajectory.json, iter_K/{scene.py,video.mp4,critique.json}
Anything incomplete (no history line, no critique yet) is rendered and labelled "in progress".
"""

import difflib
import json
import re
from html import escape as esc
from pathlib import Path
from urllib.parse import quote

from . import config
from .skills import _signatures

SPLITS = ("train", "test")


# ---------------------------------------------------------------- loading

def _num(path: Path, prefix: str) -> int:
    return int(path.name[len(prefix):])


def _dirs(parent: Path, prefix: str) -> list[Path]:
    found = [p for p in parent.glob(f"{prefix}*") if p.is_dir() and re.fullmatch(rf"{prefix}\d+", p.name)]
    return sorted(found, key=lambda p: _num(p, prefix))


def load_brief(d: Path, run_dir: Path, text: str | None = None) -> dict:
    """One brief's inner loop: every iteration found on disk, scored if its critique exists."""
    traj = json.loads((d / "trajectory.json").read_text()) if (d / "trajectory.json").exists() else None
    iters = []
    for it in _dirs(d, "iter_"):
        crit_path, video = it / "critique.json", it / "video.mp4"
        crit = json.loads(crit_path.read_text()) if crit_path.exists() else None
        if crit is None:
            score = None
        elif "render_error" in crit:
            score = 0.0
        else:
            score = crit["overall"]
        iters.append({
            "k": _num(it, "iter_"),
            "video": video.relative_to(run_dir).as_posix() if video.exists() else None,
            "score": score,
            "critique": crit,
        })
    scored = [i for i in iters if i["score"] is not None]
    best = max(scored, key=lambda i: (i["score"], -i["k"])) if scored else None
    return {
        "brief": (traj or {}).get("brief") or text or "(brief text not recorded)",
        "iters": iters,
        "first": iters[0] if iters and iters[0]["score"] is not None else None,
        "best": best,
        "complete": traj is not None,
    }


def load_evolve(run_dir: Path) -> dict:
    hist_path = run_dir / "history.jsonl"
    history = {}
    if hist_path.exists():
        for line in hist_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                history[rec["gen"]] = rec
    # briefs.json holds the brief text only when a brief has not written trajectory.json yet.
    briefs_file = config.ROOT / "briefs.json"
    texts = json.loads(briefs_file.read_text()) if briefs_file.exists() else {}
    gens = []
    for g in _dirs(run_dir, "gen_"):
        n = _num(g, "gen_")
        splits = {}
        for split in SPLITS:
            splits[split] = [
                load_brief(b, run_dir, (texts.get(split) or [None] * 99)[_num(b, "brief_")]
                           if _num(b, "brief_") < len(texts.get(split, [])) else None)
                for b in _dirs(g / split, "brief_")
            ]
        gens.append({"n": n, "dir": g, "history": history.get(n), "splits": splits,
                     "has_snapshot": (g / "skills").is_dir()})
    return {"gens": gens, "history": history}


# ---------------------------------------------------------------- html helpers

def _score(s: float | None) -> str:
    return "not judged yet" if s is None else f"{s:.1f} / 10"


def _video(rel: str | None, note: str = "") -> str:
    if rel is None:
        return f'<div class="novideo">{esc(note or "No video yet")}</div>'
    return (f'<video controls muted loop playsinline preload="metadata" src="{quote(rel)}"></video>')


def _issues_html(crit: dict | None, limit: int | None = None) -> str:
    if crit is None:
        return '<p class="muted">Waiting for the judge.</p>'
    if "render_error" in crit:
        tail = crit["render_error"].strip().splitlines()[-3:]
        return f'<p><span class="sev critical">render failed</span></p><pre>{esc(chr(10).join(tail))}</pre>'
    out = [f'<p class="summary">{esc(crit.get("summary", ""))}</p>']
    issues = crit.get("issues", [])
    rank = {"critical": 0, "major": 1, "minor": 2}
    issues = sorted(issues, key=lambda i: rank.get(i.get("severity", "minor"), 3))
    shown = issues if limit is None else issues[:limit]
    if shown:
        out.append("<ul class='issues'>")
        for i in shown:
            sev = i.get("severity")
            tag = f'<span class="sev {esc(sev)}">{esc(sev)}</span> ' if sev else ""
            when = f'<span class="muted">at {esc(str(i.get("t", "")))}</span> ' if i.get("t") else ""
            out.append(f"<li>{tag}{when}{esc(i.get('problem', ''))}"
                       f"<div class='fix'>Suggested fix: {esc(i.get('fix', ''))}</div></li>")
        out.append("</ul>")
        if len(issues) > len(shown):
            out.append(f'<p class="muted">and {len(issues) - len(shown)} more issues</p>')
    return "".join(out)


# ---------------------------------------------------------------- results chart

def _chart(history: dict) -> str:
    gens = sorted(history)
    if not gens:
        return '<p class="muted">No generation has finished yet, so there is no curve to draw.</p>'
    W, H, L, R, T, B = 640, 300, 44, 16, 16, 40
    x = lambda g: L + (W - L - R) * (g / max(gens[-1], 1) if gens[-1] else 0.5)
    y = lambda v: T + (H - T - B) * (1 - v / 10)
    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Mean score per generation">']
    for v in range(0, 11, 2):
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                     f'<text class="tick" x="{L - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for g in gens:
        parts.append(f'<text class="tick" x="{x(g):.1f}" y="{H - 18}" text-anchor="middle">gen {g}</text>')
    series = [("train", "first_score", "train, first attempt", "dash"),
              ("train", "best_score", "train, best of loop", ""),
              ("test", "first_score", "test, first attempt", "dash"),
              ("test", "best_score", "test, best of loop", "")]
    for split, key, _, dash in series:
        pts = [(x(g), y(history[g][split][key])) for g in gens]
        d = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        parts.append(f'<polyline class="line {split} {dash}" points="{d}"/>')
        for g, (px, py) in zip(gens, pts):
            parts.append(f'<circle class="dot {split}" cx="{px:.1f}" cy="{py:.1f}" r="4">'
                         f'<title>gen {g} {split} {key.split("_")[0]}: {history[g][split][key]}</title></circle>')
    parts.append("</svg>")
    legend = "".join(f'<span class="key"><i class="sw {s} {d}"></i>{esc(label)}</span>'
                     for s, _, label, d in series)
    return "".join(parts) + f'<div class="legend">{legend}</div>'


def _metrics_table(run: dict) -> str:
    rows = []
    for g in run["gens"]:
        h = g["history"]
        if h is None:
            rows.append(f'<tr><td>gen {g["n"]}</td><td colspan="6" class="muted">in progress</td></tr>')
            continue
        tr, te = h["train"], h["test"]
        rows.append(
            f'<tr><td>gen {g["n"]}</td><td>{tr["first_score"]}</td><td>{tr["best_score"]}</td>'
            f'<td>{te["first_score"]}</td><td>{te["best_score"]}</td>'
            f'<td>{round(tr["pass_rate"] * 100)}% / {round(te["pass_rate"] * 100)}%</td>'
            f'<td>{tr["render_failures"]} / {te["render_failures"]}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Generation</th><th>Train first</th><th>Train best</th>'
            '<th>Test first</th><th>Test best</th><th>Pass rate (train / test)</th>'
            '<th>Crashed renders (train / test)</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


# ---------------------------------------------------------------- per-brief comparison

def _last_run(run: dict, split: str, i: int) -> tuple[int, dict] | None:
    """The most recent generation's run of this brief (preferring one that finished)."""
    cands = [(g["n"], g["splits"][split][i]) for g in run["gens"] if i < len(g["splits"][split])]
    done = [c for c in cands if c[1]["complete"]]
    pool = done or [c for c in cands if c[1]["best"]]
    return pool[-1] if pool else None


def _panel(title: str, rel: str | None, entry: dict | None, note: str = "") -> str:
    if entry is None:
        return f'<div class="panel"><h4>{esc(title)}</h4>{_video(None, "Nothing rendered yet")}</div>'
    crit = entry["critique"]
    return (f'<div class="panel"><h4>{esc(title)}</h4>{_video(rel, "Render crashed, no video")}'
            f'<p class="score">{esc(_score(entry["score"]))}{esc(note)}</p>{_issues_html(crit, 3)}</div>')


def _comparison(run: dict) -> tuple[str, list]:
    cards, runs = [], []
    first_gen = run["gens"][0]
    for split in SPLITS:
        for i, b0 in enumerate(first_gen["splits"][split]):
            last = _last_run(run, split, i)
            gn, bl = last if last else (first_gen["n"], b0)
            runs.append((split, i, gn, bl))
            same = gn == first_gen["n"]
            state = "" if bl["complete"] else ' <span class="badge">in progress</span>'
            first = b0["first"]
            best = bl["best"]
            gain = ""
            if first and best and not same:
                gain = f' <span class="gain">{best["score"] - first["score"]:+.1f} vs. generation {first_gen["n"]} first attempt</span>'
            left = _panel(f"Generation {first_gen['n']}, first attempt", first and first["video"], first)
            right_title = (f"Generation {gn}, best video" if not same
                           else f"Generation {gn}, best after the revision loop")
            right = _panel(right_title, best and best["video"], best)
            cards.append(
                f'<article class="card"><header><span class="badge {split}">{split}</span>'
                f'<span class="muted">brief {i}</span>{state}{gain}</header>'
                f'<p class="brief">{esc(b0["brief"] if b0["complete"] or bl["brief"].startswith("(") else bl["brief"])}</p>'
                f'<div class="pair">{left}{right}</div></article>')
    return "".join(cards), runs


# ---------------------------------------------------------------- iteration strips

def _strip(split: str, i: int, gn: int, b: dict, open_: bool) -> str:
    gain = (b["best"]["score"] - b["iters"][0]["score"]) if b["best"] and b["iters"][0]["score"] is not None else 0
    steps = []
    for it in b["iters"]:
        nxt = ""
        if it["critique"] and "render_error" not in it["critique"] and it["k"] < len(b["iters"]):
            nxt = '<p class="muted">The coder revised the code in response to this critique.</p>'
        steps.append(
            f'<div class="step"><h4>Iteration {it["k"]}</h4>{_video(it["video"], "Render crashed, no video")}'
            f'<p class="score">{esc(_score(it["score"]))}</p>{_issues_html(it["critique"])}{nxt}</div>')
    state = "" if b["complete"] else " (in progress)"
    return (f'<details class="strip"{" open" if open_ else ""}><summary>{split} brief {i}, generation {gn}: '
            f'{len(b["iters"])} iterations, {gain:+.1f} points within the loop{state}</summary>'
            f'<p class="brief">{esc(b["brief"])}</p><div class="steps">{"".join(steps)}</div></details>')


def _strips(runs: list) -> str:
    multi = [r for r in runs if len(r[3]["iters"]) >= 2 and r[3]["iters"][0]["score"] is not None]
    if not multi:
        return '<p class="muted">No brief has more than one judged iteration yet.</p>'
    def gain(r):
        b = r[3]
        return (b["best"]["score"] - b["iters"][0]["score"]) if b["best"] else -99
    top = max(multi, key=gain)
    ordered = [top] + [r for r in multi if r is not top]
    return "".join(_strip(s, i, g, b, r is top) for r in ordered for (s, i, g, b) in [r])


# ---------------------------------------------------------------- what it learned

def _lessons(snapshot_or_live: Path) -> list[str]:
    p = snapshot_or_live / "lessons.md"
    return [l[2:].strip() for l in p.read_text().splitlines() if l.startswith("- ")] if p.exists() else []


def _final_skills(run: dict) -> tuple[Path | None, str]:
    snaps = [g for g in run["gens"] if g["has_snapshot"]]
    if not snaps:
        return None, ""
    last = snaps[-1]
    prev = run["history"].get(last["n"])
    if prev and prev.get("library_changes"):
        # The librarian already rewrote the library after this generation; its result is only in the live dir.
        return config.SKILLS_DIR, f"live library (generation {last['n'] + 1} has not started yet)"
    return last["dir"] / "skills", f"library snapshot used by generation {last['n']}"


def _helper_block(name: str, status: str, path: Path) -> str:
    sigs = _signatures(path)
    body = "".join(f"<li><code>{esc(s)}</code></li>" for s in sigs) or '<li class="muted">no public functions</li>'
    return f'<div class="helper"><h4>{esc(name)} <span class="badge {status}">{status}</span></h4><ul>{body}</ul></div>'


def _learned(run: dict) -> str:
    out = []
    for g in run["gens"]:
        h = g["history"]
        nxt = f'Produced the library for generation {g["n"] + 1}.'
        if h is None:
            out.append(f'<div class="gen"><h3>After generation {g["n"]} <span class="badge">in progress</span></h3>'
                       f'<p class="muted">The librarian has not run yet.</p></div>')
            continue
        lc = h.get("library_changes")
        if lc is None:
            out.append(f'<div class="gen"><h3>After generation {g["n"]}</h3>'
                       f'<p class="muted">Final generation: it was measured, and the library was not changed again.</p></div>')
            continue
        acc = ", ".join(f"<code>{esc(m)}</code>" for m in lc.get("accepted", [])) or "none"
        rej = lc.get("rejected", {})
        rej_html = "".join(f"<li><code>{esc(m)}</code> was rejected because its demo failed to render:"
                           f"<pre>{esc(e[-300:])}</pre></li>" for m, e in rej.items())
        sp = " The coder's system prompt was rewritten." if lc.get("system_prompt_rewritten") else ""
        out.append(
            f'<div class="gen"><h3>After generation {g["n"]}</h3><p>{esc(lc.get("rationale", ""))}</p>'
            f'<p><strong>Helpers accepted:</strong> {acc}. <strong>Lessons now in the library:</strong> '
            f'{lc.get("lessons", "?")}.{sp}</p>'
            + (f'<p><strong>Helpers rejected:</strong></p><ul>{rej_html}</ul>' if rej else "")
            + f'<p class="muted">{nxt}</p></div>')

    base = next((g for g in run["gens"] if g["has_snapshot"]), None)
    final, label = _final_skills(run)
    if base is None or final is None:
        return "".join(out) + '<p class="muted">No library snapshot found.</p>'
    b_sk = base["dir"] / "skills"
    out.append(f'<h3>Lessons: generation {base["n"]} library vs. final</h3>'
               f'<p class="muted">Final = {esc(label)}.</p>')
    old, new = _lessons(b_sk), _lessons(final)
    diff = [l for l in difflib.unified_diff(old, new, lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
    if not diff:
        out.append('<p class="muted">The lessons file has not changed.</p>')
    else:
        out.append('<div class="diff">' + "".join(
            f'<div class="{"add" if l[0] == "+" else "del"}"><b>{"added" if l[0] == "+" else "removed"}</b> {esc(l[1:])}</div>'
            for l in diff) + "</div>")
    out.append(f'<p class="muted">{len(old)} lessons before, {len(new)} after.</p>')

    sp_old = (b_sk / "system_prompt.md")
    sp_new = final / "system_prompt.md"
    if sp_old.exists() and sp_new.exists():
        changed = sp_old.read_text() != sp_new.read_text()
        out.append(f'<p><strong>System prompt:</strong> {"rewritten by the librarian" if changed else "unchanged"}.</p>')

    out.append("<h3>Helper modules</h3>")
    o_lib, n_lib = b_sk / "lib", final / "lib"
    names = sorted({p.stem for d in (o_lib, n_lib) if d.is_dir() for p in d.glob("*.py")})
    blocks = []
    for name in names:
        o, n = o_lib / f"{name}.py", n_lib / f"{name}.py"
        if not o.exists():
            blocks.append(_helper_block(name, "new", n))
        elif not n.exists():
            blocks.append(f'<div class="helper"><h4>{esc(name)} <span class="badge removed">removed</span></h4></div>')
        else:
            blocks.append(_helper_block(name, "unchanged" if o.read_text() == n.read_text() else "rewritten", n))
    out.append('<div class="helpers">' + "".join(blocks) + "</div>" if blocks
               else '<p class="muted">No helper modules.</p>')
    return "".join(out)


# ---------------------------------------------------------------- page

CSS = """
:root{--bg:#fbfaf7;--fg:#1d2330;--muted:#677084;--card:#fff;--line:#e2e0d8;--accent:#2b59c3;
--train:#2b59c3;--test:#d9731a;--add:#e3f4e6;--addfg:#17602a;--del:#fbe6e6;--delfg:#8a1f1f;--code:#f1efe8}
@media (prefers-color-scheme:dark){:root{--bg:#12151c;--fg:#e6e8ee;--muted:#9099ad;--card:#1b2029;--line:#2c3340;
--accent:#7ea2f4;--train:#7ea2f4;--test:#f0a05a;--add:#173323;--addfg:#8fd9a4;--del:#3a1d1f;--delfg:#f0a0a0;--code:#232937}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:2rem;margin:.2em 0}h2{font-size:1.5rem;margin:2.2em 0 .4em;border-top:1px solid var(--line);padding-top:1em}
h3{font-size:1.15rem;margin:1.4em 0 .4em}h4{margin:.2em 0 .5em;font-size:1rem}
.lede{font-size:1.1rem;color:var(--muted);max-width:46em}
.muted{color:var(--muted)}code,pre{font:13px/1.45 ui-monospace,Menlo,Consolas,monospace}
code{background:var(--code);padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}
pre{background:var(--code);padding:10px;border-radius:8px;overflow:auto;white-space:pre-wrap}
.loop{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;margin:1em 0}
.loop div{flex:1 1 150px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:.95rem}
.loop b{display:block;color:var(--accent)}
.card,.gen,.helper,.strip{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:14px 0}
.card header{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.brief{font-style:italic;margin:.5em 0}
.pair,.steps{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.steps{grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}
@media (max-width:640px){.pair{grid-template-columns:1fr}}
video{width:100%;aspect-ratio:16/9;background:#000;border-radius:8px;display:block}
.novideo{aspect-ratio:16/9;display:grid;place-items:center;background:var(--code);border-radius:8px;color:var(--muted);text-align:center;padding:8px}
.score{font-weight:600;margin:.5em 0 .2em}.summary{margin:.2em 0}.gain{color:var(--addfg);font-weight:600}
.issues{padding-left:1.1em;margin:.4em 0}.issues li{margin:.4em 0}.fix{color:var(--muted);font-size:.9rem}
.badge{display:inline-block;font-size:.75rem;padding:1px 8px;border-radius:99px;border:1px solid var(--line);color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.badge.train{color:var(--train);border-color:var(--train)}.badge.test{color:var(--test);border-color:var(--test)}
.badge.new{color:var(--addfg);border-color:var(--addfg)}.badge.rewritten{color:var(--accent);border-color:var(--accent)}
.sev{font-size:.72rem;text-transform:uppercase;font-weight:700;letter-spacing:.04em}
.sev.critical{color:var(--delfg)}.sev.major{color:var(--test)}.sev.minor{color:var(--muted)}
.chart{width:100%;height:auto;max-width:700px}.grid{stroke:var(--line)}.tick{fill:var(--muted);font-size:12px}
.line{fill:none;stroke-width:2.5}.line.dash{stroke-dasharray:6 4;opacity:.8}
.line.train,.dot.train{stroke:var(--train)}.line.test,.dot.test{stroke:var(--test)}
.dot{fill:var(--bg);stroke-width:2.5}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.9rem}.key{display:inline-flex;align-items:center;gap:6px}
.sw{display:inline-block;width:22px;height:0;border-top:3px solid}.sw.train{border-color:var(--train)}.sw.test{border-color:var(--test)}.sw.dash{border-top-style:dashed}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
.diff div{padding:4px 10px;border-radius:6px;margin:4px 0}.diff .add{background:var(--add);color:var(--addfg)}
.diff .del{background:var(--del);color:var(--delfg)}.diff b{text-transform:uppercase;font-size:.7rem;margin-right:8px}
.helpers{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.helper ul{padding-left:1.1em;margin:0}
summary{cursor:pointer;font-weight:600}
"""


def _intro(run_dir: Path, evolve: bool) -> str:
    inner = ('<div><b>1. Write</b>A coder LLM writes Python (Pillow and numpy) scene code for a brief.</div>'
             '<div><b>2. Render</b>The code is rendered to an MP4.</div>'
             '<div><b>3. Watch</b>NVIDIA Nemotron Omni watches the video and lists what is wrong.</div>'
             '<div><b>4. Revise</b>The coder rewrites the scene using that critique, and the loop repeats.</div>')
    outer = ('<div><b>Outer loop</b>After each generation a librarian LLM reads the training trajectories and rewrites '
             'the skill library: lessons, a system prompt and reusable helper code. Held-out test briefs are never '
             'shown to the librarian, so their scores show whether what it learned generalizes.</div>') if evolve else ""
    return (f'<h1>vidloop</h1><p class="lede">Can an agent get better at making videos by watching its own work? '
            f'Writing animation code blind is hard: the first attempt usually has flaws no one can see in the code. '
            f'vidloop closes the loop with a model that actually watches the result.</p>'
            f'<h2>The loop</h2><div class="loop">{inner}</div>'
            f'<div class="loop">{outer}</div><p class="muted">Run: <code>{esc(run_dir.name)}</code></p>')


def build(run_dir: Path) -> str:
    run_dir = run_dir.resolve()
    if (run_dir / "trajectory.json").exists():
        b = load_brief(run_dir, run_dir)
        body = (_intro(run_dir, False) +
                "<h2>Result</h2>" +
                f'<article class="card"><p class="brief">{esc(b["brief"])}</p><div class="pair">'
                f'{_panel("First attempt", b["first"] and b["first"]["video"], b["first"])}'
                f'{_panel("Best video", b["best"] and b["best"]["video"], b["best"])}</div></article>'
                "<h2>How it got there</h2>" +
                _strip("single", 0, 0, b, True).replace("single brief 0, generation 0: ", "This brief: "))
    else:
        run = load_evolve(run_dir)
        if not run["gens"]:
            raise SystemExit(f"{run_dir} has no gen_N directories and no trajectory.json: not a vidloop run")
        cards, runs = _comparison(run)
        body = (_intro(run_dir, True) +
                "<h2>Results: score per generation</h2>"
                '<p class="muted">Mean judge score out of 10. Dashed lines are the first attempt at each brief, solid '
                'lines are the best video after the revision loop. Blue briefs are what the librarian studies; orange '
                'briefs are held out.</p>' + _chart(run["history"]) + _metrics_table(run) +
                "<h2>Every brief, before and after</h2>"
                '<p class="muted">Left: the very first video the agent made, with an empty library. Right: the best '
                'video from the latest generation.</p>' + cards +
                "<h2>Inside one loop</h2>"
                '<p class="muted">Each video, its score, and the critique that drove the next revision. The most '
                'improved brief is open.</p>' + _strips(runs) +
                "<h2>What the agent taught itself</h2>" + _learned(run))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>vidloop report</title>'
            f'<style>{CSS}</style></head><body><main>{body}</main></body></html>')


def report(run_dir: Path) -> Path:
    out = run_dir / "report.html"
    out.write_text(build(run_dir))
    return out
