"""`vidloop log <film dir>`: one chronological, human-readable story of everything that happened in a film run.

It merges the model-call trace (trace.jsonl), still-gate verdicts (shot_N/still_check_K.json), clip attempts
and their verdicts (shot_N/attempt_K/result.json), storyboard revisions (storyboard_history.jsonl), notes
(notes.jsonl) and the film verdicts (result.json), so a reader can follow the back and forth between the
generators and the judges without opening each file.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from .trace import LANES, _human, spans


def _clock(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M:%S")


def _model_lines(film_dir: Path) -> list[tuple[float, str]]:
    """Model calls, with runs of the same model and stage (e.g. 5 parallel judge samples) collapsed into one line."""
    if not (film_dir / "trace.jsonl").exists():
        return []
    out, group = [], []

    def flush():
        if group:
            label = LANES[group[0]["lane"]][1].split(" (")[0]
            n = f" x{len(group)}" if len(group) > 1 else ""
            secs = max(g["t1"] for g in group) - group[0]["t0"]
            out.append((group[0]["t0"], f"{label}: {group[0]['stage']}{n} ({_human(secs)})"))

    for s in spans(film_dir):
        if group and (s["lane"], s["stage"]) == (group[0]["lane"], group[0]["stage"]) and s["t0"] - group[-1]["t1"] < 5:
            group.append(s)
            continue
        flush()
        group = [s]
    flush()
    return out


def _file_lines(film_dir: Path) -> list[tuple[float, str]]:
    out = []
    for f in sorted(film_dir.glob("shot_*/still_check_*.json")):
        v = json.loads(f.read_text())
        shot, k = f.parent.name, f.stem.rsplit("_", 1)[1]
        if "pick" in v:  # best-of-K round
            p = v["pick"]
            n = len(v["candidates"])
            passed = sum(c["ok"] for c in v["candidates"])
            head = f"[{shot}] still round {k}: {n} candidates, Nemotron passed {passed}"
            out.append((f.stat().st_mtime, f"{head}; {p['by']} picked candidate {p['best']}" if p["best"] >= 0 else
                        f"{head}; {p['by']}: none acceptable ({'; '.join(p['problems'])[:200]})"))
        elif v["ok"]:
            out.append((f.stat().st_mtime, f"[{shot}] still {k} PASSED the still gate"))
        else:
            why = "; ".join(f"{x['by']}: {x['evidence'][:90]}" for x in v["failed"])
            out.append((f.stat().st_mtime, f"[{shot}] still {k} FAILED the still gate ({why})"))
    for f in sorted(film_dir.glob("shot_*/attempt_*/result.json")):
        r = json.loads(f.read_text())
        shot, att = f.parts[-3], f.parts[-2].split("_")[1]
        failed = [i["problem"].removeprefix("FAILED CHECK: ")[:70] for i in r["verdict"]["issues"]]
        out.append((f.stat().st_mtime, f"[{shot}] clip attempt {att} judged {r['verdict']['overall']}/10"
                                       + (f", failed: {'; '.join(failed)}" if failed else "")))
    hist = film_dir / "storyboard_history.jsonl"
    if hist.exists():
        for line in hist.read_text().splitlines():
            h = json.loads(line)
            out.append((h["t"], f"storyboard: {h['reason']}"))
    notes = film_dir / "notes.jsonl"
    if notes.exists():
        for line in notes.read_text().splitlines():
            n = json.loads(line)
            if "t" in n and n.get("by") != "still gate":  # still-gate notes duplicate the still_check lines
                what = n.get("action", "note")
                out.append((n["t"], f"[shot_{n['shot']}] {n['by']} {what}: {n['note'][:160]}"))
    res = film_dir / "result.json"
    if res.exists():
        r = json.loads(res.read_text())
        codex = r.get("codex")
        out.append((res.stat().st_mtime, f"FILM judged {r['film_score']}/10, passed={r.get('passed')}"
                    + (f", Codex {'BLOCKING' if codex['blocking'] else 'ok'}: {codex['summary'][:120]}" if codex else "")))
    return out


def raw(film_dir: Path, last: int | None = None, width: int = 1500) -> str:
    """The raw exchanges with every model, in order: who was asked what, and what came back."""
    path = film_dir / "messages.jsonl"
    if not path.exists():
        return f"no raw messages in {film_dir} (run started before raw logging existed)"
    entries = [json.loads(l) for l in path.read_text().splitlines()]
    out = []
    for e in entries[-last:] if last else entries:
        meta = {k: v for k, v in e.items() if k not in ("t", "stage", "model", "request", "response")}
        out.append(f"===== {_clock(e['t'])}  {e['model']}  [{e['stage']}]" + (f"  {meta}" if meta else ""))
        out.append("--- prompt ---\n" + e["request"][:width] + (" ..." if len(e["request"]) > width else ""))
        out.append("--- reply ---\n" + e["response"][:width] + (" ..." if len(e["response"]) > width else ""))
    return "\n".join(out)


def story(film_dir: Path) -> str:
    events = sorted(_model_lines(film_dir) + _file_lines(film_dir))
    if not events:
        return f"nothing recorded in {film_dir}"
    lines = [f"{_clock(t)}  {text}" for t, text in events]
    return "\n".join(lines) + f"\n({len(events)} events, now {_clock(time.time())})"
