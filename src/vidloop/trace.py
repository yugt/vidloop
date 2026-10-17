"""Who is doing what: every model call and pipeline step appends start/end events to the run's trace.jsonl.

`vidloop status <dir>` reads it to show what is running now, recent steps and per-model totals, so the
workflow can be watched without reading logs. Tracing is on once `use(dir)` is called for a run.
"""

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

_path: Path | None = None
_lock = threading.Lock()


def use(run_dir: Path) -> None:
    global _path
    _path = run_dir / "trace.jsonl"


def _write(event: dict) -> None:
    if _path is None:
        return
    with _lock, _path.open("a") as f:
        f.write(json.dumps(event) + "\n")


def caller(depth: int = 2) -> str:
    """Name of the function `depth` frames up: 2 = the caller of the function that calls caller()."""
    return sys._getframe(depth).f_code.co_name


@contextmanager
def span(stage: str, model: str, **detail):
    sid = f"{threading.get_ident()}-{time.monotonic_ns()}"
    start = time.time()
    _write({"id": sid, "event": "start", "t": start, "stage": stage, "model": model, "pid": os.getpid(), **detail})
    try:
        yield
    except BaseException as e:
        _write({"id": sid, "event": "end", "t": time.time(), "secs": round(time.time() - start, 1), "ok": False,
                "error": f"{type(e).__name__}: {str(e)[:200]}"})
        raise
    _write({"id": sid, "event": "end", "t": time.time(), "secs": round(time.time() - start, 1), "ok": True})


def status(run_dir: Path, recent: int = 8) -> str:
    """Human-readable view: running steps, recent finished steps, totals per model."""
    path = run_dir / "trace.jsonl"
    if not path.exists():
        return f"no trace in {run_dir} (started before tracing existed?)"
    starts, ends, throttles, limits = {}, {}, [], {}
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e["event"] in ("start", "end"):
            (starts if e["event"] == "start" else ends)[e["id"]] = e
        elif e["event"] == "throttle":
            throttles.append(e)
        elif e["event"] == "ratelimit":
            limits[e["model"]] = e
    now = time.time()
    out = ["RUNNING NOW"]
    open_steps = [s for i, s in starts.items() if i not in ends]
    running = [s for s in open_steps if _alive(s.get("pid"))]
    out += [f"  {s['model']:<44} {s['stage']:<16} {_detail(s):<28} {_human(now - s['t'])}" for s in running] or ["  (nothing)"]
    interrupted = [s for s in open_steps if not _alive(s.get("pid"))]
    if interrupted:
        out.append(f"INTERRUPTED (process gone): {len(interrupted)} steps, last {interrupted[-1]['model']} {interrupted[-1]['stage']}")
    out.append("RECENTLY FINISHED")
    done = sorted((starts[i] | {"end": e} for i, e in ends.items() if i in starts), key=lambda s: s["end"]["t"])[-recent:]
    for s in done:
        mark = "ok" if s["end"]["ok"] else f"FAILED {s['end'].get('error', '')[:60]}"
        out.append(f"  {s['model']:<44} {s['stage']:<16} {_detail(s):<28} {_human(s['end']['secs'])}  {mark}")
    out.append("TOTALS PER MODEL")
    totals: dict[str, list] = {}
    for i, s in starts.items():
        t = totals.setdefault(s["model"], [0, 0.0, 0])
        t[0] += 1
        if i in ends:
            t[1] += ends[i]["secs"]
            t[2] += not ends[i]["ok"]
    out += [f"  {m:<44} {n:>4} calls  {_human(secs):>12}  {fails} failed" for m, (n, secs, fails) in sorted(totals.items())]
    out.append("PROVIDER HEADROOM")
    per = {}
    for t in throttles:
        per.setdefault(t["provider"], []).append(t)
    out += [f"  {p:<12} throttled {len(ts)} times (last {_human(now - ts[-1]['t'])} ago: HTTP {ts[-1]['status']} {ts[-1]['detail'][:60]})"
            for p, ts in per.items()] or ["  no throttling recorded"]
    for m, e in limits.items():
        rest = {k: v for k, v in e.items() if k not in ("event", "t", "model", "pid")}
        out.append(f"  {m:<44} rate limit {rest}")
    return "\n".join(out)


def _detail(s: dict) -> str:
    return " ".join(f"{k}={v}" for k, v in s.items() if k not in ("id", "event", "t", "stage", "model", "pid", "end"))[:28]


def _alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _human(secs: float) -> str:
    secs = int(secs)
    return f"{secs // 60} min {secs % 60} s" if secs >= 60 else f"{secs} s"


LANES = [("qwen-image", "Qwen-Image (stills)", (255, 170, 60)), ("happyhorse", "HappyHorse (clips)", (240, 90, 90)),
         ("Omni", "Nemotron Omni (judge)", (110, 190, 255)), ("Super", "Nemotron Super (plan/fix)", (120, 220, 140)),
         ("codex", "Codex (review)", (200, 140, 255)), ("ffmpeg", "ffmpeg (edits)", (190, 190, 190)),
         ("DeepSeek", "DeepSeek (coder)", (255, 220, 120))]


def _lane(model: str) -> int:
    return next((k for k, (key, _, _) in enumerate(LANES) if key.lower() in model.lower()), len(LANES) - 1)


def spans(run_dir: Path) -> list[dict]:
    """Closed and still-running spans as {lane, model, stage, t0, t1}; running ones end now."""
    starts, ends = {}, {}
    for line in (run_dir / "trace.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["event"] in ("start", "end"):
            (starts if e["event"] == "start" else ends)[e["id"]] = e
    now = time.time()
    out = []
    for i, s in starts.items():
        if i in ends:
            t1 = ends[i]["t"]
        elif _alive(s.get("pid")):
            t1 = now
        else:
            continue  # interrupted run: its end is unknown
        out.append({"lane": _lane(s["model"]), "model": s["model"], "stage": s["stage"], "t0": s["t"], "t1": t1,
                    "pid": s.get("pid")})
    return sorted(out, key=lambda s: s["t0"])


def idle_gaps(items: list[dict], min_secs: float = 0.0) -> list[tuple[float, float]]:
    """Intervals where no model call was running."""
    gaps, end = [], None
    for s in items:
        if end is not None and s["t0"] - end > min_secs:
            gaps.append((end, s["t0"]))
        end = s["t1"] if end is None else max(end, s["t1"])
    return gaps


def timeline(run_dir: Path, out_png: Path, since: float | None = None) -> str:
    """Gantt chart of model activity plus a utilization report: busy time per model and every idle gap."""
    from PIL import Image, ImageDraw, ImageFont

    every = spans(run_dir)
    latest = next((s["pid"] for s in reversed(every) if s.get("pid")), None)
    # Only the latest run (one process): earlier, interrupted runs would show their restarts as idle time.
    items = [s for s in every if s.get("pid") == latest]
    if since is not None:  # a time window: keep calls overlapping it, clipped to its start
        items = sorted(({**s, "t0": max(s["t0"], since)} for s in items if s["t1"] >= since), key=lambda s: s["t0"])
    if not items:
        return "no spans to draw"
    t0, t1 = items[0]["t0"], max(s["t1"] for s in items)
    wall = t1 - t0
    gaps = idle_gaps(items)
    idle = sum(b - a for a, b in gaps)

    W, row, left, top = 1600, 46, 300, 60
    H = top + row * len(LANES) + 150
    img = Image.new("RGB", (W, H), (22, 24, 30))
    d = ImageDraw.Draw(img)
    f, fs = ImageFont.load_default(size=18), ImageFont.load_default(size=14)
    x = lambda t: left + (W - left - 30) * (t - t0) / wall
    d.text((20, 18), f"Model activity, {_human(wall)} of wall time; nothing running for {_human(idle)} "
                     f"({100 * idle / wall:.0f}%)", font=f, fill=(235, 235, 240))
    for a, b in gaps:
        d.rectangle([x(a), top - 6, x(b), top + row * len(LANES)], fill=(70, 30, 30))
    for k, (_, label, color) in enumerate(LANES):
        y = top + row * k
        d.text((20, y + row / 2), label, font=fs, fill=(220, 220, 225), anchor="lm")
        d.line([left, y + row - 2, W - 30, y + row - 2], fill=(45, 48, 56))
        # Concurrent calls in one lane get their own sub-track, so parallel work is visible.
        tracks: list[float] = []
        placed = []
        for s in (s for s in items if s["lane"] == k):
            slot = next((n for n, end in enumerate(tracks) if end <= s["t0"]), None)
            if slot is None:
                tracks.append(0.0)
                slot = len(tracks) - 1
            tracks[slot] = s["t1"]
            placed.append((slot, s))
        n = max(1, len(tracks))
        h = (row - 16) / n
        for slot, s in placed:
            top_y = y + 8 + slot * h
            d.rectangle([x(s["t0"]), top_y + 1, max(x(s["t1"]), x(s["t0"]) + 2), top_y + h - 1], fill=color)
        if n > 1:
            d.text((W - 26, y + row / 2), f"x{n}", font=fs, fill=color, anchor="lm")
    for m in range(0, int(wall // 60) + 1):
        tx = x(t0 + m * 60)
        d.line([tx, top + row * len(LANES), tx, top + row * len(LANES) + 8], fill=(150, 150, 160))
        d.text((tx, top + row * len(LANES) + 12), f"{m} min", font=fs, fill=(150, 150, 160), anchor="ma")
    d.text((left, H - 60), "dark red = no model running (local Python, ffmpeg stitching, contact sheets, or waiting)",
           font=fs, fill=(200, 140, 140))
    img.save(out_png)

    busy = {}
    for s in items:
        busy.setdefault(LANES[s["lane"]][1], []).append(s["t1"] - s["t0"])
    lines = [f"wall time {_human(wall)}; some model running {100 * (1 - idle / wall):.0f}% of it; "
             f"idle {_human(idle)} in {len(gaps)} gaps"]
    lines += [f"  {lane:<28} {len(v):>3} calls, busy {_human(sum(v))}" for lane, v in busy.items()]
    lines.append("  longest idle gaps: " + ", ".join(_human(b - a) for a, b in sorted(gaps, key=lambda g: g[0] - g[1])[:5]))
    seq = [LANES[s["lane"]][1].split(" (")[0] for s in items]
    handoffs = [seq[0]] + [b for a, b in zip(seq, seq[1:]) if a != b]
    lines.append("  hand-off sequence: " + " -> ".join(handoffs[:30]) + (" ..." if len(handoffs) > 30 else ""))
    return "\n".join(lines)


def throttle(provider: str, model: str, status: int, detail: str = "") -> None:
    """A provider refused or delayed a request for capacity (HTTP 429/503 or a throttling code)."""
    _write({"event": "throttle", "t": time.time(), "provider": provider, "model": model, "status": status,
            "detail": detail[:200], "pid": os.getpid()})


_ratelimit_latest: dict[str, dict] = {}


def ratelimit(model: str, headers: dict) -> None:
    """The provider's own view of remaining capacity (x-ratelimit-* response headers), sampled per call."""
    snap = {k.removeprefix("x-ratelimit-"): v for k, v in headers.items() if k.lower().startswith("x-ratelimit-")}
    if snap:
        _ratelimit_latest[model] = snap
        _write({"event": "ratelimit", "t": time.time(), "model": model, "pid": os.getpid(), **snap})


def record(stage: str, model: str, request: str, response: str, **meta) -> None:
    """The raw exchange with a model (prompt in, text out) appended to messages.jsonl, next to trace.jsonl."""
    if _path is None:
        return
    entry = {"t": time.time(), "stage": stage, "model": model, "request": request, "response": response, **meta}
    with _lock, (_path.parent / "messages.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")


def render_messages(messages: list[dict]) -> str:
    """Chat messages as plain text, with attached media shown as placeholders."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append(f"[{m['role']}] {content}")
            continue
        parts = [p["text"] if p["type"] == "text" else f"[{p['type'].removesuffix('_url')}]" for p in content]
        out.append(f"[{m['role']}] " + "\n".join(parts))
    return "\n".join(out)
