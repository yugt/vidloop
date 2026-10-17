"""Short film from shots: storyboard -> one checked 3-5 second clip per shot -> ffmpeg stitch -> film QC.

Image-to-video models hold up for one simple camera move over a few seconds and fall apart over long,
multi-beat shots (warped steel, impossible camera paths). So a 10-20 second film is cut into shots
that are each generated and verified on their own (animate.make_shot), then joined with crossfades,
and the title is drawn with Pillow and overlaid so it stays crisp.
"""

import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

from . import assets, codex_review, config, edl, llm, notebook, still_gate, trace
from .animate import make_shot
from .files import atomic_write_text
from .judge import critique
from .loop import log

SIZE = (854, 480)  # HappyHorse 480P is 16:9
XFADE = 0.5

STORYBOARD_FORMAT = {"type": "json_schema", "json_schema": {"name": "storyboard", "schema": {
    "type": "object", "required": ["style", "cast", "title", "shots"], "properties": {
        "style": {"type": "string"},
        "cast": {"type": "string"},
        "title": {"type": "string"},
        "shots": {"type": "array", "minItems": 3, "maxItems": 4, "items": {
            "type": "object",
            "required": ["intent", "still_prompt", "subject", "motion_prompt", "expectations", "must", "duration"],
            "properties": {
                "intent": {"type": "string"}, "still_prompt": {"type": "string"}, "subject": {"type": "string"},
                "motion_prompt": {"type": "string"},
                "expectations": {"type": "array", "items": {"type": "string"}},
                "must": {"type": "array", "items": {"type": "string"}},
                "duration": {"type": "integer", "minimum": 3, "maximum": 5}}}}}}}}

STORYBOARD = """You are a film director storyboarding a {target}-second film for an image-to-video pipeline.
Each shot is made by generating ONE still image (the first frame) and animating it for 3-5 seconds with
ONE simple, achievable camera move plus natural motion. Long or multi-beat moves in one shot fail, so split
the action across 3-4 shots. Shot durations (integers 3-5) must sum to at least {need}.
Return:
- style: continuity notes every still prompt must share (time of day, light, palette, lens, film look)
- cast: one exact visual description of every recurring subject (vehicles, characters, props: make, era,
  shape, color, distinctive details). It is appended word for word to every still prompt, so a car or a
  character looks the same in every shot. "" if nothing recurs.
- title: the on-screen title text exactly as the brief asks ("" if none); it is overlaid during the final
  shot, so never put text in prompts
- shots: for each shot, intent (what it must show in the story), still_prompt (a detailed photorealistic
  first frame that already includes the style, no text), subject (one line), motion_prompt (one camera move
  plus natural motion, never text), expectations (3-4 yes/no checks a viewer could verify, YES = good),
  must (hard facts from the brief that this shot must show, as yes/no checks; a shot failing one is
  regenerated whatever its score; leave out anything the brief marks optional), duration.
BRIEF: {brief}"""


def _ffmpeg(*args: str) -> None:
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *args], check=True)


def _secs(path: Path) -> float:
    return imageio_ffmpeg.count_frames_and_secs(str(path))[1]


def storyboard(brief: str, target: float) -> dict:
    need = target + XFADE * 3
    text = llm.chat(config.LIBRARIAN_MODEL, [{"role": "user", "content": notebook.text() + STORYBOARD.format(
        target=f"{target:g}", need=f"{need:g}", brief=brief)}], max_tokens=12000, temperature=0.4,
        response_format=STORYBOARD_FORMAT)
    return json.loads(text)


REVISE_FORMAT = {"type": "json_schema", "json_schema": {"name": "revision", "schema": {
    "type": "object", "required": ["cast", "still_prompt", "subject", "motion_prompt", "expectations", "must"],
    "properties": {
        "cast": {"type": "string"}, "still_prompt": {"type": "string"}, "subject": {"type": "string"},
        "motion_prompt": {"type": "string"}, "expectations": {"type": "array", "items": {"type": "string"}},
        "must": {"type": "array", "items": {"type": "string"}}}}}}

# A full rewrite once replaced a working continuous-take escape shot with an unrelated follow shot,
# so revisions are minimal edits that keep everything the note does not mention.
REVISE = """A reviewer rejected shot {index} of this storyboard. Make the SMALLEST change that fixes the note:
edit only the words of still_prompt, motion_prompt, subject and expectations that the problem requires, and
copy everything else (camera setup, framing, action, timing, wording) verbatim. Keep the shot's intent.
must: the shot's existing must-pass checks plus any hard fact the note states (e.g. "no flame"), each a
yes/no check phrased so YES = good. cast: the existing cast verbatim, edited only if the note is about how a
recurring subject looks (it is appended word for word to every still prompt).
REVIEWER NOTE: {note}
STORYBOARD: {board}"""


def revise(board: dict, index: int, note: str) -> dict:
    """Rewrite one shot (and the shared cast) from a reviewer's note."""
    text = llm.chat(config.LIBRARIAN_MODEL, [{"role": "user", "content": notebook.text() + REVISE.format(
        index=index, note=note, board=json.dumps(board, indent=1))}], max_tokens=12000, temperature=0.3,
        response_format=REVISE_FORMAT)
    rev = json.loads(text)
    board["cast"] = rev["cast"] or board.get("cast", "")
    board["shots"][index].update({k: rev[k] for k in ("still_prompt", "subject", "motion_prompt", "expectations", "must")})
    return board


def _still_prompt(board: dict, shot: dict) -> str:
    cast = board.get("cast", "")
    return f"{shot['still_prompt']} {cast}" if cast and cast not in shot["still_prompt"] else shot["still_prompt"]


def last_frame(clip: Path) -> Image.Image:
    """The final frame of a clip, at film size."""
    out = clip.with_name("last_frame.png")
    n = imageio_ffmpeg.count_frames_and_secs(str(clip))[0]
    _ffmpeg("-i", str(clip), "-vf", f"select=eq(n\\,{n - 1})", "-frames:v", "1", "-fps_mode", "vfr", str(out))
    return Image.open(out).convert("RGB").resize(SIZE, Image.LANCZOS)


def _save_board(film_dir: Path, data: dict, reason: str) -> None:
    """Write storyboard.json and append the change to storyboard_history.jsonl, so every revision stays visible."""
    atomic_write_text(film_dir / "storyboard.json", json.dumps(data, indent=1))
    with (film_dir / "storyboard_history.jsonl").open("a") as f:
        f.write(json.dumps({"t": time.time(), "reason": reason, "storyboard": data}) + "\n")


def _note(film_dir: Path, entry: dict) -> None:
    with (film_dir / "notes.jsonl").open("a") as f:
        f.write(json.dumps({"t": time.time(), **entry}) + "\n")


STILL_TRIES = 3  # rounds of CANDIDATES stills each
# Independent shots run in parallel so the generators and the judges work at the same time; 2 matches the
# Alibaba plan's concurrency. Shots chained by frame are kept in one sequence.
PARALLEL_SHOTS = 3
_board_lock = threading.Lock()
# Speculative clip generation: started from Nemotron's top-ranked still while Codex is still choosing, so the
# Alibaba slots freed by the stills keep working; the cache's in-flight locks make the real request reuse it.
_speculation = ThreadPoolExecutor(config.CLIP_CANDIDATES * 2)


def _speculate(image: Image.Image, motion: str, duration: int, label: str) -> None:
    for j in range(config.CLIP_CANDIDATES):
        future = _speculation.submit(assets.clip_path, image, motion, duration, j)
        future.add_done_callback(lambda f, j=j: f.exception() and log(f"  [{label}] speculative clip {j} failed: {f.exception()}"))
FIX_STILL_FORMAT = {"type": "json_schema", "json_schema": {"name": "still_fix", "schema": {
    "type": "object", "required": ["still_prompt"], "properties": {"still_prompt": {"type": "string"}}}}}
FIX_STILL = """A generated still (the first frame of a shot) failed checks before animation. Make the SMALLEST edit to the
still prompt that fixes every failure: keep the camera, composition and wording the failures do not concern, and
state counts, exclusions and positions explicitly where something was miscounted or misplaced.
The revised prompt must still satisfy EVERY one of the shot's hard facts below; never trade one away to fix
another (a revision once turned 'seen from behind' into 'seen from the front' and broke the shot's start heading).
HARD FACTS: {must}
STILL PROMPT: {prompt}
WHAT THE IMAGE ACTUALLY SHOWS (a literal description): {description}
FAILED CHECKS (with evidence): {failed}
Return the full revised still_prompt."""


def good_still(film_dir: Path, board: dict, index: int) -> Image.Image:
    """Best-of-K still gate. Each round generates CANDIDATES stills in parallel (different seeds), Nemotron judges
    check them all concurrently, and Codex (when enabled) compares them side by side and picks one; if none passes,
    the shared problems drive one prompt revision for the next round."""
    shot, k_n = board["shots"][index], config.CANDIDATES
    # still_must: only what a FIRST frame can show. Mixing in end-of-shot facts ("by the end the front faces the
    # camera") once made the still judges reject correct stills and the reviser turn the car the wrong way.
    must, context = shot.get("still_must") or shot.get("must", []), f"SHOT INTENT: {shot['intent']}\n"
    for r in range(STILL_TRIES):
        prompt = _still_prompt(board, shot)
        images, checks = [None] * k_n, {}
        with ThreadPoolExecutor(2 * k_n + 1) as pool:
            # Streaming: each candidate is judged by Nemotron the moment it arrives.
            generating = {pool.submit(assets.image, prompt, SIZE, seed=r * k_n + j): j for j in range(k_n)}
            for done in as_completed(generating):
                j = generating[done]
                images[j] = done.result()
                images[j].save(film_dir / f"shot_{index}" / f"still_r{r + 1}_c{j}.png")
                checks[j] = pool.submit(still_gate.check, images[j], must, context, use_codex=False)
            # Overlap: Codex compares all candidates independently while the Nemotron judges are still working.
            codex = pool.submit(still_gate.pick, images, None, must, context) if config.CODEX_REVIEW else None
            verdicts = [checks[j].result() for j in range(k_n)]
            if codex and not shot.get("continues"):
                # Start the clips for Nemotron's best-ranked candidate now, while Codex decides.
                top = min(range(k_n), key=lambda j: len(verdicts[j]["failed"]))
                log(f"  [shot_{index}] speculatively animating candidate {top} while Codex picks")
                _speculate(images[top], shot["motion_prompt"], shot["duration"], f"shot_{index}")
            choice = codex.result() if codex else still_gate.pick(images, verdicts, must, context)
        if choice["best"] < 0:  # merge Nemotron's failures into the diagnosis that drives the prompt fix
            omni = sorted({f["check"] for v in verdicts for f in v["failed"]})
            choice["problems"] = choice["problems"] + [f"Nemotron judges also failed: {c}" for c in omni]
        atomic_write_text(film_dir / f"shot_{index}" / f"still_check_{r + 1}.json", json.dumps(
            {"still_prompt": shot["still_prompt"], "ok": choice["best"] >= 0, "pick": choice,
             "candidates": verdicts, "failed": [] if choice["best"] >= 0 else
             [{"check": p, "by": choice["by"], "evidence": ""} for p in choice["problems"]]}, indent=1))
        if choice["best"] >= 0:
            log(f"  [shot_{index}] round {r + 1}: {choice['by']} picked candidate {choice['best']} of {k_n}")
            return images[choice["best"]]
        failed = "; ".join(choice["problems"]) + (f". Advice: {choice['advice']}" if choice["advice"] else "")
        log(f"  [shot_{index}] round {r + 1}: none of {k_n} candidates passed: {failed[:160]}")
        with _board_lock:
            _note(film_dir, {"shot": index, "by": "still gate", "note": f"{k_n} candidates failed: {failed}"})
        descriptions = " | ".join(f"candidate {j}: {v['description'][:400]}" for j, v in enumerate(verdicts))
        fixed = json.loads(llm.chat(config.LIBRARIAN_MODEL, [{"role": "user", "content": notebook.text() + FIX_STILL.format(
            prompt=shot["still_prompt"], description=descriptions, failed=failed, must=json.dumps(must))}],
            max_tokens=8000, temperature=0.3, response_format=FIX_STILL_FORMAT))["still_prompt"]
        shot["still_prompt"] = fixed
        with _board_lock:
            saved = json.loads((film_dir / "storyboard.json").read_text())
            saved["shots"][index]["still_prompt"] = fixed
            _save_board(film_dir, saved, f"still gate: shot {index} still prompt revised after round {r + 1}")
    raise RuntimeError(f"shot {index}: no still passed the still gate in {STILL_TRIES} rounds of {k_n}")


def shoot(film_dir: Path, board: dict, index: int, note: str = "") -> dict:
    """Generate (or regenerate, with a reviewer's note) one shot and return its summary.
    A shot marked "continues" starts from the previous shot's last frame (frame chaining), so the two
    join with a hard cut into one continuous take."""
    shot = board["shots"][index]
    if shot.get("still_image"):
        # A reviewer-approved first frame: used as is, the still gate is skipped (the clip is still judged).
        (film_dir / f"shot_{index}").mkdir(parents=True, exist_ok=True)
        still = Image.open(shot["still_image"]).convert("RGB").resize(SIZE, Image.LANCZOS)
        log(f"  [shot_{index}] using the reviewer-approved still {Path(shot['still_image']).name}")
    elif shot.get("continues") and index > 0:
        still = last_frame(film_dir / f"shot_{index - 1}" / "best.mp4")
    else:
        (film_dir / f"shot_{index}").mkdir(parents=True, exist_ok=True)
        still = good_still(film_dir, board, index)
    intent = shot["intent"] + (f" Reviewer note: {note}" if note else "")
    plan = {k: shot[k] for k in ("subject", "motion_prompt", "expectations")} | {"must": shot.get("must", [])}
    summary = make_shot(still, film_dir / f"shot_{index}", shot["duration"], plan=plan, intent=intent)
    atomic_write_text(film_dir / f"shot_{index}" / "summary.json", json.dumps(summary, indent=1))
    return summary


def _title_png(film_dir: Path, title: str) -> Path:
    img = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=56)
    x, y = SIZE[0] / 2, SIZE[1] * 0.78
    for dx, dy in ((2, 2), (0, 3), (3, 0)):
        d.text((x + dx, y + dy), title, font=font, fill=(0, 0, 0, 150), anchor="mm")
    d.text((x, y), title, font=font, fill=(255, 255, 255, 255), anchor="mm")
    path = film_dir / "title.png"
    img.save(path)
    return path


def _hard_cut(shot: dict) -> bool:
    return bool(shot.get("continues")) or shot.get("join") == "cut"


def _clip(film_dir: Path, board: dict, i: int) -> Path:
    """The shot's clip as it goes into the film: its best take, run through its EDL if it has one."""
    best = film_dir / f"shot_{i}" / "best.mp4"
    ops = board["shots"][i].get("edit") or []
    return edl.apply(best, best.with_name("cut.mp4"), ops, SIZE) if ops else best


def timeline(film_dir: Path, board: dict) -> list[dict]:
    """Each shot's start and end in film seconds, mirroring stitch(), so film-level timestamps map to shots."""
    out, start = [], 0.0
    for i in range(len(board["shots"])):
        if i > 0 and not _hard_cut(board["shots"][i]):
            start -= XFADE
        secs = _secs(_clip(film_dir, board, i))
        out.append({"shot": i, "start": round(start, 2), "end": round(start + secs, 2)})
        start += secs
    return out


def stitch(film_dir: Path, board: dict, target: float) -> Path:
    """Crossfade the shots' best clips, overlay the title over the final shot, trim to the target length."""
    clips = [_clip(film_dir, board, i) for i in range(len(board["shots"]))]
    inputs, chains = [], []
    for i, c in enumerate(clips):
        inputs += ["-i", str(c)]
        chains.append(f"[{i}:v]scale={SIZE[0]}:{SIZE[1]},fps={config.FPS},format=yuv420p,setsar=1[v{i}]")
    last, offset = "v0", 0.0
    for i in range(1, len(clips)):
        if _hard_cut(board["shots"][i]):
            # Chained shot: it starts on the previous shot's last frame, so a hard cut is seamless.
            offset += _secs(clips[i - 1])
            # concat outputs a 1/1000000 timebase; xfade needs every input at the fps timebase.
            chains.append(f"[{last}][v{i}]concat=n=2:v=1:a=0,settb=1/{config.FPS}[x{i}]")
        else:
            offset += _secs(clips[i - 1]) - XFADE
            chains.append(f"[{last}][v{i}]xfade=transition=fade:duration={XFADE}:offset={offset:.3f}[x{i}]")
        last = f"x{i}"
    total = min(target, offset + _secs(clips[-1]))
    if board["title"]:
        title_at = max(0.0, total - 2.5)
        inputs += ["-loop", "1", "-t", f"{total:.3f}", "-i", str(_title_png(film_dir, board["title"]))]
        chains.append(f"[{len(clips)}:v]format=rgba,fade=in:st={title_at:.2f}:d=0.8:alpha=1[t]")
        chains.append(f"[{last}][t]overlay=0:0[out]")
        last = "out"
    out = film_dir / "film.mp4"
    _ffmpeg(*inputs, "-filter_complex", ";".join(chains), "-map", f"[{last}]", "-t", f"{total:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out))
    return out


# The shot judge sees one shot at a time, so it passed a car that changed model between shots.
CONTINUITY_CHECKS = [
    "Every recurring vehicle keeps the same model, shape and color in every shot.",
    "Every recurring character keeps the same appearance in every shot.",
    "No single action (a take-off, a turn, a landing) is split by a cut or a jump in camera angle.",
]

DIAGNOSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "diagnosis", "schema": {
    "type": "object", "required": ["action", "shot", "note", "edl", "join"], "properties": {
        "action": {"type": "string", "enum": ["edit", "regenerate", "none"]},
        "shot": {"type": "integer"}, "note": {"type": "string"},
        "edl": edl.EDL_SCHEMA, "join": {"type": "string", "enum": ["", "fade", "cut"]}}}}}

DIAGNOSE = """A film made of these shots failed some checks. Fix the most important failure (story beats and
continuity before style) by choosing ONE action on ONE shot:
- "edit": the shot's footage is right but needs editing. Give its complete new EDL (replaces any existing one),
  applied in order to the shot's generated clip. Ops:""" + edl.OPS_HELP.replace("{", "{{").replace("}", "}}") + """
  You may also set join ("fade" or "cut") for how this shot joins the previous one ("" = unchanged).
  Prefer an edit whenever trimming, holding, mirroring, zooming past an artifact, grading or re-joining fixes it:
  it is instant and cannot break what already works.
- "regenerate": the footage itself is wrong (wrong subject, missing action, broken physics). Write a reviewer note:
  what is wrong, what must stay the same, what to change.
- "none": no single shot can fix the failures (e.g. pacing across the whole film).
Timestamps in the evidence are film seconds; the timeline maps them to shots.
STORYBOARD: {board}
TIMELINE: {timeline}
FAILED FILM CHECKS (with the judge's evidence): {failed}
PER-SHOT RESULTS: {shots}"""


def diagnose(film_dir: Path) -> dict:
    board = json.loads((film_dir / "storyboard.json").read_text())
    failed = json.loads((film_dir / "result.json").read_text())["failed"]
    shots = [json.loads((film_dir / f"shot_{i}" / "summary.json").read_text()) for i in range(len(board["shots"]))]
    messages = [{"role": "user", "content": notebook.text() + DIAGNOSE.format(
        board=json.dumps({k: board.get(k) for k in ("cast", "shots")}, indent=1),
        timeline=json.dumps(timeline(film_dir, board)), failed=json.dumps(failed, indent=1),
        shots=json.dumps(shots, indent=1))}]
    # The diagnosis once ran past 8000 tokens with five Codex issues to weigh; give it room and retry.
    for attempt in range(3):
        try:
            return json.loads(llm.chat(config.LIBRARIAN_MODEL, messages, max_tokens=16000, temperature=0.2,
                                       response_format=DIAGNOSE_FORMAT))
        except llm.TruncatedError:
            log(f"  diagnosis cut off, attempt {attempt + 1}/3")
    raise RuntimeError("the diagnosis could not finish in 3 attempts")


def improve(film_dir: Path, rounds: int) -> dict:
    """The closed loop: while the film judge fails it, diagnose the responsible shot, regenerate it with a
    minimal-change note, re-stitch and re-judge. Then distill what the film taught into the notebook."""
    result = json.loads((film_dir / "result.json").read_text())
    for r in range(rounds):
        if result["passed"]:
            break
        d = diagnose(film_dir)
        n_shots = len(json.loads((film_dir / "storyboard.json").read_text())["shots"])
        if d["action"] == "none" or not 0 <= d["shot"] < n_shots:
            log(f"round {r + 1}: no single shot can fix the failures; stopping")
            break
        if d["action"] == "edit":
            log(f"round {r + 1}: edit shot {d['shot']}: {d['edl']} join={d['join'] or 'unchanged'} ({d['note'][:100]})")
            result = edit(film_dir, d["shot"], d["edl"], d["join"], d["note"])
        else:
            log(f"round {r + 1}: regenerate shot {d['shot']}: {d['note'][:140]}")
            result = redo(film_dir, d["shot"], d["note"], by="judge")
    log(f"notebook: {notebook.distill(film_dir)}")
    return result


def contact_sheets(film_dir: Path, n_shots: int) -> None:
    """One frame per 0.5 s for each shot, and one per second for the film: for human review."""
    for i in range(n_shots):
        _ffmpeg("-i", str(film_dir / f"shot_{i}" / ("cut.mp4" if (film_dir / f"shot_{i}" / "cut.mp4").exists() else "best.mp4")),
                "-vf", "fps=2,scale=240:-1,tile=10x1",
                "-frames:v", "1", str(film_dir / f"shot_{i}" / "sheet.png"))
    _ffmpeg("-i", str(film_dir / "film.mp4"), "-vf", "fps=1,scale=320:-1,tile=5x4", "-frames:v", "1",
            str(film_dir / "film_sheet.png"))


def shoot_all(film_dir: Path, board: dict, indices) -> dict[int, dict]:
    """Shoot the given shots, PARALLEL_SHOTS at a time; a shot chained to its predecessor runs after it."""
    indices = list(indices)
    chains: list[list[int]] = []
    for i in indices:
        if chains and board["shots"][i].get("continues") and chains[-1][-1] == i - 1:
            chains[-1].append(i)
        else:
            chains.append([i])
    with ThreadPoolExecutor(PARALLEL_SHOTS) as pool:
        results = pool.map(lambda chain: [(i, shoot(film_dir, board, i)) for i in chain], chains)
        done = dict(pair for chain in results for pair in chain)
    return done


def make_film(brief: str, film_dir: Path) -> dict:
    asked = re.search(r"(\d+(?:\.\d+)?)-second", brief)
    target = float(asked.group(1)) if asked else 15.0
    film_dir.mkdir(parents=True, exist_ok=True)
    trace.use(film_dir)
    board = storyboard(brief, target)
    _save_board(film_dir, {"brief": brief, "target": target, **board}, "storyboard written")
    log(f"storyboard: {len(board['shots'])} shots, title {board['title']!r}")
    done = shoot_all(film_dir, board, range(len(board["shots"])))
    shots = [done[i] for i in range(len(board["shots"]))]
    return finish(film_dir, board, brief, target, shots)


def resume(film_dir: Path) -> dict:
    """Continue an interrupted film: keep the storyboard and finished shots, generate the missing ones."""
    trace.use(film_dir)
    saved = json.loads((film_dir / "storyboard.json").read_text())
    board = {k: saved.get(k, "") for k in ("style", "cast", "title", "shots", "references")}
    todo = [i for i in range(len(board["shots"])) if not (film_dir / f"shot_{i}" / "summary.json").exists()]
    for i in sorted(set(range(len(board["shots"]))) - set(todo)):
        log(f"  [shot_{i}] loaded from disk")
    done = shoot_all(film_dir, board, todo)
    shots = [done[i] if i in done else json.loads((film_dir / f"shot_{i}" / "summary.json").read_text())
             for i in range(len(board["shots"]))]
    return finish(film_dir, board, saved["brief"], saved["target"], shots)


def edit(film_dir: Path, index: int, ops: list[dict], join: str, note: str) -> dict:
    """Apply an EDL (and join) to one shot without regenerating it, then re-stitch and re-check the film."""
    _note(film_dir, {"shot": index, "by": "judge", "action": "edit", "edl": ops, "join": join, "note": note})
    saved = json.loads((film_dir / "storyboard.json").read_text())
    saved["shots"][index]["edit"] = ops
    if join:
        saved["shots"][index]["join"] = join
    _save_board(film_dir, saved, f"edit: EDL applied to shot {index}")
    board = {k: saved.get(k, "") for k in ("style", "cast", "title", "shots", "references")}
    shots = [json.loads((film_dir / f"shot_{i}" / "summary.json").read_text()) for i in range(len(board["shots"]))]
    return finish(film_dir, board, saved["brief"], saved["target"], shots)


def redo(film_dir: Path, index: int, note: str, by: str = "human") -> dict:
    """Regenerate one shot after a reviewer rejects it, then re-stitch and re-check the film."""
    trace.use(film_dir)
    _note(film_dir, {"shot": index, "by": by, "action": "regenerate", "note": note})
    saved = json.loads((film_dir / "storyboard.json").read_text())
    board = revise({k: saved.get(k, "") for k in ("style", "cast", "title", "shots", "references")}, index, note)
    board["shots"][index].pop("edit", None)  # edits were cut for the old clip
    _save_board(film_dir, {**saved, **board}, f"redo: shot {index} revised ({by})")
    log(f"  [shot_{index}] revised still: {board['shots'][index]['still_prompt'][:120]}")
    shoot(film_dir, board, index, note)
    shots = [json.loads((film_dir / f"shot_{i}" / "summary.json").read_text()) for i in range(len(board["shots"]))]
    return finish(film_dir, board, saved["brief"], saved["target"], shots)


def finish(film_dir: Path, board: dict, brief: str, target: float, shots: list[dict]) -> dict:
    film = stitch(film_dir, board, target)
    contact_sheets(film_dir, len(board["shots"]))
    verdict = critique(brief, film, CONTINUITY_CHECKS)
    failed = [f"{i['problem']} ({i['fix']})" for i in verdict["issues"]]
    passed = verdict["overall"] >= config.PASS_SCORE
    codex = None
    if config.CODEX_REVIEW:
        # The film passes only if both reviewers agree: Omni's score and no blocking Codex issue.
        codex = codex_review.review(film_dir, board, brief)
        passed = passed and not codex["blocking"]
        failed += [f"CODEX REVIEW, shot {i['shot']} ({i['kind']}{', blocking' if i['blocking'] else ''}): "
                   f"{i['problem']} Suggested fix: {i['fix']}" for i in codex["issues"]]
    result = {"film": str(film), "seconds": round(_secs(film), 2), "shot_scores": [s["scores"] for s in shots],
              "film_score": verdict["overall"], "passed": passed, "failed": failed}
    atomic_write_text(film_dir / "result.json", json.dumps({**result, "verdict": verdict, "codex": codex}, indent=1))
    log(f"film {result['seconds']} s, film score {verdict['overall']}, passed {passed}, shots {result['shot_scores']}"
        + (f", codex {'BLOCKING' if codex['blocking'] else 'ok'} ({len(codex['issues'])} issues)" if codex else ""))
    return result
