"""Build the ~58-second hackathon demo video from the project's own runs, with a synthesized voice-over.

Order (854x480, 24 fps): the story first, then how it improved, then the technical view.
    title -> the film shot by shot (danger, escape, home) -> failed earlier versions, each tagged with who caught
    it -> how the agents work -> the still gate -> the timeline -> the notebook -> credits.
Every segment has a narration line, synthesized with Alibaba qwen-audio-3.0-tts-plus (cached per line). A
segment lasts at least as long as its line, the line starts with the segment, and the line is burned in as a
subtitle, so picture, voice and captions stay aligned.

Usage: python references/demo_video.py <danger.mp4> <escape.mp4> <home.mp4> <out.mp4>
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 854, 480, 24
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FILM = RUNS / "film-20261018-053741"
FF = imageio_ffmpeg.get_ffmpeg_exe()

BG, FG, DIM, RED, GREEN, BLUE = (16, 18, 24), (240, 240, 245), (150, 152, 165), (235, 70, 70), (90, 210, 120), (110, 190, 255)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def ff(*args: str) -> None:
    subprocess.run([FF, "-y", "-loglevel", "error", *args], check=True)


def write_frames(out: Path, dur: float, draw) -> Path:
    """Render a segment from a frame function draw(t) -> RGB image."""
    writer = imageio_ffmpeg.write_frames(str(out), (W, H), fps=FPS, codec="libx264", pix_fmt_out="yuv420p",
                                         macro_block_size=1, output_params=["-crf", "20"])
    writer.send(None)
    for i in range(int(round(dur * FPS))):
        writer.send(np.ascontiguousarray(np.asarray(draw(i / FPS).convert("RGB"))))
    writer.close()
    return out


def wrap(d: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if d.textlength(trial, font=f) > width and line:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line]


def label_png(path: Path, tag: str, tag_color, text: str, sub: str) -> Path:
    """A bottom caption bar: coloured tag, main text, smaller 'who caught it' line."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, H - 92, W, H], fill=(0, 0, 0, 185))
    tw = d.textlength(tag, font=font(18)) + 20
    d.rounded_rectangle([18, H - 80, 18 + tw, H - 52], radius=6, fill=tag_color)
    d.text((28, H - 66), tag, font=font(18), fill=(255, 255, 255), anchor="lm")
    d.text((30 + tw, H - 66), text, font=font(22), fill=FG, anchor="lm")
    d.text((20, H - 28), sub, font=font(17), fill=(200, 205, 215), anchor="lm")
    img.save(path)
    return path


def clip_seg(out: Path, src: Path, start: float, dur: float, overlay: Path | None = None) -> Path:
    base = f"[0:v]trim=start={start}:duration={dur},setpts=PTS-STARTPTS,scale={W}:{H}:force_original_aspect_ratio=decrease," \
           f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p,setsar=1"
    if overlay:
        ff("-i", str(src), "-loop", "1", "-t", str(dur), "-i", str(overlay), "-filter_complex",
           f"{base}[v];[v][1:v]overlay=0:0,format=yuv420p[o]", "-map", "[o]", "-t", str(dur), "-an",
           "-c:v", "libx264", "-crf", "20", str(out))
    else:
        ff("-i", str(src), "-filter_complex", f"{base}[o]", "-map", "[o]", "-t", str(dur), "-an",
           "-c:v", "libx264", "-crf", "20", str(out))
    return out


def image_seg(out: Path, img: Image.Image, dur: float, zoom: float = 0.04) -> Path:
    """A still image with a slow push-in."""
    base = img.convert("RGB").resize((W, H), Image.LANCZOS)

    def draw(t):
        s = 1 + zoom * t / dur
        w, h = int(W / s), int(H / s)
        return base.crop(((W - w) // 2, (H - h) // 2, (W - w) // 2 + w, (H - h) // 2 + h)).resize((W, H), Image.LANCZOS)
    return write_frames(out, dur, draw)


def title_card(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    a = min(1.0, t / 0.6)
    col = tuple(int(c * a) for c in FG)
    d.text((W / 2, 170), "vidloop", font=font(76), fill=col, anchor="mm")
    d.text((W / 2, 245), "a video studio of agents that judge and fix their own work", font=font(26),
           fill=tuple(int(c * a) for c in (200, 205, 220)), anchor="mm")
    d.text((W / 2, 330), "Built at the Crusoe Recursive Agents Hackathon, Oct 17-18, 2026", font=font(20),
           fill=tuple(int(c * a) for c in DIM), anchor="mm")
    return img


FAILURES = [  # (source, start, kind, label, who caught it)
    (RUNS / "film-20261018-040013/film.mp4", 1.0, "clip", "v1: a pretty travel film with no story", "caught by: human reviewer -> now one story beat per shot"),
    (RUNS / "film-20261018-042602/film.mp4", 4.6, "clip", "the car changes model between shots", "caught by: human -> now a shared cast + Codex continuity review"),
    (RUNS / "film-roadster-story/film_v2.mp4", 3.6, "clip", "flames: the real Roadster uses cold gas", "caught by: news footage review -> now a must-pass 'no flame' check"),
    (FILM / "_shot_0_convertible/still.png", 0.0, "image", "open-top car and two median barriers", "now caught by the still gate (Omni: 'open-top convertible, no roof')"),
    (RUNS / "film-roadster-story/_v6_failed_escape/attempt_1/clip.mp4", 1.4, "clip", "a barrel roll instead of a U-turn", "caught by Super-VL: 'rotates onto its roof instead of yawing flat'"),
    (FILM / "_shot_1_before_rawlog/attempt_1/clip.mp4", 2.2, "clip", "the car splits into two cars", "caught by Super-VL: 'a second identical car appears'"),
    (FILM / "_shot_1_arrows/attempt_1/clip_1.mp4", 3.0, "clip", "road arrows point the same way on both sides", "caught by: human -> now a must-pass road-marking check"),
]

PIPELINE = [  # (box title, model, real quote from the raw log or notebook)
    ("Storyboard", "Nemotron Super", "one story beat per shot, a shared cast"),
    ("3 stills", "Qwen-Image x3", ""),
    ("Still gate", "Nemotron Omni x9 + Codex", "'an open-top convertible with no roof visible'"),
    ("Fix the prompt", "Nemotron Super", "'exactly ONE median barrier ... no painted arrows'"),
    ("2 clips", "HappyHorse i2v x2", ""),
    ("Clip panel", "Omni + Super-VL + Codex", "'rotates onto its roof instead of yawing flat'"),
    ("Film judge", "Omni + Codex", "'the required median barrier is absent'"),
    ("Edit or redo", "Nemotron Super + ffmpeg EDL", "trim, hold, mirror, regrade, or regenerate one shot"),
    ("Notebook", "lessons for the next film", "'spell out counts and exclusions'"),
]


def pipeline_frame(t: float, dur: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 26), "How the agents work", font=font(28), fill=FG, anchor="mm")
    bw, bh, gx, gy, x0, y0 = 250, 80, 26, 26, 27, 84
    active = min(len(PIPELINE) - 1, int(t / (dur / len(PIPELINE))))
    pos = []
    for k in range(len(PIPELINE)):
        row, col = divmod(k, 3)
        col = col if row % 2 == 0 else 2 - col  # snake layout so the flow reads continuously
        pos.append((x0 + col * (bw + gx), y0 + row * (bh + gy)))
    for k in range(len(PIPELINE) - 1):
        (ax, ay), (bx, by) = pos[k], pos[k + 1]
        a = (ax + bw / 2, ay + bh / 2)
        b = (bx + bw / 2, by + bh / 2)
        d.line([a, b], fill=(70, 74, 88) if k >= active else BLUE, width=3)
    d.line([(pos[-1][0] + bw / 2, pos[-1][1] + bh / 2), (pos[0][0] + bw / 2, pos[0][1] + bh / 2)], fill=(45, 48, 58), width=2)
    for k, (title, model, _) in enumerate(PIPELINE):
        x, y = pos[k]
        on = k == active
        done = k < active
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=10, fill=(30, 60, 92) if on else (28, 30, 38),
                            outline=BLUE if on else ((80, 140, 190) if done else (60, 62, 72)), width=3 if on else 1)
        d.text((x + 14, y + 24), f"{k + 1}. {title}", font=font(21), fill=FG, anchor="lm")
        d.text((x + 14, y + 55), model, font=font(16), fill=(170, 200, 230) if on else DIM, anchor="lm")
    quote = PIPELINE[active][2]
    if quote:
        d.text((W / 2, 60), quote, font=font(18), fill=(255, 220, 140), anchor="mm")
    return img


def still_gate_frame(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 30), "Still gate: 3 candidates judged in parallel", font=font(26), fill=FG, anchor="mm")
    d.text((W / 2, 62), "9 Nemotron Omni calls + Codex, before any video is generated", font=font(18), fill=DIM, anchor="mm")
    verdicts = [(RED, "FAIL", "thrusters already firing"), (GREEN, "PICKED", "by Codex: every fact holds"),
                (RED, "FAIL", "thrusters already firing")]
    tw, th = 268, 151
    for j, (col, tag, why) in enumerate(verdicts):
        x, y = 13 + j * (tw + 15), 110
        img.paste(Image.open(FILM / f"shot_0/still_r2_c{j}.png").convert("RGB").resize((tw, th)), (x, y))
        if t > 1.0 + 0.6 * j:
            d.rectangle([x, y, x + tw, y + th], outline=col, width=5)
            d.rounded_rectangle([x, y + th + 14, x + 92, y + th + 44], radius=6, fill=col)
            d.text((x + 46, y + th + 29), tag, font=font(18), fill=(255, 255, 255), anchor="mm")
            for n, line in enumerate(wrap(d, why, font(17), tw)):
                d.text((x, y + th + 60 + 22 * n), line, font=font(17), fill=FG)
    if t > 3.0:
        d.text((W / 2, H - 92), "Codex also corrected 9 Omni votes that miscounted the lanes", font=font(18),
               fill=(255, 220, 140), anchor="mm")
    return img


def lessons_frame(lessons: list[str]):
    def draw(t):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((W / 2, 46), "Lessons the agents wrote in their own notebook", font=font(26), fill=FG, anchor="mm")
        y = 110
        for k, lesson in enumerate(lessons):
            if t < 0.4 + 0.7 * k:
                break
            for n, line in enumerate(wrap(d, lesson, font(20), W - 110)):
                d.text((60 if n == 0 else 76, y), ("- " if n == 0 else "") + line, font=font(20), fill=(220, 225, 235))
                y += 30
            y += 18
        return img
    return draw


def credits_frame(t: float) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    lines = [("Everything shown was built during the hackathon:", FG, 24),
             ("storyboarding, still gate, judge panel, closed loop, EDL edits, notebook, tracing", (210, 215, 225), 18),
             ("", FG, 10),
             ("Third-party models used:", DIM, 18),
             ("Nemotron Omni, Super, Super-VL and DeepSeek on Crusoe Managed Inference", DIM, 17),
             ("Qwen-Image-3.0 and HappyHorse-1.1 on Alibaba Cloud; Codex as second reviewer", DIM, 17),
             ("", FG, 6),
             ("github.com/yugt/vidloop", (255, 220, 140), 24)]
    y = 110
    for text, col, size in lines:
        d.text((W / 2, y), text, font=font(size), fill=col, anchor="mm")
        y += size + 22
    return img


TTS_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
TTS_DIR = RUNS / "demo/tts"
SPEED = 1.1  # the voice reads about 2.4 words per second; 1.1x keeps it natural and the demo under a minute


def tts(text: str, voice: str = "longanhuan_v3.6") -> Path:
    """Synthesize one narration line (cached by voice and text)."""
    import hashlib
    import json
    import os

    import httpx

    path = TTS_DIR / f"{hashlib.sha256((voice + text).encode()).hexdigest()[:16]}.mp3"
    if not path.exists():
        TTS_DIR.mkdir(parents=True, exist_ok=True)
        r = httpx.post(TTS_URL, headers={"Authorization": f"Bearer {os.environ['ALIBABA_API_KEY']}"}, timeout=120,
                       json={"model": "qwen-audio-3.0-tts-plus",
                             "input": {"text": text, "voice": voice, "format": "mp3", "sample_rate": 24000}})
        r.raise_for_status()
        path.write_bytes(httpx.get(json.loads(r.text)["output"]["audio"]["url"], timeout=120).content)
    return path


def audio_secs(path: Path) -> float:
    out = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True).stderr
    h, m, sec = next(l for l in out.splitlines() if "Duration" in l).split("Duration: ")[1].split(",")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def subtitle_png(path: Path, text: str, top: bool = False) -> Path:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lines = wrap(d, text, font(19), W - 80)
    y = 16 if top else H - 14 - 26 * len(lines)
    d.rectangle([0, y - 10, W, H], fill=(0, 0, 0, 150))
    for n, line in enumerate(lines):
        d.text((W / 2, y + 26 * n + 11), line, font=font(19), fill=(255, 255, 255), anchor="mm")
    img.save(path)
    return path


def fit(out: Path, src: Path, dur: float, subtitle: Path) -> Path:
    """Re-time a rendered segment to `dur` (holding its last frame if it is shorter) and burn in the subtitle."""
    ff("-i", str(src), "-loop", "1", "-t", f"{dur:.3f}", "-i", str(subtitle), "-filter_complex",
       f"[0:v]fps={FPS},tpad=stop_mode=clone:stop_duration=60,trim=duration={dur:.3f},setpts=PTS-STARTPTS,"
       f"scale={W}:{H},setsar=1[v];[v][1:v]overlay=0:0,format=yuv420p[o]",
       "-map", "[o]", "-t", f"{dur:.3f}", "-an", "-c:v", "libx264", "-crf", "20", str(out))
    return out


def _short(lesson: str, limit: int = 140) -> str:
    """A notebook lesson without its parenthetical examples, up to its first semicolon, capped at a word boundary."""
    import re
    text = re.sub(r"\s*\([^)]*\)", "", lesson).split(";")[0].strip().rstrip(".,")
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0]
    return text + "."


def build(danger: Path, escape: Path, home: Path, out: Path) -> Path:
    lessons_file = ROOT / "skills/director_lessons.md"
    wanted = ("Spell out counts", "fix the start heading", "special effects")
    all_lessons = [l[2:].strip() for l in lessons_file.read_text().splitlines() if l.startswith("- ")]
    lessons = [next(l for l in all_lessons if w.lower() in l.lower()) for w in wanted]
    lessons = [_short(l) for l in lessons]  # one screen line or two, not a paragraph
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        failures = []
        for k, (src, start, kind, text, who) in enumerate(FAILURES):
            lab = label_png(tmp / f"label_{k}.png", "FAILURE", RED, text, who)
            if kind == "image":
                plain = image_seg(tmp / f"fail_{k}_raw.mp4", Image.open(src), 1.4)
                failures.append(clip_seg(tmp / f"fail_{k}.mp4", plain, 0.0, 1.4, lab))
            else:
                failures.append(clip_seg(tmp / f"fail_{k}.mp4", src, start, 1.4, lab))
        montage = tmp / "failures.mp4"
        (tmp / "fail_list.txt").write_text("".join(f"file '{f}'\n" for f in failures))
        ff("-f", "concat", "-safe", "0", "-i", str(tmp / "fail_list.txt"), "-c:v", "libx264", "-crf", "20", str(montage))

        # (rendered segment, narration line, minimum seconds)
        plan = [
            (write_frames(tmp / "title.mp4", 3.0, title_card),
             "Vidloop: AI agents that make a film, then judge and fix it.", 3.0),
            (clip_seg(tmp / "danger.mp4", danger, 0.0, 4.0),
             "A robot drives the Roadster north. Ahead, the hillside collapses.", 4.0),
            (clip_seg(tmp / "escape.mp4", escape, 0.0, 5.0),
             "A barrier blocks any U-turn. So it takes off, turns in mid-air, and lands.", 5.0),
            (clip_seg(tmp / "home.mp4", home, 0.0, 4.0), "Safe return.", 4.0),
            (montage, "It took many failures: no story, the wrong car, flames, a barrel roll, a car split in two. "
                      "Each is now a check the agents run themselves.", 9.8),
            (write_frames(tmp / "pipeline.mp4", 10.5, lambda t: pipeline_frame(t, 10.5)),
             "Nemotron writes the storyboard. Qwen draws three stills; nine Nemotron judges and Codex check them. "
             "HappyHorse animates the best; Omni, Super-VL and Codex judge each clip.", 9.0),
            (write_frames(tmp / "gate.mp4", 5.0, still_gate_frame),
             "Bad stills never reach the video model.", 5.0),
            (image_seg(tmp / "timeline.mp4", Image.open(FILM / "timeline.png"), 4.0, zoom=0.03),
             "All models run in parallel.", 3.5),
            (write_frames(tmp / "lessons.mp4", 4.0, lessons_frame(lessons)),
             "Mistakes go into a notebook the next film reads.", 3.5),
            (write_frames(tmp / "credits.mp4", 3.0, credits_frame), "Built this weekend on Crusoe.", 2.5),
        ]
        segs, audio = [], []
        for k, (seg, line, minimum) in enumerate(plan):
            top = seg == montage  # the failure labels occupy the bottom of the frame
            voice = tts(line)
            dur = max(minimum, audio_secs(voice) / SPEED + 0.3)
            segs.append(fit(tmp / f"seg_{k:02d}.mp4", seg, dur, subtitle_png(tmp / f"sub_{k}.png", line, top)))
            padded = tmp / f"aud_{k:02d}.wav"
            ff("-i", str(voice), "-af", f"atempo={SPEED},apad=whole_dur={dur:.3f},atrim=duration={dur:.3f}", "-ar", "24000", "-ac", "1",
               str(padded))
            audio.append(padded)
        (tmp / "v.txt").write_text("".join(f"file '{s}'\n" for s in segs))
        (tmp / "a.txt").write_text("".join(f"file '{a}'\n" for a in audio))
        ff("-f", "concat", "-safe", "0", "-i", str(tmp / "v.txt"), "-f", "concat", "-safe", "0", "-i", str(tmp / "a.txt"),
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "128k", "-shortest", str(out))
    return out


if __name__ == "__main__":
    print(build(*(Path(a).resolve() for a in sys.argv[1:5])))
