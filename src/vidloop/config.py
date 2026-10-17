"""Models, paths and loop limits. Every value can be overridden by env var."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("VIDLOOP_ROOT", Path(__file__).resolve().parents[2]))
# Parallel evolve runs each need their own library, since the librarian rewrites it in place.
SKILLS_DIR = Path(os.environ.get("VIDLOOP_SKILLS_DIR", ROOT / "skills")).resolve()
SKILL_LIB_DIR = SKILLS_DIR / "lib"
RUNS_DIR = ROOT / "runs"

BASE_URL = os.environ.get("CRUSOE_BASE_URL", "https://api.inference.crusoecloud.com/v1")

# Roles are spread across models: each model has its own per-minute rate limit.
CODER_MODEL = os.environ.get("VIDLOOP_CODER_MODEL", "deepseek-ai/DeepSeek-V4-Pro")
JUDGE_MODEL = os.environ.get("VIDLOOP_JUDGE_MODEL", "nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B")
# Alibaba image generation (DashScope API shape). Token Plan host by default; a pay-as-you-go key
# works by pointing VIDLOOP_IMAGE_BASE_URL at https://dashscope.aliyuncs.com.
IMAGE_BASE_URL = os.environ.get("VIDLOOP_IMAGE_BASE_URL", "https://token-plan.cn-beijing.maas.aliyuncs.com")
IMAGE_MODEL = os.environ.get("VIDLOOP_IMAGE_MODEL", "qwen-image-3.0-pro")
VIDEO_MODEL = os.environ.get("VIDLOOP_VIDEO_MODEL", "happyhorse-1.1-i2v")
# Whether the coder is told about generate_image (off = code-only baseline).
IMAGE_TOOL = os.environ.get("VIDLOOP_IMAGE_TOOL", "0") == "1"
VIDEO_TOOL = os.environ.get("VIDLOOP_VIDEO_TOOL", "0") == "1"
# Second film reviewer: Codex compares all shots' contact sheets (needs the codex CLI; bills the OpenAI account).
CODEX_REVIEW = os.environ.get("VIDLOOP_CODEX_REVIEW", "0") == "1"
# Best-of-K: still candidates generated and judged in parallel per round; Codex (if on) picks among them.
CANDIDATES = int(os.environ.get("VIDLOOP_CANDIDATES", "3"))
CLIP_CANDIDATES = int(os.environ.get("VIDLOOP_CLIP_CANDIDATES", "2"))
# Concurrent Alibaba requests across all shots (the upgraded plan allows 4).
ALIBABA_CONCURRENCY = int(os.environ.get("VIDLOOP_ALIBABA_CONCURRENCY", "4"))
LIBRARIAN_MODEL = os.environ.get("VIDLOOP_LIBRARIAN_MODEL", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B")

WIDTH = 640
HEIGHT = 360
FPS = 24
MAX_DURATION = 10.0
# Long enough for one first-time image-to-video clip (1-5 minutes) plus images.
RENDER_TIMEOUT_S = 600

MAX_ITERS = int(os.environ.get("VIDLOOP_MAX_ITERS", "5"))
PASS_SCORE = float(os.environ.get("VIDLOOP_PASS_SCORE", "9.0"))
JUDGE_SAMPLES = int(os.environ.get("VIDLOOP_JUDGE_SAMPLES", "5"))
