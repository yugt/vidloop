"""`vidloop probe`: measure how much concurrency a provider actually accepts.

The trace shows when calls ran, not how many more the provider would have taken, and Crusoe sends no
x-ratelimit headers. So this fires N concurrent minimal requests per level, bypassing the pipeline's own
caps and cache, and reports latency and rejections per level. Image and video probes spend Alibaba credits.
"""

import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from . import assets, config, llm


def _crusoe(i: int) -> None:
    llm.chat(config.JUDGE_MODEL, [{"role": "user", "content": f"Reply with OK {i}."}], max_tokens=2000,
             extra_body={"reasoning_effort": "low"})


def _image(i: int) -> None:
    resp = httpx.post(f"{config.IMAGE_BASE_URL}/api/v1/services/aigc/multimodal-generation/generation",
                      headers={"Authorization": f"Bearer {os.environ['ALIBABA_API_KEY']}"},
                      json={"model": config.IMAGE_MODEL,
                            "input": {"messages": [{"role": "user", "content": [{"text": "a plain grey test card"}]}]},
                            "parameters": {"size": "1328*1328", "seed": 1000 + i, "watermark": False}}, timeout=180)
    if resp.status_code != 200 or "Throttling" in resp.text[:300]:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:120]}")


def _video(i: int) -> None:
    from PIL import Image
    img = Image.new("RGB", (854, 480), (120 + i, 120, 120))
    assets._fetch_clip(assets._png(img), f"slow camera push-in {i}", 3, 1000 + i)


PROBES = {"crusoe": _crusoe, "image": _image, "video": _video}


def run(provider: str, levels: list[int], out_dir: Path) -> str:
    fn = PROBES[provider]
    rows, n = [], 0
    for level in levels:
        def timed(i):
            t = time.time()
            try:
                fn(n + i)
                return time.time() - t, None
            except Exception as e:  # recorded per request: the point is to count rejections
                return time.time() - t, f"{type(e).__name__}: {str(e)[:100]}"
        start = time.time()
        with ThreadPoolExecutor(level) as pool:
            results = list(pool.map(timed, range(level)))
        n += level
        ok = [s for s, err in results if err is None]
        errors = [err for _, err in results if err]
        rows.append({"concurrent": level, "ok": len(ok), "failed": len(errors), "wall_s": round(time.time() - start, 1),
                     "latency_median_s": round(statistics.median(ok), 1) if ok else None,
                     "latency_max_s": round(max(ok), 1) if ok else None, "errors": errors[:3]})
    path = out_dir / f"probe-{provider}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(rows, indent=1))
    lines = [f"{'concurrent':>10} {'ok':>4} {'failed':>6} {'wall':>8} {'median':>8} {'max':>8}  errors"]
    for r in rows:
        lines.append(f"{r['concurrent']:>10} {r['ok']:>4} {r['failed']:>6} {r['wall_s']:>7}s "
                     f"{r['latency_median_s'] or '-':>7}s {r['latency_max_s'] or '-':>7}s  {'; '.join(r['errors'])[:90]}")
    return "\n".join(lines) + f"\n(saved {path})"
