"""Crusoe Managed Inference client (OpenAI-compatible) with retry on 429/503."""

import json
import os
import random
import re
import time

from openai import APIStatusError, OpenAI

from . import config, trace

_client: OpenAI | None = None


class TruncatedError(RuntimeError):
    """The model hit max_tokens before producing an answer."""


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.BASE_URL, api_key=os.environ["CRUSOE_API_KEY"])
    return _client


def chat(model: str, messages: list[dict], max_tokens: int = 16000, temperature: float = 0.4,
         extra_body: dict | None = None, response_format: dict | None = None) -> str:
    """Return the assistant text with any <think> block removed."""
    stage = trace.caller(2)
    with trace.span(stage, model):
        text = _chat(model, messages, max_tokens, temperature, extra_body, response_format)
    trace.record(stage, model, trace.render_messages(messages), text)
    return text


def _chat(model, messages, max_tokens, temperature, extra_body, response_format) -> str:
    for attempt in range(6):
        try:
            raw = client().chat.completions.with_raw_response.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                extra_body=extra_body, **({"response_format": response_format} if response_format else {}),
            )
            trace.ratelimit(model, dict(raw.headers))
            resp = raw.parse()
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                raise TruncatedError(f"{model} hit max_tokens={max_tokens} before answering")
            text = choice.message.content or ""
            return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        except APIStatusError as e:
            if e.status_code in (429, 503):
                trace.throttle("crusoe", model, e.status_code, str(e))
            if e.status_code not in (429, 503) or attempt == 5:
                raise
            time.sleep(min(60, 2**attempt) + random.random())
    raise AssertionError("unreachable")


def extract_json(text: str) -> dict:
    """Parse the last ```json fenced block, or the outermost {...} span."""
    fenced = re.findall(r"```json\s*(.*?)```", text, flags=re.S)
    if fenced:
        return json.loads(fenced[-1])
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model output:\n{text[:2000]}")
    return json.loads(text[start : end + 1])


def extract_python(text: str) -> str:
    """Return the last ```python fenced block."""
    blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.S)
    if not blocks:
        raise ValueError(f"no ```python block in model output:\n{text[:2000]}")
    return blocks[-1].strip() + "\n"
