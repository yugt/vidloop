import json

import pytest

from vidloop import trace


def test_status_shows_running_finished_and_failed_steps(tmp_path):
    trace.use(tmp_path)
    with trace.span("still", "qwen-image-3.0-pro", prompt="bridge"):
        pass
    with pytest.raises(ValueError):
        with trace.span("clip", "happyhorse-1.1-i2v"):
            raise ValueError("task failed")
    trace._write({"id": "x", "event": "start", "t": 0, "stage": "_sample", "model": "omni", "pid": __import__("os").getpid()})
    trace._write({"id": "y", "event": "start", "t": 0, "stage": "describe", "model": "dead", "pid": 999999999})
    out = trace.status(tmp_path)
    running, rest = out.split("RECENTLY FINISHED")
    assert "omni" in running and "qwen" not in running
    assert "INTERRUPTED (process gone): 1 steps, last dead describe" in out
    assert "dead" not in running.split("INTERRUPTED")[0]
    assert "qwen-image-3.0-pro" in rest and "ok" in rest
    assert "FAILED ValueError: task failed" in rest
    lines = (tmp_path / "trace.jsonl").read_text().splitlines()
    assert all(json.loads(l)["id"] for l in lines)


def test_caller_names_the_function_that_called_the_traced_function():
    def traced():
        return trace.caller(2)

    def storyboard():
        return traced()

    assert storyboard() == "storyboard"


def test_raw_exchanges_are_recorded_with_media_as_placeholders(tmp_path):
    trace.use(tmp_path)
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Is the roof closed?"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    trace.record("check", "omni", trace.render_messages(msgs), '{"verdict": "no"}')
    e = json.loads((tmp_path / "messages.jsonl").read_text())
    assert e["request"] == "[user] Is the roof closed?\n[image]" and "base64" not in e["request"]
    assert e["response"] == '{"verdict": "no"}'


def test_status_reports_throttling_per_provider_and_ignores_it_in_spans(tmp_path):
    trace.use(tmp_path)
    with trace.span("still", "qwen-image-3.0-pro"):
        trace.throttle("alibaba", "qwen-image-3.0-pro", 429, "Throttling.RateQuota")
    trace.throttle("alibaba", "happyhorse-1.1-i2v", 429, "Throttling")
    out = trace.status(tmp_path)
    assert "alibaba      throttled 2 times" in out
    assert len(trace.spans(tmp_path)) == 1  # throttle events are not spans
