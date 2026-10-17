import pytest

from vidloop.llm import extract_json, extract_python


def test_json_prefers_last_fenced_block_over_braces_in_prose():
    text = 'I considered {"overall": 1} but then:\n```json\n{"issues": [], "summary": "ok"}\n```'
    assert extract_json(text) == {"issues": [], "summary": "ok"}


def test_json_unfenced_object_with_nested_braces():
    text = 'Here you go:\n{"issues": [{"t": "0:01", "severity": "minor"}], "summary": "a {b}"}\nDone.'
    assert extract_json(text)["issues"][0]["severity"] == "minor"


def test_json_empty_output_raises():
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("")


def test_python_takes_last_block_when_model_shows_old_then_new():
    text = "Old:\n```python\nx = 1\n```\nFixed:\n```python\nx = 2\n```"
    assert extract_python(text) == "x = 2\n"


def test_python_missing_block_raises():
    with pytest.raises(ValueError, match="no ```python block"):
        extract_python("DURATION = 3.0")
