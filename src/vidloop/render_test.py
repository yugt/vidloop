from vidloop.render import render


def _scene(tmp_path, body):
    scene = tmp_path / "scene.py"
    scene.write_text(body)
    return scene


def test_silent_crash_still_reports_a_diagnostic(tmp_path):
    scene = _scene(tmp_path, "import os\nos._exit(3)\n")
    result = render(scene, tmp_path / "out.mp4")
    assert not result.ok
    assert "status 3" in result.error


def test_duration_that_rounds_to_zero_frames_is_rejected(tmp_path):
    scene = _scene(tmp_path, "from PIL import Image\nDURATION = 0.01\n"
                             "def frame(t, W, H):\n    return Image.new('RGB', (W, H))\n")
    result = render(scene, tmp_path / "out.mp4")
    assert not result.ok
    assert "gives no frames" in result.error


def test_scene_does_not_inherit_the_crusoe_key(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUSOE_API_KEY", "should-not-leak")
    scene = _scene(tmp_path, "import os\nassert 'CRUSOE_API_KEY' not in os.environ, 'leaked'\n"
                             "from PIL import Image\nDURATION = 0.5\n"
                             "def frame(t, W, H):\n    return Image.new('RGB', (W, H))\n")
    assert render(scene, tmp_path / "out.mp4").ok
