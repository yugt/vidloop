"""The librarian's gate is what keeps self-written code from breaking every later render.
These tests run real renders in a subprocess."""

from vidloop.librarian import _validate

GOOD = '''
from PIL import Image

def solid(c: int) -> Image.Image:
    """A solid gray frame."""
    return Image.new("RGB", (640, 360), (c, c, c))

def demo(t: float, W: int, H: int) -> Image.Image:
    return solid(int(255 * t)).resize((W, H))
'''


def test_accepts_helper_whose_demo_renders():
    assert _validate("solid_fill", GOOD) is None


def test_rejects_helper_whose_demo_crashes_with_its_traceback():
    error = _validate("bad", "def demo(t, W, H):\n    return 1 / 0\n")
    assert "ZeroDivisionError" in error


def test_rejects_helper_without_demo():
    error = _validate("nodemo", "def f():\n    pass\n")
    assert "ImportError" in error and "demo" in error


def test_rejects_demo_returning_wrong_size():
    code = "from PIL import Image\ndef demo(t, W, H):\n    return Image.new('RGB', (10, 10))\n"
    assert "expected (640, 360)" in _validate("tiny", code)


def test_helper_can_import_existing_library_module():
    code = "from easing import progress\n" + GOOD.replace("int(255 * t)", "int(255 * progress(t, 0, 1))")
    assert _validate("uses_easing", code) is None


def test_render_error_ends_with_python_exception_not_ffmpeg_noise():
    error = _validate("bad2", "def demo(t, W, H):\n    return 1 / 0\n")
    assert error.strip().splitlines()[-1].startswith("ZeroDivisionError")
