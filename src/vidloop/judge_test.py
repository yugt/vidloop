import hashlib
import json

import pytest

from vidloop import judge


def test_cached_checklist_with_too_few_checks_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "CHECKLIST_DIR", tmp_path)
    brief = "A 3-second test brief."
    path = tmp_path / f"{hashlib.sha256(brief.encode()).hexdigest()[:16]}.json"
    path.write_text(json.dumps({"brief": brief, "checks": []}))
    with pytest.raises(ValueError, match="need at least"):
        judge.checklist(brief)
