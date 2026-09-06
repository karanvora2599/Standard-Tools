"""The decision record must not live somewhere designed to be deleted.

`_audit_dir` defaulted to `~/.cache/standard_quant_tools/audit`. A cache is
by definition the directory a user is invited to empty -- every "free up disk
space" tool clears it, and the XDG spec says an application must be able to
recreate anything in there. The audit trail is the one file that cannot be
recreated, and the one an incident review reads.
"""

from __future__ import annotations

import pathlib

import pytest

from standard_quant_tools.audit.paths import _audit_dir


@pytest.fixture(autouse=True)
def _no_override(monkeypatch, tmp_path):
    monkeypatch.delenv("SQT_AUDIT_DIR", raising=False)
    # A home with no legacy trail in it, so the default is what is measured.
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def test_the_default_is_not_under_a_cache_directory():
    assert ".cache" not in str(_audit_dir()).lower().replace("\\", "/").split("/")


def test_an_explicit_directory_still_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "chosen"))
    assert _audit_dir() == tmp_path / "chosen"


def test_an_existing_trail_keeps_its_home_rather_than_being_orphaned(_no_override, caplog):
    """Moving the default would make an upgrade look like a deletion.

    The new directory starts empty, so the chain appears to begin at genesis
    and the index that exists to make a missing day detectable has nothing to
    compare against — which is the same event as someone removing a day.
    """
    legacy = _no_override / ".cache" / "standard_quant_tools" / "audit"
    legacy.mkdir(parents=True)
    (legacy / "2026-09-05.jsonl").write_text("{}\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        resolved = _audit_dir()

    assert resolved == legacy, "an existing chain must stay continuous"
    assert "CACHE directory" in caplog.text, "and the operator must be told"


def test_an_empty_legacy_directory_does_not_pin_the_default(_no_override):
    """A leftover empty folder is not a trail worth staying for."""
    (_no_override / ".cache" / "standard_quant_tools" / "audit").mkdir(parents=True)

    assert _audit_dir() != _no_override / ".cache" / "standard_quant_tools" / "audit"
