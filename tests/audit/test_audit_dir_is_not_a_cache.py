"""The decision record must not live somewhere designed to be deleted.

`_audit_dir` defaulted to `~/.cache/standard_quant_tools/audit`. A cache is
by definition the directory a user is invited to empty -- every "free up disk
space" tool clears it, and the XDG spec says an application must be able to
recreate anything in there. The audit trail is the one file that cannot be
recreated, and the one an incident review reads.
"""

from __future__ import annotations

import pathlib
import warnings

import pytest

from standard_quant_tools.audit import paths as audit_paths
from standard_quant_tools.audit.paths import AuditLocationWarning, _audit_dir


@pytest.fixture(autouse=True)
def _no_override(monkeypatch, tmp_path):
    monkeypatch.delenv("SQT_AUDIT_DIR", raising=False)
    # A home with no legacy trail in it, so the default is what is measured.
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    # The legacy-location warning is once per PROCESS; each test here starts
    # as a fresh process would.
    monkeypatch.setattr(audit_paths, "_legacy_location_warned", False)
    return tmp_path


def _plant_legacy_trail(home: pathlib.Path) -> pathlib.Path:
    legacy = home / ".cache" / "standard_quant_tools" / "audit"
    legacy.mkdir(parents=True)
    (legacy / "2026-09-05.jsonl").write_text("{}\n", encoding="utf-8")
    return legacy


def test_the_default_is_not_under_a_cache_directory():
    assert ".cache" not in str(_audit_dir()).lower().replace("\\", "/").split("/")


def test_an_explicit_directory_still_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "chosen"))
    assert _audit_dir() == tmp_path / "chosen"


def test_an_existing_trail_keeps_its_home_rather_than_being_orphaned(_no_override):
    """Moving the default would make an upgrade look like a deletion.

    The new directory starts empty, so the chain appears to begin at genesis
    and the index that exists to make a missing day detectable has nothing to
    compare against — which is the same event as someone removing a day.

    The operator is told through `warnings.warn`, which reaches stderr and
    pytest's warnings summary. It used to be a log call on a logger the
    package gives only a NullHandler, so a plain script, the CLI and the MCP
    server never showed it; this test pinned that it was emitted, not that
    anyone could see it.
    """
    legacy = _plant_legacy_trail(_no_override)

    with pytest.warns(AuditLocationWarning, match="CACHE directory"):
        resolved = _audit_dir()

    assert resolved == legacy, "an existing chain must stay continuous"


def test_the_warning_is_given_once_per_process_not_once_per_call(_no_override):
    """`_audit_dir` runs on every tool call. Its comment said "say so
    once", and the message repeated on every call a host logged."""
    _plant_legacy_trail(_no_override)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            _audit_dir()

    location_warnings = [w for w in caught if w.category is AuditLocationWarning]
    assert len(location_warnings) == 1


def test_no_legacy_trail_means_no_warning(_no_override):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _audit_dir()

    assert not [w for w in caught if w.category is AuditLocationWarning]


def test_an_empty_legacy_directory_does_not_pin_the_default(_no_override):
    """A leftover empty folder is not a trail worth staying for."""
    (_no_override / ".cache" / "standard_quant_tools" / "audit").mkdir(parents=True)

    assert _audit_dir() != _no_override / ".cache" / "standard_quant_tools" / "audit"


class TestDescribeAuditLogSaysWhereTheTrailLives:
    """The process-wide warning fires once and reaches whoever reads stderr.
    An agent asking what the log is gets the same fact as a field."""

    def test_a_trail_in_the_legacy_location_is_flagged(self, _no_override):
        from standard_quant_tools.agent.runtimes.meta.audit_tools import (
            AuditLogInput,
            describe_audit_log,
        )

        legacy = _plant_legacy_trail(_no_override)
        with pytest.warns(AuditLocationWarning):
            result = describe_audit_log(AuditLogInput())

        assert result.audit_dir == str(legacy)
        assert result.audit_dir_is_legacy_cache is True
        assert any("SQT_AUDIT_DIR" in w for w in result.warnings)

    def test_a_trail_anywhere_else_is_not(self, _no_override, monkeypatch):
        from standard_quant_tools.agent.runtimes.meta.audit_tools import (
            AuditLogInput,
            describe_audit_log,
        )

        monkeypatch.setenv("SQT_AUDIT_DIR", str(_no_override / "durable"))
        result = describe_audit_log(AuditLogInput())

        assert result.audit_dir_is_legacy_cache is False
        assert not any("cache directory" in w for w in result.warnings)
