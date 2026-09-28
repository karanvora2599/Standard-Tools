"""
A sidecar another process is replacing is waited for, not called damaged.

A sidecar is replaced with `os.replace`, and on Windows a read that meets
the replace -- or a writer holding the file open -- fails with access
denied or a sharing violation for a few milliseconds. The read reported
that as a damaged record and told the caller to publish again under a fresh
name. It is retried now with the bounds `write_bytes_atomically` uses for
its rename, and a refusal that outlasts them is named for what it is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.error import ValidationError


@pytest.fixture
def runs(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv("SQT_RUNS_DIR", str(root))
    return root


@pytest.fixture
def ref(runs):
    curve = pd.Series(
        [3.0] * 20, index=pd.date_range("2025-01-01", periods=20), name="equity"
    )
    return handoff.publish(curve, "equity_curve", "run1", "curve")


def _refusal(winerror):
    error = PermissionError(13, "The process cannot access the file")
    if winerror is not None:
        error.winerror = winerror
    return error


def _sidecar_refused(monkeypatch, times: int, winerror=32):
    """Make reading the sidecar fail `times` times; returns the attempt log."""
    real = Path.read_text
    attempts = []

    def read_text(self, *args, **kwargs):
        if self.name.endswith("._handoff.json"):
            attempts.append(self.name)
            if len(attempts) <= times:
                raise _refusal(winerror)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(handoff, "_FIRST_REPLACE_WAIT", 0.0001)
    return attempts


class TestABriefRefusalIsWaitedOut:
    @pytest.mark.parametrize("winerror", [32, 5], ids=["sharing", "access"])
    def test_the_read_is_retried_and_the_record_read(self, ref, monkeypatch, winerror):
        attempts = _sidecar_refused(monkeypatch, times=2, winerror=winerror)
        assert handoff.describe(ref)["storage"] == "local"
        assert len(attempts) == 3

    def test_resolving_waits_too(self, ref, monkeypatch):
        attempts = _sidecar_refused(monkeypatch, times=1)
        assert handoff.resolve(ref, expect="equity_curve").iloc[0] == 3.0
        assert len(attempts) == 2

    def test_a_refusal_that_persists_is_not_called_damage(self, ref, monkeypatch):
        attempts = _sidecar_refused(monkeypatch, times=10**6)
        with pytest.raises(ValidationError) as refused:
            handoff.describe(ref)
        message = str(refused.value)
        assert "could not be read" in message and "retry" in message
        assert "is damaged" not in message and "fresh run_id" not in message
        assert len(attempts) == handoff._REPLACE_ATTEMPTS


class TestEverythingElseIsAsBefore:
    def test_a_permission_error_that_is_not_transient_is_not_retried(
        self, ref, monkeypatch
    ):
        """Null case: with no Windows error code -- a real lack of
        permission -- the read fails once and is reported as before."""
        attempts = _sidecar_refused(monkeypatch, times=10**6, winerror=None)
        with pytest.raises(ValidationError, match="is damaged"):
            handoff.describe(ref)
        assert len(attempts) == 1

    def test_damaged_content_is_still_damage(self, ref, runs):
        """Null case: bytes that are not a record are refused as before."""
        (runs / "run1" / "curve._handoff.json").write_bytes(b"{not json")
        with pytest.raises(ValidationError, match="is damaged"):
            handoff.describe(ref)
