"""
`last_request_id()` names a record that exists, and a dispatch leaves the
request context as it found it on every way out.

The id was set when it was minted, before anything was written, and never
cleared -- so after a call whose arguments failed validation it still named
the PREVIOUS call's record, and `explain_decision` or `replay_decision`
described a different call. With recording off, or a write that failed, it
named a record that was never written. The context variables were reset
only after a successful write, so an audit failure left a dead request id
stamped on every later log line. See the CHANGELOG entry of 2026-09-28.
"""

import json

import pytest

from standard_quant_tools import audit
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.audit.context import _data_sources_var, _request_id_var
from standard_quant_tools.audit.writer import AuditWriter
from standard_quant_tools.error import AuditIntegrityError

CALL = ("list_strategies", {"strategy_type": "sma_crossover"})


@pytest.fixture(autouse=True)
def audit_dir(tmp_path, monkeypatch):
    directory = tmp_path / "audit"
    monkeypatch.setenv("SQT_AUDIT_DIR", str(directory))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
    monkeypatch.delenv("SQT_AUDIT_FAIL_CLOSED", raising=False)
    return directory


def _last_record(directory):
    day = audit._iter_day_files(directory)[-1]
    return json.loads(day.read_text(encoding="utf-8").splitlines()[-1])


class TestTheIdNamesAWrittenRecord:
    def test_a_successful_call_names_its_own_record(self, audit_dir):
        """Null case."""
        dispatch(*CALL)
        assert audit.last_request_id() == _last_record(audit_dir)["request_id"]

    def test_arguments_the_tool_refuses_leave_no_id(self):
        dispatch(*CALL)
        assert audit.last_request_id() is not None
        with pytest.raises(Exception):
            dispatch("list_strategies", {"no_such_argument": 1})
        assert audit.last_request_id() is None

    def test_an_unknown_tool_leaves_no_id(self):
        dispatch(*CALL)
        with pytest.raises(ValueError):
            dispatch("no_such_tool", {})
        assert audit.last_request_id() is None

    def test_a_runtime_dispatch_refused_before_it_runs_leaves_no_id(self):
        from standard_quant_tools.agent.runtimes import all_runtimes

        runtime = all_runtimes()["meta"]
        runtime.dispatch(*CALL)
        assert audit.last_request_id() is not None
        with pytest.raises(ValueError):
            runtime.dispatch("no_such_tool", {})
        assert audit.last_request_id() is None

    def test_a_modeling_dispatch_refused_before_it_runs_leaves_no_id(self):
        from standard_quant_tools.modeling.agent.dispatch import modeling_dispatch

        dispatch(*CALL)
        with pytest.raises(ValueError):
            modeling_dispatch("no_such_tool", {})
        assert audit.last_request_id() is None

    def test_recording_off_leaves_no_id(self, monkeypatch):
        dispatch(*CALL)
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "0")
        dispatch(*CALL)
        assert audit.last_request_id() is None

    def test_a_write_that_failed_open_leaves_no_id(self, monkeypatch):
        def _disk_full(self, record):
            raise OSError("no space left on device")

        monkeypatch.setattr(AuditWriter, "write", _disk_full)
        dispatch(*CALL)
        assert audit.last_request_id() is None


class TestTheContextIsResetOnEveryWayOut:
    def test_after_a_corrupted_chain_refuses_the_write(self, monkeypatch):
        def _corrupt(self, record):
            raise AuditIntegrityError("audit chain is corrupt")

        monkeypatch.setattr(AuditWriter, "write", _corrupt)
        with pytest.raises(AuditIntegrityError):
            dispatch(*CALL)
        assert _request_id_var.get() is None
        assert _data_sources_var.get() is None

    def test_after_a_fail_closed_write_failure(self, monkeypatch):
        monkeypatch.setenv("SQT_AUDIT_FAIL_CLOSED", "1")

        def _disk_full(self, record):
            raise OSError("no space left on device")

        monkeypatch.setattr(AuditWriter, "write", _disk_full)
        with pytest.raises(OSError):
            dispatch(*CALL)
        assert _request_id_var.get() is None
        assert _data_sources_var.get() is None

    def test_after_an_ordinary_call(self):
        """Null case."""
        dispatch(*CALL)
        assert _request_id_var.get() is None
        assert _data_sources_var.get() is None
