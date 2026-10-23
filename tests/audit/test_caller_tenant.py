"""
Who a call was for: `audit.caller(tenant=...)` and the record's `tenant`.

A host serving several desks from one process and one audit directory had
no way to say which desk a call was made for. The record carried the call
and never the caller, so the host could not slice the chain by desk when it
exported a bundle: it either declared every desk it served or refused.

The host now declares the caller around its calls, and every record written
inside the block names it. Outside any block the record says None, and a
day written before the field existed still verifies with a stamped record
chained onto it.
"""

import contextvars
import json
import threading
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import BaseModel

from standard_quant_tools import audit
from standard_quant_tools.audit.dispatch import _run_and_record
from standard_quant_tools.audit.writer import AuditWriter

from .test_standalone_verifier import _load_standalone_module

DATE = "2024-05-01"


@pytest.fixture(scope="module")
def standalone():
    return _load_standalone_module()


@pytest.fixture
def audit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "audit"
    monkeypatch.setenv("SQT_AUDIT_DIR", str(directory))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
    return directory


class _Probe(BaseModel):
    payload: Dict[str, Any] = {}


def _ok(model):
    return {"ok": True}


def _records(directory: Path):
    day = audit._iter_day_files(directory)[-1]
    return [json.loads(x) for x in day.read_text(encoding="utf-8").splitlines()]


class TestTheStamp:
    def test_a_call_inside_the_block_names_its_tenant(self, audit_dir: Path):
        with audit.caller(tenant="desk-a"):
            _run_and_record("probe_tool", _ok, _Probe())
        (record,) = _records(audit_dir)
        assert record["tenant"] == "desk-a"
        assert audit.verify_audit_trail_integrity(audit_dir) == []

    def test_a_call_outside_any_block_names_none(self, audit_dir: Path):
        _run_and_record("probe_tool", _ok, _Probe())
        (record,) = _records(audit_dir)
        assert "tenant" in record and record["tenant"] is None

    def test_blocks_nest_and_restore(self, audit_dir: Path):
        with audit.caller(tenant="desk-a"):
            with audit.caller(tenant="desk-b"):
                _run_and_record("probe_tool", _ok, _Probe())
            _run_and_record("probe_tool", _ok, _Probe())
            with audit.caller(tenant=None):
                _run_and_record("probe_tool", _ok, _Probe())
        _run_and_record("probe_tool", _ok, _Probe())
        assert [r["tenant"] for r in _records(audit_dir)] == [
            "desk-b",
            "desk-a",
            None,
            None,
        ]

    def test_the_block_is_left_on_an_exception(self, audit_dir: Path):
        def boom(model):
            raise RuntimeError("the tool failed")

        with pytest.raises(RuntimeError):
            with audit.caller(tenant="desk-a"):
                _run_and_record("probe_tool", boom, _Probe())
        assert audit.current_tenant() is None
        (record,) = _records(audit_dir)
        assert record["status"] == "error" and record["tenant"] == "desk-a"

    def test_the_tenant_is_read_before_the_tool_runs(self, audit_dir: Path):
        """A tool that changes the context while it runs does not change
        whom its own record names."""

        def rebinds(model):
            audit.context._tenant_var.set("someone-else")
            return {"ok": True}

        with audit.caller(tenant="desk-a"):
            contextvars.copy_context().run(
                _run_and_record, "probe_tool", rebinds, _Probe()
            )
        (record,) = _records(audit_dir)
        assert record["tenant"] == "desk-a"

    def test_a_nested_dispatch_inherits_the_tenant(self, audit_dir: Path):
        def outer(model):
            _run_and_record("inner_tool", _ok, _Probe())
            return {"ok": True}

        with audit.caller(tenant="desk-a"):
            _run_and_record("outer_tool", outer, _Probe())
        assert {r["tool_name"]: r["tenant"] for r in _records(audit_dir)} == {
            "inner_tool": "desk-a",
            "outer_tool": "desk-a",
        }

    def test_another_thread_does_not_see_the_block(self, audit_dir: Path):
        """A context variable, not a global: a desk's block on one thread
        never stamps a call another thread makes at the same time."""
        inside = threading.Event()
        done = threading.Event()

        def other_thread():
            inside.wait(5)
            _run_and_record("other_tool", _ok, _Probe())
            done.set()

        worker = threading.Thread(target=other_thread)
        worker.start()
        with audit.caller(tenant="desk-a"):
            inside.set()
            assert done.wait(5)
        worker.join(5)
        (record,) = _records(audit_dir)
        assert record["tool_name"] == "other_tool" and record["tenant"] is None

    @pytest.mark.parametrize("bad", ["", "   ", 7, b"desk-a"])
    def test_a_blank_or_non_string_tenant_is_refused(self, bad, audit_dir: Path):
        with pytest.raises(ValueError, match="tenant must be"):
            with audit.caller(tenant=bad):
                pass
        assert audit.current_tenant() is None

    def test_the_tenant_is_inside_the_hash(self, audit_dir: Path, standalone):
        """Moving a record to another desk breaks the chain in both
        verifiers, like an edit to any other field."""
        with audit.caller(tenant="desk-a"):
            _run_and_record("probe_tool", _ok, _Probe())
        day = audit._iter_day_files(audit_dir)[-1]
        line = json.loads(day.read_text(encoding="utf-8"))
        line["tenant"] = "desk-b"
        day.write_text(json.dumps(line) + "\n", encoding="utf-8")
        assert audit.verify_audit_trail_integrity(audit_dir)
        assert standalone.verify_trail(audit_dir)


def _record(request_id: str, tenant=None) -> "audit.DecisionRecord":
    return audit.DecisionRecord(
        request_id=request_id,
        timestamp_utc=f"{DATE}T00:00:00+00:00",
        tool_name="probe",
        tenant=tenant,
        input={"value": 1},
        cpp_available=False,
        duration_ms=1.0,
        status="ok",
    )


class TestADayWrittenBeforeTheFieldStillVerifies:
    def test_old_records_verify_and_a_stamped_one_chains_onto_them(
        self, audit_dir: Path, standalone, monkeypatch: pytest.MonkeyPatch
    ):
        audit_dir.mkdir(parents=True)
        day = audit_dir / f"{DATE}.jsonl"
        prev = AuditWriter(audit_dir=audit_dir)._bootstrap_new_day(day)
        with open(day, "a", encoding="utf-8") as handle:
            for request_id in ("old-1", "old-2"):
                record = _record(request_id)
                record.prev_record_hash = prev
                payload = json.loads(
                    record.model_dump_json(exclude={"record_hash", "tenant"})
                )
                prev = audit.hash_payload({**payload, "record_hash": None})
                handle.write(json.dumps({**payload, "record_hash": prev}) + "\n")
        assert audit.verify_audit_trail_integrity(audit_dir) == []

        monkeypatch.setattr(
            AuditWriter, "_path_for", lambda self, when: self._dir / f"{DATE}.jsonl"
        )
        AuditWriter().write(_record("new-1", tenant="desk-a"))

        lines = _records(audit_dir)
        assert ["tenant" in x for x in lines] == [False, False, True]
        assert lines[2]["prev_record_hash"] == lines[1]["record_hash"]
        assert audit.verify_audit_trail_integrity(audit_dir) == []
        assert standalone.verify_trail(audit_dir) == []
        reread = audit.DecisionRecord(**lines[0])
        assert reread.tenant is None


def test_explain_decision_names_the_tenant(audit_dir: Path):
    from standard_quant_tools.agent import dispatch

    with audit.caller(tenant="desk-a"):
        _run_and_record("probe_tool", _ok, _Probe())
    (record,) = _records(audit_dir)
    explained = dispatch("explain_decision", {"request_id": record["request_id"]})
    assert explained["tenant"] == "desk-a"
