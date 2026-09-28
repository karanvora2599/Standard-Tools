"""
A decision record says WHICH compiled build ran, not only that one did.

`cpp_available` was true for an extension weeks older than the Python
calling it, so a record computed by old kernels read exactly like one
computed by current ones. Each record now carries `native_build`: the
import-time verdict and the short digest of the sources the extension was
built from ("match:df27c6e4af54"), or why no extension ran ("stale:…",
"absent", "disabled").

The field is new, so the other half of this file is the null case: a day
written before it existed still verifies in both verifiers, and a new
record chains onto it.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel

from standard_quant_tools import _native_build as nb
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
    payload: Dict[str, Any]


def _lines(directory: Path) -> List[Dict[str, Any]]:
    day = audit._iter_day_files(directory)[-1]
    return [json.loads(line) for line in day.read_text(encoding="utf-8").splitlines()]


class TestANewRecordNamesTheBuild:
    def test_it_carries_the_import_time_verdict(self, audit_dir: Path, standalone):
        _run_and_record("probe_tool", lambda model: {"ok": True}, _Probe(payload={}))

        record = _lines(audit_dir)[-1]
        assert record["native_build"] == nb.native_build_status().label
        assert audit.verify_audit_trail_integrity(audit_dir) == []
        assert standalone.verify_trail(audit_dir) == []

    def test_a_refused_build_is_recorded_as_refused(
        self, audit_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The case the field exists for: an extension was present, built
        from other sources, and the Python path ran instead."""
        stale = nb.NativeBuildStatus(
            nb.STALE, used=False, built_digest="0a1b2c3d4e5f" + "0" * 52
        )
        monkeypatch.setattr(nb, "_status", stale)
        _run_and_record("probe_tool", lambda model: {"ok": True}, _Probe(payload={}))
        assert _lines(audit_dir)[-1]["native_build"] == "stale:0a1b2c3d4e5f"

    def test_no_extension_is_recorded_as_the_verdict_alone(
        self, audit_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(nb, "_status", nb.NativeBuildStatus(nb.ABSENT))
        _run_and_record("probe_tool", lambda model: {"ok": True}, _Probe(payload={}))
        assert _lines(audit_dir)[-1]["native_build"] == "absent"


def _write_before_the_field(day: Path, records: List["audit.DecisionRecord"]) -> None:
    """Append records as a writer that did not know `native_build` did:
    chained, hashed over the line as written, and without the key."""
    prev = AuditWriter(audit_dir=day.parent)._bootstrap_new_day(day)
    with open(day, "a", encoding="utf-8") as handle:
        for record in records:
            record.prev_record_hash = prev
            payload = json.loads(
                record.model_dump_json(exclude={"record_hash", "native_build"})
            )
            record_hash = audit.hash_payload({**payload, "record_hash": None})
            handle.write(json.dumps({**payload, "record_hash": record_hash}) + "\n")
            prev = record_hash


def _record(request_id: str) -> "audit.DecisionRecord":
    return audit.DecisionRecord(
        request_id=request_id,
        timestamp_utc=f"{DATE}T00:00:00+00:00",
        tool_name="probe",
        input={"value": 1},
        cpp_available=True,
        duration_ms=1.0,
        status="ok",
    )


class TestADayWrittenBeforeTheFieldStillVerifies:
    def test_old_records_verify_and_a_new_one_chains_onto_them(
        self, audit_dir: Path, standalone, monkeypatch: pytest.MonkeyPatch
    ):
        audit_dir.mkdir(parents=True)
        day = audit_dir / f"{DATE}.jsonl"
        _write_before_the_field(day, [_record("old-1"), _record("old-2")])
        assert audit.verify_audit_trail_integrity(audit_dir) == []
        assert standalone.verify_trail(audit_dir) == []

        monkeypatch.setattr(
            AuditWriter, "_path_for", lambda self, when: self._dir / f"{DATE}.jsonl"
        )
        new = _record("new-1")
        new.native_build = "match:df27c6e4af54"
        AuditWriter().write(new)

        lines = [json.loads(x) for x in day.read_text(encoding="utf-8").splitlines()]
        assert [("native_build" in x) for x in lines] == [False, False, True]
        assert lines[2]["prev_record_hash"] == lines[1]["record_hash"]
        assert audit.verify_audit_trail_integrity(audit_dir) == []
        assert standalone.verify_trail(audit_dir) == []

    def test_an_old_record_reads_back_with_no_build_named(self, tmp_path: Path):
        day = tmp_path / f"{DATE}.jsonl"
        _write_before_the_field(day, [_record("old-1")])
        reread = audit.DecisionRecord(**json.loads(day.read_text(encoding="utf-8")))
        assert reread.native_build is None
