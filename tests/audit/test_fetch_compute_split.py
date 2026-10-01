"""
Where a call's time went: fetching market data, or computing on it.

`duration_ms` is end to end, so a Hurst analysis that took a minute could
not say whether the kernel or the vendor was slow, and every argument about
speed started by guessing. Each data source now carries its own `fetch_ms`
and the record carries the two sums: `fetch_ms` over its data sources and
`compute_ms`, the rest of `duration_ms`.

A provider reports a data access when it has the frame and says nothing
when it starts, so the audit times it as a lap: the time since the call
started or since its previous data access completed. The laps cover the
call up to its last data access; what follows is computation. A failed call
gets no `compute_ms`, because its last fetch may have failed without
reporting itself.

The fields are new, so the other half of this file is the null case: a day
written before they existed still verifies in both verifiers and under a
signed checkpoint, and a new record chains onto it.
"""

import contextvars
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

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


def _access(symbol: str = "AAPL", **extra: Any) -> None:
    audit.record_data_access(
        symbol, "2024-01-01", "2024-06-01", "1d", "live_fetch", "abc", **extra
    )


def _last(directory: Path) -> Dict[str, Any]:
    day = audit._iter_day_files(directory)[-1]
    return json.loads(day.read_text(encoding="utf-8").splitlines()[-1])


class TestTheSplit:
    def test_one_fetch_then_compute(self, audit_dir: Path):
        def tool(model):
            time.sleep(0.06)  # the vendor
            _access()
            time.sleep(0.04)  # the kernel
            return {"ok": True}

        _run_and_record("probe_tool", tool, _Probe())
        record = _last(audit_dir)
        (source,) = record["data_sources"]
        assert source["fetch_ms"] >= 60.0
        assert record["fetch_ms"] == source["fetch_ms"]
        assert record["compute_ms"] >= 40.0
        assert record["fetch_ms"] + record["compute_ms"] == pytest.approx(
            record["duration_ms"], abs=0.002
        )

    def test_each_source_carries_its_own_lap_and_they_sum(self, audit_dir: Path):
        def tool(model):
            time.sleep(0.03)
            _access("AAPL")
            time.sleep(0.05)
            _access("SPY")
            return {"ok": True}

        _run_and_record("probe_tool", tool, _Probe())
        record = _last(audit_dir)
        first, second = record["data_sources"]
        assert first["fetch_ms"] >= 30.0
        assert second["fetch_ms"] >= 50.0
        assert record["fetch_ms"] == pytest.approx(
            first["fetch_ms"] + second["fetch_ms"], abs=0.002
        )
        assert record["fetch_ms"] <= record["duration_ms"]
        assert record["compute_ms"] >= 0.0

    def test_a_provider_that_timed_its_fetch_is_believed(self, audit_dir: Path):
        def tool(model):
            _access(fetch_ms=1234.5)
            return {"ok": True}

        _run_and_record("probe_tool", tool, _Probe())
        record = _last(audit_dir)
        assert record["data_sources"][0]["fetch_ms"] == 1234.5
        assert record["fetch_ms"] == 1234.5
        # More than the call took: the floor keeps compute from going
        # negative.
        assert record["compute_ms"] == 0.0

    def test_concurrent_fetches_share_one_clock(self, audit_dir: Path):
        """A context copied into a worker thread carries the same clock,
        so the laps still partition the call instead of each counting the
        whole shared wait."""

        def fetch(symbol: str) -> None:
            time.sleep(0.05)
            _access(symbol)

        def tool(model):
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [
                    pool.submit(contextvars.copy_context().run, fetch, s)
                    for s in ("AAPL", "NVDA", "SPY")
                ]
                for future in futures:
                    future.result()
            return {"ok": True}

        _run_and_record("probe_tool", tool, _Probe())
        record = _last(audit_dir)
        assert len(record["data_sources"]) == 3
        assert record["fetch_ms"] >= 50.0
        assert record["fetch_ms"] <= record["duration_ms"]

    def test_a_call_that_fetches_nothing_is_all_compute(self, audit_dir: Path):
        """Null case."""
        _run_and_record("probe_tool", lambda model: {"ok": True}, _Probe())
        record = _last(audit_dir)
        assert record["data_sources"] == []
        assert record["fetch_ms"] == 0.0
        assert record["compute_ms"] == record["duration_ms"]

    def test_a_failed_call_is_not_split(self, audit_dir: Path):
        """The time after its last completed access may be a fetch that
        failed without reporting itself, so it is not called compute."""

        def tool(model):
            time.sleep(0.02)
            _access("AAPL")
            raise RuntimeError("the second fetch never came back")

        with pytest.raises(RuntimeError):
            _run_and_record("probe_tool", tool, _Probe())
        record = _last(audit_dir)
        assert record["status"] == "error"
        assert record["fetch_ms"] == record["data_sources"][0]["fetch_ms"]
        assert record["compute_ms"] is None

    def test_outside_a_record_nothing_is_timed(self):
        """A provider called directly has no record to report into."""
        _access()  # must not raise
        assert audit.context._fetch_clock_var.get() is None

    def test_the_clock_does_not_outlive_its_call(self, audit_dir: Path):
        _run_and_record("probe_tool", lambda model: {"ok": True}, _Probe())
        assert audit.context._fetch_clock_var.get() is None


# ── Days written before the fields ───────────────────────────────────────────

_NEW_FIELDS = {"fetch_ms", "compute_ms", "native_isa", "output_hash_rounded"}


def _record(request_id: str) -> "audit.DecisionRecord":
    return audit.DecisionRecord(
        request_id=request_id,
        timestamp_utc=f"{DATE}T00:00:00+00:00",
        tool_name="probe",
        input={"value": 1},
        data_sources=[
            {
                "symbol": "AAPL",
                "start": "2024-01-01",
                "end": "2024-06-01",
                "interval": "1d",
                "source": "disk_cache",
                "content_hash": "abc",
            }
        ],
        cpp_available=True,
        duration_ms=1.0,
        status="ok",
    )


def _write_before_the_fields(day: Path, records: List["audit.DecisionRecord"]) -> None:
    """Append records as a writer that knew none of the new fields did:
    chained, hashed over the line as written, and without the keys."""
    prev = AuditWriter(audit_dir=day.parent)._bootstrap_new_day(day)
    with open(day, "a", encoding="utf-8") as handle:
        for record in records:
            record.prev_record_hash = prev
            payload = json.loads(
                record.model_dump_json(exclude={"record_hash"} | _NEW_FIELDS)
            )
            record_hash = audit.hash_payload({**payload, "record_hash": None})
            handle.write(json.dumps({**payload, "record_hash": record_hash}) + "\n")
            prev = record_hash


def _new_record(request_id: str) -> "audit.DecisionRecord":
    record = _record(request_id)
    record.data_sources[0]["fetch_ms"] = 12.5
    record.fetch_ms = 12.5
    record.compute_ms = 0.0
    record.native_isa = "avx2+fma"
    record.output_hash_rounded = "0123456789abcdef"
    return record


class TestADayWrittenBeforeTheFieldsStillVerifies:
    def test_old_records_verify_and_a_new_one_chains_onto_them(
        self, audit_dir: Path, standalone, monkeypatch: pytest.MonkeyPatch
    ):
        audit_dir.mkdir(parents=True)
        day = audit_dir / f"{DATE}.jsonl"
        _write_before_the_fields(day, [_record("old-1"), _record("old-2")])
        assert audit.verify_audit_trail_integrity(audit_dir) == []
        assert standalone.verify_trail(audit_dir) == []

        monkeypatch.setattr(
            AuditWriter, "_path_for", lambda self, when: self._dir / f"{DATE}.jsonl"
        )
        AuditWriter().write(_new_record("new-1"))

        lines = [json.loads(x) for x in day.read_text(encoding="utf-8").splitlines()]
        assert [_NEW_FIELDS <= set(x) for x in lines] == [False, False, True]
        assert lines[2]["prev_record_hash"] == lines[1]["record_hash"]
        assert lines[2]["data_sources"][0]["fetch_ms"] == 12.5
        assert audit.verify_audit_trail_integrity(audit_dir) == []
        assert standalone.verify_trail(audit_dir) == []

    def test_an_edit_to_a_new_field_is_still_caught(
        self, audit_dir: Path, standalone, monkeypatch: pytest.MonkeyPatch
    ):
        """The new fields are inside the hash like every other: moving a
        fetch time breaks the chain in both verifiers."""
        monkeypatch.setattr(
            AuditWriter, "_path_for", lambda self, when: self._dir / f"{DATE}.jsonl"
        )
        AuditWriter().write(_new_record("new-1"))
        day = audit_dir / f"{DATE}.jsonl"
        line = json.loads(day.read_text(encoding="utf-8"))
        line["compute_ms"] = 999.0
        day.write_text(json.dumps(line) + "\n", encoding="utf-8")
        assert audit.verify_audit_trail_integrity(audit_dir)
        assert standalone.verify_trail(audit_dir)

    def test_an_old_record_reads_back_unsplit(self, tmp_path: Path):
        day = tmp_path / f"{DATE}.jsonl"
        _write_before_the_fields(day, [_record("old-1")])
        reread = audit.DecisionRecord(**json.loads(day.read_text(encoding="utf-8")))
        assert reread.fetch_ms is None and reread.compute_ms is None
        assert reread.native_isa is None and reread.output_hash_rounded is None


@pytest.mark.skipif(not audit.HAS_CRYPTOGRAPHY, reason="cryptography is not installed")
class TestASignedCheckpointStillHolds:
    def test_a_day_signed_before_the_fields_is_extended_by_a_new_record(
        self, audit_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        private_bytes, public_bytes = audit.generate_keypair()
        private_path = tmp_path / "signing.private"
        public_path = tmp_path / "signing.public"
        private_path.write_bytes(private_bytes)
        public_path.write_bytes(public_bytes)

        audit_dir.mkdir(parents=True)
        _write_before_the_fields(
            audit_dir / f"{DATE}.jsonl", [_record("old-1"), _record("old-2")]
        )
        audit.checkpoint_and_sign(DATE, audit_dir=audit_dir, key_path=private_path)
        assert (
            audit.verify_checkpoint_state(DATE, public_path, audit_dir=audit_dir)
            == "valid"
        )

        monkeypatch.setattr(
            AuditWriter, "_path_for", lambda self, when: self._dir / f"{DATE}.jsonl"
        )
        AuditWriter().write(_new_record("new-1"))
        found = audit.verify_checkpoint(DATE, public_path, audit_dir=audit_dir)
        assert found.state == "extended"
        assert (found.records_signed, found.records_after) == (2, 1)
