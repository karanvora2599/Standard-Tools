"""
A decision record's hash is taken over exactly what is written and re-read.

The writer used to hash a record's live values and then write pydantic's
JSON of them, while the verifier hashes the line it reads back. Wherever
the two spell a value differently the stored hash can never be reproduced:
a NaN was hashed as NaN and written as null, a timestamp gained a `T`, a
set was hashed sorted and written unsorted, an integer-keyed mapping sorted
its keys two ways. The day then reported "content altered" for ever, on a
call nobody touched -- usually one the tool had correctly refused. A numpy
value was worse: the JSON writer raised, the write failed open, and the
call left no record at all.

Also pinned here: a day already written by that writer keeps verifying as
before, and a line it wrote for a non-finite input is recognised for what
it is -- by restoring the value and reproducing the stored hash -- without
letting a real edit of the same line through. See the CHANGELOG entry of
2026-09-27.
"""

import datetime as dt
import enum
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from standard_quant_tools import audit, cli
from standard_quant_tools.audit.dispatch import _run_and_record
from standard_quant_tools.audit.json_native import to_json_native
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
    return directory


def _record(value: Any, request_id: str = "r1") -> "audit.DecisionRecord":
    return audit.DecisionRecord(
        request_id=request_id,
        timestamp_utc=f"{DATE}T00:00:00+00:00",
        tool_name="probe",
        input={"value": value},
        cpp_available=False,
        duration_ms=1.0,
        status="ok",
    )


def _old_rule_hash(record: "audit.DecisionRecord") -> str:
    """How the writer hashed a record before: over the live values."""
    return audit.hash_payload(
        {**record.model_dump(exclude={"record_hash"}), "record_hash": None}
    )


def _write_the_old_way(day: Path, records: List["audit.DecisionRecord"]) -> None:
    """Append records exactly as the previous writer did: chained, hashed
    over the live values, written as pydantic's JSON."""
    prev = AuditWriter(audit_dir=day.parent)._bootstrap_new_day(day)
    with open(day, "a", encoding="utf-8") as f:
        for record in records:
            record.prev_record_hash = prev
            record.record_hash = _old_rule_hash(record)
            f.write(record.model_dump_json() + "\n")
            prev = record.record_hash


class _Colour(enum.Enum):
    RED = 1


# Every value here broke the old rule: the line on disk verified as altered,
# or the write raised and no record was left.
_VALUES_THE_OLD_RULE_BROKE = {
    "nan in a list": [1.0, float("nan")],
    "infinity": float("inf"),
    "negative infinity": float("-inf"),
    "bytes": b"ab",
    "pandas timestamp": pd.Timestamp("2024-01-02 03:04:05"),
    "datetime": dt.datetime(2024, 1, 2, 3, 4, 5),
    "set": {"zeta", "alpha", "mid"},
    "integer keys": {2: "a", 10: "b"},
    "numpy nan": np.float64("nan"),
    "numpy array": np.array([1.0, np.nan]),
    "numpy integer": np.int64(3),
    "numpy datetimes": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
    "enum member": _Colour.RED,
}


class TestEveryRecordedValueVerifies:
    @pytest.mark.parametrize(
        "value",
        list(_VALUES_THE_OLD_RULE_BROKE.values()),
        ids=list(_VALUES_THE_OLD_RULE_BROKE),
    )
    def test_a_record_written_now_verifies_in_both_verifiers(
        self, tmp_path: Path, standalone, value
    ):
        path = AuditWriter(audit_dir=tmp_path).write(_record(value))

        assert audit.verify_audit_log_integrity(path) == []
        assert standalone.verify_log_file(path) == []
        assert audit.verify_audit_trail_integrity(tmp_path) == []
        assert standalone.verify_trail(tmp_path) == []

    def test_a_non_finite_value_is_recorded_as_a_token_replay_can_restore(
        self, tmp_path: Path
    ):
        """Recorded as null, a NaN call would replay as a different call:
        the replay rebuilds the input model from the record. The tokens are
        read back by pydantic as the values they name."""
        record = _record(None)
        record.input = {"xs": [1.0, float("nan")], "hi": float("inf")}
        record.input["lo"] = float("-inf")
        path = AuditWriter(audit_dir=tmp_path).write(record)

        written = json.loads(path.read_text(encoding="utf-8"))["input"]
        assert written == {"xs": [1.0, "NaN"], "hi": "Infinity", "lo": "-Infinity"}

        class Restored(BaseModel):
            xs: List[float]
            hi: float
            lo: float

        restored = Restored(**written)
        assert restored.xs[0] == 1.0 and math.isnan(restored.xs[1])
        assert restored.hi == float("inf") and restored.lo == float("-inf")

    def test_the_normaliser_leaves_json_native_values_alone(self):
        """Null case: the conversion is the identity on what JSON already
        is, and applying it twice changes nothing."""
        native = {
            "s": "é",
            "i": 2**60,
            "f": 0.1 + 0.2,
            "neg_zero": -0.0,
            "b": True,
            "n": None,
            "l": [1, [2, {"k": 3.5}]],
            "d": {},
        }
        assert to_json_native(native) == native
        once = to_json_native(_VALUES_THE_OLD_RULE_BROKE)
        assert to_json_native(once) == once


class TestExistingRecordsHashExactlyAsBefore:
    def test_a_json_native_record_hashes_bit_identically_under_both_rules(
        self, tmp_path: Path
    ):
        """The null case the whole change rests on: for a record made of
        JSON-native values the parsed form of the line equals the live
        values, so the new rule gives the old hash, bit for bit."""
        record = audit.DecisionRecord(
            request_id="de06f2e1b7db47d7938069970bdc10ab",
            timestamp_utc="2026-07-19T16:19:37.695789+00:00",
            tool_name="run_sma_backtest",
            input={
                "symbol": "AAPL",
                "parameters": {"fast_period": 10, "slow_period": 50},
                "weights": [0.1, 0.2, 0.30000000000000004, 1e-17, 1e16],
                "note": "naïve — ünïcode",
                "flag": False,
                "missing": None,
            },
            data_sources=[
                {
                    "symbol": "AAPL",
                    "start": "2022-01-01",
                    "end": "2022-06-01",
                    "source": "live_fetch",
                    "content_hash": "1d975f555f10aeb8",
                }
            ],
            cpp_available=True,
            n_workers=4,
            duration_ms=6765.812,
            output_hash="8a2b0ca80ac84ba1",
            status="ok",
            git_commit_sha="463b874696913a8ec813c9a789465a443b66a15b",
            package_version="0.1.0",
            random_seed=42,
        )
        path = AuditWriter(audit_dir=tmp_path).write(record)

        assert record.record_hash == _old_rule_hash(record)
        reread = audit.DecisionRecord(**json.loads(path.read_text(encoding="utf-8")))
        assert _old_rule_hash(reread) == record.record_hash

    def test_a_day_written_the_old_way_verifies_and_the_chain_continues(
        self, audit_dir: Path, standalone, monkeypatch: pytest.MonkeyPatch
    ):
        audit_dir.mkdir(parents=True)
        day = audit_dir / f"{DATE}.jsonl"
        _write_the_old_way(day, [_record(1.5, "old-1"), _record("x", "old-2")])
        assert audit.verify_audit_trail_integrity(audit_dir) == []
        assert standalone.verify_trail(audit_dir) == []

        monkeypatch.setattr(
            AuditWriter, "_path_for", lambda self, when: self._dir / f"{DATE}.jsonl"
        )
        AuditWriter().write(_record(float("nan"), "new-1"))

        assert len(day.read_text(encoding="utf-8").splitlines()) == 3
        assert audit.verify_audit_trail_integrity(audit_dir) == []
        assert standalone.verify_trail(audit_dir) == []


class _FreeForm(BaseModel):
    payload: Dict[str, Any]


class TestDispatchRecordsWhatTheToolWasGiven:
    def test_a_numpy_array_in_a_free_form_input_leaves_a_record(self, audit_dir: Path):
        """The JSON writer could not serialise a numpy value, the write
        failed open, and the call left no record at all."""
        _run_and_record(
            "probe_tool",
            lambda model: {"ok": True},
            _FreeForm(payload={"a": np.array([1.0, np.nan]), "n": np.int64(3)}),
        )

        day_files = audit._iter_day_files(audit_dir)
        assert len(day_files) == 1
        lines = day_files[0].read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["input"] == {"payload": {"a": [1.0, "NaN"], "n": 3}}
        assert audit.verify_audit_trail_integrity(audit_dir) == []

    def test_a_refused_nan_input_leaves_a_day_that_verifies(self, audit_dir: Path):
        """The tool refuses the NaN, correctly, and the record of the
        refusal is written -- which used to mark the day tampered for
        ever."""
        from standard_quant_tools.agent.tools import dispatch
        from standard_quant_tools.error import ValidationError

        with pytest.raises(ValidationError, match="non-finite"):
            dispatch(
                "analyze_basis_history",
                {
                    "spot_prices": [100.0, float("nan"), 102.0],
                    "futures_prices": [101.0, 102.0, 103.0],
                },
            )

        day_files = audit._iter_day_files(audit_dir)
        assert day_files, "the refused call left no record"
        record = json.loads(day_files[-1].read_text(encoding="utf-8").splitlines()[-1])
        assert record["status"] == "error"
        assert record["input"]["spot_prices"] == [100.0, "NaN", 102.0]
        assert audit.verify_audit_trail_integrity(audit_dir) == []

    def test_an_ordinary_call_records_its_input_unchanged(self, audit_dir: Path):
        """Null case: a JSON-native input is recorded as given."""
        _run_and_record(
            "probe_tool",
            lambda model: {"ok": True},
            _FreeForm(payload={"symbol": "AAPL", "window": 20, "weights": [0.5, 0.5]}),
        )
        line = audit._iter_day_files(audit_dir)[0].read_text(encoding="utf-8")
        assert json.loads(line)["input"] == {
            "payload": {"symbol": "AAPL", "window": 20, "weights": [0.5, 0.5]}
        }


def _old_nan_line(directory: Path, value: Any) -> Path:
    """A day whose one record the previous writer wrote for a non-finite
    input: hashed over NaN, written with null."""
    directory.mkdir(parents=True, exist_ok=True)
    day = directory / f"{DATE}.jsonl"
    record = _record(None)
    record.input = value
    _write_the_old_way(day, [record])
    return day


def _edit_input(day: Path, edit) -> None:
    line = json.loads(day.read_text(encoding="utf-8").splitlines()[0])
    edit(line)
    day.write_text(json.dumps(line) + "\n", encoding="utf-8")


class TestALineWrittenForANonFiniteInputBeforeTheFix:
    def test_it_is_described_rather_than_reported_as_altered(self, tmp_path: Path):
        day = _old_nan_line(tmp_path, {"x": [1.0, float("nan"), 3.0]})
        assert json.loads(day.read_text(encoding="utf-8"))["input"]["x"][1] is None

        notes: List[str] = []
        assert audit.verify_audit_log_integrity(day, notes=notes) == []
        assert len(notes) == 1
        assert "input.x[1]=NaN" in notes[0]
        assert "not altered" in notes[0]

    @pytest.mark.parametrize(
        "edit",
        [
            lambda line: line["input"]["x"].__setitem__(0, None),  # 1.0 -> null
            lambda line: line["input"]["x"].__setitem__(2, 4.0),  # 3.0 -> 4.0
            lambda line: line.__setitem__("status", "error"),
        ],
        ids=["a value nulled", "a value changed", "another field changed"],
    )
    def test_a_real_edit_of_the_same_line_is_still_altered(self, tmp_path: Path, edit):
        day = _old_nan_line(tmp_path, {"x": [1.0, float("nan"), 3.0]})
        _edit_input(day, edit)

        notes: List[str] = []
        problems = audit.verify_audit_log_integrity(day, notes=notes)
        assert any("altered after it was written" in p for p in problems)
        assert notes == []

    def test_one_nan_among_many_unset_parameters_is_still_found(self, tmp_path: Path):
        """Optional parameters default to None, so a real input carries many
        nulls; restorations of one null are tried before two."""
        value: Dict[str, Any] = {f"optional_{i}": None for i in range(40)}
        value["spot"] = float("nan")
        day = _old_nan_line(tmp_path, value)

        notes: List[str] = []
        assert audit.verify_audit_log_integrity(day, notes=notes) == []
        assert "input.spot=NaN" in notes[0]

    def test_both_infinities_are_found_together(self, tmp_path: Path):
        day = _old_nan_line(
            tmp_path, {"hi": float("inf"), "lo": float("-inf"), "unset": None}
        )
        notes: List[str] = []
        assert audit.verify_audit_log_integrity(day, notes=notes) == []
        assert "input.hi=Infinity" in notes[0] and "input.lo=-Infinity" in notes[0]

    def test_sqt_verify_prints_the_note_and_passes(
        self, audit_dir: Path, capsys: pytest.CaptureFixture
    ):
        _old_nan_line(audit_dir, {"x": [float("nan")]})

        assert cli.main(["verify"]) == 0
        output = capsys.readouterr().out
        assert "OK" in output
        assert "note(s), not problems" in output
