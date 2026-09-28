"""
Parity test for scripts/verify_audit_log.py against the real
standard_quant_tools.audit module.

scripts/verify_audit_log.py is a deliberate, stdlib-only reimplementation of
audit.py's hash_payload / verify_audit_log_integrity /
verify_audit_trail_integrity, so an external auditor can verify a log bundle
without installing the project. That's a known duplication-by-design
maintenance risk: if audit.py's hash_payload canonicalization ever changes
and this file isn't updated to match, the two implementations would
silently disagree about what counts as tampered. This test is what catches
that drift -- it must be run (and pass) any time audit.py's hashing changes.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from standard_quant_tools import audit

from .. import REPO_ROOT

_SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_audit_log.py"


def _load_standalone_module() -> ModuleType:
    """Load scripts/verify_audit_log.py as a module without adding it to
    sys.modules permanently or requiring scripts/ to be a package."""
    spec = importlib.util.spec_from_file_location(
        "verify_audit_log_standalone", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def standalone():
    return _load_standalone_module()


class TestHashPayloadParity:
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"a": 1},
            {"b": 1, "a": 2},  # key order must not matter, in both implementations
            {"a": None, "b": [1, 2, 3], "c": {"nested": True}},
            {"float": 1.23456789, "negative": -1, "zero": 0},
            [1, 2, 3],
            "a bare string",
            None,
            # A realistic DecisionRecord shape, the actual real-world payload
            # both implementations hash in production.
            {
                "request_id": "abc123",
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "tool_name": "analyze_stock_risk",
                "input": {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"},
                "data_sources": [
                    {
                        "symbol": "AAPL",
                        "start": "2022-01-01",
                        "end": "2023-01-01",
                        "interval": "1d",
                        "source": "live_fetch",
                        "content_hash": "deadbeef",
                    }
                ],
                "cpp_available": False,
                "n_workers": None,
                "duration_ms": 12.345,
                "output_hash": "cafef00d",
                "status": "ok",
                "error_type": None,
                "error_message": None,
                "git_commit_sha": "0123456789abcdef",
                "package_version": "0.1.0",
                "random_seed": 42,
                "strategy_source_hash": None,
                "prev_record_hash": "0" * 16,
                "record_hash": None,
            },
        ],
    )
    def test_hash_payload_matches_real_implementation(self, standalone, payload):
        assert standalone.hash_payload(payload) == audit.hash_payload(payload)

    def test_genesis_hash_constant_matches(self, standalone):
        assert standalone._GENESIS_HASH == audit._GENESIS_HASH

    def test_day_file_regex_matches(self, standalone):
        for name in ["2024-01-01.jsonl", "_chain_index.jsonl", "2024-01-01.jsonl.lock"]:
            assert bool(standalone._DAY_FILE_RE.match(name)) == bool(
                audit._DAY_FILE_RE.match(name)
            )


class TestStandaloneVerifierEndToEnd:
    """Confirms the standalone script's verify_log_file/verify_trail agree
    with the real library's verify_audit_log_integrity/
    verify_audit_trail_integrity against real, dispatch()-produced data --
    not just that the hash function matches in isolation."""

    def test_clean_trail_agrees(self, standalone, tmp_path: Path):
        w = audit.AuditWriter(audit_dir=tmp_path)
        day1 = tmp_path / "2024-01-01.jsonl"
        head1 = w._bootstrap_new_day(day1)
        r1 = audit.DecisionRecord(
            request_id="r1",
            timestamp_utc="2024-01-01T00:00:00+00:00",
            tool_name="t1",
            input={},
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        r1.prev_record_hash = head1
        r1.record_hash = audit.hash_payload(
            {**r1.model_dump(exclude={"record_hash"}), "record_hash": None}
        )
        day1.write_text(r1.model_dump_json() + "\n", encoding="utf-8")

        assert audit.verify_audit_trail_integrity(tmp_path) == []
        assert standalone.verify_trail(tmp_path) == []

    def test_tampered_record_detected_by_both(self, standalone, tmp_path: Path):
        w = audit.AuditWriter(audit_dir=tmp_path)
        day1 = tmp_path / "2024-01-01.jsonl"
        head1 = w._bootstrap_new_day(day1)
        r1 = audit.DecisionRecord(
            request_id="r1",
            timestamp_utc="2024-01-01T00:00:00+00:00",
            tool_name="t1",
            input={},
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        r1.prev_record_hash = head1
        r1.record_hash = audit.hash_payload(
            {**r1.model_dump(exclude={"record_hash"}), "record_hash": None}
        )
        day1.write_text(r1.model_dump_json() + "\n", encoding="utf-8")

        record = json.loads(day1.read_text(encoding="utf-8").splitlines()[0])
        record["status"] = "tampered"
        day1.write_text(json.dumps(record) + "\n", encoding="utf-8")

        real_problems = audit.verify_audit_trail_integrity(tmp_path)
        standalone_problems = standalone.verify_trail(tmp_path)
        assert real_problems and standalone_problems
        assert len(real_problems) == len(standalone_problems)

    def test_cli_entrypoint_exits_zero_on_clean_trail(self, standalone, tmp_path: Path):
        w = audit.AuditWriter(audit_dir=tmp_path)
        day1 = tmp_path / "2024-01-01.jsonl"
        head1 = w._bootstrap_new_day(day1)
        r1 = audit.DecisionRecord(
            request_id="r1",
            timestamp_utc="2024-01-01T00:00:00+00:00",
            tool_name="t1",
            input={},
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        r1.prev_record_hash = head1
        r1.record_hash = audit.hash_payload(
            {**r1.model_dump(exclude={"record_hash"}), "record_hash": None}
        )
        day1.write_text(r1.model_dump_json() + "\n", encoding="utf-8")

        assert standalone.main([str(tmp_path)]) == 0

    def test_cli_entrypoint_exits_nonzero_without_args(self, standalone):
        with pytest.raises(SystemExit):
            standalone.main([])


def _old_writer_day(directory: Path, value) -> Path:
    """One record as the previous writer wrote it for this input: hashed
    over the live values, written as pydantic's JSON (a NaN becomes null)."""
    w = audit.AuditWriter(audit_dir=directory)
    day = directory / "2024-01-01.jsonl"
    record = audit.DecisionRecord(
        request_id="r1",
        timestamp_utc="2024-01-01T00:00:00+00:00",
        tool_name="t1",
        input=value,
        cpp_available=False,
        duration_ms=1.0,
        status="ok",
    )
    record.prev_record_hash = w._bootstrap_new_day(day)
    record.record_hash = audit.hash_payload(
        {**record.model_dump(exclude={"record_hash"}), "record_hash": None}
    )
    day.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    return day


class TestBothVerifiersReadAnOldNonFiniteLineAlike:
    """The library and the auditor's copy must accept exactly the same
    lines as written-before-the-fix, with the same words, and refuse the
    same edits -- the explanation is bounded, and a bound applied in one
    copy only would make them disagree about a line with many nulls."""

    def test_the_same_note_and_no_problem(self, standalone, tmp_path: Path):
        value = {f"unset_{i}": None for i in range(12)}
        value["spot"] = [100.0, float("nan"), float("inf")]
        _old_writer_day(tmp_path, value)

        real_notes: list = []
        standalone_notes: list = []
        assert audit.verify_audit_trail_integrity(tmp_path, notes=real_notes) == []
        assert standalone.verify_trail(tmp_path, notes=standalone_notes) == []
        assert real_notes and real_notes == standalone_notes

    @pytest.mark.parametrize(
        "edit",
        [
            lambda line: line["input"]["spot"].__setitem__(0, None),
            lambda line: line["input"]["spot"].__setitem__(0, 101.0),
            lambda line: line.__setitem__("tool_name", "t2"),
        ],
        ids=["a value nulled", "a value changed", "another field changed"],
    )
    def test_an_edit_is_a_problem_for_both(self, standalone, tmp_path: Path, edit):
        day = _old_writer_day(tmp_path, {"spot": [100.0, float("nan")]})
        line = json.loads(day.read_text(encoding="utf-8").splitlines()[0])
        edit(line)
        day.write_text(json.dumps(line) + "\n", encoding="utf-8")

        real_notes: list = []
        standalone_notes: list = []
        real = audit.verify_audit_trail_integrity(tmp_path, notes=real_notes)
        mirrored = standalone.verify_trail(tmp_path, notes=standalone_notes)
        assert real and len(real) == len(mirrored)
        assert real_notes == standalone_notes == []

    def test_the_bounds_are_the_same_in_both(self, standalone):
        from standard_quant_tools.audit import verify as verify_module

        assert standalone._EXPLAIN_MAX_TRIALS == verify_module._EXPLAIN_MAX_TRIALS
        assert standalone._EXPLAIN_MAX_BYTES == verify_module._EXPLAIN_MAX_BYTES

    def test_the_auditor_script_prints_the_note_and_passes(
        self, standalone, tmp_path: Path, capsys
    ):
        _old_writer_day(tmp_path, {"spot": [float("nan")]})

        assert standalone.main([str(tmp_path)]) == 0
        output = capsys.readouterr().out
        assert "OK" in output and "note(s), not problems" in output


class TestRecordsWrittenNowVerifyInBoth:
    @pytest.mark.parametrize(
        "value",
        [
            {"x": [1.0, float("nan")], "y": float("-inf")},
            {"b": b"\x00\xff"},
            {"s": {"zeta", "alpha"}, "d": {2: "a", 10: "b"}},
        ],
        ids=["non-finite", "bytes", "set and integer keys"],
    )
    def test_the_writer_and_both_verifiers_agree(
        self, standalone, tmp_path: Path, value
    ):
        record = audit.DecisionRecord(
            request_id="r1",
            timestamp_utc="2024-01-01T00:00:00+00:00",
            tool_name="t1",
            input=value,
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        audit.AuditWriter(audit_dir=tmp_path).write(record)

        assert audit.verify_audit_trail_integrity(tmp_path) == []
        assert standalone.verify_trail(tmp_path) == []


def _two_written_days(directory: Path) -> None:
    """Two days written through the real writer's index, one record each."""
    w = audit.AuditWriter(audit_dir=directory)
    for date in ("2024-01-01", "2024-01-02"):
        day = directory / f"{date}.jsonl"
        record = audit.DecisionRecord(
            request_id=f"r-{date}",
            timestamp_utc=f"{date}T00:00:00+00:00",
            tool_name="t1",
            input={"n": 1},
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        record.prev_record_hash = w._bootstrap_new_day(day)
        payload = json.loads(record.model_dump_json(exclude={"record_hash"}))
        record.record_hash = audit.hash_payload({**payload, "record_hash": None})
        day.write_text(record.model_dump_json() + "\n", encoding="utf-8")


#: Damage the two verifiers must read identically: where it goes, and what.
_DAMAGE = [
    ("2024-01-01.jsonl", b"{not json\n"),
    ("2024-01-01.jsonl", b"[1,2,3]\n"),
    ("2024-01-01.jsonl", b"\xff\xfe\n"),
    ("2024-01-02.jsonl", b'{"record_hash": "cut'),
    ("_chain_index.jsonl", b"[1]\n"),
    ("_chain_index.jsonl", b'"a string"\n'),
    ("2023-12-31.jsonl", b'{"request_id": "planted", "record_hash": "0"}\n'),
]


class TestBothVerifiersReadDamageAndTheHeadAlike:
    """Every verification change is made in both copies, word for word:
    a line that is not a record, a day before the index, and the head the
    check ran through."""

    @pytest.mark.parametrize(
        "where,junk", _DAMAGE, ids=[f"{w}:{j[:12]!r}" for w, j in _DAMAGE]
    )
    def test_the_same_problems_notes_and_head(self, standalone, tmp_path, where, junk):
        _two_written_days(tmp_path)
        with open(tmp_path / where, "ab") as f:
            f.write(junk)

        real_notes: list = []
        real_head: dict = {}
        mirrored_notes: list = []
        mirrored_head: dict = {}
        real = audit.verify_audit_trail_integrity(
            tmp_path, notes=real_notes, head=real_head
        )
        mirrored = standalone.verify_trail(
            tmp_path, notes=mirrored_notes, head=mirrored_head
        )

        assert real, "the damage must be reported"
        assert real == mirrored
        assert real_notes == mirrored_notes
        assert real_head == mirrored_head

    def test_the_head_sentence_is_the_same_text(self, standalone, tmp_path):
        _two_written_days(tmp_path)
        head: dict = {}
        audit.verify_audit_trail_integrity(tmp_path, head=head)
        empty: dict = {}
        audit.verify_audit_trail_integrity(tmp_path / "absent", head=empty)

        for case in (head, empty):
            assert standalone.describe_head(case) == audit.describe_head(case)
