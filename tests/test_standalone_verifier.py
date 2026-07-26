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

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "verify_audit_log.py"
)


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
