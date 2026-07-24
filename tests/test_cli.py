"""
Tests for cli.py — the audit-trail CLI (sqt replay/compare/report).

report/compare are tested against hand-built synthetic JSONL records (pure
text I/O, no re-execution needed). replay is tested against a record
produced by a real dispatch() call — same pattern as test_audit.py's
TestVerifyReplay — since verify_replay() re-runs the actual recorded tool.
"""

import json
from pathlib import Path

import pytest

from standard_quant_tools import cli
from standard_quant_tools.agent import dispatch


@pytest.fixture(autouse=True)
def audit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect decision-record writes to a temp directory for every test."""
    d = tmp_path / "audit"
    monkeypatch.setenv("SQT_AUDIT_DIR", str(d))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
    return d


@pytest.fixture(autouse=True)
def redirect_ohlcv_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the parquet OHLCV cache so tests never touch the real user cache."""
    import standard_quant_tools.data.yfinance_provider as provider_module

    monkeypatch.setattr(provider_module, "_CACHE_ROOT", tmp_path / "ohlcv")
    provider_module._session_cache.clear()


def _write_record(
    directory: Path, record: dict, filename: str = "2026-01-01.jsonl"
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _minimal_record(request_id: str, **overrides) -> dict:
    base = {
        "request_id": request_id,
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "tool_name": "analyze_stock_risk",
        "input": {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"},
        "data_sources": [],
        "cpp_available": False,
        "n_workers": None,
        "duration_ms": 12.3,
        "output_hash": "abc123",
        "status": "ok",
        "error_type": None,
        "error_message": None,
        "git_commit_sha": "deadbeef",
        "package_version": "0.1.0",
        "random_seed": None,
        "strategy_source_hash": None,
    }
    base.update(overrides)
    return base


class TestFindRecord:
    def test_finds_record_across_multiple_files(self, audit_dir: Path):
        _write_record(audit_dir, _minimal_record("aaa"), "2026-01-01.jsonl")
        _write_record(audit_dir, _minimal_record("bbb"), "2026-01-02.jsonl")
        found = cli.find_record("bbb", audit_dir)
        assert found["request_id"] == "bbb"

    def test_missing_record_raises(self, audit_dir: Path):
        _write_record(audit_dir, _minimal_record("aaa"))
        with pytest.raises(ValueError, match="No decision record"):
            cli.find_record("zzz", audit_dir)

    def test_missing_directory_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No decision record"):
            cli.find_record("aaa", tmp_path / "nonexistent")


class TestCmdReport:
    def test_report_contains_full_record(self, audit_dir: Path):
        _write_record(audit_dir, _minimal_record("aaa", tool_name="run_sma_backtest"))
        output = cli.cmd_report("aaa", audit_dir)
        parsed = json.loads(output)
        assert parsed["request_id"] == "aaa"
        assert parsed["tool_name"] == "run_sma_backtest"


class TestCmdCompare:
    def test_identical_records_report_no_input_diff(self, audit_dir: Path):
        _write_record(audit_dir, _minimal_record("aaa"))
        _write_record(audit_dir, _minimal_record("bbb"))
        output = cli.cmd_compare("aaa", "bbb", audit_dir)
        assert "input: identical" in output

    def test_differing_inputs_are_listed(self, audit_dir: Path):
        _write_record(
            audit_dir,
            _minimal_record(
                "aaa",
                input={"symbol": "AAPL", "benchmark": "SPY", "period": "1y"},
            ),
        )
        _write_record(
            audit_dir,
            _minimal_record(
                "bbb",
                input={"symbol": "MSFT", "benchmark": "SPY", "period": "1y"},
            ),
        )
        output = cli.cmd_compare("aaa", "bbb", audit_dir)
        assert "symbol" in output
        assert "'AAPL'" in output
        assert "'MSFT'" in output

    def test_differing_status_marked_not_equal(self, audit_dir: Path):
        _write_record(audit_dir, _minimal_record("aaa", status="ok"))
        _write_record(audit_dir, _minimal_record("bbb", status="error"))
        output = cli.cmd_compare("aaa", "bbb", audit_dir)
        assert "status" in output
        assert "!=" in output


class TestCmdReplay:
    def test_replay_matches_unmodified_record(self, patched_factory, audit_dir: Path):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        record_path = next(audit_dir.glob("*.jsonl"))
        record = json.loads(record_path.read_text(encoding="utf-8").splitlines()[0])

        output = cli.cmd_replay(record["request_id"], audit_dir)
        assert "output_match : True" in output


class TestMainEntrypoint:
    def test_report_via_main(self, audit_dir: Path, capsys):
        _write_record(audit_dir, _minimal_record("aaa"))
        exit_code = cli.main(["report", "aaa"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert '"request_id": "aaa"' in captured.out

    def test_compare_via_main(self, audit_dir: Path, capsys):
        _write_record(audit_dir, _minimal_record("aaa"))
        _write_record(audit_dir, _minimal_record("bbb"))
        exit_code = cli.main(["compare", "aaa", "bbb"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Comparing aaa vs bbb" in captured.out

    def test_missing_record_returns_nonzero_exit(self, audit_dir: Path, capsys):
        exit_code = cli.main(["report", "zzz"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "error" in captured.err
