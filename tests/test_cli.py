"""
Tests for cli.py — the audit-trail CLI (sqt replay/compare/report).

report/compare are tested against hand-built synthetic JSONL records (pure
text I/O, no re-execution needed). replay is tested against a record
produced by a real dispatch() call — same pattern as test_audit.py's
TestVerifyReplay — since verify_replay() re-runs the actual recorded tool.
"""

import json
import stat
import zipfile
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
    """Redirect the parquet OHLCV cache so tests never touch the real user
    cache. _CACHE_ROOT/_session_cache are read by functions defined in
    standard_quant_tools.data._cache (shared by every provider) -- patching
    yfinance_provider's re-exported name would not affect what those
    functions actually see."""
    import standard_quant_tools.data._cache as cache_module

    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "ohlcv")
    cache_module._session_cache.clear()


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
        # audit_dir now also contains _chain_index.jsonl (see audit.py's
        # cross-day chain continuity) which also matches *.jsonl but isn't a
        # decision-record file -- filter it out rather than grabbing
        # whichever file glob() happens to yield first.
        day_files = [
            p for p in audit_dir.glob("*.jsonl") if cli.audit._DAY_FILE_RE.match(p.name)
        ]
        record_path = day_files[0]
        record = json.loads(record_path.read_text(encoding="utf-8").splitlines()[0])

        output = cli.cmd_replay(record["request_id"], audit_dir)
        assert "output_match : True" in output


class TestCmdVerify:
    def test_clean_trail_reports_no_problems(self, patched_factory, audit_dir: Path):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        dispatch(
            "analyze_stock_risk", {"symbol": "MSFT", "benchmark": "SPY", "period": "1y"}
        )
        assert cli.cmd_verify(audit_dir=audit_dir) == []

    def test_tampered_record_reported(self, patched_factory, audit_dir: Path):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        day_files = [
            p for p in audit_dir.glob("*.jsonl") if cli.audit._DAY_FILE_RE.match(p.name)
        ]
        path = day_files[0]
        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        record["status"] = "tampered"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        problems = cli.cmd_verify(audit_dir=audit_dir)
        assert problems

    def test_single_file_mode_checks_only_that_file(
        self, patched_factory, audit_dir: Path
    ):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        day_files = [
            p for p in audit_dir.glob("*.jsonl") if cli.audit._DAY_FILE_RE.match(p.name)
        ]
        assert cli.cmd_verify(file=day_files[0]) == []

    def test_missing_audit_dir_reports_no_problems(self, tmp_path: Path):
        assert cli.cmd_verify(audit_dir=tmp_path / "nonexistent") == []

    def test_verify_via_main_clean(self, patched_factory, audit_dir: Path, capsys):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        exit_code = cli.main(["verify"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "OK" in captured.out

    def test_verify_via_main_tampered(self, patched_factory, audit_dir: Path, capsys):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        day_files = [
            p for p in audit_dir.glob("*.jsonl") if cli.audit._DAY_FILE_RE.match(p.name)
        ]
        path = day_files[0]
        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        record["status"] = "tampered"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

        exit_code = cli.main(["verify"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "problem" in captured.out


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


class TestCmdHoldAndRelease:
    def test_hold_and_release_round_trip(self, audit_dir: Path):
        path = cli.cmd_hold("2024-01-01", reason="litigation", audit_dir=audit_dir)
        assert path.exists()
        assert cli.cmd_release_hold("2024-01-01", audit_dir=audit_dir) is True
        assert not path.exists()

    def test_hold_via_main(self, audit_dir: Path, capsys):
        exit_code = cli.main(["hold", "2024-01-01", "--reason", "litigation"])
        assert exit_code == 0
        assert "Hold placed" in capsys.readouterr().out

    def test_release_hold_via_main_when_not_held(self, audit_dir: Path, capsys):
        exit_code = cli.main(["release-hold", "2024-01-01"])
        assert exit_code == 0
        assert "No hold existed" in capsys.readouterr().out


class TestCmdGC:
    def test_dry_run_lists_without_deleting(self, audit_dir: Path):
        cli.audit.AuditWriter(audit_dir=audit_dir)
        day_path = audit_dir / "2020-01-01.jsonl"
        w = cli.audit.AuditWriter(audit_dir=audit_dir)
        head = w._bootstrap_new_day(day_path)
        record = cli.audit.DecisionRecord(
            request_id="r1",
            timestamp_utc="2020-01-01T00:00:00+00:00",
            tool_name="t1",
            input={},
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        record.prev_record_hash = head
        record.record_hash = cli.audit.hash_payload(
            {**record.model_dump(exclude={"record_hash"}), "record_hash": None}
        )
        day_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

        result = cli.cmd_gc(confirm=False, retention_days=0, audit_dir=audit_dir)
        assert result == ["2020-01-01"]
        assert day_path.exists()

    def test_confirm_actually_deletes(self, audit_dir: Path):
        w = cli.audit.AuditWriter(audit_dir=audit_dir)
        day_path = audit_dir / "2020-01-01.jsonl"
        head = w._bootstrap_new_day(day_path)
        record = cli.audit.DecisionRecord(
            request_id="r1",
            timestamp_utc="2020-01-01T00:00:00+00:00",
            tool_name="t1",
            input={},
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        record.prev_record_hash = head
        record.record_hash = cli.audit.hash_payload(
            {**record.model_dump(exclude={"record_hash"}), "record_hash": None}
        )
        day_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

        result = cli.cmd_gc(confirm=True, retention_days=0, audit_dir=audit_dir)
        assert result == ["2020-01-01"]
        assert not day_path.exists()

    def test_gc_via_main_no_candidates(self, audit_dir: Path, capsys):
        exit_code = cli.main(["gc"])
        assert exit_code == 0
        assert "No day files" in capsys.readouterr().out


class TestCmdSeal:
    def test_seal_via_main(self, patched_factory, audit_dir: Path, capsys):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        day_files = [
            p for p in audit_dir.glob("*.jsonl") if cli.audit._DAY_FILE_RE.match(p.name)
        ]
        date = day_files[0].stem

        exit_code = cli.main(["seal", date])
        assert exit_code == 0
        assert "Sealed" in capsys.readouterr().out

        # Restore write permission so pytest's tmp_path teardown can clean up.
        import os as os_module

        os_module.chmod(day_files[0], stat.S_IWRITE | stat.S_IREAD)

    def test_seal_missing_date_via_main_returns_error_exit(
        self, audit_dir: Path, capsys
    ):
        exit_code = cli.main(["seal", "2099-01-01"])
        assert exit_code == 1
        assert "error" in capsys.readouterr().err


class TestCmdExport:
    def test_export_via_main_creates_bundle(
        self, patched_factory, audit_dir: Path, tmp_path: Path, capsys
    ):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        day_files = [
            p for p in audit_dir.glob("*.jsonl") if cli.audit._DAY_FILE_RE.match(p.name)
        ]
        date = day_files[0].stem
        out_path = tmp_path / "bundle.zip"

        exit_code = cli.main(
            ["export", "--start", date, "--end", date, "--out", str(out_path)]
        )
        assert exit_code == 0
        assert "Exported bundle" in capsys.readouterr().out
        assert out_path.exists()
        with zipfile.ZipFile(out_path) as zf:
            assert f"{date}.jsonl" in zf.namelist()
            assert "manifest.json" in zf.namelist()


class TestCmdKeygenAnchorAndCheckpointVerify:
    pytestmark = pytest.mark.skipif(
        not cli.audit.HAS_CRYPTOGRAPHY, reason="cryptography is not installed"
    )

    def test_keygen_via_main_writes_keypair(self, tmp_path: Path, capsys):
        out_dir = tmp_path / "keys"
        exit_code = cli.main(["keygen", "--out", str(out_dir)])
        assert exit_code == 0
        output = capsys.readouterr().out
        assert "Private key" in output
        assert "Public key" in output
        assert "local development only" in output
        assert (out_dir / "audit_signing_key.private").exists()
        assert (out_dir / "audit_signing_key.public").exists()

    def test_anchor_and_verify_checkpoint_round_trip_via_main(
        self, patched_factory, audit_dir: Path, tmp_path: Path, capsys
    ):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        day_files = [
            p for p in audit_dir.glob("*.jsonl") if cli.audit._DAY_FILE_RE.match(p.name)
        ]
        date = day_files[0].stem

        keys_dir = tmp_path / "keys"
        cli.main(["keygen", "--out", str(keys_dir)])
        priv_path = keys_dir / "audit_signing_key.private"
        pub_path = keys_dir / "audit_signing_key.public"

        exit_code = cli.main(["anchor", date, "--key", str(priv_path)])
        assert exit_code == 0
        assert "Checkpoint signed" in capsys.readouterr().out
        assert (audit_dir / f"{date}.checkpoint.json").exists()

        exit_code = cli.main(
            ["verify", "--checkpoint", date, "--pubkey", str(pub_path)]
        )
        assert exit_code == 0
        assert "Signature valid." in capsys.readouterr().out

    def test_verify_checkpoint_without_pubkey_errors(
        self, patched_factory, audit_dir: Path, capsys
    ):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        exit_code = cli.main(["verify", "--checkpoint", "2024-01-01"])
        assert exit_code == 1
        assert "--pubkey" in capsys.readouterr().err

    def test_verify_checkpoint_missing_date_fails(
        self, patched_factory, audit_dir: Path, tmp_path: Path, capsys
    ):
        keys_dir = tmp_path / "keys"
        cli.main(["keygen", "--out", str(keys_dir)])
        pub_path = keys_dir / "audit_signing_key.public"

        exit_code = cli.main(
            ["verify", "--checkpoint", "2099-01-01", "--pubkey", str(pub_path)]
        )
        assert exit_code == 1
        assert "invalid" in capsys.readouterr().out.lower()

    def test_anchor_without_key_or_env_var_errors(
        self,
        patched_factory,
        audit_dir: Path,
        capsys,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("SQT_AUDIT_SIGNING_KEY_PATH", raising=False)
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        day_files = [
            p for p in audit_dir.glob("*.jsonl") if cli.audit._DAY_FILE_RE.match(p.name)
        ]
        date = day_files[0].stem

        exit_code = cli.main(["anchor", date])
        assert exit_code == 1
        assert "error" in capsys.readouterr().err
