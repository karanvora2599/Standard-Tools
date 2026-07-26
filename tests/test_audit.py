"""
Tests for the decision-record audit trail (src/standard_quant_tools/audit.py).

Uses the same patterns as test_parquet_cache.py / test_data.py: `patched_factory`
for tests that don't care about real data provenance, and a direct
`patch("yfinance.Ticker")` + redirected `_CACHE_ROOT` for tests that exercise
the real YFinanceProvider code path (needed to verify data-source provenance
and the async context-propagation fix).
"""

import json
import logging
import stat
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import standard_quant_tools.data.yfinance_provider as provider_module
from standard_quant_tools import audit
from standard_quant_tools.agent import dispatch
from standard_quant_tools.data.yfinance_provider import _parquet_path
from standard_quant_tools.error import DataNotFoundError

# ── Fixtures ─────────────────────────────────────────────────────────────────


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
    monkeypatch.setattr(provider_module, "_CACHE_ROOT", tmp_path / "ohlcv")
    provider_module._session_cache.clear()


def _audit_records(directory: Path):
    """Decision records only -- excludes the chain-index witness log
    (_chain_index.jsonl), which lives in the same directory and matches the
    same *.jsonl glob but holds index entries, not DecisionRecords."""
    records = []
    for f in sorted(directory.glob("*.jsonl")):
        if not audit._DAY_FILE_RE.match(f.name):
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


# ── Hashing ──────────────────────────────────────────────────────────────────


class TestHashing:
    def test_hash_dataframe_deterministic(self, sample_ohlcv: pd.DataFrame):
        assert audit.hash_dataframe(sample_ohlcv) == audit.hash_dataframe(
            sample_ohlcv.copy()
        )

    def test_hash_dataframe_differs_on_change(self, sample_ohlcv: pd.DataFrame):
        mutated = sample_ohlcv.copy()
        mutated.iloc[0, 0] = float(mutated.iloc[0, 0]) + 1.0
        assert audit.hash_dataframe(sample_ohlcv) != audit.hash_dataframe(mutated)

    def test_hash_payload_ignores_key_order(self):
        assert audit.hash_payload({"b": 1, "a": 2}) == audit.hash_payload(
            {"a": 2, "b": 1}
        )

    def test_hash_payload_differs_on_change(self):
        assert audit.hash_payload({"a": 1}) != audit.hash_payload({"a": 2})


# ── dispatch() → decision record ──────────────────────────────────────────────


class TestDispatchAudit:
    def test_successful_call_writes_one_record(self, patched_factory, audit_dir: Path):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        records = _audit_records(audit_dir)
        assert len(records) == 1
        assert records[0]["tool_name"] == "analyze_stock_risk"
        assert records[0]["status"] == "ok"
        assert records[0]["output_hash"] is not None

    def test_error_call_writes_error_record(
        self, patched_factory, mock_provider, audit_dir: Path
    ):
        mock_provider.get_ohlcv.side_effect = DataNotFoundError("symbol not found")
        with pytest.raises(DataNotFoundError):
            dispatch(
                "analyze_stock_risk",
                {"symbol": "NOPE", "benchmark": "SPY", "period": "1y"},
            )
        records = _audit_records(audit_dir)
        assert len(records) == 1
        assert records[0]["status"] == "error"
        assert records[0]["error_type"] == "DataNotFoundError"

    def test_disabled_audit_writes_nothing(
        self, patched_factory, audit_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "0")
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        assert not audit_dir.exists() or _audit_records(audit_dir) == []

    def test_record_includes_reproducibility_provenance(
        self, patched_factory, audit_dir: Path
    ):
        """
        git_commit_sha/package_version are best-effort: this repo's own git
        context makes git_commit_sha deterministically non-null when running
        from a checkout, but the assertion tolerates None so it doesn't
        break in a sandbox without git — the point is the keys are always
        present and package_version is always resolvable (it's a plain
        module attribute, no subprocess involved).
        """
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        records = _audit_records(audit_dir)
        assert len(records) == 1
        assert "git_commit_sha" in records[0]
        assert records[0]["git_commit_sha"] is None or isinstance(
            records[0]["git_commit_sha"], str
        )
        assert records[0]["package_version"] == "0.1.0"

    def test_record_includes_strategy_source_hash_when_applicable(
        self, patched_factory, audit_dir: Path
    ):
        dispatch(
            "run_sma_backtest",
            {
                "symbol": "AAPL",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "strategy_type": "sma_crossover",
                "parameters": {"fast_period": 10, "slow_period": 30},
            },
        )
        records = _audit_records(audit_dir)
        assert len(records) == 1
        assert records[0]["strategy_source_hash"] is not None
        assert isinstance(records[0]["strategy_source_hash"], str)

    def test_strategy_source_hash_none_when_not_applicable(
        self, patched_factory, audit_dir: Path
    ):
        """analyze_stock_risk's input model has neither a `strategy` nor a
        `strategy_type` field — the hash must be None, not an error."""
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        records = _audit_records(audit_dir)
        assert len(records) == 1
        assert records[0]["strategy_source_hash"] is None

    def test_record_includes_random_seed_when_present(
        self, patched_factory, audit_dir: Path
    ):
        dispatch(
            "get_robustness_diagnostics",
            {
                "symbol": "AAPL",
                "start_date": "2022-01-01",
                "end_date": "2023-01-01",
                "strategy": "sma_crossover",
                "param_grid": {"fast_period": [10], "slow_period": [30]},
                "n_bootstrap_iterations": 20,
                "random_seed": 42,
            },
        )
        records = _audit_records(audit_dir)
        assert len(records) == 1
        assert records[0]["random_seed"] == 42

    def test_random_seed_none_when_absent(self, patched_factory, audit_dir: Path):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        records = _audit_records(audit_dir)
        assert records[0]["random_seed"] is None

    def test_request_id_correlates_log_and_record(
        self,
        patched_factory,
        audit_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.DEBUG, logger="standard_quant_tools")
        caplog.handler.addFilter(audit.RequestIdFilter())

        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )

        records = _audit_records(audit_dir)
        assert len(records) == 1
        written_request_id = records[0]["request_id"]

        log_request_ids = {getattr(r, "request_id", None) for r in caplog.records}
        assert written_request_id in log_request_ids


# ── Async context propagation (get_portfolio_analysis) ────────────────────────


class TestAsyncProvenance:
    def test_portfolio_call_captures_all_ticker_sources(
        self,
        audit_dir: Path,
        sample_ohlcv: pd.DataFrame,
    ):
        """
        get_portfolio_analysis fetches per-ticker data via get_ohlcv_async,
        which runs in a thread pool executor. Without copying the calling
        context into that thread (see the fix in get_ohlcv_async), the
        per-ticker data_sources entries would silently be dropped and only
        the synchronously-fetched benchmark would show up.
        """
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = sample_ohlcv.rename(
                columns=str.lower
            )
            dispatch(
                "get_portfolio_analysis",
                {
                    "tickers": ["AAPL", "MSFT"],
                    "weights": [0.5, 0.5],
                    "start_date": "2022-01-01",
                    "end_date": "2022-06-01",
                    "benchmark": "SPY",
                },
            )

        records = _audit_records(audit_dir)
        assert len(records) == 1
        symbols = {s["symbol"] for s in records[0]["data_sources"]}
        assert symbols == {"AAPL", "MSFT", "SPY"}


# ── verify_replay() ────────────────────────────────────────────────────────────


class TestVerifyReplay:
    def test_replay_matches_unmodified_record(self, patched_factory, audit_dir: Path):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        record = _audit_records(audit_dir)[0]

        result = audit.verify_replay(record)
        assert result.output_match is True

    def test_replay_flags_tampered_cache(
        self, audit_dir: Path, tmp_path: Path, sample_ohlcv: pd.DataFrame
    ):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = sample_ohlcv.rename(
                columns=str.lower
            )
            dispatch(
                "run_hurst_analysis",
                {
                    "symbol": "AAPL",
                    "start_date": "2022-01-01",
                    "end_date": "2022-06-01",
                    "method": "dfa",
                },
            )
        record = _audit_records(audit_dir)[0]

        # Rewrite the cached Parquet file with altered values — simulates the
        # provider silently revising historical data between the original
        # call and the replay.
        pq_path = _parquet_path("AAPL", "2022-01-01", "2022-06-01", "1d")
        cached = pd.read_parquet(provider_module._CACHE_ROOT / pq_path.name)
        cached["Close"] = cached["Close"] * 1.5
        cached.to_parquet(provider_module._CACHE_ROOT / pq_path.name)

        result = audit.verify_replay(record)
        assert any(not m["match"] for m in result.data_source_matches)
        assert result.notes  # a diagnostic note should explain the mismatch


class TestReplayDataSourceAsymmetry:
    def test_source_missing_from_replay_is_still_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """
        verify_replay used to only iterate the replay's *new* data sources,
        so a symbol/range present in the original record but no longer
        touched by the replay (e.g. the tool changed which tickers it
        fetches) was silently dropped from data_source_matches instead of
        being surfaced as a mismatch.
        """
        from pydantic import BaseModel

        from standard_quant_tools.agent import tools as tools_module

        class DummyInput(BaseModel):
            symbol: str = "AAPL"

        class DummyOutput(BaseModel):
            value: float = 1.0

        def dummy_fn(inp: DummyInput) -> DummyOutput:
            # The replay only re-touches AAPL — MSFT has "disappeared".
            audit.record_data_access(
                "AAPL",
                "2022-01-01",
                "2022-06-01",
                "1d",
                source="live_fetch",
                content_hash="new-aapl-hash",
            )
            return DummyOutput()

        monkeypatch.setitem(
            tools_module._TOOL_DISPATCH, "dummy_tool", (dummy_fn, DummyInput)
        )

        record = {
            "request_id": "test-req",
            "tool_name": "dummy_tool",
            "input": {"symbol": "AAPL"},
            "data_sources": [
                {
                    "symbol": "AAPL",
                    "start": "2022-01-01",
                    "end": "2022-06-01",
                    "interval": "1d",
                    "source": "live_fetch",
                    "content_hash": "old-aapl-hash",
                },
                {
                    "symbol": "MSFT",
                    "start": "2022-01-01",
                    "end": "2022-06-01",
                    "interval": "1d",
                    "source": "live_fetch",
                    "content_hash": "old-msft-hash",
                },
            ],
            "output_hash": None,
        }

        result = audit.verify_replay(record)

        msft_entries = [m for m in result.data_source_matches if m["symbol"] == "MSFT"]
        assert len(msft_entries) == 1, (
            "a source present in the original record but absent from the "
            "replay must still appear in data_source_matches"
        )
        assert msft_entries[0]["match"] is False
        assert msft_entries[0]["new_hash"] is None
        assert msft_entries[0]["old_hash"] == "old-msft-hash"


# ── Hash-chain tamper evidence ─────────────────────────────────────────────────


class TestHashChainIntegrity:
    def test_chained_records_pass_integrity_check(
        self, patched_factory, audit_dir: Path
    ):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        dispatch(
            "analyze_stock_risk", {"symbol": "MSFT", "benchmark": "SPY", "period": "1y"}
        )
        records = _audit_records(audit_dir)
        assert len(records) == 2
        assert records[0]["prev_record_hash"] == audit._GENESIS_HASH
        assert records[1]["prev_record_hash"] == records[0]["record_hash"]
        assert records[0]["record_hash"] is not None
        assert records[0]["record_hash"] != records[1]["record_hash"]

        jsonl_files = [
            p for p in audit_dir.glob("*.jsonl") if audit._DAY_FILE_RE.match(p.name)
        ]
        assert len(jsonl_files) == 1
        assert audit.verify_audit_log_integrity(jsonl_files[0]) == []
        # And the chain index (a separate file, created for the first write
        # of this new day) is itself clean when checked as part of the
        # full cross-day trail.
        assert audit.verify_audit_trail_integrity(audit_dir) == []

    def test_editing_a_record_breaks_the_chain(self, patched_factory, audit_dir: Path):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        dispatch(
            "analyze_stock_risk", {"symbol": "MSFT", "benchmark": "SPY", "period": "1y"}
        )
        jsonl_path = next(iter(audit_dir.glob("*.jsonl")))
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()

        first = json.loads(lines[0])
        first["status"] = "tampered"  # alter content without recomputing hashes
        lines[0] = json.dumps(first)
        jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        problems = audit.verify_audit_log_integrity(jsonl_path)
        assert problems, "editing a record's content must be detected"
        assert any("record_hash" in p for p in problems)

    def test_removing_a_record_breaks_the_chain(self, patched_factory, audit_dir: Path):
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        dispatch(
            "analyze_stock_risk", {"symbol": "MSFT", "benchmark": "SPY", "period": "1y"}
        )
        dispatch(
            "analyze_stock_risk",
            {"symbol": "GOOGL", "benchmark": "SPY", "period": "1y"},
        )
        jsonl_path = next(iter(audit_dir.glob("*.jsonl")))
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        del lines[1]  # remove the middle record
        jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        problems = audit.verify_audit_log_integrity(jsonl_path)
        assert problems, "removing a record must break the chain for what follows"
        assert any("chain broken" in p for p in problems)

    def test_clean_nonexistent_file_reports_no_problems(self, tmp_path: Path):
        assert audit.verify_audit_log_integrity(tmp_path / "no-such-file.jsonl") == []

    def test_write_creates_and_releases_a_sidecar_lock_file(
        self, patched_factory, audit_dir: Path
    ):
        """
        The FIRST write to a fresh audit_dir creates two sidecar lock files,
        not one: the day file's own lock, plus _chain_index.jsonl.lock
        (acquired once, only for the first write of a new calendar day, to
        bootstrap the cross-day chain index — see AuditWriter._bootstrap_new_day).
        This is an intentional new sidecar from the cross-day chain-continuity
        feature, not a regression of the original single-lock-file behavior.
        """
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        lock_files = list(audit_dir.glob("*.jsonl.lock"))
        day_locks = [p for p in lock_files if p.name != "_chain_index.jsonl.lock"]
        index_locks = [p for p in lock_files if p.name == "_chain_index.jsonl.lock"]
        assert len(day_locks) == 1
        assert len(index_locks) == 1
        # A second write must succeed without either earlier lock being held,
        # and must NOT re-acquire the index lock (same day, not a new day file).
        dispatch(
            "analyze_stock_risk", {"symbol": "MSFT", "benchmark": "SPY", "period": "1y"}
        )
        assert len(_audit_records(audit_dir)) == 2
        assert len(list(audit_dir.glob("*.jsonl.lock"))) == 2


# ── Cross-day chain continuity (the chain index) ───────────────────────────────


def _hand_written_record(tool_name: str, prev_hash: str) -> "audit.DecisionRecord":
    """Build and hash a DecisionRecord the same way AuditWriter.write() does,
    for tests that need to write directly to a specific day's file (bypassing
    dispatch()'s real-clock-driven day selection) to control which calendar
    day a record lands on."""
    record = audit.DecisionRecord(
        request_id="r-" + tool_name,
        timestamp_utc="2024-01-01T00:00:00+00:00",
        tool_name=tool_name,
        input={},
        cpp_available=False,
        duration_ms=1.0,
        status="ok",
    )
    record.prev_record_hash = prev_hash
    record.record_hash = audit.hash_payload(
        {**record.model_dump(exclude={"record_hash"}), "record_hash": None}
    )
    return record


class TestChainIndexContinuity:
    def test_second_day_chains_onto_previous_days_last_hash(self, tmp_path: Path):
        w = audit.AuditWriter(audit_dir=tmp_path)

        day1 = tmp_path / "2024-01-01.jsonl"
        head1 = w._bootstrap_new_day(day1)
        r1 = _hand_written_record("day1-call", head1)
        day1.write_text(r1.model_dump_json() + "\n", encoding="utf-8")

        day2 = tmp_path / "2024-01-02.jsonl"
        head2 = w._bootstrap_new_day(day2)
        assert head2 == r1.record_hash, (
            "the second day's bootstrap chain head must equal the first "
            "day's last record hash"
        )
        r2 = _hand_written_record("day2-call", head2)
        day2.write_text(r2.model_dump_json() + "\n", encoding="utf-8")

        assert audit.verify_audit_trail_integrity(tmp_path) == []

    def test_gap_day_chains_onto_most_recent_existing_file(self, tmp_path: Path):
        """A day with zero audit activity (e.g. a weekend) must not break the
        chain for the next day that does have activity -- it should skip
        straight back to the most recent day file that actually exists."""
        w = audit.AuditWriter(audit_dir=tmp_path)

        day1 = tmp_path / "2024-01-01.jsonl"
        head1 = w._bootstrap_new_day(day1)
        r1 = _hand_written_record("friday", head1)
        day1.write_text(r1.model_dump_json() + "\n", encoding="utf-8")

        # 2024-01-02 (Saturday) never happens -- no file is ever created for it.
        day3 = tmp_path / "2024-01-03.jsonl"
        head3 = w._bootstrap_new_day(day3)
        assert head3 == r1.record_hash, (
            "a gap day must not be treated as a break -- the next active day "
            "should chain onto the most recent day that actually has a file"
        )
        r3 = _hand_written_record("monday", head3)
        day3.write_text(r3.model_dump_json() + "\n", encoding="utf-8")

        assert audit.verify_audit_trail_integrity(tmp_path) == []

    def test_deleted_day_file_detected_by_trail_verification(self, tmp_path: Path):
        w = audit.AuditWriter(audit_dir=tmp_path)

        day1 = tmp_path / "2024-01-01.jsonl"
        head1 = w._bootstrap_new_day(day1)
        r1 = _hand_written_record("day1-call", head1)
        day1.write_text(r1.model_dump_json() + "\n", encoding="utf-8")

        day2 = tmp_path / "2024-01-02.jsonl"
        head2 = w._bootstrap_new_day(day2)
        r2 = _hand_written_record("day2-call", head2)
        day2.write_text(r2.model_dump_json() + "\n", encoding="utf-8")

        assert audit.verify_audit_trail_integrity(tmp_path) == []

        day1.unlink()  # simulate an attacker deleting an entire day's records
        problems = audit.verify_audit_trail_integrity(tmp_path)
        assert problems, "a deleted day file must be detected"
        assert any("2024-01-01" in p and "no longer exists" in p for p in problems)

    def test_day_file_without_index_entry_detected(self, tmp_path: Path):
        """The reverse tamper direction: the index entry is removed but the
        day file itself is left in place."""
        w = audit.AuditWriter(audit_dir=tmp_path)

        day1 = tmp_path / "2024-01-01.jsonl"
        head1 = w._bootstrap_new_day(day1)
        r1 = _hand_written_record("day1-call", head1)
        day1.write_text(r1.model_dump_json() + "\n", encoding="utf-8")

        day2 = tmp_path / "2024-01-02.jsonl"
        head2 = w._bootstrap_new_day(day2)
        r2 = _hand_written_record("day2-call", head2)
        day2.write_text(r2.model_dump_json() + "\n", encoding="utf-8")

        # Remove day2's index entry only, leaving day2.jsonl untouched.
        index_path = tmp_path / audit._INDEX_FILENAME
        lines = [
            line
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if '"2024-01-02"' not in line
        ]
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        problems = audit.verify_audit_trail_integrity(tmp_path)
        assert any("no corresponding chain index entry" in p for p in problems)

    def test_index_own_chain_tamper_detected(self, tmp_path: Path):
        w = audit.AuditWriter(audit_dir=tmp_path)
        day1 = tmp_path / "2024-01-01.jsonl"
        head1 = w._bootstrap_new_day(day1)
        r1 = _hand_written_record("day1-call", head1)
        day1.write_text(r1.model_dump_json() + "\n", encoding="utf-8")

        day2 = tmp_path / "2024-01-02.jsonl"
        w._bootstrap_new_day(day2)

        index_path = tmp_path / audit._INDEX_FILENAME
        entry = json.loads(index_path.read_text(encoding="utf-8").splitlines()[0])
        entry["chain_head"] = "tampered00000000"  # alter without recomputing index_hash
        index_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        problems = audit.verify_audit_trail_integrity(tmp_path)
        assert any("index_hash" in p for p in problems)

    def test_fsync_called_on_write(
        self, patched_factory, audit_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import os as os_module

        calls = []
        real_fsync = os_module.fsync
        monkeypatch.setattr(
            os_module, "fsync", lambda fd: (calls.append(fd), real_fsync(fd))[1]
        )
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        assert len(calls) >= 1, "os.fsync must be called when writing a decision record"


# ── Legal hold ──────────────────────────────────────────────────────────────


class TestLegalHold:
    def test_hold_creates_sidecar_and_is_held_reports_true(self, tmp_path: Path):
        assert not audit.is_held("2024-01-01", audit_dir=tmp_path)
        path = audit.hold_day("2024-01-01", audit_dir=tmp_path, reason="litigation")
        assert path.exists()
        assert audit.is_held("2024-01-01", audit_dir=tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["reason"] == "litigation"
        assert payload["date"] == "2024-01-01"

    def test_hold_works_ahead_of_any_activity(self, tmp_path: Path):
        """A hold is a statement about a date, not a file -- placing one on a
        date with no audit file yet must not raise."""
        path = audit.hold_day("2099-01-01", audit_dir=tmp_path)
        assert path.exists()

    def test_release_hold_removes_sidecar_and_reports_true(self, tmp_path: Path):
        audit.hold_day("2024-01-01", audit_dir=tmp_path)
        assert audit.release_hold("2024-01-01", audit_dir=tmp_path) is True
        assert not audit.is_held("2024-01-01", audit_dir=tmp_path)

    def test_release_hold_on_unheld_date_reports_false(self, tmp_path: Path):
        assert audit.release_hold("2024-01-01", audit_dir=tmp_path) is False

    def test_holding_twice_overwrites_reason(self, tmp_path: Path):
        path = audit.hold_day("2024-01-01", audit_dir=tmp_path, reason="first")
        audit.hold_day("2024-01-01", audit_dir=tmp_path, reason="second")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["reason"] == "second"


# ── Retention / gc ──────────────────────────────────────────────────────────


class TestRetentionGC:
    def _write_day(self, tmp_path: Path, date: str) -> Path:
        w = audit.AuditWriter(audit_dir=tmp_path)
        day_path = tmp_path / f"{date}.jsonl"
        head = w._bootstrap_new_day(day_path)
        record = _hand_written_record(f"call-{date}", head)
        day_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")
        return day_path

    def test_no_retention_configured_yields_no_candidates(self, tmp_path: Path):
        self._write_day(tmp_path, "2020-01-01")
        assert audit.gc_candidates(audit_dir=tmp_path) == []

    def test_retention_days_param_finds_old_days(self, tmp_path: Path):
        self._write_day(tmp_path, "2020-01-01")
        assert audit.gc_candidates(audit_dir=tmp_path, retention_days=0) == [
            "2020-01-01"
        ]

    def test_retention_days_env_var_used_when_param_omitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        self._write_day(tmp_path, "2020-01-01")
        monkeypatch.setenv("SQT_AUDIT_RETENTION_DAYS", "0")
        assert audit.gc_candidates(audit_dir=tmp_path) == ["2020-01-01"]

    def test_held_day_excluded_from_candidates(self, tmp_path: Path):
        self._write_day(tmp_path, "2020-01-01")
        audit.hold_day("2020-01-01", audit_dir=tmp_path)
        assert audit.gc_candidates(audit_dir=tmp_path, retention_days=0) == []

    def test_gc_dry_run_does_not_delete(self, tmp_path: Path):
        day_path = self._write_day(tmp_path, "2020-01-01")
        result = audit.gc(audit_dir=tmp_path, retention_days=0, dry_run=True)
        assert result == ["2020-01-01"]
        assert day_path.exists()

    def test_gc_confirm_deletes_candidates_only(self, tmp_path: Path):
        old_path = self._write_day(tmp_path, "2020-01-01")
        held_path = self._write_day(tmp_path, "2020-01-02")
        audit.hold_day("2020-01-02", audit_dir=tmp_path)

        deleted = audit.gc(audit_dir=tmp_path, retention_days=0, dry_run=False)

        assert deleted == ["2020-01-01"]
        assert not old_path.exists()
        assert held_path.exists()

    def test_gc_never_runs_automatically_from_dispatch(
        self, patched_factory, audit_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A retention window being configured must not cause dispatch() to
        delete anything on its own -- gc() is only ever invoked explicitly."""
        monkeypatch.setenv("SQT_AUDIT_RETENTION_DAYS", "0")
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        assert len(_audit_records(audit_dir)) == 1


# ── Sealing ─────────────────────────────────────────────────────────────────


class TestSealing:
    def test_seal_makes_day_file_read_only(self, tmp_path: Path):
        w = audit.AuditWriter(audit_dir=tmp_path)
        day_path = tmp_path / "2024-01-01.jsonl"
        head = w._bootstrap_new_day(day_path)
        record = _hand_written_record("call", head)
        day_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

        sealed_path = audit.seal_day("2024-01-01", audit_dir=tmp_path)
        assert sealed_path == day_path
        with pytest.raises(PermissionError):
            with open(day_path, "a", encoding="utf-8"):
                pass

        # Cleanup: restore write permission so pytest's tmp_path teardown
        # can remove the file on platforms that enforce this at delete time.
        import os as os_module

        os_module.chmod(day_path, stat.S_IWRITE | stat.S_IREAD)

    def test_seal_missing_day_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            audit.seal_day("2099-01-01", audit_dir=tmp_path)

    def test_seal_does_not_touch_the_chain_index(self, tmp_path: Path):
        w = audit.AuditWriter(audit_dir=tmp_path)
        day_path = tmp_path / "2024-01-01.jsonl"
        head = w._bootstrap_new_day(day_path)
        record = _hand_written_record("call", head)
        day_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

        audit.seal_day("2024-01-01", audit_dir=tmp_path)

        # The index must remain writable -- a later day's bootstrap still
        # needs to append to it.
        day2_path = tmp_path / "2024-01-02.jsonl"
        w._bootstrap_new_day(day2_path)  # must not raise

        import os as os_module

        os_module.chmod(day_path, stat.S_IWRITE | stat.S_IREAD)


# ── Redaction ───────────────────────────────────────────────────────────────


class TestRedaction:
    def test_no_fields_configured_returns_input_unchanged(self):
        payload = {"symbol": "AAPL", "account_id": "12345"}
        assert audit._redact(payload, []) is payload

    def test_top_level_field_redacted(self):
        redacted = audit._redact(
            {"symbol": "AAPL", "account_id": "12345"}, ["account_id"]
        )
        assert redacted["symbol"] == "AAPL"
        assert redacted["account_id"] != "12345"
        assert redacted["account_id"].startswith("<redacted:")

    def test_nested_dotted_field_redacted(self):
        redacted = audit._redact(
            {"client": {"ssn": "123-45-6789", "name": "A"}}, ["client.ssn"]
        )
        assert redacted["client"]["ssn"].startswith("<redacted:")
        assert redacted["client"]["name"] == "A"

    def test_missing_field_silently_skipped(self):
        redacted = audit._redact({"symbol": "AAPL"}, ["account_id", "client.ssn"])
        assert redacted == {"symbol": "AAPL"}

    def test_redaction_placeholder_is_deterministic(self):
        a = audit._redact({"account_id": "12345"}, ["account_id"])
        b = audit._redact({"account_id": "12345"}, ["account_id"])
        assert a["account_id"] == b["account_id"]

    def test_redaction_does_not_mutate_original_dict(self):
        original = {"account_id": "12345"}
        audit._redact(original, ["account_id"])
        assert original["account_id"] == "12345"

    def test_redact_fields_reads_comma_separated_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("SQT_AUDIT_REDACT_FIELDS", "account_id, client.ssn")
        assert audit._redact_fields() == ["account_id", "client.ssn"]

    def test_redact_fields_empty_when_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SQT_AUDIT_REDACT_FIELDS", raising=False)
        assert audit._redact_fields() == []

    def test_dispatch_redacts_configured_field_in_written_record(
        self, patched_factory, audit_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("SQT_AUDIT_REDACT_FIELDS", "symbol")
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        records = _audit_records(audit_dir)
        assert records[0]["input"]["symbol"] != "AAPL"
        assert records[0]["input"]["symbol"].startswith("<redacted:")
        assert records[0]["input"]["benchmark"] == "SPY"  # unconfigured field untouched


# ── Export bundle ───────────────────────────────────────────────────────────


class TestExportBundle:
    def _write_day(self, tmp_path: Path, date: str, tool_name: str = "call") -> Path:
        w = audit.AuditWriter(audit_dir=tmp_path)
        day_path = tmp_path / f"{date}.jsonl"
        head = w._bootstrap_new_day(day_path)
        record = _hand_written_record(tool_name, head)
        day_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")
        return day_path

    def test_bundle_contains_day_files_index_manifest_and_verifier(
        self, tmp_path: Path
    ):
        self._write_day(tmp_path, "2024-01-01")
        out_path = tmp_path.parent / "bundle.zip"

        result = audit.export_bundle("2024-01-01", "2024-01-01", out_path, tmp_path)

        assert result == out_path
        with zipfile.ZipFile(out_path) as zf:
            names = set(zf.namelist())
        assert "2024-01-01.jsonl" in names
        assert "_chain_index.jsonl" in names
        assert "manifest.json" in names
        assert "verify_audit_log.py" in names
        assert "README.txt" in names

    def test_date_range_excludes_out_of_range_days(self, tmp_path: Path):
        self._write_day(tmp_path, "2024-01-01")
        self._write_day(tmp_path, "2024-01-02")
        self._write_day(tmp_path, "2024-01-03")
        out_path = tmp_path.parent / "bundle.zip"

        audit.export_bundle("2024-01-02", "2024-01-02", out_path, tmp_path)

        with zipfile.ZipFile(out_path) as zf:
            names = set(zf.namelist())
        assert "2024-01-02.jsonl" in names
        assert "2024-01-01.jsonl" not in names
        assert "2024-01-03.jsonl" not in names

    def test_manifest_sha256_matches_bundled_file_content(self, tmp_path: Path):
        self._write_day(tmp_path, "2024-01-01")
        out_path = tmp_path.parent / "bundle.zip"
        audit.export_bundle("2024-01-01", "2024-01-01", out_path, tmp_path)

        with zipfile.ZipFile(out_path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            by_name = {f["name"]: f for f in manifest["files"]}
            content = zf.read("2024-01-01.jsonl")
            import hashlib

            assert (
                by_name["2024-01-01.jsonl"]["sha256"]
                == hashlib.sha256(content).hexdigest()
            )
            assert by_name["2024-01-01.jsonl"]["record_count"] == 1

    def test_bundled_verifier_confirms_clean_trail(self, tmp_path: Path):
        """End-to-end: a bundle built from a real clean trail must itself
        verify clean using the exact verifier script it ships."""
        import importlib.util

        self._write_day(tmp_path, "2024-01-01")
        out_dir = tmp_path.parent / "extracted"
        out_path = tmp_path.parent / "bundle.zip"
        audit.export_bundle("2024-01-01", "2024-01-01", out_path, tmp_path)

        with zipfile.ZipFile(out_path) as zf:
            zf.extractall(out_dir)

        spec = importlib.util.spec_from_file_location(
            "bundled_verifier", out_dir / "verify_audit_log.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.verify_trail(out_dir) == []
