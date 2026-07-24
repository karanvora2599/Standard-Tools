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
    records = []
    for f in sorted(directory.glob("*.jsonl")):
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

        jsonl_files = list(audit_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        assert audit.verify_audit_log_integrity(jsonl_files[0]) == []

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
        dispatch(
            "analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"}
        )
        lock_files = list(audit_dir.glob("*.jsonl.lock"))
        assert len(lock_files) == 1
        # A second write must succeed without the earlier lock being held.
        dispatch(
            "analyze_stock_risk", {"symbol": "MSFT", "benchmark": "SPY", "period": "1y"}
        )
        assert len(_audit_records(audit_dir)) == 2
