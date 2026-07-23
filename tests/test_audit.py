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
        assert audit.hash_dataframe(sample_ohlcv) == audit.hash_dataframe(sample_ohlcv.copy())

    def test_hash_dataframe_differs_on_change(self, sample_ohlcv: pd.DataFrame):
        mutated = sample_ohlcv.copy()
        mutated.iloc[0, 0] = float(mutated.iloc[0, 0]) + 1.0
        assert audit.hash_dataframe(sample_ohlcv) != audit.hash_dataframe(mutated)

    def test_hash_payload_ignores_key_order(self):
        assert audit.hash_payload({"b": 1, "a": 2}) == audit.hash_payload({"a": 2, "b": 1})

    def test_hash_payload_differs_on_change(self):
        assert audit.hash_payload({"a": 1}) != audit.hash_payload({"a": 2})


# ── dispatch() → decision record ──────────────────────────────────────────────

class TestDispatchAudit:
    def test_successful_call_writes_one_record(self, patched_factory, audit_dir: Path):
        dispatch("analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"})
        records = _audit_records(audit_dir)
        assert len(records) == 1
        assert records[0]["tool_name"] == "analyze_stock_risk"
        assert records[0]["status"] == "ok"
        assert records[0]["output_hash"] is not None

    def test_error_call_writes_error_record(self, patched_factory, mock_provider, audit_dir: Path):
        mock_provider.get_ohlcv.side_effect = DataNotFoundError("symbol not found")
        with pytest.raises(DataNotFoundError):
            dispatch("analyze_stock_risk", {"symbol": "NOPE", "benchmark": "SPY", "period": "1y"})
        records = _audit_records(audit_dir)
        assert len(records) == 1
        assert records[0]["status"] == "error"
        assert records[0]["error_type"] == "DataNotFoundError"

    def test_disabled_audit_writes_nothing(self, patched_factory, audit_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "0")
        dispatch("analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"})
        assert not audit_dir.exists() or _audit_records(audit_dir) == []

    def test_record_includes_reproducibility_provenance(self, patched_factory, audit_dir: Path):
        """
        git_commit_sha/package_version are best-effort: this repo's own git
        context makes git_commit_sha deterministically non-null when running
        from a checkout, but the assertion tolerates None so it doesn't
        break in a sandbox without git — the point is the keys are always
        present and package_version is always resolvable (it's a plain
        module attribute, no subprocess involved).
        """
        dispatch("analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"})
        records = _audit_records(audit_dir)
        assert len(records) == 1
        assert "git_commit_sha" in records[0]
        assert records[0]["git_commit_sha"] is None or isinstance(records[0]["git_commit_sha"], str)
        assert records[0]["package_version"] == "0.1.0"

    def test_request_id_correlates_log_and_record(
        self, patched_factory, audit_dir: Path, caplog: pytest.LogCaptureFixture,
    ):
        caplog.set_level(logging.DEBUG, logger="standard_quant_tools")
        caplog.handler.addFilter(audit.RequestIdFilter())

        dispatch("analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"})

        records = _audit_records(audit_dir)
        assert len(records) == 1
        written_request_id = records[0]["request_id"]

        log_request_ids = {getattr(r, "request_id", None) for r in caplog.records}
        assert written_request_id in log_request_ids


# ── Async context propagation (get_portfolio_analysis) ────────────────────────

class TestAsyncProvenance:
    def test_portfolio_call_captures_all_ticker_sources(
        self, audit_dir: Path, sample_ohlcv: pd.DataFrame,
    ):
        """
        get_portfolio_analysis fetches per-ticker data via get_ohlcv_async,
        which runs in a thread pool executor. Without copying the calling
        context into that thread (see the fix in get_ohlcv_async), the
        per-ticker data_sources entries would silently be dropped and only
        the synchronously-fetched benchmark would show up.
        """
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = sample_ohlcv.rename(columns=str.lower)
            dispatch("get_portfolio_analysis", {
                "tickers": ["AAPL", "MSFT"],
                "weights": [0.5, 0.5],
                "start_date": "2022-01-01",
                "end_date": "2022-06-01",
                "benchmark": "SPY",
            })

        records = _audit_records(audit_dir)
        assert len(records) == 1
        symbols = {s["symbol"] for s in records[0]["data_sources"]}
        assert symbols == {"AAPL", "MSFT", "SPY"}


# ── verify_replay() ────────────────────────────────────────────────────────────

class TestVerifyReplay:
    def test_replay_matches_unmodified_record(self, patched_factory, audit_dir: Path):
        dispatch("analyze_stock_risk", {"symbol": "AAPL", "benchmark": "SPY", "period": "1y"})
        record = _audit_records(audit_dir)[0]

        result = audit.verify_replay(record)
        assert result.output_match is True

    def test_replay_flags_tampered_cache(self, audit_dir: Path, tmp_path: Path, sample_ohlcv: pd.DataFrame):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = sample_ohlcv.rename(columns=str.lower)
            dispatch("run_hurst_analysis", {
                "symbol": "AAPL",
                "start_date": "2022-01-01",
                "end_date": "2022-06-01",
                "method": "dfa",
            })
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
