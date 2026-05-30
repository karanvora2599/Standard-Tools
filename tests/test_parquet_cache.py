"""
Tests for the Parquet-based persistent OHLCV cache in YFinanceProvider.
All tests redirect _CACHE_ROOT to a pytest tmp_path so they never touch
the real user cache directory.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import standard_quant_tools.data.yfinance_provider as provider_module
from standard_quant_tools.data.yfinance_provider import (
    YFinanceProvider,
    _is_historical,
    _norm_date,
    _parquet_path,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def redirect_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect all Parquet writes/reads to a temp directory for every test."""
    monkeypatch.setattr(provider_module, "_CACHE_ROOT", tmp_path)
    # Also clear the session TTL cache between tests so reads always hit the disk path
    provider_module._session_cache.clear()


@pytest.fixture
def minimal_ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0] * 5,
            "High": [101.0] * 5,
            "Low": [99.0] * 5,
            "Close": [100.5] * 5,
            "Volume": [1_000_000.0] * 5,
        },
        index=dates,
    )


# ── Unit tests for helper functions ──────────────────────────────────────────

class TestNormDate:
    def test_string_passthrough(self):
        assert _norm_date("2023-06-15") == "2023-06-15"

    def test_datetime_truncated(self):
        from datetime import datetime
        assert _norm_date(datetime(2023, 6, 15, 10, 30)) == "2023-06-15"

    def test_date_object(self):
        from datetime import date
        assert _norm_date(date(2023, 6, 15)) == "2023-06-15"


class TestIsHistorical:
    def test_past_date_is_historical(self):
        assert _is_historical("2020-01-01") is True

    def test_future_date_not_historical(self):
        assert _is_historical("2099-12-31") is False

    def test_today_not_historical(self):
        from datetime import date
        assert _is_historical(date.today().isoformat()) is False

    def test_yesterday_is_historical(self):
        from datetime import date, timedelta
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        assert _is_historical(yesterday) is True


class TestParquetPath:
    def test_returns_path_object(self, tmp_path: Path):
        p = _parquet_path("AAPL", "2022-01-01", "2023-01-01", "1d")
        assert isinstance(p, Path)

    def test_filename_contains_symbol(self):
        p = _parquet_path("MSFT", "2022-01-01", "2023-01-01", "1d")
        assert "MSFT" in p.name

    def test_slash_in_symbol_sanitized(self):
        p = _parquet_path("BRK/B", "2022-01-01", "2023-01-01", "1d")
        assert "/" not in p.name

    def test_different_intervals_different_paths(self):
        p1 = _parquet_path("AAPL", "2022-01-01", "2023-01-01", "1d")
        p2 = _parquet_path("AAPL", "2022-01-01", "2023-01-01", "1h")
        assert p1 != p2


# ── Integration tests for the caching behaviour ───────────────────────────────

class TestParquetCacheWrite:
    def test_cache_file_created_after_fetch(self, tmp_path: Path, minimal_ohlcv: pd.DataFrame):
        """A Parquet file should be written for historical ranges after a live fetch."""
        with patch.object(YFinanceProvider, "_fetch_from_yfinance", return_value=minimal_ohlcv, create=True):
            prov = YFinanceProvider()
            # Patch the internal yfinance call so no network is needed
            with patch("yfinance.Ticker") as mock_ticker:
                mock_ticker.return_value.history.return_value = minimal_ohlcv.rename(
                    columns=str.lower
                )
                # Use a historical end date
                prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

        expected = _parquet_path("AAPL", "2022-01-01", "2022-06-01", "1d")
        # Redirect to tmp_path (done by autouse fixture)
        from standard_quant_tools.data.yfinance_provider import _CACHE_ROOT
        pq = _CACHE_ROOT / expected.name
        assert pq.exists(), f"Expected Parquet file at {pq}"

    def test_no_cache_for_current_data(self, tmp_path: Path, minimal_ohlcv: pd.DataFrame):
        """Data fetched with today as end_date should NOT be written to Parquet."""
        from datetime import date
        today = date.today().isoformat()

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = (
                minimal_ohlcv.copy().rename(columns=str.lower)
            )
            prov = YFinanceProvider()
            prov.get_ohlcv("AAPL", "2022-01-01", today)

        from standard_quant_tools.data.yfinance_provider import _CACHE_ROOT
        parquets = list(_CACHE_ROOT.glob("*.parquet"))
        assert len(parquets) == 0, "No Parquet file should be written for current-day data"

    def test_cache_loaded_on_second_call(self, tmp_path: Path, minimal_ohlcv: pd.DataFrame):
        """Second call for a historical range must read from Parquet, not yfinance."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = (
                minimal_ohlcv.copy().rename(columns=str.lower)
            )
            prov = YFinanceProvider()
            # First call — writes to yfinance + saves Parquet
            result1 = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")
            first_call_count = mock_ticker.call_count

            # Clear session TTL cache so the function body runs again
            provider_module._session_cache.clear()

            # Second call — should read Parquet, not call yfinance again
            result2 = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

        assert mock_ticker.call_count == first_call_count, (
            "yfinance.Ticker should not be called again when Parquet cache exists"
        )
        # Parquet round-trip drops DatetimeIndex freq metadata — ignore it
        pd.testing.assert_frame_equal(result1, result2, check_freq=False)

    def test_cache_returns_correct_columns(self, tmp_path: Path, minimal_ohlcv: pd.DataFrame):
        """Data loaded from Parquet must have the same five columns as a live fetch."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = (
                minimal_ohlcv.copy().rename(columns=str.lower)
            )
            prov = YFinanceProvider()
            prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")
            provider_module._session_cache.clear()
            result = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_cache_dir_uses_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """SQT_CACHE_DIR env variable should control the cache directory."""
        custom_dir = tmp_path / "custom_cache"
        monkeypatch.setenv("SQT_CACHE_DIR", str(custom_dir))
        # Re-import to pick up the env var (simulate fresh import)
        import importlib
        importlib.reload(provider_module)
        assert str(custom_dir) in str(provider_module._CACHE_ROOT)
        # Restore
        importlib.reload(provider_module)
