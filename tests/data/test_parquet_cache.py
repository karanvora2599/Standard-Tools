"""
Tests for the Parquet-based persistent OHLCV cache in YFinanceProvider.
All tests redirect _CACHE_ROOT to a pytest tmp_path so they never touch
the real user cache directory.
"""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import standard_quant_tools.data._cache as cache_module
import standard_quant_tools.data.yfinance_provider as provider_module
from standard_quant_tools.data._cache import _parquet_path
from standard_quant_tools.data.yfinance_provider import (
    YFinanceProvider,
    _is_historical,
    _norm_date,
)
from standard_quant_tools.error import ValidationError

# ── Helpers ───────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def redirect_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect all Parquet writes/reads to a temp directory for every test.

    _CACHE_ROOT/_session_cache are read by functions defined in
    standard_quant_tools.data._cache (extracted there so Bloomberg/Polygon
    can share them) -- patching yfinance_provider's re-exported name would
    not affect what those functions actually see, so the patch target is
    the defining module, not the re-exporting one.
    """
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path)
    # Also clear the session TTL cache between tests so reads always hit the disk path
    cache_module._session_cache.clear()


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

    def test_resolved_path_is_inside_cache_root(self, tmp_path: Path):
        p = _parquet_path("AAPL", "2022-01-01", "2023-01-01", "1d")
        assert p.resolve().is_relative_to(tmp_path.resolve())


class TestParquetPathContainment:
    """
    Regression tests (operational item C): symbol/start/end/interval are
    all LLM-reachable via get_ohlcv's own parameters and go straight into
    the cache filename -- only "/" in symbol was ever sanitized. Applies
    the same slug-plus-resolved-containment approach as artifacts.py.
    """

    @pytest.mark.parametrize(
        "bad_symbol",
        [
            "../../etc/passwd",
            "..\\..\\Windows\\System32",
            "AAPL/../../x",
            "AAPL\x00.txt",
            "C:\\Windows",
            "",
        ],
    )
    def test_path_traversal_symbol_raises(self, bad_symbol):
        with pytest.raises(ValidationError, match="not a valid identifier|empty"):
            _parquet_path(bad_symbol, "2022-01-01", "2023-01-01", "1d")

    def test_invalid_interval_raises(self):
        with pytest.raises(ValidationError, match="interval"):
            _parquet_path("AAPL", "2022-01-01", "2023-01-01", "../../etc")

    def test_unnormalized_start_date_raises(self):
        with pytest.raises(ValidationError, match="start"):
            _parquet_path("AAPL", "../../etc/passwd", "2023-01-01", "1d")

    def test_unnormalized_end_date_raises(self):
        with pytest.raises(ValidationError, match="end"):
            _parquet_path("AAPL", "2022-01-01", "../../etc/passwd", "1d")

    def test_legitimate_slash_ticker_still_works(self):
        """BRK/B (and similar real tickers) must still work -- only actual
        traversal attempts are rejected, not every symbol containing '/'."""
        p = _parquet_path("BRK/B", "2022-01-01", "2023-01-01", "1d")
        assert "/" not in p.name
        assert "BRK" in p.name and p.name.endswith(".parquet")

    def test_slash_and_dash_tickers_do_not_collide(self):
        """
        Regression test: '/' used to be encoded by replacing it with '-',
        which made BRK/B and BRK-B -- two genuinely different symbols in real
        ticker vocabularies -- resolve to the SAME cache file, so one symbol
        could be served the other's cached bars.
        """
        slash = _parquet_path("BRK/B", "2022-01-01", "2023-01-01", "1d")
        dash = _parquet_path("BRK-B", "2022-01-01", "2023-01-01", "1d")
        assert slash != dash
        assert "/" not in slash.name and "/" not in dash.name


class TestNormDateValidation:
    """
    _norm_date's job here is rejecting non-date-SHAPED strings before they
    reach the cache filename (a path-traversal concern), not full calendar
    validation (e.g. month=13) -- so these cases are all strings that don't
    even match the YYYY-MM-DD shape after truncation to 10 characters.
    """

    @pytest.mark.parametrize(
        "bad_date",
        [
            "../../etc/passwd",
            "not-a-date",
            "2022/01/01",
        ],
    )
    def test_malformed_date_string_raises(self, bad_date):
        with pytest.raises(ValidationError, match="YYYY-MM-DD"):
            _norm_date(bad_date)

    def test_valid_date_string_passes(self):
        assert _norm_date("2022-01-01") == "2022-01-01"


# ── Integration tests for the caching behaviour ───────────────────────────────


class TestParquetCacheWrite:
    def test_cache_file_created_after_fetch(
        self, tmp_path: Path, minimal_ohlcv: pd.DataFrame
    ):
        """A Parquet file should be written for historical ranges after a live fetch."""
        with patch.object(
            YFinanceProvider,
            "_fetch_from_yfinance",
            return_value=minimal_ohlcv,
            create=True,
        ):
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
        from standard_quant_tools.data._cache import _CACHE_ROOT

        pq = _CACHE_ROOT / expected.name
        assert pq.exists(), f"Expected Parquet file at {pq}"

    def test_no_cache_for_current_data(
        self, tmp_path: Path, minimal_ohlcv: pd.DataFrame
    ):
        """Data fetched with today as end_date should NOT be written to Parquet."""
        from datetime import date

        today = date.today().isoformat()

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = minimal_ohlcv.copy().rename(
                columns=str.lower
            )
            prov = YFinanceProvider()
            prov.get_ohlcv("AAPL", "2022-01-01", today)

        from standard_quant_tools.data._cache import _CACHE_ROOT

        parquets = list(_CACHE_ROOT.glob("*.parquet"))
        assert (
            len(parquets) == 0
        ), "No Parquet file should be written for current-day data"

    def test_cache_loaded_on_second_call(
        self, tmp_path: Path, minimal_ohlcv: pd.DataFrame
    ):
        """Second call for a historical range must read from Parquet, not yfinance."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = minimal_ohlcv.copy().rename(
                columns=str.lower
            )
            prov = YFinanceProvider()
            # First call — writes to yfinance + saves Parquet
            result1 = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")
            first_call_count = mock_ticker.call_count

            # Clear session TTL cache so the function body runs again
            cache_module._session_cache.clear()

            # Second call — should read Parquet, not call yfinance again
            result2 = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

        assert (
            mock_ticker.call_count == first_call_count
        ), "yfinance.Ticker should not be called again when Parquet cache exists"
        # Parquet round-trip drops DatetimeIndex freq metadata — ignore it
        pd.testing.assert_frame_equal(result1, result2, check_freq=False)


class TestTimezoneNormalization:
    """
    Regression tests: yfinance attaches the listing exchange's own timezone
    to its returned index (even for daily bars) -- e.g. tz-aware
    'America/New_York'. Every downstream consumer (agent/tools.py's
    pd.Timestamp(iso_date) signal/target-weight keys, portfolio_engine.py's
    per-ticker index intersection, signal_fill_policy's reindex) builds or
    compares against tz-naive, midnight-normalized timestamps -- a tz-aware
    provider index would make those either raise or (reindex doesn't raise)
    silently produce an all-NaN/all-zero result.
    """

    def test_tz_aware_index_normalized_to_naive(self, tmp_path: Path):
        dates = pd.date_range("2022-01-03", periods=5, freq="B", tz="America/New_York")
        tz_aware_df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1_000_000.0] * 5,
            },
            index=dates,
        )
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = tz_aware_df
            prov = YFinanceProvider()
            result = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

        assert result.index.tz is None
        # Matches a plain, tz-naive ISO-date timestamp exactly (midnight, no
        # residual time-of-day component from the exchange's local open time).
        assert pd.Timestamp("2022-01-03") in result.index

    def test_tz_aware_utc_index_also_normalized(self, tmp_path: Path):
        dates = pd.date_range("2022-01-03", periods=5, freq="B", tz="UTC")
        tz_aware_df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1_000_000.0] * 5,
            },
            index=dates,
        )
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = tz_aware_df
            prov = YFinanceProvider()
            result = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

        assert result.index.tz is None
        assert pd.Timestamp("2022-01-03") in result.index

    def test_preexisting_tz_aware_cache_file_normalized_on_read(self, tmp_path: Path):
        """
        A Parquet cache file written before this fix (or by an older
        yfinance version that returned tz-aware data) must still come back
        tz-naive on a cache-hit read -- the fix has to apply on both the
        live-fetch path and the disk-cache-hit path, not just one.
        """
        dates = pd.date_range("2022-01-03", periods=5, freq="B", tz="UTC")
        stale_cached_df = pd.DataFrame(
            {
                "Open": [100.0] * 5,
                "High": [101.0] * 5,
                "Low": [99.0] * 5,
                "Close": [100.5] * 5,
                "Volume": [1_000_000.0] * 5,
            },
            index=dates,
        )
        path = _parquet_path("AAPL", "2022-01-01", "2022-06-01", "1d")
        cache_module._CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        stale_cached_df.to_parquet(path)

        prov = YFinanceProvider()
        result = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

        assert result.index.tz is None
        assert pd.Timestamp("2022-01-03") in result.index

    def test_cache_returns_correct_columns(
        self, tmp_path: Path, minimal_ohlcv: pd.DataFrame
    ):
        """Data loaded from Parquet must have the same five columns as a live fetch."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = minimal_ohlcv.copy().rename(
                columns=str.lower
            )
            prov = YFinanceProvider()
            prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")
            cache_module._session_cache.clear()
            result = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_cache_dir_uses_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """SQT_CACHE_DIR env variable should control the cache directory."""
        import importlib

        custom_dir = tmp_path / "custom_cache"
        monkeypatch.setenv("SQT_CACHE_DIR", str(custom_dir))
        # _CACHE_ROOT is computed at import time in _cache.py; provider_module
        # binds its own reference to it via `from ._cache import ...`, so
        # both modules need reloading (cache first) for this to propagate,
        # and both need reloading back afterward in a finally -- otherwise
        # this test leaves provider_module's bound references (e.g.
        # _session_cache) pointing at objects reload just replaced in
        # cache_module, silently breaking cache isolation for every test
        # that runs after this one in the same process.
        importlib.reload(cache_module)
        importlib.reload(provider_module)
        try:
            assert str(custom_dir) in str(cache_module._CACHE_ROOT)
        finally:
            monkeypatch.delenv("SQT_CACHE_DIR", raising=False)
            importlib.reload(cache_module)
            importlib.reload(provider_module)


class TestCacheHardening:
    """
    Regression tests for get_ohlcv's cache-hit paths: an in-memory
    session-cache hit must still be audited, both cache-hit paths must
    return data a caller can't corrupt for other callers by mutating it,
    a corrupt Parquet file must be evicted and refetched rather than
    raising, and concurrent disk writes must never collide on a temp
    filename.
    """

    def test_session_cache_hit_still_records_audit(self, minimal_ohlcv: pd.DataFrame):
        with (
            patch("yfinance.Ticker") as mock_ticker,
            patch.object(provider_module.audit, "record_data_access") as mock_record,
        ):
            mock_ticker.return_value.history.return_value = minimal_ohlcv.rename(
                columns=str.lower
            )
            prov = YFinanceProvider()
            prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")
            prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")  # session-cache hit

        assert mock_record.call_count == 2
        sources = [c.kwargs.get("source") for c in mock_record.call_args_list]
        assert sources == ["live_fetch", "session_cache"]

    def test_session_cache_hit_returns_independent_copy(
        self, minimal_ohlcv: pd.DataFrame
    ):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = minimal_ohlcv.rename(
                columns=str.lower
            )
            prov = YFinanceProvider()
            first = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")
            first.iloc[0, first.columns.get_loc("Close")] = -999.0

            second = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

        assert second["Close"].iloc[0] != -999.0

    def test_disk_cache_hit_returns_independent_copy(self, minimal_ohlcv: pd.DataFrame):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = minimal_ohlcv.rename(
                columns=str.lower
            )
            prov = YFinanceProvider()
            prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

            cache_module._session_cache.clear()
            second = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")  # disk hit
            second.iloc[0, second.columns.get_loc("Close")] = -999.0

            cache_module._session_cache.clear()
            third = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")  # disk hit

        assert third["Close"].iloc[0] != -999.0

    def test_corrupt_parquet_evicted_and_refetched(self, minimal_ohlcv: pd.DataFrame):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = minimal_ohlcv.rename(
                columns=str.lower
            )
            prov = YFinanceProvider()
            prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")
            first_call_count = mock_ticker.call_count

            pq_path = _parquet_path("AAPL", "2022-01-01", "2022-06-01", "1d")
            pq_path.write_bytes(b"this is not a valid parquet file")
            cache_module._session_cache.clear()

            result = prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

        assert mock_ticker.call_count == first_call_count + 1, (
            "a corrupt cache file must trigger a live refetch, not propagate "
            "the read error"
        )
        assert pq_path.exists(), "a fresh, valid Parquet file must be rewritten"
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), minimal_ohlcv.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(pq_path).reset_index(drop=True),
            minimal_ohlcv.reset_index(drop=True),
        )

    def test_temp_filename_unique_across_writes_same_process_same_thread(
        self, minimal_ohlcv: pd.DataFrame
    ):
        """
        Two sequential disk-cache writes in the same process and thread
        (so os.getpid() and threading.get_ident() are identical both times)
        must still use different temp filenames — proving uniqueness comes
        from more than just the PID, which alone doesn't protect against
        two threads in the same process racing on the same cache file.
        """
        tmp_names = []
        orig_replace = Path.replace

        def spy_replace(self, target):
            tmp_names.append(self.name)
            return orig_replace(self, target)

        with (
            patch("yfinance.Ticker") as mock_ticker,
            patch.object(Path, "replace", spy_replace),
        ):
            mock_ticker.return_value.history.return_value = minimal_ohlcv.rename(
                columns=str.lower
            )
            prov = YFinanceProvider()
            prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

            pq_path = _parquet_path("AAPL", "2022-01-01", "2022-06-01", "1d")
            pq_path.unlink()
            cache_module._session_cache.clear()

            prov.get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

        assert len(tmp_names) == 2
        assert tmp_names[0] != tmp_names[1]

    def test_fresh_provider_instance_does_not_share_session_cache_entry(
        self, minimal_ohlcv: pd.DataFrame
    ):
        """
        The session-cache key must be scoped per provider instance (like the
        old @cached decorator's default hashkey, which included self) — a
        fresh instance must re-check the disk/network rather than silently
        reuse another instance's cached result. This matters for audit
        replay, which constructs a fresh provider specifically to re-read
        data and detect tampering.
        """
        with (
            patch("yfinance.Ticker") as mock_ticker,
            patch.object(provider_module.audit, "record_data_access") as mock_record,
        ):
            mock_ticker.return_value.history.return_value = minimal_ohlcv.rename(
                columns=str.lower
            )
            YFinanceProvider().get_ohlcv("AAPL", "2022-01-01", "2022-06-01")
            YFinanceProvider().get_ohlcv("AAPL", "2022-01-01", "2022-06-01")

        sources = [c.kwargs.get("source") for c in mock_record.call_args_list]
        assert sources == ["live_fetch", "disk_cache"], (
            "a second, distinct provider instance must not transparently "
            "hit the first instance's session-cache entry"
        )


class TestCacheInvalidSymbol:
    """A symbol yfinance itself can fetch fine but that _parquet_path can't
    safely encode into a cache filename (e.g. one containing a space) used
    to hard-fail get_ohlcv with a ValidationError, unlike PolygonProvider,
    which already degrades gracefully by skipping the disk cache for that
    call. YFinanceProvider now mirrors that via _safe_parquet_path."""

    def test_cache_invalid_symbol_still_returns_data(self, minimal_ohlcv: pd.DataFrame):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = minimal_ohlcv.rename(
                columns=str.lower
            )
            prov = YFinanceProvider()
            result = prov.get_ohlcv("AAPL US Equity", "2022-01-01", "2022-06-01")

        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), minimal_ohlcv.reset_index(drop=True)
        )

    def test_cache_invalid_symbol_does_not_write_to_disk(
        self, minimal_ohlcv: pd.DataFrame, tmp_path: Path
    ):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = minimal_ohlcv.rename(
                columns=str.lower
            )
            YFinanceProvider().get_ohlcv("AAPL US Equity", "2022-01-01", "2022-06-01")

        assert list(tmp_path.iterdir()) == []

    def test_cache_invalid_symbol_refetches_every_call(
        self, minimal_ohlcv: pd.DataFrame
    ):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = minimal_ohlcv.rename(
                columns=str.lower
            )
            prov = YFinanceProvider()
            prov.get_ohlcv("AAPL US Equity", "2022-01-01", "2022-06-01")
            cache_module._session_cache.clear()
            prov.get_ohlcv("AAPL US Equity", "2022-01-01", "2022-06-01")

        assert mock_ticker.call_count == 2


class TestIntervalAwareNormalization:
    """
    `_normalize_ohlcv_index` called `.normalize()` unconditionally, setting
    every timestamp to midnight. For intraday bars that destroyed the
    series' time-series identity outright — four hourly bars became four
    copies of the same date.

    It ran on yfinance's live fetch AND on both providers' Parquet cache
    reads, so it also made the same request answer differently depending on
    whether it was served live or from cache: Polygon's live `_parse_aggs`
    preserves intraday timestamps, the cache read did not.
    """

    @pytest.mark.parametrize(
        "interval,expected",
        [
            ("1m", True),
            ("2m", True),
            ("5m", True),
            ("15m", True),
            ("30m", True),
            ("60m", True),
            ("90m", True),
            ("1h", True),
            # The tokens that merely start with a digit and 'm': classifying
            # a monthly bar as intraday would skip the date normalization
            # every downstream consumer depends on.
            ("1d", False),
            ("5d", False),
            ("1wk", False),
            ("1mo", False),
            ("3mo", False),
        ],
    )
    def test_interval_classification(self, interval, expected):
        assert cache_module.is_intraday_interval(interval) is expected

    def test_intraday_timestamps_preserved(self):
        idx = pd.DatetimeIndex(
            [
                "2026-08-11 09:30",
                "2026-08-11 10:30",
                "2026-08-11 11:30",
                "2026-08-11 12:30",
            ]
        )
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0]}, index=idx)
        out = cache_module._normalize_ohlcv_index(df, "1h")
        assert out.index.nunique() == 4, "intraday bars collapsed to one date"
        assert list(out.index) == list(idx)

    @pytest.mark.parametrize("interval", ["1d", "5d", "1wk", "1mo", "3mo"])
    def test_daily_and_coarser_bit_identical_to_previous_behavior(self, interval):
        """Daily output must not shift at all — this change is only about
        intraday, and a silent daily change would move every existing
        index by a timezone offset."""
        idx = pd.date_range("2026-01-01", periods=5, tz="America/New_York")
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)

        legacy_idx = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        out = cache_module._normalize_ohlcv_index(df, interval)
        assert out.index.equals(legacy_idx)

    def test_default_interval_keeps_previous_behavior(self):
        """A caller that doesn't pass an interval must not silently gain a
        time component it isn't prepared for."""
        idx = pd.DatetimeIndex(["2026-08-11 09:30", "2026-08-12 15:45"])
        df = pd.DataFrame({"Close": [1.0, 2.0]}, index=idx)
        out = cache_module._normalize_ohlcv_index(df)
        assert (out.index == out.index.normalize()).all()


class TestIntradayCacheIdentity:
    """
    `_norm_date` truncated every bound to 10 characters, so two genuinely
    different intraday ranges on the same day resolved to one cache file and
    the second silently served the first's bars.
    """

    def test_distinct_intraday_ranges_get_distinct_cache_files(self):
        a = _parquet_path(
            "AAPL",
            cache_module._norm_cache_bound("2026-08-11 09:30:00", "1h"),
            cache_module._norm_cache_bound("2026-08-11 12:00:00", "1h"),
            "1h",
        )
        b = _parquet_path(
            "AAPL",
            cache_module._norm_cache_bound("2026-08-11 13:00:00", "1h"),
            cache_module._norm_cache_bound("2026-08-11 16:00:00", "1h"),
            "1h",
        )
        assert a != b

    def test_daily_cache_keys_unchanged(self):
        """Existing on-disk cache files must stay addressable."""
        assert cache_module._norm_cache_bound("2022-01-01", "1d") == "2022-01-01"
        assert (
            cache_module._norm_cache_bound(datetime(2022, 1, 1, 15, 30), "1d")
            == "2022-01-01"
        )

    def test_bare_date_under_intraday_stays_a_date(self):
        """ "the whole day" is a different request from "the day starting at
        00:00:00" — conflating them reintroduces the collision."""
        assert cache_module._norm_cache_bound("2026-08-11", "1h") == "2026-08-11"

    @pytest.mark.parametrize(
        "bad", ["2026-08-11T../etc", "../../etc/passwd", "2026-08-11T09"]
    )
    def test_widened_bound_format_still_rejects_traversal(self, bad):
        with pytest.raises(ValidationError):
            _parquet_path("AAPL", bad, "2026-08-12", "1h")
