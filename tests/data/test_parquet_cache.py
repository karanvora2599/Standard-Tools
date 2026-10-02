"""
Tests for the Parquet-based persistent OHLCV cache in YFinanceProvider.
All tests redirect _CACHE_ROOT to a pytest tmp_path so they never touch
the real user cache directory.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.data._cache as cache_module
import standard_quant_tools.data.yfinance_provider as provider_module
from standard_quant_tools.cli import cmd_cache_gc
from standard_quant_tools.cli import main as cli_main
from standard_quant_tools.data._cache import _parquet_path
from standard_quant_tools.data.databento import DATASET_SUMMARY
from standard_quant_tools.data.polygon_provider import PolygonProvider
from standard_quant_tools.data.yfinance_provider import (
    YFinanceProvider,
    _is_historical,
    _norm_date,
)
from standard_quant_tools.error import ValidationError

from .test_databento_provider import (
    BASIC,
    CONSOLIDATED,
    DEPTH,
    SINCE_2023,
    WIDE,
    StubClient,
)
from .test_databento_provider import _provider as _databento

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
        # The UTC date: the guard compared against the LOCAL date, so east
        # of UTC+5:30 a session still trading was already yesterday.
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).date()
        assert _is_historical(today.isoformat()) is False

    def test_yesterday_is_historical(self):
        from datetime import datetime, timedelta, timezone

        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
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
        # Today on the guard's clock, which is UTC: the local date can already
        # be yesterday there, and yesterday is historical.
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).date().isoformat()

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
        """SQT_CACHE_DIR controls the cache directory.

        The root is resolved at first use now, not at import, so this no
        longer reloads the module to see a new value: it clears the
        resolved root, and the next use reads the variable.
        """
        custom_dir = tmp_path / "custom_cache"
        monkeypatch.setenv("SQT_CACHE_DIR", str(custom_dir))
        monkeypatch.setattr(cache_module, "_CACHE_ROOT", None)
        assert cache_module.cache_root() == custom_dir
        assert cache_module._CACHE_ROOT == custom_dir
        path = _parquet_path("AAPL", "2022-01-01", "2022-06-01", "1d")
        assert path.parent == custom_dir.resolve()


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
            patch.object(
                provider_module.audit, "recording_data_access", return_value=True
            ),
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

        The write goes through the library's one atomic writer, which
        renames with `os.replace` rather than `Path.replace`, so that is
        what is spied on.
        """
        tmp_names = []
        orig_replace = os.replace

        def spy_replace(src, dst):
            tmp_names.append(Path(src).name)
            return orig_replace(src, dst)

        with (
            patch("yfinance.Ticker") as mock_ticker,
            patch("os.replace", spy_replace),
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
            patch.object(
                provider_module.audit, "recording_data_access", return_value=True
            ),
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
            # Databento's one-second bars: read as daily, every bar of a day
            # was flattened onto its midnight.
            ("1s", True),
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

    def test_one_second_bars_keep_their_seconds(self):
        idx = pd.date_range("2026-08-11 14:30:00", periods=4, freq="s")
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0, 4.0]}, index=idx)
        out = cache_module._normalize_ohlcv_index(df, "1s")
        assert list(out.index) == list(idx)

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


class TestInclusiveEndContract:
    """
    `DataProvider.get_ohlcv`'s `end_date` is an INCLUSIVE observation cutoff
    (data/base.py). The underlying vendors disagreed natively — Polygon's
    aggregates `to` and Bloomberg's `endDate` are inclusive, yfinance's
    `ticker.history(end=...)` is exclusive — and nothing reconciled them, so
    the same call returned a different window depending only on which
    provider served it. On the default provider it silently dropped the
    final bar, which is why score_model(as_of=X) excluded X while still
    reporting X as the as-of date.
    """

    def test_bare_date_covers_the_whole_day(self):
        bound = cache_module.inclusive_end_timestamp("2023-01-01", "1d")
        assert bound >= pd.Timestamp("2023-01-01 23:59:59")
        assert bound < pd.Timestamp("2023-01-02")

    def test_explicit_intraday_timestamp_is_exact(self):
        bound = cache_module.inclusive_end_timestamp("2026-08-11 12:00:00", "1h")
        assert bound == pd.Timestamp("2026-08-11 12:00:00")

    def test_trim_keeps_the_boundary_bar_and_drops_beyond(self):
        idx = pd.date_range("2022-12-30", "2023-01-03", freq="D")
        df = pd.DataFrame({"Close": range(len(idx))}, index=idx)
        out = cache_module.trim_to_inclusive_end(df, "2023-01-01", "1d")
        assert out.index.max() == pd.Timestamp("2023-01-01")
        assert pd.Timestamp("2023-01-02") not in out.index

    def test_yfinance_requests_an_exclusive_bound_and_returns_inclusive(self):
        """
        The end-to-end regression: yfinance must be asked for a bound PAST
        the inclusive window, and the caller must get the boundary bar back.
        """
        captured = {}

        def fake_history(**kw):
            captured.update(kw)
            # Simulate yfinance's real, exclusive-`end` behavior.
            idx = pd.date_range(
                "2022-12-28", "2023-01-05", freq="D", tz="America/New_York"
            )
            end = pd.Timestamp(kw["end"])
            if end.tzinfo is None:
                end = end.tz_localize("America/New_York")
            idx = idx[idx < end]
            return pd.DataFrame(
                {
                    c: [1.0] * len(idx)
                    for c in ["Open", "High", "Low", "Close", "Volume"]
                },
                index=idx,
            )

        cache_module._session_cache.clear()
        with patch("yfinance.Ticker") as ticker_cls:
            inst = MagicMock()
            inst.history = lambda **kw: fake_history(**kw)
            ticker_cls.return_value = inst
            df = YFinanceProvider().get_ohlcv("AAPL", "2022-12-28", "2023-01-01")

        assert pd.Timestamp(captured["end"]) > pd.Timestamp("2023-01-01")
        assert df.index.max() == pd.Timestamp(
            "2023-01-01"
        ), "the caller's inclusive end date must be present in the result"
        # And the over-fetch used to obtain it must not leak through.
        assert df.index.max() <= cache_module.inclusive_end_timestamp(
            "2023-01-01", "1d"
        )


class TestCacheFormatVersioning:
    """
    Cached frames written before the inclusive-end contract are missing
    their final bar. Serving one on a cache hit would answer the same
    request differently than a live fetch — the cache/live parity failure
    this layer exists to prevent — so the filename carries a format
    generation and old files are simply never looked up again.
    """

    def test_cache_filename_carries_a_format_version(self):
        name = _parquet_path("AAPL", "2022-01-01", "2023-01-01", "1d").name
        assert name.startswith(f"{cache_module._CACHE_FORMAT_VERSION}_")

    def test_old_generation_files_are_not_addressed(self):
        """A v1-era filename must not be what the current code looks up."""
        current = _parquet_path("AAPL", "2022-01-01", "2023-01-01", "1d").name
        legacy = "yfinance_AAPL_2022-01-01_2023-01-01_1d.parquet"
        assert current != legacy


# ── The cache root: resolved at first use, blank is the default ───────────────


class TestTheCacheRootIsReadAtFirstUse:
    """
    The root was read at import with a bare `os.environ.get`: an empty
    `SQT_CACHE_DIR` meant the working directory, a relative one moved with
    it, and a value in a local `.env` was never seen because nothing had
    loaded the file yet. It is read through `env_path` at first use now.
    """

    @pytest.fixture(autouse=True)
    def _elsewhere(self, tmp_path, monkeypatch):
        # A regression that accepted a relative root would create it under
        # the working directory; make that a throwaway one.
        monkeypatch.chdir(tmp_path)

    def _unresolved(self, monkeypatch, value):
        monkeypatch.setattr(cache_module, "_CACHE_ROOT", None)
        monkeypatch.setenv("SQT_CACHE_DIR", value)

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_setting_is_the_default_not_the_working_directory(
        self, monkeypatch, blank
    ):
        self._unresolved(monkeypatch, blank)
        root = cache_module.cache_root()
        assert root == Path.home() / ".cache" / "standard_quant_tools" / "ohlcv"
        assert root.is_absolute()

    def test_a_relative_setting_is_refused_by_name(self, monkeypatch):
        self._unresolved(monkeypatch, "relative/cache")
        with pytest.raises(ValidationError, match="SQT_CACHE_DIR is a relative"):
            cache_module.cache_root()

    def test_the_refusal_is_not_mistaken_for_an_uncacheable_symbol(self, monkeypatch):
        """`_safe_parquet_path` turns a symbol the cache cannot encode into
        "skip the cache for this call". A misconfigured root is not that:
        skipping would quietly turn the disk cache off for every call."""
        self._unresolved(monkeypatch, "relative/cache")
        with pytest.raises(ValidationError, match="SQT_CACHE_DIR"):
            cache_module._safe_parquet_path("AAPL", "2022-01-01", "2022-06-01", "1d")

    def test_a_fetch_is_refused_before_any_network_call(self, monkeypatch):
        self._unresolved(monkeypatch, "relative/cache")
        with patch("yfinance.Ticker") as mock_ticker:
            with pytest.raises(ValidationError, match="SQT_CACHE_DIR"):
                YFinanceProvider().get_ohlcv("AAPL", "2022-01-01", "2022-06-01")
        assert mock_ticker.call_count == 0
        assert not (Path.cwd() / "relative").exists()

    def test_resolved_once_then_fixed_for_the_process(self, tmp_path, monkeypatch):
        self._unresolved(monkeypatch, str(tmp_path / "first"))
        assert cache_module.cache_root() == tmp_path / "first"
        monkeypatch.setenv("SQT_CACHE_DIR", str(tmp_path / "second"))
        assert cache_module.cache_root() == tmp_path / "first"

    def test_the_root_is_readable_by_name_before_first_use(self, tmp_path, monkeypatch):
        """The describe tools and the external-data fence read `_CACHE_ROOT`
        directly; before the first use it resolves rather than reading as a
        placeholder."""
        monkeypatch.delattr(cache_module, "_CACHE_ROOT")
        monkeypatch.setenv("SQT_CACHE_DIR", str(tmp_path / "named"))
        assert cache_module._CACHE_ROOT == tmp_path / "named"


# ── The write: one atomic writer, nothing left behind ─────────────────────────


class TestAWriteLeavesNothingBehind:
    """
    The cache had its own atomic write, which removed its temp file only on
    success: a refused rename left one behind every time (on Windows, every
    write while a reader had the entry open), and a KeyboardInterrupt left
    one too. It goes through the library's one writer now, which removes
    the temp in a `finally`.
    """

    def _entry(self):
        return _parquet_path("AAPL", "2022-01-01", "2022-06-01", "1d")

    def test_a_refused_rename_leaves_no_temp_file(self, tmp_path, minimal_ohlcv):
        def refuse(src, dst):
            raise PermissionError(13, "Access is denied")

        with patch("os.replace", refuse):
            cache_module._write_parquet_atomic(self._entry(), minimal_ohlcv)
        assert list(tmp_path.iterdir()) == []

    def test_an_interrupt_propagates_and_leaves_no_temp_file(
        self, tmp_path, minimal_ohlcv
    ):
        def interrupt(src, dst):
            raise KeyboardInterrupt

        with patch("os.replace", interrupt):
            with pytest.raises(KeyboardInterrupt):
                cache_module._write_parquet_atomic(self._entry(), minimal_ohlcv)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.skipif(
        sys.platform != "win32", reason="a sharing violation is a Windows refusal"
    )
    def test_a_sharing_violation_on_the_rename_is_retried(
        self, tmp_path, minimal_ohlcv
    ):
        """A reader holding the entry open makes Windows refuse the rename
        (ERROR_SHARING_VIOLATION); the write waits it out instead of
        dropping the entry's update."""
        real = os.replace
        attempts = []

        def refused_once(src, dst):
            attempts.append(Path(src).name)
            if len(attempts) == 1:
                raise PermissionError(13, "being used by another process", None, 32)
            return real(src, dst)

        with patch("os.replace", refused_once):
            cache_module._write_parquet_atomic(self._entry(), minimal_ohlcv)
        assert len(attempts) == 2
        assert [p.name for p in tmp_path.iterdir()] == [self._entry().name]

    def test_a_successful_write_leaves_exactly_the_entry(self, tmp_path, minimal_ohlcv):
        cache_module._write_parquet_atomic(self._entry(), minimal_ohlcv)
        assert [p.name for p in tmp_path.iterdir()] == [self._entry().name]
        pd.testing.assert_frame_equal(
            pd.read_parquet(self._entry()), minimal_ohlcv, check_freq=False
        )


# ── Orphaned temp files: collected, but never a live writer's ────────────────

_OLD_TEMP = "v3_yfinance_AAPL_2022-01-01_2022-06-01_1d.31852.32040.d5e6b0d1.tmp.parquet"
_NEW_TEMP = (
    ".v3_yfinance_AAPL_2022-01-01_2022-06-01_1d.parquet."
    "0123456789abcdef0123456789abcdef.tmp"
)


def _plant(root: Path, name: str, age_seconds: float) -> Path:
    path = root / name
    path.write_bytes(b"partial")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


class TestOrphanedTempFilesAreCollected:
    """
    A leaked temp was named with the CURRENT generation prefix, so the
    dead-generation collection never saw it and `sqt cache gc` could not
    remove it. Both names a cache writer has used are collected now, once
    they are older than any live write could be.
    """

    def test_old_orphans_are_listed_then_collected(self, tmp_path, capsys):
        old = _plant(tmp_path, _OLD_TEMP, 2 * 3600)
        new = _plant(tmp_path, _NEW_TEMP, 2 * 3600)
        entry = tmp_path / "v3_yfinance_AAPL_2022-01-01_2022-06-01_1d.parquet"
        entry.write_bytes(b"the entry")

        listed = {p.name for p in cache_module.orphaned_temps()}
        assert listed == {old.name, new.name}
        assert cli_main(["cache", "gc"]) == 0
        assert "Orphaned temp files (dry-run): 2 file(s)" in capsys.readouterr().out
        assert old.exists() and new.exists()

        assert cli_main(["cache", "gc", "--confirm"]) == 0
        assert "Deleted orphaned temp files: 2 file(s)" in capsys.readouterr().out
        assert [p.name for p in tmp_path.iterdir()] == [entry.name]

    def test_a_young_temp_file_is_left_for_its_writer(self, tmp_path):
        young = _plant(tmp_path, _NEW_TEMP, 5)
        assert cache_module.orphaned_temps() == []
        assert cmd_cache_gc(confirm=True) == []
        assert young.exists()

    def test_only_names_a_cache_writer_produces_are_candidates(self, tmp_path):
        names = (
            "notes.tmp",
            ".hidden.tmp",
            "v3_yfinance_AAPL_a_b_1d.parquet",
            "unversioned.1.2.abcdef12.tmp.parquet",
        )
        for name in names:
            _plant(tmp_path, name, 2 * 3600)
        assert cache_module.orphaned_temps() == []
        cmd_cache_gc(confirm=True)
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(names)

    def test_a_dead_generation_temp_is_listed_once(self, tmp_path):
        dead = _plant(
            tmp_path, "v1_yfinance_AAPL_a_b_1d.1.2.abcdef12.tmp.parquet", 7200
        )
        assert [p.name for p in cache_module.dead_generations()] == [dead.name]
        assert cache_module.orphaned_temps() == []
        assert [p.name for p in cmd_cache_gc()] == [dead.name]

    def test_a_negative_age_is_refused(self):
        with pytest.raises(ValidationError, match="min_age_seconds"):
            cache_module.orphaned_temps(min_age_seconds=-1)


# ── The read: what the live path checks, for every provider ───────────────────

_WINDOW = ("2025-03-03", "2025-03-07")
_SESSIONS = pd.bdate_range(*_WINDOW)
_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
_DATABENTO_RANGES = {
    DATASET_SUMMARY: ("2024-07-01T00:00:00+00:00", "2026-09-02T00:00:00+00:00"),
    CONSOLIDATED: SINCE_2023,
    BASIC: WIDE,
    DEPTH: WIDE,
}


def _full_frame(index=_SESSIONS, close: float = 200.0) -> pd.DataFrame:
    price = close + np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "Open": price,
            "High": price + 1,
            "Low": price - 1,
            "Close": price,
            "Volume": np.full(len(index), 1_000_000.0),
        },
        index=index,
    )


class _Yfinance:
    name = "yfinance"

    def entry(self) -> Path:
        return _parquet_path("AAPL", *_WINDOW, "1d")

    def fetch(self):
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.history.return_value = _full_frame(close=100.0).rename(
                columns=str.lower
            )
            frame = YFinanceProvider().get_ohlcv("AAPL", *_WINDOW)
        return frame, ticker.return_value.history.call_count


class _Polygon:
    name = "polygon"

    def entry(self) -> Path:
        return _parquet_path("AAPL", *_WINDOW, "1d", provider="polygon")

    def fetch(self):
        results = [
            {
                "o": 100.0 + i,
                "h": 101.0 + i,
                "l": 99.0 + i,
                "c": 100.0 + i,
                "v": 1_000_000,
                "t": int(pd.Timestamp(day, tz="America/New_York").value // 10**6),
            }
            for i, day in enumerate(_SESSIONS)
        ]
        with patch(
            "standard_quant_tools.data.polygon_provider._polygon_get",
            return_value={"status": "OK", "results": results},
        ) as get:
            frame = PolygonProvider(api_key="test-key").get_ohlcv("AAPL", *_WINDOW)
        return frame, get.call_count


class _Databento:
    name = "databento"

    def entry(self) -> Path:
        return _parquet_path(
            "AAPL", *_WINDOW, "1d", provider=f"databento-{DATASET_SUMMARY}"
        )

    def fetch(self):
        client = StubClient(_DATABENTO_RANGES)
        frame = _databento(client).get_ohlcv("AAPL", *_WINDOW)
        return frame, len(client.calls)


_PROVIDERS = [_Yfinance(), _Polygon(), _Databento()]
_IDS = [p.name for p in _PROVIDERS]


@pytest.fixture
def _quiet_vendors(monkeypatch):
    for name in (
        "DATABENTO_DATASET",
        "DATABENTO_DEPTH_DATASET",
        "DATABENTO_OHLCV_DATASET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("standard_quant_tools.data._retry.time.sleep", lambda s: None)


def _planted(provider, frame: pd.DataFrame) -> Path:
    path = provider.entry()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return path


@pytest.mark.usefixtures("_quiet_vendors")
class TestTheCacheReadChecksWhatTheLivePathChecks:
    """
    Every provider's read returned whatever the file held once it parsed,
    so a readable Parquet file that was not a plausible answer -- one
    `Close` column, another window's bars, a null close -- was served as a
    hit with no request made. The live paths refuse each of those. The one
    shared read now does too, evicts the entry and fetches.
    """

    @pytest.mark.parametrize("provider", _PROVIDERS, ids=_IDS)
    def test_a_close_only_file_is_evicted_and_refetched(self, provider):
        path = _planted(
            provider, pd.DataFrame({"Close": [999.0] * len(_SESSIONS)}, _SESSIONS)
        )
        frame, calls = provider.fetch()
        assert calls == 1
        assert list(frame.columns) == _COLUMNS
        assert (frame["Close"] != 999.0).all()
        assert list(pd.read_parquet(path).columns) == _COLUMNS

    @pytest.mark.parametrize("provider", _PROVIDERS, ids=_IDS)
    def test_another_windows_bars_are_evicted_and_refetched(self, provider):
        _planted(
            provider,
            _full_frame(index=pd.bdate_range("2023-01-02", "2023-01-06"), close=999.0),
        )
        frame, calls = provider.fetch()
        assert calls == 1
        assert frame.index.min() >= pd.Timestamp(_WINDOW[0])
        assert (frame["Close"] != 999.0).all()

    @pytest.mark.parametrize("provider", _PROVIDERS, ids=_IDS)
    def test_a_null_close_is_evicted_and_refetched(self, provider):
        bad = _full_frame(close=999.0)
        bad.iloc[2, bad.columns.get_loc("Close")] = np.nan
        _planted(provider, bad)
        frame, calls = provider.fetch()
        assert calls == 1
        assert frame["Close"].notna().all()

    @pytest.mark.parametrize("provider", _PROVIDERS, ids=_IDS)
    def test_a_plausible_entry_is_served_without_a_request(self, provider):
        _planted(provider, _full_frame(close=200.0))
        frame, calls = provider.fetch()
        assert calls == 0
        assert frame["Close"].tolist() == [200.0, 201.0, 202.0, 203.0, 204.0]

    def test_a_live_answer_the_read_would_refuse_is_not_written(self):
        """Writing it would only buy an eviction and a refetch on the next
        call; the answer is served live either way."""
        outside = _full_frame(index=pd.bdate_range("2023-01-02", "2023-01-06"))
        with patch("yfinance.Ticker") as ticker:
            ticker.return_value.history.return_value = outside.rename(columns=str.lower)
            YFinanceProvider().get_ohlcv("AAPL", *_WINDOW)
        assert not _parquet_path("AAPL", *_WINDOW, "1d").exists()


def _sharing_violation(*_args, **_kwargs):
    raise PermissionError(
        13, "The process cannot access the file because it is being used"
    )


@pytest.mark.usefixtures("_quiet_vendors")
class TestASharingViolationIsNotCorruption:
    """
    On Windows a reader that opens an entry while another process renames a
    new version over it gets `PermissionError`. Every provider treated that
    as corruption and deleted a valid entry, costing a metered refetch; and
    in yfinance and Polygon the delete itself was unguarded, so a second
    refusal escaped into the retry layer and came out as an APIError with
    no request made. The read is retried now, and a read that still cannot
    open is a miss that keeps the entry.
    """

    @pytest.mark.parametrize("provider", _PROVIDERS, ids=_IDS)
    def test_a_read_refused_once_is_retried_and_the_entry_kept(self, provider):
        path = _planted(provider, _full_frame(close=200.0))
        real = pd.read_parquet
        attempts = []

        def refused_once(*args, **kwargs):
            attempts.append(args)
            if len(attempts) == 1:
                _sharing_violation()
            return real(*args, **kwargs)

        with patch.object(pd, "read_parquet", refused_once):
            frame, calls = provider.fetch()
        assert len(attempts) == 2
        assert calls == 0
        assert path.exists()
        assert frame["Close"].iloc[0] == 200.0

    @pytest.mark.parametrize("provider", _PROVIDERS, ids=_IDS)
    def test_a_refused_read_and_a_refused_delete_serve_the_live_answer(self, provider):
        path = _planted(provider, _full_frame(close=200.0))
        with (
            patch.object(pd, "read_parquet", _sharing_violation),
            patch.object(Path, "unlink", _sharing_violation),
        ):
            frame, calls = provider.fetch()
        assert calls == 1
        assert list(frame.columns) == _COLUMNS
        assert path.exists()

    def test_content_that_does_not_parse_is_still_evicted(self, tmp_path):
        """The null case: corruption is a ValueError from the Parquet reader,
        not a permission error, and it is evicted as before."""
        path = _planted(_Yfinance(), _full_frame())
        path.write_bytes(b"this is not a valid parquet file")
        assert cache_module._read_cached_ohlcv(path, "1d", *_WINDOW) is None
        assert not path.exists()
