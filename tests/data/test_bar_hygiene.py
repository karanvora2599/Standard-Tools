"""
Bars a provider cannot serve as they came, and a last bar still trading.

EVERY TEST HERE IS OFFLINE. yfinance is replaced by a stub that answers
from a frame the test builds; Databento by an injected client; Polygon and
Bloomberg are exercised through their pure parsers and a stubbed request.

The three conditions, each with the clean series beside it:

  - a row after the last bar with a Close -- the placeholder yfinance lists
    outside market hours for the next session -- is dropped and disclosed,
    and the frame is otherwise the series without it;
  - a row with no Close inside the window is dropped as a missing bar;
  - a last daily bar whose session has not closed is kept and flagged.

A window with no Close at all is refused once, not retried.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.data._cache as cache_module
import standard_quant_tools.data.bar_hygiene as hygiene
import standard_quant_tools.data.yfinance_provider as yf_module
from standard_quant_tools.data.bar_hygiene import (
    CME_TRADE_DATE,
    DISCLOSURE_KEYS,
    MISSING_KEY,
    PARTIAL_CLOSE_KEY,
    PARTIAL_KEY,
    PARTIAL_SESSION_KEY,
    PLACEHOLDER_KEY,
    US_EQUITY,
    UTC_DAY,
    collect_served_bars,
    drop_unusable_closes,
)
from standard_quant_tools.data.yfinance_provider import YFinanceProvider
from standard_quant_tools.error import (
    APIError,
    NonRetryableAPIError,
    VendorUnavailableError,
)

# In nanoseconds, the unit the provider returns: pandas 3 builds a range in
# microseconds, and the frames would then differ in their index dtype alone.
SESSIONS = pd.bdate_range("2026-06-01", "2026-09-29").as_unit("ns")
LAST = "2026-09-29"
NEXT = "2026-09-30"


def _bars(index: pd.DatetimeIndex, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index))))
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.uniform(1e6, 5e6, len(index)),
        },
        index=index,
    )


def _no_close_row(day: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [np.nan],
            "High": [np.nan],
            "Low": [np.nan],
            "Close": [np.nan],
            "Volume": [0.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp(day)]),
    )


class _Market:
    """`yfinance`, answering every symbol from one frame and counting the
    requests made."""

    def __init__(self, frames: Dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: List[str] = []
        self.failure: Optional[BaseException] = None
        market = self

        class _Ticker:
            def __init__(self, symbol: str) -> None:
                self.symbol = symbol

            def history(self, start=None, end=None, interval="1d", **_kw):
                market.calls.append(self.symbol)
                if market.failure is not None:
                    raise market.failure
                frame = market.frames[self.symbol].copy()
                index = pd.DatetimeIndex(frame.index)
                keep = np.ones(len(frame), dtype=bool)
                if start is not None:
                    keep &= index >= pd.Timestamp(start).tz_localize(None).normalize()
                if end is not None:
                    keep &= index < pd.Timestamp(end).tz_localize(None)
                frame = frame[keep]
                frame.index = pd.DatetimeIndex(frame.index).tz_localize(
                    "America/New_York"
                )
                return frame

        self.Ticker = _Ticker


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """A cache of this test's own, no retry wait, and an empty session tier
    on the way in and out."""
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr("standard_quant_tools.data._retry.time.sleep", lambda s: None)
    with cache_module._session_cache_lock:
        cache_module._session_cache.clear()
    yield
    with cache_module._session_cache_lock:
        cache_module._session_cache.clear()


def _install(monkeypatch, frames: Dict[str, pd.DataFrame]) -> _Market:
    market = _Market(frames)
    monkeypatch.setattr(yf_module, "yf", market)
    return market


def _now(monkeypatch, instant: str) -> None:
    monkeypatch.setattr(hygiene, "_utc_now", lambda: pd.Timestamp(instant, tz="UTC"))


def _no_disclosures(frame: pd.DataFrame) -> bool:
    return not any(key in frame.attrs for key in DISCLOSURE_KEYS)


# ── a trailing row with no Close ─────────────────────────────────────────────


class TestTheOvernightPlaceholderIsDropped:
    def test_the_series_comes_back_without_it_and_says_so(self, monkeypatch, caplog):
        clean = _bars(SESSIONS)
        _install(monkeypatch, {"AAPL": pd.concat([clean, _no_close_row(NEXT)])})
        _now(monkeypatch, "2026-10-01 00:05")

        with caplog.at_level(logging.WARNING), collect_served_bars() as served:
            frame = YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", NEXT)

        pd.testing.assert_frame_equal(frame, clean, check_freq=False)
        assert frame.attrs[PLACEHOLDER_KEY] == [NEXT]
        assert MISSING_KEY not in frame.attrs
        assert any("trailing bar(s) with no Close" in r.message for r in caplog.records)
        (warning,) = served.warnings()
        assert warning.startswith("AAPL: 1 bar(s) at the end of the window")
        assert NEXT in warning

    def test_a_clean_closed_series_is_untouched(self, monkeypatch, caplog):
        clean = _bars(SESSIONS)
        _install(monkeypatch, {"AAPL": clean})
        _now(monkeypatch, "2026-10-01 00:05")

        with caplog.at_level(logging.WARNING), collect_served_bars() as served:
            frame = YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", LAST)

        pd.testing.assert_frame_equal(frame, clean, check_freq=False)
        assert _no_disclosures(frame)
        assert served.warnings() == []
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_a_row_past_the_window_is_trimmed_not_reported(self, monkeypatch):
        """The window ends before the placeholder's date, so the row was
        never part of the answer and nothing was dropped from it."""
        clean = _bars(SESSIONS)
        _install(monkeypatch, {"AAPL": pd.concat([clean, _no_close_row(NEXT)])})

        frame = YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", LAST)

        pd.testing.assert_frame_equal(frame, clean, check_freq=False)
        assert _no_disclosures(frame)


# ── a hole inside the window ─────────────────────────────────────────────────


class TestAHoleIsDroppedAsAMissingBar:
    def test_the_row_is_dropped_and_named(self, monkeypatch):
        clean = _bars(SESSIONS)
        holed = clean.copy()
        holed.iloc[[20, 45], holed.columns.get_loc("Close")] = np.nan
        _install(monkeypatch, {"AAPL": holed})

        with collect_served_bars() as served:
            frame = YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", LAST)

        expected = clean.drop(clean.index[[20, 45]])
        pd.testing.assert_frame_equal(frame, expected, check_freq=False)
        dates = [str(clean.index[i].date()) for i in (20, 45)]
        assert frame.attrs[MISSING_KEY] == dates
        assert PLACEHOLDER_KEY not in frame.attrs
        (warning,) = served.warnings()
        assert "dropped as missing bars" in warning
        assert all(d in warning for d in dates)

    def test_what_is_cached_is_what_was_served(self, monkeypatch):
        """A historical window is written to disk after the drop, with its
        disclosure, and a fresh provider reading it back gets the same frame
        and the same attrs without asking the vendor again."""
        holed = _bars(SESSIONS)
        holed.iloc[20, holed.columns.get_loc("Close")] = np.nan
        market = _install(monkeypatch, {"AAPL": holed})

        live = YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", LAST)
        cached = YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", LAST)

        assert market.calls == ["AAPL"]
        pd.testing.assert_frame_equal(cached, live, check_freq=False)
        assert cached.attrs[MISSING_KEY] == live.attrs[MISSING_KEY]
        assert not cached["Close"].isna().any()

    def test_a_cached_file_holding_a_null_close_is_still_evicted(self, monkeypatch):
        clean = _bars(SESSIONS)
        market = _install(monkeypatch, {"AAPL": clean})
        path = cache_module._parquet_path(
            "AAPL", "2026-06-01", LAST, "1d", provider="yfinance"
        )
        stale = clean.copy()
        stale.iloc[5, stale.columns.get_loc("Close")] = np.nan
        path.parent.mkdir(parents=True, exist_ok=True)
        stale.to_parquet(path)

        frame = YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", LAST)

        assert market.calls == ["AAPL"], "the null-Close file was served"
        pd.testing.assert_frame_equal(frame, clean, check_freq=False)

    def test_the_cache_still_refuses_to_store_a_null_close(self, tmp_path):
        holed = _bars(SESSIONS)
        holed.iloc[3, holed.columns.get_loc("Close")] = np.nan
        path = tmp_path / "cache" / "x.parquet"
        cache_module._write_cached_ohlcv(path, holed, "1d", "2026-06-01", LAST)
        assert not path.exists()


# ── nothing usable ───────────────────────────────────────────────────────────


class TestADeterministicRefusalIsNotRetried:
    def test_a_window_with_no_close_at_all_is_refused_once(self, monkeypatch):
        empty = _bars(SESSIONS)
        empty["Close"] = np.nan
        market = _install(monkeypatch, {"AAPL": empty})

        with pytest.raises(NonRetryableAPIError, match="none of them has a Close"):
            YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", LAST)
        assert market.calls == ["AAPL"]

    def test_missing_columns_are_refused_once(self, monkeypatch):
        market = _install(monkeypatch, {"AAPL": _bars(SESSIONS).drop(columns="Volume")})

        with pytest.raises(NonRetryableAPIError, match="missing columns"):
            YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", LAST)
        assert market.calls == ["AAPL"]

    def test_a_transient_failure_is_still_retried(self, monkeypatch):
        """Three attempts, then named as the vendor's failure -- a
        `VendorUnavailableError`, which no outer retry repeats (see the
        CHANGELOG entry of 2026-10-02)."""
        market = _install(monkeypatch, {"AAPL": _bars(SESSIONS)})
        market.failure = ConnectionError("connection reset by peer")

        with pytest.raises(APIError) as caught:
            YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", LAST)
        assert isinstance(caught.value, VendorUnavailableError)
        assert "No data found" not in str(caught.value)
        assert market.calls == ["AAPL"] * 3

    def test_an_impossible_date_is_refused_before_any_request(self, monkeypatch):
        """'2019-13-45' has the shape of a date, reached the vendor, failed
        there, and was asked three times."""
        from standard_quant_tools.error import ValidationError

        market = _install(monkeypatch, {"AAPL": _bars(SESSIONS)})
        with pytest.raises(ValidationError, match="not a calendar date"):
            YFinanceProvider().get_ohlcv("AAPL", "2019-13-45", LAST)
        assert market.calls == []

    def test_the_refusal_names_the_remedy(self):
        frame = _bars(SESSIONS[:3])
        frame["Close"] = np.nan
        with pytest.raises(NonRetryableAPIError) as caught:
            drop_unusable_closes(frame, "AAPL", provider="yfinance")
        text = str(caught.value)
        assert "Widen the window" in text and "another provider" in text


# ── a last bar still trading ─────────────────────────────────────────────────


class TestAFormingLastBarIsFlagged:
    def test_mid_session_the_last_bar_is_flagged(self, monkeypatch, caplog):
        _install(monkeypatch, {"AAPL": _bars(SESSIONS)})
        _now(monkeypatch, f"{LAST} 13:41")  # 09:41 in New York

        with caplog.at_level(logging.WARNING), collect_served_bars() as served:
            frame = YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", LAST)

        assert frame.attrs[PARTIAL_KEY] is True
        assert frame.attrs[PARTIAL_SESSION_KEY] == LAST
        assert frame.attrs[PARTIAL_CLOSE_KEY] == "2026-09-29T20:00:00+00:00"
        assert len(frame) == len(SESSIONS), "the bar is kept, not dropped"
        assert any("has not closed" in r.message for r in caplog.records)
        (warning,) = served.warnings()
        assert warning.startswith(f"AAPL: the last bar ({LAST}) is a session")

    def test_after_the_close_nothing_is_flagged(self, monkeypatch):
        _install(monkeypatch, {"AAPL": _bars(SESSIONS)})
        _now(monkeypatch, f"{LAST} 20:01")

        with collect_served_bars() as served:
            frame = YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", LAST)

        assert _no_disclosures(frame)
        assert served.warnings() == []

    def test_without_the_calendar_the_close_is_four_in_new_york(self, monkeypatch):
        monkeypatch.setattr(hygiene, "_exchange_calendar", lambda code: None)
        _install(monkeypatch, {"AAPL": _bars(SESSIONS)})

        _now(monkeypatch, f"{LAST} 19:59")
        assert (
            YFinanceProvider()
            .get_ohlcv("AAPL", "2026-06-01", LAST)
            .attrs.get(PARTIAL_KEY)
        )
        _now(monkeypatch, f"{LAST} 20:00")
        assert _no_disclosures(YFinanceProvider().get_ohlcv("AAPL", "2026-06-01", LAST))

    def test_the_calendar_knows_an_early_close(self, monkeypatch):
        """The day after Thanksgiving closes at 13:00 New York. At 13:30 the
        calendar says the session is over; the fixed 16:00 does not."""
        pytest.importorskip("exchange_calendars")
        day = "2025-11-28"
        frame = _bars(
            pd.bdate_range("2025-11-03", day).drop(pd.Timestamp("2025-11-27"))
        )
        _install(monkeypatch, {"AAPL": frame})
        _now(monkeypatch, f"{day} 18:30")

        assert _no_disclosures(YFinanceProvider().get_ohlcv("AAPL", "2025-11-03", day))
        monkeypatch.setattr(hygiene, "_exchange_calendar", lambda code: None)
        assert (
            YFinanceProvider()
            .get_ohlcv("AAPL", "2025-11-03", day)
            .attrs.get(PARTIAL_KEY)
        )

    def test_a_cached_frame_is_judged_when_it_is_served(self, monkeypatch):
        _install(monkeypatch, {"AAPL": _bars(SESSIONS)})
        provider = YFinanceProvider()
        _now(monkeypatch, f"{LAST} 13:41")
        assert provider.get_ohlcv("AAPL", "2026-06-01", LAST).attrs.get(PARTIAL_KEY)
        _now(monkeypatch, f"{LAST} 21:00")
        assert _no_disclosures(provider.get_ohlcv("AAPL", "2026-06-01", LAST))

    def test_an_intraday_bar_is_never_partial(self, monkeypatch):
        index = pd.date_range(f"{LAST} 13:30", f"{LAST} 15:30", freq="1h")
        _install(monkeypatch, {"AAPL": _bars(index)})
        _now(monkeypatch, f"{LAST} 15:45")

        frame = YFinanceProvider().get_ohlcv("AAPL", LAST, LAST, interval="1h")

        assert len(frame) == 3
        assert _no_disclosures(frame)

    def test_a_future_closes_with_its_cme_trade_date(self, monkeypatch):
        _install(monkeypatch, {"ES=F": _bars(SESSIONS)})
        _now(monkeypatch, f"{LAST} 20:30")  # 15:30 in Chicago
        assert (
            YFinanceProvider()
            .get_ohlcv("ES=F", "2026-06-01", LAST)
            .attrs.get(PARTIAL_KEY)
        )
        _now(monkeypatch, f"{LAST} 21:30")  # 16:30 in Chicago
        assert _no_disclosures(YFinanceProvider().get_ohlcv("ES=F", "2026-06-01", LAST))

    def test_a_weekly_bar_is_partial_until_its_week_closes(self, monkeypatch):
        weeks = pd.date_range("2026-06-01", "2026-09-28", freq="W-MON")
        _install(monkeypatch, {"AAPL": _bars(weeks)})
        _now(monkeypatch, "2026-09-30 15:00")  # Wednesday of the last week

        frame = YFinanceProvider().get_ohlcv(
            "AAPL", "2026-06-01", "2026-09-30", interval="1wk"
        )
        assert frame.attrs[PARTIAL_KEY] is True
        assert frame.attrs[PARTIAL_CLOSE_KEY] == "2026-10-02T20:00:00+00:00"

        closed = YFinanceProvider().get_ohlcv(
            "AAPL", "2026-06-01", "2026-09-27", interval="1wk"
        )
        assert _no_disclosures(closed)


class TestTheClocks:
    def test_the_us_equity_close_without_a_calendar(self, monkeypatch):
        monkeypatch.setattr(hygiene, "_exchange_calendar", lambda code: None)
        assert str(US_EQUITY.session_close(pd.Timestamp("2026-01-15"))) == (
            "2026-01-15 21:00:00+00:00"
        )
        assert str(US_EQUITY.session_close(pd.Timestamp("2026-07-15"))) == (
            "2026-07-15 20:00:00+00:00"
        )

    def test_the_cme_trade_date_closes_at_four_in_chicago(self):
        assert str(CME_TRADE_DATE.session_close(pd.Timestamp("2026-07-15"))) == (
            "2026-07-15 21:00:00+00:00"
        )

    def test_a_utc_day_closes_at_midnight(self):
        assert str(UTC_DAY.session_close(pd.Timestamp("2026-07-15"))) == (
            "2026-07-16 00:00:00+00:00"
        )

    def test_the_yahoo_symbol_picks_its_clock(self):
        clock = yf_module._session_clock
        assert clock("AAPL") is US_EQUITY
        assert clock("BRK-B") is US_EQUITY
        assert clock("ES=F") is CME_TRADE_DATE
        assert clock("BTC-USD") is UTC_DAY
        assert clock("VOD.L").calendar == "XLON"
        assert clock("VOD.L").timezone == "Europe/London"


# ── Databento ────────────────────────────────────────────────────────────────


class _Store:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_df(self) -> pd.DataFrame:
        return self._frame


class _DatabentoClient:
    """Answers every bar request with one float-dollar frame, sliced to the
    window, and counts the requests."""

    def __init__(self, frame: pd.DataFrame, served_by: str) -> None:
        self.calls: List[dict] = []
        owner = self

        class _Metadata:
            def get_dataset_range(self, dataset):
                if dataset != served_by:
                    raise RuntimeError(f"403 not_entitled for {dataset}")
                return {
                    "start": "2018-01-01T00:00:00+00:00",
                    "end": "2026-09-30T00:00:00+00:00",
                }

        class _Timeseries:
            def get_range(self, **kwargs):
                owner.calls.append(kwargs)
                if kwargs["dataset"] != served_by:
                    raise RuntimeError(f"403 not_entitled for {kwargs['dataset']}")
                start = pd.Timestamp(str(kwargs["start"]))
                end = pd.Timestamp(str(kwargs["end"]))
                start = start if start.tzinfo else start.tz_localize("UTC")
                end = end if end.tzinfo else end.tz_localize("UTC")
                index = frame.index
                return _Store(frame[(index >= start) & (index < end)])

        self.metadata = _Metadata()
        self.timeseries = _Timeseries()


def _vendor_bars(index: pd.DatetimeIndex) -> pd.DataFrame:
    bars = _bars(index)
    out = pd.DataFrame(
        {
            "open": bars["Open"].to_numpy(),
            "high": bars["High"].to_numpy(),
            "low": bars["Low"].to_numpy(),
            "close": bars["Close"].to_numpy(),
            "volume": np.full(len(index), 1_000_000, dtype="int64"),
        },
        index=index.tz_localize("UTC"),
    )
    return out


class TestDatabentoFollowsTheSameRule:
    @pytest.fixture
    def summary(self):
        from standard_quant_tools.data.databento import DATASET_SUMMARY

        return DATASET_SUMMARY

    def _provider(self, client):
        from standard_quant_tools.data.databento_provider import DatabentoProvider

        return DatabentoProvider(api_key="not-used", client=client)

    def test_a_null_close_is_dropped_not_refused(self, monkeypatch, summary):
        """It used to refuse the whole frame ("contain a null Close")."""
        _now(monkeypatch, "2026-10-01 12:00")
        raw = _vendor_bars(SESSIONS)
        raw.iloc[10, raw.columns.get_loc("close")] = np.nan
        raw.iloc[-1, raw.columns.get_loc("close")] = np.nan
        client = _DatabentoClient(raw, summary)

        with collect_served_bars() as served:
            frame = self._provider(client).get_ohlcv("NVDA", "2026-06-01", LAST)

        assert frame.attrs[MISSING_KEY] == [str(SESSIONS[10].date())]
        assert frame.attrs[PLACEHOLDER_KEY] == [LAST]
        assert len(frame) == len(SESSIONS) - 2
        assert not frame["Close"].isna().any()
        assert len(served.warnings()) == 2

    def test_a_clean_frame_carries_no_disclosure(self, monkeypatch, summary):
        _now(monkeypatch, "2026-10-01 12:00")
        client = _DatabentoClient(_vendor_bars(SESSIONS), summary)
        with collect_served_bars() as served:
            frame = self._provider(client).get_ohlcv("NVDA", "2026-06-01", LAST)
        assert _no_disclosures(frame)
        assert served.warnings() == []

    def test_a_window_with_no_close_is_refused_once(self, monkeypatch, summary):
        raw = _vendor_bars(SESSIONS)
        raw["close"] = np.nan
        client = _DatabentoClient(raw, summary)

        with pytest.raises(NonRetryableAPIError, match="none of them has a Close"):
            self._provider(client).get_ohlcv("NVDA", "2026-06-01", LAST)
        assert len([c for c in client.calls if c["dataset"] == summary]) == 1

    def test_missing_columns_are_refused_once(self, monkeypatch, summary):
        raw = _vendor_bars(SESSIONS).drop(columns="volume")
        client = _DatabentoClient(raw, summary)

        with pytest.raises(NonRetryableAPIError, match="missing"):
            self._provider(client).get_ohlcv("NVDA", "2026-06-01", LAST)
        assert len([c for c in client.calls if c["dataset"] == summary]) == 1

    def test_the_summary_feed_closes_with_the_session(self, monkeypatch, summary):
        client = _DatabentoClient(_vendor_bars(SESSIONS), summary)
        _now(monkeypatch, f"{LAST} 19:00")
        assert (
            self._provider(client)
            .get_ohlcv("NVDA", "2026-06-01", LAST)
            .attrs.get(PARTIAL_KEY)
        )
        _now(monkeypatch, f"{LAST} 20:30")
        assert _no_disclosures(
            self._provider(client).get_ohlcv("NVDA", "2026-06-01", LAST)
        )

    def test_a_utc_day_feed_is_forming_until_utc_midnight(self):
        from standard_quant_tools.data.databento_provider import _bar_clock

        frame = pd.DataFrame()
        frame.attrs["dataset"] = "EQUS.MINI"
        assert _bar_clock("NVDA", frame) is UTC_DAY
        frame.attrs["dataset"] = "EQUS.SUMMARY"
        assert _bar_clock("NVDA", frame) is US_EQUITY
        assert _bar_clock("ES.c.0", frame) is CME_TRADE_DATE


# ── Polygon and Bloomberg ────────────────────────────────────────────────────


class TestPolygonAndBloombergFollowTheSameRule:
    def test_a_polygon_bar_with_no_close_is_a_missing_bar(self, monkeypatch):
        import standard_quant_tools.data.polygon_provider as polygon

        stamps = [
            int(pd.Timestamp(d, tz="America/New_York").value // 1_000_000)
            for d in SESSIONS[-5:]
        ]
        results = [
            {"o": 10.0, "h": 11.0, "l": 9.0, "c": 10.5, "v": 100, "t": t}
            for t in stamps
        ]
        results[2] = {"v": 0, "t": stamps[2]}
        calls: List[str] = []

        def _get(path, params, api_key):
            calls.append(path)
            return {"results": results}

        monkeypatch.setattr(polygon, "_polygon_get", _get)
        _now(monkeypatch, "2026-10-01 12:00")

        with collect_served_bars() as served:
            frame = polygon.PolygonProvider(api_key="k").get_ohlcv(
                "AAPL", str(SESSIONS[-5].date()), LAST
            )

        assert len(frame) == 4
        assert frame.attrs[MISSING_KEY] == [str(SESSIONS[-3].date())]
        assert len(served.warnings()) == 1
        assert len(calls) == 1

    def test_a_polygon_bar_with_a_close_but_no_high_is_refused_once(self):
        from standard_quant_tools.data.polygon_provider import _parse_aggs

        with pytest.raises(NonRetryableAPIError, match="missing"):
            _parse_aggs(
                [{"o": 1.0, "l": 0.5, "c": 1.0, "t": 1672704000000}], "X", "day"
            )

    def test_a_bloomberg_bar_with_no_last_price_is_a_missing_bar(self):
        import datetime

        from standard_quant_tools.data.bloomberg_provider import _parse_historical_bars

        bars = [
            {
                "date": datetime.date(2023, 1, 3),
                "PX_OPEN": 1.0,
                "PX_HIGH": 2.0,
                "PX_LOW": 0.5,
                "PX_LAST": 1.5,
            },
            {"date": datetime.date(2023, 1, 4)},
            {
                "date": datetime.date(2023, 1, 5),
                "PX_OPEN": 1.0,
                "PX_HIGH": 2.0,
                "PX_LOW": 0.5,
                "PX_LAST": 1.6,
            },
        ]
        parsed = _parse_historical_bars(bars, "AAPL")
        kept = drop_unusable_closes(parsed, "AAPL", provider="bloomberg")
        assert list(kept["Close"]) == [1.5, 1.6]
        assert kept.attrs[MISSING_KEY] == ["2023-01-04"]
