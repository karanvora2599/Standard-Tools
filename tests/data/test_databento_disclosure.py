"""
What the Databento provider now says about the data it serves, and the
answers it no longer gets wrong -- every one against a stubbed client.

Each class plants an answer the old code got wrong without a word: a tape
and quotes from two venues, a rejected key reported as a date range, a
receive stamp that could not be chosen, vendor warnings dropped on the
floor, a futures week of six UTC days, a window served whole by the sample
feed, an intraday end one bar short, a bad symbol costing a dozen
requests, a currency future refused, and prints of no size counted as
trades. Each has a null case beside it: the clean input must stay quiet.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import resolve as resolve_runtime
from standard_quant_tools.data import _cache as cache_module
from standard_quant_tools.data.databento import (
    DATASET_FUTURES,
    DATASET_SUMMARY,
    cross_venue_warning,
    normalize_trades,
    print_counts,
)
from standard_quant_tools.data.databento_provider import DatabentoProvider
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import (
    DataNotFoundError,
    InvalidSymbolError,
    NonRetryableAPIError,
    ValidationError,
    VendorUnavailableError,
)

from .test_databento_provider import (
    BASIC,
    CONSOLIDATED,
    DEPTH,
    SINCE_2023,
    WIDE,
    StubClient,
    _provider,
)

SUMMARY = DATASET_SUMMARY
SINCE_2024_07 = ("2024-07-01T00:00:00+00:00", "2026-09-02T00:00:00+00:00")
ALL = {SUMMARY: SINCE_2024_07, CONSOLIDATED: SINCE_2023, BASIC: WIDE, DEPTH: WIDE}
PROVIDER_LOG = "standard_quant_tools.data.databento_provider"

TAPE_START, TAPE_END = "2025-03-03T14:30:00", "2025-03-03T14:31:00"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr("standard_quant_tools.data._retry.time.sleep", lambda s: None)
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    for name in (
        "DATABENTO_DATASET",
        "DATABENTO_DEPTH_DATASET",
        "DATABENTO_OHLCV_DATASET",
    ):
        monkeypatch.delenv(name, raising=False)


# ── vendor-shaped frames ─────────────────────────────────────────────────


def _stamps(n):
    base = pd.Timestamp("2025-03-03 14:30", tz="UTC")
    event = base + pd.to_timedelta(np.arange(n) * 1000, unit="us")
    # 259 microseconds of capture latency on every record.
    return event, event + pd.Timedelta(microseconds=259)


def _trades(sizes=None, prices=None, flags=None, publishers=None):
    """A trades frame the way `.to_df()` returns one: `ts_recv` on the
    INDEX, `ts_event` a column, float dollars."""
    sizes = [100, 200, 50, 300, 10] if sizes is None else sizes
    n = len(sizes)
    prices = [250.0, 250.01, 250.0, 250.02, 250.01][:n] if prices is None else prices
    event, recv = _stamps(n)
    data = {
        "ts_event": event,
        "price": np.asarray(prices, dtype=float),
        "size": np.asarray(sizes, dtype="uint32"),
        "flags": np.asarray(flags if flags is not None else [0] * n, dtype="uint8"),
    }
    if publishers is not None:
        data["publisher_id"] = np.asarray(publishers, dtype="uint16")
    return pd.DataFrame(data, index=pd.DatetimeIndex(recv, name="ts_recv"))


def _quotes(n=5):
    event, recv = _stamps(n)
    return pd.DataFrame(
        {
            "ts_event": event,
            "bid_px_00": np.full(n, 249.99),
            "ask_px_00": np.full(n, 250.03),
            "bid_sz_00": np.full(n, 300, dtype="uint32"),
            "ask_sz_00": np.full(n, 300, dtype="uint32"),
            "flags": np.zeros(n, dtype="uint8"),
        },
        index=pd.DatetimeIndex(recv, name="ts_recv"),
    )


def _by_schema(trades=None, quotes=None, refuse=None):
    """A rule answering trades and quotes with their own frames, and
    refusing `(dataset, schema)` pairs in `refuse` the way a venue that
    does not carry a schema does."""
    refuse = set(refuse or ())

    def rule(kw):
        if (kw["dataset"], kw["schema"]) in refuse:
            return RuntimeError(f"422 {kw['schema']} is not available for this dataset")
        if kw["schema"] == "trades":
            return _trades() if trades is None else trades
        if kw["schema"] == "mbp-1":
            return _quotes() if quotes is None else quotes
        return None

    return rule


class _HttpError(Exception):
    """The shape of the vendor client's HTTP errors: a status attribute,
    and a message that starts with the request id."""

    def __init__(self, status, text, request_id="a1b2c3"):
        self.http_status = status
        super().__init__(f"Request {request_id}: {status} {text}")


# ── 1. a tape and quotes from two venues ─────────────────────────────────


class TestATapeAndQuotesFromTwoVenuesAreDisclosed:
    """XNAS.BASIC carrying trades but not top-of-book quotes answered the
    tape while XNAS.ITCH answered the quotes, and nothing said so."""

    def _pair(self, **kw):
        client = StubClient(
            {BASIC: WIDE, DEPTH: WIDE},
            rules=[_by_schema(refuse={(BASIC, "mbp-1")})],
        )
        provider = _provider(client)
        trades = provider.get_trades("AAPL", TAPE_START, TAPE_END, **kw)
        quotes = provider.get_quotes("AAPL", TAPE_START, TAPE_END, **kw)
        return trades, quotes, client

    def test_the_fallback_is_named_on_the_frame_and_warned(self, caplog):
        with caplog.at_level(logging.WARNING, logger=PROVIDER_LOG):
            trades, quotes, _client = self._pair()
        assert trades.attrs["dataset"] == BASIC
        assert quotes.attrs["dataset"] == DEPTH
        assert quotes.attrs["fallback_from"] == [BASIC]
        warning = [n for n in quotes.attrs["vendor_notes"] if n.startswith("WARNING")]
        assert warning and BASIC in warning[0] and "dataset='XNAS.ITCH'" in warning[0]
        assert any("XNAS.BASIC" in r.getMessage() for r in caplog.records)

    def test_the_pair_is_flagged_where_it_is_paired(self):
        trades, quotes, _client = self._pair()
        message = cross_venue_warning(trades, quotes)
        assert message is not None
        assert "XNAS.BASIC" in message and "XNAS.ITCH" in message

    def test_a_pinned_dataset_asks_only_that_venue(self):
        client = StubClient({BASIC: WIDE, DEPTH: WIDE}, rules=[_by_schema()])
        provider = _provider(client)
        trades = provider.get_trades("AAPL", TAPE_START, TAPE_END, dataset=DEPTH)
        quotes = provider.get_quotes("AAPL", TAPE_START, TAPE_END, dataset=DEPTH)
        assert client.datasets_called() == [DEPTH, DEPTH]
        assert cross_venue_warning(trades, quotes) is None
        assert "fallback_from" not in quotes.attrs

    def test_an_empty_dataset_name_is_refused(self):
        provider = _provider(StubClient({BASIC: WIDE}, rules=[_by_schema()]))
        with pytest.raises(ValidationError, match="names no dataset"):
            provider.get_trades("AAPL", TAPE_START, TAPE_END, dataset="  ")

    def test_null_one_venue_for_both_says_nothing(self):
        client = StubClient({BASIC: WIDE, DEPTH: WIDE}, rules=[_by_schema()])
        provider = _provider(client)
        trades = provider.get_trades("AAPL", TAPE_START, TAPE_END)
        quotes = provider.get_quotes("AAPL", TAPE_START, TAPE_END)
        assert trades.attrs["dataset"] == quotes.attrs["dataset"] == BASIC
        assert cross_venue_warning(trades, quotes) is None
        assert not [n for n in quotes.attrs["vendor_notes"] if n.startswith("WARNING")]

    def test_null_a_frame_naming_no_dataset_is_not_judged(self):
        assert cross_venue_warning(pd.DataFrame(), pd.DataFrame()) is None


class TestTheAgentCarriesTheVenue:
    def _data(self, tool, arguments, provider):
        with patch.object(DataFactory, "get_provider", lambda *a, **k: provider):
            return resolve_runtime("data").dispatch(tool, arguments)

    def _args(self, name, **extra):
        return {
            "symbol": "AAPL",
            "start_date": TAPE_START,
            "end_date": TAPE_END,
            "run_id": "venues",
            "name": name,
            **extra,
        }

    def test_the_quote_fetch_warns_and_the_pairing_tool_repeats_it(self):
        client = StubClient(
            {BASIC: WIDE, DEPTH: WIDE},
            rules=[_by_schema(refuse={(BASIC, "mbp-1")})],
        )
        provider = _provider(client)
        tape = self._data("fetch_tick_tape", self._args("tape"), provider)
        panel = self._data("fetch_quote_panel", self._args("quotes"), provider)
        assert tape["dataset"] == BASIC and panel["dataset"] == DEPTH
        assert any("XNAS.BASIC" in w for w in panel["warnings"])
        spread = resolve_runtime("microstructure").dispatch(
            "get_effective_spread_series",
            {
                "tick_tape_ref": tape["ref"],
                "quote_panel_ref": panel["ref"],
                "run_id": "venues",
                "name": "spread",
            },
        )
        assert any(
            "trades come from XNAS.BASIC and the quotes from XNAS.ITCH" in w
            for w in spread["warnings"]
        )

    def test_the_dataset_argument_reaches_the_provider(self):
        client = StubClient({BASIC: WIDE, DEPTH: WIDE}, rules=[_by_schema()])
        provider = _provider(client)
        result = self._data(
            "fetch_tick_tape", self._args("pinned", dataset=DEPTH), provider
        )
        assert result["dataset"] == DEPTH
        assert client.datasets_called() == [DEPTH]

    def test_a_provider_without_named_datasets_refuses_the_argument(self):
        from tests.agent.test_microstructure_tools import _tape, _TickProvider

        provider = _TickProvider(*_tape([(0, 100.0, 1)], [(0, 99.9, 100.1, 1, 1)]))
        with pytest.raises(ValidationError, match="routes no named datasets"):
            self._data("fetch_tick_tape", self._args("x", dataset=DEPTH), provider)

    def test_the_metrics_tool_names_the_two_venues(self, monkeypatch):
        client = StubClient(
            {BASIC: WIDE, DEPTH: WIDE},
            rules=[_by_schema(refuse={(BASIC, "mbp-1")})],
        )
        provider = _provider(client)
        monkeypatch.setattr(
            "standard_quant_tools.agent.runtimes.portfolio.tools.DataFactory"
            ".get_provider",
            staticmethod(lambda *a, **k: provider),
        )
        from standard_quant_tools.agent.tools import dispatch

        result = dispatch(
            "get_microstructure_metrics",
            {
                "symbol": "AAPL",
                "start": TAPE_START,
                "end": TAPE_END,
                "realized_horizon_seconds": None,
                "source": "databento",
            },
        )
        assert any("quotes from XNAS.ITCH" in n for n in result["notes"])


# ── prints of no size, and prints at a fraction of a cent ────────────────


class TestZeroSizeAndSubPennyPrintsAreCounted:
    SIZES = [100, 0, 200, 0, 50, 0]
    PRICES = [250.0, 250.005, 250.01, 250.0, 250.0037, 250.02]
    PUBLISHERS = [81, 93, 81, 93, 93, 81]

    def test_the_normalizer_counts_both_by_publisher(self):
        frame = _trades(self.SIZES, self.PRICES, publishers=self.PUBLISHERS)
        out, notes = normalize_trades(frame)
        counts = print_counts(out)
        assert counts["zero_size"] == 3 and counts["sub_penny"] == 2
        assert counts["zero_size_by_publisher"] == {"93": 2, "81": 1}
        note = [n for n in notes if "size 0" in n]
        assert note and "3 of 6 prints" in note[0] and "kept" in note[0]

    def test_the_provider_puts_the_counts_on_the_tape(self):
        frame = _trades(self.SIZES, self.PRICES)
        client = StubClient({BASIC: WIDE}, rules=[_by_schema(trades=frame)])
        tape = _provider(client).get_trades("AAPL", TAPE_START, TAPE_END)
        assert tape.attrs["print_counts"]["zero_size"] == 3
        assert tape.attrs["print_counts"]["sub_penny"] == 2
        assert len(tape) == 6, "the prints are counted, not dropped"

    def test_the_fetch_tool_warns(self):
        frame = _trades(self.SIZES, self.PRICES)
        client = StubClient({BASIC: WIDE}, rules=[_by_schema(trades=frame)])
        provider = _provider(client)
        with patch.object(DataFactory, "get_provider", lambda *a, **k: provider):
            result = resolve_runtime("data").dispatch(
                "fetch_tick_tape",
                {
                    "symbol": "AAPL",
                    "start_date": TAPE_START,
                    "end_date": TAPE_END,
                    "run_id": "prints",
                    "name": "tape",
                },
            )
        assert any("3 zero-size and 2 sub-penny" in w for w in result["warnings"])

    def test_null_a_clean_tape_has_no_note(self):
        out, notes = normalize_trades(_trades())
        counts = print_counts(out)
        assert counts["zero_size"] == 0 and counts["sub_penny"] == 0
        assert not [n for n in notes if "size 0" in n or "fraction" in n]


# ── 2. an HTTP status is read, not a word in the message ─────────────────


class TestTheVendorsStatusDecidesWhatAFailureMeans:
    def test_a_rejected_key_is_named_after_one_lookup(self):
        client = StubClient(ALL)
        calls = []

        def rejected(dataset):
            calls.append(dataset)
            raise _HttpError(401, "auth_authentication_failed\nAuthentication failed.")

        client.metadata.get_dataset_range = rejected
        provider = _provider(client)
        with pytest.raises(NonRetryableAPIError, match="DATABENTO_API_KEY") as caught:
            provider.get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
        assert "No dataset covers" not in str(caught.value)
        assert len(calls) == 1 and client.calls == []
        assert provider._denied == set()

    def test_a_rejected_key_on_a_data_request_is_named_too(self):
        client = StubClient(ALL, rules=[lambda kw: _HttpError(401, "unauthorized")])
        with pytest.raises(NonRetryableAPIError, match="DATABENTO_API_KEY"):
            _provider(client).get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
        assert len(client.calls) == 1, "not retried, not passed to the next feed"

    def test_a_500_whose_request_id_contains_403_is_not_a_denial(self):
        """Nor an empty answer: it is the vendor's failure, asked once more
        on the same feed and then named, with no lesser feed asked in its
        place (see the CHANGELOG entry of 2026-10-02)."""
        failing = [
            lambda kw: (
                _HttpError(500, "internal error", request_id="f403e9")
                if kw["dataset"] == SUMMARY
                else None
            )
        ]
        client = StubClient(ALL, rules=failing)
        provider = _provider(client)
        with pytest.raises(VendorUnavailableError) as caught:
            provider.get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
        assert SUMMARY not in provider._denied
        assert client.datasets_called() == [SUMMARY, SUMMARY]
        assert caught.value.status == 500 and caught.value.dataset == SUMMARY
        assert "failed on its side" in str(caught.value)
        assert "returned no" not in str(caught.value)

    def test_the_word_author_is_not_an_entitlement(self):
        assert not DatabentoProvider._is_denial(
            RuntimeError("422 symbology author mismatch")
        )

    def test_null_a_403_is_remembered(self):
        denied = [
            lambda kw: (
                _HttpError(403, "not_entitled") if kw["dataset"] == SUMMARY else None
            )
        ]
        provider = _provider(StubClient(ALL, rules=denied))
        provider.get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
        assert SUMMARY in provider._denied

    def test_every_feed_declined_is_an_entitlement_refusal(self):
        client = StubClient({})  # every range lookup answers 403
        with pytest.raises(NonRetryableAPIError, match="entitlement problem") as caught:
            _provider(client).get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
        assert "No dataset covers" not in str(caught.value)
        assert client.calls == []


# ── 3. the receive stamp, where `.to_df()` puts it ───────────────────────


class TestTheReceiveStampIsFoundOnTheIndex:
    def test_auto_takes_ts_recv_from_the_index_and_keeps_both(self):
        frame = _trades()
        out, notes = normalize_trades(frame)
        recv = pd.DatetimeIndex(frame.index)
        assert (pd.DatetimeIndex(out["timestamp"]) == recv).all()
        assert (pd.DatetimeIndex(out["ts_recv"]) == recv).all()
        assert (pd.DatetimeIndex(out["ts_event"]) == frame["ts_event"]).all()
        assert any(n.startswith("timestamp taken from ts_recv") for n in notes)

    def test_asking_for_ts_recv_by_name_finds_it(self):
        out, _notes = normalize_trades(_trades(), timestamp="ts_recv")
        assert (pd.DatetimeIndex(out["timestamp"]) == _trades().index).all()

    def test_the_provider_indexes_by_it_and_says_so(self):
        client = StubClient({BASIC: WIDE}, rules=[_by_schema()])
        tape = _provider(client).get_trades("AAPL", TAPE_START, TAPE_END)
        assert tape.attrs["timestamp_source"] == "ts_recv"
        assert (tape.index == _trades().index).all()
        latency = pd.DatetimeIndex(tape["ts_recv"]) - pd.DatetimeIndex(tape["ts_event"])
        assert (latency == pd.Timedelta(microseconds=259)).all()

    def test_null_an_index_named_ts_event_is_ts_event(self):
        frame = _trades().reset_index(drop=True)
        frame.index = pd.DatetimeIndex(frame.pop("ts_event"), name="ts_event")
        out, notes = normalize_trades(frame)
        assert (pd.DatetimeIndex(out["timestamp"]) == frame.index).all()
        assert any(n.startswith("timestamp taken from ts_event") for n in notes)

    def test_null_a_missing_stamp_asked_for_by_name_is_refused(self):
        frame = _trades().reset_index(drop=True)
        with pytest.raises(ValidationError, match="ts_recv"):
            normalize_trades(frame, timestamp="ts_recv")


# ── 4. the vendor's warnings reach the result ────────────────────────────


class TestTheVendorsWarningsAreKept:
    def test_a_flagged_trade_reaches_the_frame_and_the_fetch(self, caplog):
        frame = _trades(flags=[0, 4, 0, 0, 0])
        client = StubClient({BASIC: WIDE}, rules=[_by_schema(trades=frame)])
        provider = _provider(client)
        with caplog.at_level(logging.WARNING, logger=PROVIDER_LOG):
            tape = provider.get_trades("AAPL", TAPE_START, TAPE_END)
        notes = tape.attrs["vendor_notes"]
        assert any(n.startswith("WARNING: 1 of 5 records set flag 4") for n in notes)
        assert any(n.startswith("price_scale=") for n in notes)
        assert any("flag 4" in r.getMessage() for r in caplog.records)
        with patch.object(DataFactory, "get_provider", lambda *a, **k: provider):
            result = resolve_runtime("data").dispatch(
                "fetch_tick_tape",
                {
                    "symbol": "AAPL",
                    "start_date": TAPE_START,
                    "end_date": TAPE_END,
                    "run_id": "flags",
                    "name": "tape",
                },
            )
        assert any("flag 4" in w for w in result["warnings"])
        assert any(n.startswith("timestamp taken from") for n in result["vendor_notes"])

    def test_a_flagged_book_reaches_the_depth_fetch(self):
        from .test_databento_provider import _mbp10

        book = _mbp10(rows=20)
        book.loc[3, "flags"] = 4
        client = StubClient({DEPTH: WIDE}, default=book)
        provider = _provider(client)
        with patch.object(DataFactory, "get_provider", lambda *a, **k: provider):
            result = resolve_runtime("data").dispatch(
                "fetch_order_book",
                {
                    "symbol": "AAPL",
                    "start_date": "2026-03-02",
                    "end_date": "2026-03-02",
                    "run_id": "flags",
                    "name": "book",
                },
            )
        assert any("flag 4" in w for w in result["warnings"])
        assert any("sentinel" in n for n in result["vendor_notes"])

    def test_null_a_clean_tape_warns_of_nothing(self):
        client = StubClient({BASIC: WIDE}, rules=[_by_schema()])
        tape = _provider(client).get_trades("AAPL", TAPE_START, TAPE_END)
        assert not [n for n in tape.attrs["vendor_notes"] if n.startswith("WARNING")]


# ── 5. a futures daily bar is a CME trade date ───────────────────────────


def _globex_week(first_date, days=5, *, stray_saturday=False):
    """Hourly bars for `days` Globex sessions: each opens at 17:00 Chicago
    on the evening before its date and closes at 16:00, with the hour-long
    halt between. Close counts up by one per bar; volume is 1,000 a bar."""
    stamps = []
    for date in pd.bdate_range(first_date, periods=days):
        # Monday's session opens on Sunday evening.
        prior = date - pd.Timedelta(days=1)
        open_ = pd.Timestamp(f"{prior.date()} 17:00", tz="America/Chicago")
        stamps.extend(open_ + pd.Timedelta(hours=h) for h in range(23))
    if stray_saturday:
        last = pd.bdate_range(first_date, periods=days)[-1]
        stamps.append(pd.Timestamp(f"{last.date()} 17:30", tz="America/Chicago"))
    index = pd.DatetimeIndex(stamps).tz_convert("UTC")
    close = 5000.0 + np.arange(len(index))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(index), 1_000, dtype="uint64"),
        },
        index=index,
    )


class TestAFuturesDailyBarIsACmeTradeDate:
    def test_a_week_is_five_trade_dates_not_six_utc_days(self):
        hourly = _globex_week("2025-03-03")
        client = StubClient({DATASET_FUTURES: WIDE}, default=hourly)
        frame = _provider(client).get_ohlcv("ES.c.0", "2025-03-03", "2025-03-07")
        assert [c["schema"] for c in client.calls] == ["ohlcv-1h"]
        assert [str(d.date()) for d in frame.index] == [
            "2025-03-03",
            "2025-03-04",
            "2025-03-05",
            "2025-03-06",
            "2025-03-07",
        ]
        assert (frame["Volume"] == 23_000).all()
        # Monday's close is the 15:00 Chicago bar, the session's last.
        monday_last = pd.Timestamp("2025-03-03 15:00", tz="America/Chicago")
        assert frame["Close"].iloc[0] == hourly.loc[monday_last, "close"]
        assert "CME trade date" in frame.attrs["session"]
        utc_days = hourly.index.tz_convert("UTC").normalize().unique()
        assert len(utc_days) == 6, "the planted week does span six UTC days"

    def test_the_roll_holds_across_a_daylight_saving_change(self):
        # Chicago moved to daylight time on Sunday 2025-03-09.
        hourly = _globex_week("2025-03-10")
        client = StubClient({DATASET_FUTURES: WIDE}, default=hourly)
        frame = _provider(client).get_ohlcv("ES.c.0", "2025-03-10", "2025-03-14")
        assert len(frame) == 5 and (frame["Volume"] == 23_000).all()

    def test_a_print_after_fridays_close_belongs_to_monday(self):
        hourly = _globex_week("2025-03-03", stray_saturday=True)
        client = StubClient({DATASET_FUTURES: WIDE}, default=hourly)
        frame = _provider(client).get_ohlcv("ES.c.0", "2025-03-03", "2025-03-07")
        assert len(frame) == 5, "no Saturday bar"

    def test_a_weekend_holds_no_trade_date(self):
        hourly = _globex_week("2025-03-03")
        client = StubClient({DATASET_FUTURES: WIDE}, default=hourly)
        with pytest.raises(DataNotFoundError, match="no CME trade date"):
            _provider(client).get_ohlcv("ES.c.0", "2025-03-08", "2025-03-09")

    def test_the_metadata_says_what_a_daily_bar_is(self):
        notes = " ".join(_provider(StubClient({})).get_metadata("ES.c.0").notes)
        assert "CME TRADE DATE" in notes and "settlement" in notes

    def test_null_an_equity_day_is_still_the_summary_feeds_daily_bar(self):
        client = StubClient(ALL)
        frame = _provider(client).get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
        assert [c["schema"] for c in client.calls] == ["ohlcv-1d"]
        assert "session" not in frame.attrs


# ── 6. a window the better feed covers only part of ──────────────────────


class TestACoverageDowngradeIsDisclosed:
    WINDOW = ("2024-06-24", "2024-12-31")

    def test_the_sample_feed_answering_the_whole_window_is_said(self, caplog):
        with caplog.at_level(logging.WARNING, logger=PROVIDER_LOG):
            frame = _provider(StubClient(ALL)).get_ohlcv("AAPL", *self.WINDOW)
        assert frame.attrs["dataset"] == CONSOLIDATED
        downgrade = frame.attrs["coverage_downgrade"]
        assert downgrade["preferred"] == SUMMARY
        assert downgrade["covers_from"] == "2024-07-01"
        assert downgrade["served"] == CONSOLIDATED
        assert "split the request at 2024-07-01" in downgrade["advice"]
        assert any("2024-07-01" in r.getMessage() for r in caplog.records)
        # The data itself is unchanged: the window is not clamped.
        assert str(frame.index[0].date()) == "2024-06-24"

    def test_a_warm_read_says_it_too(self):
        _provider(StubClient(ALL)).get_ohlcv("AAPL", *self.WINDOW)
        warm = StubClient(ALL)
        again = _provider(warm).get_ohlcv("AAPL", *self.WINDOW)
        assert warm.calls == []
        assert again.attrs["coverage_downgrade"]["preferred"] == SUMMARY

    def test_the_fetch_tool_and_the_preflight_warn(self):
        provider = _provider(StubClient(ALL, billable=1_000))
        arguments = {
            "symbol": "AAPL",
            "start_date": self.WINDOW[0],
            "end_date": self.WINDOW[1],
        }
        with patch.object(DataFactory, "get_provider", lambda *a, **k: provider):
            data = resolve_runtime("data")
            fetched = data.dispatch(
                "fetch_ohlcv", {**arguments, "run_id": "downgrade", "name": "bars"}
            )
            preflight = data.dispatch(
                "preflight_vendor_request", {**arguments, "vendor_schema": "ohlcv-1d"}
            )
        advice = "Split the request at 2024-07-01: EQUS.SUMMARY answers"
        assert any(
            "COVERAGE DOWNGRADE" in w and advice in w for w in fetched["warnings"]
        )
        assert preflight["dataset"] == CONSOLIDATED
        assert any(
            "COVERAGE DOWNGRADE" in w and advice in w for w in preflight["warnings"]
        )

    @pytest.mark.parametrize(
        "window", [("2024-07-01", "2024-12-31"), ("2023-06-01", "2023-06-30")]
    )
    def test_null_a_window_one_feed_answers_whole_says_nothing(self, window):
        frame = _provider(StubClient(ALL)).get_ohlcv("AAPL", *window)
        assert "coverage_downgrade" not in frame.attrs


# ── 7. an explicit intraday end is inclusive ─────────────────────────────


def _minute_bars(first, last):
    index = pd.date_range(first, last, freq="min", tz="UTC")
    close = np.arange(len(index), dtype=float) + 100.0
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 10},
        index=index,
    )


class TestAnIntradayEndIncludesItsBar:
    def test_the_bar_at_the_end_is_served(self):
        client = StubClient(
            {BASIC: WIDE}, default=_minute_bars("2025-03-03 14:00", "2025-03-03 14:20")
        )
        frame = _provider(client).get_ohlcv(
            "AAPL", "2025-03-03T14:00:00", "2025-03-03T14:10:00", interval="1m"
        )
        assert len(frame) == 11
        assert frame.index[-1] == pd.Timestamp("2025-03-03 14:10")
        assert client.calls[0]["end"] == "2025-03-03T14:11:00"

    def test_null_a_bare_date_gains_no_next_day_bar(self):
        client = StubClient(
            {BASIC: WIDE}, default=_minute_bars("2025-03-03 23:55", "2025-03-04 00:05")
        )
        frame = _provider(client).get_ohlcv(
            "AAPL", "2025-03-03", "2025-03-03", interval="1m"
        )
        assert frame.index[-1] == pd.Timestamp("2025-03-03 23:59")
        assert client.calls[0]["end"] == "2025-03-04T00:00:00"

    def test_null_a_tape_stays_half_open(self):
        client = StubClient({BASIC: WIDE}, rules=[_by_schema()])
        _provider(client).get_trades("AAPL", TAPE_START, TAPE_END)
        assert client.calls[0]["end"] == "2025-03-03T14:31:00"


# ── 8. a symbol the vendor does not know ─────────────────────────────────


class _Store:
    def __init__(self, frame, not_found=(), via_metadata=False):
        self._frame = frame
        if via_metadata:
            self.metadata = type("M", (), {"not_found": list(not_found)})()
        else:
            self.symbology = {"not_found": list(not_found), "partial": []}

    def to_df(self):
        return self._frame


def _vendor(answer):
    client = StubClient(ALL)

    def get_range(**kw):
        client.calls.append(kw)
        return answer(kw)

    client.timeseries.get_range = get_range
    return client


class TestAnUnknownSymbolFailsOnTheFirstAnswer:
    @pytest.mark.parametrize("via_metadata", [False, True])
    def test_one_request_and_the_symbol_is_named(self, via_metadata):
        client = _vendor(
            lambda kw: _Store(
                pd.DataFrame(), not_found=kw["symbols"], via_metadata=via_metadata
            )
        )
        with pytest.raises(InvalidSymbolError, match="ZZZZZ") as caught:
            _provider(client).get_ohlcv("ZZZZZ", "2025-03-03", "2025-03-07")
        assert len(client.calls) == 1
        assert SUMMARY in str(caught.value)

    def test_null_an_empty_answer_that_resolved_is_the_old_refusal(self):
        """Now a `DataNotFoundError`, which is never retried (see the
        CHANGELOG entry of 2026-10-02)."""
        client = _vendor(lambda kw: _Store(pd.DataFrame(), not_found=[]))
        with pytest.raises(DataNotFoundError, match="Datasets tried") as caught:
            _provider(client).get_ohlcv("AAPL", "2025-03-03", "2025-03-07")
        assert not isinstance(caught.value, InvalidSymbolError)


# ── 9. futures roots that start with a digit ─────────────────────────────


class TestANumericFuturesRootResolves:
    @pytest.mark.parametrize(
        "symbol,raw",
        [
            ("6EZ5", "6EZ5"),
            ("6bh26", "6BH26"),
            ("6JM6", "6JM6"),
            ("M2KZ5", "M2KZ5"),
            ("SR3Z5", "SR3Z5"),
            ("ESZ6", "ESZ6"),
            ("MESU5", "MESU5"),
        ],
    )
    def test_the_contract_routes_to_the_futures_dataset(self, symbol, raw):
        route = DatabentoProvider.resolve_symbol(symbol)
        assert (route.raw, route.stype_in, route.family) == (
            raw,
            "raw_symbol",
            "future",
        )

    def test_a_currency_contract_is_fetched_from_globex(self):
        client = StubClient({DATASET_FUTURES: WIDE}, default=_globex_week("2025-03-03"))
        _provider(client).get_ohlcv("6EZ5", "2025-03-03", "2025-03-07")
        assert client.calls[0]["dataset"] == DATASET_FUTURES
        assert client.calls[0]["symbols"] == ["6EZ5"]

    @pytest.mark.parametrize("symbol,family", [("NVDA", "equity"), ("AAPL", "equity")])
    def test_null_an_equity_is_still_an_equity(self, symbol, family):
        assert DatabentoProvider.resolve_symbol(symbol).family == family

    def test_null_a_bare_root_is_still_ambiguous(self):
        with pytest.raises(ValidationError, match="ambiguous"):
            DatabentoProvider.resolve_symbol("ES")

    @pytest.mark.parametrize("symbol", ["9Z5", "A1"])
    def test_null_a_digit_alone_is_not_a_root(self, symbol):
        with pytest.raises(ValidationError, match="6EZ5"):
            DatabentoProvider.resolve_symbol(symbol)
