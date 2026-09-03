"""
The first depth provider, and the vendor quirks it has to survive.

EVERY TEST HERE IS OFFLINE. The client is injected, which is the point:
dataset preference, the daily finalization walk-back, entitlement memory
and the end-anchoring that makes a weekend request work are exactly the
parts that are expensive to get wrong and impossible to exercise against a
live API in a suite. A provider whose only test is "it fetched something
once, by hand" has no test.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.data.databento import (
    FIXED_PRICE_SCALE,
    UNDEF_ORDER_SIZE,
    UNDEF_PRICE,
)
from standard_quant_tools.data.databento_provider import DatabentoProvider, _to_utc
from standard_quant_tools.error import APIError, ValidationError

CONSOLIDATED = "EQUS.MINI"
BASIC = "XNAS.BASIC"
DEPTH = "XNAS.ITCH"


class _Store:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_df(self) -> pd.DataFrame:
        return self._frame


class _Metadata:
    def __init__(self, ranges) -> None:
        self._ranges = ranges

    def get_dataset_range(self, dataset: str):
        if dataset not in self._ranges:
            raise RuntimeError(f"403 not_entitled for {dataset}")
        first, last = self._ranges[dataset]
        return {"start": first, "end": last}


class _Timeseries:
    def __init__(self, owner) -> None:
        self._owner = owner

    def get_range(self, **kwargs):
        self._owner.calls.append(kwargs)
        for rule in self._owner.rules:
            outcome = rule(kwargs)
            if outcome is not None:
                if isinstance(outcome, Exception):
                    raise outcome
                return _Store(outcome)
        return _Store(self._owner.default)


class StubClient:
    """A Databento client that answers from rules, and records every call."""

    def __init__(self, ranges, rules=None, default=None) -> None:
        self.metadata = _Metadata(ranges)
        self.timeseries = _Timeseries(self)
        self.rules = list(rules or [])
        self.default = default if default is not None else pd.DataFrame()
        self.calls: list = []

    def datasets_called(self):
        return [c["dataset"] for c in self.calls]


def _bars(rows: int = 5, fixed_point: bool = True) -> pd.DataFrame:
    index = pd.date_range("2024-03-01", periods=rows, freq="D", tz="UTC")
    close = 100.0 + np.arange(rows)
    scale = FIXED_PRICE_SCALE if fixed_point else 1
    return pd.DataFrame(
        {
            "open": np.round(close * scale).astype("int64" if fixed_point else float),
            "high": np.round((close + 1) * scale).astype(
                "int64" if fixed_point else float
            ),
            "low": np.round((close - 1) * scale).astype(
                "int64" if fixed_point else float
            ),
            "close": np.round(close * scale).astype("int64" if fixed_point else float),
            "volume": np.full(rows, 1_000_000, dtype="int64"),
        },
        index=index,
    )


def _mbp10(rows: int = 40, live_levels: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    mid = 250.0 + np.cumsum(rng.normal(0, 0.01, rows))
    base = pd.Timestamp("2026-03-02 14:30", tz="UTC").value
    data = {
        "ts_recv": (base + np.arange(rows) * 1_000_000).astype("int64"),
        "ts_event": (base + np.arange(rows) * 1_000_000 - 200_000).astype("int64"),
        "flags": np.zeros(rows, dtype="int64"),
        "action": np.array(["A"] * rows),
        "side": np.array(["B"] * rows),
    }
    for level in range(10):
        offset = 0.01 * level
        bid = np.round((mid - 0.01 - offset) * FIXED_PRICE_SCALE).astype("int64")
        ask = np.round((mid + 0.01 + offset) * FIXED_PRICE_SCALE).astype("int64")
        bid_size = rng.integers(100, 3000, rows).astype("int64")
        ask_size = rng.integers(100, 3000, rows).astype("int64")
        if level >= live_levels:
            bid[:] = UNDEF_PRICE
            ask[:] = UNDEF_PRICE
            bid_size[:] = UNDEF_ORDER_SIZE
            ask_size[:] = UNDEF_ORDER_SIZE
        data[f"bid_px_{level:02d}"] = bid
        data[f"ask_px_{level:02d}"] = ask
        data[f"bid_sz_{level:02d}"] = bid_size
        data[f"ask_sz_{level:02d}"] = ask_size
        data[f"bid_ct_{level:02d}"] = rng.integers(1, 30, rows).astype("int64")
        data[f"ask_ct_{level:02d}"] = rng.integers(1, 30, rows).astype("int64")
    return pd.DataFrame(data)


WIDE = ("2018-01-01T00:00:00+00:00", "2026-09-02T00:00:00+00:00")
SINCE_2023 = ("2023-03-28T00:00:00+00:00", "2026-09-02T00:00:00+00:00")


def _provider(client, **kw) -> DatabentoProvider:
    return DatabentoProvider(api_key="not-used", client=client, **kw)


class TestSymbolMapping:
    @pytest.mark.parametrize(
        "given,expected",
        [("NVDA", "NVDA"), ("BRK.B", "BRKB"), ("BRK-B", "BRKB"), ("aapl", "AAPL")],
    )
    def test_share_classes_lose_the_separator(self, given, expected) -> None:
        assert DatabentoProvider.to_raw_symbol(given) == expected

    @pytest.mark.parametrize("given", ["", "TOOLONGSYM", "ES=F", "BTC/USD"])
    def test_anything_else_is_refused(self, given) -> None:
        with pytest.raises(ValidationError, match="raw_symbol"):
            DatabentoProvider.to_raw_symbol(given)


class TestTheInclusiveEndDate:
    """This library's end date is INCLUSIVE and Databento's range is
    half-open. A bare date that is not pushed forward returns nothing."""

    def test_a_bare_end_date_reaches_the_end_of_that_day(self) -> None:
        assert _to_utc("2026-03-02", end_of_day=True) == datetime(
            2026, 3, 3, tzinfo=timezone.utc
        )

    def test_a_bare_start_date_is_not_pushed(self) -> None:
        assert _to_utc("2026-03-02", end_of_day=False) == datetime(
            2026, 3, 2, tzinfo=timezone.utc
        )

    def test_a_timestamp_with_a_time_is_left_alone(self) -> None:
        assert _to_utc("2026-03-02T15:30:00", end_of_day=True) == datetime(
            2026, 3, 2, 15, 30, tzinfo=timezone.utc
        )

    def test_an_unparseable_date_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="ISO date"):
            _to_utc("last tuesday", end_of_day=False)


class TestDatasetPreference:
    def test_the_consolidated_feed_is_tried_first(self) -> None:
        client = StubClient(
            {CONSOLIDATED: SINCE_2023, BASIC: WIDE, DEPTH: WIDE}, default=_bars()
        )
        _provider(client).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        assert client.datasets_called()[0] == CONSOLIDATED

    def test_a_range_predating_it_skips_it_without_asking(self) -> None:
        """
        It does not exist before 2023-03-28, so asking is a guaranteed miss
        and a wasted round trip.
        """
        client = StubClient(
            {CONSOLIDATED: SINCE_2023, BASIC: WIDE, DEPTH: WIDE}, default=_bars()
        )
        _provider(client).get_ohlcv("NVDA", "2019-01-02", "2019-06-30")
        assert CONSOLIDATED not in client.datasets_called()
        assert client.datasets_called()[0] == BASIC

    def test_it_falls_through_to_the_next_dataset_on_an_empty_answer(self) -> None:
        empty_for_consolidated = [
            lambda kw: pd.DataFrame() if kw["dataset"] == CONSOLIDATED else None
        ]
        client = StubClient(
            {CONSOLIDATED: SINCE_2023, BASIC: WIDE, DEPTH: WIDE},
            rules=empty_for_consolidated,
            default=_bars(),
        )
        frame = _provider(client).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        assert len(frame) == 5
        assert client.datasets_called()[:2] == [CONSOLIDATED, BASIC]

    def test_exhausting_every_dataset_refuses_by_name(self) -> None:
        client = StubClient(
            {CONSOLIDATED: SINCE_2023, BASIC: WIDE, DEPTH: WIDE},
            rules=[lambda kw: pd.DataFrame()],
        )
        with pytest.raises(APIError, match="Datasets tried"):
            _provider(client).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")


class TestEntitlementDenialsAreRemembered:
    def test_a_denied_dataset_is_not_asked_twice(self) -> None:
        """
        A 403 is a fact about the subscription, not about the request.
        Re-asking turns one refusal into a per-call latency cost.
        """
        denied = [
            lambda kw: (
                RuntimeError("403 Forbidden: not_entitled")
                if kw["dataset"] == CONSOLIDATED
                else None
            )
        ]
        client = StubClient(
            {CONSOLIDATED: SINCE_2023, BASIC: WIDE, DEPTH: WIDE},
            rules=denied,
            default=_bars(),
        )
        provider = _provider(client)
        provider.get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        first_round = client.datasets_called()
        assert CONSOLIDATED in first_round

        client.calls.clear()
        provider.get_ohlcv("NVDA", "2024-02-01", "2024-02-10")
        assert CONSOLIDATED not in client.datasets_called()

    def test_a_dataset_with_no_range_is_skipped(self) -> None:
        client = StubClient({BASIC: WIDE, DEPTH: WIDE}, default=_bars())
        _provider(client).get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
        assert CONSOLIDATED not in client.datasets_called()


class TestTheDailyFinalizationLag:
    """
    `ohlcv-1d` finalizes a day or two behind the live feed while the
    dataset range reports the LIVE edge, so the honest end lands in the
    unfinalized tail and is refused.
    """

    def test_the_end_walks_back_until_it_resolves(self) -> None:
        state = {"refusals": 2}

        def unfinalized(kw):
            if kw["schema"] != "ohlcv-1d":
                return None
            if state["refusals"] > 0:
                state["refusals"] -= 1
                return RuntimeError("422 data_end_after_available_end")
            return None

        client = StubClient(
            {CONSOLIDATED: SINCE_2023}, rules=[unfinalized], default=_bars()
        )
        frame = _provider(client).get_ohlcv("NVDA", "2024-03-01", "2024-03-10")
        assert len(frame) == 5
        ends = [c["end"] for c in client.calls]
        assert len(ends) == 3, ends
        assert ends[0] > ends[1] > ends[2], ends

    def test_only_that_error_is_retried(self) -> None:
        """Anything else is a real failure, and re-asking would hide it."""

        def boom(kw):
            return RuntimeError("500 internal error")

        client = StubClient({CONSOLIDATED: SINCE_2023}, rules=[boom])
        with pytest.raises(APIError):
            _provider(client).get_ohlcv("NVDA", "2024-03-01", "2024-03-10")
        # One attempt per dataset, not six walk-backs.
        assert len(client.calls) == 1

    def test_an_intraday_schema_does_not_walk_back(self) -> None:
        client = StubClient({CONSOLIDATED: SINCE_2023}, default=_bars())
        _provider(client).get_ohlcv("NVDA", "2024-03-01", "2024-03-02", interval="1h")
        assert len(client.calls) == 1
        assert client.calls[0]["schema"] == "ohlcv-1h"


class TestTheRequestIsClampedToWhatWasPublished:
    def test_an_end_past_the_edge_is_pulled_back(self) -> None:
        """
        The weekend case. Anchoring to wall-clock `now` asks for data that
        was never published; anchoring to the dataset's edge returns the
        last session, which is what the caller meant.
        """
        edge = ("2023-03-28T00:00:00+00:00", "2026-03-06T00:00:00+00:00")
        client = StubClient({CONSOLIDATED: edge}, default=_bars())
        _provider(client).get_ohlcv("NVDA", "2026-03-02", "2026-03-08", interval="1h")
        requested_end = client.calls[0]["end"]
        assert requested_end <= "2026-03-06T00:00:00"

    def test_an_empty_window_is_refused_before_any_call(self) -> None:
        client = StubClient({CONSOLIDATED: SINCE_2023}, default=_bars())
        with pytest.raises(ValidationError, match="empty window"):
            _provider(client).get_ohlcv("NVDA", "2024-03-10", "2024-03-01")
        assert client.calls == []


class TestBars:
    def test_fixed_point_prices_are_scaled(self) -> None:
        client = StubClient({CONSOLIDATED: SINCE_2023}, default=_bars(fixed_point=True))
        frame = _provider(client).get_ohlcv("NVDA", "2024-03-01", "2024-03-10")
        assert 99.0 < frame["Close"].iloc[0] < 101.0

    def test_float_dollars_are_not_scaled_again(self) -> None:
        client = StubClient(
            {CONSOLIDATED: SINCE_2023}, default=_bars(fixed_point=False)
        )
        frame = _provider(client).get_ohlcv("NVDA", "2024-03-01", "2024-03-10")
        assert 99.0 < frame["Close"].iloc[0] < 101.0

    def test_the_column_contract_is_this_librarys(self) -> None:
        client = StubClient({CONSOLIDATED: SINCE_2023}, default=_bars())
        frame = _provider(client).get_ohlcv("NVDA", "2024-03-01", "2024-03-10")
        assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_an_unsupported_interval_names_what_exists(self) -> None:
        client = StubClient({CONSOLIDATED: SINCE_2023}, default=_bars())
        with pytest.raises(ValidationError, match=r"publishes \['1d', '1h'"):
            _provider(client).get_ohlcv("NVDA", "2024-03-01", "2024-03-10", "1wk")


class TestTheFirstOrderBook:
    """
    `DataProvider.get_order_book` has been a declared contract with no
    implementation since it was written. Every depth measure in
    `analysis/order_book.py` was built against synthetic books.
    """

    def test_it_returns_this_librarys_column_contract(self) -> None:
        client = StubClient({DEPTH: WIDE}, default=_mbp10())
        book = _provider(client).get_order_book("NVDA", "2026-03-02", "2026-03-02")
        for column in (
            "timestamp",
            "bid_price_0",
            "bid_size_0",
            "ask_price_0",
            "ask_size_0",
        ):
            assert column in book.columns
        assert "bid_px_00" not in book.columns

    def test_book_metrics_reads_it_without_translation(self) -> None:
        """The whole point: the analytics do not learn about Databento."""
        from standard_quant_tools.analysis.order_book import book_metrics

        client = StubClient({DEPTH: WIDE}, default=_mbp10())
        book = _provider(client).get_order_book("NVDA", "2026-03-02", "2026-03-02")
        metrics = book_metrics(book)
        assert 240 < metrics["mean_microprice"] < 260
        assert metrics["depth_slope"] is not None

    def test_prices_are_dollars_and_sentinels_are_gone(self) -> None:
        client = StubClient({DEPTH: WIDE}, default=_mbp10())
        book = _provider(client).get_order_book("NVDA", "2026-03-02", "2026-03-02")
        prices = [c for c in book.columns if "price" in c]
        assert (
            book[prices].max().max() < 1000
        ), "an unmasked int64-max sentinel would appear here as a $9.2bn quote"

    def test_it_reads_only_the_depth_dataset(self) -> None:
        """
        Top of book is not depth. Serving the consolidated feed here would
        return a one-level book whose imbalance is zero by construction --
        a balanced market rather than missing data, which is the exact
        substitution the base class refuses to make.
        """
        client = StubClient(
            {CONSOLIDATED: WIDE, BASIC: WIDE, DEPTH: WIDE}, default=_mbp10()
        )
        _provider(client).get_order_book("NVDA", "2026-03-02", "2026-03-02")
        assert set(client.datasets_called()) == {DEPTH}

    def test_levels_caps_the_depth_read(self) -> None:
        client = StubClient({DEPTH: WIDE}, default=_mbp10(live_levels=10))
        book = _provider(client).get_order_book(
            "NVDA", "2026-03-02", "2026-03-02", levels=3
        )
        assert "bid_price_2" in book.columns
        assert "bid_price_3" not in book.columns

    @pytest.mark.parametrize("levels", [0, 11, -1])
    def test_an_impossible_depth_is_refused(self, levels) -> None:
        client = StubClient({DEPTH: WIDE}, default=_mbp10())
        with pytest.raises(ValidationError, match="mbp-10 carries ten"):
            _provider(client).get_order_book(
                "NVDA", "2026-03-02", "2026-03-02", levels=levels
            )

    def test_limit_bounds_what_comes_back(self) -> None:
        client = StubClient({DEPTH: WIDE}, default=_mbp10(rows=100))
        book = _provider(client).get_order_book(
            "NVDA", "2026-03-02", "2026-03-02", limit=10
        )
        assert len(book) == 10


class TestHonestSelfReport:
    def test_bars_are_reported_unadjusted(self) -> None:
        """
        The field that will surprise someone. Databento serves what the
        venue published, so a split is a real -50% bar -- and every other
        provider here reports adjusted=True.
        """
        meta = _provider(StubClient({})).get_metadata("NVDA")
        assert meta.adjusted is False
        assert meta.provider == "databento"
        assert meta.timezone == "UTC"

    def test_it_does_not_claim_point_in_time(self) -> None:
        assert _provider(StubClient({})).get_metadata("NVDA").point_in_time is False

    def test_fundamentals_are_refused_rather_than_faked(self) -> None:
        """Empty ratios would be indistinguishable from a company that
        genuinely has none."""
        with pytest.raises(ValidationError, match="publishes no"):
            _provider(StubClient({})).get_financial_ratios("NVDA")

    def test_ticker_info_returns_what_it_knows(self) -> None:
        info = _provider(StubClient({})).get_ticker_info("nvda")
        assert info.symbol == "NVDA"


class TestCredentials:
    def test_a_missing_key_refuses_with_the_reason(self, monkeypatch) -> None:
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        provider = DatabentoProvider()
        with pytest.raises(APIError, match="DATABENTO_API_KEY"):
            provider.get_ohlcv("NVDA", "2024-01-02", "2024-01-10")

    def test_the_refusal_says_why_it_is_not_a_spec_field(self, monkeypatch) -> None:
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        with pytest.raises(APIError, match="lineage"):
            DatabentoProvider().get_ohlcv("NVDA", "2024-01-02", "2024-01-10")
