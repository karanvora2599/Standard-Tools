"""
The optional tick-data capability on the provider contract.

`get_trades` / `get_quotes` are the first methods on `DataProvider` that not
every provider can serve. They are concrete-with-raise rather than abstract
on purpose: marking them abstract would break yfinance and Bloomberg at
IMPORT time to express something better said at the point of use.

These tests are all offline. The Polygon parsing is exercised against
synthetic payloads through the same seam the rest of that module's tests
use, so nothing here needs a key or a paid plan -- which matters, because
the endpoints themselves are not on the free tier and CI cannot reach them.
"""

import pandas as pd
import pytest

from standard_quant_tools.data.base import DataProvider
from standard_quant_tools.data.polygon_provider import (
    PolygonProvider,
    _parse_ticks,
    _tick_range_ns,
)
from standard_quant_tools.error import DataNotFoundError, ValidationError

TRADE_COLUMNS = {"price": "price", "size": "size", "exchange": "exchange"}


class TestTheContract:
    def test_the_methods_are_concrete_not_abstract(self):
        # If these became abstract, every provider that cannot serve ticks
        # would fail to instantiate -- which is a worse answer than a clear
        # error when someone actually asks for a tick.
        assert "get_trades" not in DataProvider.__abstractmethods__
        assert "get_quotes" not in DataProvider.__abstractmethods__

    @pytest.mark.parametrize("method", ["get_trades", "get_quotes"])
    def test_a_provider_without_ticks_says_which_provider(self, method):
        from standard_quant_tools.data.factory import DataFactory

        provider = DataFactory.get_provider("yfinance")
        with pytest.raises(NotImplementedError) as excinfo:
            getattr(provider, method)("AAPL", "2024-01-01", "2024-01-02")
        message = str(excinfo.value)
        assert "YFinanceProvider" in message, "the error must name the provider"
        assert "Polygon" in message, "the error must name what does work"

    def test_the_error_refuses_to_offer_bars_as_a_substitute(self):
        # The tempting wrong answer is to synthesize ticks from OHLCV. Every
        # microstructure measure downstream would treat the fiction as fact.
        from standard_quant_tools.data.factory import DataFactory

        provider = DataFactory.get_provider("yfinance")
        with pytest.raises(NotImplementedError, match="cannot substitute"):
            provider.get_trades("AAPL", "2024-01-01", "2024-01-02")

    def test_polygon_declares_both(self):
        assert PolygonProvider.get_trades is not DataProvider.get_trades
        assert PolygonProvider.get_quotes is not DataProvider.get_quotes


class TestRangeHandling:
    def test_the_range_is_nanoseconds(self):
        start, end = _tick_range_ns("2024-01-02", "2024-01-03")
        assert end - start == 86_400 * 1_000_000_000, "one day in nanoseconds"

    def test_the_range_is_half_open(self):
        # A closed range either double-counts the boundary tick when two
        # windows are concatenated or drops it, and which one is invisible
        # until someone concatenates.
        a_start, a_end = _tick_range_ns("2024-01-02", "2024-01-03")
        b_start, _b_end = _tick_range_ns("2024-01-03", "2024-01-04")
        assert a_end == b_start, "adjacent windows must abut without overlapping"

    @pytest.mark.parametrize(
        "start,end",
        [("2024-01-03", "2024-01-02"), ("2024-01-02", "2024-01-02")],
    )
    def test_a_non_advancing_range_is_refused(self, start, end):
        with pytest.raises(ValidationError, match="must be after"):
            _tick_range_ns(start, end)


class TestTickParsing:
    def test_nanosecond_resolution_survives(self):
        # The aggregates endpoint in the same module uses MILLISECONDS.
        # Parsing these with that unit would date every tick to 1970, and
        # the frame would still look structurally fine.
        rows = [
            {"sip_timestamp": 1_700_000_000_123_456_789, "price": 101.5, "size": 100},
            {"sip_timestamp": 1_700_000_000_123_456_999, "price": 101.6, "size": 50},
        ]
        frame = _parse_ticks(rows, TRADE_COLUMNS, "AAPL", "trades")
        assert frame.index.dtype == "datetime64[ns]"
        assert frame.index[0].year == 2023, "millisecond unit would give 1970"
        # 210 nanoseconds apart -- a resolution a millisecond unit destroys.
        assert (frame.index[1] - frame.index[0]).value == 210

    def test_rows_without_a_timestamp_are_dropped_not_defaulted(self):
        # A tick with no time cannot be ordered against the others, and
        # every measure built on this data is an ordering.
        rows = [
            {"sip_timestamp": 1_700_000_000_000_000_000, "price": 10.0, "size": 1},
            {"price": 11.0, "size": 2},
        ]
        frame = _parse_ticks(rows, TRADE_COLUMNS, "AAPL", "trades")
        assert len(frame) == 1

    def test_the_participant_timestamp_is_a_fallback(self):
        rows = [{"participant_timestamp": 1_700_000_000_000_000_000, "price": 10.0}]
        frame = _parse_ticks(rows, {"price": "price"}, "AAPL", "trades")
        assert len(frame) == 1

    def test_output_is_sorted_by_time(self):
        rows = [
            {"sip_timestamp": 2_000_000_000_000_000_000, "price": 2.0, "size": 1},
            {"sip_timestamp": 1_000_000_000_000_000_000, "price": 1.0, "size": 1},
        ]
        frame = _parse_ticks(rows, TRADE_COLUMNS, "AAPL", "trades")
        assert frame.index.is_monotonic_increasing
        assert frame["price"].iloc[0] == 1.0

    def test_an_empty_payload_is_not_an_empty_frame(self):
        # Returning an empty frame would let a caller compute a spread over
        # nothing and report it as a measurement.
        with pytest.raises(DataNotFoundError, match="no trades"):
            _parse_ticks([], TRADE_COLUMNS, "AAPL", "trades")

    def test_a_payload_with_no_usable_timestamps_says_so_specifically(self):
        with pytest.raises(DataNotFoundError, match="none carrying a usable"):
            _parse_ticks([{"price": 1.0}], TRADE_COLUMNS, "AAPL", "trades")

    def test_quote_columns_map_through(self):
        rows = [
            {
                "sip_timestamp": 1_700_000_000_000_000_000,
                "bid_price": 10.0,
                "bid_size": 3,
                "ask_price": 10.02,
                "ask_size": 5,
            }
        ]
        frame = _parse_ticks(
            rows,
            {
                "bid_price": "bid_price",
                "bid_size": "bid_size",
                "ask_price": "ask_price",
                "ask_size": "ask_size",
            },
            "AAPL",
            "quotes",
        )
        assert list(frame.columns) == ["bid_price", "bid_size", "ask_price", "ask_size"]
        assert frame["ask_price"].iloc[0] > frame["bid_price"].iloc[0]


class TestPlanTierError:
    def test_a_403_is_re_raised_as_a_plan_problem(self, monkeypatch):
        # _polygon_get turns 401/403 into "check your key", which is right
        # for an expired key and misleading for a valid one on the wrong
        # plan. These endpoints are the only ones in the module where the
        # second case is the likely one.
        import standard_quant_tools.data.polygon_provider as module
        from standard_quant_tools.error import NonRetryableAPIError

        def _boom(*_args, **_kwargs):
            raise NonRetryableAPIError("Polygon.io rejected the request (HTTP 403)")

        monkeypatch.setenv("SQT_POLYGON_API_KEY", "test-key")
        monkeypatch.setattr(module, "_polygon_get", _boom)
        provider = module.PolygonProvider()

        with pytest.raises(NonRetryableAPIError, match="not on the free"):
            provider.get_trades("AAPL", "2024-01-02", "2024-01-03")
        with pytest.raises(NonRetryableAPIError, match="not on the free"):
            provider.get_quotes("AAPL", "2024-01-02", "2024-01-03")

    def test_a_successful_call_parses_without_network(self, monkeypatch):
        import standard_quant_tools.data.polygon_provider as module

        payload = {
            "results": [
                {
                    "sip_timestamp": 1_700_000_000_000_000_000,
                    "price": 42.0,
                    "size": 7,
                    "exchange": 4,
                }
            ]
        }
        monkeypatch.setenv("SQT_POLYGON_API_KEY", "test-key")
        monkeypatch.setattr(module, "_polygon_get", lambda *a, **k: payload)
        provider = module.PolygonProvider()

        frame = provider.get_trades("AAPL", "2024-01-02", "2024-01-03")
        assert isinstance(frame, pd.DataFrame)
        assert frame["price"].iloc[0] == 42.0
        assert frame.index.name == "timestamp"
