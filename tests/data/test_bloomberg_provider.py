"""
Tests for data/bloomberg_provider.py.

Scope, stated explicitly: blpapi is not installed in this environment (it's
Bloomberg's own SDK, distributed outside PyPI's normal mirror network, and
Desktop API additionally requires a real, running, logged-in Bloomberg
Terminal to do anything at all). This file thoroughly tests:
  - every pure helper function (ticker normalization, timezone heuristic,
    date formatting, historical-bar/reference-field parsing) — these have
    zero dependency on blpapi and are the actual business logic;
  - config resolution (host/port from explicit args, env vars, or defaults);
  - the graceful, real (not mocked) failure path when blpapi isn't
    installed, for both BloombergProvider() directly and via DataFactory.

It deliberately does NOT mock blpapi's Session/Event/Message/Element wire
objects to test _open_session/_send_request/_drain_historical_response/
_drain_reference_response — a hand-rolled mock of a wire protocol this
suite has never run against a real Terminal would provide false confidence,
not real coverage. Verify that layer with a real Terminal
(`@pytest.mark.integration`-style manual test) before depending on it.
"""

import datetime

import pandas as pd
import pytest

from standard_quant_tools.data.base import FinancialRatios, TickerInfo
from standard_quant_tools.data.bloomberg_provider import (
    HAS_BLPAPI,
    BloombergProvider,
    _bloomberg_timezone,
    _parse_financial_ratios,
    _parse_historical_bars,
    _parse_ticker_info,
    _resolve_bloomberg_config,
    _to_bbg_date,
    _to_bloomberg_ticker,
)
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    InvalidSymbolError,
    ValidationError,
)


class TestToBloombergTicker:
    def test_bare_symbol_gets_us_equity_suffix(self):
        assert _to_bloomberg_ticker("AAPL") == "AAPL US Equity"

    def test_already_qualified_ticker_passed_through(self):
        assert _to_bloomberg_ticker("VOD LN Equity") == "VOD LN Equity"
        assert _to_bloomberg_ticker("EURUSD Curncy") == "EURUSD Curncy"
        assert _to_bloomberg_ticker("SPX Index") == "SPX Index"

    def test_whitespace_is_stripped(self):
        assert _to_bloomberg_ticker("  AAPL  ") == "AAPL US Equity"

    def test_empty_symbol_raises(self):
        with pytest.raises(InvalidSymbolError):
            _to_bloomberg_ticker("")
        with pytest.raises(InvalidSymbolError):
            _to_bloomberg_ticker("   ")

    def test_unrecognized_trailing_token_still_gets_suffix(self):
        # "GOOG Foo" doesn't end in a recognized market-sector keyword, so
        # it's treated as a bare symbol needing " US Equity", not silently
        # assumed to already be Bloomberg-qualified.
        assert _to_bloomberg_ticker("GOOG Foo") == "GOOG Foo US Equity"


class TestBloombergTimezone:
    def test_us_default(self):
        assert _bloomberg_timezone("AAPL US Equity") == "America/New_York"

    def test_non_us_yellow_keys(self):
        assert _bloomberg_timezone("VOD LN Equity") == "Europe/London"
        assert _bloomberg_timezone("SAP GR Equity") == "Europe/Berlin"
        assert _bloomberg_timezone("7203 JP Equity") == "Asia/Tokyo"
        assert _bloomberg_timezone("0700 HK Equity") == "Asia/Hong_Kong"

    def test_unrecognized_yellow_key_defaults_to_new_york(self):
        assert _bloomberg_timezone("XYZ ZZ Equity") == "America/New_York"

    def test_ticker_with_no_yellow_key_defaults_to_new_york(self):
        assert _bloomberg_timezone("SPX Index") == "America/New_York"


class TestToBbgDate:
    def test_string_date(self):
        assert _to_bbg_date("2023-01-05") == "20230105"

    def test_date_object(self):
        assert _to_bbg_date(datetime.date(2023, 1, 5)) == "20230105"

    def test_datetime_object(self):
        assert _to_bbg_date(datetime.datetime(2023, 1, 5, 14, 30)) == "20230105"

    def test_invalid_format_raises(self):
        with pytest.raises(ValidationError):
            _to_bbg_date("01/05/2023")
        with pytest.raises(ValidationError):
            _to_bbg_date("not-a-date")


class TestResolveBloombergConfig:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SQT_BLOOMBERG_HOST", raising=False)
        monkeypatch.delenv("SQT_BLOOMBERG_PORT", raising=False)
        assert _resolve_bloomberg_config() == ("localhost", 8194)

    def test_explicit_args_win_over_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SQT_BLOOMBERG_HOST", "envhost")
        monkeypatch.setenv("SQT_BLOOMBERG_PORT", "9999")
        assert _resolve_bloomberg_config(host="explicit", port=1234) == (
            "explicit",
            1234,
        )

    def test_env_vars_used_when_no_explicit_args(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SQT_BLOOMBERG_HOST", "bbg.internal")
        monkeypatch.setenv("SQT_BLOOMBERG_PORT", "8195")
        assert _resolve_bloomberg_config() == ("bbg.internal", 8195)

    def test_invalid_port_env_var_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SQT_BLOOMBERG_PORT", "not-a-port")
        with pytest.raises(ValidationError):
            _resolve_bloomberg_config()


class TestParseHistoricalBars:
    def test_builds_expected_dataframe(self):
        bars = [
            {
                "date": datetime.date(2023, 1, 3),
                "PX_OPEN": 100.0,
                "PX_HIGH": 105.0,
                "PX_LOW": 99.0,
                "PX_LAST": 103.0,
                "PX_VOLUME": 1_000_000.0,
            },
            {
                "date": datetime.date(2023, 1, 4),
                "PX_OPEN": 103.0,
                "PX_HIGH": 108.0,
                "PX_LOW": 102.0,
                "PX_LAST": 107.0,
                "PX_VOLUME": 1_200_000.0,
            },
        ]
        df = _parse_historical_bars(bars, "AAPL")
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 2
        assert df["Close"].iloc[0] == 103.0
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_missing_volume_defaults_to_zero(self):
        bars = [
            {
                "date": datetime.date(2023, 1, 3),
                "PX_OPEN": 100.0,
                "PX_HIGH": 105.0,
                "PX_LOW": 99.0,
                "PX_LAST": 103.0,
            }
        ]
        df = _parse_historical_bars(bars, "AAPL")
        assert df["Volume"].iloc[0] == 0.0

    def test_empty_bars_raises_data_not_found(self):
        with pytest.raises(DataNotFoundError):
            _parse_historical_bars([], "AAPL")

    def test_missing_required_field_raises_api_error(self):
        bars = [
            {"date": datetime.date(2023, 1, 3), "PX_OPEN": 100.0}
        ]  # no PX_LAST etc.
        with pytest.raises(APIError):
            _parse_historical_bars(bars, "AAPL")


class TestParseTickerInfo:
    def test_maps_available_fields(self):
        info = _parse_ticker_info(
            "AAPL",
            {
                "LONG_COMP_NAME": "Apple Inc",
                "GICS_SECTOR_NAME": "Information Technology",
                "GICS_INDUSTRY_NAME": "Technology Hardware",
                "CNTRY_OF_DOMICILE": "US",
            },
        )
        assert isinstance(info, TickerInfo)
        assert info.name == "Apple Inc"
        assert info.sector == "Information Technology"
        assert info.country == "US"
        assert info.website is None  # not exposed by the fields this provider requests

    def test_missing_fields_default_to_unknown(self):
        info = _parse_ticker_info("AAPL", {})
        assert info.name == "Unknown"
        assert info.sector == "Unknown"
        assert info.industry == "Unknown"

    def test_falls_back_to_name_field_when_long_comp_name_absent(self):
        info = _parse_ticker_info("AAPL", {"NAME": "Apple"})
        assert info.name == "Apple"


class TestParseFinancialRatios:
    def test_maps_available_fields(self):
        ratios = _parse_financial_ratios(
            {
                "PE_RATIO": 28.5,
                "BEST_PE_RATIO": 30.1,
                "CUR_MKT_CAP": 2_800_000_000_000,
            }
        )
        assert isinstance(ratios, FinancialRatios)
        assert ratios.trailing_pe == 28.5
        assert ratios.forward_pe == 30.1
        assert ratios.market_cap == 2_800_000_000_000

    def test_missing_fields_are_none(self):
        ratios = _parse_financial_ratios({})
        assert ratios.trailing_pe is None
        assert ratios.market_cap is None


class TestGracefulFailureWithoutBlpapi:
    """blpapi is genuinely not installed in this environment, so these
    exercise the real (not mocked) failure path."""

    def test_has_blpapi_is_false_here(self):
        assert HAS_BLPAPI is False

    def test_direct_construction_raises_actionable_api_error(self):
        with pytest.raises(APIError, match="blpapi is not installed"):
            BloombergProvider()

    def test_via_data_factory_raises_same_error(self):
        with pytest.raises(APIError, match="blpapi is not installed"):
            DataFactory.get_provider("bloomberg")

    def test_abstract_methods_are_all_implemented(self):
        # Confirms BloombergProvider satisfies the full DataProvider ABC
        # contract independent of whether it can actually be constructed.
        assert BloombergProvider.__abstractmethods__ == frozenset()
