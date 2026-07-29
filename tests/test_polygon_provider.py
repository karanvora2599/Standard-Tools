"""
Tests for data/polygon_provider.py.

Unlike Bloomberg (a vendor SDK that isn't installed in this environment),
Polygon is a plain REST API — the only thing standing between these tests
and a real network call is an API key. Scope, stated explicitly:
  - every pure helper function (_parse_aggs, _parse_ticker_info,
    _parse_financial_ratios, _resolve_polygon_api_key, _norm_date) is
    tested directly with no network involved;
  - get_ohlcv/get_ticker_info/get_financial_ratios are tested by
    monkeypatching `_polygon_get` (the single network seam), not by mocking
    `urllib.request.urlopen`'s wire format — that would just be re-testing
    `_polygon_get` itself, which has its own dedicated tests below;
  - a small number of `_polygon_get`-level tests mock `urlopen` directly to
    cover HTTP status handling (401/403/404/429/other, invalid JSON, a
    body-level "status": "ERROR" payload) without a real API key.
No real network call is made anywhere in this file.
"""

import json
import urllib.error
from email.message import Message
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from standard_quant_tools.data.base import FinancialRatios, TickerInfo
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.data.polygon_provider import (
    PolygonProvider,
    _norm_date,
    _parse_aggs,
    _parse_financial_ratios,
    _parse_ticker_info,
    _polygon_get,
    _resolve_polygon_api_key,
)
from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    InvalidSymbolError,
    ValidationError,
)


class TestResolvePolygonApiKey:
    def test_explicit_arg_wins(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SQT_POLYGON_API_KEY", "env-key")
        assert _resolve_polygon_api_key("explicit-key") == "explicit-key"

    def test_falls_back_to_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SQT_POLYGON_API_KEY", "env-key")
        assert _resolve_polygon_api_key() == "env-key"

    def test_no_key_anywhere_raises_api_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SQT_POLYGON_API_KEY", raising=False)
        with pytest.raises(APIError, match="No Polygon.io API key found"):
            _resolve_polygon_api_key()


class TestNormDate:
    def test_valid_date_string(self):
        assert _norm_date("2023-01-05") == "2023-01-05"

    def test_truncates_datetime_like_string(self):
        assert _norm_date("2023-01-05T00:00:00") == "2023-01-05"

    def test_invalid_format_raises(self):
        with pytest.raises(ValidationError):
            _norm_date("01/05/2023")
        with pytest.raises(ValidationError):
            _norm_date("not-a-date")


class TestParseAggs:
    def test_builds_expected_dataframe_for_daily_bars(self):
        results = [
            {
                "o": 100.0,
                "h": 105.0,
                "l": 99.0,
                "c": 103.0,
                "v": 1_000_000,
                "t": 1672704000000,
            },
            {
                "o": 103.0,
                "h": 108.0,
                "l": 102.0,
                "c": 107.0,
                "v": 1_200_000,
                "t": 1672790400000,
            },
        ]
        df = _parse_aggs(results, "AAPL", "day")
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 2
        assert df["Close"].iloc[0] == 103.0
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is None

    def test_missing_volume_defaults_to_zero(self):
        results = [{"o": 100.0, "h": 105.0, "l": 99.0, "c": 103.0, "t": 1672704000000}]
        df = _parse_aggs(results, "AAPL", "day")
        assert df["Volume"].iloc[0] == 0.0

    def test_missing_required_field_raises_api_error(self):
        results = [{"o": 100.0, "t": 1672704000000}]  # no h/l/c
        with pytest.raises(APIError):
            _parse_aggs(results, "AAPL", "day")

    def test_intraday_bars_keep_time_component(self):
        results = [
            {
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "c": 100.5,
                "v": 500,
                "t": 1672747800000,
            }
        ]
        df = _parse_aggs(results, "AAPL", "minute")
        idx = pd.DatetimeIndex(df.index)
        assert idx.tz is None
        # Intraday bars are not normalized to midnight the way daily bars are.
        assert not (idx == idx.normalize()).all()


class TestParseTickerInfo:
    def test_maps_available_fields(self):
        info = _parse_ticker_info(
            "AAPL",
            {
                "name": "Apple Inc.",
                "sic_description": "ELECTRONIC COMPUTERS",
                "total_employees": 161_000,
                "homepage_url": "https://www.apple.com",
                "address": {"city": "Cupertino", "country": "US"},
                "locale": "us",
            },
        )
        assert isinstance(info, TickerInfo)
        assert info.name == "Apple Inc."
        assert info.sector == "ELECTRONIC COMPUTERS"
        assert info.industry == "ELECTRONIC COMPUTERS"
        assert info.city == "Cupertino"
        assert info.country == "US"
        assert info.website == "https://www.apple.com"

    def test_missing_fields_default_to_unknown(self):
        info = _parse_ticker_info("AAPL", {})
        assert info.name == "Unknown"
        assert info.sector == "Unknown"
        assert info.industry == "Unknown"
        assert info.website is None

    def test_locale_us_fills_country_when_address_absent(self):
        info = _parse_ticker_info("AAPL", {"locale": "us"})
        assert info.country == "US"


class TestParseFinancialRatios:
    def test_derives_ratios_from_financials_and_market_cap(self):
        details = {"market_cap": 1_000_000_000.0}
        financials = {
            "income_statement": {
                "net_income_loss": {"value": 100_000_000.0},
                "revenues": {"value": 500_000_000.0},
            },
            "balance_sheet": {
                "equity": {"value": 400_000_000.0},
                "liabilities": {"value": 200_000_000.0},
            },
        }
        ratios = _parse_financial_ratios(details, financials)
        assert isinstance(ratios, FinancialRatios)
        assert ratios.market_cap == 1_000_000_000
        assert ratios.trailing_pe == pytest.approx(10.0)
        assert ratios.price_to_book == pytest.approx(2.5)
        assert ratios.debt_to_equity == pytest.approx(0.5)
        assert ratios.return_on_equity == pytest.approx(0.25)
        assert ratios.profit_margins == pytest.approx(0.2)
        assert ratios.forward_pe is None
        assert ratios.dividend_yield is None

    def test_missing_financials_yields_all_none_except_market_cap(self):
        ratios = _parse_financial_ratios({"market_cap": 500.0}, {})
        assert ratios.market_cap == 500
        assert ratios.trailing_pe is None
        assert ratios.price_to_book is None
        assert ratios.debt_to_equity is None
        assert ratios.return_on_equity is None
        assert ratios.profit_margins is None

    def test_no_market_cap_yields_all_none(self):
        ratios = _parse_financial_ratios({}, {})
        assert ratios.market_cap is None
        assert ratios.trailing_pe is None


class TestPolygonGet:
    """Covers _polygon_get's HTTP/JSON handling by mocking urlopen directly
    — the one seam where mocking the wire format is actually the point."""

    def _mock_response(self, body: dict):
        cm = MagicMock()
        cm.__enter__.return_value.read.return_value = json.dumps(body).encode()
        return cm

    def test_successful_response_returns_parsed_json(self):
        with patch(
            "urllib.request.urlopen",
            return_value=self._mock_response({"status": "OK", "results": []}),
        ):
            payload = _polygon_get(
                "/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-05", {}, "key"
            )
        assert payload["status"] == "OK"

    def test_error_status_in_body_raises_api_error(self):
        with patch(
            "urllib.request.urlopen",
            return_value=self._mock_response(
                {"status": "ERROR", "error": "bad request"}
            ),
        ):
            with pytest.raises(APIError, match="bad request"):
                _polygon_get("/v3/reference/tickers/AAPL", {}, "key")

    def test_invalid_json_raises_api_error(self):
        cm = MagicMock()
        cm.__enter__.return_value.read.return_value = b"not json"
        with patch("urllib.request.urlopen", return_value=cm):
            with pytest.raises(APIError, match="invalid JSON"):
                _polygon_get("/v3/reference/tickers/AAPL", {}, "key")

    def _http_error(
        self, code: int, reason: str, body: bytes
    ) -> urllib.error.HTTPError:
        err = urllib.error.HTTPError("url", code, reason, Message(), None)
        err.read = lambda *args, **kwargs: body  # type: ignore[assignment]
        return err

    def test_401_raises_api_error_about_key(self):
        err = self._http_error(401, "Unauthorized", b"unauthorized")
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(APIError, match="SQT_POLYGON_API_KEY"):
                _polygon_get("/v3/reference/tickers/AAPL", {}, "key")

    def test_404_raises_data_not_found(self):
        err = self._http_error(404, "Not Found", b"not found")
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(DataNotFoundError):
                _polygon_get("/v3/reference/tickers/NOTREAL", {}, "key")

    def test_429_raises_api_error_about_rate_limit(self):
        err = self._http_error(429, "Too Many Requests", b"rate limited")
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(APIError, match="rate limit"):
                _polygon_get("/v3/reference/tickers/AAPL", {}, "key")

    def test_other_http_error_raises_generic_api_error(self):
        err = self._http_error(500, "Server Error", b"server error")
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(APIError, match="HTTP 500"):
                _polygon_get("/v3/reference/tickers/AAPL", {}, "key")

    def test_url_error_raises_api_error(self):
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("no network")
        ):
            with pytest.raises(APIError, match="request failed"):
                _polygon_get("/v3/reference/tickers/AAPL", {}, "key")


class TestTickerUrlPathEncoding:
    """A symbol is LLM/user-reachable and gets interpolated straight into a
    URL path segment (not just a query-string param) for both the aggs
    endpoint and the ticker-details endpoint. Guards against it being used
    to inject extra path segments / query params — e.g. a symbol like
    'AAPL?extra=1&apiKey=x' must not be able to smuggle its own apiKey or
    alter the request path _polygon_get ends up hitting."""

    def _provider(self) -> PolygonProvider:
        return PolygonProvider(api_key="test-key")

    def test_ohlcv_symbol_with_injection_characters_is_escaped_in_path(self):
        malicious = "AAPL?extra=1&apiKey=stolen"
        with patch(
            "standard_quant_tools.data.polygon_provider._polygon_get",
            return_value={"status": "OK", "results": []},
        ) as mock_get:
            with pytest.raises(DataNotFoundError):
                self._provider().get_ohlcv(malicious, "2023-01-01", "2023-06-01")
        path = mock_get.call_args[0][0]
        assert "?" not in path
        assert "&" not in path
        assert path == (
            "/v2/aggs/ticker/AAPL%3FEXTRA%3D1%26APIKEY%3DSTOLEN/range/"
            "1/day/2023-01-01/2023-06-01"
        )

    def test_ticker_details_symbol_with_injection_characters_is_escaped_in_path(self):
        with patch(
            "standard_quant_tools.data.polygon_provider._polygon_get",
            return_value={"status": "OK", "results": None},
        ) as mock_get:
            with pytest.raises(DataNotFoundError):
                self._provider().get_ticker_info("AAPL/../../secrets")
        path = mock_get.call_args[0][0]
        assert path == "/v3/reference/tickers/AAPL%2F..%2F..%2FSECRETS"

    def test_crypto_style_colon_prefix_is_preserved(self):
        with patch(
            "standard_quant_tools.data.polygon_provider._polygon_get",
            return_value={"status": "OK", "results": []},
        ) as mock_get:
            with pytest.raises(DataNotFoundError):
                self._provider().get_ohlcv("X:BTCUSD", "2023-01-01", "2023-06-01")
        path = mock_get.call_args[0][0]
        assert "/v2/aggs/ticker/X:BTCUSD/range/" in path


class TestPolygonProviderConstruction:
    def test_no_key_anywhere_raises_api_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SQT_POLYGON_API_KEY", raising=False)
        with pytest.raises(APIError, match="No Polygon.io API key found"):
            PolygonProvider()

    def test_via_data_factory_raises_same_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SQT_POLYGON_API_KEY", raising=False)
        with pytest.raises(APIError, match="No Polygon.io API key found"):
            DataFactory.get_provider("polygon")

    def test_constructs_with_explicit_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SQT_POLYGON_API_KEY", raising=False)
        provider = PolygonProvider(api_key="explicit-key")
        assert provider._api_key == "explicit-key"

    def test_abstract_methods_are_all_implemented(self):
        assert PolygonProvider.__abstractmethods__ == frozenset()


class TestGetOhlcvValidation:
    """Interval/symbol validation happens before any network call, so these
    are testable without mocking _polygon_get at all."""

    def _provider(self) -> PolygonProvider:
        return PolygonProvider(api_key="test-key")

    def test_empty_symbol_raises_invalid_symbol(self):
        with pytest.raises(InvalidSymbolError):
            self._provider().get_ohlcv("", "2023-01-01", "2023-06-01")

    def test_unsupported_interval_raises_validation_error(self):
        with pytest.raises(
            ValidationError, match="not supported by the Polygon provider"
        ):
            self._provider().get_ohlcv(
                "AAPL", "2023-01-01", "2023-06-01", interval="90m"
            )


class TestGetOhlcvMockedFetch:
    """Mocks _polygon_get (the network seam), not urlopen — these exercise
    get_ohlcv's own logic (empty results, next_url truncation warning,
    audit recording) independent of HTTP wire details."""

    def _provider(self) -> PolygonProvider:
        return PolygonProvider(api_key="test-key")

    def test_empty_results_raises_data_not_found(self):
        with patch(
            "standard_quant_tools.data.polygon_provider._polygon_get",
            return_value={"status": "OK", "results": []},
        ):
            with pytest.raises(DataNotFoundError):
                self._provider().get_ohlcv("NOTREAL", "2023-01-01", "2023-06-01")

    def test_returns_dataframe_on_success(self):
        payload = {
            "status": "OK",
            "results": [
                {
                    "o": 100.0,
                    "h": 105.0,
                    "l": 99.0,
                    "c": 103.0,
                    "v": 1000,
                    "t": 1672704000000,
                },
            ],
        }
        with patch(
            "standard_quant_tools.data.polygon_provider._polygon_get",
            return_value=payload,
        ):
            df = self._provider().get_ohlcv("AAPL", "2023-01-01", "2023-06-01")
        assert not df.empty
        assert df["Close"].iloc[0] == 103.0

    def test_next_url_present_logs_warning_but_still_returns(self, caplog):
        payload = {
            "status": "OK",
            "next_url": "https://api.polygon.io/v2/aggs/...&cursor=abc",
            "results": [
                {
                    "o": 100.0,
                    "h": 105.0,
                    "l": 99.0,
                    "c": 103.0,
                    "v": 1000,
                    "t": 1672704000000,
                },
            ],
        }
        with patch(
            "standard_quant_tools.data.polygon_provider._polygon_get",
            return_value=payload,
        ):
            with caplog.at_level("WARNING"):
                df = self._provider().get_ohlcv("AAPL", "2023-01-01", "2023-06-01")
        assert not df.empty
        assert any("truncating" in record.message for record in caplog.records)


class TestAsyncFetch:
    def test_get_ohlcv_async_returns_same_data(self):
        import asyncio

        payload = {
            "status": "OK",
            "results": [
                {
                    "o": 100.0,
                    "h": 105.0,
                    "l": 99.0,
                    "c": 103.0,
                    "v": 1000,
                    "t": 1672704000000,
                },
            ],
        }
        with patch(
            "standard_quant_tools.data.polygon_provider._polygon_get",
            return_value=payload,
        ):
            provider = PolygonProvider(api_key="test-key")
            df = asyncio.run(
                provider.get_ohlcv_async("AAPL", "2023-01-01", "2023-06-01")
            )
        assert not df.empty


class TestGetMetadata:
    def test_reports_honest_defaults(self):
        provider = PolygonProvider(api_key="test-key")
        meta = provider.get_metadata("AAPL")
        assert meta.provider == "polygon"
        assert meta.adjusted is True
        assert meta.survivorship_free is False
        assert meta.point_in_time is False
        assert meta.timezone == "America/New_York"
