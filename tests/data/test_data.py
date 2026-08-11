"""
Tests for the data layer.

Unit tests: mock the yfinance calls (no network).
Integration tests: require live network; marked with @pytest.mark.integration.
"""

from pathlib import Path

import pandas as pd
import pytest

import standard_quant_tools.data._cache as cache_module
from standard_quant_tools.data.base import FinancialRatios, TickerInfo
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    InvalidSymbolError,
)


@pytest.fixture(autouse=True)
def redirect_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect all Parquet writes/reads to a temp directory for every test
    so the real persistent disk cache never leaks between tests -- see
    tests/test_parquet_cache.py for the original pattern this mirrors.
    Most tests here use a fully-mocked DataFactory and never touch the
    cache, but TestLiveYFinance's integration tests do."""
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path)
    cache_module._session_cache.clear()


# ── Mocked unit tests ─────────────────────────────────────────────────────────


class TestGetOHLCV:
    def test_returns_dataframe_with_correct_columns(
        self, patched_factory, sample_ohlcv
    ):
        provider = DataFactory.get_provider()
        result = provider.get_ohlcv("AAPL", "2023-01-01", "2024-01-01")
        assert isinstance(result, pd.DataFrame)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    def test_returns_nonempty_dataframe(self, patched_factory):
        provider = DataFactory.get_provider()
        result = provider.get_ohlcv("AAPL", "2023-01-01", "2024-01-01")
        assert not result.empty

    def test_invalid_symbol_raises_error(self, patched_factory, mock_provider):
        mock_provider.get_ohlcv.side_effect = InvalidSymbolError("empty symbol")
        provider = DataFactory.get_provider()
        with pytest.raises(InvalidSymbolError):
            provider.get_ohlcv("", "2023-01-01", "2024-01-01")

    def test_data_not_found_raises_correct_error(self, patched_factory, mock_provider):
        mock_provider.get_ohlcv.side_effect = DataNotFoundError("symbol not found")
        provider = DataFactory.get_provider()
        with pytest.raises(DataNotFoundError):
            provider.get_ohlcv("NOTREAL_XYZ", "2023-01-01", "2024-01-01")


class TestGetTickerInfo:
    def test_returns_ticker_info_model(self, patched_factory):
        provider = DataFactory.get_provider()
        result = provider.get_ticker_info("AAPL")
        assert isinstance(result, TickerInfo)

    def test_ticker_info_has_correct_symbol(self, patched_factory):
        provider = DataFactory.get_provider()
        result = provider.get_ticker_info("AAPL")
        assert result.symbol == "AAPL"

    def test_ticker_info_has_name_and_sector(self, patched_factory):
        provider = DataFactory.get_provider()
        result = provider.get_ticker_info("AAPL")
        assert result.name != ""
        assert result.sector != ""


class TestGetFinancialRatios:
    def test_returns_financial_ratios_model(self, patched_factory):
        provider = DataFactory.get_provider()
        result = provider.get_financial_ratios("AAPL")
        assert isinstance(result, FinancialRatios)

    def test_ratios_are_numeric_or_none(self, patched_factory):
        provider = DataFactory.get_provider()
        result = provider.get_financial_ratios("AAPL")
        for field in (
            "forward_pe",
            "trailing_pe",
            "price_to_book",
            "debt_to_equity",
            "return_on_equity",
            "profit_margins",
            "dividend_yield",
        ):
            value = getattr(result, field)
            assert value is None or isinstance(value, (int, float))


class TestDataFactory:
    def test_factory_unknown_source_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown data provider"):
            DataFactory.get_provider(source="unknown_provider_xyz")

    def test_factory_returns_provider_for_yfinance(self):
        from standard_quant_tools.data.yfinance_provider import YFinanceProvider

        provider = DataFactory.get_provider("yfinance")
        assert isinstance(provider, YFinanceProvider)


class TestAsyncFetch:
    def test_get_ohlcv_async_returns_same_data(self, patched_factory, sample_ohlcv):
        import asyncio

        provider = DataFactory.get_provider()
        result = asyncio.run(
            provider.get_ohlcv_async("AAPL", "2023-01-01", "2024-01-01")
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


# ── Live integration tests ─────────────────────────────────────────────────────


@pytest.mark.integration
class TestLiveYFinance:
    """These tests hit the real yfinance API. Run with: pytest -m integration"""

    def test_live_ohlcv_fetch(self):
        provider = DataFactory.get_provider("yfinance")
        df = provider.get_ohlcv("AAPL", "2023-01-01", "2023-06-01")
        assert not df.empty
        assert "Close" in df.columns
        assert (df["Close"] > 0).all()

    def test_live_ticker_info(self):
        provider = DataFactory.get_provider("yfinance")
        info = provider.get_ticker_info("MSFT")
        assert info.symbol == "MSFT"
        assert info.sector is not None

    def test_live_financial_ratios(self):
        provider = DataFactory.get_provider("yfinance")
        ratios = provider.get_financial_ratios("AAPL")
        assert isinstance(ratios, FinancialRatios)

    def test_live_invalid_symbol_raises(self):
        provider = DataFactory.get_provider("yfinance")
        with pytest.raises((DataNotFoundError, InvalidSymbolError, APIError)):
            provider.get_ohlcv("INVALID_SYM_XYZ_99999", "2023-01-01", "2023-06-01")

    def test_live_empty_symbol_raises_invalid_symbol(self):
        provider = DataFactory.get_provider("yfinance")
        with pytest.raises(InvalidSymbolError):
            provider.get_ohlcv("", "2023-01-01", "2023-06-01")
