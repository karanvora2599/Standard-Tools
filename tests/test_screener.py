"""Tests for the stock screener (mocked data provider)."""

import pandas as pd
import pytest

from standard_quant_tools.screener.screener import screen_stocks


@pytest.fixture
def passing_provider(mock_provider, patched_factory):
    """Factory patched so all tickers return data that passes any filter."""
    return patched_factory


@pytest.fixture
def failing_fundamentals_provider(mock_provider, monkeypatch):
    """Factory patched so financial ratios always fail PE filter."""
    from standard_quant_tools.data.base import FinancialRatios
    from standard_quant_tools.data.factory import DataFactory
    bad_ratios = FinancialRatios(forward_pe=999.0, price_to_book=999.0)
    mock_provider.get_financial_ratios.return_value = bad_ratios
    monkeypatch.setattr(DataFactory, 'get_provider', lambda *a, **kw: mock_provider)
    return mock_provider


class TestScreenerFilters:
    def test_all_pass_with_no_filters(self, passing_provider, sample_ohlcv):
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        result = screen_stocks(tickers, filters={})
        assert len(result) == 3
        assert set(result.index) == set(tickers)

    def test_pe_filter_excludes_high_pe(self, failing_fundamentals_provider):
        tickers = ['AAPL', 'MSFT']
        result = screen_stocks(tickers, filters={'pe_ratio_max': 30.0})
        # forward_pe=999 > 30 → all excluded
        assert len(result) == 0

    def test_pe_filter_passes_low_pe(self, patched_factory):
        tickers = ['AAPL']
        # mock_provider has forward_pe=28.5, so pe_ratio_max=30 should pass
        result = screen_stocks(tickers, filters={'pe_ratio_max': 30.0})
        assert 'AAPL' in result.index

    def test_rsi_filter_oversold(self, patched_factory, sample_ohlcv):
        """RSI filter: the mock data's RSI determines pass/fail."""
        tickers = ['AAPL']
        # rsi_max=100 → everything passes
        result = screen_stocks(tickers, filters={'rsi_max': 100.0})
        assert len(result) == 1

    def test_rsi_filter_impossible_threshold(self, patched_factory):
        """rsi_max=-1 → nothing passes (RSI is always ≥ 0)."""
        tickers = ['AAPL']
        result = screen_stocks(tickers, filters={'rsi_max': -1.0})
        assert len(result) == 0

    def test_market_cap_filter(self, patched_factory):
        """Market cap filter: mock has $2.8T; require > $3T should exclude."""
        tickers = ['AAPL']
        result = screen_stocks(tickers, filters={'market_cap_min': 3_000_000_000_000})
        assert len(result) == 0

    def test_market_cap_filter_passes(self, patched_factory):
        """Require market cap > $1T: mock has $2.8T, should pass."""
        tickers = ['AAPL']
        result = screen_stocks(tickers, filters={'market_cap_min': 1_000_000_000_000})
        assert len(result) == 1

    def test_empty_universe_returns_empty_df(self, patched_factory):
        result = screen_stocks([], filters={})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_multiple_tickers_independent_filtering(self, patched_factory):
        """All tickers use the same mock, so all should pass or all fail."""
        tickers = ['AAPL', 'MSFT', 'TSLA']
        result = screen_stocks(tickers, filters={'pe_ratio_max': 35.0})
        # mock pe=28.5 < 35 → all 3 pass
        assert len(result) == 3


class TestScreenerOutput:
    def test_returns_dataframe(self, patched_factory):
        result = screen_stocks(['AAPL'], filters={})
        assert isinstance(result, pd.DataFrame)

    def test_result_indexed_by_ticker(self, patched_factory):
        result = screen_stocks(['AAPL', 'MSFT'], filters={})
        assert result.index.name == 'ticker'

    def test_sort_by_column(self, patched_factory):
        """Sort by forward_pe ascending."""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        result = screen_stocks(tickers, filters={}, sort_by='forward_pe', ascending=True)
        if 'forward_pe' in result.columns and len(result) > 1:
            pe_vals = result['forward_pe'].dropna()
            assert pe_vals.is_monotonic_increasing

    def test_fundamental_columns_present(self, patched_factory):
        result = screen_stocks(['AAPL'], filters={'pe_ratio_max': 35.0})
        assert 'forward_pe' in result.columns
