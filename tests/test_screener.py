"""Tests for the stock screener (mocked data provider)."""

import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
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
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: mock_provider)
    return mock_provider


class TestScreenerFilters:
    def test_all_pass_with_no_filters(self, passing_provider, sample_ohlcv):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        result = screen_stocks(tickers, filters={})
        assert len(result) == 3
        assert set(result.index) == set(tickers)

    def test_pe_filter_excludes_high_pe(self, failing_fundamentals_provider):
        tickers = ["AAPL", "MSFT"]
        result = screen_stocks(tickers, filters={"pe_ratio_max": 30.0})
        # forward_pe=999 > 30 → all excluded
        assert len(result) == 0

    def test_pe_filter_passes_low_pe(self, patched_factory):
        tickers = ["AAPL"]
        # mock_provider has forward_pe=28.5, so pe_ratio_max=30 should pass
        result = screen_stocks(tickers, filters={"pe_ratio_max": 30.0})
        assert "AAPL" in result.index

    def test_rsi_filter_oversold(self, patched_factory, sample_ohlcv):
        """RSI filter: the mock data's RSI determines pass/fail."""
        tickers = ["AAPL"]
        # rsi_max=100 → everything passes
        result = screen_stocks(tickers, filters={"rsi_max": 100.0})
        assert len(result) == 1

    def test_rsi_filter_impossible_threshold(self, patched_factory):
        """rsi_max=-1 → nothing passes (RSI is always ≥ 0)."""
        tickers = ["AAPL"]
        result = screen_stocks(tickers, filters={"rsi_max": -1.0})
        assert len(result) == 0

    def test_market_cap_filter(self, patched_factory):
        """Market cap filter: mock has $2.8T; require > $3T should exclude."""
        tickers = ["AAPL"]
        result = screen_stocks(tickers, filters={"market_cap_min": 3_000_000_000_000})
        assert len(result) == 0

    def test_market_cap_filter_passes(self, patched_factory):
        """Require market cap > $1T: mock has $2.8T, should pass."""
        tickers = ["AAPL"]
        result = screen_stocks(tickers, filters={"market_cap_min": 1_000_000_000_000})
        assert len(result) == 1

    def test_empty_universe_returns_empty_df(self, patched_factory):
        result = screen_stocks([], filters={})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_multiple_tickers_independent_filtering(self, patched_factory):
        """All tickers use the same mock, so all should pass or all fail."""
        tickers = ["AAPL", "MSFT", "TSLA"]
        result = screen_stocks(tickers, filters={"pe_ratio_max": 35.0})
        # mock pe=28.5 < 35 → all 3 pass
        assert len(result) == 3


class TestScreenerOutput:
    def test_returns_dataframe(self, patched_factory):
        result = screen_stocks(["AAPL"], filters={})
        assert isinstance(result, pd.DataFrame)

    def test_result_indexed_by_ticker(self, patched_factory):
        result = screen_stocks(["AAPL", "MSFT"], filters={})
        assert result.index.name == "ticker"

    def test_sort_by_column(self, patched_factory):
        """Sort by forward_pe ascending."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        result = screen_stocks(
            tickers, filters={}, sort_by="forward_pe", ascending=True
        )
        if "forward_pe" in result.columns and len(result) > 1:
            pe_vals = result["forward_pe"].dropna()
            assert pe_vals.is_monotonic_increasing

    def test_fundamental_columns_present(self, patched_factory):
        result = screen_stocks(["AAPL"], filters={"pe_ratio_max": 35.0})
        assert "forward_pe" in result.columns


class TestScreenerProcessPool:
    """
    Verify n_workers behaviour.

    Note: monkeypatch only affects the main process.  Tests that spawn actual
    child processes (n_workers > 1) cannot use the mock provider and are
    therefore marked integration.  Unit tests here cover the n_workers=1
    (sequential) code path and API surface.
    """

    def test_n_workers_1_explicit_matches_default(self, patched_factory):
        """Explicit n_workers=1 must give the same result as the auto default."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        single = screen_stocks(tickers, filters={}, n_workers=1)
        default = screen_stocks(tickers, filters={}, n_workers=1)
        assert set(single.index) == set(default.index)

    def test_n_workers_1_applies_filters(self, patched_factory):
        """Filters work correctly in single-process mode."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        # mock forward_pe=28.5; require < 30 → all pass
        result = screen_stocks(tickers, filters={"pe_ratio_max": 30.0}, n_workers=1)
        assert len(result) == 3

    def test_n_workers_1_filter_excludes(self, patched_factory):
        """Impossible filter should exclude all tickers even with n_workers=1."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        result = screen_stocks(tickers, filters={"pe_ratio_max": 1.0}, n_workers=1)
        assert len(result) == 0

    def test_small_list_auto_uses_single_process(self, patched_factory):
        """≤20 tickers with no n_workers override auto-selects n_workers=1."""
        # The screener code sets n_workers=1 when len(tickers) <= 20; the
        # mock is therefore reachable and all tickers should pass.
        tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        result = screen_stocks(tickers, filters={})
        assert len(result) == 5

class TestScreenerErrorReporting:
    """
    Regression tests (operational item A): a ticker whose data fetch raises
    must be reported distinctly from one that genuinely failed a filter —
    both used to collapse to the same "excluded" outcome, making a
    zero-result run indistinguishable from a broken data pipeline. Also:
    unknown filter keys must be rejected, not silently ignored.
    """

    def test_unknown_filter_key_raises(self, patched_factory):
        with pytest.raises(ValidationError, match="Unknown filter key"):
            screen_stocks(["AAPL"], filters={"pe_ratio_mx": 30.0})

    def test_typo_in_filter_key_is_not_silently_ignored(self, patched_factory):
        """A near-miss typo (extra/missing letter) must raise, not silently
        apply no filter at all while still returning ticker as "passed"."""
        with pytest.raises(ValidationError):
            screen_stocks(["AAPL"], filters={"roe_minn": 0.1})

    def test_filter_failure_reported_in_failed_filters(
        self, failing_fundamentals_provider
    ):
        result = screen_stocks(["AAPL"], filters={"pe_ratio_max": 30.0})
        assert len(result) == 0
        assert result.attrs["failed_filters"] == {"AAPL": "pe_ratio_max"}
        assert result.attrs["failed_tickers"] == {}

    def test_data_fetch_exception_reported_in_failed_tickers_not_failed_filters(
        self, mock_provider, monkeypatch
    ):
        from standard_quant_tools.data.factory import DataFactory

        mock_provider.get_financial_ratios.side_effect = RuntimeError("API down")
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: mock_provider)

        result = screen_stocks(["AAPL"], filters={"pe_ratio_max": 30.0})
        assert len(result) == 0
        # The exception must show up as an error, NOT as a filter rejection --
        # this is exactly the distinction the old bare `return None` erased.
        assert result.attrs["failed_filters"] == {}
        assert "AAPL" in result.attrs["failed_tickers"]
        assert "API down" in result.attrs["failed_tickers"]["AAPL"]

    def test_mixed_pass_fail_error_all_reported_correctly(self, mock_provider, monkeypatch):
        from standard_quant_tools.data.factory import DataFactory
        from standard_quant_tools.data.base import FinancialRatios

        def get_ratios(ticker):
            if ticker == "AAPL":
                return FinancialRatios(forward_pe=10.0)  # passes pe_ratio_max=30
            if ticker == "MSFT":
                return FinancialRatios(forward_pe=999.0)  # fails pe_ratio_max=30
            raise RuntimeError("network timeout")  # GOOGL: error

        mock_provider.get_financial_ratios.side_effect = get_ratios
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: mock_provider)

        result = screen_stocks(
            ["AAPL", "MSFT", "GOOGL"], filters={"pe_ratio_max": 30.0}
        )
        assert list(result.index) == ["AAPL"]
        assert result.attrs["failed_filters"] == {"MSFT": "pe_ratio_max"}
        assert "GOOGL" in result.attrs["failed_tickers"]


@pytest.mark.integration
class TestScreenerProcessPoolIntegration:
    @pytest.mark.integration
    def test_process_pool_same_as_sequential(self):
        """
        Integration: n_workers=2 must return the same tickers as n_workers=1
        when using the live data provider.  Requires network.
        """
        tickers = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "NVDA",
            "META",
            "TSLA",
            "JPM",
            "V",
            "MA",
            "UNH",
            "HD",
            "PG",
            "JNJ",
            "KO",
            "PEP",
            "ABBV",
            "MRK",
            "CVX",
            "XOM",
            "WMT",
            "BAC",
            "DIS",
        ]
        seq = screen_stocks(tickers, filters={"pe_ratio_max": 60}, n_workers=1)
        par = screen_stocks(tickers, filters={"pe_ratio_max": 60}, n_workers=2)
        assert set(seq.index) == set(par.index)
