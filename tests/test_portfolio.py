"""Tests for the portfolio module: build_portfolio, portfolio_metrics, correlation_matrix."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.portfolio import (
    build_portfolio,
    correlation_matrix,
    portfolio_metrics,
)


@pytest.fixture(scope='module')
def multi_asset_returns():
    """3-asset returns DataFrame for portfolio tests."""
    np.random.seed(99)
    n = 252
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    # Correlated assets with different drift/vol
    base = np.random.normal(0.0003, 0.01, n)
    r1 = base + np.random.normal(0, 0.005, n)
    r2 = base * 0.8 + np.random.normal(0, 0.007, n)
    r3 = base * 0.5 + np.random.normal(0, 0.012, n)
    return pd.DataFrame({'A': r1, 'B': r2, 'C': r3}, index=dates)


class TestBuildPortfolio:
    def test_output_is_series(self, multi_asset_returns):
        result = build_portfolio(multi_asset_returns, [1/3, 1/3, 1/3])
        assert isinstance(result, pd.Series)

    def test_output_length_matches_input(self, multi_asset_returns):
        result = build_portfolio(multi_asset_returns, [1/3, 1/3, 1/3])
        assert len(result) == len(multi_asset_returns)

    def test_equal_weights_is_row_mean(self, multi_asset_returns):
        """Equal weights → portfolio return = arithmetic mean of asset returns."""
        n = multi_asset_returns.shape[1]
        weights = [1/n] * n
        result = build_portfolio(multi_asset_returns, weights)
        expected = multi_asset_returns.mean(axis=1)
        pd.testing.assert_series_equal(result, expected, check_names=False, rtol=1e-10)

    def test_single_asset_weight_one_is_identity(self, multi_asset_returns):
        """100% in asset A → portfolio returns = asset A returns."""
        result = build_portfolio(multi_asset_returns, [1.0, 0.0, 0.0])
        pd.testing.assert_series_equal(result, multi_asset_returns['A'], check_names=False, rtol=1e-10)

    def test_mismatched_weights_length_raises(self, multi_asset_returns):
        with pytest.raises(ValidationError):
            build_portfolio(multi_asset_returns, [0.5, 0.5])

    def test_weights_not_summing_to_one_raises(self, multi_asset_returns):
        with pytest.raises(ValidationError):
            build_portfolio(multi_asset_returns, [0.5, 0.5, 0.5])


class TestPortfolioMetrics:
    def test_returns_required_keys(self, multi_asset_returns):
        result = portfolio_metrics(multi_asset_returns, [1/3, 1/3, 1/3])
        required = {
            'annualized_return', 'annualized_volatility', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'calmar_ratio',
            'var_95', 'cvar_95', 'total_return',
        }
        assert required.issubset(result.keys())

    def test_annualized_vol_lower_than_max_single_asset(self, multi_asset_returns):
        """Diversification: portfolio vol should be ≤ max individual asset vol."""
        portfolio_vol = portfolio_metrics(multi_asset_returns, [1/3, 1/3, 1/3])['annualized_volatility']
        max_asset_vol = multi_asset_returns.std().max() * np.sqrt(252)
        assert portfolio_vol <= max_asset_vol * 1.05  # 5% tolerance for imperfect correlation

    def test_max_drawdown_nonpositive(self, multi_asset_returns):
        result = portfolio_metrics(multi_asset_returns, [1/3, 1/3, 1/3])
        assert result['max_drawdown'] <= 0

    def test_var_less_than_cvar(self, multi_asset_returns):
        result = portfolio_metrics(multi_asset_returns, [1/3, 1/3, 1/3])
        assert result['cvar_95'] >= result['var_95']

    def test_information_ratio_returned_with_benchmark(self, multi_asset_returns):
        benchmark = multi_asset_returns['A']
        result = portfolio_metrics(
            multi_asset_returns, [1/3, 1/3, 1/3], benchmark_returns=benchmark
        )
        assert 'information_ratio' in result
        assert isinstance(result['information_ratio'], float)

    def test_weights_stored_in_result(self, multi_asset_returns):
        weights = [0.5, 0.3, 0.2]
        result = portfolio_metrics(multi_asset_returns, weights)
        assert result['weights'] == pytest.approx(weights, abs=1e-10)


class TestCorrelationMatrix:
    def test_returns_dataframe(self, multi_asset_returns):
        result = correlation_matrix(multi_asset_returns)
        assert isinstance(result, pd.DataFrame)

    def test_diagonal_is_one(self, multi_asset_returns):
        result = correlation_matrix(multi_asset_returns)
        for col in result.columns:
            assert result.loc[col, col] == pytest.approx(1.0, abs=1e-10)

    def test_symmetric(self, multi_asset_returns):
        result = correlation_matrix(multi_asset_returns)
        pd.testing.assert_frame_equal(result, result.T, rtol=1e-10)

    def test_values_bounded_minus_1_to_1(self, multi_asset_returns):
        result = correlation_matrix(multi_asset_returns)
        assert (result >= -1.0).all(axis=None) and (result <= 1.0).all(axis=None)

    def test_perfectly_correlated_assets_yield_one(self):
        n = 100
        r = pd.Series(np.random.normal(0, 0.01, n))
        df = pd.DataFrame({'X': r, 'Y': r})
        result = correlation_matrix(df)
        assert result.loc['X', 'Y'] == pytest.approx(1.0, abs=1e-10)
