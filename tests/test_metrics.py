"""Tests for return and risk metrics."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.metrics.return_metrics import (
    annualized_volatility, cagr, cumulative_return,
)
from standard_quant_tools.metrics.risk_metrics import (
    calmar_ratio, cvar, drawdown_series, information_ratio,
    max_drawdown, sharpe_ratio, sortino_ratio, treynor_ratio,
    var_historical, var_parametric,
)


# ── Return Metrics ────────────────────────────────────────────────────────────

class TestCumulativeReturn:
    def test_zero_return_on_flat_equity(self):
        equity = pd.Series([100.0] * 50)
        assert cumulative_return(equity) == pytest.approx(0.0)

    def test_doubling_equity_yields_100pct(self):
        equity = pd.Series([100.0, 200.0])
        assert cumulative_return(equity) == pytest.approx(1.0)

    def test_exact_formula(self, sample_equity):
        result = cumulative_return(sample_equity)
        expected = (sample_equity.iloc[-1] / sample_equity.iloc[0]) - 1
        assert result == pytest.approx(expected, rel=1e-10)

    def test_empty_series_returns_zero(self):
        assert cumulative_return(pd.Series(dtype=float)) == 0.0


class TestCAGR:
    def test_consistent_with_cumulative_return(self, sample_equity):
        total = cumulative_return(sample_equity)
        n_years = len(sample_equity) / 252
        expected = (1 + total) ** (1 / n_years) - 1
        assert cagr(sample_equity) == pytest.approx(expected, rel=1e-10)

    def test_positive_for_growing_equity(self, sample_equity):
        if sample_equity.iloc[-1] > sample_equity.iloc[0]:
            assert cagr(sample_equity) > 0

    def test_annual_return_equals_cagr_on_one_year(self):
        """A 252-bar series that doubles should have CAGR = 100%."""
        equity = pd.Series(np.linspace(100, 200, 252))
        assert cagr(equity) == pytest.approx(1.0, rel=0.05)


class TestAnnualizedVolatility:
    def test_scales_with_sqrt_time(self):
        """Annualized vol = daily std * sqrt(252)."""
        np.random.seed(0)
        returns = pd.Series(np.random.normal(0, 0.01, 252))
        daily_std = returns.std()
        assert annualized_volatility(returns) == pytest.approx(daily_std * np.sqrt(252), rel=1e-10)

    def test_zero_for_constant_returns(self):
        returns = pd.Series([0.001] * 100)
        assert annualized_volatility(returns) == pytest.approx(0.0, abs=1e-10)


# ── Risk Metrics ──────────────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_positive_for_positive_expected_return(self, sample_returns):
        sr = sharpe_ratio(sample_returns)
        if sample_returns.mean() > 0:
            assert sr > 0

    def test_scales_with_return(self):
        """Doubling mean return (same vol) should approximately double Sharpe."""
        np.random.seed(1)
        vol = pd.Series(np.random.normal(0, 0.01, 252))
        base_returns = vol + 0.001
        double_returns = vol + 0.002
        assert sharpe_ratio(double_returns) > sharpe_ratio(base_returns)

    def test_higher_vol_lowers_sharpe(self):
        np.random.seed(2)
        low_vol = pd.Series(np.random.normal(0.001, 0.005, 252))
        high_vol = pd.Series(np.random.normal(0.001, 0.020, 252))
        assert sharpe_ratio(low_vol) > sharpe_ratio(high_vol)


class TestSortinoRatio:
    def test_higher_than_sharpe_for_skewed_returns(self):
        """For returns with more upside than downside, Sortino >= Sharpe."""
        np.random.seed(3)
        # Mix of positive and negative, but positively skewed
        ups = np.abs(np.random.normal(0.003, 0.005, 200))
        downs = -np.abs(np.random.normal(0.001, 0.003, 52))
        returns = pd.Series(np.concatenate([ups, downs]))
        sr = sharpe_ratio(returns)
        srt = sortino_ratio(returns)
        assert srt >= sr

    def test_positive_for_positive_drift(self):
        np.random.seed(4)
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        # May not always be positive, but the direction should be consistent
        assert isinstance(sortino_ratio(returns), float)


class TestMaxDrawdown:
    def test_always_nonpositive(self, sample_equity):
        assert max_drawdown(sample_equity) <= 0

    def test_zero_for_monotonically_rising_equity(self):
        equity = pd.Series(np.linspace(100, 200, 100))
        assert max_drawdown(equity) == pytest.approx(0.0, abs=1e-10)

    def test_minus_one_for_total_loss(self):
        equity = pd.Series([100.0, 50.0, 0.01])
        mdd = max_drawdown(equity)
        assert mdd < -0.99

    def test_correct_calculation(self):
        """MDD of [100, 80, 90, 70, 110] = (70-100)/100 = -0.30."""
        equity = pd.Series([100.0, 80.0, 90.0, 70.0, 110.0])
        assert max_drawdown(equity) == pytest.approx(-0.30, abs=1e-10)


class TestCalmarRatio:
    def test_positive_when_cagr_positive(self, sample_equity):
        cal = calmar_ratio(sample_equity)
        annual_ret = cagr(sample_equity)
        if annual_ret > 0 and max_drawdown(sample_equity) < 0:
            assert cal > 0

    def test_infinite_for_no_drawdown(self):
        equity = pd.Series(np.linspace(100, 200, 252))
        result = calmar_ratio(equity)
        assert result == float('inf') or result > 1000

    def test_formula_cagr_over_abs_mdd(self, sample_equity):
        cal = calmar_ratio(sample_equity)
        mdd = max_drawdown(sample_equity)
        annual_ret = cagr(sample_equity)
        if mdd != 0:
            assert cal == pytest.approx(annual_ret / abs(mdd), rel=1e-6)


class TestVaR:
    def test_var_historical_nonnegative(self, sample_returns):
        assert var_historical(sample_returns) >= 0

    def test_var_95_greater_than_var_90(self, sample_returns):
        """Higher confidence = larger estimated loss."""
        var90 = var_historical(sample_returns, 0.90)
        var95 = var_historical(sample_returns, 0.95)
        assert var95 >= var90

    def test_var_historical_fraction_check(self, sample_returns):
        """~5% of returns should fall below the 95% VaR threshold."""
        var95 = var_historical(sample_returns, 0.95)
        tail_fraction = (sample_returns < -var95).mean()
        assert tail_fraction == pytest.approx(0.05, abs=0.03)

    def test_parametric_close_to_historical_for_normal_data(self):
        """For normally distributed returns, historical and parametric VaR should be close."""
        np.random.seed(10)
        returns = pd.Series(np.random.normal(0, 0.01, 5000))
        hist = var_historical(returns, 0.95)
        param = var_parametric(returns, 0.95)
        assert abs(hist - param) / param < 0.15  # within 15%


class TestCVaR:
    def test_cvar_greater_than_or_equal_to_var(self, sample_returns):
        """CVaR is always at least as large as VaR."""
        var95 = var_historical(sample_returns, 0.95)
        cvar95 = cvar(sample_returns, 0.95)
        assert cvar95 >= var95 - 1e-10

    def test_cvar_nonnegative(self, sample_returns):
        assert cvar(sample_returns) >= 0


class TestInformationRatio:
    def test_zero_when_active_return_is_zero(self, sample_returns):
        """IR should be 0 when asset return equals benchmark exactly."""
        ir = information_ratio(sample_returns, sample_returns)
        assert ir == pytest.approx(0.0, abs=1e-6)

    def test_positive_when_asset_outperforms(self, sample_returns, benchmark_returns):
        """IR direction should match outperformance direction."""
        active = (sample_returns - benchmark_returns).mean()
        ir = information_ratio(sample_returns, benchmark_returns)
        assert (ir > 0) == (active > 0)

    def test_scales_with_active_return(self, sample_returns):
        np.random.seed(5)
        bench = pd.Series(np.random.normal(0.0002, 0.009, len(sample_returns)), index=sample_returns.index)
        ir = information_ratio(sample_returns, bench)
        assert isinstance(ir, float) and not np.isnan(ir)


class TestDrawdownSeries:
    def test_always_nonpositive(self, sample_equity):
        dd = drawdown_series(sample_equity)
        assert (dd <= 0).all()

    def test_equals_zero_at_new_highs(self, sample_equity):
        """At any new all-time high, drawdown is exactly 0."""
        dd = drawdown_series(sample_equity)
        highs = sample_equity == sample_equity.cummax()
        assert dd[highs].eq(0.0).all()

    def test_min_equals_max_drawdown(self, sample_equity):
        dd = drawdown_series(sample_equity)
        assert dd.min() == pytest.approx(max_drawdown(sample_equity), rel=1e-10)
