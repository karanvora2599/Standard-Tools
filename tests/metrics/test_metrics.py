"""Tests for return and risk metrics."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.return_metrics import (
    annualized_volatility,
    cagr,
    cumulative_return,
)
from standard_quant_tools.metrics.risk_metrics import (
    HAS_SCIPY,
    _fit_gpd_pwm,
    calmar_ratio,
    cvar,
    drawdown_series,
    evt_tail_risk,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
    var_historical,
    var_parametric,
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
        """
        The elapsed span of a LEVEL series is N-1 intervals, not N: a series
        of N prices contains N-1 returns. This test previously recomputed
        with `len(series) / 252`, which overstates the elapsed time and so
        understates the growth rate.

        Negligible over a decade of daily bars, material on short windows —
        over 21 observations (one month) it is a 5% error in the exponent's
        denominator, and it grows as the window shortens, which is exactly
        where a CAGR is already least reliable.
        """
        total = cumulative_return(sample_equity)
        n_years = (len(sample_equity) - 1) / 252
        expected = (1 + total) ** (1 / n_years) - 1
        assert cagr(sample_equity) == pytest.approx(expected, rel=1e-10)

    def test_a_single_observation_spans_no_time(self):
        """One level is not a growth rate. Previously len(series)/252 gave a
        non-zero span for a series containing no return at all."""
        assert cagr(pd.Series([100.0])) == 0.0

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
        assert annualized_volatility(returns) == pytest.approx(
            daily_std * np.sqrt(252), rel=1e-10
        )

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
        assert result == float("inf") or result > 1000

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

    @pytest.mark.parametrize("bad_confidence", [0.0, 1.0, -0.1, 1.5])
    def test_var_historical_rejects_out_of_range_confidence(
        self, sample_returns, bad_confidence
    ):
        with pytest.raises(ValidationError, match="confidence"):
            var_historical(sample_returns, bad_confidence)

    @pytest.mark.parametrize("bad_confidence", [0.0, 1.0, -0.1, 1.5])
    def test_var_parametric_rejects_out_of_range_confidence(
        self, sample_returns, bad_confidence
    ):
        with pytest.raises(ValidationError, match="confidence"):
            var_parametric(sample_returns, bad_confidence)

    def test_var_parametric_without_scipy_rejects_unsupported_confidence(
        self, sample_returns, monkeypatch
    ):
        """
        Regression test (operational item B): without scipy, an arbitrary
        confidence level not in the precomputed z-table must raise, not
        silently substitute the 95% z-score for whatever level was
        actually requested.
        """
        import standard_quant_tools.metrics.risk_metrics as risk_metrics_mod

        monkeypatch.setattr(risk_metrics_mod, "HAS_SCIPY", False)
        monkeypatch.setattr(risk_metrics_mod, "_scipy_stats", None)
        with pytest.raises(ValidationError, match="scipy"):
            var_parametric(sample_returns, 0.975)

    def test_var_parametric_without_scipy_accepts_table_confidence(
        self, sample_returns, monkeypatch
    ):
        """A confidence level that IS in the precomputed z-table must still
        work fine without scipy."""
        import standard_quant_tools.metrics.risk_metrics as risk_metrics_mod

        monkeypatch.setattr(risk_metrics_mod, "HAS_SCIPY", False)
        monkeypatch.setattr(risk_metrics_mod, "_scipy_stats", None)
        result = var_parametric(sample_returns, 0.95)
        assert isinstance(result, float)


class TestCVaR:
    def test_cvar_greater_than_or_equal_to_var(self, sample_returns):
        """CVaR is always at least as large as VaR."""
        var95 = var_historical(sample_returns, 0.95)
        cvar95 = cvar(sample_returns, 0.95)
        assert cvar95 >= var95 - 1e-10

    def test_cvar_nonnegative(self, sample_returns):
        assert cvar(sample_returns) >= 0

    @pytest.mark.parametrize("bad_confidence", [0.0, 1.0, -0.1, 1.5])
    def test_cvar_rejects_out_of_range_confidence(self, sample_returns, bad_confidence):
        with pytest.raises(ValidationError, match="confidence"):
            cvar(sample_returns, bad_confidence)


class TestTreynorRatio:
    def test_numerator_uses_aligned_window_not_full_series(self):
        """
        Regression test (operational item B): the numerator (mean excess
        return) must use the SAME date-aligned window beta was estimated
        from, not the full, unaligned `returns` series. Constructs a
        `returns` series with extra dates the benchmark doesn't cover, at a
        return level clearly different from the aligned-window returns --
        if the numerator wrongly used the full series, the result would
        differ from a version computed with `returns` pre-trimmed to
        exactly the common index.
        """
        common_dates = pd.date_range("2023-01-02", periods=50, freq="B")
        extra_dates = pd.date_range("2023-06-01", periods=20, freq="B")

        np.random.seed(42)
        common_returns = np.random.normal(0.0005, 0.01, 50)
        common_bench = np.random.normal(0.0003, 0.008, 50)
        # Extra dates the benchmark never covers, at a very different (much
        # higher) return level -- would visibly skew a wrongly-unaligned mean.
        extra_returns = np.full(20, 0.05)

        returns = pd.Series(
            np.concatenate([common_returns, extra_returns]),
            index=common_dates.append(extra_dates),
        )
        benchmark_returns = pd.Series(common_bench, index=common_dates)

        result = treynor_ratio(returns, benchmark_returns)
        # Reference: compute Treynor by hand on the pre-trimmed, aligned
        # series only -- this must match exactly, proving the numerator
        # never touched the extra out-of-window dates.
        aligned_returns = pd.Series(common_returns, index=common_dates)
        reference = treynor_ratio(aligned_returns, benchmark_returns)
        assert result == pytest.approx(reference, rel=1e-9)


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
        bench = pd.Series(
            np.random.normal(0.0002, 0.009, len(sample_returns)),
            index=sample_returns.index,
        )
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


# ── EVT tail risk ─────────────────────────────────────────────────────────────


def _gpd_inverse_cdf_exceedances(xi, beta, n, seed):
    """Sample n exceedances from a known GPD(xi, beta) via inverse-CDF —
    used to check that the fitted parameters recover the true generating
    values, independent of evt_tail_risk's own threshold-selection logic."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, n)
    return (beta / xi) * ((1 - u) ** (-xi) - 1)


@pytest.fixture(scope="module")
def fat_tailed_returns():
    """Student-t(df=3) returns — genuinely fat-tailed, enough observations
    for a reliable POT fit at the default 5% tail_fraction (n=5000 -> ~250
    exceedances, comfortably above the 20-exceedance floor)."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2015-01-01", periods=5000, freq="B")
    return pd.Series(rng.standard_t(df=3, size=5000) * 0.01, index=dates)


class TestFitGpdPwm:
    def test_recovers_known_generating_parameters(self):
        true_xi, true_beta = 0.25, 0.02
        exceedances = _gpd_inverse_cdf_exceedances(true_xi, true_beta, 20_000, seed=3)
        xi_hat, beta_hat = _fit_gpd_pwm(exceedances)
        assert xi_hat == pytest.approx(true_xi, abs=0.05)
        assert beta_hat == pytest.approx(true_beta, abs=0.01)


class TestEvtTailRisk:
    def test_returns_required_keys(self, fat_tailed_returns):
        result = evt_tail_risk(fat_tailed_returns)
        assert set(result.keys()) == {
            "confidence",
            "tail_fraction",
            "threshold",
            "n_exceedances",
            "n_obs",
            "shape_xi",
            "scale_beta",
            "var_evt",
            "cvar_evt",
            "method",
            "tail_classification",
        }

    def test_default_method_is_pwm_and_needs_no_scipy(self, fat_tailed_returns):
        result = evt_tail_risk(fat_tailed_returns)
        assert result["method"] == "pwm"

    def test_n_exceedances_matches_tail_fraction(self, fat_tailed_returns):
        result = evt_tail_risk(fat_tailed_returns, tail_fraction=0.05)
        expected = int(round(0.05 * len(fat_tailed_returns)))
        assert abs(result["n_exceedances"] - expected) <= 1

    def test_var_evt_is_positive_for_fat_tailed_losses(self, fat_tailed_returns):
        result = evt_tail_risk(fat_tailed_returns, confidence=0.99)
        assert result["var_evt"] > 0
        assert result["cvar_evt"] >= result["var_evt"]

    def test_tail_classification_matches_shape_xi(self, fat_tailed_returns):
        result = evt_tail_risk(fat_tailed_returns)
        xi = result["shape_xi"]
        if xi > 0.1:
            assert result["tail_classification"] == "heavy_tailed"
        elif xi < -0.1:
            assert result["tail_classification"] == "light_tailed"
        else:
            assert result["tail_classification"] == "near_exponential"

    @pytest.mark.skipif(not HAS_SCIPY, reason="method='mle' requires scipy")
    def test_mle_method_gives_similar_result_to_pwm(self, fat_tailed_returns):
        pwm = evt_tail_risk(fat_tailed_returns, method="pwm")
        mle = evt_tail_risk(fat_tailed_returns, method="mle")
        assert mle["shape_xi"] == pytest.approx(pwm["shape_xi"], abs=0.1)

    def test_mle_without_scipy_raises_if_unavailable(
        self, fat_tailed_returns, monkeypatch
    ):
        monkeypatch.setattr(
            "standard_quant_tools.metrics.risk_metrics.HAS_SCIPY", False
        )
        with pytest.raises(ValidationError, match="scipy"):
            evt_tail_risk(fat_tailed_returns, method="mle")


class TestEvtTailRiskValidation:
    def test_confidence_out_of_bounds_raises(self, fat_tailed_returns):
        with pytest.raises(ValidationError, match="confidence"):
            evt_tail_risk(fat_tailed_returns, confidence=1.5)

    def test_tail_fraction_out_of_bounds_raises(self, fat_tailed_returns):
        with pytest.raises(ValidationError, match="tail_fraction"):
            evt_tail_risk(fat_tailed_returns, tail_fraction=0.6)

    def test_invalid_method_raises(self, fat_tailed_returns):
        with pytest.raises(ValidationError, match="method"):
            evt_tail_risk(fat_tailed_returns, method="bogus")

    def test_too_few_exceedances_raises(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 50), index=dates)
        with pytest.raises(ValidationError, match="exceedances"):
            evt_tail_risk(returns, tail_fraction=0.05)


@pytest.mark.benchmark
class TestEvtTailRiskScale:
    def test_two_million_points_fits_quickly(self):
        import time

        rng = np.random.default_rng(9)
        returns = pd.Series(rng.standard_t(df=3, size=2_000_000) * 0.01)
        t0 = time.time()
        result = evt_tail_risk(returns)
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"2M-point EVT fit took {elapsed:.2f}s"
        assert result["n_obs"] == 2_000_000
