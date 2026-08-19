"""Tests for regression and analysis functions: beta, alpha, R-squared."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.regression import calculate_beta, rolling_beta
from standard_quant_tools.error import ValidationError


class TestCalculateBeta:
    def test_returns_required_keys(self, sample_returns, benchmark_returns):
        result = calculate_beta(sample_returns, benchmark_returns)
        assert set(result.keys()) == {"alpha", "beta", "r_squared"}

    def test_beta_one_when_asset_equals_benchmark(self, sample_returns):
        """When asset and benchmark are identical, beta must be exactly 1."""
        result = calculate_beta(sample_returns, sample_returns)
        assert result["beta"] == pytest.approx(1.0, abs=1e-6)

    def test_alpha_near_zero_when_asset_equals_benchmark(self, sample_returns):
        result = calculate_beta(sample_returns, sample_returns)
        assert result["alpha"] == pytest.approx(0.0, abs=1e-10)

    def test_r_squared_one_when_perfectly_correlated(self, sample_returns):
        result = calculate_beta(sample_returns, sample_returns)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_r_squared_bounded_0_to_1(self, sample_returns, benchmark_returns):
        result = calculate_beta(sample_returns, benchmark_returns)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_beta_two_for_double_leverage(self, sample_returns):
        """An asset with exactly 2x leverage should have beta = 2."""
        leveraged = sample_returns * 2
        result = calculate_beta(leveraged, sample_returns)
        assert result["beta"] == pytest.approx(2.0, abs=1e-6)

    def test_negative_beta_for_inverse_asset(self, sample_returns):
        inverse = -sample_returns
        result = calculate_beta(inverse, sample_returns)
        assert result["beta"] == pytest.approx(-1.0, abs=1e-6)

    def test_low_r_squared_for_uncorrelated_assets(self):
        """Uncorrelated random series should have R² close to 0."""
        np.random.seed(42)
        r1 = pd.Series(np.random.normal(0, 0.01, 500))
        r2 = pd.Series(np.random.normal(0, 0.01, 500))
        result = calculate_beta(r1, r2)
        assert result["r_squared"] < 0.10

    def test_index_alignment_handles_different_lengths(self, sample_returns):
        """calculate_beta should align on common index and not crash."""
        short_bench = sample_returns.iloc[:100]
        result = calculate_beta(sample_returns, short_bench)
        assert isinstance(result["beta"], float)

    def test_nan_in_input_raises(self, sample_returns, benchmark_returns):
        bad = sample_returns.copy()
        bad.iloc[5] = np.nan
        with pytest.raises(ValidationError, match="non-finite"):
            calculate_beta(bad, benchmark_returns)

    def test_minimal_data_is_not_estimable(self):
        """
        A single overlapping point cannot support an OLS fit, so all three
        statistics are NaN — "not estimable".

        This test previously asserted a "safe zero dict" and called that the
        intended behaviour. Zero is not safe here: 0.0 is ALSO a legitimate
        beta (a market-neutral asset), so nothing downstream could tell a
        failed estimate from a real measurement. Two consumers read it the
        wrong way — the screener passed an unestimable ticker through a
        beta_max ceiling, and treynor_ratio turned "no overlapping benchmark
        data" into a plausible-looking risk-adjusted return.
        """
        result = calculate_beta(pd.Series([0.01]), pd.Series([0.01]))
        assert set(result) == {"alpha", "beta", "r_squared"}
        assert all(np.isnan(v) for v in result.values())

    def test_not_estimable_is_distinguishable_from_a_real_zero_beta(self):
        """The property the zero sentinel destroyed: these two states must
        not produce the same number."""
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-02", periods=300, freq="B")
        mkt = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
        independent = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
        estimable = calculate_beta(independent, mkt)
        assert np.isfinite(estimable["beta"]), "a real fit stays a number"
        assert np.isnan(calculate_beta(pd.Series([0.01]), pd.Series([0.01]))["beta"])


class TestRollingBeta:
    def test_returns_dataframe_with_rolling_beta_column(
        self, sample_returns, benchmark_returns
    ):
        result = rolling_beta(sample_returns, benchmark_returns, window=60)
        assert "Rolling_Beta" in result.columns

    def test_output_length_matches_input(self, sample_returns, benchmark_returns):
        result = rolling_beta(sample_returns, benchmark_returns, window=60)
        assert len(result) == len(sample_returns)

    def test_nan_prefix_equals_window_minus_one(
        self, sample_returns, benchmark_returns
    ):
        window = 60
        result = rolling_beta(sample_returns, benchmark_returns, window=window)
        assert result["Rolling_Beta"].iloc[: window - 1].isna().all()

    def test_rolling_beta_of_identical_series_is_one(self, sample_returns):
        result = rolling_beta(sample_returns, sample_returns, window=30)
        valid = result["Rolling_Beta"].dropna()
        assert (valid - 1.0).abs().max() < 1e-6

    def test_rolling_beta_of_double_leverage_is_two(self, sample_returns):
        leveraged = sample_returns * 2
        result = rolling_beta(leveraged, sample_returns, window=30)
        valid = result["Rolling_Beta"].dropna()
        assert (valid - 2.0).abs().max() < 1e-6

    def test_rolling_window_does_not_use_future_data(
        self, sample_returns, benchmark_returns
    ):
        """Beta at bar t must only depend on bars [t-window+1 .. t].
        The cov/var rolling approach and OLS may differ slightly; use rel tolerance."""
        window = 30
        result = rolling_beta(sample_returns, benchmark_returns, window=window)
        idx_t = sample_returns.index[window]
        slice_asset = sample_returns.iloc[1 : window + 1]  # window bars ending at t
        slice_bench = benchmark_returns.iloc[1 : window + 1]
        manual = calculate_beta(slice_asset, slice_bench)
        rolling_val = float(result.loc[idx_t, "Rolling_Beta"])
        assert rolling_val == pytest.approx(manual["beta"], rel=0.05)

    def test_constant_benchmark_window_yields_nan_not_inf(self, sample_returns):
        """A window with zero benchmark variance (e.g. a constant
        benchmark) used to divide by zero -- must produce NaN for that
        window, not inf/-inf that could silently poison downstream math."""
        constant_benchmark = pd.Series(1.0, index=sample_returns.index)
        result = rolling_beta(sample_returns, constant_benchmark, window=30)
        valid = result["Rolling_Beta"].dropna()
        assert valid.empty
        assert not np.isinf(result["Rolling_Beta"]).any()
