"""Tests for regression and analysis functions: beta, alpha, R-squared."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.regression import calculate_beta, rolling_beta


class TestCalculateBeta:
    def test_returns_required_keys(self, sample_returns, benchmark_returns):
        result = calculate_beta(sample_returns, benchmark_returns)
        assert set(result.keys()) == {'alpha', 'beta', 'r_squared'}

    def test_beta_one_when_asset_equals_benchmark(self, sample_returns):
        """When asset and benchmark are identical, beta must be exactly 1."""
        result = calculate_beta(sample_returns, sample_returns)
        assert result['beta'] == pytest.approx(1.0, abs=1e-6)

    def test_alpha_near_zero_when_asset_equals_benchmark(self, sample_returns):
        result = calculate_beta(sample_returns, sample_returns)
        assert result['alpha'] == pytest.approx(0.0, abs=1e-10)

    def test_r_squared_one_when_perfectly_correlated(self, sample_returns):
        result = calculate_beta(sample_returns, sample_returns)
        assert result['r_squared'] == pytest.approx(1.0, abs=1e-6)

    def test_r_squared_bounded_0_to_1(self, sample_returns, benchmark_returns):
        result = calculate_beta(sample_returns, benchmark_returns)
        assert 0.0 <= result['r_squared'] <= 1.0

    def test_beta_two_for_double_leverage(self, sample_returns):
        """An asset with exactly 2x leverage should have beta = 2."""
        leveraged = sample_returns * 2
        result = calculate_beta(leveraged, sample_returns)
        assert result['beta'] == pytest.approx(2.0, abs=1e-6)

    def test_negative_beta_for_inverse_asset(self, sample_returns):
        inverse = -sample_returns
        result = calculate_beta(inverse, sample_returns)
        assert result['beta'] == pytest.approx(-1.0, abs=1e-6)

    def test_low_r_squared_for_uncorrelated_assets(self):
        """Uncorrelated random series should have R² close to 0."""
        np.random.seed(42)
        r1 = pd.Series(np.random.normal(0, 0.01, 500))
        r2 = pd.Series(np.random.normal(0, 0.01, 500))
        result = calculate_beta(r1, r2)
        assert result['r_squared'] < 0.10

    def test_index_alignment_handles_different_lengths(self, sample_returns):
        """calculate_beta should align on common index and not crash."""
        short_bench = sample_returns.iloc[:100]
        result = calculate_beta(sample_returns, short_bench)
        assert isinstance(result['beta'], float)

    def test_minimal_data_returns_zeros(self):
        """With only 1 data point, should return safe zero dict."""
        r1 = pd.Series([0.01])
        r2 = pd.Series([0.01])
        result = calculate_beta(r1, r2)
        assert result == {'alpha': 0.0, 'beta': 0.0, 'r_squared': 0.0}


class TestRollingBeta:
    def test_returns_dataframe_with_rolling_beta_column(self, sample_returns, benchmark_returns):
        result = rolling_beta(sample_returns, benchmark_returns, window=60)
        assert 'Rolling_Beta' in result.columns

    def test_output_length_matches_input(self, sample_returns, benchmark_returns):
        result = rolling_beta(sample_returns, benchmark_returns, window=60)
        assert len(result) == len(sample_returns)

    def test_nan_prefix_equals_window_minus_one(self, sample_returns, benchmark_returns):
        window = 60
        result = rolling_beta(sample_returns, benchmark_returns, window=window)
        assert result['Rolling_Beta'].iloc[:window - 1].isna().all()

    def test_rolling_beta_of_identical_series_is_one(self, sample_returns):
        result = rolling_beta(sample_returns, sample_returns, window=30)
        valid = result['Rolling_Beta'].dropna()
        assert (valid - 1.0).abs().max() < 1e-6

    def test_rolling_beta_of_double_leverage_is_two(self, sample_returns):
        leveraged = sample_returns * 2
        result = rolling_beta(leveraged, sample_returns, window=30)
        valid = result['Rolling_Beta'].dropna()
        assert (valid - 2.0).abs().max() < 1e-6

    def test_rolling_window_does_not_use_future_data(self, sample_returns, benchmark_returns):
        """Beta at bar t must only depend on bars [t-window+1 .. t].
        The cov/var rolling approach and OLS may differ slightly; use rel tolerance."""
        window = 30
        result = rolling_beta(sample_returns, benchmark_returns, window=window)
        idx_t = sample_returns.index[window]
        slice_asset = sample_returns.iloc[1:window + 1]   # window bars ending at t
        slice_bench = benchmark_returns.iloc[1:window + 1]
        manual = calculate_beta(slice_asset, slice_bench)
        rolling_val = float(result.loc[idx_t, 'Rolling_Beta'])
        assert rolling_val == pytest.approx(manual['beta'], rel=0.05)
