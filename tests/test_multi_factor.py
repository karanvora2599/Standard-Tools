"""Tests for multi-factor OLS regression and rolling factor loadings."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.multi_factor import (
    multi_factor_regression,
    rolling_factor_loadings,
)
from standard_quant_tools.error import ValidationError

# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def two_factor_data():
    """
    Synthetic asset with exact two-factor structure plus noise.
    y = 0.001 + 1.2*f1 + 0.4*f2 + noise
    True: alpha≈0.001, loading_f1≈1.2, loading_f2≈0.4
    """
    np.random.seed(0)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    f1 = pd.Series(np.random.normal(0, 0.01, n), index=dates, name="mkt")
    f2 = pd.Series(np.random.normal(0, 0.005, n), index=dates, name="smb")
    noise = np.random.normal(0, 0.003, n)
    asset = pd.Series(0.001 + 1.2 * f1.values + 0.4 * f2.values + noise, index=dates)
    factors = pd.DataFrame({"mkt": f1, "smb": f2})
    return asset, factors


@pytest.fixture(scope="module")
def single_factor_data(sample_returns, benchmark_returns):
    """Wrap conftest fixtures into (asset, factors) form for single-factor tests."""
    factors = pd.DataFrame({"bench": benchmark_returns})
    return sample_returns, factors


# ── multi_factor_regression ────────────────────────────────────────────────────


class TestMultiFactorRegressionKeys:
    def test_returns_all_required_keys(self, two_factor_data):
        asset, factors = two_factor_data
        result = multi_factor_regression(asset, factors)
        assert set(result.keys()) == {
            "alpha",
            "loadings",
            "t_stats",
            "p_values",
            "r_squared",
            "adj_r_squared",
            "n_obs",
        }

    def test_loadings_keys_match_factor_columns(self, two_factor_data):
        asset, factors = two_factor_data
        result = multi_factor_regression(asset, factors)
        assert set(result["loadings"].keys()) == {"mkt", "smb"}

    def test_t_stats_keys_include_alpha(self, two_factor_data):
        asset, factors = two_factor_data
        result = multi_factor_regression(asset, factors)
        assert "alpha" in result["t_stats"]
        assert "mkt" in result["t_stats"]
        assert "smb" in result["t_stats"]

    def test_p_values_keys_match_t_stats(self, two_factor_data):
        asset, factors = two_factor_data
        result = multi_factor_regression(asset, factors)
        assert result["p_values"].keys() == result["t_stats"].keys()


class TestMultiFactorRegressionValues:
    def test_known_loadings_recovered(self, two_factor_data):
        """With 500 obs and low noise the OLS should recover true loadings closely."""
        asset, factors = two_factor_data
        result = multi_factor_regression(asset, factors)
        assert result["loadings"]["mkt"] == pytest.approx(1.2, abs=0.05)
        assert result["loadings"]["smb"] == pytest.approx(0.4, abs=0.05)

    def test_perfect_fit_r_squared_is_one(self):
        """y = exact linear combination → R² must be 1."""
        np.random.seed(1)
        n = 200
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        f1 = np.random.normal(0, 0.01, n)
        f2 = np.random.normal(0, 0.01, n)
        y = 0.0005 + 2.0 * f1 - 0.5 * f2
        asset = pd.Series(y, index=dates)
        factors = pd.DataFrame({"a": f1, "b": f2}, index=dates)
        result = multi_factor_regression(asset, factors)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_r_squared_bounded_zero_to_one(self, two_factor_data):
        asset, factors = two_factor_data
        result = multi_factor_regression(asset, factors)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_adj_r_squared_leq_r_squared(self, two_factor_data):
        asset, factors = two_factor_data
        result = multi_factor_regression(asset, factors)
        assert result["adj_r_squared"] <= result["r_squared"]

    def test_p_values_bounded_zero_to_one(self, two_factor_data):
        asset, factors = two_factor_data
        result = multi_factor_regression(asset, factors)
        for name, pv in result["p_values"].items():
            assert 0.0 <= pv <= 1.0, f"p_value[{name}] = {pv} out of range"

    def test_significant_factor_has_low_p_value(self, two_factor_data):
        """mkt loading ≈ 1.2 on 500 obs should be highly significant (p < 0.01)."""
        asset, factors = two_factor_data
        result = multi_factor_regression(asset, factors)
        assert result["p_values"]["mkt"] < 0.01

    def test_n_obs_matches_overlap(self, two_factor_data):
        asset, factors = two_factor_data
        result = multi_factor_regression(asset, factors)
        assert result["n_obs"] == len(asset)

    def test_single_factor_loadings_match_calculate_beta(self, single_factor_data):
        """
        Single-factor multi_factor_regression must give the same beta as
        calculate_beta (both use OLS, so they should agree to floating-point precision).
        """
        from standard_quant_tools.analysis.regression import calculate_beta

        asset, factors = single_factor_data
        mfr = multi_factor_regression(asset, factors)
        cb = calculate_beta(asset, factors["bench"])
        assert mfr["loadings"]["bench"] == pytest.approx(cb["beta"], abs=1e-8)
        assert mfr["r_squared"] == pytest.approx(cb["r_squared"], abs=1e-6)

    def test_uncorrelated_factors_low_r_squared(self):
        """Noise asset regressed on unrelated factors → R² near 0."""
        np.random.seed(99)
        n = 300
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        asset = pd.Series(np.random.normal(0, 0.01, n), index=dates)
        factors = pd.DataFrame(
            {
                "f1": np.random.normal(0, 0.01, n),
                "f2": np.random.normal(0, 0.01, n),
            },
            index=dates,
        )
        result = multi_factor_regression(asset, factors)
        assert result["r_squared"] < 0.05

    def test_index_alignment_on_partial_overlap(self, two_factor_data):
        """Should run cleanly even when factor index is longer than asset index."""
        asset, factors = two_factor_data
        extended_factors = pd.concat(
            [
                factors,
                pd.DataFrame(
                    {"mkt": [0.0], "smb": [0.0]},
                    index=pd.date_range("2100-01-01", periods=1, freq="B"),
                ),
            ]
        )
        result = multi_factor_regression(asset, extended_factors)
        assert result["n_obs"] == len(asset)

    def test_insufficient_data_returns_nan(self):
        """Fewer observations than parameters should return NaN for all metrics."""
        dates = pd.date_range("2022-01-01", periods=3, freq="B")
        asset = pd.Series([0.01, 0.02, -0.01], index=dates)
        factors = pd.DataFrame(
            {
                "f1": [0.005, 0.01, -0.005],
                "f2": [0.003, -0.002, 0.007],
                "f3": [0.001, 0.002, 0.003],
            },
            index=dates,
        )
        # n=3, k=4 (3 factors + intercept) → n < k+1
        result = multi_factor_regression(asset, factors)
        assert np.isnan(result["r_squared"])
        assert np.isnan(result["alpha"])

    def test_three_factor_model_recovers_all_loadings(self):
        """Verify the solver handles 3 factors correctly."""
        np.random.seed(7)
        n = 400
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        f1 = np.random.normal(0, 0.01, n)
        f2 = np.random.normal(0, 0.008, n)
        f3 = np.random.normal(0, 0.006, n)
        y = 0.0002 + 1.0 * f1 + 0.5 * f2 - 0.3 * f3 + np.random.normal(0, 0.002, n)
        asset = pd.Series(y, index=dates)
        factors = pd.DataFrame({"mkt": f1, "smb": f2, "hml": f3}, index=dates)
        result = multi_factor_regression(asset, factors)
        assert result["loadings"]["mkt"] == pytest.approx(1.0, abs=0.08)
        assert result["loadings"]["smb"] == pytest.approx(0.5, abs=0.08)
        assert result["loadings"]["hml"] == pytest.approx(-0.3, abs=0.08)


# ── rolling_factor_loadings ────────────────────────────────────────────────────


class TestRollingFactorLoadings:
    def test_returns_dataframe(self, two_factor_data):
        asset, factors = two_factor_data
        result = rolling_factor_loadings(asset, factors, window=60)
        assert isinstance(result, pd.DataFrame)

    def test_columns_include_alpha_and_factors(self, two_factor_data):
        asset, factors = two_factor_data
        result = rolling_factor_loadings(asset, factors, window=60)
        assert "alpha" in result.columns
        assert "mkt" in result.columns
        assert "smb" in result.columns

    def test_output_length_matches_input(self, two_factor_data):
        asset, factors = two_factor_data
        result = rolling_factor_loadings(asset, factors, window=60)
        assert len(result) == len(asset)

    def test_nan_prefix_length(self, two_factor_data):
        asset, factors = two_factor_data
        window = 60
        result = rolling_factor_loadings(asset, factors, window=window)
        assert result.iloc[: window - 1].isna().all(axis=None)

    def test_no_nan_after_warmup(self, two_factor_data):
        asset, factors = two_factor_data
        window = 60
        result = rolling_factor_loadings(asset, factors, window=window)
        assert not result.iloc[window - 1 :].isna().any(axis=None)

    def test_nan_in_asset_returns_raises(self, two_factor_data):
        asset, factors = two_factor_data
        bad = asset.copy()
        bad.iloc[30] = np.nan
        with pytest.raises(ValidationError, match="non-finite"):
            rolling_factor_loadings(bad, factors, window=60)

    def test_stable_loadings_converge_to_true_values(self, two_factor_data):
        """
        When the DGP is stationary, the rolling mean of the loadings should
        be close to the full-sample estimates.
        """
        asset, factors = two_factor_data
        window = 60
        rolling = rolling_factor_loadings(asset, factors, window=window)
        full = multi_factor_regression(asset, factors)

        mean_mkt = rolling["mkt"].dropna().mean()
        mean_smb = rolling["smb"].dropna().mean()
        assert mean_mkt == pytest.approx(full["loadings"]["mkt"], abs=0.05)
        assert mean_smb == pytest.approx(full["loadings"]["smb"], abs=0.05)

    def test_underdetermined_window_is_all_nan(self):
        """
        window < k+2 leaves fewer observations than the k+1 coefficients being
        estimated, so every window is underdetermined and the whole result is
        NaN — on BOTH backends.

        This assertion used to check only `result.shape`, which is why it
        passed while the two paths silently disagreed: the C++ kernel bailed
        to all-NaN for window < k+2, while numpy.linalg.lstsq returned its
        minimum-norm solution. Assert the values, not just the shape, so that
        divergence can never hide here again.
        """
        np.random.seed(5)
        n = 50
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        asset = pd.Series(np.random.normal(0, 0.01, n), index=dates)
        factors = pd.DataFrame({"f": np.random.normal(0, 0.01, n)}, index=dates)
        # k=1, so k+2 == 3: windows of 1 and 2 are both underdetermined.
        for window in (1, 2):
            result = rolling_factor_loadings(asset, factors, window=window)
            assert result.shape == (n, 2)
            assert result.isna().all().all(), f"window={window} must be all-NaN"

    def test_smallest_determined_window_produces_values(self):
        """window == k+2 is the smallest determined window — must NOT be NaN."""
        np.random.seed(6)
        n = 50
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        asset = pd.Series(np.random.normal(0, 0.01, n), index=dates)
        factors = pd.DataFrame({"f": np.random.normal(0, 0.01, n)}, index=dates)
        result = rolling_factor_loadings(asset, factors, window=3)
        assert result.shape == (n, 2)
        assert not result.iloc[2:].isna().all().all()
