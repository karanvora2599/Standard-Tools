"""Tests for PCA on returns: pca_returns and factor_contributions."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.pca import factor_contributions, pca_returns


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def one_factor_data():
    """
    4-asset universe driven almost entirely by one common factor.
    PC1 should explain > 95% of variance.
    """
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    factor = np.random.normal(0, 1, n)
    returns = pd.DataFrame({
        "A": factor * 1.0 + np.random.normal(0, 0.1, n),
        "B": factor * 0.8 + np.random.normal(0, 0.1, n),
        "C": factor * 1.2 + np.random.normal(0, 0.1, n),
        "D": factor * 0.9 + np.random.normal(0, 0.1, n),
    }, index=dates)
    return returns


@pytest.fixture(scope="module")
def two_factor_data():
    """
    6-asset universe with two independent factors (3 assets each).
    PC1 and PC2 should together explain > 90% of variance.
    """
    np.random.seed(7)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    f1 = np.random.normal(0, 1, n)
    f2 = np.random.normal(0, 1, n)
    returns = pd.DataFrame({
        "A1": f1 + np.random.normal(0, 0.15, n),
        "A2": f1 * 0.9 + np.random.normal(0, 0.15, n),
        "A3": f1 * 1.1 + np.random.normal(0, 0.15, n),
        "B1": f2 + np.random.normal(0, 0.15, n),
        "B2": f2 * 0.8 + np.random.normal(0, 0.15, n),
        "B3": f2 * 1.2 + np.random.normal(0, 0.15, n),
    }, index=dates)
    return returns


@pytest.fixture(scope="module")
def pca_one_factor(one_factor_data):
    return pca_returns(one_factor_data)


@pytest.fixture(scope="module")
def pca_two_factor(two_factor_data):
    return pca_returns(two_factor_data)


# ── pca_returns — output structure ─────────────────────────────────────────────

class TestPcaReturnsKeys:
    def test_returns_required_keys(self, pca_one_factor):
        assert set(pca_one_factor.keys()) == {
            "explained_variance_ratio",
            "cumulative_variance_ratio",
            "loadings",
            "factor_returns",
            "n_components",
            "n_obs",
        }

    def test_evr_is_series(self, pca_one_factor):
        assert isinstance(pca_one_factor["explained_variance_ratio"], pd.Series)

    def test_loadings_is_dataframe(self, pca_one_factor):
        assert isinstance(pca_one_factor["loadings"], pd.DataFrame)

    def test_factor_returns_is_dataframe(self, pca_one_factor):
        assert isinstance(pca_one_factor["factor_returns"], pd.DataFrame)


class TestPcaReturnsShapes:
    def test_loadings_shape(self, one_factor_data, pca_one_factor):
        n_assets = len(one_factor_data.columns)
        n_comp = pca_one_factor["n_components"]
        assert pca_one_factor["loadings"].shape == (n_assets, n_comp)

    def test_factor_returns_shape(self, one_factor_data, pca_one_factor):
        n_comp = pca_one_factor["n_components"]
        assert pca_one_factor["factor_returns"].shape == (len(one_factor_data), n_comp)

    def test_n_components_limits_output(self, one_factor_data):
        result = pca_returns(one_factor_data, n_components=2)
        assert result["n_components"] == 2
        assert result["loadings"].shape[1] == 2
        assert result["factor_returns"].shape[1] == 2
        assert len(result["explained_variance_ratio"]) == 2

    def test_columns_named_pc1_pc2(self, pca_one_factor):
        assert list(pca_one_factor["loadings"].columns) == ["PC1", "PC2", "PC3", "PC4"]
        assert list(pca_one_factor["factor_returns"].columns) == ["PC1", "PC2", "PC3", "PC4"]

    def test_loadings_index_matches_asset_tickers(self, one_factor_data, pca_one_factor):
        assert list(pca_one_factor["loadings"].index) == list(one_factor_data.columns)

    def test_factor_returns_index_matches_dates(self, one_factor_data, pca_one_factor):
        assert list(pca_one_factor["factor_returns"].index) == list(one_factor_data.index)

    def test_n_obs_matches_non_nan_rows(self, one_factor_data, pca_one_factor):
        assert pca_one_factor["n_obs"] == len(one_factor_data.dropna())


class TestPcaReturnsVariance:
    def test_evr_sums_to_one(self, pca_one_factor):
        """Full decomposition must account for all variance."""
        assert pca_one_factor["explained_variance_ratio"].sum() == pytest.approx(1.0, abs=1e-9)

    def test_cumvar_is_monotone(self, pca_one_factor):
        cumvar = pca_one_factor["cumulative_variance_ratio"]
        assert (cumvar.diff().dropna() >= 0).all()

    def test_cumvar_ends_at_one(self, pca_one_factor):
        assert pca_one_factor["cumulative_variance_ratio"].iloc[-1] == pytest.approx(1.0, abs=1e-9)

    def test_each_evr_bounded_0_to_1(self, pca_one_factor):
        evr = pca_one_factor["explained_variance_ratio"]
        assert (evr >= 0).all()
        assert (evr <= 1).all()

    def test_pcs_in_decreasing_evr_order(self, pca_one_factor):
        evr = pca_one_factor["explained_variance_ratio"].values
        assert (np.diff(evr) <= 0).all()

    def test_one_factor_pc1_dominates(self, pca_one_factor):
        """One-factor data: PC1 should explain > 95% of variance."""
        evr = pca_one_factor["explained_variance_ratio"]
        assert evr["PC1"] > 0.95

    def test_two_factor_first_two_pcs_dominate(self, pca_two_factor):
        """Two-factor data: PC1+PC2 should explain > 85% of variance."""
        cumvar = pca_two_factor["cumulative_variance_ratio"]
        assert cumvar["PC2"] > 0.85


class TestPcaReturnsGeometry:
    def test_loading_columns_have_unit_norm(self, pca_one_factor):
        """Each eigenvector must have unit L2 norm."""
        loadings = pca_one_factor["loadings"].to_numpy()
        norms = np.linalg.norm(loadings, axis=0)
        np.testing.assert_allclose(norms, 1.0, atol=1e-9)

    def test_factor_returns_are_pairwise_uncorrelated(self, pca_one_factor):
        """Principal components must be orthogonal."""
        F = pca_one_factor["factor_returns"].to_numpy()
        corr = np.corrcoef(F.T)
        off_diag = corr - np.eye(corr.shape[0])
        assert np.abs(off_diag).max() < 1e-9

    def test_dominant_loadings_are_positive(self, pca_one_factor):
        """Sign convention: largest-magnitude loading per PC should be positive."""
        loadings = pca_one_factor["loadings"].to_numpy()
        for col in range(loadings.shape[1]):
            max_idx = np.argmax(np.abs(loadings[:, col]))
            assert loadings[max_idx, col] > 0

    def test_one_factor_all_loadings_same_sign_on_pc1(self, pca_one_factor):
        """When all assets share one factor, all PC1 loadings should be positive."""
        pc1_loadings = pca_one_factor["loadings"]["PC1"]
        assert (pc1_loadings > 0).all()


class TestPcaReturnsEdgeCases:
    def test_n_components_exceeds_assets_capped(self, one_factor_data):
        """Asking for more PCs than assets should silently cap at n_assets."""
        result = pca_returns(one_factor_data, n_components=100)
        assert result["n_components"] == len(one_factor_data.columns)

    def test_nan_rows_dropped(self, one_factor_data):
        data_with_nan = one_factor_data.copy()
        data_with_nan.iloc[0, 0] = np.nan
        result = pca_returns(data_with_nan)
        assert result["n_obs"] == len(one_factor_data) - 1

    def test_raises_on_insufficient_data(self):
        tiny = pd.DataFrame({"A": [0.01], "B": [0.02]})
        with pytest.raises(ValueError):
            pca_returns(tiny)

    def test_no_standardize_runs_without_error(self, one_factor_data):
        result = pca_returns(one_factor_data, standardize=False)
        assert result["n_components"] == len(one_factor_data.columns)


# ── factor_contributions ───────────────────────────────────────────────────────

class TestFactorContributions:
    def test_returns_dataframe(self, one_factor_data):
        df = factor_contributions(one_factor_data, n_components=3)
        assert isinstance(df, pd.DataFrame)

    def test_index_is_asset_tickers(self, one_factor_data):
        df = factor_contributions(one_factor_data, n_components=3)
        assert set(df.index) == set(one_factor_data.columns)

    def test_columns_are_pc_names(self, one_factor_data):
        n = 3
        df = factor_contributions(one_factor_data, n_components=n)
        assert list(df.columns) == [f"PC{i + 1}" for i in range(n)]

    def test_contributions_nonnegative(self, one_factor_data):
        df = factor_contributions(one_factor_data, n_components=3)
        assert (df >= -1e-10).all(axis=None)

    def test_row_sum_bounded_by_one(self, one_factor_data):
        """Total R² across all PCs cannot exceed 1."""
        df = factor_contributions(one_factor_data, n_components=4)
        row_sums = df.sum(axis=1)
        assert (row_sums <= 1.0 + 1e-9).all()

    def test_one_factor_pc1_dominates_all_assets(self, one_factor_data):
        """In one-factor data, PC1 contribution should be the largest for every asset."""
        df = factor_contributions(one_factor_data, n_components=4)
        for ticker in df.index:
            assert df.loc[ticker, "PC1"] == df.loc[ticker].max()

    def test_two_factor_data_pc1_pc2_dominant(self, two_factor_data):
        """PC1+PC2 combined contribution > 0.7 for every asset in 2-factor data."""
        df = factor_contributions(two_factor_data, n_components=3)
        pc1_pc2 = df["PC1"] + df["PC2"]
        assert (pc1_pc2 > 0.70).all()

    def test_shape_matches_assets_and_components(self, two_factor_data):
        n = 3
        df = factor_contributions(two_factor_data, n_components=n)
        assert df.shape == (len(two_factor_data.columns), n)
