"""Tests for correlation & diversification analytics."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.correlation import (
    diversification_ratio,
    pairwise_correlation_summary,
)
from standard_quant_tools.error import ValidationError


class TestDiversificationRatio:
    def test_perfectly_correlated_assets_yield_ratio_one(self):
        """Identical return streams -> zero diversification benefit -> DR == 1.0."""
        np.random.seed(0)
        n = 200
        base = np.random.normal(0, 0.01, n)
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        returns_df = pd.DataFrame({"A": base, "B": base, "C": base}, index=dates)
        dr = diversification_ratio(returns_df)
        assert dr == pytest.approx(1.0, abs=1e-8)

    def test_uncorrelated_equal_vol_assets_hand_computed(self):
        """
        n uncorrelated, equal-volatility assets, equal-weighted: DR = sqrt(n)
        exactly (weighted_avg_vol = sigma; portfolio_vol = sigma/sqrt(n)).
        """
        rng = np.random.default_rng(1)
        n_assets = 4
        n_obs = 5000  # large sample so realized correlation/vol ~ population values
        dates = pd.date_range("2015-01-01", periods=n_obs, freq="B")
        data = {f"T{i}": rng.normal(0, 0.02, n_obs) for i in range(n_assets)}
        returns_df = pd.DataFrame(data, index=dates)
        dr = diversification_ratio(returns_df)
        assert dr == pytest.approx(np.sqrt(n_assets), rel=0.05)

    def test_equal_weight_default_matches_explicit_equal_weights(self):
        rng = np.random.default_rng(2)
        n = 300
        dates = pd.date_range("2021-01-01", periods=n, freq="B")
        returns_df = pd.DataFrame(
            {"A": rng.normal(0, 0.01, n), "B": rng.normal(0, 0.015, n)}, index=dates
        )
        default_dr = diversification_ratio(returns_df)
        explicit_dr = diversification_ratio(returns_df, weights=[0.5, 0.5])
        assert default_dr == pytest.approx(explicit_dr)

    def test_single_asset_raises(self):
        dates = pd.date_range("2021-01-01", periods=50, freq="B")
        returns_df = pd.DataFrame({"A": np.random.normal(0, 0.01, 50)}, index=dates)
        with pytest.raises(ValidationError, match="at least 2 assets"):
            diversification_ratio(returns_df)

    def test_mismatched_weights_length_raises(self):
        dates = pd.date_range("2021-01-01", periods=50, freq="B")
        returns_df = pd.DataFrame(
            {"A": np.random.normal(0, 0.01, 50), "B": np.random.normal(0, 0.01, 50)},
            index=dates,
        )
        with pytest.raises(ValidationError, match="length"):
            diversification_ratio(returns_df, weights=[0.3, 0.3, 0.4])

    def test_weights_not_summing_to_one_raises(self):
        dates = pd.date_range("2021-01-01", periods=50, freq="B")
        returns_df = pd.DataFrame(
            {"A": np.random.normal(0, 0.01, 50), "B": np.random.normal(0, 0.01, 50)},
            index=dates,
        )
        with pytest.raises(ValidationError, match="sum to 1.0"):
            diversification_ratio(returns_df, weights=[0.5, 0.6])


class TestPairwiseCorrelationSummary:
    def test_identical_pair_has_correlation_one(self):
        np.random.seed(3)
        n = 100
        base = np.random.normal(0, 0.01, n)
        independent = np.random.normal(0, 0.01, n)
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        returns_df = pd.DataFrame({"A": base, "B": base, "C": independent}, index=dates)
        summary = pairwise_correlation_summary(returns_df)
        assert summary["highest_correlated_pair"]["correlation"] == pytest.approx(
            1.0, abs=1e-8
        )
        assert {
            summary["highest_correlated_pair"]["a"],
            summary["highest_correlated_pair"]["b"],
        } == {"A", "B"}

    def test_correlation_matrix_is_symmetric_with_unit_diagonal(self):
        rng = np.random.default_rng(4)
        n = 100
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        returns_df = pd.DataFrame(
            {
                "A": rng.normal(0, 0.01, n),
                "B": rng.normal(0, 0.01, n),
                "C": rng.normal(0, 0.01, n),
            },
            index=dates,
        )
        summary = pairwise_correlation_summary(returns_df)
        corr = summary["correlation_matrix"]
        np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0, atol=1e-10)
        np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T, atol=1e-10)

    def test_avg_pairwise_correlation_matches_manual_mean(self):
        rng = np.random.default_rng(5)
        n = 150
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        returns_df = pd.DataFrame(
            {
                "A": rng.normal(0, 0.01, n),
                "B": rng.normal(0, 0.01, n),
                "C": rng.normal(0, 0.01, n),
            },
            index=dates,
        )
        summary = pairwise_correlation_summary(returns_df)
        corr = summary["correlation_matrix"]
        n_assets = 3
        manual = np.mean(
            [corr.iloc[i, j] for i in range(n_assets) for j in range(i + 1, n_assets)]
        )
        assert summary["avg_pairwise_correlation"] == pytest.approx(manual)

    def test_highest_and_lowest_pairs_are_distinct_when_three_assets(self):
        np.random.seed(6)
        n = 200
        base = np.random.normal(0, 0.01, n)
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        returns_df = pd.DataFrame({"A": base, "B": base, "C": -base}, index=dates)
        summary = pairwise_correlation_summary(returns_df)
        assert summary["highest_correlated_pair"]["correlation"] == pytest.approx(
            1.0, abs=1e-8
        )
        assert summary["lowest_correlated_pair"]["correlation"] == pytest.approx(
            -1.0, abs=1e-8
        )

    def test_single_asset_raises(self):
        dates = pd.date_range("2021-01-01", periods=50, freq="B")
        returns_df = pd.DataFrame({"A": np.random.normal(0, 0.01, 50)}, index=dates)
        with pytest.raises(ValidationError, match="at least 2 assets"):
            pairwise_correlation_summary(returns_df)
