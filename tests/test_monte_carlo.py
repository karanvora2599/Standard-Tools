"""Tests for Monte Carlo forward-path simulation (block-bootstrap resampling)."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.monte_carlo import simulate_forward_paths
from standard_quant_tools.error import ValidationError


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(7)
    return pd.Series(rng.normal(0.0005, 0.01, 300))


class TestSimulateForwardPaths:
    def test_output_keys_present(self, sample_returns):
        result = simulate_forward_paths(
            sample_returns, horizon_days=30, n_simulations=200, seed=1
        )
        expected_keys = {
            "terminal_median",
            "terminal_p5",
            "terminal_p95",
            "prob_loss",
            "terminal_var_95",
            "terminal_cvar_95",
            "equity_band_p5",
            "equity_band_p50",
            "equity_band_p95",
        }
        assert expected_keys.issubset(result.keys())

    def test_band_lengths_match_horizon(self, sample_returns):
        result = simulate_forward_paths(
            sample_returns, horizon_days=45, n_simulations=200, seed=1
        )
        assert len(result["equity_band_p5"]) == 45
        assert len(result["equity_band_p50"]) == 45
        assert len(result["equity_band_p95"]) == 45

    def test_bands_are_ordered_every_day(self, sample_returns):
        result = simulate_forward_paths(
            sample_returns, horizon_days=30, n_simulations=500, seed=2
        )
        p5 = np.array(result["equity_band_p5"])
        p50 = np.array(result["equity_band_p50"])
        p95 = np.array(result["equity_band_p95"])
        assert (p5 <= p50).all()
        assert (p50 <= p95).all()

    def test_reproducible_with_same_seed(self, sample_returns):
        r1 = simulate_forward_paths(
            sample_returns, horizon_days=30, n_simulations=200, seed=99
        )
        r2 = simulate_forward_paths(
            sample_returns, horizon_days=30, n_simulations=200, seed=99
        )
        assert r1["terminal_median"] == r2["terminal_median"]
        assert r1["equity_band_p50"] == r2["equity_band_p50"]

    def test_different_seeds_give_different_results(self, sample_returns):
        r1 = simulate_forward_paths(
            sample_returns, horizon_days=30, n_simulations=200, seed=1
        )
        r2 = simulate_forward_paths(
            sample_returns, horizon_days=30, n_simulations=200, seed=2
        )
        assert r1["terminal_median"] != r2["terminal_median"]

    def test_zero_return_series_yields_flat_paths(self):
        """All-zero returns -> every resampled block is zero -> every path
        stays exactly at initial_capital, so all percentile bands collapse
        to the same value and prob_loss is 0 (never strictly below)."""
        flat = pd.Series(np.zeros(100))
        result = simulate_forward_paths(
            flat, horizon_days=20, n_simulations=100, initial_capital=10_000.0, seed=3
        )
        np.testing.assert_allclose(result["equity_band_p5"], 10_000.0)
        np.testing.assert_allclose(result["equity_band_p50"], 10_000.0)
        np.testing.assert_allclose(result["equity_band_p95"], 10_000.0)
        assert result["prob_loss"] == pytest.approx(0.0)

    def test_positive_drift_median_terminal_above_initial(self):
        """Strong, consistent positive daily return -> terminal median must
        exceed initial_capital (deterministic compounding, no noise)."""
        strong_positive = pd.Series(np.full(100, 0.01))
        result = simulate_forward_paths(
            strong_positive,
            horizon_days=50,
            n_simulations=50,
            initial_capital=10_000.0,
            seed=4,
        )
        assert result["terminal_median"] > 10_000.0
        assert result["prob_loss"] == pytest.approx(0.0)

    def test_negative_drift_prob_loss_is_one(self):
        strong_negative = pd.Series(np.full(100, -0.01))
        result = simulate_forward_paths(
            strong_negative,
            horizon_days=50,
            n_simulations=50,
            initial_capital=10_000.0,
            seed=5,
        )
        assert result["terminal_median"] < 10_000.0
        assert result["prob_loss"] == pytest.approx(1.0)

    def test_initial_capital_scales_terminal_values_linearly(self, sample_returns):
        r1 = simulate_forward_paths(
            sample_returns,
            horizon_days=30,
            n_simulations=200,
            initial_capital=10_000.0,
            seed=11,
        )
        r2 = simulate_forward_paths(
            sample_returns,
            horizon_days=30,
            n_simulations=200,
            initial_capital=20_000.0,
            seed=11,
        )
        assert r2["terminal_median"] == pytest.approx(
            r1["terminal_median"] * 2, rel=1e-9
        )

    @pytest.mark.parametrize("bad_horizon", [0, -5])
    def test_invalid_horizon_days_raises(self, sample_returns, bad_horizon):
        with pytest.raises(ValidationError, match="horizon_days"):
            simulate_forward_paths(sample_returns, horizon_days=bad_horizon)

    @pytest.mark.parametrize("bad_n", [0, -10])
    def test_invalid_n_simulations_raises(self, sample_returns, bad_n):
        with pytest.raises(ValidationError, match="n_simulations"):
            simulate_forward_paths(sample_returns, horizon_days=30, n_simulations=bad_n)

    def test_invalid_initial_capital_raises(self, sample_returns):
        with pytest.raises(ValidationError, match="initial_capital"):
            simulate_forward_paths(sample_returns, horizon_days=30, initial_capital=0.0)

    def test_block_size_larger_than_series_raises(self, sample_returns):
        with pytest.raises(ValidationError, match="block_size"):
            simulate_forward_paths(sample_returns, horizon_days=30, block_size=10_000)

    def test_empty_returns_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            simulate_forward_paths(pd.Series([], dtype=float), horizon_days=30)

    def test_nan_in_returns_raises(self, sample_returns):
        """NaN/Inf input contract: simulate_forward_paths_into validates
        initial_capital's finiteness but never checked `values` itself --
        a single NaN/Inf in the historical returns being resampled from
        poisons `equity` permanently for every path/bar downstream of
        when it's sampled, with no explicit check anywhere in the native
        kernel."""
        bad = sample_returns.copy()
        bad.iloc[len(bad) // 2] = np.nan
        with pytest.raises(ValidationError, match="non-finite"):
            simulate_forward_paths(bad, horizon_days=30)

    def test_inf_in_returns_raises(self, sample_returns):
        bad = sample_returns.copy()
        bad.iloc[3] = np.inf
        with pytest.raises(ValidationError, match="non-finite"):
            simulate_forward_paths(bad, horizon_days=30)
