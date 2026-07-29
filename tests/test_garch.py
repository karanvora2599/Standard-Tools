"""Tests for GARCH(1,1) conditional volatility: garch_volatility_forecast."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.garch import (
    HAS_SCIPY,
    _garch11_variance_recursion,
    garch_volatility_forecast,
)
from standard_quant_tools.error import ValidationError

pytestmark = pytest.mark.skipif(
    not HAS_SCIPY, reason="GARCH MLE fitting requires scipy"
)


def _simulate_garch11(n, omega, alpha, beta, seed=42):
    rng = np.random.default_rng(seed)
    sigma2 = np.empty(n)
    resid = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    resid[0] = rng.standard_normal() * np.sqrt(sigma2[0])
    for t in range(1, n):
        sigma2[t] = omega + alpha * resid[t - 1] ** 2 + beta * sigma2[t - 1]
        resid[t] = rng.standard_normal() * np.sqrt(sigma2[t])
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.Series(resid, index=dates)


# ── _garch11_variance_recursion — hand-computed ───────────────────────────────


class TestVarianceRecursion:
    def test_matches_hand_computed_values(self):
        resid_sq = np.array([1.0, 4.0, 9.0])
        sigma2 = _garch11_variance_recursion(resid_sq, omega=0.1, alpha=0.2, beta=0.5)
        expected0 = resid_sq.mean()
        expected1 = 0.1 + 0.2 * 1.0 + 0.5 * expected0
        expected2 = 0.1 + 0.2 * 4.0 + 0.5 * expected1
        assert sigma2[0] == pytest.approx(expected0)
        assert sigma2[1] == pytest.approx(expected1)
        assert sigma2[2] == pytest.approx(expected2)

    def test_never_returns_non_positive_variance(self):
        # omega=0, alpha=0, beta=0 would otherwise decay toward/through 0.
        resid_sq = np.zeros(10)
        sigma2 = _garch11_variance_recursion(resid_sq, omega=0.0, alpha=0.0, beta=0.0)
        assert np.all(sigma2 > 0)


# ── garch_volatility_forecast — parameter recovery on simulated data ─────────


class TestParameterRecovery:
    def test_recovers_true_parameters_on_simulated_process(self):
        true_omega, true_alpha, true_beta = 1e-6, 0.08, 0.90
        returns = _simulate_garch11(50_000, true_omega, true_alpha, true_beta)
        result = garch_volatility_forecast(returns, forecast_horizon=5)
        assert result["converged"] is True
        assert result["alpha"] == pytest.approx(true_alpha, abs=0.03)
        assert result["beta"] == pytest.approx(true_beta, abs=0.05)
        assert result["persistence"] < 1.0

    def test_forecast_decays_toward_long_run_variance(self):
        true_omega, true_alpha, true_beta = 1e-6, 0.08, 0.90
        returns = _simulate_garch11(20_000, true_omega, true_alpha, true_beta)
        result = garch_volatility_forecast(returns, forecast_horizon=252)
        forecast = result["forecast_annualized_vol"]
        long_run = result["long_run_annualized_vol"]
        # Far-horizon forecast should be close to the long-run level;
        # the very first step should be closer to current vol than long-run
        # is (unless they happen to already coincide).
        assert abs(forecast[-1] - long_run) < abs(forecast[0] - long_run) + 1e-9


# ── garch_volatility_forecast — output structure ──────────────────────────────


class TestOutputStructure:
    @pytest.fixture(scope="class")
    def result(self):
        returns = _simulate_garch11(2000, 1e-6, 0.08, 0.90)
        return garch_volatility_forecast(returns, forecast_horizon=10)

    def test_has_required_keys(self, result):
        expected = {
            "omega",
            "alpha",
            "beta",
            "persistence",
            "converged",
            "log_likelihood",
            "aic",
            "bic",
            "n_obs",
            "current_annualized_vol",
            "long_run_annualized_vol",
            "forecast_annualized_vol",
        }
        assert expected <= set(result.keys())

    def test_forecast_length_matches_horizon(self, result):
        assert len(result["forecast_annualized_vol"]) == 10

    def test_persistence_equals_alpha_plus_beta(self, result):
        assert result["persistence"] == pytest.approx(
            result["alpha"] + result["beta"]
        )

    def test_vols_are_non_negative(self, result):
        assert result["current_annualized_vol"] >= 0
        assert result["long_run_annualized_vol"] >= 0
        assert all(v >= 0 for v in result["forecast_annualized_vol"])


# ── Validation ─────────────────────────────────────────────────────────────────


class TestValidation:
    def test_non_positive_horizon_raises(self):
        returns = _simulate_garch11(2000, 1e-6, 0.08, 0.90)
        with pytest.raises(ValidationError, match="forecast_horizon"):
            garch_volatility_forecast(returns, forecast_horizon=0)

    def test_too_few_observations_raises(self):
        returns = _simulate_garch11(50, 1e-6, 0.08, 0.90)
        with pytest.raises(ValidationError, match="at least 100"):
            garch_volatility_forecast(returns)


# ── Scale / performance sanity (not a hard perf assertion elsewhere) ─────────


@pytest.mark.benchmark
class TestScale:
    def test_two_million_points_fits_quickly(self):
        import time

        rng = np.random.default_rng(7)
        dates = pd.date_range("2000-01-01", periods=2_000_000, freq="min")
        returns = pd.Series(rng.standard_normal(2_000_000) * 0.01, index=dates)
        t0 = time.time()
        result = garch_volatility_forecast(returns, forecast_horizon=10)
        elapsed = time.time() - t0
        assert elapsed < 15.0, f"2M-point GARCH fit took {elapsed:.2f}s"
        assert result["n_obs"] == 2_000_000
