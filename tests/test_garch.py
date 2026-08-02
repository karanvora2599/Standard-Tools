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


# ── garch_volatility_forecast — forecast seed uses the last observed return ──


class TestForecastSeed:
    """Regression coverage for a real bug: the forecast used to seed itself
    from sigma2[-1] (the model's own fitted variance for the last observed
    bar, computed only through resid_sq[-2]) without ever taking one more
    step to incorporate resid_sq[-1], the actual last observed squared
    return -- silently understating the forecast right after a shock."""

    def test_forecast_first_step_equals_current_vol_exactly(self):
        # current_annualized_vol and forecast_annualized_vol[0] both claim to
        # describe the same thing (the next bar's conditional vol) -- they
        # must be numerically identical, not just close.
        returns = _simulate_garch11(2000, 1e-6, 0.08, 0.90)
        result = garch_volatility_forecast(returns, forecast_horizon=10)
        assert result["forecast_annualized_vol"][0] == pytest.approx(
            result["current_annualized_vol"], rel=1e-9
        )

    def test_current_vol_reacts_to_an_outsized_last_bar_shock(self):
        # Two otherwise-identical series differing only in the sign/scale of
        # the very last return: current_annualized_vol must be able to move
        # in response to that last bar. Under the bug, current_var was
        # computed from information only through the SECOND-to-last bar, so
        # this last-bar shock had zero effect on the reported "current" vol.
        rng = np.random.default_rng(7)
        base = rng.normal(0, 0.01, 500)
        calm = pd.Series(np.append(base[:-1], 0.005))
        shocked = pd.Series(np.append(base[:-1], 0.5))

        calm_result = garch_volatility_forecast(calm, forecast_horizon=1)
        shocked_result = garch_volatility_forecast(shocked, forecast_horizon=1)

        assert (
            shocked_result["current_annualized_vol"]
            > calm_result["current_annualized_vol"] * 2
        )

    def test_current_var_matches_one_more_hand_computed_recursion_step(self):
        # Direct hand-computation, same style as
        # TestVarianceRecursion.test_matches_hand_computed_values: current_var
        # must equal omega + alpha*resid_sq[-1] + beta*sigma2[-1], where
        # sigma2 is the recursion's own output array (i.e. one explicit step
        # beyond sigma2[-1] itself).
        omega, alpha, beta = 1e-6, 0.08, 0.90
        returns = _simulate_garch11(300, omega, alpha, beta)
        resid = (returns - returns.mean()).to_numpy()
        resid_sq = resid**2

        result = garch_volatility_forecast(returns, forecast_horizon=1)
        fitted_omega, fitted_alpha, fitted_beta = (
            result["omega"],
            result["alpha"],
            result["beta"],
        )
        fitted_sigma2 = _garch11_variance_recursion(
            resid_sq, fitted_omega, fitted_alpha, fitted_beta
        )
        expected_current_var = (
            fitted_omega + fitted_alpha * resid_sq[-1] + fitted_beta * fitted_sigma2[-1]
        )
        expected_annualized = float(np.sqrt(expected_current_var * 252))
        assert result["current_annualized_vol"] == pytest.approx(
            expected_annualized, rel=1e-6
        )


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
        assert result["persistence"] == pytest.approx(result["alpha"] + result["beta"])

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

    def test_inf_in_returns_raises(self):
        """Inf input contract: returns.dropna() strips NaN but not +/-Inf,
        and garch11_variance_recursion_into's floor-clamp (mean <
        kMinSigma2) is false for Inf, so it would otherwise silently
        propagate through the entire native recursion uncaught."""
        returns = _simulate_garch11(200, 1e-6, 0.08, 0.90)
        returns.iloc[50] = np.inf
        with pytest.raises(ValidationError, match="non-finite"):
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
