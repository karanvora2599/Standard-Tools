"""Tests for backtest/robustness.py — bootstrap CI, parameter sensitivity, DSR."""

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.robustness import (
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    parameter_sensitivity,
)
from standard_quant_tools.error import ValidationError


class TestBlockBootstrapCi:
    def test_empty_returns_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            block_bootstrap_ci(pd.Series(dtype=float), lambda r: float(r.mean()))

    def test_block_size_out_of_range_raises(self):
        returns = pd.Series([0.01] * 10)
        with pytest.raises(ValidationError, match="block_size"):
            block_bootstrap_ci(returns, lambda r: float(r.mean()), block_size=20)

    def test_invalid_confidence_raises(self):
        returns = pd.Series([0.01] * 30)
        with pytest.raises(ValidationError, match="confidence"):
            block_bootstrap_ci(
                returns, lambda r: float(r.mean()), block_size=5, confidence=1.5
            )

    def test_constant_series_ci_collapses_to_point_estimate(self):
        """Every resampled block of a constant series is identical, so the
        bootstrap distribution has zero spread — a deterministic check that
        doesn't depend on RNG behavior."""
        returns = pd.Series([0.002] * 100)
        result = block_bootstrap_ci(
            returns,
            lambda r: float(r.mean()),
            n_iterations=50,
            block_size=10,
            seed=0,
        )
        assert result["point_estimate"] == pytest.approx(0.002)
        assert result["ci_lower"] == pytest.approx(0.002, abs=1e-9)
        assert result["ci_upper"] == pytest.approx(0.002, abs=1e-9)

    def test_ci_contains_point_estimate(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.0005, 0.01, 500))
        result = block_bootstrap_ci(
            returns,
            lambda r: float(r.mean()),
            n_iterations=300,
            block_size=20,
            seed=1,
        )
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_output_keys_present(self):
        returns = pd.Series(np.random.default_rng(3).normal(0, 0.01, 50))
        result = block_bootstrap_ci(
            returns, lambda r: float(r.mean()), n_iterations=20, block_size=5
        )
        for key in (
            "point_estimate",
            "ci_lower",
            "ci_upper",
            "confidence",
            "n_iterations",
            "block_size",
            "bootstrap_scheme",
            "bootstrap_scale",
        ):
            assert key in result


class TestTheIntervalCoversAShortSeries:
    """
    With moving blocks, a nominal 95% interval on 60 returns with the
    default 20-day blocks covered the true mean 72.5% of the time on these
    seeds: the draws under-reach the first and last 19 returns and a block
    bootstrap's variance of a mean is short by (1 - b/n). Circular blocks
    widened by sqrt(n / (n - b)): 89%.
    """

    @staticmethod
    def _coverage(**kwargs):
        mu, covered, trials = 0.0008, 0, 200
        for seed in range(trials):
            returns = pd.Series(np.random.default_rng(seed).normal(mu, 0.012, 60))
            result = block_bootstrap_ci(
                returns,
                lambda r: float(r.mean()),
                n_iterations=400,
                block_size=20,
                seed=seed,
                **kwargs,
            )
            covered += result["ci_lower"] <= mu <= result["ci_upper"]
        return covered / trials

    def test_sixty_returns_with_twenty_day_blocks(self):
        rate = self._coverage()
        assert rate >= 0.85, f"a nominal 95% interval covered the mean {rate:.0%}"

    def test_the_moving_draw_is_kept_on_request_and_is_what_was_too_narrow(self):
        assert self._coverage(circular=False) < 0.8

    def test_every_return_is_drawn_equally_often(self):
        """One day carries the whole sum; its draws average the sample mean
        only if that day is reached as often as any other."""
        returns = pd.Series(np.r_[1.0, np.zeros(59)])
        draws = []

        def recorded(r):
            draws.append(float(r.mean()))
            return draws[-1]

        block_bootstrap_ci(returns, recorded, n_iterations=2000, block_size=20, seed=0)
        point, resampled = draws[0], np.mean(draws[1:])
        assert resampled == pytest.approx(point, rel=0.05)

    def test_the_draws_are_widened_and_say_by_how_much(self):
        returns = pd.Series(np.random.default_rng(4).normal(0, 0.01, 120))
        result = block_bootstrap_ci(
            returns, lambda r: float(r.mean()), n_iterations=200, block_size=20, seed=1
        )
        assert result["bootstrap_scheme"] == "circular"
        assert result["bootstrap_scale"] == pytest.approx(math.sqrt(120 / 100))
        moving = block_bootstrap_ci(
            returns,
            lambda r: float(r.mean()),
            n_iterations=200,
            block_size=20,
            seed=1,
            circular=False,
        )
        assert (moving["bootstrap_scheme"], moving["bootstrap_scale"]) == (
            "moving",
            1.0,
        )


class TestParameterSensitivity:
    def test_empty_grid_raises(self):
        with pytest.raises(ValidationError, match="empty"):
            parameter_sensitivity(pd.DataFrame())

    def test_missing_metric_col_raises(self):
        grid = pd.DataFrame({"sharpe_ratio": [1.0, 0.5]})
        with pytest.raises(ValidationError, match="total_return"):
            parameter_sensitivity(grid, metric_col="total_return")

    def test_hand_computed_gaps(self):
        grid = pd.DataFrame({"sharpe_ratio": [2.0, 1.5, 1.0, 0.5, 0.2, 0.1]})
        result = parameter_sensitivity(grid)
        assert result["n_trials"] == 6
        assert result["best"] == pytest.approx(2.0)
        assert result["median"] == pytest.approx(
            0.75
        )  # median of [2,1.5,1,0.5,0.2,0.1]
        assert result["best_minus_median"] == pytest.approx(1.25)
        assert result["best_minus_rank2"] == pytest.approx(0.5)  # 2.0 - 1.5
        # ranks 2-5: [1.5, 1.0, 0.5, 0.2], mean = 0.8
        assert result["best_minus_top5_mean"] == pytest.approx(2.0 - 0.8)

    def test_single_trial_gaps_are_zero(self):
        grid = pd.DataFrame({"sharpe_ratio": [1.2]})
        result = parameter_sensitivity(grid)
        assert result["n_trials"] == 1
        assert result["best_minus_rank2"] == 0.0
        assert result["best_minus_top5_mean"] == 0.0


class TestDeflatedSharpeRatio:
    def test_n_obs_below_two_raises(self):
        with pytest.raises(ValidationError, match="n_obs"):
            deflated_sharpe_ratio(1.0, sharpe_trials_std=0.5, n_trials=10, n_obs=1)

    def test_degenerate_skew_kurtosis_raises(self):
        with pytest.raises(ValidationError, match="degenerate"):
            deflated_sharpe_ratio(
                1.0,
                sharpe_trials_std=0.5,
                n_trials=10,
                n_obs=252,
                skew=100.0,
                kurtosis=3.0,
            )

    def test_single_trial_has_no_deflation(self):
        result = deflated_sharpe_ratio(
            1.0, sharpe_trials_std=0.5, n_trials=1, n_obs=252
        )
        assert result["expected_max_sharpe"] == 0.0

    def test_more_trials_deflates_more(self):
        """The whole point of DSR: holding the observed Sharpe fixed, more
        trials searched -> higher expected_max_sharpe bar -> lower DSR."""
        kwargs = dict(observed_sharpe=1.5, sharpe_trials_std=0.5, n_obs=252)
        few = deflated_sharpe_ratio(n_trials=2, **kwargs)
        many = deflated_sharpe_ratio(n_trials=200, **kwargs)
        assert many["expected_max_sharpe"] > few["expected_max_sharpe"]
        assert many["deflated_sharpe_ratio"] < few["deflated_sharpe_ratio"]

    def test_dsr_in_unit_interval(self):
        result = deflated_sharpe_ratio(
            1.2, sharpe_trials_std=0.4, n_trials=50, n_obs=500
        )
        assert 0.0 <= result["deflated_sharpe_ratio"] <= 1.0

    def test_higher_observed_sharpe_gives_higher_dsr(self):
        kwargs = dict(sharpe_trials_std=0.4, n_trials=20, n_obs=252)
        low = deflated_sharpe_ratio(observed_sharpe=0.3, **kwargs)
        high = deflated_sharpe_ratio(observed_sharpe=2.0, **kwargs)
        assert high["deflated_sharpe_ratio"] > low["deflated_sharpe_ratio"]
