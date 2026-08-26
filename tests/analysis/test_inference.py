"""
Error bars, checked by COVERAGE rather than by shape.

A confidence interval has exactly one property worth testing: a 95% interval
must contain the true value 95% of the time. Everything else -- that it is
symmetric, that it is wide, that it looks plausible -- is decoration. So the
central test here generates 200 samples from a process whose true Sharpe is
known by construction and counts how often the interval covers it.

That is also the only test that would catch the mistake this module is most
likely to make. An IID bootstrap produces intervals that look entirely
reasonable and are systematically too narrow; nothing about the returned
numbers reveals it, and only a coverage test does.

THE NULL CASES ARE PAIRED throughout: distributions that are the same must
not be called different, a stable correlation must not be called unstable,
and at zero autocorrelation the blocked and IID bootstraps must agree -- the
last of which is what shows the block correction costs nothing when it is
not needed.
"""

import math

import numpy as np
import pytest

from standard_quant_tools.analysis.inference import (
    STATISTICS,
    bootstrap_statistic,
    compare_distributions,
    decompose_returns,
    rolling_correlation_stability,
)
from standard_quant_tools.error import ValidationError

MU, SIGMA = 0.0008, 0.012
TRUE_SHARPE = MU / SIGMA * math.sqrt(252)


def _ar1(phi, n=756, seed=0, sigma=SIGMA, mu=MU):
    rng = np.random.default_rng(seed)
    value, out = 0.0, []
    for _ in range(n):
        value = phi * value + rng.normal(0, sigma)
        out.append(value + mu)
    return np.array(out)


class TestBootstrapCoverage:
    def test_the_interval_covers_the_true_value_at_about_the_nominal_rate(self):
        """
        THE ONLY PROPERTY A CONFIDENCE INTERVAL HAS. Generated from a process
        whose true Sharpe is known by construction, so "covers" is not a
        judgement call.
        """
        covered = 0
        trials = 150
        for seed in range(trials):
            sample = np.random.default_rng(seed).normal(MU, SIGMA, 756)
            result = bootstrap_statistic(
                sample, statistic="sharpe", n_bootstrap=300, seed=seed
            )
            covered += result["lower"] <= TRUE_SHARPE <= result["upper"]
        rate = covered / trials
        assert 0.85 <= rate <= 0.99, (
            f"a nominal 95% interval covered the true Sharpe {rate:.0%} of " "the time"
        )

    def test_a_wider_confidence_gives_a_wider_interval(self):
        sample = np.random.default_rng(0).normal(MU, SIGMA, 756)
        narrow = bootstrap_statistic(sample, confidence=0.80, n_bootstrap=300, seed=1)
        wide = bootstrap_statistic(sample, confidence=0.99, n_bootstrap=300, seed=1)
        assert wide["interval_width"] > narrow["interval_width"]

    def test_more_data_gives_a_narrower_interval(self):
        short = bootstrap_statistic(
            np.random.default_rng(2).normal(MU, SIGMA, 252), n_bootstrap=300, seed=2
        )
        long = bootstrap_statistic(
            np.random.default_rng(2).normal(MU, SIGMA, 2520), n_bootstrap=300, seed=2
        )
        assert long["interval_width"] < short["interval_width"]


class TestBlockedVersusIid:
    @pytest.mark.parametrize("statistic", ["sharpe", "max_drawdown"])
    def test_at_zero_autocorrelation_the_two_agree(self, statistic):
        """
        The null case, and the one that shows the correction is not simply
        inflating everything. On an IID series the blocked bootstrap must
        reproduce the IID one.
        """
        sample = _ar1(0.0, seed=3)
        iid = bootstrap_statistic(
            sample, statistic=statistic, block_size=1, n_bootstrap=600, seed=1
        )
        blocked = bootstrap_statistic(
            sample, statistic=statistic, n_bootstrap=600, seed=1
        )
        ratio = blocked["interval_width"] / iid["interval_width"]
        assert 0.8 < ratio < 1.25, f"{statistic}: ratio {ratio:.2f} at phi=0"

    @pytest.mark.parametrize("statistic", ["sharpe", "max_drawdown"])
    def test_under_persistence_the_iid_interval_is_too_narrow(self, statistic):
        """
        Measured at phi=0.8: the IID interval understates the Sharpe's by
        2.24x and the drawdown's by 1.63x. Averaged over seeds because a
        single draw's ratio is noisy.
        """
        ratios = []
        for seed in range(8):
            sample = _ar1(0.8, seed=seed)
            iid = bootstrap_statistic(
                sample, statistic=statistic, block_size=1, n_bootstrap=400, seed=seed
            )
            blocked = bootstrap_statistic(
                sample, statistic=statistic, n_bootstrap=400, seed=seed
            )
            ratios.append(blocked["interval_width"] / iid["interval_width"])
        assert np.mean(ratios) > 1.3, (
            f"{statistic}: blocked/IID width ratio {np.mean(ratios):.2f} on a "
            "strongly autocorrelated series"
        )

    def test_the_sharpe_is_more_affected_than_the_drawdown(self):
        """
        The correction to a claim I had backwards. Reasoning from "path
        dependence" says maximum drawdown should suffer most; it does not.
        The Sharpe depends directly on the variance estimate, which is what
        serial correlation distorts.
        """
        sharpe, drawdown = [], []
        for seed in range(8):
            sample = _ar1(0.8, seed=seed)
            for statistic, target in (("sharpe", sharpe), ("max_drawdown", drawdown)):
                iid = bootstrap_statistic(
                    sample,
                    statistic=statistic,
                    block_size=1,
                    n_bootstrap=400,
                    seed=seed,
                )
                blocked = bootstrap_statistic(
                    sample, statistic=statistic, n_bootstrap=400, seed=seed
                )
                target.append(blocked["interval_width"] / iid["interval_width"])
        assert np.mean(sharpe) > np.mean(drawdown)

    def test_an_iid_bootstrap_says_it_is_one(self):
        result = bootstrap_statistic(
            _ar1(0.5, seed=4), block_size=1, n_bootstrap=300, seed=1
        )
        assert any("IID bootstrap" in w for w in result["warnings"])

    def test_the_block_size_is_reported(self):
        result = bootstrap_statistic(_ar1(0.0, n=1000, seed=5), n_bootstrap=300)
        assert result["block_size"] == 10  # 1000 ** (1/3)


class TestBootstrapReporting:
    def test_an_interval_containing_zero_is_called_out(self):
        """A sample that does not distinguish the strategy from no edge has
        to say so, whatever the point estimate is."""
        sample = np.random.default_rng(6).normal(0.0001, 0.02, 300)
        result = bootstrap_statistic(sample, n_bootstrap=400, seed=1)
        if result["contains_zero"]:
            assert any("CONTAINS ZERO" in w for w in result["warnings"])

    def test_the_estimated_bias_is_reported(self):
        result = bootstrap_statistic(
            np.random.default_rng(7).normal(MU, SIGMA, 500), n_bootstrap=400, seed=1
        )
        assert result["estimated_bias"] is not None

    @pytest.mark.parametrize("statistic", list(STATISTICS))
    def test_every_advertised_statistic_computes(self, statistic):
        sample = np.random.default_rng(8).normal(MU, SIGMA, 400)
        result = bootstrap_statistic(
            sample, statistic=statistic, n_bootstrap=200, seed=1
        )
        assert result["point_estimate"] is not None
        assert result["lower"] <= result["upper"]

    def test_an_unknown_statistic_lists_the_available_ones(self):
        with pytest.raises(ValidationError, match="Available"):
            bootstrap_statistic(
                np.random.default_rng(9).normal(0, 0.01, 200), statistic="alpha"
            )

    def test_it_is_reproducible_from_the_seed(self):
        sample = np.random.default_rng(10).normal(MU, SIGMA, 400)
        a = bootstrap_statistic(sample, n_bootstrap=300, seed=42)
        b = bootstrap_statistic(sample, n_bootstrap=300, seed=42)
        assert a["lower"] == b["lower"] and a["upper"] == b["upper"]

    def test_too_little_data_is_refused(self):
        with pytest.raises(ValidationError, match="at least"):
            bootstrap_statistic([0.01, 0.02, -0.01])


class TestCompareDistributions:
    def test_two_samples_from_one_distribution_are_not_called_different(self):
        rng = np.random.default_rng(0)
        result = compare_distributions(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
        assert result["same_distribution_at_05"]

    def test_the_false_positive_rate_is_near_nominal(self):
        fires = 0
        for seed in range(150):
            a = np.random.default_rng(seed).normal(0, 1, 300)
            b = np.random.default_rng(seed + 5000).normal(0, 1, 300)
            fires += compare_distributions(a, b)["p_value"] < 0.05
        assert fires / 150 < 0.15, f"{fires}/150 identical distributions flagged"

    def test_a_mean_shift_is_detected(self):
        rng = np.random.default_rng(1)
        result = compare_distributions(rng.normal(0, 1, 500), rng.normal(0.5, 1, 500))
        assert result["p_value"] < 0.01
        assert not result["same_distribution_at_05"]

    def test_ks_misses_a_tail_only_difference_and_the_result_says_so(self):
        """
        THE LIMITATION THIS TOOL EXISTS TO SURFACE. A normal against a t(3)
        has the same mean and a far higher kurtosis; KS returns p=0.22 --
        nowhere near rejection -- while the 1st percentile has moved by a
        factor of 1.86. Reading only the p-value would conclude nothing
        changed.
        """
        rng = np.random.default_rng(2)
        result = compare_distributions(rng.normal(0, 1, 500), rng.standard_t(3, 500))
        kurtosis = next(s for s in result["moment_shifts"] if s["moment"] == "kurtosis")
        assert kurtosis["change"] > 3, "the t(3) sample should be far fatter-tailed"
        assert result["tail_ratio_p01"] > 1.5
        assert any("LEAST sensitive in the tails" in w for w in result["warnings"])

    def test_the_moment_that_moved_is_named(self):
        rng = np.random.default_rng(3)
        result = compare_distributions(rng.normal(0, 1, 500), rng.normal(0, 2, 500))
        std = next(s for s in result["moment_shifts"] if s["moment"] == "std")
        assert std["relative_change"] > 0.5

    def test_a_thin_sample_is_flagged_as_underpowered(self):
        rng = np.random.default_rng(4)
        result = compare_distributions(rng.normal(0, 1, 20), rng.normal(0, 1, 20))
        assert any("little power" in w for w in result["warnings"])

    def test_the_labels_are_used_in_the_output(self):
        rng = np.random.default_rng(5)
        result = compare_distributions(
            rng.normal(0, 1, 200),
            rng.normal(0, 1, 200),
            label_a="in_sample",
            label_b="out_of_sample",
        )
        assert "in_sample" in result["moments"]
        assert "out_of_sample" in result["moments"]

    def test_too_little_data_is_refused(self):
        with pytest.raises(ValidationError, match="at least"):
            compare_distributions([0.1, 0.2], [0.3, 0.4])


class TestRollingCorrelationStability:
    @staticmethod
    def _pair(n=1000, seed=3, flip=False):
        rng = np.random.default_rng(seed)
        factor = rng.normal(0, 1, n)
        if flip:
            other = np.concatenate([factor[: n // 2], -factor[n // 2 :]])
        else:
            other = 0.8 * factor
        return factor, other + rng.normal(0, 0.3, n)

    def test_a_sign_flipping_pair_is_caught(self):
        """
        The pathology: a full-sample correlation near zero that is +0.9 for
        half the sample and -0.9 for the other. A hedge sized on the average
        is wrong in both regimes.
        """
        a, b = self._pair(flip=True)
        result = rolling_correlation_stability(a, b, window=63)
        assert abs(result["full_sample_correlation"]) < 0.2
        assert result["sign_flips"] >= 1
        assert result["max_rolling"] - result["min_rolling"] > 1.0
        assert any("changes SIGN" in w for w in result["warnings"])

    def test_a_stable_pair_is_not_flagged(self):
        """The null case."""
        a, b = self._pair(flip=False)
        result = rolling_correlation_stability(a, b, window=63)
        assert result["sign_flips"] == 0
        assert result["fraction_within_0_2"] > 0.8
        assert not any("changes SIGN" in w for w in result["warnings"])

    def test_the_overlap_of_the_windows_is_declared(self):
        a, b = self._pair()
        result = rolling_correlation_stability(a, b, window=63)
        assert result["n_independent_windows"] < result["n_windows"]
        assert any("not a sample size" in w for w in result["warnings"])

    def test_the_stress_correlation_is_reported(self):
        a, b = self._pair()
        result = rolling_correlation_stability(a, b, window=63)
        assert result["stress_correlation"] is not None

    def test_misaligned_series_are_refused(self):
        with pytest.raises(ValidationError, match="aligned"):
            rolling_correlation_stability(np.arange(100.0), np.arange(90.0))

    def test_too_little_data_for_two_windows_is_refused(self):
        with pytest.raises(ValidationError, match="fewer than two windows"):
            rolling_correlation_stability(
                np.random.default_rng(0).normal(0, 1, 100),
                np.random.default_rng(1).normal(0, 1, 100),
                window=63,
            )


class TestDecomposeReturns:
    def test_the_volatility_drag_matches_half_the_variance(self):
        """
        The identity behind the whole tool: geometric = arithmetic minus
        roughly sigma^2/2. Measured agreement to two basis points annualized.
        """
        sample = np.random.default_rng(4).normal(0.0008, 0.03, 2520)
        result = decompose_returns(sample)
        predicted = 0.5 * float(sample.var(ddof=1))
        assert result["volatility_drag"] == pytest.approx(predicted, rel=0.05)

    def test_the_geometric_mean_is_below_the_arithmetic_one(self):
        result = decompose_returns(np.random.default_rng(5).normal(0.001, 0.02, 1000))
        assert result["geometric_mean"] < result["arithmetic_mean"]

    def test_a_constant_series_has_no_drag(self):
        result = decompose_returns([0.001] * 500)
        assert result["volatility_drag"] == pytest.approx(0.0, abs=1e-12)

    def test_heavy_drag_is_flagged(self):
        result = decompose_returns(np.random.default_rng(6).normal(0.0008, 0.04, 1500))
        assert any("Volatility drag" in w for w in result["warnings"])

    def test_a_result_resting_on_five_days_is_called_out(self):
        """A strategy whose whole return comes from five observations is a
        lottery ticket with good statistics."""
        returns = list(np.random.default_rng(7).normal(-0.0004, 0.005, 500))
        returns[100:105] = [0.35] * 5
        result = decompose_returns(returns)
        if result["total_without_best_5"] < 0 < result["total_return"]:
            assert any("five best days" in w for w in result["warnings"])

    def test_the_win_loss_profile_is_reported(self):
        result = decompose_returns(np.random.default_rng(8).normal(0.0005, 0.01, 800))
        assert result["n_positive"] + result["n_negative"] <= result["n_observations"]
        assert 0 <= result["win_rate"] <= 1
        assert result["mean_win"] > 0 > result["mean_loss"]
