"""
Whether the overfitting detectors detect overfitting.

THE STRUCTURE OF EVERY TEST HERE is a pair: one series built to BE
overfitted and one built not to be, with the detector required to tell them
apart. A detector that flags everything is as useless as one that flags
nothing, and only the pair reveals which you have.

The simulations are the whole test. `_noise_grid` builds N strategies drawn
from a zero-mean normal -- there is no edge anywhere in it, by construction,
so anything a detector finds there is a false positive. `_edge_grid` plants
a genuine drift in exactly one column. PBO must come back near 0.5 on the
first and near 0 on the second; if it does not separate them the
implementation is wrong regardless of what it returns on real data.

CALIBRATION IS CHECKED AS A RATE where the statistic claims one. The reality
check claims a 5% false-positive rate and is measured over many independent
draws rather than asserted on one -- a single seed cannot distinguish a
broken test from an unlucky one, which is a lesson this file's Granger
equivalent learned the hard way.
"""

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtesting.overfitting import (
    _norm_ppf,
    combinatorial_purged_cv,
    deflated_sharpe_ratio,
    parameter_decay,
    probability_of_backtest_overfitting,
    reality_check,
    regime_stratified_performance,
)
from standard_quant_tools.error import ValidationError


def _noise_grid(n=1000, k=20, seed=3):
    """K strategies with NO edge whatsoever. Anything found here is false."""
    return pd.DataFrame(np.random.default_rng(seed).normal(0, 0.01, (n, k)))


def _edge_grid(n=1000, k=20, drift=0.0012, seed=3):
    """The same, with a genuine drift planted in column 0."""
    frame = _noise_grid(n, k, seed)
    frame[0] = np.random.default_rng(seed + 100).normal(drift, 0.01, n)
    return frame


class TestNormPpf:
    """The inverse normal is written out because scipy is not a dependency.
    If it drifts, every threshold in this module drifts with it."""

    @pytest.mark.parametrize(
        "p,expected",
        [
            (0.5, 0.0),
            (0.975, 1.959964),
            (0.99, 2.326348),
            (0.995, 2.575829),
            (0.001, -3.090232),
            (0.025, -1.959964),
        ],
    )
    def test_it_matches_the_tabulated_quantiles(self, p, expected):
        assert _norm_ppf(p) == pytest.approx(expected, abs=1e-5)

    def test_it_is_symmetric(self):
        for p in (0.01, 0.1, 0.3):
            assert _norm_ppf(p) == pytest.approx(-_norm_ppf(1 - p), abs=1e-9)

    def test_a_probability_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValidationError, match=r"\(0, 1\)"):
            _norm_ppf(1.5)


class TestDeflatedSharpe:
    def test_the_best_of_twenty_noise_strategies_is_not_significant(self):
        """
        THE CENTRAL TEST. Twenty strategies with no edge; the best shows an
        annualized Sharpe near 1.3 purely from being the maximum of twenty
        draws. Undeflated it looks publishable. Deflated it must not.
        """
        grid = _noise_grid(n=504, k=20, seed=0).to_numpy()
        sharpes = grid.mean(0) / grid.std(0, ddof=1) * math.sqrt(252)
        best = pd.Series(grid[:, int(sharpes.argmax())])
        result = deflated_sharpe_ratio(best, n_trials=20, trial_sharpes=sharpes)
        assert result["observed_sharpe"] > 0.8
        assert not result["significant_at_95"]
        assert result["deflation_threshold"] > 0.8

    def test_the_deflation_threshold_rises_with_the_number_of_trials(self):
        returns = pd.Series(np.random.default_rng(1).normal(0.0008, 0.01, 1260))
        thresholds = [
            deflated_sharpe_ratio(returns, n_trials=n)["deflation_threshold"]
            for n in (1, 10, 100, 1000)
        ]
        assert thresholds == sorted(thresholds)
        assert thresholds[0] == 0.0

    def test_the_same_result_becomes_less_significant_as_trials_grow(self):
        returns = pd.Series(np.random.default_rng(1).normal(0.0012, 0.01, 1260))
        few = deflated_sharpe_ratio(returns, n_trials=1)
        many = deflated_sharpe_ratio(returns, n_trials=500)
        assert few["observed_sharpe"] == many["observed_sharpe"]
        assert many["deflated_sharpe_probability"] < few["deflated_sharpe_probability"]

    def test_a_strong_edge_with_one_trial_survives(self):
        returns = pd.Series(np.random.default_rng(2).normal(0.0025, 0.01, 1260))
        result = deflated_sharpe_ratio(returns, n_trials=1)
        assert result["significant_at_95"]

    def test_negative_skew_is_penalised(self):
        """
        The short-volatility payoff -- many small gains, rare large losses --
        is exactly the shape that most needs deflating, and normal-theory
        Sharpe inference does not see it.
        """
        rng = np.random.default_rng(4)
        n = 1260
        symmetric = rng.normal(0.0008, 0.01, n)
        skewed = np.where(rng.random(n) < 0.97, 0.0012, -0.030)
        a = deflated_sharpe_ratio(pd.Series(symmetric), n_trials=1)
        b = deflated_sharpe_ratio(pd.Series(skewed), n_trials=1)
        assert b["skewness"] < -0.5
        assert any("Negative skew" in w for w in b["warnings"])
        assert b["sharpe_standard_error"] > a["sharpe_standard_error"]

    def test_fat_tails_widen_the_standard_error(self):
        rng = np.random.default_rng(5)
        n = 1260
        thin = pd.Series(rng.normal(0.0005, 0.01, n))
        fat = pd.Series(rng.standard_t(3, n) * 0.006 + 0.0005)
        thin_result = deflated_sharpe_ratio(thin, n_trials=1)
        fat_result = deflated_sharpe_ratio(fat, n_trials=1)
        assert fat_result["kurtosis"] > thin_result["kurtosis"]

    def test_the_trial_sharpe_variance_is_used_when_given(self):
        """
        100 near-identical parameter settings deflate much less than 100
        genuinely different ideas, and only the trial distribution knows
        which you have.
        """
        returns = pd.Series(np.random.default_rng(6).normal(0.001, 0.01, 1000))
        tight = deflated_sharpe_ratio(
            returns,
            n_trials=50,
            trial_sharpes=np.full(50, 0.5) + np.linspace(0, 0.01, 50),
        )
        wide = deflated_sharpe_ratio(
            returns,
            n_trials=50,
            trial_sharpes=np.random.default_rng(7).normal(0, 1.5, 50),
        )
        assert wide["deflation_threshold"] > tight["deflation_threshold"]

    def test_the_uncounted_trials_are_named(self):
        returns = pd.Series(np.random.default_rng(8).normal(0.001, 0.01, 500))
        result = deflated_sharpe_ratio(returns, n_trials=5)
        assert any("uncounted" in w for w in result["warnings"])

    def test_zero_trials_is_refused(self):
        returns = pd.Series(np.random.default_rng(9).normal(0.001, 0.01, 500))
        with pytest.raises(ValidationError, match="at least 1"):
            deflated_sharpe_ratio(returns, n_trials=0)

    @pytest.mark.parametrize("level", [0.0, 0.001, -0.002, 1e-8, 1e6])
    def test_a_flat_series_has_no_sharpe_to_deflate(self, level):
        """
        Caught by a RELATIVE test, because a constant series does not have a
        standard deviation of zero in floating point. numpy returns 2.2e-19
        for a flat 0.001 series -- the deviations are taken against an
        accumulated mean and the rounding does not cancel -- so `std <= 0`
        passes and the Sharpe comes back as 7.3e16. Finite, no NaN, and
        complete nonsense. Parametrized across scales because an absolute
        epsilon would pass at 1e6 and wrongly reject genuine returns at 1e-8.
        """
        with pytest.raises(ValidationError, match="no dispersion"):
            deflated_sharpe_ratio(pd.Series(np.full(300, level)), n_trials=1)

    def test_a_genuinely_tiny_but_varying_series_is_not_rejected(self):
        """The other side of the relative test: small is not constant."""
        returns = pd.Series(np.random.default_rng(20).normal(1e-8, 1e-8, 500))
        result = deflated_sharpe_ratio(returns, n_trials=1)
        assert math.isfinite(result["observed_sharpe"])

    def test_one_trial_sharpe_is_refused_because_the_variance_is_the_point(self):
        returns = pd.Series(np.random.default_rng(10).normal(0.001, 0.01, 500))
        with pytest.raises(ValidationError, match="VARIANCE"):
            deflated_sharpe_ratio(returns, n_trials=5, trial_sharpes=[0.4])


class TestPBO:
    def test_a_pure_noise_grid_gives_a_pbo_near_one_half(self):
        """
        The definitional check. With no edge anywhere, picking the in-sample
        best is exactly as good as picking at random, which is PBO = 0.5.
        """
        result = probability_of_backtest_overfitting(_noise_grid(), n_splits=8)
        assert 0.30 < result["pbo"] < 0.70

    def test_a_grid_containing_a_real_edge_gives_a_low_pbo(self):
        """The other side of the pair: when one configuration genuinely
        works, the in-sample winner keeps winning out of sample."""
        result = probability_of_backtest_overfitting(_edge_grid(), n_splits=8)
        assert result["pbo"] < 0.2

    def test_the_two_cases_are_clearly_separated(self):
        noise = probability_of_backtest_overfitting(_noise_grid(), n_splits=8)["pbo"]
        edge = probability_of_backtest_overfitting(_edge_grid(), n_splits=8)["pbo"]
        assert noise - edge > 0.25

    def test_a_collinear_grid_is_flagged_as_one_strategy(self):
        """
        A hundred configurations correlated at 0.99 are one strategy with a
        parameter nudged. Every split ranks them identically and the PBO is
        measuring nothing.
        """
        rng = np.random.default_rng(11)
        base = rng.normal(0, 0.01, 1000)
        grid = pd.DataFrame({i: base + rng.normal(0, 0.0005, 1000) for i in range(15)})
        result = probability_of_backtest_overfitting(grid, n_splits=8)
        assert result["median_configuration_correlation"] > 0.95
        assert any("collinear" in w for w in result["warnings"])

    def test_the_number_of_combinations_is_the_binomial_coefficient(self):
        result = probability_of_backtest_overfitting(_noise_grid(), n_splits=8)
        assert result["n_combinations"] == 70  # C(8, 4)

    def test_it_says_pbo_is_about_the_procedure_not_the_strategy(self):
        result = probability_of_backtest_overfitting(_noise_grid(), n_splits=8)
        assert any("SELECTION PROCEDURE" in w for w in result["warnings"])

    def test_one_configuration_is_refused(self):
        with pytest.raises(ValidationError, match="no choice to measure"):
            probability_of_backtest_overfitting(_noise_grid(k=1), n_splits=8)

    def test_an_odd_number_of_splits_is_refused(self):
        with pytest.raises(ValidationError, match="even"):
            probability_of_backtest_overfitting(_noise_grid(), n_splits=7)

    def test_too_little_data_per_chunk_is_refused(self):
        with pytest.raises(ValidationError, match="under 5 observations"):
            probability_of_backtest_overfitting(_noise_grid(n=30), n_splits=8)


class TestCombinatorialPurgedCV:
    def test_no_training_label_window_touches_the_test_set(self):
        """
        THE WHOLE POINT. A label built from a 5-day forward return at t is a
        function of prices through t+5. If t is in training and t+3 is in
        test, the training label contains the test answer -- which is why
        plain k-fold shows 0.6 AUC and production shows 0.5.
        """
        horizon = 5
        result = combinatorial_purged_cv(
            1000, n_splits=6, n_test_splits=2, label_horizon=horizon
        )
        leaks = 0
        for path in result["paths"]:
            test = set(path["test_index"])
            for i in path["train_index"]:
                if any(j in test for j in range(i, min(i + horizon + 1, 1000))):
                    leaks += 1
        assert leaks == 0, f"{leaks} training observations leak into their test set"

    def test_the_embargo_leaves_a_gap_after_every_test_block(self):
        result = combinatorial_purged_cv(
            1000, n_splits=6, n_test_splits=2, embargo_pct=0.01, label_horizon=1
        )
        embargo = result["embargo_observations"]
        assert embargo == 10
        for path in result["paths"]:
            test = sorted(path["test_index"])
            train = set(path["train_index"])
            for i in test:
                if i + 1 in test:
                    continue  # not the end of a block
                following = [k for k in range(i + 1, i + embargo + 1) if k in train]
                assert not following, (
                    f"training observation inside the {embargo}-observation "
                    f"embargo after test index {i}"
                )

    def test_the_path_count_is_the_binomial_coefficient(self):
        assert (
            combinatorial_purged_cv(1000, n_splits=6, n_test_splits=2)["n_paths"] == 15
        )
        assert (
            combinatorial_purged_cv(1000, n_splits=6, n_test_splits=3)["n_paths"] == 20
        )

    def test_a_longer_label_horizon_purges_more(self):
        short = combinatorial_purged_cv(1000, label_horizon=1)["mean_purged"]
        long = combinatorial_purged_cv(1000, label_horizon=20)["mean_purged"]
        assert long > short

    def test_train_and_test_never_intersect(self):
        result = combinatorial_purged_cv(500, n_splits=5, n_test_splits=2)
        for path in result["paths"]:
            assert not set(path["train_index"]) & set(path["test_index"])

    def test_a_zero_embargo_is_flagged_rather_than_silently_accepted(self):
        result = combinatorial_purged_cv(1000, embargo_pct=0.0)
        assert result["embargo_observations"] == 0
        assert any("no embargo" in w for w in result["warnings"])

    def test_over_aggressive_purging_is_flagged(self):
        result = combinatorial_purged_cv(
            200, n_splits=4, n_test_splits=2, label_horizon=40
        )
        if result["mean_train_size"] < 200 * 0.3:
            assert any("most of the sample" in w for w in result["warnings"])

    def test_too_few_observations_are_refused(self):
        with pytest.raises(ValidationError, match="too few"):
            combinatorial_purged_cv(20)

    def test_test_splits_must_be_fewer_than_splits(self):
        with pytest.raises(ValidationError, match="fewer than"):
            combinatorial_purged_cv(1000, n_splits=4, n_test_splits=4)


class TestRealityCheck:
    def test_the_false_positive_rate_is_near_nominal(self):
        """
        Measured as a RATE over independent draws. A single seed cannot tell
        a broken test from an unlucky one.
        """
        fires = 0
        trials = 40
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            strategy = pd.Series(rng.normal(0, 0.01, 500))
            benchmarks = pd.DataFrame(rng.normal(0, 0.01, (500, 10)))
            fires += reality_check(strategy, benchmarks, n_bootstrap=300, seed=seed)[
                "significant_at_05"
            ]
        assert fires / trials < 0.20, (
            f"{fires}/{trials} no-edge strategies came back significant "
            "against a nominal 5%"
        )

    def test_a_genuine_edge_is_detected(self):
        rng = np.random.default_rng(9)
        strategy = pd.Series(rng.normal(0.0015, 0.01, 500))
        benchmarks = pd.DataFrame(rng.normal(0, 0.01, (500, 10)))
        result = reality_check(strategy, benchmarks, n_bootstrap=1000, seed=1)
        assert result["significant_at_05"]
        assert result["p_value"] < 0.05

    def test_the_bootstrap_is_blocked_and_says_so(self):
        """Resampling individual days destroys the serial correlation that
        drives drawdowns, making the null too narrow."""
        rng = np.random.default_rng(10)
        result = reality_check(
            pd.Series(rng.normal(0, 0.01, 400)),
            pd.DataFrame(rng.normal(0, 0.01, (400, 5))),
            block_size=20,
            n_bootstrap=200,
        )
        assert result["block_size"] == 20
        assert any("Block bootstrap" in w for w in result["warnings"])

    def test_the_stationarity_assumption_is_declared(self):
        rng = np.random.default_rng(11)
        result = reality_check(
            pd.Series(rng.normal(0, 0.01, 400)),
            pd.DataFrame(rng.normal(0, 0.01, (400, 5))),
            n_bootstrap=200,
        )
        assert any("STATIONARY" in w for w in result["warnings"])

    def test_it_is_reproducible_from_the_seed(self):
        rng = np.random.default_rng(12)
        strategy = pd.Series(rng.normal(0.0005, 0.01, 400))
        benchmarks = pd.DataFrame(rng.normal(0, 0.01, (400, 5)))
        a = reality_check(strategy, benchmarks, n_bootstrap=200, seed=42)
        b = reality_check(strategy, benchmarks, n_bootstrap=200, seed=42)
        assert a["p_value"] == b["p_value"]

    def test_no_benchmarks_is_refused(self):
        with pytest.raises(ValidationError, match="no benchmark"):
            reality_check(
                pd.Series(np.random.default_rng(13).normal(0, 0.01, 400)),
                pd.DataFrame(),
            )


class TestRegimeStratified:
    @staticmethod
    def _concentrated(n=1000, seed=14):
        """All the P&L in one regime, and a flat rest."""
        rng = np.random.default_rng(seed)
        returns = rng.normal(0, 0.01, n)
        labels = np.array(["calm"] * n, dtype=object)
        returns[200:400] += 0.004
        labels[200:400] = "boom"
        return pd.Series(returns), pd.Series(labels)

    def test_concentration_in_one_regime_is_surfaced(self):
        """
        The failure this catches: a Sharpe of 1.2 earned entirely in one
        18-month window. The full-sample number is arithmetically correct
        and completely misleading.
        """
        returns, labels = self._concentrated()
        result = regime_stratified_performance(returns, labels)
        assert result["pnl_concentration"] > 0.7
        assert any("bet on that regime" in w for w in result["warnings"])

    def test_evenly_spread_performance_is_not_flagged(self):
        """The null case."""
        rng = np.random.default_rng(15)
        returns = pd.Series(rng.normal(0.0006, 0.01, 1200))
        labels = pd.Series(rng.choice(["a", "b", "c"], 1200))
        result = regime_stratified_performance(returns, labels)
        assert result["pnl_concentration"] < 0.75

    def test_regimes_are_ordered_by_contribution(self):
        returns, labels = self._concentrated()
        result = regime_stratified_performance(returns, labels)
        contributions = [r["total_return"] for r in result["by_regime"]]
        assert contributions == sorted(contributions, reverse=True)

    def test_the_shares_of_sample_sum_to_one(self):
        returns, labels = self._concentrated()
        result = regime_stratified_performance(returns, labels)
        assert sum(r["share_of_sample"] for r in result["by_regime"]) == pytest.approx(
            1.0
        )

    def test_a_thin_regime_is_flagged_as_a_single_draw(self):
        rng = np.random.default_rng(16)
        returns = pd.Series(rng.normal(0.0005, 0.01, 500))
        labels = pd.Series(["a"] * 490 + ["rare"] * 10)
        result = regime_stratified_performance(returns, labels)
        assert any("single draws" in w for w in result["warnings"])

    def test_one_regime_is_refused_because_it_is_the_full_sample(self):
        returns = pd.Series(np.random.default_rng(17).normal(0, 0.01, 300))
        with pytest.raises(ValidationError, match="distinct regime"):
            regime_stratified_performance(returns, pd.Series(["only"] * 300))


class TestParameterDecay:
    def test_a_broad_optimum_is_recognised_as_robust(self):
        """A parameter whose neighbours perform almost as well describes a
        real effect: the exact value is not doing the work."""
        params = np.arange(5, 55, 5)
        smooth = -0.001 * (params - 30) ** 2 + 1.2
        result = parameter_decay(params, smooth)
        assert result["spike_ratio"] > 0.85
        assert any("broad optimum" in w for w in result["warnings"])

    def test_a_spike_is_recognised_as_noise(self):
        """
        The other side. In a noisy objective a lone spike is almost always
        the single setting that happened to fit the sample.
        """
        params = np.arange(5, 55, 5)
        scores = np.full(len(params), 0.2)
        scores[5] = 1.8
        result = parameter_decay(params, scores)
        assert result["spike_ratio"] < 0.5
        assert any("SPIKE" in w for w in result["warnings"])

    def test_the_two_shapes_are_clearly_separated(self):
        params = np.arange(5, 55, 5)
        smooth = -0.001 * (params - 30) ** 2 + 1.2
        spike = np.full(len(params), 0.2)
        spike[5] = 1.8
        assert (
            parameter_decay(params, smooth)["spike_ratio"]
            > parameter_decay(params, spike)["spike_ratio"] + 0.4
        )

    def test_an_optimum_at_the_grid_edge_is_flagged(self):
        """The true optimum may lie outside the grid -- or performance is
        monotone, which usually means the parameter stands in for something
        else."""
        params = np.arange(5, 55, 5)
        monotone = np.linspace(0.1, 1.5, len(params))
        result = parameter_decay(params, monotone)
        assert result["best_at_grid_edge"]
        assert any("EDGE" in w for w in result["warnings"])

    def test_a_flat_surface_is_flagged_as_barely_mattering(self):
        params = np.arange(5, 55, 5)
        flat = np.full(len(params), 0.5) + np.linspace(0, 0.01, len(params))
        result = parameter_decay(params, flat)
        assert result["plateau_fraction"] > 0.9
        assert any("barely matters" in w for w in result["warnings"])

    def test_it_says_robustness_is_not_sufficiency(self):
        """A broad plateau at a bad level is still a bad level."""
        params = np.arange(5, 55, 5)
        result = parameter_decay(params, -0.001 * (params - 30) ** 2 + 1.2)
        assert any("nowhere near sufficient" in w for w in result["warnings"])

    def test_the_surface_is_returned_sorted_by_parameter(self):
        result = parameter_decay([30, 10, 20, 50, 40], [0.9, 0.2, 0.5, 0.3, 0.7])
        order = [p["parameter"] for p in result["surface"]]
        assert order == sorted(order)

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValidationError, match="against"):
            parameter_decay([1, 2, 3, 4, 5], [0.1, 0.2])

    def test_too_few_points_are_refused(self):
        with pytest.raises(ValidationError, match="at least 5"):
            parameter_decay([1, 2, 3], [0.1, 0.5, 0.2])
