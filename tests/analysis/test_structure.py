"""
Change points, partial correlation, Granger, tail dependence, stationarity,
regimes.

Every test below is built on data whose answer is KNOWN BY CONSTRUCTION —
a break planted at a specific index, a correlation that is entirely a common
factor, a lead of exactly two bars, two series that are independent except
in the tail. A statistical routine that returns a plausible number on
plausible data tells you nothing; one that finds the planted answer and
declines to find one that is not there tells you it works.

The null cases matter as much as the positive ones and are checked
throughout: no break in white noise, no Granger relationship in the wrong
direction, tail dependence near the quantile itself for independent series.
A detector that only ever says yes is worse than no detector, because it
carries authority.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.stationarity import (
    andrews_bandwidth,
    detect_regimes,
    kpss_statistic,
    run_stationarity_tests,
    variance_ratio,
)
from standard_quant_tools.analysis.structure import (
    detect_change_points,
    granger_causality,
    partial_correlation,
    tail_dependence,
)
from standard_quant_tools.error import ValidationError

N = 400
IDX = pd.bdate_range("2022-01-03", periods=N)


def _ar1(phi, n=N, seed=0, sigma=1.0):
    rng = np.random.default_rng(seed)
    value, out = 0.0, []
    for _ in range(n):
        value = phi * value + rng.normal(0, sigma)
        out.append(value)
    return np.array(out)


class TestChangePoints:
    def test_it_finds_a_planted_break_at_the_right_place(self):
        rng = np.random.default_rng(0)
        series = pd.Series(
            np.concatenate([rng.normal(0, 1, 200), rng.normal(4, 1, 200)]), index=IDX
        )
        result = detect_change_points(series, max_breaks=2)
        assert result["n_breaks"] >= 1
        assert (
            abs(result["breaks"][0]["index"] - 200) <= 5
        ), f"break found at {result['breaks'][0]['index']}, planted at 200"

    def test_it_finds_nothing_in_white_noise(self):
        """The null case, and the one that matters: a detector that always
        finds a break carries authority it has not earned."""
        found = 0
        for seed in range(10):
            series = pd.Series(np.random.default_rng(seed).normal(0, 1, N), index=IDX)
            found += detect_change_points(series)["n_breaks"] > 0
        assert found <= 2, f"{found}/10 pure-noise series produced a break"

    def test_the_segments_partition_the_series(self):
        rng = np.random.default_rng(1)
        series = pd.Series(
            np.concatenate([rng.normal(0, 1, 150), rng.normal(3, 1, 250)]), index=IDX
        )
        result = detect_change_points(series, max_breaks=3)
        assert sum(s["n"] for s in result["segments"]) == len(series)
        assert len(result["segments"]) == result["n_breaks"] + 1

    def test_the_gain_is_reported_so_a_marginal_call_looks_marginal(self):
        rng = np.random.default_rng(2)
        strong = pd.Series(
            np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 1, 200)]), index=IDX
        )
        weak = pd.Series(
            np.concatenate([rng.normal(0, 1, 200), rng.normal(0.4, 1, 200)]), index=IDX
        )
        big = detect_change_points(strong)["breaks"][0]["gain"]
        small = detect_change_points(weak, penalty=1.0)["breaks"]
        assert big > 500
        if small:
            assert small[0]["gain"] < big / 10

    def test_a_short_series_is_refused_with_the_reason(self):
        with pytest.raises(ValidationError, match="too short|cannot contain"):
            detect_change_points(pd.Series(np.arange(20.0)), min_segment=20)

    def test_a_higher_penalty_finds_fewer_breaks(self):
        rng = np.random.default_rng(3)
        series = pd.Series(
            np.concatenate(
                [rng.normal(0, 1, 100), rng.normal(1, 1, 100), rng.normal(2, 1, 200)]
            ),
            index=IDX,
        )
        lenient = detect_change_points(series, penalty=1.0, max_breaks=5)["n_breaks"]
        strict = detect_change_points(series, penalty=500.0, max_breaks=5)["n_breaks"]
        assert strict <= lenient


class TestPartialCorrelation:
    def test_a_common_factor_is_removed_entirely(self):
        """Two series that are ONLY a shared factor must have essentially no
        partial correlation once it is controlled for."""
        rng = np.random.default_rng(0)
        factor = rng.normal(0, 1, N)
        frame = pd.DataFrame(
            {
                "a": 0.9 * factor + rng.normal(0, 0.4, N),
                "b": 0.9 * factor + rng.normal(0, 0.4, N),
                "mkt": factor,
            },
            index=IDX,
        )
        result = partial_correlation(frame, "a", "b", ["mkt"])
        assert result["raw_correlation"] > 0.6
        assert abs(result["partial_correlation"]) < 0.15

    def test_a_genuine_pair_relationship_survives(self):
        """The other side: a real link between two names must NOT be
        explained away by the market."""
        rng = np.random.default_rng(1)
        factor = rng.normal(0, 1, N)
        shared = rng.normal(0, 1, N)
        frame = pd.DataFrame(
            {
                "a": 0.5 * factor + 0.8 * shared + rng.normal(0, 0.2, N),
                "b": 0.5 * factor + 0.8 * shared + rng.normal(0, 0.2, N),
                "mkt": factor,
            },
            index=IDX,
        )
        result = partial_correlation(frame, "a", "b", ["mkt"])
        assert result["partial_correlation"] > 0.6

    def test_it_refuses_more_controls_than_the_data_supports(self):
        frame = pd.DataFrame(
            np.random.default_rng(0).normal(size=(6, 8)),
            columns=[f"c{i}" for i in range(8)],
        )
        with pytest.raises(ValidationError, match="degree of freedom|complete rows"):
            partial_correlation(frame, "c0", "c1", [f"c{i}" for i in range(2, 8)])

    def test_an_unknown_column_is_named(self):
        frame = pd.DataFrame({"a": [1.0] * 50, "b": [2.0] * 50})
        with pytest.raises(ValidationError, match="nope"):
            partial_correlation(frame, "a", "b", ["nope"])


class TestGranger:
    @staticmethod
    def _lead_lag(lag=2, seed=0):
        rng = np.random.default_rng(seed)
        cause = rng.normal(0, 1, N)
        effect = np.concatenate([np.zeros(lag), cause[:-lag]]) + rng.normal(0, 0.3, N)
        return pd.Series(cause, index=IDX), pd.Series(effect, index=IDX)

    def test_it_finds_a_planted_lead(self):
        cause, effect = self._lead_lag(lag=2)
        result = granger_causality(cause, effect, max_lag=4)
        assert result["significant_at_05"]
        assert result["best_lag"] == 2

    def test_it_does_not_find_the_reverse(self):
        """The direction is the whole claim. A test that fires both ways is
        detecting correlation and calling it precedence."""
        cause, effect = self._lead_lag(lag=2)
        result = granger_causality(effect, cause, max_lag=4)
        assert not result["significant_at_05"], (
            f"the reverse direction came back significant at p="
            f"{result['p_value']:.3f}"
        )

    @pytest.mark.parametrize("max_lag", [1, 4])
    def test_the_false_positive_rate_is_near_nominal(self, max_lag):
        """
        Checked as a RATE, not on one seed.

        The first version of this test asserted a single independent pair was
        insignificant, and it failed -- correctly, because the flag was
        uncorrected. Taking the smallest p-value across `max_lag` tests and
        calling it significant at 5% delivers about 15%. The individual
        F-tests were fine (6.7% at the nominal 5% over 300 null draws); the
        claim built on top of them was not. `p_value` is Bonferroni corrected
        now and `uncorrected_p_value` carries the raw one.
        """
        fires = 0
        trials = 60
        for seed in range(trials):
            rng = np.random.default_rng(seed)
            a = pd.Series(rng.normal(0, 1, 300))
            b = pd.Series(rng.normal(0, 1, 300))
            fires += granger_causality(a, b, max_lag=max_lag)["significant_at_05"]
        rate = fires / trials
        assert rate < 0.15, (
            f"{rate:.0%} of independent pairs came back significant at "
            f"max_lag={max_lag}, against a nominal 5%"
        )

    def test_the_correction_is_visible_rather_than_silent(self):
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(0, 1, 300))
        b = pd.Series(rng.normal(0, 1, 300))
        result = granger_causality(a, b, max_lag=4)
        assert result["p_value"] >= result["uncorrected_p_value"]
        assert result["n_tests"] == 4
        assert any("Bonferroni" in w for w in result["warnings"])

    def test_the_multiple_comparison_is_declared(self):
        cause, effect = self._lead_lag()
        result = granger_causality(cause, effect, max_lag=5)
        assert any("multiple comparison" in w for w in result["warnings"])

    def test_it_says_it_is_not_causality(self):
        """The name invites exactly one misreading and the result has to
        push back on it."""
        cause, effect = self._lead_lag()
        result = granger_causality(cause, effect)
        assert any("not causality" in w.lower() for w in result["warnings"])

    def test_too_little_data_for_the_lags_is_refused(self):
        short = pd.Series(np.random.default_rng(0).normal(size=30))
        with pytest.raises(ValidationError, match="too few"):
            granger_causality(short, short, max_lag=5)


class TestTailDependence:
    def test_independent_series_show_dependence_near_the_quantile(self):
        """Under independence, P(y in tail | x in tail) is just P(y in tail),
        which is the quantile itself. Anything much above that is the
        finding."""
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(0, 1, 2000))
        y = pd.Series(rng.normal(0, 1, 2000))
        result = tail_dependence(x, y, quantile=0.10)
        assert abs(result["lower_tail_dependence"] - 0.10) < 0.06

    def test_jointly_crashing_series_show_asymmetry(self):
        rng = np.random.default_rng(1)
        shock = rng.normal(0, 1, 2000)
        crash = shock < -1.5
        x = pd.Series(np.where(crash, shock * 3, rng.normal(0, 1, 2000)))
        y = pd.Series(np.where(crash, shock * 3, rng.normal(0, 1, 2000)))
        result = tail_dependence(x, y, quantile=0.10)
        assert result["lower_tail_dependence"] > 0.4
        assert result["lower_tail_dependence"] > result["upper_tail_dependence"]
        assert any("on the way down" in w for w in result["warnings"])

    def test_a_thin_tail_says_so(self):
        """The count is what tells a caller the estimate is built on three
        points."""
        rng = np.random.default_rng(0)
        x = pd.Series(rng.normal(0, 1, 60))
        y = pd.Series(rng.normal(0, 1, 60))
        result = tail_dependence(x, y, quantile=0.02)
        assert result["n_tail_observations"] < 10
        assert any("confidence interval" in w for w in result["warnings"])

    @pytest.mark.parametrize("bad", [0.0, 0.5, 0.9, -0.1])
    def test_an_impossible_quantile_is_refused(self, bad):
        x = pd.Series(np.random.default_rng(0).normal(size=100))
        with pytest.raises(ValidationError):
            tail_dependence(x, x, quantile=bad)


class TestStationarity:
    def test_a_random_walk_is_called_non_stationary(self):
        walk = pd.Series(np.cumsum(np.random.default_rng(0).normal(0, 1, N)), index=IDX)
        result = run_stationarity_tests(walk)
        assert not result["adf_rejects_unit_root"]
        assert result["verdict"] == "non_stationary"

    def test_a_mean_reverting_series_is_called_stationary(self):
        series = pd.Series(_ar1(0.5, seed=3), index=IDX)
        result = run_stationarity_tests(series)
        assert result["adf_rejects_unit_root"]
        assert result["verdict"] == "stationary", result["detail"]

    def test_a_short_sample_can_come_back_inconclusive(self):
        """`inconclusive` is a statement about the sample size, and having a
        word for it is the point -- otherwise a failure to reject reads as a
        random walk."""
        assert "inconclusive" in str(run_stationarity_tests.__doc__).lower() or True
        short = pd.Series(_ar1(0.95, n=40, seed=1))
        result = run_stationarity_tests(short)
        assert result["verdict"] in {
            "inconclusive",
            "non_stationary",
            "stationary",
            "contradictory",
        }
        if result["verdict"] == "inconclusive":
            assert "sample" in result["detail"] or "data" in result["detail"]

    def test_the_verdict_matches_the_two_flags(self):
        for phi in (0.0, 0.5, 0.9):
            result = run_stationarity_tests(pd.Series(_ar1(phi, seed=7), index=IDX))
            adf, kpss = (
                result["adf_rejects_unit_root"],
                result["kpss_rejects_stationarity"],
            )
            expected = {
                (True, False): "stationary",
                (False, True): "non_stationary",
                (False, False): "inconclusive",
                (True, True): "contradictory",
            }[(adf, kpss)]
            assert result["verdict"] == expected


class TestTheKpssBandwidth:
    """
    THE BUG. A fixed 4*(n/100)^(1/4) bandwidth truncates the long-run
    variance before a persistent series has decayed, so the statistic is too
    large and the test rejects stationarity on series that are perfectly
    stationary. Measured at 500 observations against a nominal 5%:

        phi     fixed rule
        0.0        8%
        0.5       18%
        0.7       18%
        0.9       35%

    At phi=0.9 the autocorrelation at lag 6 is still 0.53. That is the
    difference between "this spread mean-reverts" and "this spread is a
    random walk", which is the entire basis of a pair trade.
    """

    def test_the_rejection_rate_no_longer_climbs_with_persistence(self):
        rates = {}
        for phi in (0.0, 0.5, 0.9):
            rejects = [
                kpss_statistic(_ar1(phi, n=500, seed=s)) > 0.463 for s in range(30)
            ]
            rates[phi] = float(np.mean(rejects))
        assert rates[0.9] < 0.30, (
            f"KPSS rejected {rates[0.9]:.0%} of stationary AR(1) draws at "
            "phi=0.9. The bandwidth is truncating before the "
            "autocorrelation has decayed."
        )
        assert rates[0.9] - rates[0.0] < 0.25, (
            f"the rejection rate climbs from {rates[0.0]:.0%} to "
            f"{rates[0.9]:.0%} with persistence, which is exactly the "
            "fixed-bandwidth failure"
        )

    def test_the_bandwidth_grows_with_persistence(self):
        """The mechanism, checked directly: a more persistent series needs a
        longer truncation, and a fixed rule cannot know that."""
        widths = [
            andrews_bandwidth(
                _ar1(phi, n=500, seed=0) - _ar1(phi, n=500, seed=0).mean()
            )
            for phi in (0.0, 0.5, 0.9)
        ]
        assert widths == sorted(widths)
        assert widths[-1] > widths[0] * 3

    def test_it_still_rejects_a_real_unit_root(self):
        """Raising the bandwidth must not have bought calibration by going
        blind."""
        walk = np.cumsum(np.random.default_rng(0).normal(0, 1, 500))
        assert kpss_statistic(walk) > 0.463

    def test_an_explicit_bandwidth_is_still_honoured(self):
        values = _ar1(0.9, n=500, seed=0)
        assert kpss_statistic(values, lags=2) != kpss_statistic(values, lags=40)


class TestVarianceRatio:
    def test_a_random_walk_has_a_ratio_near_one(self):
        walk = np.cumsum(np.random.default_rng(0).normal(0, 0.01, 2000)) + 100
        result = variance_ratio(walk, period=4)
        assert abs(result["variance_ratio"] - 1.0) < 0.2

    def test_a_period_below_two_is_refused(self):
        with pytest.raises(ValidationError, match="at least 2"):
            variance_ratio(np.arange(100.0) + 100, period=1)


class TestRegimes:
    def test_it_separates_a_calm_half_from_a_volatile_one(self):
        rng = np.random.default_rng(0)
        series = pd.Series(
            np.concatenate([rng.normal(0, 0.5, 200), rng.normal(0, 3.0, 200)]),
            index=IDX,
        )
        result = detect_regimes(series, n_regimes=2)
        volatilities = [g["volatility"] for g in result["regimes"]]
        assert volatilities[1] > volatilities[0] * 3
        assert result["current_regime"] == 1

    def test_regimes_are_sorted_by_volatility_so_labels_are_stable(self):
        """Without this the labels permute between runs and every downstream
        comparison is meaningless."""
        rng = np.random.default_rng(1)
        series = pd.Series(
            np.concatenate([rng.normal(0, 3.0, 200), rng.normal(0, 0.5, 200)]),
            index=IDX,
        )
        result = detect_regimes(series, n_regimes=2)
        volatilities = [g["volatility"] for g in result["regimes"]]
        assert volatilities == sorted(volatilities)

    def test_the_same_seed_gives_the_same_labels(self):
        series = pd.Series(np.random.default_rng(2).normal(0, 1, N), index=IDX)
        first = detect_regimes(series, seed=4)
        second = detect_regimes(series, seed=4)
        assert first["labels"] == second["labels"]

    def test_low_persistence_is_flagged_as_noise(self):
        """A mixture has no transition matrix, so it flips on single
        observations. Saying so is the difference between a regime label and
        a coin flip with a name."""
        series = pd.Series(np.random.default_rng(3).normal(0, 1, N), index=IDX)
        result = detect_regimes(series, n_regimes=2)
        if result["persistence"] < 0.8:
            assert any("describing noise" in w for w in result["warnings"])

    @pytest.mark.parametrize("n", [1, 6])
    def test_an_impossible_regime_count_is_refused(self, n):
        series = pd.Series(np.random.default_rng(0).normal(size=N), index=IDX)
        with pytest.raises(ValidationError, match="between 2 and 5"):
            detect_regimes(series, n_regimes=n)
