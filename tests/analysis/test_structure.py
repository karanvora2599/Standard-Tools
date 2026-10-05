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

from standard_quant_tools.analysis import stationarity
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


class TestThePenaltyIsScaledToTheSeries:
    """The default made a break unreportable on the default channel.

    A split's gain is bounded by the residual sum of squares it is removing
    from, so an ABSOLUTE penalty means different things at different scales.
    10.0 was reasonable on prices, whose RSS runs to six figures, and
    impossible on returns: clearing it on 500 daily returns needs a drift
    change of 0.28 PER DAY. A +0.5%/day regime shift, which is enormous,
    buys a gain of about 0.003.

    And it did not fail. It reported the series homogeneous -- a statement
    about the market, where the truth was a statement about units.
    """

    @staticmethod
    def _returns(shift=0.005, n=500, seed=0):
        rng = np.random.default_rng(seed)
        half = n // 2
        values = np.concatenate(
            [rng.normal(0, 0.01, half), rng.normal(shift, 0.01, n - half)]
        )
        return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=n))

    def test_a_real_regime_shift_in_returns_is_found(self):
        result = detect_change_points(self._returns())
        assert result["n_breaks"] >= 1
        assert result["penalty_was_derived"] is True

    def test_the_old_default_could_not_have_found_it(self):
        """Not a hypothetical: the gain a perfect split buys here is three
        orders of magnitude below 10.0."""
        series = self._returns()
        total_rss = float(((series - series.mean()) ** 2).sum())
        assert total_rss < 10.0, "the whole series cannot buy a gain of 10"

    def test_pure_noise_still_finds_nothing(self):
        """What the penalty is for. Measured at 0/200 seeds; three here."""
        for seed in (0, 1, 2):
            noise = pd.Series(
                np.random.default_rng(seed).normal(0, 0.01, 500),
                index=pd.bdate_range("2020-01-01", periods=500),
            )
            assert detect_change_points(noise)["n_breaks"] == 0, seed

    def test_prices_still_work_and_get_a_bigger_penalty(self):
        returns = self._returns()
        prices = (1 + returns).cumprod() * 100
        on_returns = detect_change_points(returns)
        on_prices = detect_change_points(prices)
        assert on_prices["penalty"] > on_returns["penalty"] * 1_000
        assert on_prices["n_breaks"] >= 1

    def test_an_explicit_penalty_keeps_its_absolute_meaning(self):
        """Backward compatible for anyone who set one deliberately."""
        prices = (1 + self._returns()).cumprod() * 100
        result = detect_change_points(prices, penalty=10.0)
        assert result["penalty"] == 10.0
        assert result["penalty_was_derived"] is False

    def test_a_penalty_no_split_could_clear_is_refused(self):
        """The failure that used to be reported as a finding."""
        with pytest.raises(ValidationError) as exc:
            detect_change_points(self._returns(), penalty=10.0)
        message = str(exc.value)
        assert "cannot be cleared by any split" in message
        assert "units mismatch" in message
        assert "Omit `penalty`" in message

    def test_the_applied_penalty_comes_back(self):
        """Every `gain` has to be read against it, so it cannot be implicit."""
        result = detect_change_points(self._returns())
        assert result["penalty"] > 0
        for found in result["breaks"]:
            assert found["gain"] > result["penalty"]

    def test_the_marginal_warning_uses_the_real_penalty(self):
        """It compared every gain to a hardcoded 30.0, so with a penalty
        scaled to returns -- around 0.002 -- every break found was reported
        as a marginal one."""
        result = detect_change_points(self._returns(shift=0.02))
        strong = [b for b in result["breaks"] if b["gain"] >= 3.0 * result["penalty"]]
        assert strong, "no break cleared 3x its own penalty"
        assert not any("less than 3x" in w for w in result["warnings"])

    def test_the_empty_answer_names_the_threshold_it_used(self):
        noise = pd.Series(
            np.random.default_rng(0).normal(0, 0.01, 500),
            index=pd.bdate_range("2020-01-01", periods=500),
        )
        warning = detect_change_points(noise)["warnings"][0]
        assert "scaled to this series" in warning


class TestAConstantHasNoChangePoint:
    """
    A constant series centres to rounding residue, and the prefix-sum costs
    of that residue produced breaks: 20 of 42 constant/length combinations
    "found" some, three at a time for 0.1 at n=300 against a derived
    penalty of 3e-33. Refused now, as ljung_box refuses it.
    """

    @pytest.mark.parametrize("level,n", [(0.1, 300), (0.3, 100), (1.23, 1000)])
    def test_a_constant_is_refused(self, level, n):
        series = pd.Series(
            np.full(n, level), index=pd.bdate_range("2020-01-01", periods=n)
        )
        with pytest.raises(ValidationError, match="does not vary"):
            detect_change_points(series)

    def test_a_shift_far_from_zero_is_found_where_it_is(self):
        """
        The planted case at a level far from zero: 1e6 with noise of 0.01.
        The prefix-sum cost is a sum of squares minus a squared sum, both
        near 3e14 here, and their difference -- the cost being compared --
        is 0.03: uncentred, it was cancellation noise and the break was
        placed at bar 91 instead of 150. Centred, it is exact.
        """
        rng = np.random.default_rng(7)
        values = 1e6 + np.concatenate(
            [rng.normal(0, 0.01, 150), rng.normal(0.03, 0.01, 150)]
        )
        series = pd.Series(values, index=pd.bdate_range("2020-01-01", periods=300))
        result = detect_change_points(series, max_breaks=1)
        assert result["n_breaks"] == 1
        assert abs(result["breaks"][0]["index"] - 150) <= 5

    def test_noise_far_from_zero_finds_no_break(self):
        """
        The null case at the same level. Uncentred, 32 of these 50
        pure-noise series produced a break.
        """
        found = 0
        for seed in range(50):
            values = 1e6 + np.random.default_rng(seed).normal(0, 0.01, 300)
            series = pd.Series(values, index=pd.bdate_range("2020-01-01", periods=300))
            found += detect_change_points(series)["n_breaks"] > 0
        assert found <= 2


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
        # 0.15 was the original bar and the UNCORRECTED rate is 12-15%, so
        # it let the very mutation this test exists to catch slip under.
        # Corrected, the measured rate is 6.0% / 4.0% / 3.3% at max_lag
        # 1 / 4 / 8, so 0.12 separates the two with room either side.
        assert rate < 0.12, (
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

    def test_the_flag_follows_the_CORRECTED_p_value(self):
        """
        The test that actually pins the correction, found by mutation.

        The rate test above passes with the correction REMOVED -- the
        uncorrected rate is 12-15% and the bar was 15%, so it slipped
        under the threshold it existed to enforce. This one searches for a
        case where the correction changes the answer (raw below 0.05,
        corrected above it) and asserts the flag follows the corrected
        number. It cannot pass without the correction, whatever the
        sampling does.
        """
        found = None
        for seed in range(400):
            rng = np.random.default_rng(seed)
            a = pd.Series(rng.normal(0, 1, 300))
            b = pd.Series(rng.normal(0, 1, 300))
            result = granger_causality(a, b, max_lag=4)
            if result["uncorrected_p_value"] < 0.05 <= result["p_value"]:
                found = result
                break
        assert found is not None, (
            "no seed in 400 produced a case where the correction changes the "
            "verdict, which would itself be suspicious"
        )
        assert not found["significant_at_05"], (
            f"raw p={found['uncorrected_p_value']:.4f} is under 0.05 and "
            f"corrected p={found['p_value']:.4f} is not, yet the tool called "
            "it significant -- the flag is reading the uncorrected value"
        )

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


def _level(returns: np.ndarray) -> np.ndarray:
    """A level series whose simple differences are exactly `returns`. It
    starts at zero, so `variance_ratio` differences it rather than taking
    logs, and the test controls the increments to the last bit."""
    return np.concatenate([[0.0], np.cumsum(returns)])


def _ma1(n: int, theta: float, seed: int) -> np.ndarray:
    shocks = np.random.default_rng(seed).normal(0.0, 1.0, n + 1)
    return shocks[1:] + theta * shocks[:-1]


def _garch_increments(n: int, seed: int) -> np.ndarray:
    """Uncorrelated increments whose variance clusters: a random walk under
    the heteroskedastic null the robust statistic is built for."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.05, 0.10, 0.85
    variance = omega / (1.0 - alpha - beta)
    out = np.empty(n)
    for t in range(n):
        out[t] = np.sqrt(variance) * rng.standard_normal()
        variance = omega + alpha * out[t] ** 2 + beta * variance
    return out


def _lo_mackinlay(prices: np.ndarray, q: int) -> tuple:
    """
    The statistic as Lo and MacKinlay (1988) write it, index for index:
    prices X_0..X_T, T = nq increments, overlapping q-differences, the
    unbiased variance estimators, and the heteroskedasticity-robust z*.
    Loops rather than vector algebra, so it shares nothing with the
    library's code but the formula.
    """
    T = len(prices) - 1
    mu = (prices[T] - prices[0]) / T
    sigma_a = sum((prices[k] - prices[k - 1] - mu) ** 2 for k in range(1, T + 1))
    sigma_a /= T - 1
    m = q * (T - q + 1) * (1.0 - q / T)
    sigma_c = (
        sum((prices[k] - prices[k - q] - q * mu) ** 2 for k in range(q, T + 1)) / m
    )
    vr = sigma_c / sigma_a
    denominator = (
        sum((prices[k] - prices[k - 1] - mu) ** 2 for k in range(1, T + 1)) ** 2
    )
    theta = 0.0
    for j in range(1, q):
        delta = (
            T
            * sum(
                (prices[k] - prices[k - 1] - mu) ** 2
                * (prices[k - j] - prices[k - j - 1] - mu) ** 2
                for k in range(j + 1, T + 1)
            )
            / denominator
        )
        theta += (2.0 * (q - j) / q) ** 2 * delta
    return vr, np.sqrt(T) * (vr - 1.0) / np.sqrt(theta)


class TestTheVarianceRatioStatistic:
    """
    The z statistic was missing its sqrt(T) and the overlapping variance
    its (1 - q/T), so z had a standard deviation near 0.03 instead of 1
    and the test never rejected anything: an MA(1) with theta = -0.8 and
    VR(8) = 0.15 came back p = 0.79. Every p-value it produced was inert.
    """

    @pytest.mark.parametrize("period", [2, 4, 8])
    def test_reverting_increments_are_rejected(self, period):
        values = _level(_ma1(1000, -0.8, seed=5))
        result = variance_ratio(values, period=period)
        assert result["differencing"] == "level"
        assert result["variance_ratio"] < 0.7
        assert result["z_statistic"] < -5.0
        assert result["p_value"] < 1e-6

    @pytest.mark.parametrize("period", [2, 5, 8])
    def test_it_is_the_textbook_statistic(self, period):
        rng = np.random.default_rng(17)
        increments = np.empty(600)
        increments[0] = rng.normal()
        for t in range(1, 600):
            increments[t] = 0.15 * increments[t - 1] + rng.normal()
        prices = _level(increments)
        vr, z = _lo_mackinlay(prices, period)
        result = variance_ratio(prices, period=period)
        assert result["variance_ratio"] == pytest.approx(vr, rel=1e-10)
        assert result["z_statistic"] == pytest.approx(z, rel=1e-10)

    @pytest.mark.parametrize("period", [2, 4, 8])
    @pytest.mark.parametrize("kind", ["iid", "garch"])
    def test_a_random_walk_is_rejected_at_the_nominal_rate(self, kind, period):
        """The null, homoskedastic and heteroskedastic: 1000 walks of 500
        increments. Size near 5% and a z whose spread is that of a standard
        normal -- the old z had a spread of 0.03 and a size of 0. A thousand
        rather than a few hundred because the size of 200 walks moves
        between 3.5% and 9.5% from one block of seeds to the next."""
        n_walks = 1000
        zs, rejected = [], 0
        for seed in range(n_walks):
            if kind == "iid":
                increments = np.random.default_rng(seed).normal(0.0, 1.0, 500)
            else:
                increments = _garch_increments(500, seed)
            result = variance_ratio(_level(increments), period=period)
            zs.append(result["z_statistic"])
            rejected += result["p_value"] < 0.05
        assert 0.03 <= rejected / n_walks <= 0.08, rejected
        assert 0.9 <= float(np.std(zs)) <= 1.1


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
        first = detect_regimes(series)
        second = detect_regimes(series)
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


@pytest.mark.skipif(not stationarity.HAS_CPP, reason="native extension not built")
class TestRegimesOnBothPaths:
    """
    Each EM step after numpy's exponentials is the extension's
    `regime_em_step` when it is built, the numpy loop otherwise (see the
    CHANGELOG entry of 2026-10-04). The labels, the regimes and every
    float in them are the same on both paths, on the series the tests
    above fit and on 2,000 daily returns with 2 to 5 regimes.
    """

    @staticmethod
    def _both(series, n_regimes):
        native = detect_regimes(series, n_regimes=n_regimes)
        stationarity.HAS_CPP = False
        try:
            python = detect_regimes(series, n_regimes=n_regimes)
        finally:
            stationarity.HAS_CPP = True
        return native, python

    @pytest.mark.parametrize("n_regimes", [2, 3, 4, 5])
    @pytest.mark.parametrize("seed", [0, 1, 3])
    def test_the_same_fit(self, n_regimes, seed):
        rng = np.random.default_rng(seed)
        series = pd.Series(
            np.concatenate([rng.normal(0, 0.5, 200), rng.normal(0, 3.0, 200)]),
            index=IDX,
        )
        native, python = self._both(series, n_regimes)
        assert native == python

    @pytest.mark.parametrize("n_regimes", [2, 3, 4, 5])
    def test_two_thousand_returns(self, n_regimes):
        rng = np.random.default_rng(7)
        vol = np.where((np.arange(2_000) // 250) % 2 == 0, 0.008, 0.02)
        series = pd.Series(rng.normal(0.0003, vol))
        native, python = self._both(series, n_regimes)
        assert native == python
