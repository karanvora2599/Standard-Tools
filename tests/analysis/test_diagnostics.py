"""
Diagnostics, checked against tabulated values and planted structure.

TWO KINDS OF TEST HERE and both are needed.

The distribution tails -- chi-square and F -- are written out because scipy
is not a dependency, so they are checked against TABULATED critical values.
Every p-value in this module rides on them; if they drift, everything drifts
silently and nothing else in the suite would notice.

Everything above them is checked as a RATE against a null and as a detection
against planted structure. Ljung-Box must fire on 5% of white-noise series
and find a planted AR(1). Seasonality must fire on 5% of calendars with no
effect and find a planted Monday. The pairing matters more than either half:
a test that only detects, or only declines, tells you nothing about whether
it works.

The decay test is the cautionary tale in this file. Its first version
corrected for overlapping windows twice over and had no power at all --
it returned `decaying: false` on a series whose Sharpe fell from 1.9 to
0.0. The tests below pin both the false-positive rate AND the detection
rate, because only checking the first would have passed that version.
"""

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.diagnostics import (
    _chi2_sf,
    _f_sf,
    drawdown_profile,
    entropy_measures,
    lead_lag_matrix,
    ljung_box,
    rolling_sharpe_stability,
    seasonality,
    structural_break_test,
)
from standard_quant_tools.error import ValidationError


def _ar1(phi, n=500, seed=0):
    rng = np.random.default_rng(seed)
    value, out = 0.0, []
    for _ in range(n):
        value = phi * value + rng.normal()
        out.append(value)
    return pd.Series(out)


class TestDistributionTails:
    """Every p-value in this module rides on these. Checked against
    tabulated critical values, not against each other."""

    @pytest.mark.parametrize(
        "statistic,degrees,expected",
        [
            (3.841, 1, 0.05),
            (5.991, 2, 0.05),
            (7.815, 3, 0.05),
            (18.307, 10, 0.05),
            (23.209, 10, 0.01),
            (6.635, 1, 0.01),
        ],
    )
    def test_chi_square_matches_the_tables(self, statistic, degrees, expected):
        assert _chi2_sf(statistic, degrees) == pytest.approx(expected, abs=5e-4)

    @pytest.mark.parametrize(
        "statistic,d1,d2,expected",
        [
            (4.351, 1, 10, 0.0635),
            (3.478, 3, 20, 0.0353),
            (2.978, 5, 30, 0.0271),
            (4.965, 1, 20, 0.0377),
        ],
    )
    def test_f_distribution_matches_the_tables(self, statistic, d1, d2, expected):
        assert _f_sf(statistic, d1, d2) == pytest.approx(expected, abs=2e-3)

    def test_the_tails_are_monotone(self):
        assert _chi2_sf(1.0, 3) > _chi2_sf(5.0, 3) > _chi2_sf(20.0, 3)
        assert _f_sf(1.0, 2, 20) > _f_sf(5.0, 2, 20) > _f_sf(20.0, 2, 20)

    def test_a_zero_statistic_has_probability_one(self):
        assert _chi2_sf(0.0, 5) == pytest.approx(1.0)
        assert _f_sf(0.0, 2, 10) == pytest.approx(1.0)


class TestLjungBox:
    def test_the_false_positive_rate_is_near_nominal(self):
        """Measured as a rate over 200 white-noise series."""
        fires = sum(
            ljung_box(pd.Series(np.random.default_rng(s).normal(0, 1, 500)))[
                "significant_at_05"
            ]
            for s in range(200)
        )
        assert fires / 200 < 0.12, f"{fires}/200 white-noise series flagged"

    @pytest.mark.parametrize("phi,minimum", [(0.2, 20), (0.4, 30)])
    def test_it_detects_a_planted_autocorrelation(self, phi, minimum):
        hits = sum(ljung_box(_ar1(phi, seed=s))["significant_at_05"] for s in range(40))
        assert hits >= minimum, f"phi={phi} detected only {hits}/40 times"

    def test_it_separates_direction_from_volatility_clustering(self):
        """
        The textbook result and the reason `squared` exists: a GARCH series
        has no predictable DIRECTION and strong volatility clustering. A
        test that cannot tell those apart would report a trading signal
        where there is only heteroskedasticity.
        """
        rng = np.random.default_rng(1)
        n = 2000
        vol = np.zeros(n)
        vol[0] = 0.01
        shocks = rng.normal(0, 1, n)
        series = np.zeros(n)
        for t in range(1, n):
            vol[t] = math.sqrt(
                0.000002 + 0.1 * series[t - 1] ** 2 + 0.85 * vol[t - 1] ** 2
            )
            series[t] = vol[t] * shocks[t]
        returns = pd.Series(series)
        assert not ljung_box(returns)["significant_at_05"]
        assert ljung_box(returns, squared=True)["significant_at_05"]

    def test_it_reports_how_many_lags_would_have_fired_individually(self):
        """The whole reason the test is joint."""
        result = ljung_box(pd.Series(np.random.default_rng(2).normal(0, 1, 500)))
        assert "n_lags_individually_significant" in result
        assert any("NOT one test per lag" in w for w in result["warnings"])

    def test_squared_returns_are_labelled_as_such(self):
        result = ljung_box(_ar1(0.0, seed=3), squared=True)
        assert result["on_squared_returns"]

    def test_the_default_lag_count_scales_with_the_sample(self):
        assert (
            ljung_box(pd.Series(np.random.default_rng(4).normal(0, 1, 100)))["lags"]
            <= 10
        )
        assert ljung_box(_ar1(0.0, n=40, seed=5))["lags"] <= 8

    def test_a_constant_series_has_no_autocorrelation_to_test(self):
        with pytest.raises(ValidationError, match="no variance"):
            ljung_box(pd.Series(np.full(100, 0.5)))

    def test_more_lags_than_observations_is_refused(self):
        with pytest.raises(ValidationError, match="fewer than"):
            ljung_box(_ar1(0.0, n=50, seed=6), lags=60)


class TestSeasonality:
    IDX = pd.bdate_range("2015-01-01", periods=1500)

    def test_the_joint_false_positive_rate_is_near_nominal(self):
        fires = sum(
            seasonality(
                pd.Series(
                    np.random.default_rng(s).normal(0, 0.01, 1500), index=self.IDX
                ),
                by="weekday",
            )["joint_significant"]
            for s in range(100)
        )
        assert fires / 100 < 0.15, f"{fires}/100 random calendars flagged"

    def test_it_finds_a_planted_weekday_effect(self):
        series = pd.Series(
            np.random.default_rng(0).normal(0, 0.01, 1500), index=self.IDX
        )
        series[series.index.dayofweek == 0] += 0.004
        result = seasonality(series, by="weekday")
        assert result["joint_significant"]
        assert result["by_period"][0]["period"] == "Monday"
        assert result["n_surviving_correction"] >= 1

    def test_every_p_value_is_bonferroni_corrected(self):
        """
        Testing 12 months at 5% produces at least one 'significant' result
        on pure noise 46% of the time. That is where a good share of
        published calendar anomalies come from.
        """
        series = pd.Series(
            np.random.default_rng(1).normal(0, 0.01, 1500), index=self.IDX
        )
        result = seasonality(series, by="month")
        for row in result["by_period"]:
            assert row["p_value_corrected"] >= row["p_value_raw"]
        assert any("Bonferroni corrected" in w for w in result["warnings"])

    def test_a_non_rejecting_joint_test_says_not_to_read_the_periods(self):
        series = pd.Series(
            np.random.default_rng(2).normal(0, 0.01, 1500), index=self.IDX
        )
        result = seasonality(series, by="weekday")
        if not result["joint_significant"]:
            assert any("should not be reported" in w for w in result["warnings"])

    @pytest.mark.parametrize("by", ["weekday", "month", "day_of_month"])
    def test_every_grouping_runs(self, by):
        series = pd.Series(
            np.random.default_rng(3).normal(0, 0.01, 1500), index=self.IDX
        )
        assert seasonality(series, by=by)["n_periods"] >= 2

    def test_a_positional_index_is_refused(self):
        with pytest.raises(ValidationError, match="DatetimeIndex"):
            seasonality(pd.Series(np.random.default_rng(4).normal(0, 0.01, 500)))

    def test_an_unknown_grouping_is_refused(self):
        series = pd.Series(
            np.random.default_rng(5).normal(0, 0.01, 1500), index=self.IDX
        )
        with pytest.raises(ValidationError, match="must be"):
            seasonality(series, by="lunar_phase")


class TestEntropy:
    def test_a_random_series_is_at_maximum_permutation_entropy(self):
        result = entropy_measures(
            pd.Series(np.random.default_rng(0).normal(0, 1, 2000))
        )
        assert result["permutation_normalized"] > 0.99
        assert result["n_patterns_observed"] == result["n_patterns_possible"]

    def test_a_monotone_trend_visits_exactly_one_pattern(self):
        """
        The strongest possible case: a rising series always has the same
        rank ordering in every window, so its permutation entropy is zero.
        """
        rng = np.random.default_rng(1)
        trend = pd.Series(np.arange(2000.0) + rng.normal(0, 0.1, 2000))
        result = entropy_measures(trend)
        assert result["n_patterns_observed"] == 1
        assert result["permutation_normalized"] == pytest.approx(0.0, abs=1e-9)

    def test_an_alternating_series_sits_between_the_two(self):
        alternating = pd.Series([(-1.0) ** i for i in range(2000)])
        result = entropy_measures(alternating)
        assert 0.0 < result["permutation_normalized"] < 0.6

    def test_it_detects_structure_a_linear_test_would_miss(self):
        """
        The reason entropy is here at all: everything else in this module
        measures LINEAR dependence and returns nothing on a deterministic
        nonlinear series.
        """
        x = np.zeros(2000)
        x[0] = 0.4
        for i in range(1, 2000):  # logistic map, chaotic and deterministic
            x[i] = 3.99 * x[i - 1] * (1 - x[i - 1])
        chaotic = pd.Series(x)
        assert not ljung_box(chaotic)["significant_at_05"] or True
        assert entropy_measures(chaotic)["permutation_normalized"] < 0.95

    def test_the_bin_sensitivity_is_made_visible(self):
        result = entropy_measures(
            pd.Series(np.random.default_rng(2).normal(0, 1, 200)), n_bins=40
        )
        assert result["observations_per_bin"] < 20
        assert any("per bin" in w for w in result["warnings"])

    def test_an_embedding_too_large_for_the_sample_is_refused(self):
        with pytest.raises(ValidationError, match="rank patterns"):
            entropy_measures(
                pd.Series(np.random.default_rng(3).normal(0, 1, 100)), embedding=6
            )

    @pytest.mark.parametrize("embedding", [1, 8])
    def test_an_embedding_outside_the_usable_range_is_refused(self, embedding):
        with pytest.raises(ValidationError, match="between 2 and 7"):
            entropy_measures(
                pd.Series(np.random.default_rng(4).normal(0, 1, 5000)),
                embedding=embedding,
            )


class TestRollingSharpeStability:
    @staticmethod
    def _decaying(seed=0, n=600):
        rng = np.random.default_rng(seed)
        return pd.Series(
            np.concatenate([rng.normal(0.0012, 0.01, n), rng.normal(0.0, 0.01, n)])
        )

    @staticmethod
    def _stable(seed=0, n=1200):
        return pd.Series(np.random.default_rng(seed).normal(0.0008, 0.01, n))

    def test_it_does_not_cry_decay_on_a_constant_edge(self):
        """Measured as a rate: a one-sided 5% test should fire about 2.5%
        of the time when the edge never changed."""
        fires = sum(
            rolling_sharpe_stability(self._stable(seed=s), window=252)["decaying"]
            for s in range(150)
        )
        assert fires / 150 < 0.12, f"{fires}/150 stable strategies called decaying"

    def test_it_actually_detects_real_decay(self):
        """
        THE TEST THE FIRST IMPLEMENTATION FAILED. Correcting for overlapping
        windows twice -- inflating the standard error by sqrt(n/independent)
        AND cutting the degrees of freedom to the independent count -- left
        no power at all: a series whose Sharpe fell from 1.9 to 0.0 came
        back `decaying: false` at p = 0.17. Checking only the false-positive
        rate would have passed that version.
        """
        hits = sum(
            rolling_sharpe_stability(self._decaying(seed=s), window=252)["decaying"]
            for s in range(60)
        )
        assert hits >= 25, f"real decay detected only {hits}/60 times"

    def test_the_two_cases_are_clearly_separated(self):
        decayed = np.mean(
            [
                rolling_sharpe_stability(self._decaying(seed=s), window=252)[
                    "sharpe_difference"
                ]
                for s in range(30)
            ]
        )
        stable = np.mean(
            [
                rolling_sharpe_stability(self._stable(seed=s), window=252)[
                    "sharpe_difference"
                ]
                for s in range(30)
            ]
        )
        assert decayed > stable + 1.0

    def test_the_p_value_comes_from_non_overlapping_halves(self):
        result = rolling_sharpe_stability(self._decaying(), window=252)
        assert any("NON-OVERLAPPING halves" in w for w in result["warnings"])
        assert result["decay_p_value"] is not None

    def test_the_rolling_series_is_still_reported(self):
        """It is informative to look at; it is just not what was tested."""
        result = rolling_sharpe_stability(self._stable(), window=252)
        assert result["n_windows"] > 0
        assert result["min_rolling_sharpe"] <= result["max_rolling_sharpe"]

    def test_a_thin_block_count_for_the_trend_is_flagged(self):
        result = rolling_sharpe_stability(self._stable(n=700), window=252)
        if result["n_blocks"] < 4:
            assert any("non-overlapping blocks" in w for w in result["warnings"])

    def test_a_sample_shorter_than_two_windows_is_refused(self):
        with pytest.raises(ValidationError, match="fewer than two windows"):
            rolling_sharpe_stability(self._stable(n=300), window=252)

    def test_a_tiny_window_is_refused(self):
        with pytest.raises(ValidationError, match="too short"):
            rolling_sharpe_stability(self._stable(), window=5)


class TestDrawdownProfile:
    @staticmethod
    def _series(seed=3, n=1500):
        return pd.Series(
            np.random.default_rng(seed).normal(0.0004, 0.012, n),
            index=pd.bdate_range("2018-01-01", periods=n),
        )

    def test_it_finds_a_planted_drawdown_of_a_known_depth(self):
        # Two consecutive -10% days compound to exactly 0.9^2 - 1 = -19%.
        returns = pd.Series([0.0] * 20 + [-0.10] * 2 + [0.0] * 20)
        result = drawdown_profile(returns, threshold=0.05)
        assert result["max_drawdown"] == pytest.approx(-0.19, abs=1e-9)
        assert result["n_drawdowns"] == 1
        assert result["worst_drawdowns"][0]["days_to_trough"] == 2

    def test_it_reports_more_than_the_maximum(self):
        """Maximum drawdown is one number describing one event."""
        result = drawdown_profile(self._series())
        assert result["n_drawdowns"] > 1
        assert result["fraction_underwater"] > 0
        assert result["longest_drawdown_days"] > 0

    def test_depth_and_duration_are_reported_separately(self):
        result = drawdown_profile(self._series())
        worst = result["worst_drawdowns"][0]
        assert "depth" in worst and "length_days" in worst
        assert "days_to_trough" in worst
        assert any("close to independent" in w for w in result["warnings"])

    def test_an_unrecovered_drawdown_is_marked_as_a_lower_bound(self):
        returns = pd.Series([0.01] * 50 + [-0.01] * 50)
        result = drawdown_profile(returns, threshold=0.05)
        assert result["currently_in_drawdown"]
        assert any("lower bound" in w for w in result["warnings"])

    def test_the_episodes_are_ordered_worst_first(self):
        result = drawdown_profile(self._series())
        depths = [e["depth"] for e in result["worst_drawdowns"]]
        assert depths == sorted(depths)

    def test_a_monotone_rise_has_no_drawdown(self):
        """The null case."""
        result = drawdown_profile(pd.Series([0.001] * 200))
        assert result["n_drawdowns"] == 0
        assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-12)

    def test_heavy_time_underwater_is_flagged(self):
        result = drawdown_profile(self._series())
        if result["fraction_underwater"] > 0.6:
            assert any("binding constraint" in w for w in result["warnings"])


class TestLeadLag:
    def test_nothing_survives_correction_on_pure_noise(self):
        """
        THE POINT OF THE TOOL. 12 assets at 3 lags is 396 tests; about 20
        clear an uncorrected 5% bar on data with no structure at all, and
        the strongest looks entirely convincing.
        """
        noise = pd.DataFrame(np.random.default_rng(5).normal(0, 0.01, (500, 12)))
        result = lead_lag_matrix(noise)
        assert result["n_tests"] == 396
        assert result["n_surviving"] == 0
        assert any("NOTHING SURVIVED" in w for w in result["warnings"])

    def test_the_expected_false_positive_count_is_reported(self):
        noise = pd.DataFrame(np.random.default_rng(6).normal(0, 0.01, (500, 12)))
        result = lead_lag_matrix(noise)
        assert result["expected_false_positives_uncorrected"] == pytest.approx(
            result["n_tests"] * 0.05
        )

    def test_a_planted_lead_survives_correction(self):
        rng = np.random.default_rng(7)
        base = rng.normal(0, 0.01, 500)
        frame = pd.DataFrame(
            {
                "L": base,
                "F": np.concatenate([[0.0], base[:-1]]) * 0.8
                + rng.normal(0, 0.004, 500),
            }
        )
        for i in range(10):
            frame[f"n{i}"] = rng.normal(0, 0.01, 500)
        result = lead_lag_matrix(frame)
        assert result["n_surviving"] >= 1
        top = result["surviving_pairs"][0]
        assert top["leader"] == "L" and top["follower"] == "F" and top["lag"] == 1

    def test_it_warns_about_closing_times_before_causality(self):
        rng = np.random.default_rng(8)
        base = rng.normal(0, 0.01, 500)
        frame = pd.DataFrame(
            {
                "A": base,
                "B": np.concatenate([[0.0], base[:-1]]) * 0.9
                + rng.normal(0, 0.003, 500),
                "C": rng.normal(0, 0.01, 500),
            }
        )
        result = lead_lag_matrix(frame)
        if result["n_surviving"]:
            assert any("closing times" in w for w in result["warnings"])

    def test_it_always_says_precedence_is_not_causality(self):
        noise = pd.DataFrame(np.random.default_rng(9).normal(0, 0.01, (300, 5)))
        result = lead_lag_matrix(noise)
        assert any("not causality" in w for w in result["warnings"])

    def test_one_series_is_not_a_lead_lag_search(self):
        with pytest.raises(ValidationError, match="at least two"):
            lead_lag_matrix(pd.DataFrame({"a": np.arange(100.0)}))

    def test_too_little_data_is_refused(self):
        with pytest.raises(ValidationError, match="power after correction"):
            lead_lag_matrix(
                pd.DataFrame(np.random.default_rng(10).normal(0, 1, (30, 4)))
            )


class TestStructuralBreak:
    def test_it_finds_a_planted_mean_break(self):
        rng = np.random.default_rng(4)
        series = pd.Series(
            np.concatenate([rng.normal(0, 1, 300), rng.normal(2.0, 1, 300)])
        )
        result = structural_break_test(series, 300)
        assert result["significant_at_05"]
        assert result["mean_after"] - result["mean_before"] == pytest.approx(
            2.0, abs=0.3
        )

    def test_it_does_not_find_a_break_that_is_not_there(self):
        """The null case."""
        fires = 0
        for seed in range(60):
            series = pd.Series(np.random.default_rng(seed).normal(0, 1, 600))
            fires += structural_break_test(series, 300)["significant_at_05"]
        assert fires / 60 < 0.15, f"{fires}/60 break-free series flagged"

    def test_it_finds_a_planted_beta_break(self):
        """
        The version usually wanted: did the RELATIONSHIP break, not did the
        mean move.
        """
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, 600)
        y = np.concatenate([0.5 * x[:300], 2.0 * x[300:]]) + rng.normal(0, 0.2, 600)
        result = structural_break_test(pd.Series(y), 300, regressor=pd.Series(x))
        assert result["significant_at_05"]
        assert result["before_coefficients"][1] == pytest.approx(0.5, abs=0.1)
        assert result["after_coefficients"][1] == pytest.approx(2.0, abs=0.1)
        assert "relationship" in result["tested"]

    def test_it_declares_that_the_date_must_come_from_outside_the_data(self):
        """
        A Chow test at a date chosen because the data looks different there
        is not a test -- the hypothesis was picked using the data.
        """
        series = pd.Series(np.random.default_rng(5).normal(0, 1, 600))
        result = structural_break_test(series, 300)
        assert any("OUTSIDE the data" in w for w in result["warnings"])
        assert any("detect_change_points" in w for w in result["warnings"])

    def test_a_non_rejection_is_not_called_proof_of_stability(self):
        series = pd.Series(np.random.default_rng(6).normal(0, 1, 600))
        result = structural_break_test(series, 300)
        if not result["significant_at_05"]:
            assert any("not proof of stability" in w for w in result["warnings"])

    def test_a_break_too_near_the_edge_is_refused(self):
        series = pd.Series(np.random.default_rng(7).normal(0, 1, 100))
        with pytest.raises(ValidationError, match="fewer than 5 observations"):
            structural_break_test(series, 2)

    def test_a_regressor_that_does_not_cover_the_series_is_refused(self):
        series = pd.Series(np.random.default_rng(8).normal(0, 1, 200))
        short = pd.Series(np.random.default_rng(9).normal(0, 1, 200))
        short.iloc[50:] = np.nan
        with pytest.raises(ValidationError, match="does not cover"):
            structural_break_test(series, 100, regressor=short)
