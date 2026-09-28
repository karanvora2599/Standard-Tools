"""
Liquidity estimators, tested against spreads that were PLANTED.

Every estimator here claims to recover a quantity from data that does not
contain it directly. The only way to know whether it does is to build a
series with a known answer and check. So each test below simulates an
efficient random walk, adds a bid-ask bounce of a specified size, and asks
the estimator what the spread was.

THE NULL CASES ARE THE IMPORTANT ONES and they are the reason several of
these tests exist at all. Roll's estimator returns a confident 10 bps on a
series with a spread of exactly zero -- the sampling noise in a lag-1
autocovariance swamps the signal, and taking a square root only when the
covariance lands negative discards the other half of that noise. A test
suite that only checked "does it find a planted 50 bps spread" would pass
and ship an estimator that hallucinates liquidity costs on every liquid
name in the universe.

So: for every estimator, one test that it finds what is there, and one that
it declines to find what is not.
"""

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.microstructure_estimators import (
    amihud_illiquidity,
    corwin_schultz_spread,
    estimate_vpin,
    intraday_volume_profile,
    kyle_lambda,
    order_flow_imbalance,
    roll_spread,
)
from standard_quant_tools.error import ValidationError


def bounce_series(n=2000, spread=0.5, sigma=0.01, price=100.0, seed=0):
    """An efficient random walk plus a bid-ask bounce of a KNOWN size."""
    rng = np.random.default_rng(seed)
    efficient = price * np.exp(np.cumsum(rng.normal(0, sigma, n)))
    side = rng.choice([-1.0, 1.0], n)
    return pd.Series(efficient + side * spread / 2.0)


def ohlc_frame(n=600, spread=0.01, sigma=0.015, seed=7):
    """Daily OHLC where the high/low straddle a planted proportional spread."""
    rng = np.random.default_rng(seed)
    rows, price = [], 100.0
    for _ in range(n):
        path = price * np.exp(np.cumsum(rng.normal(0, sigma / math.sqrt(20), 20)))
        rows.append((path.max() * (1 + spread / 2), path.min() * (1 - spread / 2)))
        price = path[-1]
    return pd.DataFrame(rows, columns=["high", "low"])


class TestRollSpread:
    def test_it_recovers_a_planted_spread_that_clears_the_noise_floor(self):
        result = roll_spread(bounce_series(spread=1.0, seed=1))
        assert result["significant"]
        assert result["spread_estimate"] == pytest.approx(1.0, rel=0.15)

    def test_it_refuses_to_call_a_zero_spread_series_illiquid(self):
        """
        THE TEST THAT MATTERS. On a random walk with no spread at all, the
        formula returns a confident-looking 0.098 on a $100 stock. Nothing
        in Roll's algebra reveals that -- it is sampling noise in the
        autocovariance, half of which is discarded by only taking a root
        when the covariance lands negative. Without the significance gate
        this estimator invents a liquidity cost for every liquid name.
        """
        result = roll_spread(bounce_series(spread=0.0, seed=1))
        assert result["significant"] is False
        assert result["spread_estimate"] < result["smallest_detectable_spread"]
        assert any("NOT DISTINGUISHABLE FROM ZERO" in w for w in result["warnings"])

    @pytest.mark.parametrize("spread", [0.0, 0.02, 0.10])
    def test_spreads_below_the_noise_floor_are_all_declared_unmeasurable(self, spread):
        result = roll_spread(bounce_series(spread=spread, seed=1))
        assert result["significant"] is False

    @pytest.mark.parametrize("spread", [0.5, 1.0, 2.0])
    def test_spreads_above_the_noise_floor_are_measured_accurately(self, spread):
        result = roll_spread(bounce_series(spread=spread, seed=1))
        assert result["significant"]
        assert result["spread_estimate"] == pytest.approx(spread, rel=0.15)

    def test_the_noise_floor_falls_as_the_sample_grows(self):
        """More data buys resolution, and the reported floor has to show it."""
        short = roll_spread(bounce_series(n=200, spread=0.0, seed=2))
        long = roll_spread(bounce_series(n=4000, spread=0.0, seed=2))
        assert long["smallest_detectable_spread"] < short["smallest_detectable_spread"]

    def test_a_trending_series_is_undefined_rather_than_zero(self):
        """
        The convention of substituting zero for a positive covariance biases
        every downstream average downward, and the zeros cluster in exactly
        the trending periods where liquidity is most interesting.
        """
        rng = np.random.default_rng(3)
        trending = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.004, 0.004, 500))))
        result = roll_spread(trending)
        assert result["spread_estimate"] is None
        assert result["serial_covariance"] > 0
        assert any("different facts" in w for w in result["warnings"])

    def test_a_rolling_window_reports_how_often_it_was_undefined(self):
        result = roll_spread(bounce_series(spread=1.0, seed=4), window=60)
        assert result["rolling"]["n_windows"] > 0
        assert result["rolling"]["n_undefined"] >= 0
        assert result["undefined_fraction"] == pytest.approx(
            result["rolling"]["n_undefined"] / result["rolling"]["n_windows"]
        )

    def test_a_high_undefined_fraction_warns_about_conditioning(self):
        rng = np.random.default_rng(5)
        trending = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.003, 0.005, 800))))
        result = roll_spread(trending, window=50)
        if result["undefined_fraction"] > 0.25:
            assert any("conditioned on" in w for w in result["warnings"])

    def test_the_spread_is_also_reported_in_basis_points(self):
        result = roll_spread(bounce_series(spread=1.0, price=100.0, seed=6))
        assert result["spread_bps"] == pytest.approx(
            result["spread_estimate"] / result["mean_price"] * 1e4, rel=1e-9
        )
        assert result["half_spread_bps"] == pytest.approx(
            result["spread_bps"] / 2, rel=1e-9
        )

    def test_too_little_data_is_refused(self):
        with pytest.raises(ValidationError, match="at least"):
            roll_spread(pd.Series([100.0, 101.0, 100.5]))

    def test_a_tiny_window_is_refused(self):
        with pytest.raises(ValidationError, match="too short"):
            roll_spread(bounce_series(), window=5)


class TestCorwinSchultz:
    def test_it_recovers_a_wide_planted_spread(self):
        result = corwin_schultz_spread(ohlc_frame(spread=0.01))
        assert result["spread_bps"] == pytest.approx(100.0, rel=0.25)

    def test_the_negative_fraction_flags_the_spreads_it_cannot_measure(self):
        """
        Measured: a 20 bps planted spread comes back as 56 bps with 44% of
        daily estimates negative, while a 100 bps spread comes back at 103
        bps with 29% negative. The negative fraction is what separates the
        two, so it has to be reported and the threshold has to sit between
        them.
        """
        narrow = corwin_schultz_spread(ohlc_frame(spread=0.002))
        wide = corwin_schultz_spread(ohlc_frame(spread=0.010))
        assert narrow["negative_fraction"] > wide["negative_fraction"]
        assert any("noise rather than" in w for w in narrow["warnings"])

    def test_a_wider_spread_produces_a_wider_estimate(self):
        estimates = [
            corwin_schultz_spread(ohlc_frame(spread=s))["spread_bps"]
            for s in (0.002, 0.005, 0.010, 0.020)
        ]
        assert estimates == sorted(estimates)

    def test_the_raw_mean_is_returned_alongside_the_floored_one(self):
        """Flooring negatives at zero is Corwin-Schultz's own recommendation
        and it turns a symmetric error into a one-sided bias. Both numbers
        have to be visible."""
        result = corwin_schultz_spread(ohlc_frame(spread=0.002))
        assert result["raw_mean_bps"] < result["spread_bps"]

    def test_a_high_below_its_low_is_refused(self):
        frame = ohlc_frame(n=50)
        frame.loc[10, "high"] = frame.loc[10, "low"] - 1.0
        with pytest.raises(ValidationError, match="high below its low"):
            corwin_schultz_spread(frame)

    def test_a_missing_column_names_what_it_wanted(self):
        with pytest.raises(ValidationError, match="low"):
            corwin_schultz_spread(pd.DataFrame({"high": np.arange(50.0)}))

    @staticmethod
    def _gapping_frame(jump=0.03, every=20, n=400, seed=11):
        """A name that gaps UP hard every twentieth bar and drifts between."""
        rng = np.random.default_rng(seed)
        price, rows = 100.0, []
        for i in range(n):
            if i % every == 0 and i:
                price *= 1 + jump
            path = price * np.exp(np.cumsum(rng.normal(0, 0.012 / math.sqrt(20), 20)))
            rows.append((path.max() * 1.0005, path.min() * 0.9995))
            price = path[-1]
        return pd.DataFrame(rows, columns=["high", "low"])

    def test_the_overnight_gap_adjustment_is_applied_and_counted(self):
        """
        The docstring claimed this adjustment for a long time before the
        code did it -- the two-day range was a plain max/min, which does not
        remove a gap. Measured on a name gapping 3% every twentieth bar: 31
        of 399 pairs gap, and removing the gap moves the raw mean from
        -39.795562 to -21.852966 bps.
        """
        result = corwin_schultz_spread(self._gapping_frame())

        assert result["n_gap_adjusted"] == 31
        assert result["raw_mean_bps"] == pytest.approx(-21.852966, abs=1e-4)
        assert any("GAPPED" in w for w in result["warnings"])

    def test_a_continuous_name_is_not_gap_adjusted_at_all(self):
        result = corwin_schultz_spread(ohlc_frame(spread=0.010))

        assert result["n_gap_adjusted"] == 0
        assert not any("GAPPED" in w for w in result["warnings"])

    def test_the_gap_correction_hides_in_the_floor_not_in_the_headline(self):
        """
        WHY THE BUG SURVIVED. A gap large enough to matter drives that
        pair's estimate deeply negative (-984 bps on the worst pair here),
        and Corwin-Schultz's own zero-floor then swallows the whole
        correction. spread_bps is unchanged to six decimal places while
        raw_mean_bps moves by 18 bps, so the headline number could not have
        revealed that the adjustment was missing.
        """
        result = corwin_schultz_spread(self._gapping_frame())

        assert result["spread_bps"] == pytest.approx(37.540011, abs=1e-4)
        assert result["raw_mean_bps"] < result["spread_bps"] - 50.0

    def test_the_direction_of_the_gap_bias_is_downward(self):
        """
        The docstring also had the SIGN backwards: it said gaps bias the
        estimate up. The inflated two-day range enters gamma, and gamma is
        SUBTRACTED from alpha, so an unadjusted gap biases it down. Gapping
        the same underlying harder has to push the raw estimate lower.
        """
        mild = corwin_schultz_spread(self._gapping_frame(jump=0.005))
        harsh = corwin_schultz_spread(self._gapping_frame(jump=0.06))

        assert harsh["n_gap_adjusted"] > mild["n_gap_adjusted"]
        assert harsh["raw_mean_bps"] < mild["raw_mean_bps"]


class TestAmihud:
    @staticmethod
    def _frame(sigma, volume, price=100.0, n=400, seed=5):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "close": price * np.exp(np.cumsum(rng.normal(0, sigma, n))),
                "volume": rng.uniform(volume * 0.8, volume * 1.2, n),
            }
        )

    def test_an_illiquid_name_scores_far_higher_than_a_liquid_one(self):
        liquid = amihud_illiquidity(self._frame(0.01, 1e7))
        illiquid = amihud_illiquidity(self._frame(0.03, 2e4, price=50.0))
        assert illiquid["mean_illiquidity"] > liquid["mean_illiquidity"] * 100

    def test_more_volume_at_the_same_volatility_means_more_liquid(self):
        thin = amihud_illiquidity(self._frame(0.02, 1e5))
        thick = amihud_illiquidity(self._frame(0.02, 1e7))
        assert thick["mean_illiquidity"] < thin["mean_illiquidity"]

    def test_it_leads_with_a_percentile_because_the_raw_value_is_meaningless(self):
        result = amihud_illiquidity(self._frame(0.02, 1e6))
        assert result["current_percentile"] is not None
        assert 0 <= result["current_percentile"] <= 100
        assert any("not interpretable" in w for w in result["warnings"])

    def test_it_says_it_is_not_a_spread(self):
        result = amihud_illiquidity(self._frame(0.02, 1e6))
        assert any("NOT a spread" in w for w in result["warnings"])

    def test_the_scaling_convention_is_declared(self):
        """Published values use several scalings and comparing across them
        silently is off by orders of magnitude."""
        result = amihud_illiquidity(self._frame(0.02, 1e6))
        assert result["scaling"] == "1e6"
        assert any("scaling convention" in w for w in result["warnings"])

    def test_zero_volume_days_are_dropped_rather_than_dividing_by_zero(self):
        frame = self._frame(0.02, 1e6)
        frame.loc[5:10, "volume"] = 0.0
        result = amihud_illiquidity(frame)
        assert math.isfinite(result["mean_illiquidity"])


class TestTheAmihudWindowIsAWindow:
    """
    `max(2, int(window))` stood in this function, so window=-5 became a
    2-bar average and nothing said otherwise. The caller asked for something
    impossible and was quietly given something else -- the failure mode that
    is worse than an exception, because the number that comes back is
    plausible.

    The tool boundary already declared `ge=2`, and the sibling in
    `backtest.liquidity` refuses a non-positive window outright. A direct
    library caller was the only one who could reach the silent rewrite.
    """

    @staticmethod
    def _frame(n=400, seed=5):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "close": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
                "volume": rng.lognormal(12.0, 0.4, n),
            }
        )

    @pytest.mark.parametrize("bad_window", [-5, -1, 0, 1])
    def test_a_window_below_two_is_refused_not_rewritten(self, bad_window):
        with pytest.raises(ValidationError, match="is not a window"):
            amihud_illiquidity(self._frame(), window=bad_window)

    def test_the_message_says_why_two_is_the_floor(self):
        with pytest.raises(ValidationError, match="halves of the rolling"):
            amihud_illiquidity(self._frame(), window=0)

    def test_a_legal_window_still_works(self):
        result = amihud_illiquidity(self._frame(), window=21)
        assert result["window"] == 21
        assert result["current_illiquidity"] is not None


class TestKyleLambda:
    @staticmethod
    def _planted(lam=2e-6, n=800, noise=0.05, seed=6):
        rng = np.random.default_rng(seed)
        volume = rng.uniform(1e5, 5e5, n)
        sign = rng.choice([-1.0, 1.0], n)
        change = lam * sign * volume + rng.normal(0, noise, n)
        return pd.DataFrame({"close": 100 + np.cumsum(change), "volume": volume})

    def test_it_recovers_a_planted_impact_coefficient(self):
        result = kyle_lambda(self._planted(lam=2e-6))
        assert result["kyle_lambda"] == pytest.approx(2e-6, rel=0.10)
        assert result["r_squared"] > 0.9

    @pytest.mark.parametrize("lam", [5e-7, 2e-6, 8e-6])
    def test_a_deeper_market_gives_a_smaller_lambda(self, lam):
        result = kyle_lambda(self._planted(lam=lam))
        assert result["kyle_lambda"] == pytest.approx(lam, rel=0.15)

    def test_the_impact_of_a_one_percent_order_is_reported_in_basis_points(self):
        result = kyle_lambda(self._planted())
        assert result["impact_of_1pct_adv_bps"] == pytest.approx(
            result["impact_of_1pct_adv"] / result["mean_price"] * 1e4, rel=1e-9
        )

    def test_a_meaningless_regression_is_declared_meaningless(self):
        """A lambda from a regression explaining 2% of the variance has a
        standard error larger than itself, and saying so is the difference
        between a number and a number you can size an order with."""
        result = kyle_lambda(self._planted(lam=1e-12, noise=1.0))
        if result["r_squared"] < 0.05:
            assert any(
                "standard error larger than itself" in w for w in result["warnings"]
            )

    def test_the_tick_rule_limitation_is_always_stated(self):
        result = kyle_lambda(self._planted())
        assert any("TICK RULE" in w for w in result["warnings"])

    def test_a_rolling_window_summarises_the_spread_of_estimates(self):
        result = kyle_lambda(self._planted(), window=100)
        assert result["rolling"]["n_windows"] > 0
        assert result["rolling"]["p25"] <= result["rolling"]["median_lambda"]
        assert result["rolling"]["median_lambda"] <= result["rolling"]["p75"]


class TestOrderFlowImbalance:
    @staticmethod
    def _frame(n=500, seed=8):
        return pd.DataFrame(
            {
                "close": 100
                * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.015, n))),
                "volume": np.random.default_rng(seed + 1).uniform(1e5, 9e5, n),
            }
        )

    def test_persistence_is_measured_without_the_window_overlap_artefact(self):
        """
        A rolling sum at window=5 shares four of its five observations with
        the previous point, so its lag-1 autocorrelation is about 1 - 1/w
        whatever the data does -- measured at +0.76, +0.89 and +0.96 for
        windows of 5, 10 and 21 on PURE NOISE. That describes the window,
        not the flow. Persistence is therefore computed on non-overlapping
        windows, and the artefact is returned separately so the difference
        is visible rather than assumed away.
        """
        for window in (5, 10, 21):
            result = order_flow_imbalance(self._frame(), window=window)
            assert abs(result["persistence"]) < 0.35
            assert result["overlapping_persistence"] > 0.7
            assert result["overlapping_persistence"] == pytest.approx(
                1 - 1 / window, abs=0.10
            )

    def test_random_data_shows_no_real_persistence(self):
        result = order_flow_imbalance(self._frame(), window=5)
        assert abs(result["persistence"]) < 0.2
        assert any("essentially" in w for w in result["warnings"])

    def test_the_artefact_is_explained_in_the_warnings(self):
        result = order_flow_imbalance(self._frame())
        assert any("NON-OVERLAPPING" in w for w in result["warnings"])

    def test_a_persistently_rising_series_is_mostly_buy_volume(self):
        frame = pd.DataFrame(
            {"close": np.linspace(100, 140, 300), "volume": np.full(300, 1e6)}
        )
        result = order_flow_imbalance(frame)
        assert result["buy_volume_fraction"] > 0.95
        assert result["current_imbalance"] > 0.9

    def test_the_tick_rule_caveat_is_stated(self):
        result = order_flow_imbalance(self._frame())
        assert any("TICK RULE" in w for w in result["warnings"])

    def test_a_repeated_timestamp_is_refused_by_name(self):
        """Aligning the imbalance with the next return by label raised
        pandas' 'cannot reindex on an axis with duplicate labels' on a dated
        frame with one stamp repeated. This test used to accept the
        positional answer instead, treating the repeat as just another bar;
        but two bars at one moment are two sources concatenated, nothing
        says which came first, and every bar estimator now refuses them by
        name -- still not with the pandas error."""
        frame = self._frame(n=300)
        stamps = pd.date_range("2026-01-02", periods=len(frame), freq="D")
        repeated = pd.DatetimeIndex(np.r_[stamps[:150], stamps[149:299]])
        assert repeated.has_duplicates
        dated = frame.set_axis(repeated)
        with pytest.raises(ValidationError) as exc:
            order_flow_imbalance(dated, window=5)
        assert str(stamps[149].date()) in str(exc.value)
        assert "duplicate labels" not in str(exc.value)

    def test_dated_bars_give_the_same_answer_as_plain_ones(self):
        """Null case: nothing repeated, and the date labels change nothing."""
        frame = self._frame(n=300)
        dated = frame.set_axis(
            pd.date_range("2026-01-02", periods=len(frame), freq="D")
        )
        assert order_flow_imbalance(dated, window=5) == order_flow_imbalance(
            frame, window=5
        )


class TestVpin:
    @staticmethod
    def _frame(n=500, seed=8):
        return pd.DataFrame(
            {
                "close": 100
                * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.015, n))),
                "volume": np.random.default_rng(seed + 1).uniform(1e5, 9e5, n),
            }
        )

    def test_the_buckets_hold_equal_volume(self):
        """The whole point of VPIN is measuring in volume time rather than
        clock time. If the buckets are not equal-volume it is just a
        rolling imbalance with extra steps."""
        result = estimate_vpin(self._frame(), n_buckets=40)
        assert result["n_buckets"] in (40, 41)
        assert result["bucket_volume"] > 0

    def test_one_sided_flow_scores_higher_than_two_sided_flow(self):
        one_sided = pd.DataFrame(
            {"close": np.linspace(100, 200, 400), "volume": np.full(400, 1e6)}
        )
        rng = np.random.default_rng(0)
        two_sided = pd.DataFrame(
            {
                "close": 100 + np.cumsum(rng.choice([-1.0, 1.0], 400)),
                "volume": np.full(400, 1e6),
            }
        )
        assert (
            estimate_vpin(one_sided)["current_vpin"]
            > estimate_vpin(two_sided)["current_vpin"]
        )

    def test_it_declares_that_it_is_not_the_vpin_of_the_paper(self):
        result = estimate_vpin(self._frame())
        assert any("not the VPIN of the paper" in w for w in result["warnings"])

    def test_it_declares_that_the_measure_is_contested(self):
        """Presenting VPIN as settled would be misleading -- the flash-crash
        result was challenged and the metric is arguably a transformation of
        volatility."""
        result = estimate_vpin(self._frame())
        assert any("CONTESTED" in w for w in result["warnings"])

    def test_vpin_is_bounded_between_zero_and_one(self):
        result = estimate_vpin(self._frame())
        assert 0.0 <= result["current_vpin"] <= 1.0
        assert 0.0 <= result["max_vpin"] <= 1.0


class TestIntradayVolumeProfile:
    @staticmethod
    def _intraday(days=5, u_shape=True):
        stamps = []
        for day in range(days):
            stamps.extend(
                pd.date_range(
                    f"2024-01-0{day + 2} 09:30",
                    f"2024-01-0{day + 2} 16:00",
                    freq="5min",
                )
            )
        index = pd.DatetimeIndex(stamps)
        minutes = index.hour * 60 + index.minute
        if u_shape:
            shape = 3.0 - 2.6 * np.sin(np.pi * (minutes - 570) / (960 - 570))
        else:
            shape = np.ones(len(index))
        return pd.DataFrame({"volume": shape * 1e5}, index=index)

    def test_it_finds_a_planted_u_shape(self):
        result = intraday_volume_profile(self._intraday())
        assert result["u_shaped"]
        assert result["open_share"] > result["trough_share"] * 3
        assert result["close_share"] > result["trough_share"] * 3

    def test_a_flat_day_is_not_called_u_shaped(self):
        """The null case. A detector that always finds the U-shape is
        reporting its own prior."""
        result = intraday_volume_profile(self._intraday(u_shape=False))
        assert not result["u_shaped"]
        assert any("NOT U-shaped" in w for w in result["warnings"])

    def test_the_shares_sum_to_one(self):
        result = intraday_volume_profile(self._intraday())
        assert sum(b["share_of_volume"] for b in result["profile"]) == pytest.approx(
            1.0
        )

    def test_daily_bars_are_refused_with_the_reason(self):
        index = pd.bdate_range("2023-01-02", periods=100)
        with pytest.raises(ValidationError, match="daily bars"):
            intraday_volume_profile(
                pd.DataFrame({"volume": np.full(100, 1e6)}, index=index)
            )

    def test_a_positional_index_is_refused(self):
        with pytest.raises(ValidationError, match="DatetimeIndex"):
            intraday_volume_profile(pd.DataFrame({"volume": np.full(100, 1e6)}))

    def test_it_warns_against_scheduling_against_the_clock(self):
        result = intraday_volume_profile(self._intraday())
        assert any("evenly across the CLOCK" in w for w in result["warnings"])

    def test_a_heavy_close_is_flagged_as_a_moving_target(self):
        frame = self._intraday()
        closing = frame.index.hour >= 15
        frame.loc[closing, "volume"] *= 30
        result = intraday_volume_profile(frame)
        if result["close_share"] > 0.20:
            assert any("Closing auction share" in w for w in result["warnings"])


class TestVpinBarsThatDidNotMove:
    """A bar whose close did not move has no direction the tick rule can
    read. It used to be counted as all buying, so a flat market came back
    at the maximum reading, 1.0 -- toxic flow on a tape where nothing
    happened."""

    def test_a_flat_market_has_no_imbalance(self):
        flat = pd.DataFrame({"close": [100.0] * 200, "volume": [1000.0] * 200})
        result = estimate_vpin(flat, n_buckets=50, window=10)
        assert result["current_vpin"] == 0.0
        assert result["max_vpin"] == 0.0
        assert result["undirected_volume_share"] == 1.0
        assert any("split half to each side" in w for w in result["warnings"])

    def test_a_still_bar_is_split_evenly_between_the_sides(self):
        """Every four-bar bucket is still, up, still, down: one buy, one
        sell and two bars nobody initiated. Half each makes every bucket
        exactly balanced; counting the still bars as buys made each one a
        3-to-1 imbalance of 0.5."""
        steps = np.tile([0.0, 1.0, 0.0, -1.0], 100)
        frame = pd.DataFrame({"close": 100.0 + np.cumsum(steps), "volume": 1000.0})
        result = estimate_vpin(frame, n_buckets=100, window=10)
        assert result["current_vpin"] == 0.0
        assert result["undirected_volume_share"] == pytest.approx(0.5)

    def test_bars_that_moved_keep_their_whole_volume_on_their_side(self):
        """Null case: a series that rises on every bar after the first has
        only one still bar, in the first bucket, and every later bucket is
        entirely buying -- exactly as before."""
        frame = pd.DataFrame({"close": np.linspace(100, 200, 400), "volume": 1e6})
        result = estimate_vpin(frame, n_buckets=100, window=10)
        assert result["current_vpin"] == 1.0
        assert result.get("undirected_volume_share", 1 / 400) == pytest.approx(1 / 400)

    def test_order_flow_imbalance_splits_the_same_way(self):
        """The sibling estimator's buy-volume fraction counted a still bar
        as not-buy, so a flat series read as all selling (0.0) beside a mean
        imbalance of exactly zero."""
        flat = pd.DataFrame({"close": [100.0] * 200, "volume": [1000.0] * 200})
        result = order_flow_imbalance(flat)
        assert result["buy_volume_fraction"] == 0.5
        assert result["mean_imbalance"] == 0.0

    def test_a_rising_series_is_all_buying_but_its_first_bar(self):
        """Only the first bar, which has no return, is split: it counted as
        not-buy, which put the fraction at 299/300."""
        frame = pd.DataFrame({"close": np.linspace(100, 140, 300), "volume": 1e6})
        result = order_flow_imbalance(frame)
        assert result["buy_volume_fraction"] == pytest.approx(299.5 / 300)


class TestTheProfileKeepsEveryBucket:
    """The trough is the number a volume profile exists to measure, and it
    was computed over the buckets some bar happened to fall in."""

    @staticmethod
    def _bars(hours_and_minutes, volume):
        day = pd.Timestamp("2026-03-02")
        stamps = [day + pd.Timedelta(hours=h, minutes=m) for h, m in hours_and_minutes]
        return pd.DataFrame({"volume": volume}, index=pd.DatetimeIndex(stamps))

    def _open_and_close_only(self):
        """Twenty one-minute bars at the open and twenty at the close, none
        in between: the most extreme U there is."""
        slots = [(9, 30 + m) for m in range(20)] + [(15, 30 + m) for m in range(20)]
        return self._bars(slots, [300.0] * 20 + [200.0] * 20)

    def test_the_empty_midday_is_the_trough(self):
        result = intraday_volume_profile(self._open_and_close_only(), n_buckets=13)
        assert result["n_buckets"] == 13
        assert len(result["profile"]) == 13
        assert result["n_empty_buckets"] == 11
        assert result["trough_share"] == 0.0
        assert result["u_shaped"] is True
        assert result["open_share"] == pytest.approx(0.6)
        assert result["close_share"] == pytest.approx(0.4)
        assert result["open_to_trough_ratio"] is None
        assert any("hold no bars" in w for w in result["warnings"])

    def test_the_trough_bucket_is_an_id_not_a_position(self):
        """It was a position in the list of occupied buckets, returned beside
        entries whose `bucket` field was the real id -- on the bars above it
        pointed at the close."""
        result = intraday_volume_profile(self._open_and_close_only(), n_buckets=13)
        trough = result["profile"][result["trough_bucket"]]
        assert trough["bucket"] == result["trough_bucket"]
        assert trough["share_of_volume"] == 0.0
        assert trough["n_bars"] == 0
        assert trough["mean_volume"] is None

    def test_one_empty_bucket_in_a_real_u_is_the_trough(self):
        """A U over a full session with nothing between 12:30 and 13:00 --
        bucket 6 of 13. Dropped, the trough became whichever occupied
        bucket was lowest and its id was off by one."""
        slots = [(h, m) for h in range(9, 16) for m in range(0, 60, 5)]
        slots = [(h, m) for h, m in slots if 570 <= h * 60 + m < 960]
        minutes = np.array([h * 60 + m for h, m in slots])
        shape = 3.0 - 2.6 * np.sin(np.pi * (minutes - 570) / 390)
        keep = (minutes < 750) | (minutes >= 780)
        frame = self._bars(
            [s for s, k in zip(slots, keep) if k], (shape * 1e5)[keep].tolist()
        )
        result = intraday_volume_profile(frame, n_buckets=13)
        assert result["trough_bucket"] == 6
        assert result["profile"][6]["start_time"] == "12:30"
        assert result["trough_share"] == 0.0
        assert result["n_empty_buckets"] == 1
        assert result["u_shaped"] is True

    def test_the_buckets_divide_the_session_not_the_sample(self):
        """Bars from 10:00 to 13:55 only. Over the sample's own range the
        first bucket was 'the open' at 10:00; over the session it is 09:30
        and empty, which is what the bars actually say about the open."""
        slots = [(h, m) for h in range(10, 14) for m in range(0, 60, 5)]
        frame = self._bars(slots, [1000.0] * len(slots))
        result = intraday_volume_profile(frame, n_buckets=13)
        assert result["bucket_span"] == ["09:30", "16:00"]
        assert result["profile"][0]["start_time"] == "09:30"
        assert result["profile"][12]["start_time"] == "15:30"
        assert result["open_share"] == 0.0
        assert result["close_share"] == 0.0
        assert result["u_shaped"] is False

    def test_a_zoned_index_is_bucketed_over_the_session_too(self):
        slots = [(h, m) for h in range(10, 14) for m in range(0, 60, 5)]
        frame = self._bars(slots, [1000.0] * len(slots))
        zoned = frame.tz_localize("America/New_York")
        result = intraday_volume_profile(zoned, n_buckets=13)
        assert result["bucket_span"] == ["09:30", "16:00"]
        assert [p["start_time"] for p in result["profile"]] == [
            p["start_time"]
            for p in intraday_volume_profile(frame, n_buckets=13)["profile"]
        ]

    def test_naive_bars_outside_the_session_are_profiled_as_given_and_named(self):
        """Without a zone nothing places a 04:00 bar, so the buckets span
        the bars' own times -- and the result says so and names the fix."""
        slots = [(h, m) for h in range(4, 20) for m in range(0, 60, 5)]
        frame = self._bars(slots, [1000.0] * len(slots))
        result = intraday_volume_profile(frame, n_buckets=13)
        assert result["bucket_span"] == ["04:00", "19:56"]
        assert any("index_timezone" in w for w in result["warnings"])

    def test_a_full_session_has_no_empty_bucket_and_no_note(self):
        """Null case: a session in which every bucket traded comes back as
        thirteen buckets with ids 0 to 12, one flat share each, and no note
        about empty buckets or bars outside the session -- as before."""
        slots = [(h, m) for h in range(9, 16) for m in range(0, 60, 5)]
        slots = [(h, m) for h, m in slots if 570 <= h * 60 + m < 960]
        frame = self._bars(slots, [1000.0] * len(slots))
        result = intraday_volume_profile(frame, n_buckets=13)
        assert result.get("n_empty_buckets", 0) == 0
        assert [p["bucket"] for p in result["profile"]] == list(range(13))
        assert all(p["n_bars"] > 0 for p in result["profile"])
        assert result["trough_share"] > 0
        assert not any("hold no bars" in w for w in result["warnings"])
        assert not any("outside the" in w for w in result["warnings"])

    def test_the_session_edges_are_the_ones_used(self):
        slots = [(h, m) for h in range(9, 16) for m in range(0, 60, 5)]
        slots = [(h, m) for h, m in slots if 570 <= h * 60 + m < 960]
        frame = self._bars(slots, [1000.0] * len(slots))
        result = intraday_volume_profile(frame, n_buckets=13)
        assert result["bucket_span"] == ["09:30", "16:00"]
        assert all(p["n_bars"] == 6 for p in result["profile"])


def _dated_bars(n=500, seed=0):
    """A daily random walk with a date on every bar, oldest first."""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1.0, n))
    return pd.DataFrame(
        {
            "close": close,
            "volume": rng.uniform(1e5, 5e5, n),
            "high": close + rng.uniform(0.1, 1.0, n),
            "low": close - rng.uniform(0.1, 1.0, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


#: Every estimator that reads consecutive bars as consecutive moments, with
#: the headline number each one returns.
SEQUENTIAL_ESTIMATORS = [
    pytest.param(lambda f: roll_spread(f["close"]), "spread_estimate", id="roll"),
    pytest.param(corwin_schultz_spread, "spread_bps", id="corwin_schultz"),
    pytest.param(amihud_illiquidity, "current_illiquidity", id="amihud"),
    pytest.param(kyle_lambda, "kyle_lambda", id="kyle"),
    pytest.param(order_flow_imbalance, "current_imbalance", id="order_flow"),
    pytest.param(estimate_vpin, "current_vpin", id="vpin"),
]


class TestBarsOutOfTimeOrder:
    """A dated frame out of order was estimated in the order it came. A
    shuffled random walk is white noise by construction, and Roll's
    estimator called it a significant spread of 16.99."""

    def test_a_shuffled_walk_is_not_a_spread(self):
        bars = _dated_bars()
        shuffled = bars.iloc[np.random.default_rng(1).permutation(len(bars))]
        in_order = roll_spread(bars["close"])
        result = roll_spread(shuffled["close"])
        assert result["spread_estimate"] == in_order["spread_estimate"]
        assert result["significant"] is in_order["significant"] is False

    @pytest.mark.parametrize("estimator,key", SEQUENTIAL_ESTIMATORS)
    def test_shuffled_bars_give_the_sorted_answer_and_say_so(self, estimator, key):
        bars = _dated_bars()
        shuffled = bars.iloc[np.random.default_rng(1).permutation(len(bars))]
        in_order = estimator(bars)
        result = estimator(shuffled)
        assert result[key] == in_order[key]
        assert any("NOT in time order" in w for w in result["warnings"])
        assert not any("NOT in time order" in w for w in in_order["warnings"])

    @pytest.mark.parametrize("estimator,key", SEQUENTIAL_ESTIMATORS)
    def test_a_repeated_stamp_is_refused_by_name(self, estimator, key):
        bars = _dated_bars()
        index = bars.index.to_numpy().copy()
        index[101] = index[100]
        with pytest.raises(ValidationError) as exc:
            estimator(bars.set_axis(pd.DatetimeIndex(index)))
        assert "2024-04-10" in str(exc.value)
        assert "one bar per timestamp" in str(exc.value)

    def test_a_bar_with_no_stamp_is_refused(self):
        bars = _dated_bars()
        index = bars.index.to_numpy().copy()
        index[7] = np.datetime64("NaT")
        with pytest.raises(ValidationError, match="NaT"):
            roll_spread(bars.set_axis(pd.DatetimeIndex(index))["close"])

    @pytest.mark.parametrize("estimator,key", SEQUENTIAL_ESTIMATORS)
    def test_bars_in_order_answer_as_their_positions_do(self, estimator, key):
        """Null case: dated bars already in order give the same number and
        the same warnings as the same bars on a plain index."""
        bars = _dated_bars()
        dated = estimator(bars)
        plain = estimator(bars.reset_index(drop=True))
        assert dated[key] == plain[key]
        assert dated["warnings"] == plain["warnings"]


class TestATapeWithoutQuotesIsCircular:
    """Prints bouncing half a cent either side of a price that never moves:
    the true lambda is zero. Signed by the tick rule and regressed on the
    last trade price, the bounce is in both x and y."""

    @staticmethod
    def _bouncing_tape(n=20000, seed=5):
        rng = np.random.default_rng(seed)
        stamps = pd.Timestamp("2026-03-02 14:30:00") + pd.to_timedelta(
            np.cumsum(rng.integers(20, 200, n)), unit="ms"
        )
        side = rng.choice([-1.0, 1.0], n)
        tape = pd.DataFrame(
            {"price": 100.0 + 0.005 * side, "size": 100.0},
            index=pd.DatetimeIndex(stamps),
        )
        quotes = pd.DataFrame(
            {"bid_price": 99.995, "ask_price": 100.005},
            index=pd.DatetimeIndex(stamps - pd.Timedelta(milliseconds=1)),
        )
        return tape, quotes

    def test_the_tick_rule_path_says_it_is_circular(self):
        tape, _ = self._bouncing_tape()
        result = kyle_lambda(trades=tape, freq="100ms")
        assert result["sign_source"] == "tick_rule"
        assert result["circular"] is True
        assert result["kyle_lambda"] > 0, "a slope from the bounce alone"
        assert any("CIRCULAR" in w and "Pass quotes" in w for w in result["warnings"])
        assert not any("attenuates" in w for w in result["warnings"])

    def test_the_same_tape_with_quotes_is_not(self):
        """Null case: Lee-Ready against the quotes and the midpoint as the
        price find the zero that is there."""
        tape, quotes = self._bouncing_tape()
        result = kyle_lambda(trades=tape, quotes=quotes, freq="100ms")
        assert result["sign_source"] == "lee_ready"
        assert result["circular"] is False
        assert result["kyle_lambda"] == pytest.approx(0.0, abs=1e-12)
