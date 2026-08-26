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
