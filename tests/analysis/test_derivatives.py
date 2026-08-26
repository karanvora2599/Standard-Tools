"""
Greeks, structures, the surface, and what the hedge costs.

THE TESTING STRATEGY HERE IS IDENTITIES, not recorded outputs. Every claim
below is checked against something that has to be true independently of the
implementation:

- second-order greeks against CENTRAL FINITE DIFFERENCES of the first-order
  ones, which cannot agree by construction -- an algebra slip in vanna does
  not produce a matching slip in the numerical derivative of delta;
- put-call parity, which is model-free, checked against prices the model
  itself produced;
- a smile fitted to a planted quadratic, which must recover the planted
  coefficients exactly;
- forward variance additivity, which is arithmetic;
- the sign of a delta-hedged P&L, which must flip when implied and realized
  volatility swap places and must be near-symmetric in magnitude.

A test that asserted `vanna == 0.00397781` would pass forever and catch
nothing. Each of these fails if the formula is wrong.
"""

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.derivatives import (
    _durrleman_violations,
    analyze_strategy,
    analyze_vol_term_structure,
    check_put_call_parity,
    expected_move,
    fit_volatility_smile,
    implied_forward_price,
    option_greeks,
    option_risk_scenarios,
    simulate_delta_hedge,
    volatility_cone,
)
from standard_quant_tools.analysis.pricing import price_option
from standard_quant_tools.error import ValidationError

BASE = dict(
    spot=100.0,
    strike=105.0,
    time_to_expiry=0.5,
    volatility=0.28,
    risk_free_rate=0.04,
    dividend_yield=0.015,
)


def _greeks(**overrides):
    return option_greeks(**{**BASE, **overrides})


class TestSecondOrderGreeks:
    """
    Every second-order greek against a central difference of the first-order
    one it differentiates. This is the test that actually validates the
    algebra: the analytic form and the numerical derivative share no code.
    """

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_vanna_is_the_derivative_of_delta_in_vol(self, option_type):
        h = 1e-5
        up = _greeks(option_type=option_type, volatility=BASE["volatility"] + h)[
            "delta"
        ]
        down = _greeks(option_type=option_type, volatility=BASE["volatility"] - h)[
            "delta"
        ]
        # Reported per vol POINT, so the raw derivative is divided by 100.
        numeric = (up - down) / (2 * h) / 100.0
        assert _greeks(option_type=option_type)["vanna"] == pytest.approx(
            numeric, rel=1e-5
        )

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_volga_is_the_derivative_of_vega_in_vol(self, option_type):
        h = 1e-5
        up = _greeks(option_type=option_type, volatility=BASE["volatility"] + h)["vega"]
        down = _greeks(option_type=option_type, volatility=BASE["volatility"] - h)[
            "vega"
        ]
        numeric = (up - down) / (2 * h) / 100.0
        assert _greeks(option_type=option_type)["volga"] == pytest.approx(
            numeric, rel=1e-5
        )

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_charm_is_the_decay_of_delta_in_calendar_time(self, option_type):
        h = 1e-5
        # Calendar time runs the other way from time-to-expiry, hence the sign.
        up = _greeks(option_type=option_type, time_to_expiry=BASE["time_to_expiry"] + h)
        down = _greeks(
            option_type=option_type, time_to_expiry=BASE["time_to_expiry"] - h
        )
        numeric = -(up["delta"] - down["delta"]) / (2 * h) / 365.0
        assert _greeks(option_type=option_type)["charm"] == pytest.approx(
            numeric, rel=1e-5
        )

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_speed_is_the_derivative_of_gamma_in_spot(self, option_type):
        h = 1e-3
        up = _greeks(option_type=option_type, spot=BASE["spot"] + h)["gamma"]
        down = _greeks(option_type=option_type, spot=BASE["spot"] - h)["gamma"]
        assert _greeks(option_type=option_type)["speed"] == pytest.approx(
            (up - down) / (2 * h), rel=1e-4
        )

    def test_vanna_and_volga_are_identical_for_calls_and_puts(self):
        """
        Put-call parity is linear in spot and free of volatility, so every
        second derivative touching vol is shared. If these ever differ, a
        sign was flipped in one branch only.
        """
        call, put = _greeks(option_type="call"), _greeks(option_type="put")
        assert call["vanna"] == pytest.approx(put["vanna"], rel=1e-12)
        assert call["volga"] == pytest.approx(put["volga"], rel=1e-12)
        assert call["gamma"] == pytest.approx(put["gamma"], rel=1e-12)

    def test_delta_difference_is_the_parity_slope(self):
        """call delta - put delta = exp(-qT), exactly."""
        call, put = _greeks(option_type="call"), _greeks(option_type="put")
        expected = math.exp(-BASE["dividend_yield"] * BASE["time_to_expiry"])
        assert call["delta"] - put["delta"] == pytest.approx(expected, rel=1e-12)

    def test_gamma_peaks_near_the_money(self):
        """
        Measured rather than assumed: at 28 vol over half a year the
        distance that matters is set by vol*sqrt(T) = 0.20 in log terms, so
        K=140 is only 1.7 standard deviations out and still carries 31% of
        the at-the-money gamma. K=160 is the real wing at 9%.
        """
        curve = [_greeks(strike=k)["gamma"] for k in (100.0, 120.0, 140.0, 160.0)]
        assert curve == sorted(curve, reverse=True)
        assert curve[0] > curve[3] * 5

    def test_the_units_are_declared_for_every_greek(self):
        """There is no convention for these and the mismatch is a real
        source of error, so the result has to say."""
        result = _greeks()
        for greek in (
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
            "vanna",
            "volga",
            "charm",
        ):
            assert greek in result["units"]
            assert result["units"][greek]

    def test_a_negative_volatility_is_refused(self):
        with pytest.raises(ValidationError, match="volatility"):
            _greeks(volatility=-0.2)

    def test_an_unknown_option_type_is_named_in_the_error(self):
        with pytest.raises(ValidationError, match="straddle"):
            _greeks(option_type="straddle")


class TestStrategy:
    @staticmethod
    def _leg(kind, strike, quantity, vol=0.25, t=0.25):
        return {
            "option_type": kind,
            "strike": strike,
            "quantity": quantity,
            "volatility": vol,
            "time_to_expiry": t,
        }

    def test_a_straddle_breaks_even_at_the_strike_plus_the_premium(self):
        """The one structure whose breakevens are exactly known."""
        result = analyze_strategy(
            [self._leg("call", 100, 1), self._leg("put", 100, 1)],
            spot=100.0,
            risk_free_rate=0.0,
        )
        premium = result["net_premium"]
        assert len(result["breakevens"]) == 2
        assert result["breakevens"][0] == pytest.approx(100 - premium, abs=0.05)
        assert result["breakevens"][1] == pytest.approx(100 + premium, abs=0.05)

    def test_an_at_the_money_straddle_is_close_to_delta_neutral(self):
        result = analyze_strategy(
            [self._leg("call", 100, 1), self._leg("put", 100, 1)],
            spot=100.0,
            risk_free_rate=0.0,
        )
        assert abs(result["greeks"]["delta"]) < 0.10

    def test_a_vertical_spread_has_max_profit_equal_to_width_minus_debit(self):
        result = analyze_strategy(
            [self._leg("call", 95, 1, t=0.5), self._leg("call", 105, -1, t=0.5)],
            spot=100.0,
        )
        assert result["max_profit"] == pytest.approx(
            10.0 - result["net_premium"], abs=0.05
        )
        assert not result["max_profit_unbounded"]

    def test_an_unbounded_loss_is_reported_as_unbounded(self):
        """
        A short call has no worst case and a numerical scan cannot find one.
        Returning the edge of the grid as 'max loss' would be a finite
        number standing in for an infinite risk.
        """
        result = analyze_strategy([self._leg("call", 100, -1)], spot=100.0)
        assert result["max_loss_unbounded"]
        assert any("unbounded" in w.lower() for w in result["warnings"])

    def test_a_long_position_is_a_debit_and_a_short_one_a_credit(self):
        assert (
            analyze_strategy([self._leg("call", 100, 1)], spot=100.0)["position"]
            == "debit"
        )
        assert (
            analyze_strategy([self._leg("call", 100, -1)], spot=100.0)["position"]
            == "credit"
        )

    def test_stock_legs_carry_delta_one(self):
        result = analyze_strategy(
            [{"option_type": "stock", "quantity": 100}], spot=50.0
        )
        assert result["greeks"]["delta"] == pytest.approx(100.0)
        assert result["net_premium"] == pytest.approx(5000.0)

    def test_a_covered_call_is_short_gamma(self):
        result = analyze_strategy(
            [{"option_type": "stock", "quantity": 1}, self._leg("call", 105, -1)],
            spot=100.0,
        )
        assert result["greeks"]["gamma"] < 0
        assert result["greeks"]["delta"] < 1.0

    def test_a_zero_quantity_leg_is_refused_rather_than_ignored(self):
        with pytest.raises(ValidationError, match="not a position"):
            analyze_strategy([self._leg("call", 100, 0)], spot=100.0)

    def test_an_empty_leg_list_is_refused(self):
        with pytest.raises(ValidationError, match="no legs"):
            analyze_strategy([], spot=100.0)

    def test_an_unknown_leg_type_names_the_leg_index(self):
        with pytest.raises(ValidationError, match="leg 1"):
            analyze_strategy(
                [self._leg("call", 100, 1), {"option_type": "swap", "quantity": 1}],
                spot=100.0,
            )


class TestPutCallParity:
    KW = dict(
        spot=100.0,
        strike=95.0,
        time_to_expiry=0.75,
        risk_free_rate=0.05,
        dividend_yield=0.02,
    )

    def _model_prices(self, vol=0.30):
        kw = {**self.KW, "volatility": vol}
        return (
            price_option(**kw, option_type="call")["price"],
            price_option(**kw, option_type="put")["price"],
        )

    def test_model_prices_satisfy_parity_to_machine_precision(self):
        """
        Parity is model-free, so prices the model produced must satisfy it
        exactly. A failure here means the pricer's own call and put branches
        disagree.
        """
        call, put = self._model_prices()
        result = check_put_call_parity(call_price=call, put_price=put, **self.KW)
        assert abs(result["violation_bps_of_strike"]) < 1e-6
        assert result["within_tolerance"]

    def test_the_implied_dividend_recovers_the_true_one(self):
        """
        The diagnostic that identifies WHY parity failed. On consistent
        prices it must return the dividend that was actually used.
        """
        call, put = self._model_prices()
        result = check_put_call_parity(call_price=call, put_price=put, **self.KW)
        assert result["implied_dividend_yield"] == pytest.approx(0.02, abs=1e-8)

    def test_a_rich_call_is_flagged_with_the_conversion_direction(self):
        call, put = self._model_prices()
        result = check_put_call_parity(call_price=call + 2.0, put_price=put, **self.KW)
        assert not result["within_tolerance"]
        assert result["violation"] > 0
        assert any("conversion" in w for w in result["warnings"])

    def test_a_rich_put_is_flagged_with_the_reversal_direction(self):
        call, put = self._model_prices()
        result = check_put_call_parity(call_price=call, put_price=put + 2.0, **self.KW)
        assert result["violation"] < 0
        assert any("reversal" in w for w in result["warnings"])

    def test_the_likely_causes_are_named_before_the_arbitrage_is(self):
        """
        A parity break is a stale quote far more often than free money, and
        the warning has to say so or it will be traded on.
        """
        call, put = self._model_prices()
        result = check_put_call_parity(call_price=call + 3.0, put_price=put, **self.KW)
        text = " ".join(result["warnings"]).lower()
        assert "stale" in text or "timestamp" in text
        assert "borrow" in text

    def test_a_negative_price_is_refused(self):
        with pytest.raises(ValidationError, match="negative"):
            check_put_call_parity(call_price=-1.0, put_price=2.0, **self.KW)


class TestVolatilitySmile:
    FORWARD, T = 100.0, 0.5
    STRIKES = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)

    def _planted(self, atm=0.25, skew=-0.30, curvature=0.80):
        x = np.log(self.STRIKES / self.FORWARD)
        return atm + skew * x + curvature * x**2

    def test_it_recovers_a_planted_quadratic_exactly(self):
        result = fit_volatility_smile(
            self.STRIKES,
            self._planted(),
            forward=self.FORWARD,
            time_to_expiry=self.T,
        )
        assert result["atm_vol"] == pytest.approx(0.25, abs=1e-9)
        assert result["skew"] == pytest.approx(-0.30, abs=1e-9)
        assert result["curvature"] == pytest.approx(0.80, abs=1e-9)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-9)

    def test_the_fit_is_in_log_moneyness_not_strike(self):
        """
        The same shape, refit after a rally, must produce the same skew. A
        parabola in STRIKE would not -- its vertex sits at a fixed price, so
        the fitted skew would drift with the level of the underlying.
        """
        low = fit_volatility_smile(
            self.STRIKES, self._planted(), forward=100.0, time_to_expiry=self.T
        )
        high = fit_volatility_smile(
            self.STRIKES * 1.10,
            self._planted(),
            forward=110.0,
            time_to_expiry=self.T,
        )
        assert high["skew"] == pytest.approx(low["skew"], rel=1e-9)
        assert high["curvature"] == pytest.approx(low["curvature"], rel=1e-9)

    def test_a_downward_skew_comes_back_negative(self):
        """Equity smiles slope down in log-moneyness; the sign has to match
        the convention or every downstream reading is backwards."""
        result = fit_volatility_smile(
            self.STRIKES,
            self._planted(skew=-0.5),
            forward=self.FORWARD,
            time_to_expiry=self.T,
        )
        assert result["skew"] < 0

    def test_too_few_strikes_is_refused_with_the_reason(self):
        with pytest.raises(ValidationError, match="interpolation|at least"):
            fit_volatility_smile(
                [95.0, 100.0, 105.0],
                [0.26, 0.25, 0.26],
                forward=100.0,
                time_to_expiry=0.5,
            )

    def test_mismatched_input_lengths_are_refused(self):
        with pytest.raises(ValidationError, match="different lengths"):
            fit_volatility_smile(
                self.STRIKES, self._planted()[:-1], forward=100.0, time_to_expiry=0.5
            )

    def test_a_poor_fit_is_declared_rather_than_returned_clean(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.15, self.STRIKES.size)
        result = fit_volatility_smile(
            self.STRIKES,
            np.abs(self._planted() + noise),
            forward=self.FORWARD,
            time_to_expiry=self.T,
        )
        if result["r_squared"] < 0.9:
            assert any("quadratic does not describe" in w for w in result["warnings"])

    def test_the_fitted_range_is_stated_because_it_does_not_extrapolate(self):
        result = fit_volatility_smile(
            self.STRIKES, self._planted(), forward=self.FORWARD, time_to_expiry=self.T
        )
        assert result["strike_range"] == [80.0, 120.0]
        assert any("extrapolate" in w for w in result["warnings"])

    def test_a_concave_smile_is_flagged_as_a_butterfly_arbitrage(self):
        """
        CONCAVITY is what breaks the density, not convexity -- a butterfly
        arbitrage is literally a concave price in strike. The first version
        of this test planted an extreme POSITIVE curvature and saw nothing,
        which was correct: the +w''/2 term means convexity pushes
        Durrleman's g up. Negative curvature is what pushes it below zero.
        """
        result = fit_volatility_smile(
            self.STRIKES,
            self._planted(curvature=-4.0),
            forward=self.FORWARD,
            time_to_expiry=self.T,
        )
        assert result["arbitrage_violations"]
        assert result["arbitrage_violations"][0]["reason"] == "negative implied density"
        assert any("arbitrage" in w.lower() for w in result["warnings"])

    def test_a_negative_fitted_vol_is_reported_as_its_own_failure(self):
        """
        Two ways the check fails and they need different diagnoses: a
        negative FITTED VOL is a broken fit, while a negative density is an
        arbitrage in otherwise-sane quotes. The branch is tested directly
        because it is not reachable through `fit_volatility_smile` with
        valid inputs -- every input vol must be positive, so a least-squares
        quadratic through them stays positive inside the fitted range. That
        makes it a defensive branch, and a defensive branch still has to say
        the right thing when it fires.
        """
        violations = _durrleman_violations(0.05, -1.5, 0.0, np.array([-0.2, 0.2]), 0.5)
        assert violations
        assert violations[0]["reason"] == "negative fitted vol"

    def test_a_well_behaved_smile_has_no_violations(self):
        """The null case: a gentle, realistic smile must come back clean, or
        the detector is useless."""
        result = fit_volatility_smile(
            self.STRIKES,
            self._planted(atm=0.25, skew=-0.10, curvature=0.15),
            forward=self.FORWARD,
            time_to_expiry=self.T,
        )
        assert not result["arbitrage_violations"]


class TestVolatilityCone:
    @staticmethod
    def _prices(vol_per_day=0.02, n=800, seed=0):
        rng = np.random.default_rng(seed)
        return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol_per_day, n))))

    def test_the_median_recovers_the_generating_volatility(self):
        cone = volatility_cone(self._prices())
        row = next(r for r in cone["cone"] if r["horizon_days"] == 21)
        assert row["median"] == pytest.approx(0.02 * math.sqrt(252), rel=0.10)

    def test_short_horizons_have_a_wider_distribution_than_long_ones(self):
        """
        The cone SHAPE is the information: short-horizon realized vol
        averages fewer days, so its distribution is wider. A cone that is
        not wider at the short end is describing overlapping windows.
        """
        cone = volatility_cone(self._prices(n=1500))
        rows = {r["horizon_days"]: r for r in cone["cone"]}
        short = rows[5]["p90"] - rows[5]["p10"]
        long = rows[126]["p90"] - rows[126]["p10"]
        assert short > long

    def test_the_independent_window_count_is_reported(self):
        """
        Rolling windows overlap, so the observation count overstates the
        confidence by roughly the horizon. The honest number has to be
        visible next to the percentiles.
        """
        cone = volatility_cone(self._prices())
        for row in cone["cone"]:
            assert row["independent_windows"] <= row["n_windows"]
            assert row["independent_windows"] >= 1

    def test_an_implied_vol_is_placed_as_a_percentile(self):
        cone = volatility_cone(self._prices(), current_implied={21: 0.60})
        row = next(r for r in cone["cone"] if r["horizon_days"] == 21)
        assert row["implied_percentile"] > 90
        assert row["implied_vs_median"] > 0

    def test_thin_horizons_are_flagged(self):
        cone = volatility_cone(self._prices(n=200))
        thin = [r for r in cone["cone"] if r["independent_windows"] < 10]
        if thin:
            assert any("INDEPENDENT" in w for w in cone["warnings"])

    def test_too_little_history_is_refused(self):
        with pytest.raises(ValidationError, match="not enough history"):
            volatility_cone(pd.Series(np.linspace(100, 110, 40)))


class TestTermStructure:
    def test_forward_variance_is_additive(self):
        """
        Total variance adds across time, so the forward vol between two
        expiries is determined by arithmetic, not by the difference of the
        quotes. This is the number a calendar spread actually prices.
        """
        result = analyze_vol_term_structure({30 / 365: 0.25, 60 / 365: 0.28})
        expected = math.sqrt((0.28**2 * 60 - 0.25**2 * 30) / 30)
        assert result["forward_vols"][0]["forward_vol"] == pytest.approx(
            expected, rel=1e-12
        )

    def test_the_forward_vol_is_not_the_quoted_difference(self):
        """The mistake this tool exists to prevent."""
        result = analyze_vol_term_structure({30 / 365: 0.25, 60 / 365: 0.28})
        forward = result["forward_vols"][0]["forward_vol"]
        assert abs(forward - 0.28) > 0.02

    def test_contango_and_backwardation_are_named(self):
        assert analyze_vol_term_structure({0.1: 0.20, 0.5: 0.28})["shape"] == "contango"
        assert (
            analyze_vol_term_structure({0.1: 0.40, 0.5: 0.22})["shape"]
            == "backwardation"
        )

    def test_a_calendar_arbitrage_is_detected(self):
        """
        Total variance must be non-decreasing in maturity. Steep enough
        backwardation breaks that, and it is an arbitrage in the quotes.
        """
        result = analyze_vol_term_structure({30 / 365: 0.60, 60 / 365: 0.20})
        assert result["arbitrage_violations"]
        assert any("arbitrage" in w.lower() for w in result["warnings"])
        assert result["forward_vols"][0]["forward_vol"] is None

    def test_mild_backwardation_is_not_an_arbitrage(self):
        """The null case: backwardation is normal ahead of an event and
        must not be flagged as free money."""
        result = analyze_vol_term_structure({30 / 365: 0.30, 60 / 365: 0.28})
        assert not result["arbitrage_violations"]
        assert result["shape"] == "backwardation"

    def test_backwardation_warns_that_it_prices_an_event(self):
        result = analyze_vol_term_structure({30 / 365: 0.40, 60 / 365: 0.25})
        assert any("event" in w.lower() for w in result["warnings"])

    def test_one_expiry_is_not_a_term_structure(self):
        with pytest.raises(ValidationError, match="at least two"):
            analyze_vol_term_structure({0.25: 0.30})


class TestExpectedMove:
    def test_the_straddle_convention_is_eighty_percent_of_one_sd(self):
        """
        Two conventions circulate and they differ by ~20%. Both are
        returned; confusing them misprices event trades.
        """
        result = expected_move(spot=100.0, implied_vol=0.32, days=30)
        assert result["straddle_approximation"] == pytest.approx(
            0.8 * result["one_sd_move"], rel=1e-12
        )

    def test_the_move_scales_as_the_square_root_of_time(self):
        one = expected_move(spot=100.0, implied_vol=0.30, days=10)["one_sd_move"]
        four = expected_move(spot=100.0, implied_vol=0.30, days=40)["one_sd_move"]
        assert four == pytest.approx(2.0 * one, rel=1e-12)

    def test_it_says_the_number_is_not_a_bound(self):
        """The whole reason this tool exists: 'the expected move' gets read
        as a ceiling and it is exceeded a third of the time."""
        result = expected_move(spot=100.0, implied_vol=0.30, days=30)
        assert result["theoretical_exceedance_pct"] == pytest.approx(31.7)
        assert any("not a bound" in w for w in result["warnings"])

    def test_realized_moves_give_the_honest_exceedance_rate(self):
        rng = np.random.default_rng(0)
        # Moves far fatter than the implied 30-day vol implies.
        moves = np.abs(rng.normal(0, 0.15, 100))
        result = expected_move(
            spot=100.0, implied_vol=0.20, days=30, realized_moves=moves
        )
        assert result["realized"]["n_observations"] == 100
        assert result["realized"]["exceeded_implied_pct"] > 45
        assert any("fatter-tailed" in w for w in result["warnings"])

    def test_a_thin_realized_sample_is_flagged(self):
        result = expected_move(
            spot=100.0, implied_vol=0.30, days=30, realized_moves=[0.02, 0.05, 0.01]
        )
        assert any("standard error" in w for w in result["warnings"])


class TestDeltaHedge:
    KW = dict(spot=100.0, strike=100.0, time_to_expiry=0.25, n_hedges=63, n_paths=300)

    def test_selling_rich_vol_makes_money_and_selling_cheap_vol_loses(self):
        """
        The sign of the whole trade. If this ever comes back the same way
        round for both, the P&L accounting is broken in a way no single
        number would reveal.
        """
        rich = simulate_delta_hedge(
            **self.KW, implied_vol=0.35, realized_vol=0.20, seed=1
        )
        cheap = simulate_delta_hedge(
            **self.KW, implied_vol=0.20, realized_vol=0.35, seed=1
        )
        assert rich["mean_pnl"] > 0
        assert cheap["mean_pnl"] < 0

    def test_hedging_at_the_realized_vol_is_roughly_a_wash(self):
        result = simulate_delta_hedge(
            **self.KW, implied_vol=0.30, realized_vol=0.30, seed=3
        )
        assert abs(result["mean_pnl"]) < 0.5 * result["std_pnl"]

    def test_the_hedging_error_scales_as_one_over_root_n(self):
        """
        The documented tradeoff, measured. Quadrupling the rehedge count
        should roughly halve the P&L standard deviation -- NOT quarter it,
        which is the intuition this test exists to correct.
        """
        few = simulate_delta_hedge(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            implied_vol=0.30,
            realized_vol=0.30,
            n_hedges=21,
            n_paths=300,
            seed=2,
        )
        many = simulate_delta_hedge(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            implied_vol=0.30,
            realized_vol=0.30,
            n_hedges=84,
            n_paths=300,
            seed=2,
        )
        ratio = few["std_pnl"] / many["std_pnl"]
        assert 1.5 < ratio < 2.6, (
            f"quadrupling the hedge count changed the error by {ratio:.2f}x, "
            "and 1/sqrt(n) predicts about 2x"
        )

    def test_transaction_costs_reduce_the_pnl(self):
        free = simulate_delta_hedge(
            **self.KW, implied_vol=0.35, realized_vol=0.20, seed=4
        )
        costly = simulate_delta_hedge(
            **self.KW,
            implied_vol=0.35,
            realized_vol=0.20,
            transaction_cost_bps=25.0,
            seed=4,
        )
        assert costly["mean_pnl"] < free["mean_pnl"]
        assert costly["mean_transaction_cost"] > 0
        assert free["mean_transaction_cost"] == 0.0

    def test_the_dispersion_is_reported_because_the_mean_is_not_the_trade(self):
        result = simulate_delta_hedge(
            **self.KW, implied_vol=0.32, realized_vol=0.28, seed=5
        )
        assert result["p05_pnl"] < result["median_pnl"] < result["p95_pnl"]
        assert result["worst_pnl"] <= result["p05_pnl"]
        assert result["best_pnl"] >= result["p95_pnl"]

    def test_the_sign_convention_is_stated(self):
        result = simulate_delta_hedge(
            **self.KW, implied_vol=0.30, realized_vol=0.30, seed=6
        )
        assert any("SHORT the option" in w for w in result["warnings"])

    def test_a_thin_path_count_is_flagged(self):
        result = simulate_delta_hedge(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            implied_vol=0.30,
            realized_vol=0.30,
            n_hedges=10,
            n_paths=20,
            seed=7,
        )
        assert any("noisy" in w for w in result["warnings"])

    def test_it_is_reproducible_from_the_seed(self):
        a = simulate_delta_hedge(**self.KW, implied_vol=0.3, realized_vol=0.25, seed=11)
        b = simulate_delta_hedge(**self.KW, implied_vol=0.3, realized_vol=0.25, seed=11)
        assert a["mean_pnl"] == b["mean_pnl"]


class TestRiskScenarios:
    KW = dict(spot=100.0, strike=100.0, time_to_expiry=0.5, volatility=0.25)

    def test_the_unshocked_cell_reprices_to_the_base_value(self):
        result = option_risk_scenarios(**self.KW, spot_shocks=[0.0], vol_shocks=[0.0])
        assert result["grid"][0]["cells"][0]["pnl"] == pytest.approx(0.0, abs=1e-9)

    def test_a_long_call_gains_on_spot_up_and_vol_up(self):
        result = option_risk_scenarios(**self.KW, quantity=1.0)
        row = next(
            r for r in result["grid"] if r["spot_shock_pct"] == pytest.approx(20.0)
        )
        cell = next(c for c in row["cells"] if c["vol_shock"] == pytest.approx(0.10))
        assert cell["pnl"] > 0

    def test_the_worst_case_for_a_short_call_is_spot_up_and_vol_up(self):
        result = option_risk_scenarios(**self.KW, quantity=-1.0)
        assert result["worst_case"]["spot_shock_pct"] > 0
        assert result["worst_case"]["vol_shock"] > 0

    @pytest.mark.parametrize("shock,floor", [(0.10, 0.008), (0.20, 0.04), (0.30, 0.09)])
    def test_the_delta_gamma_estimate_degrades_with_the_size_of_the_move(
        self, shock, floor
    ):
        """
        The reason to revalue rather than approximate, MEASURED at three
        move sizes rather than asserted at one. Delta-gamma overstates a
        long call's gain by 1.2% at a 10% move, 5.0% at 20% and 11.1% at
        30% -- the error grows with the cube of the move, which is why a
        stress test built on greeks understates a real gap.

        The floors are set below the measured values so the test pins the
        SHAPE (error grows fast with shock size) without breaking on a
        third-decimal change.
        """
        greeks = option_greeks(**self.KW, risk_free_rate=0.0, option_type="call")
        result = option_risk_scenarios(**self.KW, spot_shocks=[shock], vol_shocks=[0.0])
        exact = result["grid"][0]["cells"][0]["pnl"]
        move = self.KW["spot"] * shock
        taylor = greeks["delta"] * move + 0.5 * greeks["gamma"] * move**2
        assert taylor > exact, "delta-gamma should overstate a long call's gain"
        assert abs(exact - taylor) > floor * abs(exact)

    def test_decaying_past_expiry_is_refused(self):
        with pytest.raises(ValidationError, match="past expiry"):
            option_risk_scenarios(**self.KW, days_forward=400)

    def test_the_independence_of_the_axes_is_declared(self):
        """Spot -20% with vol unchanged is a cell in the grid and not a
        state of the world. The result has to say so."""
        result = option_risk_scenarios(**self.KW)
        assert any("independently" in w.lower() for w in result["warnings"])


class TestImpliedForward:
    def test_the_carry_forward_is_exact(self):
        result = implied_forward_price(
            spot=100.0, time_to_expiry=1.0, risk_free_rate=0.05, dividend_yield=0.02
        )
        assert result["forward"] == pytest.approx(100 * math.exp(0.03), rel=1e-12)

    def test_borrow_reduces_the_forward_separately_from_the_dividend(self):
        """
        They are folded together as 'carry' constantly, and they behave
        differently -- borrow is a floating rate that moves hundreds of bps
        in a day. Keeping them apart is the point of the tool.
        """
        without = implied_forward_price(
            spot=100.0, time_to_expiry=1.0, risk_free_rate=0.05, dividend_yield=0.02
        )
        with_borrow = implied_forward_price(
            spot=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            dividend_yield=0.02,
            borrow_rate=0.03,
        )
        assert with_borrow["forward"] < without["forward"]
        assert with_borrow["components"]["borrow"] < 0
        assert without["components"]["borrow"] == 0.0

    def test_a_dividend_above_the_rate_puts_the_forward_below_spot(self):
        result = implied_forward_price(
            spot=100.0, time_to_expiry=1.0, risk_free_rate=0.01, dividend_yield=0.05
        )
        assert result["forward"] < 100.0
        assert result["basis"] < 0
