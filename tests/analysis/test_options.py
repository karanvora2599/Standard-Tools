"""
Tests for analysis/options.py: Black-Scholes-Merton pricing, Greeks, and
implied volatility. Pure math, no network/data-provider dependency.

Correctness is validated three ways: (1) exact match against a well-known
textbook reference (Hull), (2) internal consistency invariants (put-call
parity, Greeks matching finite-difference derivatives of price), and
(3) round-trip (price -> implied_volatility recovers the original volatility).
"""

import math

import pytest

from standard_quant_tools.analysis.options import (
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)
from standard_quant_tools.error import ValidationError

# Hull's "Options, Futures, and Other Derivatives" reference example:
# S=42, K=40, T=0.5, r=0.10, sigma=0.20 -> call ~= 4.76, put ~= 0.81
HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA = 42.0, 40.0, 0.5, 0.10, 0.20


class TestBlackScholesPriceReference:
    def test_call_matches_hull_reference(self):
        price = black_scholes_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        assert price == pytest.approx(4.76, abs=0.01)

    def test_put_matches_hull_reference(self):
        price = black_scholes_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put")
        assert price == pytest.approx(0.81, abs=0.01)

    def test_put_call_parity_no_dividend(self):
        call = black_scholes_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        put = black_scholes_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put")
        rhs = HULL_S - HULL_K * math.exp(-HULL_R * HULL_T)
        assert (call - put) == pytest.approx(rhs, abs=1e-9)

    @pytest.mark.parametrize("q", [0.0, 0.02, 0.05])
    def test_put_call_parity_with_dividend_yield(self, q):
        call = black_scholes_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call", dividend_yield=q
        )
        put = black_scholes_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put", dividend_yield=q
        )
        rhs = HULL_S * math.exp(-q * HULL_T) - HULL_K * math.exp(-HULL_R * HULL_T)
        assert (call - put) == pytest.approx(rhs, abs=1e-9)

    def test_deep_itm_call_approaches_intrinsic_minus_discount(self):
        # Very high spot relative to strike: call ~ S*exp(-qT) - K*exp(-rT)
        price = black_scholes_price(1000.0, 40.0, HULL_T, HULL_R, HULL_SIGMA, "call")
        intrinsic_like = 1000.0 - 40.0 * math.exp(-HULL_R * HULL_T)
        assert price == pytest.approx(intrinsic_like, rel=1e-3)

    def test_deep_otm_call_near_zero(self):
        price = black_scholes_price(10.0, 1000.0, HULL_T, HULL_R, HULL_SIGMA, "call")
        assert price < 1e-6


class TestBlackScholesPriceValidation:
    def test_non_positive_spot_raises(self):
        with pytest.raises(ValidationError, match="spot"):
            black_scholes_price(0.0, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")

    def test_non_positive_strike_raises(self):
        with pytest.raises(ValidationError, match="strike"):
            black_scholes_price(HULL_S, -5.0, HULL_T, HULL_R, HULL_SIGMA, "call")

    def test_non_positive_time_to_expiry_raises(self):
        with pytest.raises(ValidationError, match="time_to_expiry"):
            black_scholes_price(HULL_S, HULL_K, 0.0, HULL_R, HULL_SIGMA, "call")

    def test_non_positive_volatility_raises(self):
        with pytest.raises(ValidationError, match="volatility"):
            black_scholes_price(HULL_S, HULL_K, HULL_T, HULL_R, 0.0, "call")

    def test_unknown_option_type_raises(self):
        with pytest.raises(ValidationError, match="option_type"):
            black_scholes_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "straddle")

    def test_negative_dividend_yield_raises(self):
        with pytest.raises(ValidationError, match="dividend_yield"):
            black_scholes_price(
                HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call", dividend_yield=-0.01
            )


class TestBlackScholesGreeks:
    def test_call_delta_bounds(self):
        greeks = black_scholes_greeks(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call"
        )
        assert 0.0 < greeks["delta"] < 1.0

    def test_put_delta_bounds(self):
        greeks = black_scholes_greeks(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put")
        assert -1.0 < greeks["delta"] < 0.0

    def test_gamma_and_vega_positive_and_shared_across_call_put(self):
        call_greeks = black_scholes_greeks(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call"
        )
        put_greeks = black_scholes_greeks(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put"
        )
        assert call_greeks["gamma"] > 0
        assert call_greeks["vega"] > 0
        assert call_greeks["gamma"] == pytest.approx(put_greeks["gamma"], rel=1e-9)
        assert call_greeks["vega"] == pytest.approx(put_greeks["vega"], rel=1e-9)

    def test_delta_put_call_parity(self):
        # delta_call - delta_put == exp(-qT), with or without a dividend yield
        for q in (0.0, 0.03):
            call_greeks = black_scholes_greeks(
                HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call", dividend_yield=q
            )
            put_greeks = black_scholes_greeks(
                HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put", dividend_yield=q
            )
            assert (call_greeks["delta"] - put_greeks["delta"]) == pytest.approx(
                math.exp(-q * HULL_T), abs=1e-9
            )

    def test_call_rho_positive_put_rho_negative(self):
        call_greeks = black_scholes_greeks(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call"
        )
        put_greeks = black_scholes_greeks(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "put"
        )
        assert call_greeks["rho"] > 0
        assert put_greeks["rho"] < 0

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_delta_matches_finite_difference_of_price_wrt_spot(self, option_type):
        h = 1e-4
        price_up = black_scholes_price(
            HULL_S + h, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type
        )
        price_down = black_scholes_price(
            HULL_S - h, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type
        )
        numeric_delta = (price_up - price_down) / (2 * h)
        greeks = black_scholes_greeks(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type
        )
        assert greeks["delta"] == pytest.approx(numeric_delta, abs=1e-4)

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_vega_matches_finite_difference_of_price_wrt_vol(self, option_type):
        h = 1e-4
        price_up = black_scholes_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA + h, option_type
        )
        price_down = black_scholes_price(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA - h, option_type
        )
        numeric_vega = (price_up - price_down) / (2 * h)
        greeks = black_scholes_greeks(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type
        )
        assert greeks["vega"] == pytest.approx(numeric_vega, abs=1e-3)

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_gamma_matches_finite_difference_of_delta_wrt_spot(self, option_type):
        h = 1e-3
        greeks_up = black_scholes_greeks(
            HULL_S + h, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type
        )
        greeks_down = black_scholes_greeks(
            HULL_S - h, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type
        )
        numeric_gamma = (greeks_up["delta"] - greeks_down["delta"]) / (2 * h)
        greeks = black_scholes_greeks(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type
        )
        assert greeks["gamma"] == pytest.approx(numeric_gamma, abs=1e-3)

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_theta_matches_finite_difference_of_price_wrt_time(self, option_type):
        # theta = -dPrice/dT (price decreases as time passes, i.e. T decreases
        # toward expiry) -- so theta should equal -(price(T+h)-price(T-h))/(2h)
        h = 1e-4
        price_up = black_scholes_price(
            HULL_S, HULL_K, HULL_T + h, HULL_R, HULL_SIGMA, option_type
        )
        price_down = black_scholes_price(
            HULL_S, HULL_K, HULL_T - h, HULL_R, HULL_SIGMA, option_type
        )
        numeric_theta = -(price_up - price_down) / (2 * h)
        greeks = black_scholes_greeks(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type
        )
        assert greeks["theta"] == pytest.approx(numeric_theta, abs=1e-2)

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_rho_matches_finite_difference_of_price_wrt_rate(self, option_type):
        h = 1e-4
        price_up = black_scholes_price(
            HULL_S, HULL_K, HULL_T, HULL_R + h, HULL_SIGMA, option_type
        )
        price_down = black_scholes_price(
            HULL_S, HULL_K, HULL_T, HULL_R - h, HULL_SIGMA, option_type
        )
        numeric_rho = (price_up - price_down) / (2 * h)
        greeks = black_scholes_greeks(
            HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, option_type
        )
        assert greeks["rho"] == pytest.approx(numeric_rho, abs=1e-3)


class TestImpliedVolatility:
    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("sigma_true", [0.10, 0.20, 0.35, 0.60])
    def test_round_trip_recovers_volatility(self, option_type, sigma_true):
        price = black_scholes_price(
            HULL_S, HULL_K, HULL_T, HULL_R, sigma_true, option_type
        )
        result = implied_volatility(price, HULL_S, HULL_K, HULL_T, HULL_R, option_type)
        assert result["converged"] is True
        assert result["implied_volatility"] == pytest.approx(sigma_true, abs=1e-4)

    def test_round_trip_with_dividend_yield(self):
        q = 0.03
        price = black_scholes_price(
            HULL_S, HULL_K, HULL_T, HULL_R, 0.25, "call", dividend_yield=q
        )
        result = implied_volatility(
            price, HULL_S, HULL_K, HULL_T, HULL_R, "call", dividend_yield=q
        )
        assert result["implied_volatility"] == pytest.approx(0.25, abs=1e-4)

    def test_deep_otm_option_still_converges(self):
        # Deep OTM call: tiny vega, exercises the harder end of Newton's stability
        price = black_scholes_price(HULL_S, 100.0, HULL_T, HULL_R, 0.30, "call")
        result = implied_volatility(price, HULL_S, 100.0, HULL_T, HULL_R, "call")
        assert result["converged"] is True
        assert result["implied_volatility"] == pytest.approx(0.30, abs=1e-3)

    def test_forcing_bisection_fallback_still_converges(self):
        # max_iterations=0 skips Newton entirely, forcing the bisection path
        # deterministically -- a direct test of that fallback, not just an
        # incidental exercise of it.
        price = black_scholes_price(HULL_S, HULL_K, HULL_T, HULL_R, HULL_SIGMA, "call")
        result = implied_volatility(
            price, HULL_S, HULL_K, HULL_T, HULL_R, "call", max_iterations=0
        )
        assert result["method"] == "bisection"
        assert result["converged"] is True
        assert result["implied_volatility"] == pytest.approx(HULL_SIGMA, abs=1e-4)

    def test_price_above_no_arbitrage_upper_bound_raises(self):
        # Call upper bound is S*exp(-qT) = 42 (q=0) -- anything above that is
        # not reproducible by any volatility.
        with pytest.raises(ValidationError, match="no-arbitrage"):
            implied_volatility(50.0, HULL_S, HULL_K, HULL_T, HULL_R, "call")

    def test_price_below_no_arbitrage_lower_bound_raises(self):
        # Deep ITM put (K=100 >> S=42): lower bound = K*exp(-rT) - S ~= 53.1,
        # so a quoted price of 10.0 is below it -- no volatility reproduces it.
        with pytest.raises(ValidationError, match="no-arbitrage"):
            implied_volatility(10.0, HULL_S, 100.0, HULL_T, HULL_R, "put")

    def test_non_positive_option_price_raises(self):
        with pytest.raises(ValidationError, match="option_price"):
            implied_volatility(0.0, HULL_S, HULL_K, HULL_T, HULL_R, "call")

    def test_unknown_option_type_raises(self):
        with pytest.raises(ValidationError, match="option_type"):
            implied_volatility(4.76, HULL_S, HULL_K, HULL_T, HULL_R, "straddle")
