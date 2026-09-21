"""
Phase 5 of Development/databento_live_fix_plan.md: options and futures.

The live findings (Development/databento_live_findings.md, D9, D13 and
"Also in options and futures") measured each of these on real prices. The
tests here reproduce each defect's shape offline and pin the fix:

  D9     the IV solver converges on volatility, never on an untaken step
  D13    a price at the no-arbitrage bound is admitted; 0.0 is refused
  smile  an arbitrage violation is reported by moneyness and strike
  scan   analyze_strategy scans from zero, so a put's worst case is found
  carry  the three carry components sum to the basis
  model  bachelier refuses a dividend it cannot use
  roll   roll_analysis refuses spread_ticks without tick_value
  drift  the drift band watches the hedge actually held
  engine the futures engine books the roll day when it can, and fills
         what the account can margin
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.derivatives import (
    analyze_strategy,
    fit_volatility_smile,
    implied_forward_price,
)
from standard_quant_tools.analysis.options import (
    black_scholes_price,
    implied_volatility,
)
from standard_quant_tools.analysis.pricing import price_option
from standard_quant_tools.backtest.futures_engine import run_futures_simulation
from standard_quant_tools.backtest.futures_hedge_backtest import (
    run_futures_hedge_backtest,
)
from standard_quant_tools.delta_one.futures import roll_analysis
from standard_quant_tools.error import ValidationError

# ── D9 ───────────────────────────────────────────────────────────────────


class TestTheSolverConvergesOnVolatility:
    """Four short-dated puts came back at exactly the initial guess of 0.2
    with converged=True, for true vols of 3.00, 1.20 and 0.45; over a
    700-case grid 28 'converged' answers were off by more than 0.01."""

    @pytest.mark.parametrize(
        "spot,strike,expiry,true_vol",
        [
            (260.0, 250.0, 0.000274, 3.00),
            (505.0, 500.0, 0.0027, 1.20),
            (600.0, 650.0, 0.05, 0.45),
        ],
    )
    def test_the_findings_cases_return_the_true_volatility(
        self, spot, strike, expiry, true_vol
    ):
        rate = 0.04
        price = black_scholes_price(spot, strike, expiry, rate, true_vol, "put")
        result = implied_volatility(price, spot, strike, expiry, rate, "put")
        assert result["converged"]
        assert result["implied_volatility"] == pytest.approx(true_vol, abs=1e-4)
        assert result["implied_volatility"] != pytest.approx(0.2, abs=1e-6)

    def test_no_converged_answer_on_the_grid_is_off_by_more_than_the_tolerance(self):
        spot, rate = 100.0, 0.03
        strikes = [40, 60, 80, 90, 100, 110, 120, 150, 200, 300]
        expiries = [0.001, 0.01, 0.05, 0.25, 1.0, 3.0, 5.0]
        vols = [0.05, 0.15, 0.3, 0.6, 1.0, 2.0, 3.0]
        checked = refused = 0
        for strike, expiry, vol, kind in itertools.product(
            strikes, expiries, vols, ("call", "put")
        ):
            price = black_scholes_price(spot, strike, expiry, rate, vol, kind)
            try:
                result = implied_volatility(price, spot, strike, expiry, rate, kind)
            except ValidationError:
                refused += 1  # an underflowed or unidentifiable price, by name
                continue
            if not result["converged"] or result["at_bound"]:
                continue
            checked += 1
            assert result["implied_volatility"] == pytest.approx(vol, abs=1e-4), (
                strike,
                expiry,
                vol,
                kind,
                result,
            )
        assert checked > 400
        assert refused < 400

    def test_the_result_reports_its_pricing_error(self):
        price = black_scholes_price(42.0, 40.0, 0.5, 0.1, 0.2, "call")
        result = implied_volatility(price, 42.0, 40.0, 0.5, 0.1, "call")
        assert result["price_error"] < 1e-8
        assert result["at_bound"] is False


# ── D13 ──────────────────────────────────────────────────────────────────


class TestTheBoundAdmitsItsOwnPrices:
    """77 of 700 prices the pricer had produced were refused on a strict
    bound; a deep-in-the-money call prices bit-for-bit equal to intrinsic."""

    def test_a_price_at_intrinsic_is_admitted_and_flagged(self):
        spot, strike, expiry, rate = 340.0, 200.0, 0.25, 0.04
        price = black_scholes_price(spot, strike, expiry, rate, 0.08, "call")
        lower = spot - strike * np.exp(-rate * expiry)
        assert price == pytest.approx(lower, abs=1e-9)
        result = implied_volatility(price, spot, strike, expiry, rate, "call")
        assert result["at_bound"] is True
        assert result["converged"]
        # A ceiling: every volatility at or below it prices to intrinsic.
        assert 0.0 < result["implied_volatility"] < 0.5
        assert result["price_error"] <= 1e-6

    def test_an_underflowed_price_is_refused_by_name(self):
        with pytest.raises(ValidationError, match="underflows"):
            implied_volatility(0.0, 100.0, 300.0, 0.001, 0.03, "call")

    def test_a_price_well_outside_the_bound_is_still_refused(self):
        with pytest.raises(ValidationError, match="no-arbitrage"):
            implied_volatility(50.0, 42.0, 40.0, 0.5, 0.1, "call")


# ── the smile ────────────────────────────────────────────────────────────


class TestTheSmileNamesMoneynessAndStrike:
    """A trader was told the arbitrage sat at k=1.00 while strike_range in
    the same payload said [300, 370]."""

    FORWARD, T = 100.0, 0.5
    STRIKES = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)

    def test_a_violation_carries_both(self):
        x = np.log(self.STRIKES / self.FORWARD)
        vols = 0.25 - 0.30 * x - 4.0 * x**2  # concave: a negative density
        result = fit_volatility_smile(
            self.STRIKES, vols, forward=self.FORWARD, time_to_expiry=self.T
        )
        assert result["arbitrage_violations"]
        first = result["arbitrage_violations"][0]
        assert 0.7 < first["moneyness"] < 1.3
        assert first["strike"] == pytest.approx(first["moneyness"] * self.FORWARD)
        assert 80 <= first["strike"] <= 120
        assert (
            "strike" in result["warnings"][0] and "moneyness" in result["warnings"][0]
        )


# ── the scan ─────────────────────────────────────────────────────────────


def _leg(kind, strike, qty, vol=0.25, t=0.5):
    return {
        "option_type": kind,
        "strike": strike,
        "quantity": qty,
        "volatility": vol,
        "time_to_expiry": t,
    }


class TestTheScanStartsAtZero:
    """The scan started at half the lowest strike, so a long put's worst
    case sat outside it: max_loss came back at half the truth and a bounded
    loss was labelled unbounded."""

    def test_a_short_put_has_a_bounded_worst_case_at_zero(self):
        result = analyze_strategy([_leg("put", 100, -1)], spot=100.0)
        assert not result["max_loss_unbounded"]
        assert result["max_loss_at_spot"] == 0.0
        # Worst case: the put is exercised at zero, less the premium received.
        assert result["max_loss"] == pytest.approx(
            -100.0 - result["net_premium"], abs=1e-6
        )

    def test_a_short_call_is_still_unbounded(self):
        result = analyze_strategy([_leg("call", 100, -1)], spot=100.0)
        assert result["max_loss_unbounded"]

    def test_a_long_put_reaches_its_full_profit(self):
        result = analyze_strategy([_leg("put", 100, 1)], spot=100.0)
        assert not result["max_profit_unbounded"]
        assert result["max_profit"] == pytest.approx(
            100.0 - result["net_premium"], abs=1e-6
        )


# ── carry ────────────────────────────────────────────────────────────────


class TestTheCarryComponentsSumToTheBasis:
    """Three decompositions were up to 45% short of the basis they
    decomposed, because each rate was compounded alone."""

    @pytest.mark.parametrize(
        "rate,dividend,borrow,expiry",
        [(0.05, 0.02, 0.01, 1.5), (0.10, 0.08, 0.05, 5.0), (0.03, 0.0, 0.0, 0.25)],
    )
    def test_they_sum_exactly(self, rate, dividend, borrow, expiry):
        out = implied_forward_price(
            spot=100.0,
            time_to_expiry=expiry,
            risk_free_rate=rate,
            dividend_yield=dividend,
            borrow_rate=borrow,
        )
        total = sum(out["components"].values())
        assert total == pytest.approx(out["basis"], abs=1e-9)
        assert out["components_order"] == ["financing", "dividend", "borrow"]


class TestBachelierRefusesADividend:
    def test_a_dividend_is_refused_by_name(self):
        with pytest.raises(ValidationError, match="bachelier"):
            price_option(
                spot=80.0,
                strike=80.0,
                time_to_expiry=0.5,
                volatility=12.0,
                risk_free_rate=0.03,
                model="bachelier",
                dividend_yield=0.02,
            )

    def test_without_a_dividend_it_prices(self):
        out = price_option(
            spot=80.0,
            strike=80.0,
            time_to_expiry=0.5,
            volatility=12.0,
            risk_free_rate=0.03,
            model="bachelier",
        )
        assert out["price"] > 0


class TestRollAnalysisNeedsATickValue:
    def test_spread_ticks_without_tick_value_is_refused(self):
        with pytest.raises(ValidationError, match="tick_value"):
            roll_analysis(
                front_price=5000.0,
                next_price=5010.0,
                contracts_held=10,
                multiplier=50,
                days_to_front_expiry=10,
                spread_ticks=1.0,
            )

    def test_with_a_tick_value_the_spread_is_charged(self):
        out = roll_analysis(
            front_price=5000.0,
            next_price=5010.0,
            contracts_held=10,
            multiplier=50,
            days_to_front_expiry=10,
            spread_ticks=1.0,
            tick_value=12.5,
        )
        # Both legs cross the spread: the ten held and the 9.98 rolled into.
        assert out["execution_cost"] == pytest.approx(
            (10 + 10 * 5000.0 / 5010.0) * 12.5, rel=1e-9
        )


# ── the drift band ───────────────────────────────────────────────────────


class TestTheDriftBandWatchesTheHeldHedge:
    """The rule measured the rounding residual of a fresh hedge, bounded by
    half a contract, so it could never fire: it sat through 81.8% residual
    beta on a 5% band."""

    def test_a_growing_book_trips_the_band(self):
        # A book that doubles while the future is flat: the held hedge
        # covers half the exposure by the end, far outside a 5% band.
        dates = list(pd.bdate_range("2024-01-02", periods=252))
        growing = {d: 10_000_000.0 * (1 + i / len(dates)) for i, d in enumerate(dates)}
        flat = {d: 5000.0 for d in dates}
        out = run_futures_hedge_backtest(
            portfolio_values=growing,
            future_prices=flat,
            multiplier=50,
            rehedge="drift",
            drift_band=0.05,
        )
        assert out["n_rehedges"] > 5
        # And the held hedge's residual never sits far above the band.
        assert out["held_residual_fraction_max"] < 0.06

    def test_a_flat_book_never_trips_it(self):
        dates = pd.bdate_range("2024-01-02", periods=100)
        flat_book = {d: 10_000_000.0 for d in dates}
        flat_fut = {d: 5000.0 for d in dates}
        out = run_futures_hedge_backtest(
            portfolio_values=flat_book,
            future_prices=flat_fut,
            multiplier=50,
            rehedge="drift",
            drift_band=0.05,
        )
        assert out["n_rehedges"] == 1


# ── the futures engine ───────────────────────────────────────────────────


def _prices(n: int = 20, start: float = 5000.0, step: float = 5.0):
    dates = pd.bdate_range("2024-01-02", periods=n)
    return {d: start + step * i for i, d in enumerate(dates)}


class TestTheEngineBooksTheRollDayWhenItCan:
    """The roll day's variation margin was skipped -- $7,025 per contract
    per year on a live ES year -- because a single series cannot hold the
    old contract's move."""

    def test_the_old_contracts_close_is_booked(self):
        prices = _prices()
        keys = list(prices)
        contract_map = {k: ("A" if i < 10 else "B") for i, k in enumerate(keys)}
        roll_day = keys[10]
        without = run_futures_simulation(
            prices=prices,
            target_contracts={k: 4.0 for k in prices},
            contract_map=contract_map,
            multiplier=50,
            initial_capital=1_000_000,
        )
        with_prior = run_futures_simulation(
            prices=prices,
            target_contracts={k: 4.0 for k in prices},
            contract_map=contract_map,
            multiplier=50,
            initial_capital=1_000_000,
            roll_day_prior_prices={roll_day: prices[keys[9]] + 5.0},
        )
        assert without["rolls"][0]["variation_margin_skipped"] is True
        assert any("roll day" in w for w in without["warnings"])
        assert with_prior["rolls"][0]["variation_margin_skipped"] is False
        # One day's move of 5 points on 4 contracts at 50 per point.
        assert with_prior["total_variation_margin"] - without[
            "total_variation_margin"
        ] == pytest.approx(5.0 * 4 * 50)
        assert not any("roll day" in w for w in with_prior["warnings"])


class TestAFillIsWhatTheAccountCanMargin:
    """A target the account could not margin was filled and liquidated in
    the same bar with both legs charged: 239 margin calls and 52.6% of
    starting capital in fees on a live ES year."""

    def test_an_unaffordable_target_is_sized_down_and_recorded(self):
        prices = _prices(n=10, step=0.0)
        out = run_futures_simulation(
            prices=prices,
            target_contracts={k: 100.0 for k in prices},
            multiplier=50,
            initial_capital=300_000,
            initial_margin=15_000,
            maintenance_margin=12_000,
            commission_per_contract=2.5,
        )
        (fill,) = out["margin_limited_fills"][:1]
        assert fill["requested"] == 100.0 and fill["filled"] == 20.0
        assert out["n_margin_calls"] == 0
        assert out["position_curve"].iloc[-1] == 20.0
        # Only the twenty contracts actually filled paid commission.
        assert out["total_commission"] == pytest.approx(20 * 2.5)
        assert any("sized down" in w for w in out["warnings"])

    def test_an_affordable_target_fills_in_full(self):
        prices = _prices(n=10, step=0.0)
        out = run_futures_simulation(
            prices=prices,
            target_contracts={k: 10.0 for k in prices},
            multiplier=50,
            initial_capital=300_000,
            initial_margin=15_000,
        )
        assert out["margin_limited_fills"] == []
        assert out["position_curve"].iloc[-1] == 10.0
