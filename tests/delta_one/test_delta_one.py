"""
Delta One correctness, tested against identities rather than saved numbers.

The strategy is the one `tests/analysis/test_derivatives.py` sets out: a
test asserting `basis_spread_points == 2.6278` passes forever and catches
nothing, because it re-states whatever the code did on the day it was
written. What is worth pinning here is the arithmetic that must hold no
matter what the implementation does --

  * `solve_carry` is the exact inverse of `implied_forward_price`, so
    round-tripping any rate through both returns it;
  * a forward at exactly fair value has zero basis spread, whatever the
    three components are;
  * forward carries between expiries must compose back into the total
    carry across the whole curve;
  * a hedge sized on exact arithmetic leaves exactly zero residual, and a
    rounded one leaves exactly the rounding;
  * a short's roll economics are the negation of a long's;
  * amortized execution scales as 1/horizon.

Each of those is a statement about the mathematics, so it survives a
rewrite of the code underneath it and fails if the meaning changes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from standard_quant_tools.analysis.derivatives import implied_forward_price
from standard_quant_tools.delta_one.basis import basis_history, cash_futures_basis
from standard_quant_tools.delta_one.carry import observed_carry_rate, solve_carry
from standard_quant_tools.delta_one.contracts import ContractSpec
from standard_quant_tools.delta_one.daycount import day_count, year_fraction
from standard_quant_tools.delta_one.expressions import compare_expressions
from standard_quant_tools.delta_one.futures import futures_curve, roll_analysis
from standard_quant_tools.delta_one.hedging import (
    futures_hedge,
    hedge_effectiveness,
    tracking_error,
)
from standard_quant_tools.error import ValidationError


class TestDayCount:
    def test_actual_conventions_differ_only_in_the_denominator(self):
        n365, d365 = day_count("2026-01-01", "2026-07-01", convention="ACT/365F")
        n360, d360 = day_count("2026-01-01", "2026-07-01", convention="ACT/360")
        assert n365 == n360, "ACT conventions count the same days"
        assert (d365, d360) == (365.0, 360.0)

    def test_act_act_gives_a_leap_year_its_extra_day(self):
        # 2024 is a leap year, 2023 is not. A full calendar year is exactly
        # 1.0 under ACT/ACT in both, which is the whole point of splitting
        # at the boundary rather than dividing by 365.25.
        assert year_fraction("2024-01-01", "2025-01-01", convention="ACT/ACT") == 1.0
        assert year_fraction("2023-01-01", "2024-01-01", convention="ACT/ACT") == 1.0
        # ...and ACT/365F does not, because its denominator is fixed.
        assert year_fraction("2024-01-01", "2025-01-01", convention="ACT/365F") > 1.0

    def test_reversing_the_dates_negates_the_fraction(self):
        for convention in ("ACT/365F", "ACT/360", "30/360", "ACT/ACT"):
            forward = year_fraction("2025-03-01", "2026-06-15", convention=convention)
            backward = year_fraction("2026-06-15", "2025-03-01", convention=convention)
            assert forward == pytest.approx(-backward), convention

    def test_thirty_360_makes_every_month_regular(self):
        # The convention's defining property: any two dates one month apart
        # on the same day-of-month are exactly 30/360 of a year, including
        # across February.
        assert year_fraction("2026-01-15", "2026-02-15", convention="30/360") == (
            30.0 / 360.0
        )
        assert year_fraction("2026-02-15", "2026-03-15", convention="30/360") == (
            30.0 / 360.0
        )

    def test_an_unsupported_convention_names_the_alternatives(self):
        with pytest.raises(ValidationError, match="30E/360"):
            year_fraction("2026-01-01", "2026-02-01", convention="30E/360")


class TestContractSpec:
    def test_tick_value_is_derived_and_cannot_disagree(self):
        spec = ContractSpec(symbol="ESZ5", multiplier=50, tick_size=0.25)
        assert spec.tick_value == 12.5
        # Supplying one from a different contract cannot corrupt it.
        pasted = ContractSpec.from_mapping(
            {"symbol": "ESZ5", "multiplier": 50, "tick_size": 0.25, "tick_value": 999.0}
        )
        assert pasted.tick_value == 12.5

    def test_contracts_for_inverts_notional(self):
        spec = ContractSpec(symbol="ESZ5", multiplier=50)
        exposure = 280_000_000.0
        n = spec.contracts_for(exposure, price=6200)
        assert n * spec.notional(6200) == pytest.approx(exposure)

    def test_a_missing_tick_size_gives_no_tick_value_rather_than_zero(self):
        assert ContractSpec(symbol="X", multiplier=10).tick_value is None

    def test_settlement_is_not_defaulted_to_something_wrong(self):
        with pytest.raises(ValidationError, match="physical"):
            ContractSpec(symbol="X", multiplier=10, settlement="cash-ish")


class TestCarryIsInvertible:
    @pytest.mark.parametrize(
        "target,expected",
        [
            ("financing_rate", 0.043),
            ("dividend_yield", 0.0175),
            ("borrow_rate", 0.006),
        ],
    )
    def test_solve_carry_recovers_what_the_forward_was_priced_from(
        self, target, expected
    ):
        """The inverse must undo the forward exactly, for each of the three."""
        r, q, b = 0.043, 0.0175, 0.006
        priced = implied_forward_price(
            spot=6000,
            time_to_expiry=0.25,
            risk_free_rate=r,
            dividend_yield=q,
            borrow_rate=b,
        )
        supplied = {"risk_free_rate": r, "dividend_yield": q, "borrow_rate": b}
        supplied.pop(
            {
                "financing_rate": "risk_free_rate",
                "dividend_yield": "dividend_yield",
                "borrow_rate": "borrow_rate",
            }[target]
        )
        out = solve_carry(
            spot=6000,
            forward=priced["forward"],
            time_to_expiry=0.25,
            solve_for=target,
            **supplied,
        )
        assert out["solved_rate"] == pytest.approx(expected, abs=1e-12)

    def test_net_carry_is_the_sum_of_the_three_components(self):
        r, q, b = 0.05, 0.02, 0.01
        priced = implied_forward_price(
            spot=100,
            time_to_expiry=1.5,
            risk_free_rate=r,
            dividend_yield=q,
            borrow_rate=b,
        )
        observed = observed_carry_rate(
            spot=100, forward=priced["forward"], time_to_expiry=1.5
        )
        assert observed == pytest.approx(r - q - b, abs=1e-12)

    def test_the_other_two_rates_are_required_rather_than_assumed_zero(self):
        with pytest.raises(ValidationError, match="dividend"):
            solve_carry(
                spot=100,
                forward=101,
                time_to_expiry=0.5,
                solve_for="borrow_rate",
                risk_free_rate=0.05,
            )


class TestBasis:
    def test_a_future_at_fair_value_has_no_spread(self):
        """Whatever the components are, a quote AT fair is fair."""
        for r, q, b in [(0.04, 0.0, 0.0), (0.05, 0.02, 0.01), (0.01, 0.03, 0.0)]:
            fair = implied_forward_price(
                spot=4000,
                time_to_expiry=0.4,
                risk_free_rate=r,
                dividend_yield=q,
                borrow_rate=b,
            )["forward"]
            out = cash_futures_basis(
                spot=4000,
                future_price=fair,
                time_to_expiry=0.4,
                risk_free_rate=r,
                dividend_yield=q,
                borrow_rate=b,
            )
            assert out["basis_spread_points"] == pytest.approx(0.0, abs=1e-9)
            assert out["annualized_basis_spread_bps"] == pytest.approx(0.0, abs=1e-9)
            assert out["classification"] == "fair"

    def test_the_implied_financing_makes_the_quote_fair_by_construction(self):
        out = cash_futures_basis(
            spot=6000,
            future_price=6055,
            time_to_expiry=0.25,
            risk_free_rate=0.043,
            dividend_yield=0.0175,
        )
        # Re-pricing at the implied rate must reproduce the quoted future.
        reprice = implied_forward_price(
            spot=6000,
            time_to_expiry=0.25,
            risk_free_rate=out["implied_financing_rate"],
            dividend_yield=0.0175,
            borrow_rate=0.0,
        )
        assert reprice["forward"] == pytest.approx(6055, rel=1e-9)

    def test_rich_and_cheap_are_the_two_sides_of_the_tolerance(self):
        common = dict(
            spot=6000, time_to_expiry=0.25, risk_free_rate=0.043, dividend_yield=0.0175
        )
        fair = implied_forward_price(
            spot=6000,
            time_to_expiry=0.25,
            risk_free_rate=0.043,
            dividend_yield=0.0175,
        )["forward"]
        assert (
            cash_futures_basis(future_price=fair * 1.01, tolerance_bps=1.0, **common)[
                "classification"
            ]
            == "future_rich"
        )
        assert (
            cash_futures_basis(future_price=fair * 0.99, tolerance_bps=1.0, **common)[
                "classification"
            ]
            == "future_cheap"
        )

    def test_a_constant_basis_series_has_no_dispersion(self):
        spot = np.linspace(5000, 6000, 120)
        futures = spot * 1.003  # exactly 30 bps, every day
        out = basis_history(spot=spot, futures=futures)
        assert out["std_bps"] == pytest.approx(0.0, abs=1e-9)
        assert out["current_basis_bps"] == pytest.approx(30.0, abs=1e-6)

    def test_misaligned_legs_are_refused_rather_than_truncated(self):
        with pytest.raises(ValidationError, match="aligned"):
            basis_history(spot=[100, 101, 102], futures=[100, 101])


class TestFuturesCurve:
    def test_forward_carries_compose_into_the_total(self):
        """
        The identity that makes a curve a curve: carry across the whole
        span is the time-weighted sum of the forward carries between the
        contracts, exactly as forward variance adds in the vol runtime.
        """
        contracts = [
            {"label": "a", "time_to_expiry": 0.1, "price": 6010.0},
            {"label": "b", "time_to_expiry": 0.35, "price": 6055.0},
            {"label": "c", "time_to_expiry": 0.80, "price": 6130.0},
        ]
        out = futures_curve(contracts)
        total_span = 0.80 - 0.10
        composed = sum(
            s["forward_carry_rate"] * s["years_between"]
            for s in out["calendar_spreads"]
        )
        direct = math.log(6130.0 / 6010.0)
        assert composed == pytest.approx(direct, abs=1e-12)
        assert out["curve_slope_rate"] == pytest.approx(direct / total_span, rel=1e-12)

    def test_contracts_out_of_order_give_the_same_curve(self):
        rows = [
            {"label": "c", "time_to_expiry": 0.80, "price": 6130.0},
            {"label": "a", "time_to_expiry": 0.10, "price": 6010.0},
            {"label": "b", "time_to_expiry": 0.35, "price": 6055.0},
        ]
        out = futures_curve(rows)
        assert [p["label"] for p in out["curve"]] == ["a", "b", "c"]
        assert out["shape"] == "contango"

    def test_a_falling_curve_is_backwardation(self):
        out = futures_curve(
            [
                {"time_to_expiry": 0.1, "price": 100.0},
                {"time_to_expiry": 0.5, "price": 98.0},
            ]
        )
        assert out["shape"] == "backwardation"
        assert out["calendar_spreads"][0]["forward_carry_rate"] < 0


class TestRoll:
    def test_a_short_roll_is_the_negation_of_a_long_one(self):
        common = dict(
            front_price=6041, next_price=6072, multiplier=50, days_to_front_expiry=9
        )
        long = roll_analysis(contracts_held=100, **common)
        short = roll_analysis(contracts_held=-100, **common)
        assert long["cash_impact"] == pytest.approx(-short["cash_impact"])

    def test_the_next_leg_holds_the_same_money_across_a_multiplier_change(self):
        out = roll_analysis(
            front_price=6000,
            next_price=6000,
            contracts_held=10,
            multiplier=50,
            next_multiplier=5,
            days_to_front_expiry=5,
        )
        # A micro contract is a tenth the size, so it takes ten times as many.
        assert out["next_contracts_exact"] == pytest.approx(100.0)

    def test_rolling_flat_costs_only_execution(self):
        out = roll_analysis(
            front_price=6000,
            next_price=6000,
            contracts_held=10,
            multiplier=50,
            days_to_front_expiry=5,
            # roll_yield is annualized over the gap BETWEEN the expiries, so
            # it needs that gap; without it the field is null rather than
            # annualized by the front's remaining life, which made the
            # answer depend on the roll date.
            days_between_expiries=91.0,
            cost_per_contract=2.0,
        )
        assert out["cash_impact"] == pytest.approx(0.0)
        assert out["roll_yield_rate"] == pytest.approx(0.0)
        assert out["net_roll_cost"] == pytest.approx(out["execution_cost"])

    def test_a_zero_position_is_not_a_roll(self):
        with pytest.raises(ValidationError, match="position"):
            roll_analysis(
                front_price=100,
                next_price=101,
                contracts_held=0,
                multiplier=1,
                days_to_front_expiry=1,
            )


class TestHedgeSizing:
    def test_the_exact_hedge_leaves_exactly_the_rounding(self):
        out = futures_hedge(
            portfolio_value=250e6,
            portfolio_beta=1.12,
            future_price=6200,
            multiplier=50,
        )
        residual_from_rounding = (
            out["contracts_rounded"] - out["contracts_exact"]
        ) * out["contract_notional"]
        assert out["residual_dollar_beta"] == pytest.approx(residual_from_rounding)

    def test_an_unrounded_hedge_is_exactly_flat(self):
        out = futures_hedge(
            portfolio_value=250e6,
            portfolio_beta=1.12,
            future_price=6200,
            multiplier=50,
        )
        exact_residual = (
            out["dollar_beta"] + out["contracts_exact"] * out["exposure_per_contract"]
        )
        assert exact_residual == pytest.approx(0.0, abs=1e-6)

    def test_dollar_neutral_and_beta_neutral_agree_only_at_beta_one(self):
        common = dict(portfolio_value=100e6, future_price=5000, multiplier=50)
        at_one = [
            futures_hedge(portfolio_beta=1.0, objective=o, **common)["contracts_exact"]
            for o in ("beta_neutral", "dollar_neutral")
        ]
        assert at_one[0] == pytest.approx(at_one[1])
        off_one = [
            futures_hedge(portfolio_beta=1.4, objective=o, **common)["contracts_exact"]
            for o in ("beta_neutral", "dollar_neutral")
        ]
        assert off_one[0] != pytest.approx(off_one[1])
        assert off_one[0] == pytest.approx(off_one[1] * 1.4)

    def test_a_hedge_instrument_with_higher_beta_needs_fewer_contracts(self):
        common = dict(
            portfolio_value=100e6,
            portfolio_beta=1.0,
            future_price=5000,
            multiplier=50,
        )
        low = futures_hedge(future_beta=1.0, **common)["contracts_exact"]
        high = futures_hedge(future_beta=1.25, **common)["contracts_exact"]
        assert abs(high) < abs(low)
        assert high == pytest.approx(low / 1.25)

    def test_existing_contracts_turn_the_target_into_a_trade(self):
        out = futures_hedge(
            portfolio_value=100e6,
            portfolio_beta=1.0,
            future_price=5000,
            multiplier=50,
            existing_contracts=-200,
        )
        assert out["trade_contracts_exact"] == pytest.approx(
            out["contracts_exact"] + 200
        )

    def test_an_instrument_with_no_exposure_cannot_hedge(self):
        with pytest.raises(ValidationError, match="undefined"):
            futures_hedge(
                portfolio_value=1e6,
                portfolio_beta=1.0,
                future_price=100,
                multiplier=1,
                future_beta=0.0,
            )


class TestHedgeEffectiveness:
    def test_a_perfect_hedge_removes_all_the_variance(self):
        rng = np.random.default_rng(11)
        market = rng.normal(0, 0.01, 400)
        portfolio = 1.3 * market  # no idiosyncratic component at all
        out = hedge_effectiveness(
            portfolio_returns=portfolio, hedge_returns=market, hedge_ratio=-1.3
        )
        assert out["volatility_after"] == pytest.approx(0.0, abs=1e-12)
        assert out["volatility_reduction_pct"] == pytest.approx(100.0)

    def test_the_wrong_sign_makes_it_worse_and_says_so(self):
        rng = np.random.default_rng(5)
        market = rng.normal(0, 0.01, 300)
        portfolio = market + rng.normal(0, 0.002, 300)
        out = hedge_effectiveness(
            portfolio_returns=portfolio, hedge_returns=market, hedge_ratio=+1.0
        )
        assert out["volatility_after"] > out["volatility_before"]
        assert any("sign error" in w for w in out["warnings"])

    def test_tracking_error_is_zero_against_itself(self):
        rng = np.random.default_rng(2)
        series = rng.normal(0, 0.01, 200)
        assert tracking_error(series, series) == pytest.approx(0.0, abs=1e-12)

    def test_tracking_error_scales_with_the_square_root_of_frequency(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0, 0.01, 500)
        b = rng.normal(0, 0.01, 500)
        daily = tracking_error(a, b, periods_per_year=252)
        monthly = tracking_error(a, b, periods_per_year=12)
        assert daily / monthly == pytest.approx(math.sqrt(252 / 12))


class TestExpressionComparison:
    def _exprs(self):
        return [
            {
                "label": "cash",
                "kind": "cash",
                "financing_rate": 0.045,
                "dividend_yield": 0.0175,
                "spread_bps": 2.0,
            },
            {
                "label": "etf",
                "kind": "etf",
                "financing_rate": 0.045,
                "dividend_yield": 0.017,
                "fee_rate": 0.0009,
                "spread_bps": 0.3,
            },
        ]

    def test_amortized_execution_scales_as_one_over_the_horizon(self):
        short = compare_expressions(self._exprs(), notional=1e6, horizon_years=0.25)
        long = compare_expressions(self._exprs(), notional=1e6, horizon_years=1.0)
        for a, b in zip(
            sorted(short["expressions"], key=lambda r: r["label"]),
            sorted(long["expressions"], key=lambda r: r["label"]),
        ):
            assert a["execution_bps"] == pytest.approx(4.0 * b["execution_bps"])
            # Carry does not move with the horizon; only execution does.
            assert a["carry_bps"] == pytest.approx(b["carry_bps"])

    def test_the_horizon_can_reverse_the_ranking(self):
        """The claim the tool exists to make, as an assertion."""
        exprs = [
            # Cheap to hold, expensive to trade.
            {
                "label": "basket",
                "kind": "cash",
                "financing_rate": 0.040,
                "dividend_yield": 0.018,
                "spread_bps": 4.0,
            },
            # Dearer to hold, almost free to trade.
            {
                "label": "etf",
                "kind": "etf",
                "financing_rate": 0.045,
                "dividend_yield": 0.017,
                "spread_bps": 0.2,
            },
        ]
        brief = compare_expressions(exprs, notional=1e6, horizon_years=1 / 12)
        patient = compare_expressions(exprs, notional=1e6, horizon_years=3.0)
        assert brief["cheapest"] == "etf"
        assert patient["cheapest"] == "basket"

    def test_direction_flips_every_carry_sign(self):
        long = compare_expressions(self._exprs(), notional=1e6, horizon_years=1.0)
        short = compare_expressions(
            self._exprs(), notional=1e6, horizon_years=1.0, direction="short"
        )
        for a, b in zip(
            sorted(long["expressions"], key=lambda r: r["label"]),
            sorted(short["expressions"], key=lambda r: r["label"]),
        ):
            assert a["carry_bps"] == pytest.approx(-b["carry_bps"])
            # Execution is a cost either way -- it does not flip.
            assert a["execution_bps"] == pytest.approx(b["execution_bps"])

    def test_an_unpriced_term_is_refused_rather_than_ignored(self):
        with pytest.raises(ValidationError, match="unknown fields"):
            compare_expressions(
                [{"label": "x", "kind": "cash", "expense_ratio": 0.001}],
                notional=1e6,
                horizon_years=1.0,
            )
