"""
An input that can only be a mistake is refused, not answered with nulls.

A margin rate of 1e308, a future quoted at 1e308, spread prices scaled by
1e300, a NaN holding, a spread of 1e308 bps and a risk-free rate of 1e308
each ran, overflowed somewhere inside, and came back as null numbers with a
reason. A null with a reason is the right answer when a legal input has no
defined result -- the Sharpe ratio of a flat market -- and the wrong one
when the input names nothing a market produces.

Every bound is the library's own: rates are decimals within +/-10
(1,000%), prices within the 1e12 the option pricers use, a spread at most
the whole notional. Each refusal has its null case: the largest plausible
value is still accepted and answered.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.models import (
    EfficientFrontierInput,
    EstimateTradeCostInput,
    PlanRebalanceInput,
    PortfolioSimulationInput,
)
from standard_quant_tools.agent.runtimes.delta_one import tools as D
from standard_quant_tools.agent.runtimes.delta_one.models import (
    CashFuturesBasisInput,
    EtfFairValueInput,
    SpreadMonitorInput,
)
from standard_quant_tools.agent.runtimes.portfolio import tools as P

NAN, INF = float("nan"), float("inf")


def _simulation(**overrides) -> PortfolioSimulationInput:
    return PortfolioSimulationInput(
        tickers=["AAA", "BBB"],
        start_date="2024-01-02",
        end_date="2024-06-28",
        target_weights={"AAA": {"2024-01-02": 0.5}, "BBB": {"2024-01-02": 0.5}},
        **overrides,
    )


class TestFinancingAndRiskFreeRates:
    @pytest.mark.parametrize("rate", [1e308, 10.5])
    def test_a_margin_rate_past_the_bound_is_refused(self, rate):
        with pytest.raises(PydanticValidationError, match="margin_interest_rate"):
            _simulation(margin_interest_rate=rate)

    @pytest.mark.parametrize("rate", [0.0, 0.085, 10.0])
    def test_a_margin_rate_a_broker_could_charge_is_accepted(self, rate):
        """The null case, up to the bound itself."""
        assert _simulation(margin_interest_rate=rate).margin_interest_rate == rate

    @pytest.mark.parametrize("rate", [1e308, -1e308, 10.01, -10.01])
    def test_a_frontier_rate_past_the_bound_is_refused(self, rate):
        with pytest.raises(PydanticValidationError, match="risk_free_rate"):
            EfficientFrontierInput(
                tickers=["AAA", "BBB"],
                start_date="2024-01-02",
                end_date="2024-06-28",
                risk_free_rate=rate,
            )

    @pytest.mark.parametrize("rate", [-0.0075, 0.0, 0.045, 10.0, -10.0])
    def test_a_negative_or_ordinary_frontier_rate_is_accepted(self, rate):
        """The null case: bounded on magnitude, never on sign."""
        model = EfficientFrontierInput(
            tickers=["AAA", "BBB"],
            start_date="2024-01-02",
            end_date="2024-06-28",
            risk_free_rate=rate,
        )
        assert model.risk_free_rate == rate


class TestTradeCostSpread:
    def test_a_spread_past_the_whole_notional_is_refused(self):
        with pytest.raises(PydanticValidationError, match="spread_bps"):
            EstimateTradeCostInput(notional=1_000_000.0, spread_bps=1e308)

    def test_a_spread_of_the_whole_notional_is_priced(self):
        """The null case: at the bound the breakeven move is finite."""
        result = P.estimate_trade_cost(
            EstimateTradeCostInput(
                notional=1_000_000.0, commission_model="none", spread_bps=10_000
            )
        )
        assert result.breakeven_move_bps == pytest.approx(20_000.0)
        assert not any(" is null" in note for note in result.notes)


class TestRebalanceHoldings:
    @pytest.mark.parametrize("field", ["current_weights", "target_weights", "adv"])
    @pytest.mark.parametrize("bad", [NAN, INF])
    def test_a_non_finite_holding_is_refused_by_name(self, field, bad):
        arguments = {
            "current_weights": {"AAA": 0.5, "BBB": 0.5},
            "target_weights": {"AAA": 0.2, "BBB": 0.8},
            "adv": {"AAA": 5e6, "BBB": 5e6},
            "portfolio_value": 1_000_000.0,
        }
        arguments[field] = {**arguments[field], "BBB": bad}
        with pytest.raises(PydanticValidationError) as excinfo:
            PlanRebalanceInput(**arguments)
        assert f"{field} is not finite at key(s) ['BBB']" in str(excinfo.value)

    def test_finite_holdings_are_planned(self):
        """The null case: a short and an ordinary long both plan."""
        result = P.plan_rebalance(
            PlanRebalanceInput(
                current_weights={"AAA": 0.5, "BBB": 0.5},
                target_weights={"AAA": -0.2, "BBB": 1.2},
                adv={"AAA": 5e6, "BBB": 5e6},
                portfolio_value=1_000_000.0,
            )
        )
        assert math.isfinite(result.total_turnover)
        assert not any(" is null" in line for line in result.warnings)

    def test_a_plan_already_at_target_has_no_residual(self):
        """It had none to report and reported null, which the result then
        explained as a gap too large to represent."""
        result = P.plan_rebalance(
            PlanRebalanceInput(
                current_weights={"AAA": 0.5, "BBB": 0.5},
                target_weights={"AAA": 0.5, "BBB": 0.5},
                portfolio_value=1_000_000.0,
            )
        )
        assert result.converged is True
        assert result.residual_distance == 0.0
        assert not any(" is null" in line for line in result.warnings)


class TestDeltaOnePrices:
    @pytest.mark.parametrize(
        "field,value",
        [("future_price", 1e308), ("spot", 1e308), ("future_price", 1.1e12)],
    )
    def test_a_basis_price_past_any_market_is_refused(self, field, value):
        arguments = dict(
            spot=5_000.0, future_price=5_050.0, time_to_expiry=0.25, risk_free_rate=0.04
        )
        arguments[field] = value
        with pytest.raises(PydanticValidationError, match=field):
            CashFuturesBasisInput(**arguments)

    def test_the_largest_quoted_prices_still_price(self):
        """The null case: a share class at 700,000 and its future."""
        result = D.analyze_cash_futures_basis(
            CashFuturesBasisInput(
                spot=700_000.0,
                future_price=707_000.0,
                time_to_expiry=0.5,
                risk_free_rate=0.04,
            )
        )
        assert math.isfinite(result.carry_spread_rate)
        assert not any(" is null" in line for line in result.warnings)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("etf_price", 1e308),
            ("nav", 1e308),
            ("basket_value", 1e308),
            ("cash_component", -1e308),
            ("etf_spread_bps", 1e308),
            ("basket_spread_bps", 1e308),
        ],
    )
    def test_an_etf_input_past_any_market_is_refused(self, field, value):
        with pytest.raises(PydanticValidationError, match=field):
            EtfFairValueInput(**{"etf_price": 100.3, "nav": 100.0, field: value})

    def test_an_ordinary_fund_is_priced_with_its_costs(self):
        """The null case."""
        result = D.analyze_etf_fair_value(
            EtfFairValueInput(
                etf_price=100.3,
                nav=100.0,
                basket_value=100.05,
                cash_component=-0.02,
                etf_spread_bps=2.0,
                basket_spread_bps=3.0,
            )
        )
        assert math.isfinite(result.premium_vs_reference_bps)
        assert not any(" is null" in line for line in result.warnings)


class TestSpreadMonitorPrices:
    @staticmethod
    def _monitor(**overrides):
        prices = [100.0 + 0.01 * i for i in range(12)]
        arguments = dict(
            primary_prices=[p * 1.001 for p in prices],
            reference_prices=prices,
            warmup=10,
        )
        arguments.update(overrides)
        return SpreadMonitorInput(**arguments)

    def test_a_leg_scaled_past_any_market_is_refused_by_position(self):
        with pytest.raises(PydanticValidationError) as excinfo:
            self._monitor(primary_prices=[p * 1e300 for p in range(1, 13)])
        assert "primary_prices has 12 price(s)" in str(excinfo.value)

    @pytest.mark.parametrize("bad", [NAN, INF, -1.5e12])
    def test_a_single_bad_tick_is_named(self, bad):
        prices = [100.0] * 12
        prices[7] = bad
        with pytest.raises(PydanticValidationError, match=r"position\(s\) \[7\]"):
            self._monitor(reference_prices=prices)

    def test_a_negative_leg_is_still_a_price_in_points(self):
        """The null case: absolute_points takes a leg below zero -- a front
        contract that settled at -37.63 -- and the bound is on magnitude."""
        front = [-37.63 + 0.5 * i for i in range(12)]
        result = D.monitor_spread_stream(
            self._monitor(
                primary_prices=[p + 5.0 for p in front],
                reference_prices=front,
                channel="absolute_points",
            )
        )
        assert result.state["n"] == 12
        assert not any(" is null" in line for line in result.warnings)
