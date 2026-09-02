"""
estimate_trade_cost and compare_cost_models.

estimate_trade_cost is pure arithmetic over backtest/costs.py, so its tests
assert the tool agrees with those functions called directly rather than
re-deriving the formulas -- a test that recomputed sqrt-impact by hand
would pass while the tool called the wrong function with the right shape.

compare_cost_models is tested against the property it relies on: for a
fixed signal series, total return is strictly decreasing in the commission
rate. That monotonicity is what makes the breakeven a bisection rather than
a search, so it is the thing worth pinning. The backtest itself runs on a
synthetic price frame with a stubbed provider -- these tests are about the
sweep, not about any market.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.backtest import costs
from standard_quant_tools.error import ValidationError


@pytest.fixture
def trending_prices():
    """A price path with enough structure for an SMA crossover to trade."""
    rng = np.random.default_rng(7)
    n = 400
    dates = pd.bdate_range("2021-01-04", periods=n)
    drift = np.linspace(0.0, 0.6, n)
    wobble = np.sin(np.linspace(0, 14 * np.pi, n)) * 0.12
    close = 100.0 * np.exp(drift + wobble + rng.normal(0, 0.004, n).cumsum())
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 5_000_000.0),
        },
        index=dates,
    )


@pytest.fixture
def stub_provider(monkeypatch, trending_prices):
    class _Stub:
        def get_ohlcv(self, symbol, start_date, end_date, interval="1d"):
            return trending_prices

    monkeypatch.setattr(
        "standard_quant_tools.agent.runtimes.backtest.tools.DataFactory.get_provider",
        staticmethod(lambda *a, **k: _Stub()),
    )
    return trending_prices


class TestEstimateTradeCost:
    def test_percentage_commission_matches_the_library_function(self):
        result = dispatch(
            "estimate_trade_cost",
            {"notional": 100_000, "commission_pct": 0.001, "spread_model": "none"},
        )
        assert result["legs"][0]["cost"] == pytest.approx(
            costs.percentage_commission(100_000, 0.001)
        )
        assert result["total_bps"] == pytest.approx(10.0)

    def test_directional_charges_the_side_it_was_given(self):
        common = {
            "notional": 100_000,
            "commission_model": "directional",
            "buy_rate": 0.0005,
            "sell_rate": 0.0009,
            "spread_model": "none",
        }
        buy = dispatch("estimate_trade_cost", {**common, "side": "buy"})
        sell = dispatch("estimate_trade_cost", {**common, "side": "sell"})
        assert buy["total_cost"] == pytest.approx(50.0)
        assert sell["total_cost"] == pytest.approx(90.0)

    def test_maker_rebate_is_reported_as_a_negative_leg(self):
        """The one input allowed below zero. A rebate that silently clamped
        to zero would misprice every passive fill."""
        result = dispatch(
            "estimate_trade_cost",
            {
                "notional": 100_000,
                "commission_model": "maker_taker",
                "maker_rate": -0.0002,
                "is_maker": True,
                "spread_model": "none",
            },
        )
        assert result["legs"][0]["cost"] == pytest.approx(-20.0)
        assert any("rebate" in note for note in result["notes"])

    def test_per_share_minimum_dominates_a_small_trade(self):
        result = dispatch(
            "estimate_trade_cost",
            {
                "notional": 500,
                "commission_model": "per_share",
                "shares": 10,
                "per_share_rate": 0.005,
                "min_commission": 1.0,
                "spread_model": "none",
            },
        )
        assert result["legs"][0]["cost"] == pytest.approx(1.0)
        assert any("minimum" in note for note in result["notes"])

    def test_per_share_without_shares_is_rejected(self):
        with pytest.raises(Exception) as exc:
            dispatch(
                "estimate_trade_cost",
                {"notional": 1000, "commission_model": "per_share"},
            )
        assert "shares" in str(exc.value)

    def test_impact_needs_both_inputs_or_neither(self):
        with pytest.raises(Exception) as exc:
            dispatch(
                "estimate_trade_cost",
                {"notional": 100_000, "avg_dollar_volume": 1e6},
            )
        assert "volatility" in str(exc.value)

    def test_impact_leg_matches_the_library_function(self):
        result = dispatch(
            "estimate_trade_cost",
            {
                "notional": 250_000,
                "commission_model": "none",
                "spread_model": "none",
                "avg_dollar_volume": 5e6,
                "volatility": 0.02,
            },
        )
        assert result["legs"][0]["cost"] == pytest.approx(
            costs.impact_cost(250_000, 5e6, 0.02, 1.0)
        )

    def test_large_participation_is_flagged_as_extrapolation(self):
        result = dispatch(
            "estimate_trade_cost",
            {
                "notional": 900_000,
                "commission_model": "none",
                "spread_model": "none",
                "avg_dollar_volume": 1e6,
                "volatility": 0.02,
            },
        )
        assert any("average dollar volume" in note for note in result["notes"])

    def test_pct_of_range_spread_needs_the_bar(self):
        with pytest.raises(Exception) as exc:
            dispatch(
                "estimate_trade_cost",
                {"notional": 100_000, "spread_model": "pct_of_range"},
            )
        assert "bar_high" in str(exc.value)

    def test_margin_interest_only_accrues_on_negative_cash(self):
        positive = dispatch(
            "estimate_trade_cost",
            {
                "notional": 100_000,
                "commission_model": "none",
                "spread_model": "none",
                "margin_cash": 5_000,
                "margin_annual_rate": 0.08,
            },
        )
        negative = dispatch(
            "estimate_trade_cost",
            {
                "notional": 100_000,
                "commission_model": "none",
                "spread_model": "none",
                "margin_cash": -5_000,
                "margin_annual_rate": 0.08,
                "holding_days": 30,
            },
        )
        assert positive["legs"] == []
        assert negative["legs"][0]["component"] == "margin_interest"

    def test_breakeven_move_is_the_round_trip(self):
        result = dispatch(
            "estimate_trade_cost",
            {"notional": 100_000, "commission_pct": 0.001, "spread_bps": 5.0},
        )
        assert result["breakeven_move_bps"] == pytest.approx(result["total_bps"] * 2)


class TestCompareCostModels:
    def test_costs_are_monotone_in_the_commission_rate(self, stub_provider):
        """The property the breakeven bisection depends on. Signals come
        from prices, never from equity, so a higher rate cannot change
        which dates trade -- only what they cost."""
        result = dispatch(
            "compare_cost_models",
            {
                "symbol": "TEST",
                "start_date": "2021-01-01",
                "end_date": "2022-08-01",
                "strategy_type": "sma_crossover",
                "parameters": {"fast_period": 5, "slow_period": 20},
                "scenarios": [
                    {"label": "free", "commission_pct": 0.0},
                    {"label": "cheap", "commission_pct": 0.001},
                    {"label": "dear", "commission_pct": 0.01},
                ],
                "solve_breakeven": False,
            },
        )
        returns = [row["total_return"] for row in result["scenarios"]]
        assert returns == sorted(returns, reverse=True)

    def test_every_scenario_trades_the_same_dates(self, stub_provider):
        result = dispatch(
            "compare_cost_models",
            {
                "symbol": "TEST",
                "start_date": "2021-01-01",
                "end_date": "2022-08-01",
                "strategy_type": "sma_crossover",
                "parameters": {"fast_period": 5, "slow_period": 20},
                "scenarios": [
                    {"label": "a", "commission_pct": 0.0},
                    {"label": "b", "commission_pct": 0.02},
                ],
                "solve_breakeven": False,
            },
        )
        trade_counts = {row["n_trades"] for row in result["scenarios"]}
        assert len(trade_counts) == 1, "costs changed which dates traded"

    def test_gross_is_the_ceiling_and_drag_is_never_positive(self, stub_provider):
        result = dispatch(
            "compare_cost_models",
            {
                "symbol": "TEST",
                "start_date": "2021-01-01",
                "end_date": "2022-08-01",
                "strategy_type": "sma_crossover",
                "parameters": {"fast_period": 5, "slow_period": 20},
                "scenarios": [{"label": "retail", "commission_pct": 0.001}],
                "solve_breakeven": False,
            },
        )
        row = result["scenarios"][0]
        assert row["total_return"] <= result["gross_total_return"]
        assert row["cost_drag_vs_gross"] <= 0

    def test_breakeven_is_where_the_return_actually_crosses(self, stub_provider):
        """Not just that a number came back: at the solved rate the return
        must be ~0, below it positive, above it negative."""
        base = {
            "symbol": "TEST",
            "start_date": "2021-01-01",
            "end_date": "2022-08-01",
            "strategy_type": "sma_crossover",
            "parameters": {"fast_period": 5, "slow_period": 20},
        }
        solved = dispatch(
            "compare_cost_models",
            {
                **base,
                "scenarios": [{"label": "probe", "commission_pct": 0.001}],
                "solve_breakeven": True,
            },
        )
        rate = solved["breakeven_commission_pct"]
        if rate is None:
            pytest.skip("this fixture's strategy has no crossing to find")

        probe = dispatch(
            "compare_cost_models",
            {
                **base,
                "scenarios": [
                    {
                        "label": "below",
                        "commission_pct": rate * 0.5,
                        "slippage_pct": 0.0,
                    },
                    {
                        "label": "above",
                        "commission_pct": rate * 1.5,
                        "slippage_pct": 0.0,
                    },
                ],
                "solve_breakeven": False,
            },
        )
        below, above = probe["scenarios"]
        assert below["total_return"] > 0 >= above["total_return"]

    def test_no_breakeven_when_the_strategy_loses_before_costs(
        self, monkeypatch, trending_prices
    ):
        """A losing strategy has no edge for costs to consume, and
        reporting a rate anyway would imply one existed."""
        falling = trending_prices.copy()
        falling[["Open", "High", "Low", "Close"]] = (
            falling[["Open", "High", "Low", "Close"]].iloc[::-1].to_numpy()
        )

        class _Stub:
            def get_ohlcv(self, *a, **k):
                return falling

        monkeypatch.setattr(
            "standard_quant_tools.agent.runtimes.backtest.tools.DataFactory.get_provider",
            staticmethod(lambda *a, **k: _Stub()),
        )
        result = dispatch(
            "compare_cost_models",
            {
                "symbol": "TEST",
                "start_date": "2021-01-01",
                "end_date": "2022-08-01",
                "strategy_type": "sma_crossover",
                "parameters": {"fast_period": 5, "slow_period": 20},
                "scenarios": [{"label": "retail", "commission_pct": 0.001}],
            },
        )
        if result["gross_total_return"] <= 0:
            assert result["breakeven_commission_pct"] is None
            assert any("before any costs" in note for note in result["notes"])

    def test_duplicate_scenario_labels_are_rejected(self):
        with pytest.raises(Exception) as exc:
            dispatch(
                "compare_cost_models",
                {
                    "symbol": "TEST",
                    "start_date": "2021-01-01",
                    "end_date": "2022-08-01",
                    "strategy_type": "sma_crossover",
                    "scenarios": [
                        {"label": "same", "commission_pct": 0.001},
                        {"label": "same", "commission_pct": 0.002},
                    ],
                },
            )
        assert "unique" in str(exc.value)

    def test_unknown_strategy_names_the_available_ones(self, stub_provider):
        # Caught EARLIER than it used to be. `strategy_type` was a bare
        # `str`, so the schema accepted anything and the runtime raised the
        # library's ValidationError naming the registry. It is a Literal
        # now, so the schema itself rejects it -- which is the point, since
        # the advertised JSON schema then carries an enum and a client doing
        # constrained decoding cannot emit a bad value at all.
        #
        # `dispatch`'s docstring already documents pydantic.ValidationError
        # as what a schema mismatch raises. The assertion that matters is
        # unchanged: the message still names the available strategies.
        with pytest.raises(PydanticValidationError) as exc:
            dispatch(
                "compare_cost_models",
                {
                    "symbol": "TEST",
                    "start_date": "2021-01-01",
                    "end_date": "2022-08-01",
                    "strategy_type": "not_a_strategy",
                    "scenarios": [{"label": "x", "commission_pct": 0.001}],
                },
            )
        assert "sma_crossover" in str(exc.value)


class TestSellSideCommission:
    def test_the_simulation_input_now_carries_a_sell_rate(self):
        """The portfolio engine has supported an asymmetric rate since it
        was added; no tool passed one, so an agent could not reach it."""
        from standard_quant_tools.agent.models import PortfolioSimulationInput

        assert "sell_commission_pct" in PortfolioSimulationInput.model_fields
        field = PortfolioSimulationInput.model_fields["sell_commission_pct"]
        assert field.default is None, "defaulting to a value would change costs"
