"""
The hedge simulation and the scenario grid price through the chain batch
and return what their per-contract loops returned.

`simulate_delta_hedge` asked `option_greeks` for one delta per path per
rebalance -- 10,500 calls for the default run -- and `option_risk_scenarios`
asked `price_option` for one price per cell. Both now make one batched call
per rebalance or per grid (see the CHANGELOG entry of 2026-09-28). The
references below are those loops, written out, and every number is required
to be the same double: a faster hedge simulation that drifted by a cent
would be a different simulation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from standard_quant_tools.analysis import derivatives
from standard_quant_tools.analysis import options_batch as ob
from standard_quant_tools.analysis.derivatives import (
    option_greeks,
    option_risk_scenarios,
    simulate_delta_hedge,
)
from standard_quant_tools.analysis.pricing import price_option
from standard_quant_tools.error import ValidationError

NATIVE = ob._native("black_scholes_greeks_batch") is not None


@pytest.fixture(
    params=[
        pytest.param(
            "native", marks=pytest.mark.skipif(not NATIVE, reason="extension not built")
        ),
        "python",
    ]
)
def backend(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setattr(ob, "HAS_CPP", False)
    return request.param


def _hedge_by_path(
    *,
    spot,
    strike,
    time_to_expiry,
    implied_vol,
    realized_vol,
    risk_free_rate=0.0,
    option_type="call",
    n_hedges=21,
    n_paths=500,
    transaction_cost_bps=0.0,
    seed=0,
):
    """The per-path, per-rebalance loop, one option_greeks call per step."""
    t, iv, rv = time_to_expiry, implied_vol, realized_vol
    dt = t / n_hedges
    rng = np.random.default_rng(int(seed))
    cost_rate = float(transaction_cost_bps) / 1e4
    premium = float(
        price_option(
            spot=spot,
            strike=strike,
            time_to_expiry=t,
            volatility=iv,
            risk_free_rate=risk_free_rate,
            option_type=option_type,
        )["price"]
    )
    pnl = np.zeros(n_paths)
    costs = np.zeros(n_paths)
    for p in range(n_paths):
        s, cash, shares, path_cost = spot, premium, 0.0, 0.0
        for step in range(n_hedges):
            remaining = t - step * dt
            target = option_greeks(
                spot=s,
                strike=strike,
                time_to_expiry=max(remaining, 1e-8),
                volatility=iv,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
            )["delta"]
            trade = target - shares
            traded_value = abs(trade) * s
            path_cost += traded_value * cost_rate
            cash -= trade * s + traded_value * cost_rate
            shares = target
            z = rng.standard_normal()
            s *= math.exp(
                (risk_free_rate - 0.5 * rv * rv) * dt + rv * math.sqrt(dt) * z
            )
            cash *= math.exp(risk_free_rate * dt)
        intrinsic = (
            max(s - strike, 0.0) if option_type == "call" else max(strike - s, 0.0)
        )
        pnl[p] = cash + shares * s - intrinsic
        costs[p] = path_cost
    return pnl, costs


HEDGES = [
    dict(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.25,
        implied_vol=0.30,
        realized_vol=0.25,
        n_hedges=63,
        n_paths=120,
        seed=11,
    ),
    dict(
        spot=100.0,
        strike=90.0,
        time_to_expiry=1.0,
        implied_vol=0.2,
        realized_vol=0.35,
        n_hedges=21,
        n_paths=200,
        seed=1,
        option_type="put",
        transaction_cost_bps=10.0,
        risk_free_rate=0.04,
    ),
    dict(
        spot=50.0,
        strike=55.0,
        time_to_expiry=0.1,
        implied_vol=0.5,
        realized_vol=0.5,
        n_hedges=5,
        n_paths=7,
        seed=3,
        risk_free_rate=-0.01,
    ),
]


class TestTheHedgeSimulationIsThePerPathLoop:
    @pytest.mark.parametrize("kw", HEDGES)
    def test_every_statistic_is_the_same_double(self, backend, kw):
        pnl, costs = _hedge_by_path(**kw)
        result = simulate_delta_hedge(**kw)
        assert result["mean_pnl"] == float(pnl.mean())
        assert result["median_pnl"] == float(np.median(pnl))
        assert result["std_pnl"] == float(pnl.std(ddof=1))
        assert result["p05_pnl"] == float(np.percentile(pnl, 5))
        assert result["p95_pnl"] == float(np.percentile(pnl, 95))
        assert result["worst_pnl"] == float(pnl.min())
        assert result["best_pnl"] == float(pnl.max())
        assert result["win_rate"] == float((pnl > 0).mean())
        assert result["mean_transaction_cost"] == float(costs.mean())

    def test_one_batched_greek_call_per_rebalance(self, monkeypatch):
        """The point of the change: n_hedges calls, not n_paths x n_hedges."""
        calls = []
        real = derivatives.black_scholes_greeks_batch

        def counting(*args, **kwargs):
            calls.append(np.size(args[0]))
            return real(*args, **kwargs)

        monkeypatch.setattr(derivatives, "black_scholes_greeks_batch", counting)
        simulate_delta_hedge(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            implied_vol=0.3,
            realized_vol=0.3,
            n_hedges=21,
            n_paths=40,
            seed=0,
        )
        assert calls == [40] * 21

    def test_a_spot_the_hedge_cannot_price_is_refused_by_option_greeks(self):
        """A realized volatility of 9,000% drives some path's spot below
        1e-8 within a few steps. The per-path loop refused it through
        option_greeks, and so does the batch -- with option_greeks' words,
        not a message of its own."""
        with pytest.raises(ValidationError, match="spot magnitude"):
            simulate_delta_hedge(
                spot=100.0,
                strike=100.0,
                time_to_expiry=1.0,
                implied_vol=0.3,
                realized_vol=90.0,
                n_hedges=50,
                n_paths=50,
                seed=0,
            )

    def test_an_ordinary_run_is_not_refused(self):
        """Null case for the refusal above: an ordinary volatility never
        reaches the bound."""
        result = simulate_delta_hedge(
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            implied_vol=0.3,
            realized_vol=0.3,
            n_hedges=50,
            n_paths=50,
            seed=0,
        )
        assert math.isfinite(result["mean_pnl"])


def _scenarios_by_cell(
    *,
    spot,
    strike,
    time_to_expiry,
    volatility,
    risk_free_rate=0.0,
    dividend_yield=0.0,
    option_type="call",
    quantity=1.0,
    spot_shocks=(-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20),
    vol_shocks=(-0.10, -0.05, 0.0, 0.05, 0.10),
    days_forward=0.0,
):
    """The cell-by-cell loop, one price_option call per cell."""
    quantity = float(quantity)
    remaining = time_to_expiry - float(days_forward) / 365.0
    base_value = (
        quantity
        * price_option(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            option_type=option_type,
            dividend_yield=dividend_yield,
        )["price"]
    )
    cells = []
    for ds in spot_shocks:
        shocked_spot = spot * (1.0 + float(ds))
        if shocked_spot <= 0:
            continue
        for dv in vol_shocks:
            shocked_vol = volatility + float(dv)
            if shocked_vol <= 0:
                continue
            value = (
                quantity
                * price_option(
                    spot=shocked_spot,
                    strike=strike,
                    time_to_expiry=remaining,
                    volatility=shocked_vol,
                    risk_free_rate=risk_free_rate,
                    option_type=option_type,
                    dividend_yield=dividend_yield,
                )["price"]
            )
            cells.append((float(ds) * 100.0, float(dv), value, value - base_value))
    return base_value, cells


SCENARIOS = [
    dict(spot=100.0, strike=100.0, time_to_expiry=0.5, volatility=0.25),
    dict(
        spot=100.0,
        strike=110.0,
        time_to_expiry=0.2,
        volatility=0.4,
        option_type="put",
        dividend_yield=0.03,
        risk_free_rate=0.05,
        quantity=-3,
        days_forward=5,
    ),
    dict(
        spot=100.0,
        strike=100.0,
        time_to_expiry=0.5,
        volatility=0.05,
        vol_shocks=(-0.1, -0.05, 0.0, 0.1),
        spot_shocks=(-1.5, -0.2, 0.0, 0.3),
    ),
]


class TestTheScenarioGridIsTheCellByCellLoop:
    @pytest.mark.parametrize("kw", SCENARIOS)
    def test_every_cell_is_the_same_double(self, backend, kw):
        base_value, cells = _scenarios_by_cell(**kw)
        result = option_risk_scenarios(**kw)
        assert result["base_value"] == float(base_value)
        got = [
            (row["spot_shock_pct"], cell["vol_shock"], cell["value"], cell["pnl"])
            for row in result["grid"]
            for cell in row["cells"]
        ]
        assert got == [(a, b, float(c), float(d)) for a, b, c, d in cells]

    def test_the_grid_is_one_batched_call(self, monkeypatch):
        calls = []
        real = derivatives.black_scholes_greeks_batch

        def counting(*args, **kwargs):
            calls.append(kwargs.get("grid"))
            return real(*args, **kwargs)

        monkeypatch.setattr(derivatives, "black_scholes_greeks_batch", counting)
        option_risk_scenarios(
            spot=100.0, strike=100.0, time_to_expiry=0.5, volatility=0.25
        )
        assert calls == [True]

    def test_a_nan_shock_is_still_refused_by_the_pricer(self, backend):
        """The loop skipped a shock only when `shocked <= 0`, so a NaN shock
        reached price_option and was refused there. It still is, in the same
        words, rather than being dropped from the grid."""
        with pytest.raises(ValidationError, match="spot=nan has a magnitude"):
            option_risk_scenarios(
                spot=100.0,
                strike=100.0,
                time_to_expiry=0.5,
                volatility=0.25,
                spot_shocks=(0.0, float("nan")),
            )

    def test_a_cell_past_the_volatility_bound_is_refused_as_before(self, backend):
        with pytest.raises(ValidationError, match="volatility=100.05"):
            option_risk_scenarios(
                spot=100.0,
                strike=100.0,
                time_to_expiry=0.5,
                volatility=99.95,
                vol_shocks=(0.0, 0.1),
            )

    def test_a_grid_with_no_live_column_prices_nothing(self, backend):
        """Null case: every volatility shock takes the vol to zero or below,
        so every row is empty -- as the loop left it -- and nothing is
        priced or refused."""
        result = option_risk_scenarios(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.5,
            volatility=0.05,
            vol_shocks=(-0.1, -0.05),
        )
        assert [row["cells"] for row in result["grid"]] == [[]] * 7
        assert result["worst_case"] is None
