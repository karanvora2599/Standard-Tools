"""
A strategy's capacity under square-root impact, held to closed forms.

One instrument trading |dw| = tau every period breaks even at A* = (ADV /
tau) ((alpha / tau - s/2) / (k sigma))^2, alpha the mean gross return per
period; the curve's net return there is zero, and its net Sharpe at the
reported half-Sharpe AUM is half the gross. One trade's cost is the
library's own `fixed_bps_spread` at half the spread plus `impact_cost`, the
cost model the backtests charge. An instrument's share of the impact is the
same at every AUM, so the one that binds is the one with the largest
share; a spread larger than the strategy's return breaks even at zero.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.allocation.capacity import capacity_curve
from standard_quant_tools.backtest.costs import fixed_bps_spread, impact_cost
from standard_quant_tools.error import ValidationError

P = 252


def _dates(n, start="2020-01-02"):
    return pd.bdate_range(start, periods=n)


def _gross(n, mean, sd, seed):
    return pd.Series(
        np.random.default_rng(seed).normal(mean, sd, n), index=_dates(n), name="g"
    )


def _alternating(n, tau, name="X"):
    """tau, 0, tau, 0, ... from cash: |dw| = tau every period, the first
    included."""
    return pd.DataFrame(
        {name: [tau if i % 2 == 0 else 0.0 for i in range(n)]}, index=_dates(n)
    )


def _dated(series):
    return {d.strftime("%Y-%m-%d"): float(v) for d, v in series.items()}


def _call(arguments):
    return resolve("allocation").dispatch("get_strategy_capacity_curve", arguments)


class TestTheSingleNameClosedForm:
    ADV, SIGMA, SPREAD_BPS, TAU, K = 5e7, 0.02, 2.0, 0.2, 1.0

    def _run(self, gross, **extra):
        return capacity_curve(
            gross,
            _alternating(len(gross), self.TAU),
            adv={"X": self.ADV},
            daily_volatility={"X": self.SIGMA},
            spread_bps={"X": self.SPREAD_BPS},
            periods_per_year=P,
            impact_coefficient=self.K,
            **extra,
        )

    def test_break_even_is_adv_over_tau_times_the_squared_ratio(self):
        gross = _gross(500, 0.001, 0.01, seed=1)
        out = self._run(gross)
        alpha, s = float(gross.mean()), self.SPREAD_BPS / 1e4
        closed = (self.ADV / self.TAU) * (
            (alpha / self.TAU - s / 2) / (self.K * self.SIGMA)
        ) ** 2
        assert out["breakeven_aum"] == pytest.approx(closed, rel=1e-12)
        at = self._run(gross, aum=[closed])["curve"][0]
        assert at["net_return"] == pytest.approx(0.0, abs=1e-15)

    def test_the_net_sharpe_at_the_half_sharpe_aum_is_half_the_gross(self):
        gross = _gross(750, 0.0012, 0.01, seed=2)
        out = self._run(gross)
        half = out["half_sharpe_aum"]
        assert 0 < half < out["breakeven_aum"]
        at = self._run(gross, aum=[half])["curve"][0]
        assert at["net_sharpe"] == pytest.approx(out["gross_sharpe"] / 2, rel=1e-9)
        # And it is the first such AUM: below it the Sharpe is higher.
        below = self._run(gross, aum=list(np.linspace(half / 50, half * 0.98, 20)))
        assert all(p["net_sharpe"] > out["gross_sharpe"] / 2 for p in below["curve"])

    def test_the_curve_falls_with_aum(self):
        out = self._run(_gross(500, 0.001, 0.01, seed=3))
        nets = [p["net_return"] for p in out["curve"]]
        assert all(a > b for a, b in zip(nets, nets[1:]))
        assert len(out["curve"]) == 25
        assert out["curve"][0]["aum"] == pytest.approx(out["breakeven_aum"] / 1000)


class TestTheCostModel:
    def test_one_trade_costs_the_spread_and_impact_the_backtests_charge(self):
        """A single build of 30% of AUM: half the spread by fixed_bps_spread
        plus impact_cost at that notional, both divided by AUM."""
        n, aum = 40, 2.5e7
        gross = _gross(n, 0.0, 0.01, seed=4)
        weights = pd.DataFrame({"X": [0.3] * n}, index=_dates(n))
        out = capacity_curve(
            gross,
            weights,
            adv={"X": 4e7},
            daily_volatility={"X": 0.025},
            spread_bps={"X": 6.0},
            periods_per_year=P,
            aum=[aum],
        )
        notional = 0.3 * aum
        one_trade = (
            fixed_bps_spread(notional, 3.0) + impact_cost(notional, 4e7, 0.025, 1.0)
        ) / aum
        assert out["curve"][0]["cost_per_year"] == pytest.approx(
            one_trade / n * P, rel=1e-12
        )
        assert out["curve"][0]["max_participation"] == pytest.approx(notional / 4e7)


class TestWhatBinds:
    def test_the_illiquid_name_binds_and_its_share_is_aum_free(self):
        n = 400
        gross = _gross(n, 0.0015, 0.01, seed=5)
        weights = pd.concat(
            [_alternating(n, 0.1, "LIQUID"), _alternating(n, 0.1, "THIN")], axis=1
        )
        args = dict(
            adv={"LIQUID": 1e9, "THIN": 1e7},
            daily_volatility={"LIQUID": 0.02, "THIN": 0.02},
            spread_bps={"LIQUID": 1.0, "THIN": 1.0},
            periods_per_year=P,
        )
        small = capacity_curve(gross, weights, aum=[1e5], **args)
        large = capacity_curve(gross, weights, aum=[1e9], **args)
        assert small["binding"][0]["instrument"] == "THIN"
        # sqrt(ADV ratio) = 10: THIN carries 10/11 of the impact, at any AUM.
        assert small["binding"][0]["impact_share"] == pytest.approx(10 / 11, rel=1e-12)
        assert (
            large["binding"][0]["impact_share"] == small["binding"][0]["impact_share"]
        )
        assert sum(b["impact_share"] for b in small["binding"]) == pytest.approx(1.0)

    def test_participation_past_the_law_is_flagged(self):
        n = 300
        out = capacity_curve(
            _gross(n, 0.01, 0.01, seed=6),
            _alternating(n, 0.5),
            adv={"X": 1e6},
            daily_volatility={"X": 0.01},
            spread_bps={"X": 1.0},
            periods_per_year=P,
        )
        assert out["binding"][0]["max_participation"] > 0.2
        assert any("square-root law" in w for w in out["warnings"])


class TestTheEdges:
    def test_a_spread_larger_than_the_return_breaks_even_at_zero(self):
        n = 300
        out = capacity_curve(
            _gross(n, 0.0001, 0.01, seed=7),
            _alternating(n, 0.5),
            adv={"X": 1e8},
            daily_volatility={"X": 0.02},
            spread_bps={"X": 10.0},
            periods_per_year=P,
        )
        assert out["breakeven_aum"] == 0.0
        assert any("spread alone" in w for w in out["warnings"])

    def test_a_panel_that_never_trades_has_no_break_even(self):
        n = 100
        out = _call(
            {
                "returns": _dated(_gross(n, 0.001, 0.01, seed=8)),
                "weights": {"X": {d.strftime("%Y-%m-%d"): 0.0 for d in _dates(n)}},
                "adv": {"X": 1e8},
                "daily_volatility": {"X": 0.02},
                "spread_bps": {"X": 1.0},
            }
        )
        assert out["breakeven_aum"] is None and out["half_sharpe_aum"] is None
        assert any("breakeven_aum is null" in w for w in out["warnings"])

    def test_rebalance_dates_are_charged_on_the_next_return_date(self):
        """Weekly weights dated Sundays trade on the Monday after; the row
        after the last return is dropped with a note."""
        n = 60
        gross = _gross(n, 0.001, 0.01, seed=9)
        sundays = pd.date_range("2020-01-05", periods=13, freq="W-SUN")
        weights = pd.DataFrame(
            {"X": [0.2 if i % 2 == 0 else 0.1 for i in range(13)]}, index=sundays
        )
        out = capacity_curve(
            gross,
            weights,
            adv={"X": 1e8},
            daily_volatility={"X": 0.02},
            spread_bps={"X": 0.0},
            periods_per_year=P,
            aum=[1e6],
        )
        assert out["n_weight_rows"] == 12
        assert any("after the last return" in w for w in out["warnings"])
        # 12 trades: one build of 0.2, then eleven of 0.1.
        impact = (0.2**1.5 + 11 * 0.1**1.5) * 0.02 / math.sqrt(1e8) * math.sqrt(1e6)
        assert out["curve"][0]["cost_per_year"] == pytest.approx(
            impact / n * P, rel=1e-12
        )


class TestTheTool:
    def _args(self, n=300):
        gross = _gross(n, 0.0012, 0.01, seed=10)
        rng = np.random.default_rng(11)
        weights = {
            name: {
                d.strftime("%Y-%m-%d"): float(w)
                for d, w in zip(_dates(n), rng.uniform(0.0, 0.5, n))
            }
            for name in ("A", "B")
        }
        return {
            "returns": _dated(gross),
            "weights": weights,
            "adv": {"A": 3e7, "B": 8e7},
            "daily_volatility": {"A": 0.02, "B": 0.015},
            "spread_bps": {"A": 3.0, "B": 2.0},
        }

    def test_a_weight_panel_reference_reads_as_the_inline_weights(self):
        args = self._args()
        ref = handoff.publish(
            args["weights"], kind="weight_panel", run_id="capacity_ref", name="w"
        )
        inline = _call(args)
        by_ref = _call(
            {**{k: v for k, v in args.items() if k != "weights"}, "weights_ref": ref}
        )
        assert by_ref["breakeven_aum"] == inline["breakeven_aum"]
        assert by_ref["curve"] == inline["curve"]

    def test_the_maps_must_name_the_weights_instruments(self):
        args = self._args()
        with pytest.raises(ValidationError, match="adv must name exactly"):
            _call({**args, "adv": {"A": 3e7, "C": 1e8}})

    @pytest.mark.parametrize(
        "extra,needle",
        [
            ({"weights_ref": "sqt://weight_panel/x/y"}, "exactly one of weights"),
            ({"aum": [-1.0]}, "aum"),
            ({"adv": {"A": float("nan"), "B": 1.0}}, "adv"),
            ({"impact_coefficient": 0.0}, "impact_coefficient"),
        ],
    )
    def test_the_input_model_refuses_by_name(self, extra, needle):
        with pytest.raises(PydanticValidationError, match=needle):
            _call({**self._args(), **extra})

    def test_a_weight_panel_is_not_a_returns_reference(self):
        args = self._args()
        ref = handoff.publish(
            args["weights"], kind="weight_panel", run_id="capacity_kind", name="w"
        )
        with pytest.raises(ValidationError, match="weight_panel"):
            _call(
                {
                    **{k: v for k, v in args.items() if k != "returns"},
                    "returns_ref": ref,
                }
            )
