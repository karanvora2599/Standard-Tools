"""
Continuous futures and the futures account, tested against identities.

The two properties worth pinning hardest are the ones the modules exist to
protect:

  * back-adjustment leaves the MOST RECENT segment alone, so today's value
    still matches a screen, and only history moves;
  * a futures account's equity is cash plus posted margin, and the contracts
    contribute nothing -- because their profit has already been credited as
    variation margin, and counting both would double it.

Everything else here is arithmetic that must survive a rewrite: ratio
adjustment preserves returns across a roll, difference adjustment preserves
point changes, and doubling the multiplier doubles the P&L.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.futures_engine import run_futures_simulation
from standard_quant_tools.data.continuous import build_continuous_futures
from standard_quant_tools.error import ValidationError


def _chain(gap: float = 35.0, seed: int = 5):
    """Two contracts, the back one taking over on volume partway through."""
    front_dates = pd.date_range("2026-01-05", periods=60, freq="B")
    back_dates = pd.date_range("2026-01-05", periods=90, freq="B")
    rng = np.random.default_rng(seed)
    base = 6000 * np.exp(np.cumsum(rng.normal(0.0004, 0.008, 90)))
    return [
        {
            "symbol": "ESH6",
            "expiry": "2026-03-20",
            "prices": {
                str(d.date()): float(base[i]) for i, d in enumerate(front_dates)
            },
            "volume": {
                str(d.date()): float(200_000 - 3_000 * i)
                for i, d in enumerate(front_dates)
            },
        },
        {
            "symbol": "ESM6",
            "expiry": "2026-06-19",
            "prices": {
                str(d.date()): float(base[i] + gap) for i, d in enumerate(back_dates)
            },
            "volume": {
                str(d.date()): float(20_000 + 3_000 * i)
                for i, d in enumerate(back_dates)
            },
        },
    ]


class TestContinuousFutures:
    def test_the_roll_happens_once_and_moves_forward_only(self):
        out = build_continuous_futures(_chain(), roll_rule="volume")
        assert out["n_rolls"] == 1
        assert out["contracts_used"] == ["ESH6", "ESM6"]

    def test_adjustment_never_touches_the_most_recent_segment(self):
        """Today's value must still be a real price on every adjustment."""
        last = {}
        for adjustment in ("none", "difference", "ratio"):
            out = build_continuous_futures(_chain(), adjustment=adjustment)
            series = pd.Series(out["research_series"])
            last[adjustment] = series.iloc[-1]
            tradeable = out["tradeable_contract_map"]
            newest = tradeable[max(tradeable)]
            assert series.iloc[-1] == pytest.approx(newest["price"])
        assert last["none"] == pytest.approx(last["ratio"])
        assert last["none"] == pytest.approx(last["difference"])

    def test_only_history_moves_and_the_three_disagree_there(self):
        first = {
            adjustment: pd.Series(
                build_continuous_futures(_chain(), adjustment=adjustment)[
                    "research_series"
                ]
            ).iloc[0]
            for adjustment in ("none", "difference", "ratio")
        }
        # An unadjusted series starts at the front contract's real first
        # print; both adjustments lift it by the roll gap.
        assert first["none"] < first["difference"]
        assert first["none"] < first["ratio"]
        assert first["difference"] != pytest.approx(first["ratio"])

    def test_ratio_adjustment_makes_the_roll_return_continuous(self):
        out = build_continuous_futures(_chain(), adjustment="ratio")
        series = pd.Series(out["research_series"])
        series.index = pd.to_datetime(series.index)
        series = series.sort_index()
        roll = pd.Timestamp(out["roll_dates"][0])
        position = list(series.index).index(roll)
        # The return ACROSS the roll must be a market move, not a contract
        # change: on this chain both contracts move together, so it is small.
        crossing = series.iloc[position] / series.iloc[position - 1] - 1.0
        assert abs(crossing) < 0.05

    def test_the_tradeable_map_is_never_adjusted(self):
        out = build_continuous_futures(_chain(), adjustment="difference")
        chain = {c["symbol"]: c["prices"] for c in _chain()}
        for date, row in out["tradeable_contract_map"].items():
            key = str(pd.Timestamp(date).date())
            assert row["price"] == pytest.approx(chain[row["symbol"]][key])

    def test_a_rule_cannot_run_on_data_that_was_not_supplied(self):
        chain = _chain()
        for contract in chain:
            contract.pop("volume")
        with pytest.raises(ValidationError, match="open_interest"):
            build_continuous_futures(chain, roll_rule="open_interest")

    def test_one_contract_is_already_continuous(self):
        with pytest.raises(ValidationError, match="at least two"):
            build_continuous_futures(_chain()[:1])


class TestFuturesAccount:
    def _prices(self, n=40, start=6000.0, step=5.0):
        dates = pd.date_range("2026-01-05", periods=n, freq="B")
        return {str(d.date()): start + step * i for i, d in enumerate(dates)}

    def test_equity_is_always_cash_plus_posted_margin(self):
        prices = self._prices()
        out = run_futures_simulation(
            prices=prices,
            target_contracts={k: 10.0 for k in prices},
            multiplier=50,
            initial_capital=2_000_000,
            initial_margin=15_000,
        )
        assert np.allclose(
            out["equity_curve"].to_numpy(),
            out["cash_curve"].to_numpy() + out["margin_curve"].to_numpy(),
        )

    def test_variation_margin_is_the_whole_of_the_pnl(self):
        """With no costs and no interest, P&L is exactly the price move."""
        prices = self._prices(n=20, step=5.0)
        contracts, multiplier = 10.0, 50.0
        out = run_futures_simulation(
            prices=prices,
            target_contracts={k: contracts for k in prices},
            multiplier=multiplier,
            initial_capital=2_000_000,
        )
        values = list(prices.values())
        expected = (values[-1] - values[0]) * contracts * multiplier
        assert out["total_variation_margin"] == pytest.approx(expected)
        assert out["final_equity"] == pytest.approx(2_000_000 + expected)

    def test_doubling_the_multiplier_doubles_the_pnl(self):
        prices = self._prices()
        common = dict(
            prices=prices,
            target_contracts={k: 5.0 for k in prices},
            initial_capital=5_000_000,
        )
        one = run_futures_simulation(multiplier=50, **common)
        two = run_futures_simulation(multiplier=100, **common)
        assert two["total_variation_margin"] == pytest.approx(
            2.0 * one["total_variation_margin"]
        )

    def test_a_flat_account_only_earns_interest(self):
        prices = self._prices()
        out = run_futures_simulation(
            prices=prices,
            target_contracts={k: 0.0 for k in prices},
            multiplier=50,
            initial_capital=1_000_000,
            collateral_rate=0.05,
        )
        assert out["total_variation_margin"] == pytest.approx(0.0)
        assert out["total_collateral_interest"] > 0
        assert out["final_equity"] == pytest.approx(
            1_000_000 + out["total_collateral_interest"]
        )
        assert out["max_leverage"] == pytest.approx(0.0)

    def test_a_short_makes_money_when_the_price_falls(self):
        prices = self._prices(step=-5.0)
        out = run_futures_simulation(
            prices=prices,
            target_contracts={k: -10.0 for k in prices},
            multiplier=50,
            initial_capital=2_000_000,
        )
        assert out["total_variation_margin"] > 0

    def test_the_roll_charges_both_legs(self):
        prices = self._prices(n=20)
        keys = list(prices)
        contract_map = {k: ("A" if i < 10 else "B") for i, k in enumerate(keys)}
        out = run_futures_simulation(
            prices=prices,
            target_contracts={k: 4.0 for k in prices},
            contract_map=contract_map,
            multiplier=50,
            initial_capital=1_000_000,
            commission_per_contract=2.5,
        )
        assert out["n_rolls"] == 1
        # Four contracts, closed and reopened: eight commissions.
        assert out["rolls"][0]["cost"] == pytest.approx(8 * 2.5)

    def test_an_iso_keyed_contract_map_still_rolls(self):
        """The keys arrive as strings from JSON and the index is Timestamps."""
        prices = self._prices(n=20)
        keys = list(prices)
        out = run_futures_simulation(
            prices=prices,
            target_contracts={k: 4.0 for k in prices},
            contract_map={k: ("A" if i < 10 else "B") for i, k in enumerate(keys)},
            multiplier=50,
            initial_capital=1_000_000,
        )
        assert out["n_rolls"] == 1

    def test_a_margin_call_reduces_rather_than_being_financed(self):
        # A large position into a falling market on thin capital.
        prices = self._prices(n=30, start=6000.0, step=-40.0)
        out = run_futures_simulation(
            prices=prices,
            target_contracts={k: 20.0 for k in prices},
            multiplier=50,
            initial_capital=400_000,
            initial_margin=15_000,
            maintenance_margin=12_000,
        )
        assert out["n_margin_calls"] >= 1
        assert any("margin call" in w for w in out["warnings"])
        # The position must actually have come down.
        assert abs(out["position_curve"].iloc[-1]) < 20.0

    def test_maintenance_above_initial_is_refused(self):
        prices = self._prices(n=5)
        with pytest.raises(ValidationError, match="maintenance_margin"):
            run_futures_simulation(
                prices=prices,
                target_contracts={k: 1.0 for k in prices},
                multiplier=50,
                initial_margin=1_000,
                maintenance_margin=2_000,
            )

    def test_an_unmargined_account_says_so(self):
        prices = self._prices(n=10)
        out = run_futures_simulation(
            prices=prices,
            target_contracts={k: 1.0 for k in prices},
            multiplier=50,
            initial_margin=0.0,
        )
        assert any("initial_margin is zero" in w for w in out["warnings"])

    def test_omitting_the_contract_map_says_no_roll_was_modelled(self):
        prices = self._prices(n=10)
        out = run_futures_simulation(
            prices=prices,
            target_contracts={k: 1.0 for k in prices},
            multiplier=50,
        )
        assert out["n_rolls"] == 0
        assert any("NO ROLL" in w for w in out["warnings"])
