"""
The desk instruments, tested against identities rather than saved numbers.

Same strategy as `test_delta_one.py`: what is pinned here is arithmetic that
must hold however the code underneath is written --

  * a benchmark that IS a combination of the candidates must be replicated
    exactly, and constraining the support cannot improve on the
    unconstrained fit;
  * receiving and paying a swap are exact negations of each other;
  * a TRF spread turned into a level and back is the same spread;
  * doubling an index divisor halves the dividend points;
  * index flow scales linearly with the assets tracking the index.

Each survives a rewrite and fails if the meaning changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.delta_one.dividends import dividend_points
from standard_quant_tools.delta_one.etf import etf_fair_value
from standard_quant_tools.delta_one.rebalance import index_rebalance_flow
from standard_quant_tools.delta_one.replication import optimize_replication_basket
from standard_quant_tools.delta_one.swaps import (
    price_total_return_swap,
    total_return_future,
)
from standard_quant_tools.error import ValidationError


class TestReplication:
    def _universe(self, n=25, n_obs=400, seed=17):
        rng = np.random.default_rng(seed)
        factor = rng.normal(0.0004, 0.010, n_obs)
        loads = rng.uniform(0.6, 1.5, n)
        frame = pd.DataFrame(
            {
                f"S{i:02d}": loads[i] * factor + rng.normal(0, 0.010, n_obs)
                for i in range(n)
            }
        )
        # The benchmark IS an equal-weight combination of the candidates, so
        # a perfect replication exists and the optimizer must find it.
        bench = pd.Series(frame.to_numpy() @ np.full(n, 1.0 / n))
        return frame, bench

    def test_a_spanned_benchmark_is_tracked_exactly(self):
        frame, bench = self._universe()
        out = optimize_replication_basket(
            returns=frame, benchmark_returns=bench, covariance_method="sample"
        )
        assert out["realized_tracking_error"] == pytest.approx(0.0, abs=1e-5)
        assert out["correlation"] == pytest.approx(1.0, abs=1e-5)
        assert out["beta"] == pytest.approx(1.0, abs=1e-2)

    def test_the_basket_is_always_fully_invested(self):
        frame, bench = self._universe()
        for max_names in (None, 5, 12):
            out = optimize_replication_basket(
                returns=frame, benchmark_returns=bench, max_names=max_names
            )
            assert out["net_weight"] == pytest.approx(1.0, abs=1e-6), max_names

    def test_max_names_binds_and_costs_tracking_error(self):
        frame, bench = self._universe()
        full = optimize_replication_basket(returns=frame, benchmark_returns=bench)
        small = optimize_replication_basket(
            returns=frame, benchmark_returns=bench, max_names=5
        )
        assert small["n_selected"] <= 5
        # Constraining the support cannot improve the objective it was
        # solved without.
        assert small["realized_tracking_error"] >= full["realized_tracking_error"]
        assert any("thresholding" in w for w in small["warnings"])

    def test_long_only_holds_no_shorts(self):
        frame, bench = self._universe()
        out = optimize_replication_basket(
            returns=frame, benchmark_returns=bench, long_only=True
        )
        assert min(out["weights"].values()) >= -1e-9

    def test_a_cap_on_a_name_that_is_not_a_candidate_is_refused(self):
        frame, bench = self._universe()
        with pytest.raises(ValidationError, match="not"):
            optimize_replication_basket(
                returns=frame, benchmark_returns=bench, weight_caps={"NOPE": 0.1}
            )


class TestEtfFairValue:
    def test_price_equal_to_nav_is_fair_and_has_no_arbitrage(self):
        out = etf_fair_value(etf_price=100.0, nav=100.0)
        assert out["premium_discount_bps"] == pytest.approx(0.0)
        assert out["classification"] == "fair"
        assert out["arbitrage_survives"] is False
        assert out["action"] is None

    def test_premium_and_discount_are_mirror_images(self):
        up = etf_fair_value(etf_price=101.0, nav=100.0, tolerance_bps=1.0)
        down = etf_fair_value(etf_price=99.0, nav=100.0, tolerance_bps=1.0)
        assert up["classification"] == "premium"
        assert down["classification"] == "discount"
        assert up["action"] == "create"
        assert down["action"] == "redeem"
        assert up["gross_arbitrage_bps"] == pytest.approx(
            down["gross_arbitrage_bps"], rel=1e-2
        )

    def test_costs_come_straight_off_the_gross(self):
        out = etf_fair_value(
            etf_price=100.40, nav=100.0, etf_spread_bps=1.0, basket_spread_bps=3.0
        )
        # Round trip on both legs: 2 * (1 + 3).
        assert out["execution_bps"] == pytest.approx(8.0)
        assert out["net_arbitrage_bps"] == pytest.approx(
            out["gross_arbitrage_bps"] - out["execution_bps"]
        )

    def test_a_fee_needs_a_unit_size_to_become_basis_points(self):
        out = etf_fair_value(etf_price=101.0, nav=100.0, creation_fee=500.0)
        assert out["creation_fee_bps"] == 0.0
        assert any("creation_unit_shares" in w for w in out["warnings"])


class TestTotalReturnSwap:
    def _base(self, **over):
        kwargs = dict(
            notional=100e6,
            initial_price=100.0,
            current_price=108.0,
            dividends=1.8,
            financing_rate=0.043,
            spread_bps=45.0,
            time_elapsed=0.5,
        )
        kwargs.update(over)
        return price_total_return_swap(**kwargs)

    def test_receiving_and_paying_are_exact_negations(self):
        receive = self._base(direction="receive")
        pay = self._base(direction="pay")
        assert receive["net_pnl"] == pytest.approx(-pay["net_pnl"])
        assert receive["equity_leg"] == pytest.approx(-pay["equity_leg"])
        assert receive["financing_leg"] == pytest.approx(-pay["financing_leg"])

    def test_total_return_is_price_plus_dividend(self):
        out = self._base()
        assert out["total_return"] == pytest.approx(
            out["price_return"] + out["dividend_return"]
        )
        assert out["price_return"] == pytest.approx(0.08)
        assert out["dividend_return"] == pytest.approx(0.018)

    def test_no_elapsed_time_means_no_financing(self):
        out = self._base(time_elapsed=1e-12)
        assert out["financing_accrued"] == pytest.approx(0.0, abs=1e-9)
        assert out["net_pnl"] == pytest.approx(out["equity_leg"], rel=1e-6)

    def test_act_360_accrues_more_than_act_365(self):
        common = dict(
            notional=1e6,
            initial_price=100,
            current_price=100,
            financing_rate=0.05,
            start_date="2026-01-02",
            valuation_date="2026-07-02",
        )
        a = price_total_return_swap(day_count="ACT/360", **common)
        b = price_total_return_swap(day_count="ACT/365F", **common)
        assert a["financing_accrued"] > b["financing_accrued"]
        # 365/360 exactly -- the whole of the difference between them.
        assert a["financing_accrued"] / b["financing_accrued"] == pytest.approx(
            365.0 / 360.0
        )

    def test_the_two_ways_of_giving_a_period_cannot_both_be_used(self):
        with pytest.raises(ValidationError, match="either"):
            price_total_return_swap(
                notional=1e6,
                initial_price=100,
                current_price=100,
                financing_rate=0.05,
                time_elapsed=0.5,
                start_date="2026-01-01",
                valuation_date="2026-07-01",
            )


class TestTotalReturnFuture:
    def test_the_two_conventions_round_trip(self):
        """A spread turned into a level and back must be the same spread."""
        for spread in (25.0, 95.0, -10.0):
            level = total_return_future(
                quote=spread,
                quote_convention="spread_bps",
                underlying_price=5000,
                time_to_expiry=0.5,
                reference_rate=0.043,
            )["implied_level"]
            back = total_return_future(
                quote=level,
                quote_convention="index_level",
                underlying_price=5000,
                time_to_expiry=0.5,
                reference_rate=0.043,
            )["implied_spread_bps"]
            assert back == pytest.approx(spread, abs=1e-8)

    def test_the_comparison_is_a_difference(self):
        out = total_return_future(
            quote=95.0,
            quote_convention="spread_bps",
            underlying_price=5000,
            time_to_expiry=0.5,
            reference_rate=0.043,
            comparison_spread_bps=50.0,
        )
        assert out["difference_bps"] == pytest.approx(45.0)

    def test_an_unnamed_convention_is_refused(self):
        with pytest.raises(ValidationError, match="quote_convention"):
            total_return_future(
                quote=95.0,
                quote_convention="guess",
                underlying_price=5000,
                time_to_expiry=0.5,
                reference_rate=0.043,
            )


class TestDividendPoints:
    CONS = [
        {
            "symbol": "A",
            "shares": 1e9,
            "dividend_per_share": 1.2,
            "ex_date": "2026-04-15",
        },
        {
            "symbol": "B",
            "shares": 5e8,
            "dividend_per_share": 2.4,
            "ex_date": "2026-05-20",
        },
        {
            "symbol": "C",
            "shares": 2e9,
            "dividend_per_share": 0.5,
            "ex_date": "2026-09-01",
        },
    ]

    def test_points_are_shares_times_dividend_over_the_divisor(self):
        out = dividend_points(
            self.CONS, divisor=1e8, as_of="2026-03-01", expiry="2026-06-19"
        )
        # A and B fall in the window; C goes ex after expiry.
        expected = (1e9 * 1.2 + 5e8 * 2.4) / 1e8
        assert out["total_index_points"] == pytest.approx(expected)
        assert out["n_included"] == 2
        assert out["n_excluded_after_expiry"] == 1

    def test_doubling_the_divisor_halves_the_points(self):
        common = dict(as_of="2026-03-01", expiry="2026-06-19")
        one = dividend_points(self.CONS, divisor=1e8, **common)
        two = dividend_points(self.CONS, divisor=2e8, **common)
        assert two["total_index_points"] == pytest.approx(
            one["total_index_points"] / 2.0
        )

    def test_a_dividend_already_ex_is_excluded(self):
        out = dividend_points(
            self.CONS, divisor=1e8, as_of="2026-05-01", expiry="2026-06-19"
        )
        assert out["n_excluded_already_ex"] == 1
        assert out["total_index_points"] == pytest.approx(5e8 * 2.4 / 1e8)

    def test_implying_from_a_future_needs_the_financing_too(self):
        with pytest.raises(ValidationError, match="financing_rate"):
            dividend_points(
                self.CONS,
                divisor=1e8,
                as_of="2026-03-01",
                expiry="2026-06-19",
                future_price=4985,
                spot=5000,
            )


class TestIndexRebalance:
    OLD = {"A": 0.30, "B": 0.40, "C": 0.30}
    NEW = {"A": 0.28, "B": 0.37, "XYZ": 0.0035, "C": 0.3465}

    def test_flow_scales_linearly_with_indexed_assets(self):
        small = index_rebalance_flow(
            old_weights=self.OLD, new_weights=self.NEW, indexed_assets=100e9
        )
        big = index_rebalance_flow(
            old_weights=self.OLD, new_weights=self.NEW, indexed_assets=800e9
        )
        assert big["gross_notional"] == pytest.approx(small["gross_notional"] * 8.0)

    def test_turnover_is_half_the_absolute_weight_change(self):
        out = index_rebalance_flow(
            old_weights=self.OLD, new_weights=self.NEW, indexed_assets=1e9
        )
        expected = (
            sum(
                abs(self.NEW.get(k, 0.0) - self.OLD.get(k, 0.0))
                for k in set(self.OLD) | set(self.NEW)
            )
            / 2.0
            * 100.0
        )
        assert out["turnover_pct"] == pytest.approx(expected)

    def test_an_addition_and_a_deletion_are_named(self):
        out = index_rebalance_flow(
            old_weights={"A": 0.5, "GONE": 0.5},
            new_weights={"A": 0.5, "NEW": 0.5},
            indexed_assets=1e9,
        )
        events = {row["symbol"]: row["event"] for row in out["changes"]}
        assert events["NEW"] == "addition"
        assert events["GONE"] == "deletion"
        assert out["n_additions"] == 1 and out["n_deletions"] == 1

    def test_a_smaller_auction_makes_participation_worse(self):
        common = dict(
            old_weights=self.OLD,
            new_weights=self.NEW,
            indexed_assets=800e9,
            adv={"A": 2e9, "B": 1.5e9, "C": 8e8, "XYZ": 4.5e8},
        )
        whole_day = index_rebalance_flow(auction_fraction=1.0, **common)
        auction = index_rebalance_flow(auction_fraction=0.1, **common)
        a = {r["symbol"]: r["auction_participation"] for r in whole_day["changes"]}
        b = {r["symbol"]: r["auction_participation"] for r in auction["changes"]}
        for symbol in a:
            assert b[symbol] == pytest.approx(a[symbol] * 10.0)
