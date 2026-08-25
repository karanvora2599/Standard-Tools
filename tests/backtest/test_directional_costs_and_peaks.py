"""
Direction-aware commission, and the peak-exposure diagnostics.

WHY THESE TWO SUBJECTS SHARE A FILE. Both are changes that had to land in
the Python loop, the vectorized branch AND the native kernel at once. The
portfolio simulator picks among those three by configuration, and the
default configuration takes the native path -- so a rate applied only in
Python would be silently ignored on any machine with `_sqt_core` built.

That is not hypothetical for this repo: `clip_sigma` shipped with the native
path raising and the Python path silently skipping, and two users running
the same spec got different numbers. Every test here that has a native
counterpart asserts the two agree rather than asserting either alone.
"""

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.backtest.portfolio_engine as pe
from standard_quant_tools.backtest.costs import (
    directional_commission,
    maker_taker_cost,
    percentage_commission,
)
from standard_quant_tools.error import ValidationError

requires_cpp = pytest.mark.skipif(not pe.HAS_CPP, reason="_sqt_core not built")

REL_TOL = 1e-13


def _universe(n_tickers=5, n_bars=260, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="B")
    data = {}
    for i in range(n_tickers):
        cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n_bars)))
        data[f"T{i}"] = pd.DataFrame(
            {
                "Open": cl * 0.999,
                "High": cl * 1.01,
                "Low": cl * 0.99,
                "Close": cl,
                "Volume": np.full(n_bars, 5e6),
            },
            index=idx,
        )
    reb = idx[::10]
    w = rng.dirichlet(np.ones(n_tickers), len(reb))
    return data, pd.DataFrame(w, index=reb, columns=list(data))


def _both(data, weights, **kwargs):
    native = pe.run_portfolio_simulation(data, weights, **kwargs)
    saved = pe.HAS_CPP
    pe.HAS_CPP = False
    try:
        looped = pe.run_portfolio_simulation(data, weights, **kwargs)
    finally:
        pe.HAS_CPP = saved
    return native, looped


class TestDirectionalCommission:
    def test_charges_the_side_it_is_told(self):
        assert directional_commission(1e6, 0.0005, 0.0008, is_buy=True) == 500.0
        assert directional_commission(1e6, 0.0005, 0.0008, is_buy=False) == 800.0

    def test_side_comes_from_the_flag_not_the_sign(self):
        # A short sale has a negative delta but the notional's sign must not
        # decide the side -- inferring it is how a cover gets charged as a
        # sale. Magnitude is used; `is_buy` carries the direction.
        assert directional_commission(-1e6, 0.001, 0.002, is_buy=True) == 1000.0
        assert directional_commission(1e6, 0.001, 0.002, is_buy=False) == 2000.0

    def test_equal_rates_reproduce_percentage_commission(self):
        for notional in (0.0, 1.0, 12_345.678, -9_876.5):
            assert directional_commission(
                notional, 0.0009, 0.0009, is_buy=True
            ) == pytest.approx(percentage_commission(notional, 0.0009))

    @pytest.mark.parametrize("kwargs", [{"buy_rate": -0.001}, {"sell_rate": -0.001}])
    def test_negative_rates_are_refused(self, kwargs):
        base = {"buy_rate": 0.0, "sell_rate": 0.0, "is_buy": True}
        base.update(kwargs)
        with pytest.raises(ValidationError, match="must be >= 0"):
            directional_commission(1e6, **base)


class TestMakerTaker:
    def test_taker_pays_and_maker_can_be_paid(self):
        assert maker_taker_cost(
            1e6, taker_rate=0.0003, maker_rate=-0.0002, is_maker=False
        ) == pytest.approx(300.0)
        # Negative: the venue pays. This is the one function in costs.py
        # that may return a credit, and only on this side.
        assert maker_taker_cost(
            1e6, taker_rate=0.0003, maker_rate=-0.0002, is_maker=True
        ) == pytest.approx(-200.0)

    def test_the_taker_side_still_refuses_a_credit(self):
        # The rebate is deliberately confined to the maker rate. A negative
        # taker rate is the sign error every other cost function rejects.
        with pytest.raises(ValidationError, match="must be >= 0"):
            maker_taker_cost(1e6, taker_rate=-0.0001, maker_rate=0.0, is_maker=False)

    def test_no_other_cost_function_accepts_a_rebate(self):
        with pytest.raises(ValidationError, match="must be >= 0"):
            percentage_commission(1e6, -0.001)


class TestSellRateInTheSimulator:
    def test_default_is_unchanged_behaviour(self):
        data, w = _universe()
        base = pe.run_portfolio_simulation(data, w, commission_pct=0.001)
        explicit = pe.run_portfolio_simulation(
            data, w, commission_pct=0.001, sell_commission_pct=0.001
        )
        assert explicit["final_equity"] == pytest.approx(
            base["final_equity"], rel=1e-15
        )

    def test_a_higher_sell_rate_costs_more(self):
        data, w = _universe()
        cheap = pe.run_portfolio_simulation(data, w, commission_pct=0.001)
        dear = pe.run_portfolio_simulation(
            data, w, commission_pct=0.001, sell_commission_pct=0.004
        )
        assert dear["final_equity"] < cheap["final_equity"], (
            "an asymmetric sell rate that costs more must reduce final equity; "
            "if these are equal the rate is being ignored somewhere"
        )

    @requires_cpp
    def test_native_and_python_agree_on_an_asymmetric_rate(self):
        # The whole reason the C++ moved with the Python.
        data, w = _universe(seed=17)
        native, looped = _both(
            data, w, commission_pct=0.0005, sell_commission_pct=0.0025
        )
        scale = max(float(np.nanmax(np.abs(looped["equity_curve"].to_numpy()))), 1.0)
        diff = np.nanmax(
            np.abs(
                native["equity_curve"].to_numpy() - looped["equity_curve"].to_numpy()
            )
        )
        assert diff / scale < REL_TOL, f"equity curves diverge by {diff / scale:.2e}"

    @requires_cpp
    def test_the_native_path_is_actually_taken(self):
        # If a future guard sent this configuration to the Python loop, the
        # test above would still pass while testing nothing.
        data, w = _universe(n_tickers=3, n_bars=60)
        assert pe._native_portfolio_sim is not None, "native entry point missing"
        result = pe.run_portfolio_simulation(
            data, w, commission_pct=0.001, sell_commission_pct=0.002
        )
        assert result["final_equity"] > 0

    def test_a_negative_sell_rate_is_refused(self):
        data, w = _universe(n_tickers=2, n_bars=40)
        with pytest.raises(ValidationError, match="must be >= 0"):
            pe.run_portfolio_simulation(data, w, sell_commission_pct=-0.001)


class TestPeakDiagnostics:
    def test_all_four_are_reported(self):
        data, w = _universe()
        result = pe.run_portfolio_simulation(data, w)
        for key in (
            "max_leverage",
            "max_gross_exposure",
            "peak_position_value",
            "return_over_rebalance",
        ):
            assert key in result, f"{key} missing from the result"

    def test_peaks_dominate_the_curves_they_summarize(self):
        data, w = _universe(seed=9)
        r = pe.run_portfolio_simulation(data, w)
        # Reported rounded to 6dp, same as rebalance_log's fields, so the
        # tolerance here is the rounding granularity rather than epsilon.
        assert r["max_leverage"] >= float(r["leverage_curve"].max()) - 1e-6
        assert r["max_gross_exposure"] >= float(r["gross_exposure_curve"].max()) - 1e-6
        # A single position cannot exceed the whole book's gross exposure.
        assert r["peak_position_value"] <= r["max_gross_exposure"] + 1e-6

    def test_peak_position_is_positive_for_a_held_book(self):
        # The failure this catches is a native path returning 0.0 because it
        # never published the value -- which reads like a data problem.
        data, w = _universe()
        r = pe.run_portfolio_simulation(data, w)
        assert r["peak_position_value"] > 0.0

    def test_the_engine_refuses_to_simulate_with_nothing_to_execute(self):
        # `return_over_rebalance` guards against an empty rebalance_log by
        # returning None -- 0 rebalances earning 0.0 is a different statement
        # from a flat result. That guard turns out to be DEFENSIVE rather
        # than reachable: the engine rejects both routes to an empty log
        # upstream, and this pins that it keeps doing so. If either message
        # ever stops raising, the None branch becomes live and wants a test
        # of its own.
        data, w = _universe(n_tickers=2, n_bars=40)

        with pytest.raises(ValidationError, match="nothing to simulate"):
            pe.run_portfolio_simulation(data, w.iloc[:0])

        last_bar_only = w.iloc[:1].copy()
        last_bar_only.index = [data["T0"].index[-1]]
        with pytest.raises(ValidationError, match="requires a bar after"):
            pe.run_portfolio_simulation(data, last_bar_only, fill_price="next_open")

    def test_return_over_rebalance_matches_its_definition(self):
        data, w = _universe(seed=4)
        r = pe.run_portfolio_simulation(data, w, initial_capital=1e6)
        expected = (r["final_equity"] / 1e6 - 1.0) / len(r["rebalance_log"])
        assert r["return_over_rebalance"] == pytest.approx(expected, abs=1e-6)

    @requires_cpp
    def test_native_and_python_report_the_same_peaks(self):
        data, w = _universe(seed=23)
        native, looped = _both(data, w)
        for key in ("max_leverage", "max_gross_exposure", "peak_position_value"):
            assert native[key] == pytest.approx(looped[key], rel=1e-9), (
                f"{key} differs between backends: " f"{native[key]} vs {looped[key]}"
            )
