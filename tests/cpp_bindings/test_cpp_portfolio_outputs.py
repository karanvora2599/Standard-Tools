"""
Every element `run_portfolio_simulation` returns is defined.

The kernel writes a bar only once it is marked and a rebalance row only once
it executes, and the binding handed back numpy arrays it had not initialised.
So an early stop returned whatever the allocator recycled -- measured, a
failing run's equity tail was the PREVIOUS run's equity curve -- and a
next_open trigger on the last bar left its rebalance row as garbage on a
status-0 run. The contract now: bars the simulation never marked are NaN, and
rebalance rows from n_executed on are NaN.

The Python engine never read those elements (it raises on a non-zero status
and slices the log to n_executed), so this is about direct callers of the
binding.
"""

import numpy as np
import pytest

_cpp = pytest.importorskip(
    "standard_quant_tools._sqt_core",
    reason="native extension not built",
)

N_BARS = 8
CLOSE = np.column_stack(
    [100.0 + np.arange(N_BARS, dtype=float), 50.0 + 0.5 * np.arange(N_BARS)]
)
WEIGHTS = np.array([[0.5, 0.5], [0.2, 0.8]])


def _run(exec_prices=CLOSE, rebal=(0, 4), **kwargs):
    return _cpp.run_portfolio_simulation(
        CLOSE,
        np.ascontiguousarray(exec_prices),
        WEIGHTS,
        np.array(rebal, dtype=np.int64),
        np.ones(N_BARS),
        commission_pct=0.0,
        sell_commission_pct=0.0,
        slippage_pct=0.0,
        **kwargs,
    )


def _per_bar(res):
    return np.column_stack([res["equity"], res["cash"], res["gross"], res["net"]])


class TestUnreachedElementsAreNaN:
    def test_a_failed_run_does_not_return_the_previous_runs_curve(self):
        """Planted: a healthy run fills numpy's free list with its equity
        curve, then a failing run of the same shape is allocated from it."""
        for _ in range(3):
            healthy = _run(initial_capital=7777.0)
            assert healthy["status"] == 0
            failed = _run(initial_capital=7777.0, max_gross_leverage=0.5)
            assert failed["status"] != 0
            bar = int(failed["bar"])
            assert bar == 0
            assert np.isnan(_per_bar(failed)[bar:]).all()
            assert np.isnan(failed["rebalances"][int(failed["n_executed"]) :]).all()

    def test_a_bad_price_mid_run_leaves_a_nan_tail(self):
        exec_prices = CLOSE.copy()
        exec_prices[4, 1] = np.nan  # the second rebalance buys ticker 1 here
        res = _run(exec_prices=exec_prices)
        assert res["status"] != 0 and int(res["bar"]) == 4
        assert int(res["n_executed"]) == 1
        assert np.isfinite(_per_bar(res)[:4]).all()
        assert np.isnan(_per_bar(res)[4:]).all()
        assert np.isfinite(res["rebalances"][0]).all()
        assert np.isnan(res["rebalances"][1]).all()

    def test_a_next_open_trigger_on_the_last_bar(self):
        """status 0, one row never executes -- no following Open to fill at."""
        res = _run(rebal=(0, N_BARS - 1), fill=1)
        assert res["status"] == 0
        assert int(res["n_executed"]) == 1
        assert np.isfinite(_per_bar(res)).all()
        assert np.isfinite(res["rebalances"][0]).all()
        assert np.isnan(res["rebalances"][1]).all()

    def test_a_full_run_has_no_nan(self):
        """The null case."""
        res = _run()
        assert res["status"] == 0
        assert int(res["n_executed"]) == 2
        assert np.isfinite(_per_bar(res)).all()
        assert np.isfinite(res["rebalances"]).all()

    def test_the_docstring_states_the_contract(self):
        doc = _cpp.run_portfolio_simulation.__doc__
        assert "NaN" in doc
        assert "percentage commission, no impact model" not in doc
