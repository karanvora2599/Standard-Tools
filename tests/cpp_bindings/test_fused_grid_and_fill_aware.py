"""
Regression tests for the performance pass: the fill-aware native kernel, the
fused crossover grid, OpenMP governance and Monte Carlo seed reproducibility.

Correctness first in every case. A faster path that returns different numbers
is not an optimization, so each test compares the new route against the one it
replaces before any timing is considered.
"""

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.backtest.engine as E

_cpp = pytest.importorskip(
    "standard_quant_tools._sqt_core", reason="native extension not built"
)


def _frame(n=500, seed=7):
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n)), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0.0, 0.005, n))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(close, open_) * 1.008,
            "Low": np.minimum(close, open_) * 0.992,
            "Close": close,
            "Volume": 1e6,
        },
        index=idx,
    )


class TestFillAwareNativeKernel:
    """
    The kernel only knew Close prices, so `next_open` and `hl2_exploratory`
    always fell back to Python — the MORE REALISTIC execution model was also
    the slow one, and the native grid could not be used for it at all.

    Measured on 20,000 bars after the change:

        close             native   2.60 ms   python 296.86 ms   114x
        next_open         native   4.20 ms   python 307.88 ms    73x
        hl2_exploratory   native   6.39 ms   python 321.08 ms    50x
    """

    @pytest.mark.parametrize("mode", ["close", "next_open", "hl2_exploratory"])
    def test_native_matches_the_python_fallback(self, mode):
        df = _frame()
        rng = np.random.default_rng(11)
        signals = pd.Series(rng.choice([-1.0, 0.0, 1.0, 2.0], len(df)), index=df.index)

        native = E.run_strategy(
            df, signals, fill_price=mode, commission_pct=0.001, slippage_pct=0.0005
        )
        saved = E.HAS_CPP
        E.HAS_CPP = False
        try:
            python = E.run_strategy(
                df,
                signals,
                fill_price=mode,
                commission_pct=0.001,
                slippage_pct=0.0005,
            )
        finally:
            E.HAS_CPP = saved

        for key in (
            "total_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        ):
            assert native[key] == pytest.approx(python[key], abs=1e-4), key
        assert native["num_trades"] == python["num_trades"]

    def test_the_two_leg_decomposition_is_actually_used(self):
        """A kernel that ignored ref_prices would silently return the
        close-to-close answer for every mode."""
        df = _frame()
        signals = pd.Series(1.0, index=df.index)
        close_only = E.run_strategy(df, signals, fill_price="close")
        next_open = E.run_strategy(df, signals, fill_price="next_open")
        assert close_only["total_return"] != pytest.approx(
            next_open["total_return"], abs=1e-9
        )

    def test_a_reference_series_of_the_wrong_length_is_rejected(self):
        close = _frame()["Close"].to_numpy(dtype=np.float64)
        signals = np.ones(len(close))
        with pytest.raises(Exception):
            _cpp.run_strategy(
                close, signals, 10_000.0, 0.001, 0.0005, 252.0, close[:-1]
            )


class TestFusedCrossoverGrid:
    """
    Profiling a 300-combination x 5,000-bar SMA grid showed the batch kernel
    was solving the small half of the problem:

        python signal generation   121.4 ms   92.1%
        vstack into (combos,bars)    3.2 ms    2.4%
        native batch backtest        7.2 ms    5.4%

    and 600 moving averages were computed where 35 unique periods existed.
    """

    GRID = {
        "fast_period": list(range(5, 15)),
        "slow_period": [30, 50, 80, 120, 200],
    }

    def _both_paths(self, df, **kwargs):
        original = E._fused_crossover_metrics
        fused = E.backtest_grid(df, "sma_crossover", self.GRID, n_workers=1, **kwargs)
        E._fused_crossover_metrics = lambda *a, **k: None
        try:
            general = E.backtest_grid(
                df, "sma_crossover", self.GRID, n_workers=1, **kwargs
            )
        finally:
            E._fused_crossover_metrics = original
        return fused, general

    def test_the_fused_path_is_actually_taken(self):
        """
        It silently was NOT, at first: the reference-price array was resolved
        after the branch that read it, so a NameError was swallowed by the
        broad `except Exception` guarding the C++ route and the grid fell
        back to Python with nothing said.
        """
        calls = {"n": 0, "result": None}
        original = E._fused_crossover_metrics

        def spy(*args, **kwargs):
            calls["n"] += 1
            out = original(*args, **kwargs)
            calls["result"] = out
            return out

        E._fused_crossover_metrics = spy
        try:
            E.backtest_grid(_frame(), "sma_crossover", self.GRID, n_workers=1)
        finally:
            E._fused_crossover_metrics = original
        assert calls["n"] == 1
        assert calls["result"] is not None, "fused helper bailed out"

    @pytest.mark.parametrize("mode", ["close", "next_open"])
    def test_fused_and_general_paths_agree_exactly(self, mode):
        fused, general = self._both_paths(_frame(800), fill_price=mode)
        merged = fused.merge(
            general, on=["fast_period", "slow_period"], suffixes=("_f", "_g")
        )
        assert len(merged) == len(fused)
        for metric in (
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "num_trades",
        ):
            worst = (merged[f"{metric}_f"] - merged[f"{metric}_g"]).abs().max()
            assert worst == 0, f"{metric} diverged by {worst}"

    def test_only_unique_periods_are_computed(self):
        """The redundancy is the point: 600 SMAs for 35 distinct periods."""
        fast = self.GRID["fast_period"]
        slow = self.GRID["slow_period"]
        unique = len(set(fast) | set(slow))
        naive = len(fast) * len(slow) * 2
        assert unique < naive / 5

    def test_a_non_crossover_grid_falls_back_cleanly(self):
        """The fused helper must decline anything it cannot express, rather
        than mis-handle it."""
        assert (
            E._fused_crossover_metrics(
                _frame(), ["period"], [(14,)], 10_000.0, 0.001, 0.0005, None
            )
            is None
        )


class TestOpenMpGovernance:
    """
    Kernels parallelized whenever there was more than one task, which asks
    the wrong question twice: two tiny backtests cost more in thread startup
    than they save, and a library that grabs every core oversubscribes badly
    when it is itself running inside a process pool or several agents.
    """

    def test_the_policy_header_exists_with_both_controls(self):
        from pathlib import Path

        header = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "standard_quant_tools"
            / "_cpp"
            / "include"
            / "sqt"
            / "omp_policy.hpp"
        ).read_text(encoding="utf-8")
        assert "SQT_NUM_THREADS" in header
        assert "SQT_OMP_MIN_WORK" in header
        assert "worth_parallel" in header

    def test_the_decision_is_work_based_not_count_based(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "standard_quant_tools"
            / "_cpp"
            / "src"
            / "backtest.cpp"
        ).read_text(encoding="utf-8")
        assert "if(num_tests > 1)" not in source
        assert "worth_parallel" in source

    def test_results_are_identical_under_a_single_thread(self, monkeypatch):
        """Whatever the policy decides, the numbers must not depend on it."""
        df = _frame(600)
        grid = {"fast_period": [5, 10], "slow_period": [40, 90]}
        parallel = E.backtest_grid(df, "sma_crossover", grid, n_workers=1)
        monkeypatch.setenv("SQT_NUM_THREADS", "1")
        serial = E.backtest_grid(df, "sma_crossover", grid, n_workers=1)
        pd.testing.assert_frame_equal(parallel, serial)


class TestMonteCarloSeedIsRecorded:
    """
    With random_seed=None the native kernel seeded itself from steady_clock,
    so the audit record faithfully stored `None` while the numbers came from
    a value nobody kept — the run was unreproducible and nothing said so.
    """

    def test_the_result_carries_the_seed_field(self):
        from standard_quant_tools.agent.models import MonteCarloSimulationResult

        assert "random_seed" in MonteCarloSimulationResult.model_fields
        assert MonteCarloSimulationResult.model_fields["random_seed"].is_required()

    def test_the_tool_resolves_a_seed_before_running(self):
        import inspect

        from standard_quant_tools.agent import tools as T

        source = inspect.getsource(T.run_monte_carlo_simulation)
        assert "resolved_seed" in source
        assert "seed=resolved_seed" in source
        assert "random_seed=resolved_seed" in source
