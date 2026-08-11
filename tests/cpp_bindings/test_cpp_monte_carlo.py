"""
Python integration tests for the Monte Carlo (moving-block bootstrap) C++
extension and Python wrapper.

Two execution modes:
  1. _sqt_core NOT built → cpp_* tests are skipped; wrapper tests verify the
     pure-Python fallback produces correct results (see test_monte_carlo.py).
  2. _sqt_core IS built  → all tests run; cross-validates C++ vs pure Python.

Unlike the other test_cpp_*.py files, cross-backend comparisons here use
statistical tolerance, not atol=1e-10 -- the C++ path's RNG does not
reproduce numpy's PCG64 bit stream, so results only agree in distribution,
not bit-for-bit, even with the same seed.

Run:
    pytest tests/test_cpp_monte_carlo.py -v
"""

from typing import Any

import numpy as np
import pytest

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

requires_cpp = pytest.mark.skipif(not HAS_CPP, reason="_sqt_core not built")


class TestCppSimulateForwardPathsDirect:
    """Direct calls to _sqt_core.simulate_forward_paths -- bypasses the
    Python wrapper."""

    @requires_cpp
    def test_returns_correct_shape(self):
        values = np.full(100, 0.001)
        out = _cpp.simulate_forward_paths(values, 30, 200, 10, 10_000.0, 1)
        assert out.shape == (200, 30)

    @requires_cpp
    def test_no_nan_or_inf(self):
        rng = np.random.default_rng(0)
        values = rng.standard_normal(252) * 0.01
        out = _cpp.simulate_forward_paths(values, 60, 500, 20, 10_000.0, 42)
        assert np.isfinite(out).all()

    @requires_cpp
    def test_constant_return_series_is_deterministic_compounding(self):
        # Every historical bar has the identical return r, so every block
        # drawn -- regardless of which start index the RNG picks -- is the
        # same constant-r block. Every path must equal
        # initial_capital * (1+r)**t for every t, with zero cross-path
        # variance, independent of the RNG.
        r = 0.002
        values = np.full(50, r)
        out = _cpp.simulate_forward_paths(values, 25, 100, 10, 10_000.0, 7)
        expected = 10_000.0 * (1.0 + r) ** np.arange(1, 26)
        for row in out:
            assert np.allclose(row, expected, rtol=1e-10)

    @requires_cpp
    def test_same_seed_is_reproducible(self):
        rng = np.random.default_rng(1)
        values = rng.standard_normal(200) * 0.01
        out1 = _cpp.simulate_forward_paths(values, 40, 300, 15, 10_000.0, 123)
        out2 = _cpp.simulate_forward_paths(values, 40, 300, 15, 10_000.0, 123)
        assert np.array_equal(out1, out2)

    @requires_cpp
    def test_result_is_independent_of_thread_count(self, monkeypatch):
        """If simulate_forward_paths is compiled with OpenMP, each path's
        RNG must be fully independent of every other path's -- a per-path
        buffer shared across threads (a data race) would make results
        depend on how many threads happen to run the loop. Forcing 1
        thread vs. an unconstrained thread count for the identical
        seed+inputs must produce bit-identical output either way."""
        rng = np.random.default_rng(3)
        values = rng.standard_normal(400) * 0.01

        monkeypatch.setenv("OMP_NUM_THREADS", "1")
        out_1_thread = _cpp.simulate_forward_paths(values, 50, 2000, 20, 10_000.0, 55)

        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        out_many_threads = _cpp.simulate_forward_paths(
            values, 50, 2000, 20, 10_000.0, 55
        )

        assert np.array_equal(out_1_thread, out_many_threads)

    @requires_cpp
    def test_no_seed_is_not_deterministic(self):
        rng = np.random.default_rng(2)
        values = rng.standard_normal(200) * 0.01
        out1 = _cpp.simulate_forward_paths(values, 40, 300, 15, 10_000.0, None)
        out2 = _cpp.simulate_forward_paths(values, 40, 300, 15, 10_000.0, None)
        assert not np.array_equal(out1, out2)

    @requires_cpp
    def test_invalid_block_size_raises(self):
        values = np.full(20, 0.001)
        with pytest.raises(ValueError):
            _cpp.simulate_forward_paths(values, 30, 100, 0, 10_000.0, 1)
        with pytest.raises(ValueError):
            _cpp.simulate_forward_paths(values, 30, 100, 21, 10_000.0, 1)

    @requires_cpp
    def test_non_positive_initial_capital_raises(self):
        values = np.full(20, 0.001)
        with pytest.raises(ValueError):
            _cpp.simulate_forward_paths(values, 30, 100, 5, 0.0, 1)
        with pytest.raises(ValueError):
            _cpp.simulate_forward_paths(values, 30, 100, 5, -1.0, 1)

    @requires_cpp
    def test_non_positive_horizon_or_simulations_raises(self):
        values = np.full(20, 0.001)
        with pytest.raises(ValueError):
            _cpp.simulate_forward_paths(values, 0, 100, 5, 10_000.0, 1)
        with pytest.raises(ValueError):
            _cpp.simulate_forward_paths(values, 30, 0, 5, 10_000.0, 1)


class TestCppVsPythonStatisticalParity:
    """Cross-backend comparison, loose tolerance -- the RNGs genuinely
    differ, so this checks distributional agreement, not bit-equality."""

    @requires_cpp
    def test_terminal_distribution_agrees_within_mc_noise(self):
        import pandas as pd

        from standard_quant_tools.backtest import monte_carlo as mc_module

        rng = np.random.default_rng(99)
        returns = pd.Series(rng.standard_normal(500) * 0.01 + 0.0003)

        result_cpp = mc_module.simulate_forward_paths(
            returns,
            horizon_days=60,
            n_simulations=8000,
            block_size=20,
            initial_capital=10_000.0,
            seed=7,
        )

        mc_module.HAS_CPP = False
        try:
            result_py = mc_module.simulate_forward_paths(
                returns,
                horizon_days=60,
                n_simulations=8000,
                block_size=20,
                initial_capital=10_000.0,
                seed=7,
            )
        finally:
            mc_module.HAS_CPP = True

        # Same underlying data-generating process (moving-block bootstrap
        # of the identical historical series) with a large sample size --
        # medians should agree within ordinary Monte Carlo sampling noise,
        # not exactly. This is a distributional sanity check, not a
        # correctness proof of either path individually (that's covered by
        # the deterministic-compounding test above for C++, and
        # test_monte_carlo.py for the Python fallback).
        assert result_cpp["terminal_median"] == pytest.approx(
            result_py["terminal_median"], rel=0.15
        )
        assert result_cpp["prob_loss"] == pytest.approx(result_py["prob_loss"], abs=0.1)


class TestMonteCarloWrapperDispatch:
    """Confirms the Python wrapper actually calls the C++ path when
    available, rather than silently always falling back."""

    @requires_cpp
    def test_wrapper_uses_cpp_path_when_available(self, monkeypatch):
        import pandas as pd

        from standard_quant_tools.backtest import monte_carlo as mc_module

        calls = []
        original = _cpp.simulate_forward_paths

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(mc_module._cpp_core, "simulate_forward_paths", spy)

        returns = pd.Series(np.full(50, 0.001))
        mc_module.simulate_forward_paths(
            returns,
            horizon_days=10,
            n_simulations=20,
            block_size=5,
            initial_capital=10_000.0,
            seed=1,
        )
        assert len(calls) == 1
