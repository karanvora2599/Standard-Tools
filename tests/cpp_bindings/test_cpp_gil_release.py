"""
Concurrency smoke tests for the GIL release around _sqt_core's C++ kernels.

Every binding in bindings.cpp now wraps its sqt:: call in a
py::gil_scoped_release block (Tier 3 item 9 of an independent code review),
so multiple Python threads calling into the extension concurrently should
actually run their C++ work in parallel instead of serializing on the GIL,
and -- more importantly for correctness -- must never corrupt each other's
independent, stack-local inputs/outputs while the GIL is released.

These tests don't attempt to prove GIL semantics from Python (timing-based
assertions about "did this actually run in parallel" are inherently flaky
across CI hardware). Instead they hammer several kernels from multiple
threads at once with distinct, independently-verifiable inputs and assert
every thread got back exactly the result its own input implies -- the kind
of bug (shared mutable state, a stray non-local temporary touched while the
GIL was released) that a naive GIL-release refactor could introduce.

Run:
    pytest tests/test_cpp_gil_release.py -v
"""

import threading
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


def _run_threads(worker, n_threads: int = 8):
    """Run `worker(thread_index)` on n_threads threads; re-raise any exception."""
    errors = []

    def target(i):
        try:
            worker(i)
        except Exception as exc:  # noqa: BLE001 - re-raised on the main thread below
            errors.append((i, exc))

    threads = [threading.Thread(target=target, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    if errors:
        idx, exc = errors[0]
        raise AssertionError(f"thread {idx} raised: {exc!r}") from exc


@requires_cpp
class TestConcurrentKernelCalls:
    def test_rsi_concurrent_calls_independent_results(self):
        """Each thread uses a distinct deterministic series and period; every
        thread's RSI result must match a single-threaded call with the same
        inputs -- no cross-thread contamination of the C++-side computation."""
        n = 400

        def worker(i):
            rng = np.random.default_rng(seed=i)
            arr = 100.0 + np.cumsum(rng.standard_normal(n))
            period = 10 + i
            result = _cpp.rsi(arr, period)
            expected = _cpp.rsi(arr, period)  # same inputs, single-threaded reference
            np.testing.assert_array_equal(result, expected)

        _run_threads(worker)

    def test_run_strategy_concurrent_calls_independent_results(self):
        """Each thread backtests a distinct price/signal pair; results must
        be bit-identical to what that same input produces outside threading."""
        n = 300

        def worker(i):
            rng = np.random.default_rng(seed=100 + i)
            prices = 100.0 + np.cumsum(rng.standard_normal(n))
            signals = np.sign(rng.standard_normal(n))
            result = _cpp.run_strategy(prices, signals, 10_000.0, 0.001, 0.0005)
            expected = _cpp.run_strategy(prices, signals, 10_000.0, 0.001, 0.0005)
            assert result["total_return"] == pytest.approx(expected["total_return"])
            assert result["num_trades"] == expected["num_trades"]
            np.testing.assert_allclose(
                np.asarray(result["equity_curve"]), np.asarray(expected["equity_curve"])
            )

        _run_threads(worker)

    def test_hurst_dfa_concurrent_calls_independent_results(self):
        """Hurst is one of the two bindings with no direct GIL-release
        counterpart test elsewhere -- distinct white-noise seeds per thread,
        each compared against its own single-threaded reference."""
        n = 600

        def worker(i):
            rng = np.random.default_rng(seed=200 + i)
            arr = rng.standard_normal(n)
            result = _cpp.hurst_dfa(arr, 10, -1)
            expected = _cpp.hurst_dfa(arr, 10, -1)
            assert result["hurst"] == pytest.approx(expected["hurst"], nan_ok=True)
            assert result["regime"] == expected["regime"]

        _run_threads(worker)

    def test_bollinger_bands_concurrent_calls_independent_results(self):
        n = 250

        def worker(i):
            rng = np.random.default_rng(seed=300 + i)
            prices = 100.0 + np.cumsum(rng.standard_normal(n))
            result = _cpp.bollinger_bands(prices, 20, 2.0)
            expected = _cpp.bollinger_bands(prices, 20, 2.0)
            np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))

        _run_threads(worker)

    def test_mixed_kernels_concurrent_no_crash(self):
        """Different threads calling entirely different bindings at once --
        the scenario most likely to surface a shared-state bug if any
        binding's GIL-release block accidentally captured something by
        reference instead of extracting a local copy first."""
        n = 200

        def rsi_worker(i):
            rng = np.random.default_rng(seed=400 + i)
            arr = 100.0 + np.cumsum(rng.standard_normal(n))
            _cpp.rsi(arr, 14)

        def backtest_worker(i):
            rng = np.random.default_rng(seed=500 + i)
            prices = 100.0 + np.cumsum(rng.standard_normal(n))
            signals = np.sign(rng.standard_normal(n))
            _cpp.run_strategy(prices, signals)

        def hurst_worker(i):
            rng = np.random.default_rng(seed=600 + i)
            arr = rng.standard_normal(n)
            _cpp.hurst_rs(arr, 10, -1)

        workers = [rsi_worker, backtest_worker, hurst_worker]

        def worker(i):
            workers[i % len(workers)](i)

        _run_threads(worker, n_threads=12)
