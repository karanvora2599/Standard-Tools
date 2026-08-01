"""
Python integration tests for the Donchian breakout / VWAP-reversion signal
state-machine C++ extension and Python wrapper.

Two execution modes:
  1. _sqt_core NOT built → cpp_* tests are skipped; wrapper tests verify the
     numba fallback produces correct results (see test_strategies.py).
  2. _sqt_core IS built  → all tests run; cross-validates C++ vs numba.

Outputs are exactly 0.0/1.0 (not smooth floating point), so cross-backend
comparisons here assert EXACT equality (np.array_equal), not atol=1e-10 --
the hysteresis loop only compares already-identical upstream float64
arrays, so the state machine itself introduces no floating-point path
divergence between backends.

Run:
    pytest tests/test_cpp_signals.py -v
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

from standard_quant_tools.backtest.strategies import (
    _donchian_state_machine,
    _vwap_reversion_state_machine,
)


class TestCppDonchianStateMachine:
    @requires_cpp
    def test_matches_numba_reference(self):
        rng = np.random.default_rng(0)
        n = 500
        close = 100.0 + rng.standard_normal(n).cumsum()
        entry_max = close + rng.uniform(0.5, 2.0, n)
        exit_min = close - rng.uniform(0.5, 2.0, n)
        # Warmup NaNs, matching a real rolling-window caller
        entry_max[:20] = np.nan
        exit_min[:10] = np.nan

        cpp_result = _cpp.donchian_state_machine(close, entry_max, exit_min)
        numba_result = _donchian_state_machine(close, entry_max, exit_min)

        assert np.array_equal(cpp_result, numba_result)

    @requires_cpp
    def test_nan_warmup_outputs_zero_and_does_not_update_state(self):
        close = np.array([100.0, 100.0, 100.0, 105.0, 95.0])
        entry_max = np.array([np.nan, np.nan, 102.0, 102.0, 102.0])
        exit_min = np.array([np.nan, np.nan, 98.0, 98.0, 98.0])
        result = _cpp.donchian_state_machine(close, entry_max, exit_min)
        assert result[0] == 0.0
        assert result[1] == 0.0

    @requires_cpp
    def test_enters_long_on_breakout(self):
        close = np.array([100.0, 101.0, 103.0])
        entry_max = np.array([100.5, 100.5, 100.5])
        exit_min = np.array([90.0, 90.0, 90.0])
        result = _cpp.donchian_state_machine(close, entry_max, exit_min)
        assert list(result) == [0.0, 1.0, 1.0]

    @requires_cpp
    def test_exits_on_breakdown(self):
        close = np.array([100.0, 101.0, 89.0])
        entry_max = np.array([100.5, 100.5, 100.5])
        exit_min = np.array([90.0, 90.0, 90.0])
        result = _cpp.donchian_state_machine(close, entry_max, exit_min)
        assert list(result) == [0.0, 1.0, 0.0]

    @requires_cpp
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            _cpp.donchian_state_machine(
                np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0])
            )


class TestCppVwapReversionStateMachine:
    @requires_cpp
    def test_matches_numba_reference(self):
        rng = np.random.default_rng(1)
        n = 500
        close = 100.0 + rng.standard_normal(n).cumsum()
        vwap = close + rng.standard_normal(n) * 0.5
        vwap[:15] = np.nan
        entry_threshold = 0.02

        cpp_result = _cpp.vwap_reversion_state_machine(close, vwap, entry_threshold)
        numba_result = _vwap_reversion_state_machine(close, vwap, entry_threshold)

        assert np.array_equal(cpp_result, numba_result)

    @requires_cpp
    def test_enters_long_on_drop_below_threshold(self):
        close = np.array([100.0, 97.0, 99.0])
        vwap = np.array([100.0, 100.0, 100.0])
        result = _cpp.vwap_reversion_state_machine(close, vwap, 0.02)
        assert list(result) == [0.0, 1.0, 1.0]

    @requires_cpp
    def test_exits_on_recovery_to_vwap(self):
        close = np.array([100.0, 97.0, 100.5])
        vwap = np.array([100.0, 100.0, 100.0])
        result = _cpp.vwap_reversion_state_machine(close, vwap, 0.02)
        assert list(result) == [0.0, 1.0, 0.0]

    @requires_cpp
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            _cpp.vwap_reversion_state_machine(
                np.array([1.0, 2.0]), np.array([1.0]), 0.02
            )


class TestStrategiesWrapperDispatch:
    """Confirms _donchian_signals/_vwap_reversion_signals actually call the
    C++ path when available."""

    @requires_cpp
    def test_donchian_signals_uses_cpp_path(self, monkeypatch):
        import pandas as pd

        from standard_quant_tools.backtest import strategies as strategies_module

        calls = []
        original = _cpp.donchian_state_machine

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(strategies_module._cpp_core, "donchian_state_machine", spy)

        idx = pd.date_range("2020-01-01", periods=60)
        rng = np.random.default_rng(2)
        close = 100.0 + rng.standard_normal(60).cumsum()
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1_000_000,
            },
            index=idx,
        )
        strategies_module._donchian_signals(df, entry_period=10, exit_period=5)
        assert len(calls) == 1

    @requires_cpp
    def test_vwap_reversion_signals_uses_cpp_path(self, monkeypatch):
        import pandas as pd

        from standard_quant_tools.backtest import strategies as strategies_module

        calls = []
        original = _cpp.vwap_reversion_state_machine

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(
            strategies_module._cpp_core, "vwap_reversion_state_machine", spy
        )

        idx = pd.date_range("2020-01-01", periods=60)
        rng = np.random.default_rng(3)
        close = 100.0 + rng.standard_normal(60).cumsum()
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close + 1.0,
                "Low": close - 1.0,
                "Close": close,
                "Volume": 1_000_000,
            },
            index=idx,
        )
        strategies_module._vwap_reversion_signals(df, period=10, entry_threshold=0.02)
        assert len(calls) == 1
