"""
Regression coverage for the Array1D ndim-enforcement fix: `Array1D`
(`py::array_t<double, c_style | forcecast>`) only enforces dtype and
contiguity, not ndim, so a 2-D input used to silently flatten and get
misinterpreted as a 1-D series by every kernel in _sqt_core. Every binding
that takes an Array1D parameter now calls `require_1d()` first and raises
`ValueError` for anything that isn't exactly 1-D.

One parametrized sweep across all 20 bindings, rather than 37 near-
identical hand-written tests (one per Array1D parameter) -- each entry
supplies just enough correctly-shaped extra positional args to reach the
require_1d() check for the *first* Array1D parameter before anything else
in the lambda could raise a different, unrelated error first.

Run:
    pytest tests/test_cpp_array1d_validation.py -v
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

_BAD_2D = np.ones((10, 10))
_GOOD_1D = np.ones(20)

# (binding_name, args) -- args is a tuple of positional arguments with the
# first Array1D parameter replaced by _BAD_2D; every other array argument
# is a validly-shaped 1-D array so the require_1d() check under test is
# reached deterministically, not short-circuited by some other guard.
_BINDINGS_WITH_FIRST_ARG_BAD = [
    ("hurst_dfa", (_BAD_2D, 10, -1)),
    ("hurst_rs", (_BAD_2D, 10, -1)),
    ("rolling_hurst", (_BAD_2D, 5, 1, "dfa", 10)),
    ("rsi", (_BAD_2D, 14)),
    ("adx", (_BAD_2D, _GOOD_1D, _GOOD_1D, 14)),
    ("parabolic_sar", (_BAD_2D, _GOOD_1D, 0.02, 0.02, 0.2)),
    ("wilder_atr", (_BAD_2D, _GOOD_1D, _GOOD_1D, 14)),
    ("run_strategy", (_BAD_2D, _GOOD_1D, 10_000.0, 0.001, 0.0005)),
    ("batch_run_strategy", (_BAD_2D, _GOOD_1D.reshape(1, -1), 10_000.0, 0.001, 0.0005)),
    ("ols2", (_BAD_2D, _GOOD_1D)),
    ("rolling_factor_loadings", (_BAD_2D, np.ones((20, 3)), 5)),
    ("rolling_beta", (_BAD_2D, _GOOD_1D, 5)),
    ("bollinger_bands", (_BAD_2D, 20, 2.0)),
    ("stochastic_oscillator", (_BAD_2D, _GOOD_1D, _GOOD_1D, 14, 3)),
    ("engle_granger", (_BAD_2D, _GOOD_1D, -1, True)),
    ("simulate_forward_paths", (_BAD_2D, 5, 10, 5, 10_000.0, 1)),
    ("garch11_variance_recursion", (_BAD_2D, 1e-6, 0.05, 0.9)),
    ("kalman_filter_1state", (_BAD_2D, _GOOD_1D, 1e-4, 1e-3)),
    ("kalman_filter_2state", (_BAD_2D, _GOOD_1D, 1e-4, 1e-3)),
    ("donchian_state_machine", (_BAD_2D, _GOOD_1D, _GOOD_1D)),
    ("vwap_reversion_state_machine", (_BAD_2D, _GOOD_1D, 0.02)),
]


class TestArray1DRejects2D:
    @requires_cpp
    @pytest.mark.parametrize("name,args", _BINDINGS_WITH_FIRST_ARG_BAD)
    def test_2d_input_raises(self, name, args):
        fn = getattr(_cpp, name)
        with pytest.raises(ValueError, match="1-D"):
            fn(*args)

    @requires_cpp
    def test_valid_1d_input_still_works(self):
        # Sanity check the fix didn't also reject legitimate 1-D input --
        # pick one representative binding, not all 20 (already covered by
        # every other test_cpp_*.py file's normal-path tests).
        result = _cpp.rsi(_GOOD_1D, 14)
        assert len(result) == len(_GOOD_1D)
