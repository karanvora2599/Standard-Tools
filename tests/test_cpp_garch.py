"""
Python integration tests for the GARCH(1,1) variance recursion C++
extension and Python wrapper.

Two execution modes:
  1. _sqt_core NOT built → cpp_* tests are skipped; wrapper tests verify the
     numba fallback produces correct results (see test_garch.py).
  2. _sqt_core IS built  → all tests run; cross-validates C++ vs numba.

This is a pure deterministic recursion, so cross-backend comparisons use
the standard atol=1e-10 precedent (unlike test_cpp_monte_carlo.py, which
compares distributions rather than exact values).

Run:
    pytest tests/test_cpp_garch.py -v
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

from standard_quant_tools.analysis.garch import (
    _garch11_variance_recursion_numba,
    garch_volatility_forecast,
)


class TestCppGarchVsNumba:
    """Direct comparison of _cpp.garch11_variance_recursion against the
    numba reference at the standard atol=1e-10 precedent."""

    @requires_cpp
    def test_matches_numba_reference(self):
        rng = np.random.default_rng(0)
        resid_sq = (rng.standard_normal(500) * 0.01) ** 2
        omega, alpha, beta = 1e-6, 0.05, 0.9

        cpp_result = _cpp.garch11_variance_recursion(resid_sq, omega, alpha, beta)
        numba_result = _garch11_variance_recursion_numba(resid_sq, omega, alpha, beta)

        np.testing.assert_allclose(cpp_result, numba_result, atol=1e-10)

    @requires_cpp
    def test_matches_numba_reference_across_parameter_grid(self):
        rng = np.random.default_rng(1)
        resid_sq = (rng.standard_normal(300) * 0.01) ** 2
        for omega, alpha, beta in [
            (1e-8, 0.01, 0.5),
            (1e-5, 0.2, 0.7),
            (1e-6, 0.05, 0.94),
        ]:
            cpp_result = _cpp.garch11_variance_recursion(resid_sq, omega, alpha, beta)
            numba_result = _garch11_variance_recursion_numba(
                resid_sq, omega, alpha, beta
            )
            np.testing.assert_allclose(cpp_result, numba_result, atol=1e-10)

    @requires_cpp
    def test_returns_correct_length(self):
        resid_sq = np.full(123, 1e-4)
        result = _cpp.garch11_variance_recursion(resid_sq, 1e-6, 0.05, 0.9)
        assert len(result) == 123

    @requires_cpp
    def test_empty_input_returns_empty(self):
        result = _cpp.garch11_variance_recursion(
            np.array([], dtype=float), 1e-6, 0.05, 0.9
        )
        assert len(result) == 0

    @requires_cpp
    def test_floor_at_min_sigma2(self):
        # A large negative omega/alpha/beta combination could otherwise
        # drive sigma2 negative -- must clamp at the same 1e-12 floor as
        # the numba reference.
        resid_sq = np.zeros(50)
        result = _cpp.garch11_variance_recursion(resid_sq, -1.0, 0.0, 0.0)
        assert np.all(result >= 1e-12)


class TestGarchForecastEndToEndParity:
    """Confirms garch_volatility_forecast()'s public output is identical
    whether _sqt_core is built or not -- the C++ kernel is purely an
    internal performance detail of _garch11_variance_recursion."""

    @requires_cpp
    def test_forecast_output_identical_with_and_without_cpp(self):
        import pandas as pd

        import standard_quant_tools.analysis.garch as garch_module

        rng = np.random.default_rng(2)
        returns = pd.Series(rng.standard_normal(400) * 0.01)

        result_cpp = garch_volatility_forecast(returns, forecast_horizon=10)

        garch_module.HAS_CPP = False
        try:
            result_numba = garch_volatility_forecast(returns, forecast_horizon=10)
        finally:
            garch_module.HAS_CPP = True

        assert result_cpp["omega"] == pytest.approx(result_numba["omega"], abs=1e-10)
        assert result_cpp["alpha"] == pytest.approx(result_numba["alpha"], abs=1e-10)
        assert result_cpp["beta"] == pytest.approx(result_numba["beta"], abs=1e-10)
        assert result_cpp["current_annualized_vol"] == pytest.approx(
            result_numba["current_annualized_vol"], abs=1e-10
        )
        assert result_cpp["forecast_annualized_vol"] == pytest.approx(
            result_numba["forecast_annualized_vol"], abs=1e-10
        )
