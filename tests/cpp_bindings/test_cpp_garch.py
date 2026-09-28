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
    _garch11_neg_loglik,
    _garch11_variance_recursion_numba,
    garch_volatility_forecast,
)


def _numpy_neg_loglik(resid_sq, omega, alpha, beta, penalize=True):
    """Independent reference: pure NumPy, deliberately not sharing code
    with either the C++ or numba paths -- the exact formula
    _garch11_neg_loglik's own NumPy fallback branch used before this pass."""
    sigma2 = _garch11_variance_recursion_numba(resid_sq, omega, alpha, beta)
    nll = 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + resid_sq / sigma2)
    if penalize:
        persistence = alpha + beta
        if persistence >= 1.0:
            nll += 1.0e6 * (persistence - 1.0) ** 2
    return float(nll)


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


class TestCppNegLoglikVsNumpy:
    """Direct comparison of _cpp.garch11_neg_loglik (item 3 of the
    performance architecture review: fuses the variance recursion and the
    NLL reduction into one native call) against an independent pure-NumPy
    reference."""

    @requires_cpp
    def test_matches_numpy_reference_no_penalty(self):
        rng = np.random.default_rng(3)
        resid_sq = (rng.standard_normal(500) * 0.01) ** 2
        omega, alpha, beta = 1e-6, 0.05, 0.9  # persistence < 1
        cpp_nll = _cpp.garch11_neg_loglik(resid_sq, omega, alpha, beta, True)
        ref_nll = _numpy_neg_loglik(resid_sq, omega, alpha, beta, True)
        assert cpp_nll == pytest.approx(ref_nll, abs=1e-8)

    @requires_cpp
    def test_matches_numpy_reference_across_parameter_grid(self):
        rng = np.random.default_rng(4)
        resid_sq = (rng.standard_normal(300) * 0.01) ** 2
        for omega, alpha, beta in [
            (1e-8, 0.01, 0.5),
            (1e-5, 0.2, 0.7),
            (1e-6, 0.05, 0.94),
        ]:
            cpp_nll = _cpp.garch11_neg_loglik(resid_sq, omega, alpha, beta, True)
            ref_nll = _numpy_neg_loglik(resid_sq, omega, alpha, beta, True)
            assert cpp_nll == pytest.approx(ref_nll, abs=1e-8)

    @requires_cpp
    def test_penalty_branch_matches_numpy_reference(self):
        # persistence = alpha + beta = 1.05 >= 1.0 -> soft penalty applies.
        resid_sq = np.array([1e-4, 2e-4, 3e-4, 4e-4])
        omega, alpha, beta = 1e-6, 0.2, 0.85
        with_penalty = _cpp.garch11_neg_loglik(resid_sq, omega, alpha, beta, True)
        without_penalty = _cpp.garch11_neg_loglik(resid_sq, omega, alpha, beta, False)
        assert with_penalty > without_penalty
        assert with_penalty == pytest.approx(
            _numpy_neg_loglik(resid_sq, omega, alpha, beta, True), abs=1e-8
        )
        assert without_penalty == pytest.approx(
            _numpy_neg_loglik(resid_sq, omega, alpha, beta, False), abs=1e-8
        )

    @requires_cpp
    def test_empty_input_returns_zero(self):
        result = _cpp.garch11_neg_loglik(
            np.array([], dtype=float), 1e-6, 0.05, 0.9, True
        )
        assert result == 0.0

    def test_python_dispatcher_matches_numpy_fallback(self):
        """_garch11_neg_loglik (the Python-level dispatcher) must produce
        the same result whether or not _sqt_core is built -- the C++ path
        is purely an internal performance detail, same parity contract as
        _garch11_variance_recursion's own dispatcher."""
        import standard_quant_tools.analysis.garch as garch_module

        rng = np.random.default_rng(5)
        resid_sq = (rng.standard_normal(200) * 0.01) ** 2
        params = np.array([1e-6, 0.08, 0.88])

        result_native_path = _garch11_neg_loglik(params, resid_sq, True)

        original_has_cpp = garch_module.HAS_CPP
        garch_module.HAS_CPP = False
        try:
            result_fallback_path = _garch11_neg_loglik(params, resid_sq, True)
        finally:
            garch_module.HAS_CPP = original_has_cpp

        assert result_native_path == pytest.approx(result_fallback_path, abs=1e-8)


def _simulated_garch(n, omega, alpha, beta, seed):
    """GARCH(1,1) returns with normal innovations, after a burn-in."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n + 500)
    variance = omega / (1.0 - alpha - beta)
    values = np.empty(n + 500)
    for t in range(n + 500):
        values[t] = np.sqrt(variance) * shocks[t]
        variance = omega + alpha * values[t] ** 2 + beta * variance
    return pd.Series(values[500:])


class TestGarchForecastEndToEndParity:
    """
    garch_volatility_forecast()'s C++ path (analytic gradient, jac=True) and
    its numba path (scipy's finite differences) reach the SAME optimum.

    This test used to run on iid returns at a 1% tolerance, and passed
    because both paths stopped at their shared starting point: on iid data
    there was nowhere either of them went. On a series with real clustering
    they stopped at different wrong points -- the raw-scale likelihood
    surface was so badly conditioned that the route taken decided where
    the optimizer gave up, and the two gradient sources took different
    routes, about 10 nats apart on one real series. The fit now runs on
    rescaled returns from three starts with tolerances tight enough that
    the gradient decides the stop, and both paths land on the maximum:
    measured to 1.6e-7 in the parameters and 3e-11 nats.
    """

    @requires_cpp
    @pytest.mark.parametrize(
        "true_params, seed",
        [((2e-6, 0.15, 0.80), 2), ((5e-6, 0.25, 0.65), 3), ((1e-6, 0.08, 0.90), 4)],
    )
    def test_forecast_output_matches_with_and_without_cpp(self, true_params, seed):
        import standard_quant_tools.analysis.garch as garch_module

        returns = _simulated_garch(2000, *true_params, seed=seed)
        scale = float(((returns - returns.mean()) ** 2).mean())

        result_cpp = garch_volatility_forecast(returns, forecast_horizon=10)

        garch_module.HAS_CPP = False
        try:
            result_numba = garch_volatility_forecast(returns, forecast_horizon=10)
        finally:
            garch_module.HAS_CPP = True

        # omega compared on the rescaled problem the fit runs on, where all
        # three parameters are O(1) and one tolerance means the same thing.
        assert result_cpp["omega"] / scale == pytest.approx(
            result_numba["omega"] / scale, abs=1e-4
        )
        assert result_cpp["alpha"] == pytest.approx(result_numba["alpha"], abs=1e-4)
        assert result_cpp["beta"] == pytest.approx(result_numba["beta"], abs=1e-4)
        assert result_cpp["log_likelihood"] == pytest.approx(
            result_numba["log_likelihood"], abs=1e-6
        )
        assert result_cpp["forecast_annualized_vol"] == pytest.approx(
            result_numba["forecast_annualized_vol"], rel=1e-4
        )
        assert result_cpp["converged"] is result_numba["converged"] is True
        assert result_cpp["gradient_norm"] < 1e-4
        assert result_numba["gradient_norm"] < 1e-4
