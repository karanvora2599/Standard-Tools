"""
C++ extension tests for rolling_beta and rolling_factor_loadings.

Two execution modes:
  1. _sqt_core NOT built → cpp_* tests are skipped; wrapper tests verify the
     Python fallback produces correct results.
  2. _sqt_core IS built  → all tests run; cross-validates C++ vs Python.

Run:
    pytest tests/test_cpp_regression.py -v
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

# ── Extension availability ────────────────────────────────────────────────────

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

requires_cpp = pytest.mark.skipif(not HAS_CPP, reason="_sqt_core not built")

from standard_quant_tools.analysis.regression import (
    rolling_beta as rolling_beta_wrapper,
    HAS_CPP as REGRESSION_HAS_CPP,
)
from standard_quant_tools.analysis.multi_factor import (
    rolling_factor_loadings as rfl_wrapper,
    HAS_CPP as MULTI_FACTOR_HAS_CPP,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
N   = 250
DATES = pd.date_range("2021-01-01", periods=N, freq="B")


@pytest.fixture(scope="module")
def yx_arrays():
    """y = 1.5*x + noise; true beta ≈ 1.5."""
    x = RNG.standard_normal(N) * 0.01
    y = 1.5 * x + RNG.standard_normal(N) * 0.002
    return y, x


@pytest.fixture(scope="module")
def yx_series(yx_arrays):
    y, x = yx_arrays
    return (
        pd.Series(y, index=DATES, name="asset"),
        pd.Series(x, index=DATES, name="bench"),
    )


@pytest.fixture(scope="module")
def factor_data():
    """y = 0.0005 + 1.2*f1 + 0.4*f2 + noise."""
    f1    = RNG.standard_normal(N) * 0.01
    f2    = RNG.standard_normal(N) * 0.005
    noise = RNG.standard_normal(N) * 0.002
    y     = 0.0005 + 1.2 * f1 + 0.4 * f2 + noise
    return (
        pd.Series(y,  index=DATES, name="asset"),
        pd.DataFrame({"mkt": f1, "smb": f2}, index=DATES),
    )


# ── Python reference for rolling_beta ─────────────────────────────────────────

def _py_rolling_beta(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """pandas cov/var rolling beta (the Python fallback path)."""
    sy = pd.Series(y)
    sx = pd.Series(x)
    return (sy.rolling(window).cov(sx) / sx.rolling(window).var()).to_numpy()


# ── Python reference for rolling_factor_loadings ──────────────────────────────

def _py_rfl(y: np.ndarray, X: np.ndarray, window: int) -> np.ndarray:
    """numpy lstsq rolling OLS (the Python fallback path)."""
    n, k = len(y), X.shape[1]
    out  = np.full((n, k + 1), np.nan)
    for i in range(window - 1, n):
        y_w   = y[i - window + 1: i + 1]
        X_w   = X[i - window + 1: i + 1]
        X_des = np.column_stack([np.ones(window), X_w])
        beta, *_ = np.linalg.lstsq(X_des, y_w, rcond=None)
        out[i] = beta
    return out


# ══════════════════════════════════════════════════════════════════════════════
# rolling_beta — C++ extension tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCppRollingBeta:

    @requires_cpp
    def test_output_length(self, yx_arrays):
        y, x = yx_arrays
        out  = _cpp.rolling_beta(y.astype(np.float64), x.astype(np.float64), 60)
        assert len(out) == N

    @requires_cpp
    def test_nan_prefix(self, yx_arrays):
        y, x   = yx_arrays
        window = 40
        out    = _cpp.rolling_beta(y.astype(np.float64), x.astype(np.float64), window)
        assert np.all(np.isnan(out[:window - 1]))
        assert np.all(~np.isnan(out[window - 1:]))

    @requires_cpp
    def test_known_beta_2(self):
        """y = 2*x exactly → every window's beta must be exactly 2."""
        n  = 100
        x  = np.random.default_rng(7).standard_normal(n)
        y  = 2.0 * x
        out = _cpp.rolling_beta(y, x, 30)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 2.0, atol=1e-9)

    @requires_cpp
    def test_negative_beta(self):
        """y = -x → beta = -1."""
        n  = 80
        x  = np.random.default_rng(3).standard_normal(n)
        y  = -x
        out = _cpp.rolling_beta(y, x, 20)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, -1.0, atol=1e-9)

    @requires_cpp
    def test_constant_x_returns_nan(self):
        """Var(x)=0 → denominator is zero → NaN."""
        n  = 50
        x  = np.ones(n)
        y  = np.random.default_rng(5).standard_normal(n)
        out = _cpp.rolling_beta(y, x, 20)
        assert np.all(np.isnan(out[19:]))

    @requires_cpp
    def test_matches_pandas_fallback(self, yx_arrays):
        y, x   = yx_arrays
        window = 60
        cpp_out = _cpp.rolling_beta(y.astype(np.float64), x.astype(np.float64), window)
        py_ref  = _py_rolling_beta(y, x, window)
        np.testing.assert_allclose(cpp_out, py_ref, rtol=1e-8, equal_nan=True)

    @requires_cpp
    def test_mismatched_lengths_raises(self):
        with pytest.raises(Exception):
            _cpp.rolling_beta(np.ones(10), np.ones(9), 5)

    @requires_cpp
    def test_window_larger_than_n_all_nan(self):
        y = np.random.default_rng(0).standard_normal(10)
        x = np.random.default_rng(1).standard_normal(10)
        out = _cpp.rolling_beta(y, x, 20)
        assert np.all(np.isnan(out))

    @requires_cpp
    def test_recovers_true_beta_in_expectation(self, yx_arrays):
        """Mean of rolling betas should approximate the true beta (≈1.5)."""
        y, x  = yx_arrays
        out   = _cpp.rolling_beta(y.astype(np.float64), x.astype(np.float64), 60)
        valid = out[~np.isnan(out)]
        assert valid.mean() == pytest.approx(1.5, abs=0.15)


# ══════════════════════════════════════════════════════════════════════════════
# rolling_beta — Python wrapper tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRollingBetaWrapper:

    def test_returns_dataframe(self, yx_series):
        y_s, x_s = yx_series
        result = rolling_beta_wrapper(y_s, x_s, window=60)
        assert isinstance(result, pd.DataFrame)
        assert "Rolling_Beta" in result.columns

    def test_preserves_index(self, yx_series):
        y_s, x_s = yx_series
        result = rolling_beta_wrapper(y_s, x_s, window=60)
        pd.testing.assert_index_equal(result.index, DATES)

    def test_length_matches_input(self, yx_series):
        y_s, x_s = yx_series
        result = rolling_beta_wrapper(y_s, x_s, window=60)
        assert len(result) == N

    def test_nan_prefix(self, yx_series):
        y_s, x_s = yx_series
        window   = 60
        result   = rolling_beta_wrapper(y_s, x_s, window=window)
        assert result["Rolling_Beta"].iloc[:window - 1].isna().all()
        assert result["Rolling_Beta"].iloc[window - 1:].notna().all()

    def test_known_beta_2_wrapper(self):
        """y = 2*x → all valid betas must be ≈ 2."""
        rng = np.random.default_rng(99)
        x   = pd.Series(rng.standard_normal(120) * 0.01, index=DATES[:120])
        y   = 2.0 * x
        result = rolling_beta_wrapper(y, x, window=30)
        valid  = result["Rolling_Beta"].dropna()
        assert (valid - 2.0).abs().max() < 1e-6

    def test_cpp_and_pandas_paths_agree(self, yx_series):
        """C++ and pandas fallback should produce numerically identical results."""
        if not REGRESSION_HAS_CPP:
            pytest.skip("_sqt_core not built")
        from unittest.mock import patch
        from standard_quant_tools.analysis import regression as reg_mod

        y_s, x_s = yx_series
        window   = 60

        cpp_result = rolling_beta_wrapper(y_s, x_s, window=window)

        with patch.object(reg_mod, "HAS_CPP", False), \
             patch.object(reg_mod, "_cpp_core", None):
            py_result = rolling_beta_wrapper(y_s, x_s, window=window)

        np.testing.assert_allclose(
            cpp_result["Rolling_Beta"].to_numpy(),
            py_result["Rolling_Beta"].to_numpy(),
            rtol=1e-8, equal_nan=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# rolling_factor_loadings — C++ extension tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCppRollingFactorLoadings:

    @requires_cpp
    def test_output_shape(self, factor_data):
        asset, factors = factor_data
        y   = asset.to_numpy(dtype=np.float64)
        X   = np.ascontiguousarray(factors.to_numpy(dtype=np.float64))
        out = _cpp.rolling_factor_loadings(y, X, 60)
        assert out.shape == (N, 3)  # alpha + 2 factors

    @requires_cpp
    def test_nan_prefix(self, factor_data):
        asset, factors = factor_data
        y   = asset.to_numpy(dtype=np.float64)
        X   = np.ascontiguousarray(factors.to_numpy(dtype=np.float64))
        window = 60
        out = _cpp.rolling_factor_loadings(y, X, window)
        assert np.all(np.isnan(out[:window - 1, 0]))
        assert np.all(~np.isnan(out[window - 1:, 0]))

    @requires_cpp
    def test_matches_python_lstsq(self, factor_data):
        asset, factors = factor_data
        y      = asset.to_numpy(dtype=np.float64)
        X      = np.ascontiguousarray(factors.to_numpy(dtype=np.float64))
        window = 60
        cpp_out = _cpp.rolling_factor_loadings(y, X, window)
        py_ref  = _py_rfl(y, X, window)
        np.testing.assert_allclose(cpp_out, py_ref, rtol=1e-8, equal_nan=True)

    @requires_cpp
    def test_perfect_fit_recovers_exact_coefficients(self):
        """y = 3*f1 + 0.5*f2 (no noise, no intercept) — loadings must be exact."""
        rng = np.random.default_rng(11)
        n   = 100
        f1  = rng.standard_normal(n)
        f2  = rng.standard_normal(n)
        y   = 3.0 * f1 + 0.5 * f2
        X   = np.ascontiguousarray(np.column_stack([f1, f2]))
        out = _cpp.rolling_factor_loadings(y, X, 40)
        valid = out[~np.isnan(out[:, 0])]
        np.testing.assert_allclose(valid[:, 1], 3.0,  atol=1e-8)
        np.testing.assert_allclose(valid[:, 2], 0.5,  atol=1e-8)
        np.testing.assert_allclose(valid[:, 0], 0.0,  atol=1e-8)  # intercept ≈ 0

    @requires_cpp
    def test_single_factor(self):
        """k=1 → shape must be (n, 2): [alpha, loading]."""
        rng = np.random.default_rng(22)
        n   = 80
        x   = rng.standard_normal(n)
        y   = 2.0 * x
        X   = np.ascontiguousarray(x.reshape(-1, 1))
        out = _cpp.rolling_factor_loadings(y, X, 30)
        assert out.shape == (n, 2)
        valid = out[~np.isnan(out[:, 0])]
        np.testing.assert_allclose(valid[:, 1], 2.0, atol=1e-8)

    @requires_cpp
    def test_singular_window_produces_nan(self):
        """k=2 factors that are identical → singular XtX → NaN."""
        rng = np.random.default_rng(33)
        n   = 80
        f   = rng.standard_normal(n)
        y   = rng.standard_normal(n)
        X   = np.ascontiguousarray(np.column_stack([f, f]))
        out = _cpp.rolling_factor_loadings(y, X, 20)
        valid_rows = out[19:, 0]  # after warmup
        assert np.all(np.isnan(valid_rows))

    @requires_cpp
    def test_window_larger_than_n_all_nan(self):
        rng = np.random.default_rng(0)
        n   = 20
        y   = rng.standard_normal(n)
        X   = np.ascontiguousarray(rng.standard_normal((n, 2)))
        out = _cpp.rolling_factor_loadings(y, X, 50)
        assert np.all(np.isnan(out))


# ══════════════════════════════════════════════════════════════════════════════
# rolling_factor_loadings — Python wrapper tests (C++ routing)
# ══════════════════════════════════════════════════════════════════════════════

class TestRollingFactorLoadingsWrapper:

    def test_cpp_and_python_paths_agree(self, factor_data):
        """C++ Cholesky path and numpy lstsq must agree to floating-point tolerance."""
        if not MULTI_FACTOR_HAS_CPP:
            pytest.skip("_sqt_core not built")
        from unittest.mock import patch
        from standard_quant_tools.analysis import multi_factor as mf_mod

        asset, factors = factor_data
        window         = 60

        cpp_result = rfl_wrapper(asset, factors, window=window)

        with patch.object(mf_mod, "HAS_CPP", False), \
             patch.object(mf_mod, "_cpp_core", None):
            py_result = rfl_wrapper(asset, factors, window=window)

        assert list(cpp_result.columns) == list(py_result.columns)
        np.testing.assert_allclose(
            cpp_result.to_numpy(),
            py_result.to_numpy(),
            rtol=1e-7, equal_nan=True,
        )

    def test_wrapper_columns(self, factor_data):
        asset, factors = factor_data
        result = rfl_wrapper(asset, factors, window=60)
        assert list(result.columns) == ["alpha", "mkt", "smb"]

    def test_wrapper_preserves_index(self, factor_data):
        asset, factors = factor_data
        result = rfl_wrapper(asset, factors, window=60)
        pd.testing.assert_index_equal(result.index, DATES)
