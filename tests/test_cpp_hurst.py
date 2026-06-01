"""
Python integration tests for the Hurst C++ extension (_sqt_core).

These tests run in two modes:
  1. If _sqt_core is NOT built  → the cpp_* tests are skipped; the wrapper
     tests still run and verify the Python fallback path is correct.
  2. If _sqt_core IS built      → all tests run and cross-validate C++ output
     against the Python reference implementation.

Run:
    pytest tests/test_cpp_hurst.py -v
    pytest tests/test_cpp_hurst.py -v -m "not slow"
"""

import math

import numpy as np
import pandas as pd
import pytest

# ── Extension availability ────────────────────────────────────────────────────

from typing import Any

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

requires_cpp = pytest.mark.skipif(not HAS_CPP, reason="_sqt_core not built")

# ── Reference implementations (pure Python) ──────────────────────────────────

from standard_quant_tools.analysis.hurst import (
    _dfa,
    _rs,
    _ols_slope_r2,
    hurst_exponent,
    rolling_hurst,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


@pytest.fixture
def white_noise_500():
    return RNG.standard_normal(500)


@pytest.fixture
def white_noise_1000():
    return RNG.standard_normal(1000)


@pytest.fixture
def series_500(white_noise_500):
    return pd.Series(white_noise_500)


@pytest.fixture
def series_1000(white_noise_1000):
    return pd.Series(white_noise_1000)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Direct binding tests (require _sqt_core)
# ─────────────────────────────────────────────────────────────────────────────

class TestCppBindings:

    @requires_cpp
    def test_hurst_dfa_returns_dict(self, white_noise_500):
        result = _cpp.hurst_dfa(white_noise_500, 10, -1)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"hurst", "regime", "fit_r_squared", "method", "n_obs"}

    @requires_cpp
    def test_hurst_dfa_method_field(self, white_noise_500):
        result = _cpp.hurst_dfa(white_noise_500, 10, -1)
        assert result["method"] == "dfa"

    @requires_cpp
    def test_hurst_rs_method_field(self, white_noise_500):
        result = _cpp.hurst_rs(white_noise_500, 10, -1)
        assert result["method"] == "rs"

    @requires_cpp
    def test_hurst_dfa_n_obs(self, white_noise_500):
        result = _cpp.hurst_dfa(white_noise_500, 10, -1)
        assert result["n_obs"] == 500

    @requires_cpp
    def test_hurst_dfa_returns_nan_for_short_series(self):
        short = np.array([0.1, 0.2, 0.3, 0.4])
        result = _cpp.hurst_dfa(short, 10, -1)
        assert math.isnan(result["hurst"])
        assert result["regime"] == "unknown"

    @requires_cpp
    def test_hurst_dfa_value_in_valid_range(self, white_noise_1000):
        result = _cpp.hurst_dfa(white_noise_1000, 10, -1)
        assert not math.isnan(result["hurst"])
        assert 0.0 <= result["hurst"] <= 1.5

    @requires_cpp
    def test_hurst_r2_in_valid_range(self, white_noise_1000):
        result = _cpp.hurst_dfa(white_noise_1000, 10, -1)
        assert 0.0 <= result["fit_r_squared"] <= 1.0

    @requires_cpp
    def test_rolling_hurst_returns_array(self, white_noise_500):
        out = _cpp.rolling_hurst(white_noise_500, 100, 1, "dfa", 10)
        assert isinstance(out, np.ndarray)
        assert out.shape == (500,)

    @requires_cpp
    def test_rolling_hurst_leading_nans(self, white_noise_500):
        window = 100
        out = _cpp.rolling_hurst(white_noise_500, window, 1, "dfa", 10)
        assert all(math.isnan(v) for v in out[:window - 1])

    @requires_cpp
    def test_rolling_hurst_non_nan_count(self, white_noise_500):
        n, window = 500, 100
        out = _cpp.rolling_hurst(white_noise_500, window, 1, "dfa", 10)
        non_nan = int(np.sum(~np.isnan(out)))
        assert non_nan == n - window + 1

    @requires_cpp
    def test_rolling_hurst_step2_positions(self, white_noise_500):
        n, window, step = 500, 100, 2
        out = _cpp.rolling_hurst(white_noise_500, window, step, "dfa", 10)
        for i in range(n):
            is_expected = (i >= window - 1) and ((i - (window - 1)) % step == 0)
            if is_expected:
                assert not math.isnan(out[i]), f"Expected value at index {i}"
            else:
                assert math.isnan(out[i]), f"Expected NaN at index {i}"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Cross-validation: C++ vs Python reference (require _sqt_core)
# ─────────────────────────────────────────────────────────────────────────────

class TestCppVsPython:
    """Verify that C++ output is numerically identical to the Python reference."""

    ATOL = 1e-8  # absolute tolerance for floating-point comparison

    @requires_cpp
    def test_hurst_dfa_matches_python(self, white_noise_1000):
        arr = white_noise_1000

        # Python reference
        py_sizes, py_flucts = _dfa(arr, 10, len(arr) // 4)
        py_h, py_r2 = _ols_slope_r2(np.log(py_sizes), np.log(py_flucts))

        # C++ result
        cpp = _cpp.hurst_dfa(arr, 10, -1)

        assert abs(cpp["hurst"] - float(np.clip(py_h, 0, 1.5))) < self.ATOL, (
            f"hurst mismatch: C++={cpp['hurst']:.8f}  Python={py_h:.8f}"
        )
        assert abs(cpp["fit_r_squared"] - py_r2) < self.ATOL, (
            f"R² mismatch: C++={cpp['fit_r_squared']:.8f}  Python={py_r2:.8f}"
        )

    @requires_cpp
    def test_hurst_rs_matches_python(self, white_noise_1000):
        arr = white_noise_1000

        py_sizes, py_rs = _rs(arr, 10, len(arr) // 2)
        py_h, py_r2 = _ols_slope_r2(np.log(py_sizes), np.log(py_rs))

        cpp = _cpp.hurst_rs(arr, 10, -1)

        assert abs(cpp["hurst"] - float(np.clip(py_h, 0, 1.5))) < self.ATOL
        assert abs(cpp["fit_r_squared"] - py_r2) < self.ATOL

    @requires_cpp
    @pytest.mark.slow
    def test_rolling_hurst_matches_python(self, white_noise_500):
        """Full rolling comparison — each C++ value must equal the Python value."""
        arr    = white_noise_500
        window = 100

        # Python reference (slow)
        py_series = pd.Series(arr)
        # Temporarily disable C++ so we force Python path
        import standard_quant_tools.analysis.hurst as _hmod
        original_flag = _hmod.HAS_CPP
        _hmod.HAS_CPP = False
        py_out = rolling_hurst(py_series, window=window, step=1, method="dfa").to_numpy()
        _hmod.HAS_CPP = original_flag

        # C++ result
        cpp_out = _cpp.rolling_hurst(arr, window, 1, "dfa", 10)

        for i, (c, p) in enumerate(zip(cpp_out, py_out)):
            if math.isnan(p):
                assert math.isnan(c), f"index {i}: C++ has value where Python has NaN"
            else:
                assert abs(c - p) < self.ATOL, (
                    f"index {i}: C++={c:.8f}  Python={p:.8f}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Python wrapper tests (run regardless of whether _sqt_core is built)
# ─────────────────────────────────────────────────────────────────────────────

class TestPythonWrapper:
    """Tests for the public hurst_exponent / rolling_hurst API.

    These exercise whatever path is active (C++ or Python fallback).
    """

    def test_hurst_exponent_returns_dict(self, series_500):
        result = hurst_exponent(series_500)
        assert isinstance(result, dict)
        assert "hurst" in result
        assert "regime" in result

    def test_hurst_exponent_keys(self, series_500):
        result = hurst_exponent(series_500)
        assert set(result.keys()) == {"hurst", "regime", "fit_r_squared", "method", "n_obs"}

    def test_hurst_exponent_valid_range(self, series_1000):
        result = hurst_exponent(series_1000)
        assert not math.isnan(result["hurst"])
        assert 0.0 <= result["hurst"] <= 1.5

    def test_hurst_exponent_dfa_default(self, series_500):
        result = hurst_exponent(series_500)
        assert result["method"] == "dfa"

    def test_hurst_exponent_rs_method(self, series_500):
        result = hurst_exponent(series_500, method="rs")
        assert result["method"] == "rs"

    def test_hurst_exponent_short_series_returns_nan(self):
        short = pd.Series([0.1, 0.2, 0.3, 0.4])
        result = hurst_exponent(short)
        assert math.isnan(result["hurst"])

    def test_hurst_exponent_with_nan_in_series(self):
        s = pd.Series([np.nan, 0.1, 0.2] * 200)
        result = hurst_exponent(s)
        # Should not raise and should handle NaN gracefully
        assert isinstance(result, dict)

    def test_hurst_exponent_regime_valid_string(self, series_500):
        result = hurst_exponent(series_500)
        assert result["regime"] in {"trending", "random_walk", "mean_reverting", "unknown"}

    def test_hurst_exponent_n_obs(self, series_500):
        result = hurst_exponent(series_500)
        assert result["n_obs"] == 500

    def test_rolling_hurst_returns_series(self, series_500):
        out = rolling_hurst(series_500, window=100)
        assert isinstance(out, pd.Series)

    def test_rolling_hurst_length(self, series_500):
        out = rolling_hurst(series_500, window=100)
        assert len(out) == 500

    def test_rolling_hurst_index_preserved(self, series_500):
        out = rolling_hurst(series_500, window=100)
        assert out.index.equals(series_500.index)

    def test_rolling_hurst_leading_nans(self, series_500):
        window = 100
        out = rolling_hurst(series_500, window=window)
        assert out.iloc[:window - 1].isna().all()

    def test_rolling_hurst_non_nan_values_in_range(self, series_500):
        out = rolling_hurst(series_500, window=100)
        valid = out.dropna()
        assert (valid >= 0.0).all() and (valid <= 1.5).all()

    def test_rolling_hurst_step_reduces_non_nan_count(self, series_1000):
        out_step1 = rolling_hurst(series_1000, window=200, step=1)
        out_step5 = rolling_hurst(series_1000, window=200, step=5)
        assert len(out_step5.dropna()) < len(out_step1.dropna())

    def test_rolling_hurst_rs_method(self, series_500):
        out = rolling_hurst(series_500, window=100, method="rs")
        assert isinstance(out, pd.Series)
        assert out.notna().any()

    @pytest.mark.slow
    def test_rolling_hurst_name_attribute(self, series_500):
        out = rolling_hurst(series_500, window=100)
        assert out.name == "hurst"

    def test_hurst_exponent_reproducible(self, series_500):
        r1 = hurst_exponent(series_500)
        r2 = hurst_exponent(series_500)
        assert r1["hurst"] == r2["hurst"]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Routing test: verify wrapper uses C++ when available
# ─────────────────────────────────────────────────────────────────────────────

class TestRouting:

    @requires_cpp
    def test_wrapper_routes_to_cpp(self, series_500, monkeypatch):
        """hurst_exponent should call _cpp.hurst_dfa when HAS_CPP is True."""
        import standard_quant_tools.analysis.hurst as _hmod

        calls = []
        original = _hmod._cpp.hurst_dfa

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(_hmod._cpp, "hurst_dfa", spy)
        hurst_exponent(series_500, method="dfa")
        assert len(calls) == 1

    @requires_cpp
    def test_wrapper_falls_back_to_python_when_flag_false(self, series_500, monkeypatch):
        """Setting HAS_CPP=False forces the Python path."""
        import standard_quant_tools.analysis.hurst as _hmod
        monkeypatch.setattr(_hmod, "HAS_CPP", False)
        result = hurst_exponent(series_500)
        assert isinstance(result, dict)
        assert not math.isnan(result["hurst"])
