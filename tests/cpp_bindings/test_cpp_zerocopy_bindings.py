"""
Coverage for the strict/zero-copy `_zerocopy` binding siblings
(correctness/portability pass item 18): `rolling_beta_zerocopy`,
`rolling_factor_loadings_zerocopy`, `simulate_forward_paths_zerocopy`,
`batch_run_strategy_zerocopy`, `technical_indicators_zerocopy`,
`rolling_hurst_zerocopy`.

Each takes an untyped `py::array` and manually validates dtype/C-contiguity
via `require_strict_f64_1d`/`require_strict_f64_2d` -- raising a clear
`ValueError` instead of pybind11's own generic "incompatible function
arguments" message on a mismatch -- then casts without `forcecast`, so a
correctly-typed input is used in place with zero copy. Existing default
bindings (with `forcecast`) are unchanged; these are purely additive.

Run:
    pytest tests/test_cpp_zerocopy_bindings.py -v
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


@pytest.fixture
def rng():
    return np.random.default_rng(11)


class TestRollingBetaZerocopy:
    @requires_cpp
    def test_matches_non_strict_exactly(self, rng):
        y = rng.standard_normal(150).astype(np.float64)
        x = rng.standard_normal(150).astype(np.float64)
        np.testing.assert_array_equal(
            _cpp.rolling_beta_zerocopy(y, x, 20), _cpp.rolling_beta(y, x, 20)
        )

    @requires_cpp
    def test_wrong_dtype_raises(self, rng):
        y = rng.standard_normal(150).astype(np.float32)
        x = rng.standard_normal(150).astype(np.float64)
        with pytest.raises(ValueError, match="C-contiguous float64"):
            _cpp.rolling_beta_zerocopy(y, x, 20)

    @requires_cpp
    def test_non_contiguous_raises(self, rng):
        y = rng.standard_normal(300).astype(np.float64)[::2]
        x = rng.standard_normal(150).astype(np.float64)
        with pytest.raises(ValueError, match="C-contiguous float64"):
            _cpp.rolling_beta_zerocopy(y, x, 20)


class TestRollingFactorLoadingsZerocopy:
    @requires_cpp
    def test_matches_non_strict_exactly(self, rng):
        y = rng.standard_normal(150).astype(np.float64)
        factors = rng.standard_normal((150, 3)).astype(np.float64)
        np.testing.assert_array_equal(
            _cpp.rolling_factor_loadings_zerocopy(y, factors, 20),
            _cpp.rolling_factor_loadings(y, factors, 20),
        )

    @requires_cpp
    def test_1d_factors_raises(self, rng):
        y = rng.standard_normal(150).astype(np.float64)
        factors = rng.standard_normal(150).astype(np.float64)
        with pytest.raises(ValueError, match="2-D"):
            _cpp.rolling_factor_loadings_zerocopy(y, factors, 20)


class TestSimulateForwardPathsZerocopy:
    @requires_cpp
    def test_matches_non_strict_exactly(self, rng):
        values = rng.standard_normal(200).astype(np.float64) * 0.01
        a = _cpp.simulate_forward_paths_zerocopy(values, 30, 40, 10, 10_000.0, 5)
        b = _cpp.simulate_forward_paths(values, 30, 40, 10, 10_000.0, 5)
        np.testing.assert_array_equal(a, b)

    @requires_cpp
    def test_wrong_dtype_raises(self, rng):
        values = rng.standard_normal(200).astype(np.int64)
        with pytest.raises(ValueError, match="C-contiguous float64"):
            _cpp.simulate_forward_paths_zerocopy(values, 30, 40, 10, 10_000.0, 5)


class TestBatchRunStrategyZerocopy:
    @requires_cpp
    def test_matches_non_strict_exactly(self, rng):
        prices = (rng.standard_normal(150).cumsum() + 100).astype(np.float64)
        signals = np.sign(rng.standard_normal((4, 150))).astype(np.float64)
        a = _cpp.batch_run_strategy_zerocopy(prices, signals)
        b = _cpp.batch_run_strategy(prices, signals)
        np.testing.assert_array_equal(a, b)

    @requires_cpp
    def test_1d_signals_raises(self, rng):
        prices = (rng.standard_normal(150).cumsum() + 100).astype(np.float64)
        signals = np.sign(rng.standard_normal(150)).astype(np.float64)
        with pytest.raises(ValueError, match="2-D"):
            _cpp.batch_run_strategy_zerocopy(prices, signals)


class TestTechnicalIndicatorsZerocopy:
    @requires_cpp
    def test_matches_non_strict_exactly(self, rng):
        high = (rng.standard_normal(150).cumsum() + 100).astype(np.float64)
        low = high - 1.0
        close = (high + low) / 2.0
        a = _cpp.technical_indicators_zerocopy(
            high, low, close, compute_rsi=True, compute_adx=True
        )
        b = _cpp.technical_indicators(
            high, low, close, compute_rsi=True, compute_adx=True
        )
        np.testing.assert_array_equal(a["rsi"], b["rsi"])
        np.testing.assert_array_equal(a["adx"], b["adx"])

    @requires_cpp
    def test_wrong_dtype_raises(self, rng):
        high = (rng.standard_normal(150).cumsum() + 100).astype(np.float32)
        low = high.astype(np.float64) - 1.0
        close = (high.astype(np.float64) + low) / 2.0
        with pytest.raises(ValueError, match="C-contiguous float64"):
            _cpp.technical_indicators_zerocopy(high, low, close, compute_rsi=True)


class TestRollingHurstZerocopy:
    @requires_cpp
    def test_matches_non_strict_exactly(self, rng):
        arr = rng.standard_normal(200).cumsum().astype(np.float64)
        a = _cpp.rolling_hurst_zerocopy(arr, 60, 5, "dfa", 10)
        b = _cpp.rolling_hurst(arr, 60, 5, "dfa", 10)
        np.testing.assert_array_equal(a, b)

    @requires_cpp
    def test_non_contiguous_raises(self, rng):
        arr = rng.standard_normal(400).cumsum().astype(np.float64)[::2]
        with pytest.raises(ValueError, match="C-contiguous float64"):
            _cpp.rolling_hurst_zerocopy(arr, 60, 5, "dfa", 10)
