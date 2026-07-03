"""
C++ extension tests for bollinger_bands and stochastic_oscillator.

Two execution modes:
  1. _sqt_core NOT built → cpp_* tests are skipped; wrapper routing tests
     verify the Python/pandas fallback produces correct results.
  2. _sqt_core IS built  → all tests run; cross-validates C++ vs pandas.

Run:
    pytest tests/test_cpp_new_indicators.py -v
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

from standard_quant_tools.indicators.volatility import (
    bollinger_bands as bb_wrapper,
    HAS_CPP as BB_HAS_CPP,
)
from standard_quant_tools.indicators.momentum import (
    stochastic_oscillator as stoch_wrapper,
    HAS_CPP as STOCH_HAS_CPP,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG   = np.random.default_rng(42)
N     = 300
DATES = pd.date_range("2021-01-01", periods=N, freq="B")


@pytest.fixture(scope="module")
def price_array():
    return 100.0 + np.cumsum(RNG.standard_normal(N))


@pytest.fixture(scope="module")
def price_series(price_array):
    return pd.Series(price_array, index=DATES, name="Close")


@pytest.fixture(scope="module")
def ohlc_arrays(price_array):
    spread = np.abs(RNG.standard_normal(N)) + 0.5
    high   = price_array + spread
    low    = price_array - spread
    return high, low, price_array


@pytest.fixture(scope="module")
def ohlc_series(ohlc_arrays):
    high, low, close = ohlc_arrays
    return (
        pd.Series(high,  index=DATES, name="High"),
        pd.Series(low,   index=DATES, name="Low"),
        pd.Series(close, index=DATES, name="Close"),
    )


# ── Pandas reference implementations ─────────────────────────────────────────

def _py_bollinger(prices: np.ndarray, period: int = 20, num_std: float = 2.0):
    s   = pd.Series(prices)
    sma = s.rolling(period).mean()
    std = s.rolling(period).std()
    return np.column_stack([
        (sma + num_std * std).to_numpy(),
        sma.to_numpy(),
        (sma - num_std * std).to_numpy(),
    ])


def _py_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                   k_period: int = 14, d_period: int = 3):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    ll      = l.rolling(k_period).min()
    hh      = h.rolling(k_period).max()
    K       = 100.0 * (c - ll) / (hh - ll)
    D       = K.rolling(d_period).mean()
    return np.column_stack([K.to_numpy(), D.to_numpy()])


# ══════════════════════════════════════════════════════════════════════════════
# bollinger_bands — C++ extension tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCppBollingerBands:

    @requires_cpp
    def test_output_shape(self, price_array):
        out = _cpp.bollinger_bands(price_array.astype(np.float64), 20, 2.0)
        assert out.shape == (N, 3)

    @requires_cpp
    def test_nan_prefix(self, price_array):
        period = 20
        out    = _cpp.bollinger_bands(price_array.astype(np.float64), period, 2.0)
        # rows 0..period-2 must be NaN; rows period-1.. must be finite
        assert np.all(np.isnan(out[:period - 1, 0]))
        assert np.all(~np.isnan(out[period - 1:, 0]))

    @requires_cpp
    def test_column_ordering_upper_middle_lower(self, price_array):
        out   = _cpp.bollinger_bands(price_array.astype(np.float64), 20, 2.0)
        valid = out[~np.isnan(out[:, 0])]
        assert np.all(valid[:, 0] >= valid[:, 1])   # upper >= middle
        assert np.all(valid[:, 1] >= valid[:, 2])   # middle >= lower

    @requires_cpp
    def test_wider_bands_with_larger_num_std(self, price_array):
        p   = price_array.astype(np.float64)
        out2 = _cpp.bollinger_bands(p, 20, 2.0)
        out3 = _cpp.bollinger_bands(p, 20, 3.0)
        w2   = (out2[:, 0] - out2[:, 2])[~np.isnan(out2[:, 0])]
        w3   = (out3[:, 0] - out3[:, 2])[~np.isnan(out3[:, 0])]
        assert np.all(w3 > w2)

    @requires_cpp
    def test_constant_series_zero_width(self):
        prices = np.full(50, 100.0)
        out    = _cpp.bollinger_bands(prices, 10, 2.0)
        width  = (out[:, 0] - out[:, 2])[~np.isnan(out[:, 0])]
        np.testing.assert_allclose(width, 0.0, atol=1e-10)

    @requires_cpp
    def test_middle_equals_rolling_mean(self, price_array):
        period = 20
        out    = _cpp.bollinger_bands(price_array.astype(np.float64), period, 2.0)
        ref    = pd.Series(price_array).rolling(period).mean().to_numpy()
        np.testing.assert_allclose(out[:, 1], ref, rtol=1e-10, equal_nan=True)

    @requires_cpp
    def test_matches_pandas_reference(self, price_array):
        period = 20
        out    = _cpp.bollinger_bands(price_array.astype(np.float64), period, 2.0)
        ref    = _py_bollinger(price_array, period, 2.0)
        np.testing.assert_allclose(out, ref, rtol=1e-10, equal_nan=True)

    @requires_cpp
    def test_short_series(self):
        prices = np.arange(5.0)
        out    = _cpp.bollinger_bands(prices, 10, 2.0)
        assert out.shape == (5, 3)
        assert np.all(np.isnan(out[:, 0]))

    @requires_cpp
    def test_empty_series(self):
        out = _cpp.bollinger_bands(np.array([]), 20, 2.0)
        assert out.shape[0] == 0

    @requires_cpp
    def test_custom_period(self, price_array):
        p  = price_array.astype(np.float64)
        o10 = _cpp.bollinger_bands(p, 10, 2.0)
        o20 = _cpp.bollinger_bands(p, 20, 2.0)
        # NaN prefix is shorter with period=10
        assert np.sum(np.isnan(o10[:, 0])) < np.sum(np.isnan(o20[:, 0]))


# ══════════════════════════════════════════════════════════════════════════════
# bollinger_bands — Python wrapper routing tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBollingerBandsWrapper:

    def test_cpp_and_pandas_paths_agree(self, price_series):
        """C++ and pandas fallback must produce numerically identical DataFrames."""
        if not BB_HAS_CPP:
            pytest.skip("_sqt_core not built")
        from unittest.mock import patch
        from standard_quant_tools.indicators import volatility as vol_mod

        cpp_result = bb_wrapper(price_series, period=20, num_std=2.0)

        with patch.object(vol_mod, "HAS_CPP", False), \
             patch.object(vol_mod, "_cpp_core", None):
            py_result = bb_wrapper(price_series, period=20, num_std=2.0)

        pd.testing.assert_frame_equal(cpp_result, py_result, rtol=1e-10, check_names=True)

    def test_wrapper_upper_gt_middle_gt_lower(self, price_series):
        result = bb_wrapper(price_series, period=20, num_std=2.0).dropna()
        assert (result["BB_Upper"] >= result["BB_Middle"]).all()
        assert (result["BB_Middle"] >= result["BB_Lower"]).all()

    def test_wrapper_preserves_index(self, price_series):
        result = bb_wrapper(price_series)
        pd.testing.assert_index_equal(result.index, DATES)

    def test_wrapper_columns(self, price_series):
        result = bb_wrapper(price_series)
        assert list(result.columns) == ["BB_Upper", "BB_Middle", "BB_Lower"]

    def test_wrapper_nan_prefix(self, price_series):
        period = 20
        result = bb_wrapper(price_series, period=period)
        assert result.iloc[:period - 1].isna().all(axis=None)

    def test_wrapper_middle_equals_sma(self, price_series):
        period = 20
        result = bb_wrapper(price_series, period=period)
        sma    = price_series.rolling(period).mean()
        pd.testing.assert_series_equal(
            result["BB_Middle"].dropna(), sma.dropna(), check_names=False, rtol=1e-10
        )


# ══════════════════════════════════════════════════════════════════════════════
# stochastic_oscillator — C++ extension tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCppStochasticOscillator:

    @requires_cpp
    def test_output_shape(self, ohlc_arrays):
        h, l, c = (x.astype(np.float64) for x in ohlc_arrays)
        out = _cpp.stochastic_oscillator(h, l, c, 14, 3)
        assert out.shape == (N, 2)

    @requires_cpp
    def test_nan_prefix_k(self, ohlc_arrays):
        h, l, c   = (x.astype(np.float64) for x in ohlc_arrays)
        k_period  = 14
        out       = _cpp.stochastic_oscillator(h, l, c, k_period, 3)
        # K: NaN for first k_period-1 bars
        assert np.all(np.isnan(out[:k_period - 1, 0]))
        assert np.all(~np.isnan(out[k_period - 1:, 0]))

    @requires_cpp
    def test_nan_prefix_d(self, ohlc_arrays):
        h, l, c  = (x.astype(np.float64) for x in ohlc_arrays)
        k_period = 14
        d_period = 3
        out      = _cpp.stochastic_oscillator(h, l, c, k_period, d_period)
        nan_d    = k_period + d_period - 2
        assert np.all(np.isnan(out[:nan_d, 1]))
        assert np.all(~np.isnan(out[nan_d:, 1]))

    @requires_cpp
    def test_k_bounds_0_to_100(self, ohlc_arrays):
        h, l, c = (x.astype(np.float64) for x in ohlc_arrays)
        out     = _cpp.stochastic_oscillator(h, l, c, 14, 3)
        k_valid = out[:, 0][~np.isnan(out[:, 0])]
        assert np.all(k_valid >= 0.0)
        assert np.all(k_valid <= 100.0)

    @requires_cpp
    def test_d_bounds_0_to_100(self, ohlc_arrays):
        h, l, c = (x.astype(np.float64) for x in ohlc_arrays)
        out     = _cpp.stochastic_oscillator(h, l, c, 14, 3)
        d_valid = out[:, 1][~np.isnan(out[:, 1])]
        assert np.all(d_valid >= 0.0)
        assert np.all(d_valid <= 100.0)

    @requires_cpp
    def test_close_at_high_yields_k_100(self):
        n     = 30
        high  = np.full(n, 10.0)
        low   = np.full(n, 5.0)
        close = np.full(n, 10.0)
        out   = _cpp.stochastic_oscillator(high, low, close, 5, 3)
        k_valid = out[:, 0][~np.isnan(out[:, 0])]
        np.testing.assert_allclose(k_valid, 100.0, atol=1e-10)

    @requires_cpp
    def test_close_at_low_yields_k_0(self):
        n     = 30
        high  = np.full(n, 10.0)
        low   = np.full(n, 5.0)
        close = np.full(n, 5.0)
        out   = _cpp.stochastic_oscillator(high, low, close, 5, 3)
        k_valid = out[:, 0][~np.isnan(out[:, 0])]
        np.testing.assert_allclose(k_valid, 0.0, atol=1e-10)

    @requires_cpp
    def test_d_is_sma_of_k(self, ohlc_arrays):
        """%D must equal SMA(d_period) of %K."""
        h, l, c  = (x.astype(np.float64) for x in ohlc_arrays)
        k_period = 14
        d_period = 3
        out      = _cpp.stochastic_oscillator(h, l, c, k_period, d_period)
        k_series = pd.Series(out[:, 0])
        expected_d = k_series.rolling(d_period).mean().to_numpy()
        np.testing.assert_allclose(out[:, 1], expected_d, atol=1e-10, equal_nan=True)

    @requires_cpp
    def test_matches_pandas_reference(self, ohlc_arrays):
        h, l, c = (x.astype(np.float64) for x in ohlc_arrays)
        cpp_out = _cpp.stochastic_oscillator(h, l, c, 14, 3)
        py_ref  = _py_stochastic(h, l, c, 14, 3)
        np.testing.assert_allclose(cpp_out, py_ref, atol=1e-10, equal_nan=True)

    @requires_cpp
    def test_mismatched_lengths_raises(self, ohlc_arrays):
        h, l, c = (x.astype(np.float64) for x in ohlc_arrays)
        with pytest.raises(Exception):
            _cpp.stochastic_oscillator(h[:-1], l, c, 14, 3)

    @requires_cpp
    def test_empty(self):
        out = _cpp.stochastic_oscillator(
            np.array([]), np.array([]), np.array([]), 14, 3
        )
        assert out.shape[0] == 0

    @requires_cpp
    def test_short_series_all_nan(self):
        n     = 5
        high  = np.arange(1.0, n + 1)
        low   = high - 0.5
        close = high - 0.25
        out   = _cpp.stochastic_oscillator(high, low, close, 10, 3)
        assert np.all(np.isnan(out[:, 0]))

    @requires_cpp
    def test_custom_periods(self, ohlc_arrays):
        h, l, c = (x.astype(np.float64) for x in ohlc_arrays)
        out1 = _cpp.stochastic_oscillator(h, l, c, 5,  2)
        out2 = _cpp.stochastic_oscillator(h, l, c, 14, 3)
        # Shorter period → fewer NaN rows
        assert np.sum(np.isnan(out1[:, 0])) < np.sum(np.isnan(out2[:, 0]))


# ══════════════════════════════════════════════════════════════════════════════
# stochastic_oscillator — Python wrapper routing tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStochasticOscillatorWrapper:

    def test_cpp_and_pandas_paths_agree(self, ohlc_series):
        """C++ and pandas fallback must produce numerically identical DataFrames."""
        if not STOCH_HAS_CPP:
            pytest.skip("_sqt_core not built")
        from unittest.mock import patch
        from standard_quant_tools.indicators import momentum as mom_mod

        h_s, l_s, c_s = ohlc_series
        cpp_result     = stoch_wrapper(h_s, l_s, c_s, k_period=14, d_period=3)

        with patch.object(mom_mod, "HAS_CPP", False), \
             patch.object(mom_mod, "_cpp_core", None):
            py_result = stoch_wrapper(h_s, l_s, c_s, k_period=14, d_period=3)

        pd.testing.assert_frame_equal(cpp_result, py_result, atol=1e-10, check_names=True)

    def test_wrapper_columns(self, ohlc_series):
        h_s, l_s, c_s = ohlc_series
        result = stoch_wrapper(h_s, l_s, c_s)
        assert list(result.columns) == ["Stoch_K", "Stoch_D"]

    def test_wrapper_preserves_index(self, ohlc_series):
        h_s, l_s, c_s = ohlc_series
        result = stoch_wrapper(h_s, l_s, c_s)
        pd.testing.assert_index_equal(result.index, DATES)

    def test_wrapper_k_bounded(self, ohlc_series):
        h_s, l_s, c_s = ohlc_series
        result = stoch_wrapper(h_s, l_s, c_s)
        k      = result["Stoch_K"].dropna()
        assert (k >= 0.0).all() and (k <= 100.0).all()

    def test_wrapper_d_is_sma_of_k(self, ohlc_series):
        h_s, l_s, c_s = ohlc_series
        d_period       = 3
        result         = stoch_wrapper(h_s, l_s, c_s, d_period=d_period)
        expected_d     = result["Stoch_K"].rolling(d_period).mean()
        pd.testing.assert_series_equal(
            result["Stoch_D"].dropna(), expected_d.dropna(),
            check_names=False, atol=1e-10,
        )

    def test_wrapper_close_at_high_k_100(self):
        n   = 30
        idx = pd.RangeIndex(n)
        h   = pd.Series([10.0] * n, index=idx)
        l   = pd.Series([ 5.0] * n, index=idx)
        c   = pd.Series([10.0] * n, index=idx)
        result = stoch_wrapper(h, l, c, k_period=5, d_period=3)
        np.testing.assert_allclose(result["Stoch_K"].dropna().to_numpy(), 100.0, atol=1e-10)
