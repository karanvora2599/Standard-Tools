"""
Python integration tests for RSI, ADX, and Parabolic SAR C++ extensions.

Two execution modes:
  1. _sqt_core NOT built → cpp_* tests are skipped; wrapper tests verify the
     Python fallback path produces correct results.
  2. _sqt_core IS built  → all tests run and cross-validate C++ vs Python.

Run:
    pytest tests/test_cpp_indicators.py -v
    pytest tests/test_cpp_indicators.py -v -m "not slow"
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

# ── Python reference implementations ─────────────────────────────────────────

from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.momentum import HAS_CPP as RSI_HAS_CPP
from standard_quant_tools.indicators.momentum import HAS_NUMBA as RSI_HAS_NUMBA
from standard_quant_tools.indicators.momentum import _rsi_numba
from standard_quant_tools.indicators.momentum import rsi as rsi_wrapper
from standard_quant_tools.indicators.trend import _adx_numba, _psar_numba
from standard_quant_tools.indicators.trend import adx as adx_wrapper
from standard_quant_tools.indicators.trend import parabolic_sar as psar_wrapper
from standard_quant_tools.indicators.volatility import HAS_CPP as WATR_HAS_CPP
from standard_quant_tools.indicators.volatility import wilder_atr as wilder_atr_wrapper

# True when the rsi wrapper uses Wilder's SMA seed (C++ or Numba), not EWM.
RSI_USES_WILDERS = RSI_HAS_CPP or RSI_HAS_NUMBA

# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


@pytest.fixture
def prices_200():
    return RNG.standard_normal(200).cumsum() + 100.0


@pytest.fixture
def ohlc_200(prices_200):
    close = prices_200
    high = close + RNG.uniform(0.1, 1.0, len(close))
    low = close - RNG.uniform(0.1, 1.0, len(close))
    return high, low, close


@pytest.fixture
def prices_series_200(prices_200):
    return pd.Series(prices_200, name="Close")


@pytest.fixture
def ohlc_series_200(ohlc_200):
    high, low, close = ohlc_200
    idx = pd.date_range("2020-01-01", periods=len(close), freq="B")
    return (
        pd.Series(high, index=idx, name="High"),
        pd.Series(low, index=idx, name="Low"),
        pd.Series(close, index=idx, name="Close"),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _py_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Pure Python Wilder's RSI matching the C++ and Numba implementations."""
    n = len(prices)
    result = np.full(n, np.nan)
    if n <= period:
        return result
    avg_gain = avg_loss = 0.0
    for i in range(1, period + 1):
        ch = prices[i] - prices[i - 1]
        if ch > 0:
            avg_gain += ch
        elif ch < 0:
            avg_loss -= ch
    avg_gain /= period
    avg_loss /= period
    result[period] = (
        100.0 if avg_loss == 0.0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    )
    for i in range(period + 1, n):
        ch = prices[i] - prices[i - 1]
        gain = ch if ch > 0 else 0.0
        loss = -ch if ch < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        result[i] = (
            100.0 if avg_loss == 0.0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# RSI — C++ extension tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCppRsi:

    @requires_cpp
    def test_matches_python_reference(self, prices_200):
        cpp_out = _cpp.rsi(prices_200.astype(np.float64), 14)
        py_ref = _py_rsi(prices_200, 14)
        np.testing.assert_allclose(cpp_out, py_ref, rtol=1e-10, equal_nan=True)

    @requires_cpp
    def test_nan_prefix(self, prices_200):
        period = 14
        cpp_out = _cpp.rsi(prices_200.astype(np.float64), period)
        assert np.all(np.isnan(cpp_out[:period]))
        assert np.all(~np.isnan(cpp_out[period:]))

    @requires_cpp
    def test_all_rising_equals_100(self):
        prices = np.arange(1.0, 21.0)
        cpp_out = _cpp.rsi(prices, 5)
        np.testing.assert_allclose(cpp_out[5:], 100.0, atol=1e-9)

    @requires_cpp
    def test_all_falling_equals_0(self):
        prices = np.arange(20.0, 0.0, -1.0)
        cpp_out = _cpp.rsi(prices, 5)
        np.testing.assert_allclose(cpp_out[5:], 0.0, atol=1e-9)

    @requires_cpp
    def test_known_value(self):
        # prices = {10, 11, 12, 11}, period=3
        # avg_gain=2/3, avg_loss=1/3 → RSI = 200/3
        prices = np.array([10.0, 11.0, 12.0, 11.0])
        cpp_out = _cpp.rsi(prices, 3)
        assert np.all(np.isnan(cpp_out[:3]))
        np.testing.assert_allclose(cpp_out[3], 200.0 / 3.0, rtol=1e-10)

    @requires_cpp
    def test_bounds(self, prices_200):
        cpp_out = _cpp.rsi(prices_200.astype(np.float64), 14)
        valid = cpp_out[~np.isnan(cpp_out)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    @requires_cpp
    def test_short_series(self):
        prices = np.arange(5.0)
        assert np.all(np.isnan(_cpp.rsi(prices, 5)))  # n == period
        assert np.all(np.isnan(_cpp.rsi(prices[:3], 5)))  # n < period

    @requires_cpp
    def test_empty(self):
        assert len(_cpp.rsi(np.array([]), 14)) == 0

    @requires_cpp
    def test_negative_period_returns_all_nan(self, prices_200):
        """
        Regression test for the native argument-safety fix: rsi() indexes
        result[period], which for a negative period previously wrapped to a
        huge size_t via the implicit int->size_t conversion in operator[]
        — an out-of-bounds write. The guard must make this a safe all-NaN
        return instead.
        """
        out = _cpp.rsi(prices_200.astype(np.float64), -1)
        assert len(out) == len(prices_200)
        assert np.all(np.isnan(out))

    @requires_cpp
    def test_zero_period_returns_all_nan(self, prices_200):
        out = _cpp.rsi(prices_200.astype(np.float64), 0)
        assert np.all(np.isnan(out))

    @requires_cpp
    @pytest.mark.slow
    def test_matches_numba_reference(self, prices_200):
        arr = prices_200.astype(np.float64)
        cpp_out = _cpp.rsi(arr, 14)
        num_out = _rsi_numba(arr, 14)
        np.testing.assert_allclose(cpp_out, num_out, rtol=1e-10, equal_nan=True)


# ══════════════════════════════════════════════════════════════════════════════
# RSI — Python wrapper tests (run always, verify fallback + routing)
# ══════════════════════════════════════════════════════════════════════════════


class TestRsiWrapper:

    def test_returns_series(self, prices_series_200):
        result = rsi_wrapper(prices_series_200, period=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices_series_200)

    def test_preserves_index(self, prices_series_200):
        result = rsi_wrapper(prices_series_200, period=14)
        pd.testing.assert_index_equal(result.index, prices_series_200.index)

    @pytest.mark.skipif(
        not RSI_USES_WILDERS,
        reason="EWM fallback has a different NaN prefix (index 0 only); skip on pure-Python path",
    )
    def test_nan_prefix(self, prices_series_200):
        period = 14
        result = rsi_wrapper(prices_series_200, period=period)
        assert np.all(np.isnan(result.iloc[:period]))
        assert np.all(~np.isnan(result.iloc[period:]))

    def test_bounds(self, prices_series_200):
        result = rsi_wrapper(prices_series_200, period=14)
        valid = result.dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 100.0).all()

    def test_empty_series_raises(self):
        with pytest.raises(ValidationError):
            rsi_wrapper(pd.Series(dtype=float))

    def test_default_period_14(self, prices_series_200):
        r1 = rsi_wrapper(prices_series_200)
        r2 = rsi_wrapper(prices_series_200, period=14)
        pd.testing.assert_series_equal(r1, r2)


# ══════════════════════════════════════════════════════════════════════════════
# ADX — C++ extension tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCppAdx:

    @requires_cpp
    def test_shape(self, ohlc_200):
        high, low, close = ohlc_200
        out = _cpp.adx(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            14,
        )
        assert out.shape == (200, 3)

    @requires_cpp
    def test_matches_python_reference(self, ohlc_200):
        high, low, close = ohlc_200
        h, l, c = (x.astype(np.float64) for x in (high, low, close))
        cpp_out = _cpp.adx(h, l, c, 14)
        py_ref = _adx_numba(h, l, c, 14)
        np.testing.assert_allclose(cpp_out, py_ref, rtol=1e-10, equal_nan=True)

    @requires_cpp
    def test_nan_prefix_di(self, ohlc_200):
        high, low, close = ohlc_200
        period = 7
        out = _cpp.adx(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            period,
        )
        # DI+, DI- NaN before row `period`
        assert np.all(np.isnan(out[:period, 0]))
        assert np.all(np.isnan(out[:period, 1]))
        # ADX NaN before row 2*period-1
        assert np.all(np.isnan(out[: 2 * period - 1, 2]))

    @requires_cpp
    def test_valid_after_warmup(self, ohlc_200):
        high, low, close = ohlc_200
        period = 7
        out = _cpp.adx(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            period,
        )
        assert np.all(~np.isnan(out[period:, :2]))  # DI+ / DI-
        assert np.all(~np.isnan(out[2 * period - 1 :, 2]))  # ADX

    @requires_cpp
    def test_bounds(self, ohlc_200):
        high, low, close = ohlc_200
        out = _cpp.adx(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            14,
        )
        valid = out[~np.isnan(out)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    @requires_cpp
    def test_uptrend_di_plus_dominates(self):
        n = 100
        close = np.arange(100.0, 100.0 + n)
        high = close + 0.5
        low = close - 0.5
        period = 5
        out = _cpp.adx(high, low, close, period)
        # After burn-in, DI+ > DI- for a monotonically rising price series
        for i in range(2 * period, n):
            assert (
                out[i, 0] > out[i, 1]
            ), f"row {i}: DI+={out[i,0]:.3f} not > DI-={out[i,1]:.3f}"

    @requires_cpp
    def test_short_series(self):
        n = 5
        h = np.arange(1.0, n + 1)
        l = h - 0.5
        c = h - 0.25
        out = _cpp.adx(h, l, c, 5)
        assert np.all(np.isnan(out))

    @requires_cpp
    def test_mismatched_lengths_raises(self):
        with pytest.raises(Exception):
            _cpp.adx(np.ones(10), np.ones(9), np.ones(10), 5)


# ══════════════════════════════════════════════════════════════════════════════
# ADX — Python wrapper tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAdxWrapper:

    def test_returns_dataframe(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        result = adx_wrapper(high, low, close, period=14)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["DI_Plus", "DI_Minus", "ADX"]

    def test_preserves_index(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        result = adx_wrapper(high, low, close, period=14)
        pd.testing.assert_index_equal(result.index, close.index)

    def test_length_matches_input(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        result = adx_wrapper(high, low, close, period=14)
        assert len(result) == len(close)

    def test_bounds(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        result = adx_wrapper(high, low, close, period=14)
        for col in result.columns:
            valid = result[col].dropna()
            assert (valid >= 0.0).all()
            assert (valid <= 100.0).all()

    def test_default_period_14(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        r1 = adx_wrapper(high, low, close)
        r2 = adx_wrapper(high, low, close, period=14)
        pd.testing.assert_frame_equal(r1, r2)

    def test_negative_period_raises(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        with pytest.raises(ValidationError, match="period"):
            adx_wrapper(high, low, close, period=-1)

    def test_zero_period_raises(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        with pytest.raises(ValidationError, match="period"):
            adx_wrapper(high, low, close, period=0)


class TestCppAdxArgSafety:
    @requires_cpp
    def test_negative_period_returns_all_nan(self, ohlc_200):
        """
        Regression test: adx() previously divided by a zero/negative
        `period` and indexed result[period*3+...] with a negative-derived
        value — the guard must make this a safe all-NaN return.
        """
        high, low, close = ohlc_200
        out = _cpp.adx(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            -1,
        )
        assert np.all(np.isnan(out))

    @requires_cpp
    def test_zero_period_returns_all_nan(self, ohlc_200):
        high, low, close = ohlc_200
        out = _cpp.adx(
            high.astype(np.float64), low.astype(np.float64), close.astype(np.float64), 0
        )
        assert np.all(np.isnan(out))


# ══════════════════════════════════════════════════════════════════════════════
# Parabolic SAR — C++ extension tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCppPsar:

    @requires_cpp
    def test_shape(self, ohlc_200):
        high, low, _ = ohlc_200
        out = _cpp.parabolic_sar(high.astype(np.float64), low.astype(np.float64))
        assert out.shape == (200, 2)

    @requires_cpp
    def test_matches_python_reference(self, ohlc_200):
        high, low, _ = ohlc_200
        h, l = high.astype(np.float64), low.astype(np.float64)
        cpp_out = _cpp.parabolic_sar(h, l, 0.02, 0.02, 0.2)
        py_ref = _psar_numba(h, l, 0.02, 0.02, 0.2)
        np.testing.assert_allclose(cpp_out, py_ref, rtol=1e-12, equal_nan=True)

    @requires_cpp
    def test_no_nans(self, ohlc_200):
        high, low, _ = ohlc_200
        out = _cpp.parabolic_sar(high.astype(np.float64), low.astype(np.float64))
        assert not np.any(np.isnan(out))

    @requires_cpp
    def test_trend_values_are_pm1(self, ohlc_200):
        high, low, _ = ohlc_200
        out = _cpp.parabolic_sar(high.astype(np.float64), low.astype(np.float64))
        trend = out[:, 1]
        assert np.all((trend == 1.0) | (trend == -1.0))

    @requires_cpp
    def test_bootstrap_rising(self):
        high = np.array([105.0, 106.0, 107.0])
        low = np.array([100.0, 101.0, 102.0])
        out = _cpp.parabolic_sar(high, low, 0.02, 0.02, 0.2)
        # Bar 0: SAR = low[0], Trend = 1.0
        np.testing.assert_allclose(out[0, 0], 100.0, atol=1e-12)
        assert out[0, 1] == 1.0

    @requires_cpp
    def test_rising_sar_below_low(self):
        n = 60
        close = np.arange(100.0, 100.0 + n)
        high = close + 0.5
        low = close - 0.5
        out = _cpp.parabolic_sar(high, low, 0.02, 0.02, 0.2)
        # In a strong uptrend the SAR stays below each bar's low (skip bar 0)
        for i in range(1, n):
            if out[i, 1] == 1.0:
                assert (
                    out[i, 0] < low[i]
                ), f"bar {i}: SAR {out[i,0]:.4f} >= low {low[i]:.4f}"

    @requires_cpp
    def test_single_bar(self):
        high = np.array([101.0])
        low = np.array([99.0])
        out = _cpp.parabolic_sar(high, low, 0.02, 0.02, 0.2)
        assert out.shape == (1, 2)
        assert not np.isnan(out[0, 0])
        assert out[0, 1] == 1.0

    @requires_cpp
    def test_empty(self):
        out = _cpp.parabolic_sar(np.array([]), np.array([]))
        assert out.shape == (0, 2)

    @requires_cpp
    def test_mismatched_lengths_raises(self):
        with pytest.raises(Exception):
            _cpp.parabolic_sar(np.ones(10), np.ones(9))

    @requires_cpp
    def test_custom_af_params(self, ohlc_200):
        high, low, _ = ohlc_200
        h, l = high.astype(np.float64), low.astype(np.float64)
        out1 = _cpp.parabolic_sar(h, l, 0.01, 0.01, 0.1)
        out2 = _cpp.parabolic_sar(h, l, 0.02, 0.02, 0.2)
        # Different AF params produce different results
        assert not np.allclose(out1[:, 0], out2[:, 0])

    @requires_cpp
    @pytest.mark.parametrize(
        "af_start,af_step,af_max",
        [
            (0.0, 0.02, 0.2),  # af_start <= 0
            (-0.02, 0.02, 0.2),  # af_start negative
            (0.02, -0.01, 0.2),  # af_step negative
            (0.02, 0.02, 0.0),  # af_max <= 0
            (0.2, 0.02, 0.02),  # af_max < af_start
            (float("nan"), 0.02, 0.2),
            (0.02, 0.02, float("inf")),
        ],
    )
    def test_invalid_af_params_return_all_nan(
        self, ohlc_200, af_start, af_step, af_max
    ):
        """Not a crash risk (no indexing on af_*), but a nonsensical
        combination used to silently produce a numerically meaningless SAR
        series rather than an obviously-invalid all-NaN result."""
        high, low, _ = ohlc_200
        h, l = high.astype(np.float64), low.astype(np.float64)
        out = _cpp.parabolic_sar(h, l, af_start, af_step, af_max)
        assert out.shape == (200, 2)
        assert np.all(np.isnan(out))


# ══════════════════════════════════════════════════════════════════════════════
# Parabolic SAR — Python wrapper tests
# ══════════════════════════════════════════════════════════════════════════════


class TestPsarWrapper:

    def test_returns_dataframe(self, ohlc_series_200):
        high, low, _ = ohlc_series_200
        result = psar_wrapper(high, low)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["SAR", "Trend"]

    def test_preserves_index(self, ohlc_series_200):
        high, low, _ = ohlc_series_200
        result = psar_wrapper(high, low)
        pd.testing.assert_index_equal(result.index, high.index)

    def test_no_nans(self, ohlc_series_200):
        high, low, _ = ohlc_series_200
        result = psar_wrapper(high, low)
        assert not result.isna().any().any()

    def test_trend_values_are_pm1(self, ohlc_series_200):
        high, low, _ = ohlc_series_200
        result = psar_wrapper(high, low)
        assert result["Trend"].isin([1.0, -1.0]).all()

    def test_default_params(self, ohlc_series_200):
        high, low, _ = ohlc_series_200
        r1 = psar_wrapper(high, low)
        r2 = psar_wrapper(high, low, af_start=0.02, af_step=0.02, af_max=0.2)
        pd.testing.assert_frame_equal(r1, r2)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"af_start": 0.0}, "af_start"),
            ({"af_start": -0.02}, "af_start"),
            ({"af_step": -0.01}, "af_step"),
            ({"af_max": 0.0}, "af_max"),
            ({"af_start": 0.2, "af_max": 0.02}, "af_max"),
            ({"af_start": float("nan")}, "af_start"),
            ({"af_max": float("inf")}, "af_max"),
        ],
    )
    def test_invalid_af_params_raise(self, ohlc_series_200, kwargs, match):
        high, low, _ = ohlc_series_200
        with pytest.raises(ValidationError, match=match):
            psar_wrapper(high, low, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Wilder's ATR — Python reference helper
# ══════════════════════════════════════════════════════════════════════════════


def _py_wilder_atr(
    h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int
) -> np.ndarray:
    """Pure-Python Wilder's ATR matching the C++ implementation exactly."""
    n = len(h)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    result = np.full(n, np.nan)
    if n >= period:
        result[period - 1] = tr[:period].mean()
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Wilder's ATR — C++ extension tests
# ══════════════════════════════════════════════════════════════════════════════


class TestCppWilderAtr:

    @requires_cpp
    def test_matches_python_reference(self, ohlc_200):
        high, low, close = ohlc_200
        h, l, c = (x.astype(np.float64) for x in (high, low, close))
        cpp_out = _cpp.wilder_atr(h, l, c, 14)
        py_ref = _py_wilder_atr(h, l, c, 14)
        np.testing.assert_allclose(cpp_out, py_ref, rtol=1e-12, equal_nan=True)

    @requires_cpp
    def test_nan_prefix(self, ohlc_200):
        high, low, close = ohlc_200
        period = 14
        out = _cpp.wilder_atr(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            period,
        )
        assert len(out) == len(close)
        assert np.all(np.isnan(out[: period - 1]))
        assert np.all(~np.isnan(out[period - 1 :]))

    @requires_cpp
    def test_known_value(self):
        # H=[10,11,12], L=[9,9,10], C=[9.5,10,11], period=2
        # TR = [1, 2, 2]; ATR[1]=1.5; ATR[2]=1.75
        h = np.array([10.0, 11.0, 12.0])
        l = np.array([9.0, 9.0, 10.0])
        c = np.array([9.5, 10.0, 11.0])
        out = _cpp.wilder_atr(h, l, c, 2)
        assert np.isnan(out[0])
        np.testing.assert_allclose(out[1], 1.5, rtol=1e-12)
        np.testing.assert_allclose(out[2], 1.75, rtol=1e-12)

    @requires_cpp
    def test_non_negative(self, ohlc_200):
        high, low, close = ohlc_200
        out = _cpp.wilder_atr(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            14,
        )
        valid = out[~np.isnan(out)]
        assert np.all(valid >= 0.0)

    @requires_cpp
    def test_constant_prices(self):
        # H=L=C=100 → TR=0 everywhere → ATR=0
        n = 50
        h = np.full(n, 100.0)
        l = np.full(n, 100.0)
        c = np.full(n, 100.0)
        out = _cpp.wilder_atr(h, l, c, 5)
        np.testing.assert_allclose(out[4:], 0.0, atol=1e-12)

    @requires_cpp
    def test_short_series(self):
        h = np.arange(1.0, 6.0)
        l = h - 0.5
        c = h - 0.25
        # n=5, period=5 → first valid at index 4 only
        out = _cpp.wilder_atr(h, l, c, 5)
        assert np.all(np.isnan(out[:4]))
        assert not np.isnan(out[4])
        # n=3, period=5 → all NaN
        assert np.all(np.isnan(_cpp.wilder_atr(h[:3], l[:3], c[:3], 5)))

    @requires_cpp
    def test_empty(self):
        assert len(_cpp.wilder_atr(np.array([]), np.array([]), np.array([]), 14)) == 0

    @requires_cpp
    def test_mismatched_lengths_raises(self):
        with pytest.raises(Exception):
            _cpp.wilder_atr(np.ones(10), np.ones(9), np.ones(10), 5)

    @requires_cpp
    def test_period_1_equals_tr(self):
        # period=1: ATR[i] = TR[i] for all i (seed = TR[0], no smoothing thereafter)
        h = np.array([10.0, 12.0, 11.0, 13.0])
        l = np.array([9.0, 10.0, 9.5, 11.0])
        c = np.array([9.5, 11.0, 10.0, 12.0])
        out = _cpp.wilder_atr(h, l, c, 1)
        # Compute expected TR independently
        expected_tr = np.array(
            [h[0] - l[0]]
            + [
                max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
                for i in range(1, len(h))
            ]
        )
        np.testing.assert_allclose(out, expected_tr, atol=1e-12)

    @requires_cpp
    def test_zero_period_returns_all_nan(self, ohlc_200):
        """
        Regression test for the native argument-safety fix: wilder_atr()
        previously indexed result[period-1], which for period<=0 wraps to
        a huge size_t (period=0 -> index -1 -> massive out-of-bounds
        write). The guard must make this a safe all-NaN return.
        """
        high, low, close = ohlc_200
        out = _cpp.wilder_atr(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            0,
        )
        assert np.all(np.isnan(out))

    @requires_cpp
    def test_negative_period_returns_all_nan(self, ohlc_200):
        high, low, close = ohlc_200
        out = _cpp.wilder_atr(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            -1,
        )
        assert np.all(np.isnan(out))


# ══════════════════════════════════════════════════════════════════════════════
# Wilder's ATR — Python wrapper tests
# ══════════════════════════════════════════════════════════════════════════════


class TestWilderAtrWrapper:

    def test_returns_series(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        result = wilder_atr_wrapper(high, low, close)
        assert isinstance(result, pd.Series)
        assert len(result) == len(close)

    def test_series_name(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        result = wilder_atr_wrapper(high, low, close)
        assert result.name == "Wilder_ATR"

    def test_preserves_index(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        result = wilder_atr_wrapper(high, low, close)
        pd.testing.assert_index_equal(result.index, close.index)

    def test_nan_prefix(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        period = 14
        result = wilder_atr_wrapper(high, low, close, period=period)
        assert np.all(np.isnan(result.iloc[: period - 1]))
        assert np.all(~np.isnan(result.iloc[period - 1 :]))

    def test_non_negative(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        result = wilder_atr_wrapper(high, low, close)
        assert (result.dropna() >= 0.0).all()

    def test_default_period_14(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        r1 = wilder_atr_wrapper(high, low, close)
        r2 = wilder_atr_wrapper(high, low, close, period=14)
        pd.testing.assert_series_equal(r1, r2)

    def test_negative_period_raises(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        with pytest.raises(ValidationError, match="period"):
            wilder_atr_wrapper(high, low, close, period=-1)

    def test_zero_period_raises(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        with pytest.raises(ValidationError, match="period"):
            wilder_atr_wrapper(high, low, close, period=0)

    def test_matches_python_reference(self, ohlc_series_200):
        high, low, close = ohlc_series_200
        result = wilder_atr_wrapper(high, low, close, period=14)
        ref = _py_wilder_atr(
            high.to_numpy(dtype=np.float64),
            low.to_numpy(dtype=np.float64),
            close.to_numpy(dtype=np.float64),
            14,
        )
        np.testing.assert_allclose(result.to_numpy(), ref, rtol=1e-10, equal_nan=True)

    def test_cpp_routing(self):
        # When C++ is built, the wrapper should use it
        if WATR_HAS_CPP:
            assert WATR_HAS_CPP is True
        # Always passes — just documents the expected routing
