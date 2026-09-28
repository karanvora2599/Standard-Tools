"""
A missing bar reads the same on both backends, at every door.

The rule (indicators/_missing.py): NaN is a missing bar. RSI, Wilder's ATR
and ADX skip it -- the indicator of the series with that bar dropped,
reported back at the bars that remain, NaN at the bar itself. Bollinger
Bands and the stochastic %K/%D are NaN over the windows that hold it and
resume after. An infinity is refused at every library function; a direct
kernel caller sees it treated as missing.

What this replaced, measured on the previous build with one NaN close:

    rsi, NaN in the seed window    native NaN to the end    Numba recovered
    rsi, NaN in the forward pass   both read it as a zero change
    adx, NaN high                  native recovered          Numba NaN forever
    panel with a NaN, no extension refused                  native answered

Every assertion compares the two backends against EACH OTHER, since a test
pinning one side cannot see them disagree.
"""

from typing import Any, List

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators import momentum, panel, trend, volatility
from standard_quant_tools.indicators.momentum import _rsi_numba
from standard_quant_tools.indicators.panel import technical_indicators_panel
from standard_quant_tools.indicators.trend import _adx_numba
from standard_quant_tools.indicators.volatility import _wilder_atr_kernel

_cpp: Any = pytest.importorskip(
    "standard_quant_tools._sqt_core", reason="native extension not built"
)

N = 160
GAPS: List[List[int]] = [
    [],  # the null case: no gap at all
    [0],  # the very first bar
    [5],  # inside every seed window
    [40, 41, 42],  # a run of missing bars
    [70, 120],
    [N - 1],  # the last bar
]


def _ohlc(seed: int = 7):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, N))
    high = close + rng.uniform(0.1, 1.2, N)
    low = close - rng.uniform(0.1, 1.2, N)
    return high, low, close


def _with_gaps(values: np.ndarray, gaps: List[int], fill: float = np.nan):
    out = values.copy()
    out[gaps] = fill
    return out


def _assert_same(native: np.ndarray, fallback: np.ndarray) -> None:
    native, fallback = np.asarray(native), np.asarray(fallback)
    np.testing.assert_array_equal(np.isnan(native), np.isnan(fallback))
    np.testing.assert_allclose(native, fallback, rtol=1e-12, atol=0.0, equal_nan=True)


def _fallbacks(fn):
    """The Numba-compiled kernel and, when Numba is present, its plain
    Python body -- the tier a machine with neither extension runs."""
    kernels = [fn]
    if hasattr(fn, "py_func"):
        kernels.append(fn.py_func)
    return kernels


# ── The kernels, native against fallback ────────────────────────────────────


class TestKernelsAgreeOnGaps:
    @pytest.mark.parametrize("gaps", GAPS)
    def test_rsi(self, gaps):
        close = _with_gaps(_ohlc()[2], gaps)
        for kernel in _fallbacks(_rsi_numba):
            _assert_same(_cpp.rsi(close, 14), kernel(close, 14))

    @pytest.mark.parametrize("gaps", GAPS)
    @pytest.mark.parametrize("column", [0, 1, 2])
    def test_adx(self, gaps, column):
        cols = list(_ohlc())
        cols[column] = _with_gaps(cols[column], gaps)
        for kernel in _fallbacks(_adx_numba):
            _assert_same(_cpp.adx(*cols, 14), kernel(*cols, 14))

    @pytest.mark.parametrize("gaps", GAPS)
    @pytest.mark.parametrize("column", [0, 1, 2])
    def test_wilder_atr(self, gaps, column):
        cols = list(_ohlc())
        cols[column] = _with_gaps(cols[column], gaps)
        for kernel in _fallbacks(_wilder_atr_kernel):
            _assert_same(_cpp.wilder_atr(*cols, 14), kernel(*cols, 14))

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_an_infinity_reads_as_a_missing_bar_on_both(self, bad):
        """What a direct kernel caller sees; the library refuses it first."""
        high, low, close = _ohlc()
        inf_close = _with_gaps(close, [30], bad)
        nan_close = _with_gaps(close, [30])
        _assert_same(_cpp.rsi(inf_close, 14), _cpp.rsi(nan_close, 14))
        _assert_same(_rsi_numba(inf_close, 14), _rsi_numba(nan_close, 14))
        _assert_same(
            _cpp.wilder_atr(high, low, inf_close, 14),
            _wilder_atr_kernel(high, low, inf_close, 14),
        )
        _assert_same(
            _cpp.adx(high, low, inf_close, 14), _adx_numba(high, low, inf_close, 14)
        )


class TestTheRecursionSkipsTheGap:
    """The definition, not merely agreement: the indicator of the series
    with the missing bars dropped, and NaN at them."""

    @pytest.mark.parametrize("gaps", GAPS[1:])
    def test_rsi_is_the_rsi_of_the_present_bars(self, gaps):
        close = _with_gaps(_ohlc()[2], gaps)
        present = np.isfinite(close)
        full = _cpp.rsi(close, 14)
        assert np.isnan(full[~present]).all()
        np.testing.assert_array_equal(full[present], _cpp.rsi(close[present], 14))

    @pytest.mark.parametrize("gaps", GAPS[1:])
    def test_adx_and_atr_are_those_of_the_present_bars(self, gaps):
        high, low, close = _ohlc()
        low = _with_gaps(low, gaps)
        present = np.isfinite(low)
        adx_full = _cpp.adx(high, low, close, 14)
        atr_full = _cpp.wilder_atr(high, low, close, 14)
        assert np.isnan(adx_full[~present]).all()
        assert np.isnan(atr_full[~present]).all()
        np.testing.assert_array_equal(
            adx_full[present],
            _cpp.adx(high[present], low[present], close[present], 14),
        )
        np.testing.assert_array_equal(
            atr_full[present],
            _cpp.wilder_atr(high[present], low[present], close[present], 14),
        )

    def test_a_nan_in_the_seed_window_no_longer_ends_the_series(self):
        """The native RSI summed the NaN into its seed and was NaN from
        there to the last bar."""
        close = _with_gaps(_ohlc()[2], [5])
        assert np.isfinite(_cpp.rsi(close, 14)[-1])

    def test_a_nan_in_the_forward_pass_is_not_a_flat_bar(self):
        """Both backends used to read a NaN change as zero -- a fabricated
        unchanged price. The gap bar is now NaN, and the next bar measures
        its change against the last present price."""
        close = _ohlc()[2]
        gapped = _with_gaps(close, [60])
        out = _cpp.rsi(gapped, 14)
        assert np.isnan(out[60])
        assert out[61] == _cpp.rsi(np.delete(close, 60), 14)[60]


class TestStochasticDRecovers:
    def test_an_infinite_close_no_longer_poisons_every_later_d(self):
        """An infinite %K entered %D's running sum, and inf - inf left
        every later %D NaN."""
        high, low, close = _ohlc()
        out = _cpp.stochastic_oscillator(
            high, low, _with_gaps(close, [50], np.inf), 14, 3
        )
        assert np.isnan(out[50, 0])
        assert np.isnan(out[50:53, 1]).all()
        assert np.isfinite(out[53:, 1]).all()

    @pytest.mark.parametrize("gaps", GAPS)
    @pytest.mark.parametrize("column", ["high", "low", "close"])
    def test_wrapper_backends_agree(self, gaps, column, monkeypatch):
        high, low, close = (pd.Series(a) for a in _ohlc())
        frame = {"high": high, "low": low, "close": close}
        frame[column] = pd.Series(_with_gaps(frame[column].to_numpy(), gaps))
        args = (frame["high"], frame["low"], frame["close"], 14, 3)
        native = momentum.stochastic_oscillator(*args)
        monkeypatch.setattr(momentum, "HAS_CPP", False)
        fallback = momentum.stochastic_oscillator(*args)
        for col in ("Stoch_K", "Stoch_D"):
            _assert_same(native[col].to_numpy(), fallback[col].to_numpy())


# ── The single-series functions, native against fallback ──────────────────


class TestWrappersAgreeOnGaps:
    @pytest.mark.parametrize("gaps", GAPS)
    def test_every_wrapper(self, gaps, monkeypatch):
        high, low, close = _ohlc()
        h, l_, c = (pd.Series(a) for a in (high, _with_gaps(low, gaps), close))
        cg = pd.Series(_with_gaps(close, gaps))

        def run():
            return {
                "rsi": momentum.rsi(cg, 14).to_numpy(),
                "adx": trend.adx(h, l_, c, 14).to_numpy(),
                "atr": volatility.wilder_atr(h, l_, c, 14).to_numpy(),
                "bb": volatility.bollinger_bands(cg, 20, 2.0).to_numpy(),
            }

        native = run()
        for module in (momentum, trend, volatility):
            monkeypatch.setattr(module, "HAS_CPP", False)
        fallback = run()
        for name in native:
            _assert_same(native[name], fallback[name])

    @pytest.mark.parametrize(
        "call",
        [
            lambda h, l_, c: momentum.rsi(c, 14),
            lambda h, l_, c: trend.adx(h, l_, c, 14),
            lambda h, l_, c: volatility.wilder_atr(h, l_, c, 14),
            lambda h, l_, c: volatility.bollinger_bands(c, 20),
            lambda h, l_, c: momentum.stochastic_oscillator(h, l_, c),
        ],
    )
    @pytest.mark.parametrize("native", [True, False])
    def test_every_wrapper_refuses_an_infinity(self, call, native, monkeypatch):
        if not native:
            for module in (momentum, trend, volatility):
                monkeypatch.setattr(module, "HAS_CPP", False)
        high, low, close = (pd.Series(a) for a in _ohlc())
        close.iloc[20] = np.inf
        with pytest.raises(ValidationError, match="infinite"):
            call(high, low, close)


# ── The panel: its fallback reads a gap exactly as its kernel does ─────────


def _universe(gap_ticker: str = "BBB") -> dict:
    index = pd.date_range("2024-01-01", periods=N, freq="B")
    frames = {}
    for seed, ticker in enumerate(["AAA", "BBB", "CCC"]):
        high, low, close = _ohlc(seed)
        if ticker == gap_ticker:
            close = _with_gaps(close, [8, 60])
            high = _with_gaps(high, [100])
        frames[ticker] = pd.DataFrame(
            {"High": high, "Low": low, "Close": close}, index=index
        )
    return frames


_NATIVE_FIVE = ["rsi", "adx", "atr", "bollinger_bands", "stochastic_oscillator"]


class TestPanelFallbackReadsAGapLikeTheKernel:
    def test_the_fallback_answers_what_the_kernel_answers(self, monkeypatch):
        """The fallback looped wrappers that refused NaN, so a universe with
        one missing bar was answered natively and refused without the
        extension."""
        universe = _universe()
        native = technical_indicators_panel(universe, _NATIVE_FIVE)
        for module in (panel, momentum, trend, volatility):
            monkeypatch.setattr(module, "HAS_CPP", False)
        fallback = technical_indicators_panel(universe, _NATIVE_FIVE)
        assert set(native) == set(fallback) == set(_NATIVE_FIVE)
        for name in _NATIVE_FIVE:
            pd.testing.assert_frame_equal(
                native[name], fallback[name], rtol=1e-12, atol=0.0
            )
        # The gap is a gap, not a refusal and not a poisoned tail.
        assert np.isnan(native["rsi"]["BBB"].iloc[60])
        assert np.isfinite(native["rsi"]["BBB"].iloc[-1])

    def test_null_case_a_clean_universe_is_unchanged(self, monkeypatch):
        universe = _universe(gap_ticker="none")
        native = technical_indicators_panel(universe, _NATIVE_FIVE)
        for module in (panel, momentum, trend, volatility):
            monkeypatch.setattr(module, "HAS_CPP", False)
        fallback = technical_indicators_panel(universe, _NATIVE_FIVE)
        for name in _NATIVE_FIVE:
            pd.testing.assert_frame_equal(
                native[name], fallback[name], rtol=1e-12, atol=0.0
            )

    @pytest.mark.parametrize("native", [True, False])
    def test_a_bollinger_period_below_two_is_refused_on_both(self, native, monkeypatch):
        if not native:
            for module in (panel, momentum, trend, volatility):
                monkeypatch.setattr(module, "HAS_CPP", False)
        with pytest.raises(ValidationError, match="at least 2"):
            technical_indicators_panel(
                _universe(), ["bollinger_bands"], bollinger_period=1
            )
