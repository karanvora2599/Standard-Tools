"""
Panel (whole-universe) indicator entry points.

Why this module exists, in one measurement. At 2,000 tickers x 2,000 bars:

    2000 x raw binding      _sqt_core.rsi()      38.0 ms    19.0 us/ticker
    1  x contiguous 4M-bar  _sqt_core.rsi()      32.7 ms
    -> pybind11 dispatch overhead                 5.3 ms     2.7 us  (14%)
    2000 x Python wrapper   indicators.rsi()    636.0 ms   318.0 us  (16.7x)

The C++ call boundary is not the problem. The per-ticker pandas round trip
is: `Series` -> NumPy, validation, logging, and `Series` reconstruction cost
16.7x the kernel itself. Batching the C++ calls alone would buy about 14%.

So these functions convert the whole universe once, hand the native side one
matrix, and get one matrix back -- and the kernel runs the tickers in
parallel on top of that. The per-ticker wrappers in `momentum`, `trend` and
`volatility` are unchanged and remain the right thing for a single series.

Arithmetic is identical: `technical_indicators_panel` feeds each row to the
same `*_into` kernels the single-series path uses, so panel output is
bit-identical to looping the per-ticker call. Verified in
tests/cpp_bindings/test_cpp_panel_indicators.py.

The kernel carries five of the fourteen indicators this module serves. The
other nine loop the per-ticker wrappers row by row, which is slower and is
the SAME NUMBER -- the wrapper is the definition, and the panel column is
its answer on those bars. So the shape of the API does not depend on which
indicator a caller asks for, and an indicator only the loop serves is still
obtainable as a HISTORY across a whole universe, which before the CHANGELOG
entry of 2026-09-22 it was not, at any universe size.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass


# Indicator -> field names of its trailing axis, or None for a single value
# per bar. The names are exactly the ones the per-ticker wrappers use, so a
# caller moving from `bollinger_bands(series)` to the panel finds the same
# labels rather than having to learn a second vocabulary.
#
# `atr` is WILDER'S average true range, which is what it has always been
# here and what the native kernel computes. The simple rolling-mean variant
# in `volatility.atr` is a different number, so it is a separate name --
# `atr_simple` -- rather than a parameter that would silently change what
# an existing caller's `atr` column means.
_PANEL_SHAPES: Dict[str, Optional[List[str]]] = {
    "rsi": None,
    "atr": None,
    "atr_simple": None,
    "adx": ["DI_Plus", "DI_Minus", "ADX"],
    "bollinger_bands": ["BB_Upper", "BB_Middle", "BB_Lower"],
    "stochastic_oscillator": ["Stoch_K", "Stoch_D"],
    "macd": ["MACD", "Signal", "Histogram"],
    "sma": None,
    "ema": None,
    "williams_r": None,
    "obv": None,
    "vwap": None,
    "parabolic_sar": ["SAR", "Trend"],
    "mfi": None,
}

#: The five the C++ kernel computes in one call. Everything else is looped
#: over the per-ticker wrappers, which is the same arithmetic at a lower
#: speed rather than a second implementation of it.
_NATIVE_INDICATORS = frozenset(
    {"rsi", "atr", "adx", "bollinger_bands", "stochastic_oscillator"}
)

#: These read Volume, so a panel built from High/Low/Close alone cannot
#: answer them and says so by name.
_VOLUME_INDICATORS = frozenset({"obv", "vwap", "mfi"})


def _stack_panel(
    ohlcv_by_ticker: Mapping[str, pd.DataFrame],
    tickers: Sequence[str],
    *,
    need_volume: bool = False,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Align every ticker onto one index and build (n_tickers, n_bars) matrices.

    The common index is the INTERSECTION of every ticker's bars. That is the
    only shape a dense panel can have, and it is stated here because it is a
    real difference from computing each ticker on its own full history: a
    ticker with a shorter history truncates the panel for everyone.

    Volume is stacked only when an indicator that reads it was requested.
    Requiring it unconditionally would refuse a High/Low/Close panel that
    every previously supported indicator computes from perfectly well.
    """
    index: Optional[pd.Index] = None
    for t in tickers:
        idx = ohlcv_by_ticker[t].index
        index = idx if index is None else index.intersection(idx)
    if index is None or len(index) == 0:
        raise ValidationError(
            "technical_indicators_panel: the tickers share no common bars"
        )

    high = np.empty((len(tickers), len(index)), dtype=np.float64)
    low = np.empty_like(high)
    close = np.empty_like(high)
    volume = np.empty_like(high) if need_volume else None
    for i, t in enumerate(tickers):
        frame = ohlcv_by_ticker[t].loc[index]
        columns: List[tuple] = [("High", high), ("Low", low), ("Close", close)]
        if volume is not None:
            columns.append(("Volume", volume))
        for col, dest in columns:
            if col not in frame.columns:
                extra = ""
                if col == "Volume":
                    extra = (
                        " -- obv, vwap and mfi are volume indicators, so a "
                        "High/Low/Close panel cannot answer them. Supply a "
                        "Volume column, or request only price indicators."
                    )
                raise ValidationError(
                    f"technical_indicators_panel: {t!r} is missing column "
                    f"{col!r}{extra}"
                )
            dest[i] = frame[col].to_numpy(dtype=np.float64)
    return index, high, low, close, volume  # type: ignore[return-value]


def technical_indicators_panel(
    ohlcv_by_ticker: Mapping[str, pd.DataFrame],
    indicators: Sequence[str],
    *,
    rsi_period: int = 14,
    adx_period: int = 14,
    atr_period: int = 14,
    bollinger_period: int = 20,
    bollinger_num_std: float = 2.0,
    stoch_k_period: int = 14,
    stoch_d_period: int = 3,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    sma_period: int = 14,
    ema_period: int = 14,
    williams_period: int = 14,
    vwap_period: Optional[int] = None,
    mfi_period: int = 14,
    sar_af_start: float = 0.02,
    sar_af_step: float = 0.02,
    sar_af_max: float = 0.2,
    atr_simple_period: int = 14,
) -> Dict[str, pd.DataFrame]:
    """
    Compute indicators for a whole universe in one pass.

    Args:
        ohlcv_by_ticker: ticker -> OHLCV DataFrame. Each needs High/Low/Close,
            and also Volume when "obv", "vwap" or "mfi" is requested.
        indicators: any of "rsi", "adx", "atr", "atr_simple",
            "bollinger_bands", "stochastic_oscillator", "macd", "sma", "ema",
            "williams_r", "obv", "vwap", "parabolic_sar", "mfi".
        rsi_period ... atr_simple_period: the same parameters the per-ticker
            functions take, applied to every ticker.

    Returns:
        dict of indicator name -> wide DataFrame. Single-column indicators
        ("rsi", "atr", "sma", ...) are indexed by date with one column per
        ticker. Multi-column ones ("adx", "bollinger_bands",
        "stochastic_oscillator", "macd", "parabolic_sar") use a
        (ticker, field) MultiIndex on the columns.

    Raises:
        ValidationError: on an unknown indicator name, an empty universe, a
            missing OHLC(V) column, or tickers with no bars in common.

    Five of these run in the native kernel over the whole matrix; the rest
    loop the per-ticker wrappers, which is the same arithmetic and the same
    values at a lower speed. Which path served an indicator is not
    observable in its output, and that is the point.
    """
    unknown = sorted(set(indicators) - set(_PANEL_SHAPES))
    if unknown:
        raise ValidationError(
            f"technical_indicators_panel: unknown indicator(s) {unknown}; "
            f"expected any of {sorted(_PANEL_SHAPES)}"
        )
    tickers = list(ohlcv_by_ticker)
    if not tickers:
        raise ValidationError("technical_indicators_panel: no tickers supplied")
    if not indicators:
        return {}

    wanted = set(indicators)
    index, high, low, close, volume = _stack_panel(
        ohlcv_by_ticker,
        tickers,
        need_volume=bool(wanted & _VOLUME_INDICATORS),
    )
    native_wanted = wanted & _NATIVE_INDICATORS
    looped_wanted = wanted - _NATIVE_INDICATORS
    logger.debug(
        "[panel] tickers=%d  bars=%d  indicators=%s  path=%s",
        len(tickers),
        len(index),
        sorted(wanted),
        "C++" if (HAS_CPP and _cpp_core is not None) else "per-ticker",
    )

    raw: Dict[str, np.ndarray] = {}
    if native_wanted and HAS_CPP and _cpp_core is not None:
        raw.update(
            _cpp_core.technical_indicators_panel(
                high,
                low,
                close,
                compute_rsi="rsi" in wanted,
                rsi_period=rsi_period,
                compute_adx="adx" in wanted,
                adx_period=adx_period,
                compute_atr="atr" in wanted,
                atr_period=atr_period,
                compute_bollinger="bollinger_bands" in wanted,
                bollinger_period=bollinger_period,
                bollinger_num_std=bollinger_num_std,
                compute_stochastic="stochastic_oscillator" in wanted,
                stoch_k_period=stoch_k_period,
                stoch_d_period=stoch_d_period,
            )
        )
    elif native_wanted:
        raw.update(
            _panel_fallback(
                high,
                low,
                close,
                native_wanted,
                rsi_period,
                adx_period,
                atr_period,
                bollinger_period,
                bollinger_num_std,
                stoch_k_period,
                stoch_d_period,
            )
        )
    if looped_wanted:
        raw.update(
            _panel_looped(
                high,
                low,
                close,
                volume,
                looped_wanted,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                sma_period=sma_period,
                ema_period=ema_period,
                williams_period=williams_period,
                vwap_period=vwap_period,
                mfi_period=mfi_period,
                sar_af_start=sar_af_start,
                sar_af_step=sar_af_step,
                sar_af_max=sar_af_max,
                atr_simple_period=atr_simple_period,
            )
        )

    out: Dict[str, pd.DataFrame] = {}
    for name, arr in raw.items():
        fields = _PANEL_SHAPES[name]
        if fields is None:
            out[name] = pd.DataFrame(arr.T, index=index, columns=tickers)
        else:
            cols = pd.MultiIndex.from_product(
                [tickers, fields], names=["ticker", "field"]
            )
            # (ticker, bar, field) -> (bar, ticker, field) -> (bar, ticker*field),
            # which is ticker-major and so matches from_product's column order.
            out[name] = pd.DataFrame(
                arr.transpose(1, 0, 2).reshape(len(index), -1),
                index=index,
                columns=cols,
            )
    return out


def _panel_fallback(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    wanted: set,
    rsi_period: int,
    adx_period: int,
    atr_period: int,
    bollinger_period: int,
    bollinger_num_std: float,
    stoch_k_period: int,
    stoch_d_period: int,
) -> Dict[str, np.ndarray]:
    """Pure-pandas panel, for when the extension is not built.

    Loops the existing per-ticker wrappers -- the point of the module is the
    API shape, which should not disappear just because the fast path did.
    """
    from standard_quant_tools.indicators.momentum import rsi as _rsi
    from standard_quant_tools.indicators.momentum import stochastic_oscillator as _stoch
    from standard_quant_tools.indicators.trend import adx as _adx
    from standard_quant_tools.indicators.volatility import bollinger_bands as _bb
    from standard_quant_tools.indicators.volatility import wilder_atr as _atr

    n_t, n_b = close.shape
    acc: Dict[str, np.ndarray] = {}
    if "rsi" in wanted:
        acc["rsi"] = np.empty((n_t, n_b))
    if "atr" in wanted:
        acc["atr"] = np.empty((n_t, n_b))
    if "adx" in wanted:
        acc["adx"] = np.empty((n_t, n_b, 3))
    if "bollinger_bands" in wanted:
        acc["bollinger_bands"] = np.empty((n_t, n_b, 3))
    if "stochastic_oscillator" in wanted:
        acc["stochastic_oscillator"] = np.empty((n_t, n_b, 2))

    for i in range(n_t):
        h = pd.Series(high[i])
        low_s = pd.Series(low[i])
        c = pd.Series(close[i])
        if "rsi" in wanted:
            acc["rsi"][i] = _rsi(c, rsi_period).to_numpy()
        if "atr" in wanted:
            acc["atr"][i] = _atr(h, low_s, c, atr_period).to_numpy()
        if "adx" in wanted:
            acc["adx"][i] = _adx(h, low_s, c, adx_period).to_numpy()
        if "bollinger_bands" in wanted:
            # `bollinger_bands` returns BB_Upper/BB_Middle/BB_Lower -- the
            # names this module already declares in _PANEL_SHAPES above,
            # whose own comment promises they are "exactly the ones the
            # per-ticker wrappers use". Asking
            # for the unprefixed ones raised KeyError, so this fallback had
            # never run: the native path is taken in every environment that
            # builds the extension, and 507 of the suite's tests are gated
            # on that extension without CI ever checking it loaded.
            acc["bollinger_bands"][i] = _bb(c, bollinger_period, bollinger_num_std)[
                _PANEL_SHAPES["bollinger_bands"]
            ].to_numpy()
        if "stochastic_oscillator" in wanted:
            acc["stochastic_oscillator"][i] = _stoch(
                h, low_s, c, stoch_k_period, stoch_d_period
            ).to_numpy()
    return acc


def _panel_looped(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: Optional[np.ndarray],
    wanted: set,
    *,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    sma_period: int,
    ema_period: int,
    williams_period: int,
    vwap_period: Optional[int],
    mfi_period: int,
    sar_af_start: float,
    sar_af_step: float,
    sar_af_max: float,
    atr_simple_period: int,
) -> Dict[str, np.ndarray]:
    """The nine indicators the native kernel does not carry.

    Written in `_panel_fallback`'s shape and for the same reason: each row
    goes through the per-ticker wrapper, so a panel column IS the wrapper's
    answer on those bars rather than a second implementation that has to be
    kept in agreement with it.

    The kernel covers five of the fourteen. Before the CHANGELOG entry of
    2026-09-22 the panel covered only those five, which meant a MACD, SMA,
    EMA, Williams %R, OBV, VWAP, Parabolic SAR, simple-mean ATR or MFI
    HISTORY was obtainable from nothing in this library at any universe
    size -- the single-symbol doors return the last bar only.
    """
    from standard_quant_tools.indicators.trend import ema as _ema
    from standard_quant_tools.indicators.trend import macd as _macd
    from standard_quant_tools.indicators.trend import parabolic_sar as _sar
    from standard_quant_tools.indicators.trend import sma as _sma
    from standard_quant_tools.indicators.trend import williams_r as _williams
    from standard_quant_tools.indicators.volatility import atr as _atr_simple
    from standard_quant_tools.indicators.volume import mfi as _mfi
    from standard_quant_tools.indicators.volume import obv as _obv
    from standard_quant_tools.indicators.volume import vwap as _vwap

    n_t, n_b = close.shape
    acc: Dict[str, np.ndarray] = {}
    for name in wanted:
        fields = _PANEL_SHAPES[name]
        acc[name] = (
            np.empty((n_t, n_b))
            if fields is None
            else np.empty((n_t, n_b, len(fields)))
        )

    for i in range(n_t):
        h = pd.Series(high[i])
        low_s = pd.Series(low[i])
        c = pd.Series(close[i])
        v = pd.Series(volume[i]) if volume is not None else None
        if "macd" in wanted:
            acc["macd"][i] = _macd(c, macd_fast, macd_slow, macd_signal)[
                _PANEL_SHAPES["macd"]
            ].to_numpy()
        if "sma" in wanted:
            acc["sma"][i] = _sma(c, sma_period).to_numpy()
        if "ema" in wanted:
            acc["ema"][i] = _ema(c, ema_period).to_numpy()
        if "williams_r" in wanted:
            acc["williams_r"][i] = _williams(h, low_s, c, williams_period).to_numpy()
        if "parabolic_sar" in wanted:
            acc["parabolic_sar"][i] = _sar(
                h, low_s, sar_af_start, sar_af_step, sar_af_max
            )[_PANEL_SHAPES["parabolic_sar"]].to_numpy()
        if "atr_simple" in wanted:
            acc["atr_simple"][i] = _atr_simple(
                h, low_s, c, atr_simple_period
            ).to_numpy()
        if "obv" in wanted:
            acc["obv"][i] = _obv(c, v).to_numpy()
        if "vwap" in wanted:
            acc["vwap"][i] = _vwap(h, low_s, c, v, vwap_period).to_numpy()
        if "mfi" in wanted:
            acc["mfi"][i] = _mfi(h, low_s, c, v, mfi_period).to_numpy()
    return acc
