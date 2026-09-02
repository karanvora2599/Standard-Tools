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
_PANEL_SHAPES: Dict[str, Optional[List[str]]] = {
    "rsi": None,
    "atr": None,
    "adx": ["DI_Plus", "DI_Minus", "ADX"],
    "bollinger_bands": ["BB_Upper", "BB_Middle", "BB_Lower"],
    "stochastic_oscillator": ["Stoch_K", "Stoch_D"],
}


def _stack_panel(
    ohlcv_by_ticker: Mapping[str, pd.DataFrame],
    tickers: Sequence[str],
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Align every ticker onto one index and build (n_tickers, n_bars) matrices.

    The common index is the INTERSECTION of every ticker's bars. That is the
    only shape a dense panel can have, and it is stated here because it is a
    real difference from computing each ticker on its own full history: a
    ticker with a shorter history truncates the panel for everyone.
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
    for i, t in enumerate(tickers):
        frame = ohlcv_by_ticker[t].loc[index]
        for col, dest in (("High", high), ("Low", low), ("Close", close)):
            if col not in frame.columns:
                raise ValidationError(
                    f"technical_indicators_panel: {t!r} is missing column {col!r}"
                )
            dest[i] = frame[col].to_numpy(dtype=np.float64)
    return index, high, low, close  # type: ignore[return-value]


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
) -> Dict[str, pd.DataFrame]:
    """
    Compute indicators for a whole universe in one native call.

    Args:
        ohlcv_by_ticker: ticker -> OHLCV DataFrame. Each needs High/Low/Close.
        indicators: any of "rsi", "adx", "atr", "bollinger_bands",
            "stochastic_oscillator".
        rsi_period ... stoch_d_period: the same parameters the per-ticker
            functions take, applied to every ticker.

    Returns:
        dict of indicator name -> wide DataFrame. Single-column indicators
        ("rsi", "atr") are indexed by date with one column per ticker.
        Multi-column ones ("adx", "bollinger_bands", "stochastic_oscillator")
        use a (ticker, field) MultiIndex on the columns.

    Raises:
        ValidationError: on an unknown indicator name, an empty universe, a
            missing OHLC column, or tickers with no bars in common.
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

    index, high, low, close = _stack_panel(ohlcv_by_ticker, tickers)
    wanted = set(indicators)
    logger.debug(
        "[panel] tickers=%d  bars=%d  indicators=%s  path=%s",
        len(tickers),
        len(index),
        sorted(wanted),
        "C++" if (HAS_CPP and _cpp_core is not None) else "per-ticker",
    )

    if HAS_CPP and _cpp_core is not None:
        raw = _cpp_core.technical_indicators_panel(
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
    else:
        raw = _panel_fallback(
            high,
            low,
            close,
            wanted,
            rsi_period,
            adx_period,
            atr_period,
            bollinger_period,
            bollinger_num_std,
            stoch_k_period,
            stoch_d_period,
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
