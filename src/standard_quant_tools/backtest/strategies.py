"""
Signal generation functions for built-in backtest strategies.

Each function accepts a full OHLCV DataFrame plus strategy-specific kwargs
and returns a pd.Series of signals (1 = long, 0 = flat, -1 = short).

All functions are defined at module level so they are picklable by
ProcessPoolExecutor (required for backtest_grid on Windows / spawn).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from standard_quant_tools.indicators.momentum import rsi
from standard_quant_tools.indicators.trend import adx, macd, sma
from standard_quant_tools.indicators.volatility import bollinger_bands
from standard_quant_tools.indicators.volume import vwap

logger = logging.getLogger(__name__)

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(func):  # type: ignore[misc]
        return func


_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass


@njit
def _rsi_state_machine(
    rsi_arr: np.ndarray, oversold: float, overbought: float
) -> np.ndarray:
    n = len(rsi_arr)
    values = np.zeros(n)
    in_pos = False
    for i in range(n):
        if np.isnan(rsi_arr[i]):
            # Carry the current position through a NaN (warmup) bar instead
            # of hardcoding 0.0 -- the in_pos state itself is untouched by
            # this bar either way, so a caller reading this as a real
            # position series should see the position actually held, not a
            # phantom close/reopen blip around bars this indicator can't
            # evaluate yet.
            values[i] = 1.0 if in_pos else 0.0
            continue
        if not in_pos and rsi_arr[i] < oversold:
            in_pos = True
        elif in_pos and rsi_arr[i] > overbought:
            in_pos = False
        values[i] = 1.0 if in_pos else 0.0
    return values


@njit
def _bollinger_state_machine(
    close_arr: np.ndarray, lower_arr: np.ndarray, middle_arr: np.ndarray
) -> np.ndarray:
    n = len(close_arr)
    values = np.zeros(n)
    in_pos = False
    for i in range(n):
        if np.isnan(lower_arr[i]):
            values[i] = 1.0 if in_pos else 0.0
            continue
        if not in_pos and close_arr[i] <= lower_arr[i]:
            in_pos = True
        elif in_pos and close_arr[i] >= middle_arr[i]:
            in_pos = False
        values[i] = 1.0 if in_pos else 0.0
    return values


@njit
def _donchian_state_machine(
    close_arr: np.ndarray, entry_max_arr: np.ndarray, exit_min_arr: np.ndarray
) -> np.ndarray:
    n = len(close_arr)
    values = np.zeros(n)
    in_pos = False
    for i in range(n):
        if np.isnan(entry_max_arr[i]) or np.isnan(exit_min_arr[i]):
            values[i] = 1.0 if in_pos else 0.0
            continue
        if not in_pos and close_arr[i] >= entry_max_arr[i]:
            in_pos = True
        elif in_pos and close_arr[i] <= exit_min_arr[i]:
            in_pos = False
        values[i] = 1.0 if in_pos else 0.0
    return values


@njit
def _vwap_reversion_state_machine(
    close_arr: np.ndarray, vwap_arr: np.ndarray, entry_threshold: float
) -> np.ndarray:
    n = len(close_arr)
    values = np.zeros(n)
    in_pos = False
    for i in range(n):
        if np.isnan(vwap_arr[i]):
            values[i] = 1.0 if in_pos else 0.0
            continue
        if not in_pos and close_arr[i] <= vwap_arr[i] * (1.0 - entry_threshold):
            in_pos = True
        elif in_pos and close_arr[i] >= vwap_arr[i]:
            in_pos = False
        values[i] = 1.0 if in_pos else 0.0
    return values


def _log_signals(name: str, signals: pd.Series) -> None:
    arr = signals.to_numpy()
    n = len(arr)
    n_long = int((arr == 1).sum())
    n_short = int((arr == -1).sum())
    n_flat = n - n_long - n_short
    logger.debug(
        "[signal] %s  bars=%d  long=%d(%.0f%%)  flat=%d(%.0f%%)  short=%d(%.0f%%)",
        name,
        n,
        n_long,
        100 * n_long / n if n else 0,
        n_flat,
        100 * n_flat / n if n else 0,
        n_short,
        100 * n_short / n if n else 0,
    )


def _sma_signals(
    df: pd.DataFrame,
    fast_period: int = 10,
    slow_period: int = 30,
    **_,
) -> pd.Series:
    """Long when fast SMA > slow SMA, flat otherwise."""
    logger.debug(
        "[signal] sma_crossover  fast=%d  slow=%d  bars=%d",
        fast_period,
        slow_period,
        len(df),
    )
    result = pd.Series(
        np.where(sma(df["Close"], fast_period) > sma(df["Close"], slow_period), 1, 0),
        index=df.index,
    )
    _log_signals("sma_crossover", result)
    return result


def _rsi_signals(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    **_,
) -> pd.Series:
    """Enter long when RSI < oversold; hold until RSI > overbought."""
    logger.debug(
        "[signal] rsi_mean_reversion  period=%d  oversold=%.0f  overbought=%.0f  bars=%d",
        period,
        oversold,
        overbought,
        len(df),
    )
    rsi_arr = rsi(df["Close"], period).to_numpy(dtype=float)
    result = pd.Series(
        _rsi_state_machine(rsi_arr, oversold, overbought), index=df.index
    )
    _log_signals("rsi_mean_reversion", result)
    return result


def _macd_signals(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    **_,
) -> pd.Series:
    """Long when MACD line > signal line, flat otherwise."""
    logger.debug(
        "[signal] macd_crossover  fast=%d  slow=%d  signal=%d  bars=%d",
        fast,
        slow,
        signal,
        len(df),
    )
    m = macd(df["Close"], fast, slow, signal)
    result = pd.Series(
        np.where(m["MACD"] > m["Signal"], 1, 0),
        index=df.index,
    )
    _log_signals("macd_crossover", result)
    return result


def _bollinger_signals(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    **_,
) -> pd.Series:
    """Enter long when price touches lower band; exit at middle band."""
    logger.debug(
        "[signal] bollinger_reversion  period=%d  std=%.1f  bars=%d",
        period,
        num_std,
        len(df),
    )
    bb = bollinger_bands(df["Close"], period, num_std)
    close_arr = df["Close"].to_numpy(dtype=float)
    lower_arr = bb["BB_Lower"].to_numpy(dtype=float)
    middle_arr = bb["BB_Middle"].to_numpy(dtype=float)
    result = pd.Series(
        _bollinger_state_machine(close_arr, lower_arr, middle_arr), index=df.index
    )
    _log_signals("bollinger_reversion", result)
    return result


def _donchian_signals(
    df: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
    **_,
) -> pd.Series:
    """
    Turtle-style Donchian channel breakout. Enter long when Close makes a
    new entry_period-bar high (compared against the PRIOR entry_period bars
    via .shift(1), not including today — a genuine breakout beyond the
    already-established channel, not the tautology "today's close is >=
    today's own rolling max"). Exit to flat on a new exit_period-bar low,
    same shift(1) convention. entry_period > exit_period is the classic
    asymmetric-channel design (slower entry, faster exit).

    Fully vectorized rolling max/min (pandas, O(n) — not O(n*window)) feed
    a numba-JIT state machine for the entry/exit hysteresis, the same
    pattern as _rsi_signals/_bollinger_signals — no interpreted Python loop
    regardless of series length.
    """
    logger.debug(
        "[signal] donchian_breakout  entry=%d  exit=%d  bars=%d",
        entry_period,
        exit_period,
        len(df),
    )
    entry_max = df["High"].rolling(entry_period).max().shift(1)
    exit_min = df["Low"].rolling(exit_period).min().shift(1)
    close_arr = df["Close"].to_numpy(dtype=float)
    entry_arr = entry_max.to_numpy(dtype=float)
    exit_arr = exit_min.to_numpy(dtype=float)
    if HAS_CPP and _cpp_core is not None:
        signal_arr = _cpp_core.donchian_state_machine(close_arr, entry_arr, exit_arr)
    else:
        signal_arr = _donchian_state_machine(close_arr, entry_arr, exit_arr)
    result = pd.Series(signal_arr, index=df.index)
    _log_signals("donchian_breakout", result)
    return result


def _momentum_signals(
    df: pd.DataFrame,
    lookback: int = 90,
    threshold: float = 0.0,
    **_,
) -> pd.Series:
    """
    Time-series (absolute) momentum: long when the trailing lookback-bar
    return exceeds threshold, flat otherwise. No per-bar state — a single
    vectorized pandas.Series.pct_change(periods=lookback) call, the cheapest
    strategy in this registry to evaluate on very large series (millions of
    rows): one O(n) pass, no numba, no rolling window at all beyond the
    single lagged difference pct_change already computes internally.
    """
    logger.debug(
        "[signal] momentum_timeseries  lookback=%d  threshold=%.4f  bars=%d",
        lookback,
        threshold,
        len(df),
    )
    trailing_return = df["Close"].pct_change(periods=lookback)
    result = pd.Series(np.where(trailing_return > threshold, 1, 0), index=df.index)
    _log_signals("momentum_timeseries", result)
    return result


def _vwap_reversion_signals(
    df: pd.DataFrame,
    period: int = 20,
    entry_threshold: float = 0.02,
    **_,
) -> pd.Series:
    """
    Rolling-VWAP mean reversion. Enter long when Close has dropped
    entry_threshold (fractional) below its own trailing period-bar
    volume-weighted average; exit back to flat once price recovers to VWAP.
    Aimed specifically at intraday/tick data, where VWAP — not a plain
    price mean — is the standard fair-value benchmark; contrast with
    bollinger_reversion, which reverts to a plain (unweighted) rolling mean.

    Reuses indicators.volume.vwap (already a single vectorized rolling-sum
    ratio, O(n)); the entry/exit hysteresis runs through the same
    numba-JIT state-machine pattern as bollinger_reversion/donchian_breakout.
    """
    logger.debug(
        "[signal] vwap_reversion  period=%d  entry_threshold=%.4f  bars=%d",
        period,
        entry_threshold,
        len(df),
    )
    vwap_series = vwap(df["High"], df["Low"], df["Close"], df["Volume"], period=period)
    close_arr = df["Close"].to_numpy(dtype=float)
    vwap_arr = vwap_series.to_numpy(dtype=float)
    if HAS_CPP and _cpp_core is not None:
        signal_arr = _cpp_core.vwap_reversion_state_machine(
            close_arr, vwap_arr, entry_threshold
        )
    else:
        signal_arr = _vwap_reversion_state_machine(close_arr, vwap_arr, entry_threshold)
    result = pd.Series(signal_arr, index=df.index)
    _log_signals("vwap_reversion", result)
    return result


def _adx_trend_signals(
    df: pd.DataFrame,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    **_,
) -> pd.Series:
    """
    Trend-strength-filtered directional strategy: long only when ADX
    confirms a genuinely trending market (> adx_threshold) AND +DI > -DI
    (the trend is up). No per-bar state machine — each bar's decision
    depends only on that bar's own (already-rolling) indicator values, so
    it's a single vectorized boolean AND over the adx() indicator's output.
    ADX itself carries the C++/Numba/pure-Python fallback chain
    indicators.trend.adx already has; this strategy adds no further loop
    on top of it.
    """
    logger.debug(
        "[signal] adx_trend  adx_period=%d  adx_threshold=%.1f  bars=%d",
        adx_period,
        adx_threshold,
        len(df),
    )
    adx_df = adx(df["High"], df["Low"], df["Close"], adx_period)
    trending_up = (adx_df["ADX"] > adx_threshold) & (
        adx_df["DI_Plus"] > adx_df["DI_Minus"]
    )
    result = pd.Series(np.where(trending_up, 1, 0), index=df.index)
    _log_signals("adx_trend", result)
    return result


STRATEGY_REGISTRY = {
    "sma_crossover": _sma_signals,
    "rsi_mean_reversion": _rsi_signals,
    "macd_crossover": _macd_signals,
    "bollinger_reversion": _bollinger_signals,
    "donchian_breakout": _donchian_signals,
    "momentum_timeseries": _momentum_signals,
    "vwap_reversion": _vwap_reversion_signals,
    "adx_trend": _adx_trend_signals,
}
