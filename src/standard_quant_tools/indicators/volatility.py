import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.momentum import _LastFinite
from standard_quant_tools.validation import require_finite_array, validate_series

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


@validate_series()
def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Uses C++ fused mean+std path when available (3-8× faster than two pandas
    rolling passes).  Falls back to pandas otherwise.

    Wherever a window is flat (all `period` prices identical), upper, middle
    and lower are that price exactly, on either backend -- see
    `collapse_flat_windows`.
    """
    if period <= 0:
        raise ValidationError(f"period must be > 0, got {period}")
    if not np.isfinite(num_std):
        raise ValidationError(f"num_std must be finite, got {num_std!r}")
    logger.debug(
        "[bollinger] period=%d  std=%.1f  bars=%d  path=%s",
        period,
        num_std,
        len(series),
        "C++" if (HAS_CPP and _cpp_core is not None) else "pandas",
    )

    # Checked once, unconditionally, BEFORE the C++ try/except below --
    # that except catches Exception broadly (to fall back to pandas on any
    # C++ failure), which would otherwise silently swallow a
    # ValidationError raised inside the try block and mask bad input
    # behind a confusing fallback instead of rejecting it.
    require_finite_array(series.to_numpy(dtype=np.float64), "series", "bollinger_bands")

    prices = series.to_numpy(dtype=np.float64)
    bands: Optional[np.ndarray] = None

    # ── C++ fast path ─────────────────────────────────────────────────────────
    # Only the kernel call is inside the try: an exception from anything
    # else here (the debug line used to be inside) was read as a kernel
    # failure and silently answered from pandas instead.
    if HAS_CPP and _cpp_core is not None:
        try:
            bands = np.array(_cpp_core.bollinger_bands(prices, period, num_std))
        except Exception as exc:
            logger.warning("[bollinger] C++ failed (%s) — using pandas", exc)

    # ── Pandas fallback ───────────────────────────────────────────────────────
    if bands is None:
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        bands = np.column_stack(
            [
                (sma + std * num_std).to_numpy(dtype=np.float64),
                sma.to_numpy(dtype=np.float64),
                (sma - std * num_std).to_numpy(dtype=np.float64),
            ]
        )

    collapse_flat_windows(prices, bands, period)
    result = pd.DataFrame(
        {"BB_Upper": bands[:, 0], "BB_Middle": bands[:, 1], "BB_Lower": bands[:, 2]},
        index=series.index,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[bollinger] last upper=%s  middle=%s  lower=%s",
            _LastFinite(result["BB_Upper"]),
            _LastFinite(result["BB_Middle"]),
            _LastFinite(result["BB_Lower"]),
        )
    return result


def flat_window_mask(prices: np.ndarray, period: int) -> np.ndarray:
    """
    True at bar i when the `period` bars ending at i are all the same price.

    Works along the last axis, so a (tickers, bars) panel is answered in one
    pass. It is a run-length test -- a window is flat exactly when the run
    of equal consecutive prices ending at its last bar is at least `period`
    long -- so it is exact and O(n), and it needs no rolling max and min.
    NaN never equals itself, so a window holding one is never flat.
    """
    prices = np.asarray(prices, dtype=np.float64)
    n = prices.shape[-1]
    position = np.arange(n)
    starts_run = np.ones(prices.shape, dtype=bool)
    starts_run[..., 1:] = prices[..., 1:] != prices[..., :-1]
    run_start = np.maximum.accumulate(np.where(starts_run, position, 0), axis=-1)
    return (position - run_start + 1) >= period


def collapse_flat_windows(
    prices: np.ndarray, bands: np.ndarray, period: int
) -> np.ndarray:
    """
    Set upper, middle and lower to the price wherever the window is flat.

    `prices` is (..., n) and `bands` is (..., n, 3) in upper/middle/lower
    order; `bands` is modified in place and returned.

    WHY. A window of identical prices has a mean equal to that price and a
    standard deviation of exactly zero, so the three bands ARE the price.
    What the backends compute there is rounding: pandas 3.x's online
    rolling variance leaves a standard deviation of up to 2.6e-5 on flat
    windows at real price levels, in about half of them (pandas 2.x and the
    native kernel land on zero), and the reversion strategy compares the
    close to the lower band and the middle band exactly -- so a residue
    changed which bars traded depending on the backend and the pandas
    version. Setting the answer where it is known makes it the same
    everywhere.

    A 1-bar window is left alone: its sample standard deviation is 0/0,
    not zero, and the backends' existing answers stand.
    """
    if period < 2:
        return bands
    flat = flat_window_mask(prices, period)
    if flat.any():
        bands[flat] = np.asarray(prices, dtype=np.float64)[flat][:, None]
    return bands


@validate_series()
def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    Uses np.maximum for a single-pass true range instead of pd.concat.
    """
    if period <= 0:
        raise ValidationError(f"period must be > 0, got {period}")
    if not (len(high) == len(low) == len(close)):
        raise ValidationError(
            "atr: high/low/close must all be the same length, got "
            f"{len(high)}/{len(low)}/{len(close)}"
        )
    logger.debug("[atr] period=%d  bars=%d", period, len(close))
    prev_close = close.shift(1).to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    # Same finite-input contract wilder_atr already enforces — these two are
    # siblings computing the same true range, and it made no sense for one to
    # reject NaN/Inf while the other quietly propagated it into the rolling
    # mean.
    require_finite_array(h, "high", "atr")
    require_finite_array(l, "low", "atr")
    require_finite_array(close.to_numpy(dtype=float), "close", "atr")
    tr = pd.Series(
        np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close))),
        index=close.index,
    )
    result = tr.rolling(window=period).mean()
    valid = result.dropna()
    if not valid.empty:
        logger.debug("[atr] last=%.4f", float(valid.iloc[-1]))
    return result


@validate_series()
def wilder_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range using Wilder's smoothing (not a simple rolling mean).

    TR[0] = high[0] - low[0]
    TR[i] = max(H[i]-L[i], |H[i]-C[i-1]|, |L[i]-C[i-1]|)
    Seed:    ATR[period-1] = mean(TR[0..period-1])
    Forward: ATR[i]        = (ATR[i-1]*(period-1) + TR[i]) / period

    Uses C++ fast path when built, otherwise falls back to a pure-Python loop.
    First period-1 values are NaN.
    """
    if period <= 0:
        raise ValidationError(f"period must be > 0, got {period}")

    h = high.to_numpy(dtype=np.float64)
    l = low.to_numpy(dtype=np.float64)
    c = close.to_numpy(dtype=np.float64)
    n = len(h)

    require_finite_array(h, "high", "wilder_atr")
    require_finite_array(l, "low", "wilder_atr")
    require_finite_array(c, "close", "wilder_atr")

    if HAS_CPP and _cpp_core is not None:
        raw = _cpp_core.wilder_atr(h, l, c, period)
        return pd.Series(raw, index=close.index, name="Wilder_ATR")

    # Pure-Python fallback (correct but slow; C++ path is preferred)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    result = np.full(n, np.nan)
    if n >= period:
        result[period - 1] = tr[:period].mean()
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return pd.Series(result, index=close.index, name="Wilder_ATR")
