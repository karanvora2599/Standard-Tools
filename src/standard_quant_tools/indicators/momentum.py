import logging
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators._missing import refuse_infinities
from standard_quant_tools.validation import validate_series

logger = logging.getLogger(__name__)


class _LastFinite:
    """
    An indicator's latest finite reading, formatted only when a log record
    is actually emitted.

    A debug line must not be able to fail the computation it describes.
    `last_finite` is the right call for a caller that NEEDS a reading -- it
    refuses, with a reason, when the window is longer than the data -- but
    as a logging argument it ran on every call at every log level and
    raised on legitimate warm-up output. This renders "none" instead.
    """

    __slots__ = ("_series",)

    def __init__(self, series: Any) -> None:
        self._series = series

    def __str__(self) -> str:
        values = np.asarray(self._series, dtype=float)
        finite = values[np.isfinite(values)]
        return f"{finite[-1]:.4f}" if finite.size else "none"


_cpp_core: Any = None
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Dummy decorator if numba is missing
    def njit(func):
        return func


@njit
def _rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's RSI, the fallback for rsi_into in indicators.cpp.

    A non-finite price is a missing bar and is skipped: a change is measured
    between consecutive PRESENT bars, the averages are carried across the
    gap unchanged, and the output there is NaN (see _missing.py). The
    operations and their order are the native kernel's, so the two agree on
    clean data and on data with gaps alike. The seed is a sequential sum
    rather than np.mean, whose pairwise summation is a different rounding.
    """
    n = len(prices)
    rsi = np.full(n, np.nan)

    if period <= 0 or n <= period:
        return rsi

    have_prev = False
    prev = 0.0
    n_changes = 0
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(n):
        price = prices[i]
        if not math.isfinite(price):
            continue
        if not have_prev:
            prev = price
            have_prev = True
            continue
        change = price - prev
        prev = price
        gain = change if change > 0.0 else 0.0
        loss = -change if change < 0.0 else 0.0
        n_changes += 1

        if n_changes <= period:
            # Seed: the simple mean of the first `period` changes.
            avg_gain += gain
            avg_loss += loss
            if n_changes < period:
                continue
            avg_gain /= period
            avg_loss /= period
        else:
            # Wilder's smoothing.
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@validate_series()
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    Uses C++ fast path when built, then Numba JIT, then pure Python fallback.
    All three paths use Wilder's smoothing (SMA seed, then alpha=1/period).

    A NaN price is a missing bar: the recursion skips it, so the result is
    the RSI of the series with that bar dropped, and NaN at the bar itself.
    An infinite price is refused. See `indicators/_missing.py`.
    """
    # No `if series.empty` branch here: @validate_series() above already
    # rejects an empty Series, so it was unreachable.
    if period <= 0:
        raise ValidationError(f"period must be > 0, got {period}")

    values: np.ndarray = np.asarray(series.values, dtype=np.float64)
    refuse_infinities(values, "prices", "rsi")
    path = (
        "C++"
        if (HAS_CPP and _cpp_core is not None)
        else ("numba" if HAS_NUMBA else "python")
    )
    logger.debug("[rsi] period=%d  bars=%d  path=%s", period, len(values), path)

    if HAS_CPP and _cpp_core is not None:
        rsi_vals = _cpp_core.rsi(values, period)
    else:
        rsi_vals = _rsi_numba(values, period)

    result = pd.Series(rsi_vals, index=series.index)
    valid = result.dropna()
    if not valid.empty:
        logger.debug(
            "[rsi] last=%.2f  min=%.2f  max=%.2f",
            float(valid.iloc[-1]),
            float(valid.min()),
            float(valid.max()),
        )
    return result


@validate_series()
def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.

    Uses C++ fused sliding min+max path when available (5-15× faster than two
    pandas rolling passes).  Falls back to pandas otherwise.

    A NaN is a missing bar: %K is NaN for every window holding a missing
    high or low and at a bar whose close is missing, and %D is NaN for every
    window holding such a %K; both resume after it, on either backend. An
    infinite value is refused. See `indicators/_missing.py`.
    """
    if k_period <= 0:
        raise ValidationError(f"k_period must be > 0, got {k_period}")
    if d_period <= 0:
        raise ValidationError(f"d_period must be > 0, got {d_period}")

    logger.debug(
        "[stochastic] k_period=%d  d_period=%d  bars=%d  path=%s",
        k_period,
        d_period,
        len(close),
        "C++" if (HAS_CPP and _cpp_core is not None) else "pandas",
    )

    # Checked once, unconditionally, BEFORE the C++ try/except below --
    # that except catches Exception broadly (to fall back to pandas on any
    # C++ failure), which would otherwise silently swallow a
    # ValidationError raised inside the try block and mask bad input
    # behind a confusing fallback instead of rejecting it.
    refuse_infinities(high.to_numpy(dtype=np.float64), "high", "stochastic_oscillator")
    refuse_infinities(low.to_numpy(dtype=np.float64), "low", "stochastic_oscillator")
    refuse_infinities(
        close.to_numpy(dtype=np.float64), "close", "stochastic_oscillator"
    )

    # ── C++ fast path ─────────────────────────────────────────────────────────
    # Only the kernel call is inside the try. The debug line below used to be
    # in here too, with `last_finite(d, ...)` as an argument -- evaluated
    # whether or not DEBUG was on, and raising when %D had no finite value
    # yet (k_period <= n < k_period + d_period - 1: 14 and 15 bars at the
    # defaults). The except below swallowed that as a kernel failure, fell
    # back to pandas, and the same line raised again there, so a valid call
    # was refused by its own logging.
    result: Optional[pd.DataFrame] = None
    if HAS_CPP and _cpp_core is not None:
        try:
            h_arr = high.to_numpy(dtype=np.float64)
            l_arr = low.to_numpy(dtype=np.float64)
            c_arr = close.to_numpy(dtype=np.float64)
            out = _cpp_core.stochastic_oscillator(
                h_arr, l_arr, c_arr, k_period, d_period
            )
            result = pd.DataFrame(
                {"Stoch_K": out[:, 0], "Stoch_D": out[:, 1]}, index=close.index
            )
        except Exception as exc:
            logger.warning("[stochastic] C++ failed (%s) — using pandas", exc)

    # ── Pandas fallback ───────────────────────────────────────────────────────
    if result is None:
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        # A zero-range window (flat prices across the whole lookback) makes
        # %K a 0/0. Raw pandas yields NaN there while the C++ kernel above
        # yields 0.0, so the same call returned different values depending
        # only on whether _sqt_core happened to be built. Match the compiled
        # kernel's convention so the result is build-independent;
        # `range_safe` also keeps this from being an unguarded division,
        # consistent with the degenerate-window handling in
        # spread_zscore/rolling_beta.
        price_range = highest_high - lowest_low
        range_safe = price_range.where(price_range > 0)
        k = (100 * ((close - lowest_low) / range_safe)).where(
            price_range.isna() | (price_range > 0), 0.0
        )
        # A bar whose close is missing has no %K, even over a clean flat
        # window, where the line above would have set 0.0 -- the kernel's
        # rule, so a gap reads the same on both backends.
        k = k.where(close.notna())
        d = k.rolling(window=d_period).mean()
        result = pd.DataFrame({"Stoch_K": k, "Stoch_D": d})

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[stochastic] K last=%s  D last=%s",
            _LastFinite(result["Stoch_K"]),
            _LastFinite(result["Stoch_D"]),
        )
    return result
