import logging
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import (
    require_finite_scalar,
    require_finite_series,
    require_periods_per_year,
    require_positive_start_level,
)

logger = logging.getLogger(__name__)


def _label_before(index: pd.Index) -> Any:
    """
    A label for the point one bar before `index[0]`, or None when the index
    has no notion of "before" (strings, categories, an empty index).

    A dated index steps back by its own frequency when it has one -- so a
    business-day series starting on a Monday starts its curve on the
    Friday -- and otherwise by its first gap, or by one day when there is
    no gap to read. A numeric or period index steps back by one.
    """
    if len(index) == 0:
        return None
    first = index[0]
    if isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        step: Any = index.freq
        if step is None and len(index) >= 2 and index[1] > first:
            step = index[1] - first
        return first - (step if step is not None else pd.Timedelta(days=1))
    if isinstance(index, pd.PeriodIndex) or pd.api.types.is_numeric_dtype(index):
        return first - 1
    return None


def equity_from_returns(
    returns: Union[pd.Series, np.ndarray, Sequence[float]], start: float = 1.0
) -> Union[pd.Series, np.ndarray]:
    """
    The level curve a return series compounds into, STARTING AT `start`.

    `(1 + r).cumprod()` is the curve AFTER the first return. It never
    contains the level the first return was earned on, and every metric that
    reads a level series is anchored on its first point: cumulative_return
    and cagr divide by it, and max_drawdown and drawdown_series seed their
    running peak from it. Handed that curve, each of them dropped the first
    period's return. On r = [-0.20, 0 x 40, +0.30, +0.001 x 40] -- a true
    total return of +8.24%, a CAGR of 27.6% a year over 82 daily returns at
    252 a year, and a maximum drawdown of -20% -- they reported a CAGR of
    156% a year and no drawdown at all, and a Calmar ratio of infinity.

    This prepends `start`, so n returns give n + 1 levels. Every level
    metric is then right with no change of its own: the denominator is the
    level the first return was earned on, the running peak begins there,
    and the elapsed time cagr reads off the curve, (n + 1) - 1 periods, is
    the n periods the returns span.

    A Series comes back as a Series, with the start point labelled one bar
    before the first return (see `_label_before`) and, when the index has
    no notion of "before", on a positional index 0..n. An array, or a list,
    comes back as an array with the start prepended along its last axis, so
    a matrix of resampled paths, one per row, gets one start per row.

    Every return must be finite. A missing return has no level to compound
    into -- pandas' cumprod skips it and carries the previous level, which
    is the silent assumption that the period earned nothing -- so it is
    refused; drop the gap, or fill it with 0.0 if the position really sat
    in cash. `start` must be finite and strictly positive, because it is
    the denominator of everything derived from the curve.

    A return at or below -1 is a total loss and is compounded as given: the
    curve reaches zero or goes negative, which cagr and max_drawdown
    already report as a total loss.
    """
    where = "equity_from_returns"
    require_finite_scalar(start, "start", where)
    if not start > 0:
        raise ValidationError(
            f"{where}: start={start!r}. The starting level is the denominator "
            "of every return and drawdown read off the curve, so it must be "
            "strictly positive: 1.0 for a growth-of-one curve, or the "
            "initial capital."
        )
    start = float(start)

    if isinstance(returns, pd.Series):
        values = pd.to_numeric(returns, errors="coerce").astype("float64")
        missing = int(values.isna().sum())
        require_finite_series(values, "returns", where, allow_empty=True)
        if missing:
            labels = ", ".join(str(x) for x in values.index[values.isna()][:3])
            raise ValidationError(
                f"{where}: returns has {missing} missing value(s) (at {labels}"
                f"{', ...' if missing > 3 else ''}). A missing return has no "
                "level to compound into; carrying the previous level would "
                "silently assume the period earned nothing. Drop the gap, or "
                "fill it with 0.0 if the position really sat in cash."
            )
        grown = (1.0 + values).cumprod()
        if start != 1.0:
            grown = start * grown
        label = _label_before(values.index)
        if label is None:
            return pd.Series(
                np.concatenate([[start], grown.to_numpy()]), name=returns.name
            )
        return pd.Series(
            np.concatenate([[start], grown.to_numpy()]),
            index=values.index.insert(0, label),
            name=returns.name,
        )

    array = np.asarray(returns, dtype=np.float64)
    if array.ndim == 0:
        raise ValidationError(
            f"{where}: returns is a single number, not a series. Pass a "
            "sequence of per-period returns."
        )
    if not np.isfinite(array).all():
        raise ValidationError(
            f"{where}: returns has {int((~np.isfinite(array)).sum())} "
            "non-finite value(s). Every return is compounded into every "
            "later level, so one NaN or infinity poisons the rest of the "
            "curve. Drop or fill them first."
        )
    grown_array = np.cumprod(1.0 + array, axis=-1)
    if start != 1.0:
        grown_array = start * grown_array
    lead = np.full(array.shape[:-1] + (1,), start)
    return np.concatenate([lead, grown_array], axis=-1)


def cumulative_return(series: pd.Series) -> float:
    """
    Calculate Cumulative Return over an equity/level series.

    The first value is the denominator, so it must be strictly positive.
    Unchecked, a starting level of 0.0 divided by zero (emitting a
    RuntimeWarning and returning inf), and a negative starting level returned
    a sign-flipped result that looked entirely ordinary.

    Only the START is constrained. A terminal value at or below zero is a
    wiped-out position — a real outcome run_strategy can produce, since it
    applies no bankruptcy floor — and cagr() has documented handling for it.
    """
    if series.empty:
        logger.warning("[cumulative_return] empty series — returning 0.0")
        return 0.0
    require_positive_start_level(series, "series", "cumulative_return")
    return (series.iloc[-1] / series.iloc[0]) - 1


def cagr(series: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).

    A terminal value at or below zero (a leveraged position wiped out — an
    equity curve run_strategy can produce, since it applies no bankruptcy
    floor) has no real compound growth rate: (1 + total_ret) is <= 0, and
    raising it to a fractional power yields NaN plus a RuntimeWarning. That
    NaN then propagates silently into calmar_ratio and every metric built on
    it. Reported as -1.0 (-100%/yr, total loss) instead — the conventional
    representation of a wiped-out position, and a value downstream ratios can
    actually use.
    """
    if series.empty:
        logger.warning("[cagr] empty series — returning 0.0")
        return 0.0
    require_periods_per_year(periods_per_year, "cagr")

    total_ret = cumulative_return(series)
    # N observations span N-1 return INTERVALS, not N. Using len(series)
    # overstated the elapsed time and therefore understated the growth rate.
    # Negligible on a decade of daily bars, material on short windows: over 21
    # observations (one month) it is a 5% error in the exponent's denominator,
    # and the error grows as the window shortens — exactly where a CAGR is
    # already least reliable.
    num_years = (len(series) - 1) / periods_per_year

    if num_years <= 0:
        # A single observation spans no time at all, so it has no growth RATE.
        logger.warning(
            "[cagr] series spans no elapsed time (%d observation(s)) — "
            "returning 0.0",
            len(series),
        )
        return 0.0

    if total_ret <= -1.0:
        logger.warning(
            "[cagr] terminal value is non-positive (cumulative return %.4f) — "
            "position wiped out; reporting -1.0 (total loss) rather than NaN",
            total_ret,
        )
        return -1.0

    return (1 + total_ret) ** (1 / num_years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Annualized Volatility.

    periods_per_year is validated for the same reason it is everywhere else in
    this package: it is a bare multiplier, so a zero, negative or non-finite
    value produces a confidently wrong number rather than an error.
    """
    require_finite_series(returns, "returns", "annualized_volatility")
    require_periods_per_year(periods_per_year, "annualized_volatility")
    return returns.std() * np.sqrt(periods_per_year)
