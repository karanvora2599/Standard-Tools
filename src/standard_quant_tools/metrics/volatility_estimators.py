import logging

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import (
    require_periods_per_year,
    require_positive_price_series,
)
from standard_quant_tools.validation import validate_series

logger = logging.getLogger(__name__)


def _validate_ohlc(
    func: str,
    periods_per_year: int,
    high=None,
    low=None,
    close=None,
    open_=None,
) -> None:
    """
    Shared OHLC contract for the range-based estimators.

    Every one of these takes a logarithm of a price ratio, so a non-positive
    price is not merely odd -- log(negative) is NaN, and the NaN then flows
    through the rolling window into the annualisation and out as a silently
    all-NaN volatility series. Measured before this guard: a single negative
    Low turned the whole Parkinson/Garman-Klass/Yang-Zhang output to NaN with
    nothing but a RuntimeWarning to say why.

    periods_per_year is validated for the reason it is validated everywhere
    else in this package: it is a bare multiplier under a square root, so a
    negative value produced sqrt of a negative and, again, a silently all-NaN
    series rather than an error.

    high >= low is checked too -- an inverted bar makes log(high/low)
    negative, and Parkinson's estimator squares it, so the result looks
    perfectly ordinary while describing a bar that cannot exist.
    """
    require_periods_per_year(periods_per_year, func)
    named = {"high": high, "low": low, "close": close, "open": open_}
    for name, series in named.items():
        if series is None:
            continue
        require_positive_price_series(series, name, func, allow_nan=True)
    if high is not None and low is not None:
        inverted = (high.notna() & low.notna()) & (high < low)
        if inverted.any():
            raise ValidationError(
                f"{func}: high is below low on {int(inverted.sum())} bar(s) "
                f"(first at {inverted[inverted].index[0]}). log(high/low) is "
                "then negative, and squaring it hides the inversion behind a "
                "perfectly ordinary-looking variance."
            )


_LOG_2 = float(np.log(2.0))


def _check_period(period: int) -> None:
    if period <= 1:
        raise ValidationError(
            f"period must be > 1 (need at least 2 bars for a rolling variance "
            f"estimate), got {period!r}"
        )


@validate_series()
def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Parkinson (1980) high-low range volatility estimator.

    Uses only the bar's own High/Low, ignoring Open/Close entirely -- more
    efficient than close-to-close volatility (no overnight-gap contamination
    to worry about), but blind to overnight gap risk itself since it never
    looks at the close. Assumes no drift and no overnight jumps, which is
    why Garman-Klass/Yang-Zhang exist as refinements.

    sigma^2 = mean(ln(H/L)^2) / (4 * ln 2), rolled over `period` bars.
    """
    _validate_ohlc("parkinson_volatility", periods_per_year, high=high, low=low)
    _check_period(period)
    log_hl2 = np.log(high / low) ** 2
    variance = log_hl2.rolling(period).mean() / (4.0 * _LOG_2)
    return np.sqrt(variance) * np.sqrt(periods_per_year)


@validate_series()
def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Garman-Klass (1980) OHLC volatility estimator.

    Adds the open-to-close term to Parkinson's high-low term, which uses the
    data more efficiently (lower estimator variance for the same window)
    but, like Parkinson, still assumes zero drift and no overnight jump --
    a large overnight gap inflates this estimator the same way it does
    close-to-close volatility. See yang_zhang_volatility for the version
    that isolates and accounts for the overnight component explicitly.

    per-bar term = 0.5*ln(H/L)^2 - (2*ln2 - 1)*ln(C/O)^2, rolled over
    `period` bars.
    """
    _validate_ohlc(
        "garman_klass_volatility",
        periods_per_year,
        high=high,
        low=low,
        close=close,
        open_=open_,
    )
    _check_period(period)
    log_hl2 = np.log(high / low) ** 2
    log_co2 = np.log(close / open_) ** 2
    per_bar = 0.5 * log_hl2 - (2.0 * _LOG_2 - 1.0) * log_co2
    variance = per_bar.rolling(period).mean()
    return np.sqrt(variance.clip(lower=0.0)) * np.sqrt(periods_per_year)


@validate_series()
def yang_zhang_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Yang-Zhang (2000) volatility estimator.

    Decomposes total variance into an overnight (close-to-open) component,
    an open-to-close component, and a drift-independent Rogers-Satchell
    component, then combines them with the minimum-variance weight k -- the
    only one of the three estimators here that explicitly accounts for
    overnight gap risk rather than being contaminated by or blind to it.
    Considered the most robust of the three for real equities, which
    routinely gap overnight.

    overnight_t   = ln(Open_t / Close_{t-1})
    open_close_t  = ln(Close_t / Open_t)
    rogers_satchell_t = ln(H_t/C_t)*ln(H_t/O_t) + ln(L_t/C_t)*ln(L_t/O_t)
    k = 0.34 / (1.34 + (period+1)/(period-1))
    sigma^2 = var(overnight) + k*var(open_close) + (1-k)*mean(rogers_satchell)
    """
    _validate_ohlc(
        "yang_zhang_volatility",
        periods_per_year,
        high=high,
        low=low,
        close=close,
        open_=open_,
    )
    _check_period(period)
    prev_close = close.shift(1)
    overnight = np.log(open_ / prev_close)
    open_close = np.log(close / open_)
    rogers_satchell = np.log(high / close) * np.log(high / open_) + np.log(
        low / close
    ) * np.log(low / open_)

    k = 0.34 / (1.34 + (period + 1) / (period - 1))

    var_overnight = overnight.rolling(period).var()
    var_open_close = open_close.rolling(period).var()
    mean_rs = rogers_satchell.rolling(period).mean()

    variance = var_overnight + k * var_open_close + (1.0 - k) * mean_rs
    return np.sqrt(variance.clip(lower=0.0)) * np.sqrt(periods_per_year)
