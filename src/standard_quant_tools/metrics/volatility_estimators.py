import logging

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.validation import validate_series

logger = logging.getLogger(__name__)

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
