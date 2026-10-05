import logging
import math
from functools import lru_cache
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.error import ValidationError
from standard_quant_tools.validation import require_finite_array

logger = logging.getLogger(__name__)

#: A spread this small relative to the price level is zero. An exact affine
#: pair leaves a residual at float precision -- about 1e-13 of the series --
#: so the threshold sits well above that and far below any real spread. Two
#: series whose spread is a millionth of their price are the same series.
_DEGENERATE_RTOL = 1e-9

#: The fewest aligned observations a cointegration verdict is given on, for
#: `cointegration_test` and `scan_cointegrated_pairs` alike. The ADF
#: regression on the residual spends degrees of freedom on the intercept,
#: the lag and its augmentation lags, and MacKinnon's response surface is
#: fitted on samples of 20 and more; below that the single test answered
#: p=nan at n=0 and p=0.85 at n=8 as if either were a finding.
_MIN_COINT_OBS = 20


def _degenerate_pair_reason(a_vals: np.ndarray, b_vals: np.ndarray):
    """Why this pair has no cointegration question to answer, or None.

    ONE PREDICATE FOR EVERY PATH. `cointegration_test` raises on it and
    `scan_cointegrated_pairs` flags the row instead -- a screen over a
    hundred names must not die because two of them are the same listing --
    but they must agree on WHICH pairs are answerable, or the scan and the
    single test disagree about the same two series.

    Not `has_no_dispersion`: that asks whether a series is constant relative
    to its own magnitude, and a residual of 1e-14 varies hugely relative to
    itself while being zero relative to a price near 100. The comparison
    that matters is against the SERIES scale.

    The regression runs on one BLAS thread when the caller holds
    `single_threaded_blas()`, as both callers do.
    """
    if a_vals.size == 0:
        return None
    reason = _constant_reason(b_vals)
    if reason is not None:
        return reason
    return _affine_reason(
        a_vals,
        float(np.nanmax(np.abs(a_vals))),
        np.column_stack([np.ones(a_vals.size), b_vals]),
    )


def _constant_reason(b_vals: np.ndarray) -> Optional[str]:
    """The first half of `_degenerate_pair_reason`: series_b does not move.
    A property of series_b alone, so a screen asks it once per series."""
    b_scale = float(np.nanmax(np.abs(b_vals)))
    if b_scale <= 0 or float(np.ptp(b_vals)) <= b_scale * _DEGENERATE_RTOL:
        return (
            "series_b is constant, so there is no relationship to regress "
            "series_a onto -- the hedge ratio would be arbitrary and the "
            "spread would just be series_a. Cointegration is a statement "
            "about two series that both move."
        )
    return None


def _affine_reason(
    a_vals: np.ndarray, a_scale: float, design: np.ndarray
) -> Optional[str]:
    """The second half of `_degenerate_pair_reason`: series_a is an exact
    linear function of series_b. `a_scale` is the largest |series_a| and
    `design` is [1, series_b], both of which a screen computes once per
    series rather than once per pair."""
    beta, *_ = np.linalg.lstsq(design, a_vals, rcond=None)
    residual = a_vals - design @ beta
    if a_scale > 0 and float(np.ptp(residual)) <= a_scale * _DEGENERATE_RTOL:
        return (
            "the two series are an exact linear function of one another, so "
            "the spread is a constant and there is no unit root to test for. "
            "This is a dual listing, an ETF against its sole holding, or the "
            "same column twice in a universe -- not a tradeable "
            f"relationship (hedge ratio {float(beta[1]):.6g}). A p-value on "
            "it would be arithmetic on a zero residual, and the two backends "
            "disagreed about what to invent: the native kernel answered "
            "p=0.2593/not cointegrated where statsmodels answered "
            "p=0.0/cointegrated."
        )
    return None


# ── C++ extension (optional fast path) ───────────────────────────────────────

_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass

# ── numba (Kalman filter recursion is inherently sequential) ─────────────────

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(func):  # type: ignore[misc]
        return func


def _half_life_gate(spread: np.ndarray) -> Dict[str, Any]:
    """
    The Dickey-Fuller check on a fitted spread's half-life, as result keys.

    `half_life_statistics(pd.Series(spread), fitted_residual=True)`, read
    off the array. The spread of two aligned series has one value per date
    and no gap, so the label alignment inside `half_life` keeps every row
    where it is, and the arithmetic below is that function's to the bit.
    Through the Series it was 0.7 ms of a 1.1 ms test at 500 bars, most of
    it pandas aligning the spread with its own lag.
    """
    values = spread[~np.isnan(spread)]
    require_finite_array(values, "spread", "half_life_statistics")
    stats = _half_life_statistics(
        values, fitted_residual=True, half_life_of=lambda: _half_life_of(values)
    )
    return {
        "half_life_mean_reverting": stats["mean_reverting"],
        "half_life_t_statistic": stats["t_statistic"],
        "half_life_critical_value": stats["critical_value"],
    }


def _aligned_pair(
    series_a: pd.Series, series_b: pd.Series
) -> Tuple[pd.Index, np.ndarray, np.ndarray]:
    """
    The two series on the dates they share: (index, a values, b values).

    Two series on one unique index -- one date range from one provider,
    the usual case -- are read as they stand. That is what the
    intersection and the two label lookups below return for them: the
    intersection of two equal unique indexes is that index, and looking up
    every label of a unique index in its own order returns the series. The
    lookups were 0.15 ms of a 1.1 ms test at 500 bars. Each array is a
    fresh copy, as a lookup's is.
    """
    index_a, index_b = series_a.index, series_b.index
    if (
        (index_a is index_b or index_a.equals(index_b))
        and index_a.is_unique
        and index_b.is_unique
    ):
        return (
            index_a,
            series_a.to_numpy(dtype=float, copy=True),
            series_b.to_numpy(dtype=float, copy=True),
        )
    common_idx = index_a.intersection(index_b)
    return (
        common_idx,
        series_a.loc[common_idx].to_numpy(dtype=float),
        series_b.loc[common_idx].to_numpy(dtype=float),
    )


def cointegration_test(
    series_a: pd.Series,
    series_b: pd.Series,
    autolag: str = "aic",
) -> Dict[str, Any]:
    """
    Engle-Granger two-step cointegration test.

    Regresses series_a on series_b (OLS) to find the hedge ratio, then
    runs an ADF test on the spread (residuals). Uses MacKinnon (2010)
    p-values appropriate for cointegration residuals — stricter than
    standard ADF critical values.

    Parameters
    ----------
    series_a, series_b : pd.Series
        Price (or log-price) series. Index alignment is handled automatically.
    autolag : str
        Lag selection criterion passed to the internal ADF test.
        ``'aic'`` (default) or ``'bic'``.

    Returns
    -------
    dict with keys:
        cointegrated   : bool   – True when p_value < 0.05
        hedge_ratio    : float  – OLS coefficient (a ≈ alpha + hedge_ratio * b)
        adf_statistic  : float  – ADF t-statistic on the spread
        p_value        : float  – MacKinnon cointegration p-value
        critical_values: dict   – {"1%": ..., "5%": ..., "10%": ...}
        half_life_days : float  – AR(1) half-life of the spread in bars
        half_life_mean_reverting : bool – the half-life's own Dickey-Fuller
                         t-statistic clears the 5% Engle-Granger critical
                         value. A finite half-life without it is what a
                         random walk usually produces.
        half_life_t_statistic, half_life_critical_value : float
        n_obs          : int

    Raises ValidationError on fewer than 20 aligned observations.
    """
    # Validated rather than silently coerced: the C++ path below maps
    # anything that isn't exactly "bic" onto AIC, while the statsmodels
    # fallback passes the string straight through to coint(). A typo
    # therefore ran a DIFFERENT lag-selection criterion depending on whether
    # the extension was built, and echoed the typo back either way.
    if autolag.lower() not in ("aic", "bic"):
        raise ValidationError(f"autolag must be 'aic' or 'bic', got {autolag!r}")

    common_idx, a_vals, b_vals = _aligned_pair(series_a, series_b)
    require_finite_array(a_vals, "series_a", "cointegration_test")
    require_finite_array(b_vals, "series_b", "cointegration_test")
    n = len(a_vals)
    # Ahead of the degenerate-pair guard, which on one observation reported
    # "series_b is constant" -- true, and not the reason.
    if n < _MIN_COINT_OBS:
        raise ValidationError(
            f"cointegration_test: {n} aligned observation(s); a cointegration "
            f"verdict needs at least {_MIN_COINT_OBS}. Check that the two "
            "series share dates, or widen the date range."
        )
    path = "C++" if (HAS_CPP and _cpp_core is not None) else "statsmodels"
    logger.debug("[cointegration] n_obs=%d  autolag=%s  path=%s", n, autolag, path)

    # The guard's regression and the half-life gate's run on one BLAS
    # thread, so every number returned is the same bits whatever thread
    # count the caller's BLAS has. The gate's two sums of squares are dot
    # products, which OpenBLAS splits across threads above 10,000 terms.
    with single_threaded_blas():
        return _engle_granger(a_vals, b_vals, common_idx, autolag)


def _engle_granger(
    a_vals: np.ndarray, b_vals: np.ndarray, common_idx: pd.Index, autolag: str
) -> Dict[str, Any]:
    """`cointegration_test` on the aligned, checked values."""
    n = len(a_vals)
    # ── one guard, ahead of both backends ─────────────────────────────────────
    #
    # THE TWO PATHS RETURNED OPPOSITE VERDICTS HERE. On an exactly affine
    # pair -- a dual listing, an ETF against its sole holding, the same
    # column twice in a screening universe -- the residual is identically
    # zero, and measured on the same input:
    #
    #     native (C++)              p=0.2593  adf=-2.546   cointegrated=False
    #     statsmodels fallback      p=0.0     adf=-inf     cointegrated=True
    #
    # Neither is defensible. An ADF statistic asks whether a series reverts
    # to its mean; a series that IS its mean has no such question to answer,
    # and -2.546 and -inf are both inventions. statsmodels knows -- it emits
    # `CollinearityWarning: ... Cointegration test is not reliable in this
    # case` and returns a verdict over the top of it.
    reason = _degenerate_pair_reason(a_vals, b_vals)
    if reason is not None:
        raise ValidationError(f"cointegration_test: {reason}")

    # ── C++ fast path ─────────────────────────────────────────────────────────
    if HAS_CPP and _cpp_core is not None:
        use_aic = autolag.lower() != "bic"
        raw = _cpp_core.engle_granger(a_vals, b_vals, -1, use_aic)
        spread = a_vals - float(raw["intercept"]) - float(raw["hedge_ratio"]) * b_vals
        if len(common_idx) != n:
            # Dates that repeat in both series: each lookup returned every
            # row of each date, more rows than dates, and labelling the
            # spread with the dates refuses that, as it always has.
            pd.Series(spread, index=common_idx)
        return {
            "cointegrated": bool(raw["cointegrated"]),
            "hedge_ratio": float(raw["hedge_ratio"]),
            "adf_statistic": float(raw["adf_statistic"]),
            "p_value": float(raw["p_value"]),
            "critical_values": {
                "1%": float(raw["cv_1pct"]),
                "5%": float(raw["cv_5pct"]),
                "10%": float(raw["cv_10pct"]),
            },
            "half_life_days": float(raw["half_life"]),
            **_half_life_gate(spread),
            "n_obs": int(raw["n_obs"]),
        }

    # ── statsmodels fallback ──────────────────────────────────────────────────
    X = np.column_stack([np.ones(n), b_vals])
    beta, *_ = np.linalg.lstsq(X, a_vals, rcond=None)
    hedge = float(beta[1])

    adf_t, p_val, crit_arr = coint(a_vals, b_vals, trend="c", autolag=autolag)

    crit = {
        "1%": float(crit_arr[0]),
        "5%": float(crit_arr[1]),
        "10%": float(crit_arr[2]),
    }

    spread_values = a_vals - beta[0] - hedge * b_vals
    hl = half_life(pd.Series(spread_values, index=common_idx))

    result = {
        "cointegrated": bool(p_val < 0.05),
        "hedge_ratio": hedge,
        "adf_statistic": float(adf_t),
        "p_value": float(p_val),
        "critical_values": crit,
        "half_life_days": hl,
        **_half_life_gate(spread_values),
        "n_obs": n,
    }
    logger.debug(
        "[cointegration] cointegrated=%s  p=%.4f  hedge=%.4f  half_life=%.1f days",
        result["cointegrated"],
        float(p_val),
        hedge,
        hl,
    )
    return result


def compute_spread(
    series_a: pd.Series,
    series_b: pd.Series,
    hedge_ratio: Optional[float] = None,
) -> pd.Series:
    """
    Compute the spread between two series.

    spread = series_a - hedge_ratio * series_b

    If ``hedge_ratio`` is None, it is estimated via OLS so the spread is
    the cointegration residual (zero-mean by construction).

    Parameters
    ----------
    series_a, series_b : pd.Series
    hedge_ratio : float, optional
        Pass a known or previously estimated ratio to avoid re-fitting.

    Returns
    -------
    pd.Series aligned to the common index of the two inputs.
    """
    common_idx = series_a.index.intersection(series_b.index)
    a = series_a.loc[common_idx].to_numpy(dtype=float)
    b = series_b.loc[common_idx].to_numpy(dtype=float)
    require_finite_array(a, "series_a", "compute_spread")
    require_finite_array(b, "series_b", "compute_spread")

    if hedge_ratio is None:
        if HAS_CPP and _cpp_core is not None:
            r = _cpp_core.ols2(a, b)
            spread_vals = a - r["intercept"] - r["slope"] * b
        else:
            X = np.column_stack([np.ones(len(a)), b])
            with single_threaded_blas():
                beta, *_ = np.linalg.lstsq(X, a, rcond=None)
            spread_vals = a - beta[0] - beta[1] * b
    else:
        spread_vals = a - hedge_ratio * b

    return pd.Series(spread_vals, index=common_idx, name="spread")


def half_life(spread: pd.Series) -> float:
    """
    Estimate the mean-reversion half-life of a spread via AR(1) OLS.

    Fits: delta_S_t = alpha + beta * S_{t-1} + epsilon
    Half-life = -ln(2) / beta

    Returns ``float('inf')`` when beta >= 0 (spread is not mean-reverting).
    """
    delta = spread.diff().dropna()
    lag = spread.shift(1).dropna()
    common = delta.index.intersection(lag.index)

    y = delta.loc[common].to_numpy(dtype=float)
    x = lag.loc[common].to_numpy(dtype=float)
    return _ar1_half_life(y, x)


def _half_life_of(values: np.ndarray) -> float:
    """
    `half_life` of a spread with no missing value and one row per label,
    from its values alone.

    For such a spread the first difference and the lag are both labelled
    by every date but the first, so their intersection is that index and
    the two lookups keep every row in order: `y` is the first difference
    (pandas' `diff` is the same subtraction as numpy's) and `x` the values
    before the last.
    """
    return _ar1_half_life(np.diff(values), values[:-1])


def _ar1_half_life(y: np.ndarray, x: np.ndarray) -> float:
    """The half-life from a spread's first difference `y` and its lag `x`."""
    if len(y) < 3:
        return float("inf")

    require_finite_array(y, "spread", "half_life")
    require_finite_array(x, "spread", "half_life")

    if HAS_CPP and _cpp_core is not None:
        r = _cpp_core.ols2(y, x)
        ar_coeff = r["slope"]
    else:
        X = np.column_stack([np.ones(len(y)), x])
        with single_threaded_blas():
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        ar_coeff = float(beta[1])

    if ar_coeff >= 0:
        return float("inf")

    # THE DISCRETE AR(1) HALF-LIFE, not the continuous-time OU one.
    #
    # `ar_coeff` is b from regressing the spread's first difference on its
    # own lag, so the process is spread_t = (1 + b) * spread_{t-1} + e and
    # phi = 1 + b. A shock decays by a factor of phi each BAR, so it halves
    # after log(0.5) / log(phi) bars. `-log(2) / b` is the continuous-time
    # limit of that, exact only as b -> 0, and it is biased in one
    # direction -- always "reverts slower":
    #
    #     phi     true    -ln2/b    bias
    #     0.50    1.0000  1.3863    +38.63%
    #     0.80    3.1063  3.4657    +11.57%
    #     0.95   13.5134 13.8629     +2.59%
    #     0.99   68.9676 69.3147     +0.50%
    #
    # Checked directly at phi = 0.5: a shock of 1.0 reaches 0.5 after
    # exactly one bar. `agent/models.py` screens pairs on min_half_life=5
    # and max_half_life=126 and then RANKS on this number, so the bias
    # changed which pairs passed and in what order.
    phi = 1.0 + ar_coeff
    if phi == 0.0:
        # Reverts completely in a single bar.
        return 0.0
    if abs(phi) >= 1.0:
        # |phi| >= 1 does not decay: b < 0 with phi <= -1 is an explosive
        # oscillation, not fast mean reversion.
        return float("inf")
    # `abs` so an overshooting (negative phi) spread is measured on the
    # decay of its envelope rather than returning a complex log.
    return float(np.log(0.5) / np.log(abs(phi)))


def half_life_statistics(
    spread: pd.Series, *, fitted_residual: bool = False
) -> Dict[str, Any]:
    """
    The AR(1) half-life, with the test that says whether there is one.

    `half_life` returns a finite number whenever the fitted AR(1)
    coefficient is negative, and on a random walk it is negative about half
    the time and small the rest: measured on 1000 random walks of 250
    observations, 95.5% came back finite and 84% inside a 5-126 bar screen,
    median 36. A half-life is a claim that the spread mean-reverts, so it
    comes here with the Dickey-Fuller t-statistic of that same regression
    and the 5% critical value it has to clear -- which gates the random
    walks down to about 5%.

    `fitted_residual=True` when the spread is the residual of a regression
    of one price on another (an Engle-Granger spread): the regression has
    already chosen the most stationary-looking combination, so the critical
    value is MacKinnon's for two variables (about -3.36 at 250
    observations) rather than the one-series -2.87.

    Returns a dict with `half_life` (bars; inf when the coefficient is not
    negative, as `half_life`), `ar_coefficient`, `t_statistic`,
    `critical_value`, `mean_reverting` (the t-statistic clears the 5%
    critical value and the half-life is finite) and `n_obs`.

    The regression's products run on one BLAS thread, so the t-statistic
    is the same bits at any thread count: its two sums of squares are dot
    products, which OpenBLAS splits across threads above 10,000 terms.
    """
    clean = spread.dropna()
    values = clean.to_numpy(dtype=float)
    require_finite_array(values, "spread", "half_life_statistics")
    return _half_life_statistics(
        values, fitted_residual=fitted_residual, half_life_of=lambda: half_life(clean)
    )


@lru_cache(maxsize=1024)
def _mackinnon_5pct(n_series: int, nobs: int) -> float:
    """MacKinnon's (2010) 5% critical value with a constant, for `n_series`
    series and `nobs` observations. A polynomial in 1/nobs that cost 7 us a
    call; a pair screen asks for the same few sample sizes again and
    again."""
    from statsmodels.tsa.adfvalues import mackinnoncrit

    return float(mackinnoncrit(N=n_series, regression="c", nobs=nobs)[1])


def _no_dispersion(values: np.ndarray) -> bool:
    """
    `has_no_dispersion(values)` for finite values, without the standard
    deviation where it cannot change the answer.

    That test reads the standard deviation only to call a series flat when
    it is zero or not finite, and then compares the range with the largest
    magnitude. For finite values whose largest magnitude lies between
    1e-100 and 1e100 the standard deviation is finite (no sum or square of
    them overflows), and it is zero only when every value is the same:
    two different values of that size are at least 1e-117 apart, so some
    deviation from their mean squares to a normal number, not to zero. A
    range of zero is flat by the comparison too, so there the comparison
    alone is the test's answer. Outside that band, the test itself. The
    standard deviation was 38 us of the half-life gate at 500 bars.
    """
    # Imported here: `metrics` imports `analysis` at package level, so a
    # module-level import would close a cycle.
    from standard_quant_tools.metrics.risk_metrics import (
        DISPERSION_RTOL,
        has_no_dispersion,
    )

    if values.ndim == 1 and values.size >= 2:
        scale = float(np.max(np.abs(values)))
        if 1e-100 <= scale <= 1e100:
            return float(np.ptp(values)) <= scale * DISPERSION_RTOL
    return has_no_dispersion(values)


def _half_life_statistics(
    values: np.ndarray,
    *,
    fitted_residual: bool,
    half_life_of: Callable[[], float],
) -> Dict[str, Any]:
    """`half_life_statistics` on the spread's finite values, with
    `half_life_of()` giving its half-life."""
    n = int(values.size) - 1
    nan = float("nan")
    result: Dict[str, Any] = {
        "half_life": float("inf"),
        "ar_coefficient": nan,
        "t_statistic": nan,
        "critical_value": nan,
        "mean_reverting": False,
        "n_obs": max(n, 0),
    }
    if n < 3 or _no_dispersion(values):
        # Too short to fit, or a constant: no reversion to measure.
        return result

    y = np.diff(values)
    x = values[:-1]
    design = np.column_stack([np.ones(n), x])
    dof = n - 2
    x_centred = x - x.mean()
    with single_threaded_blas():
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        s2 = float(residual @ residual) / dof if dof > 0 else nan
        sxx = float(x_centred @ x_centred)
    se = math.sqrt(s2 / sxx) if sxx > 0 and s2 == s2 else nan
    ar_coeff = float(beta[1])
    t_stat = ar_coeff / se if se == se and se > 0 else nan
    critical = _mackinnon_5pct(2 if fitted_residual else 1, n)
    hl = half_life_of()
    result.update(
        {
            "half_life": float(hl),
            "ar_coefficient": ar_coeff,
            "t_statistic": float(t_stat),
            "critical_value": critical,
            "mean_reverting": bool(
                t_stat == t_stat and t_stat < critical and math.isfinite(hl)
            ),
        }
    )
    return result


def spread_zscore(
    spread: pd.Series,
    window: Optional[int] = None,
) -> pd.Series:
    """
    Standardise a spread series into a z-score.

    Parameters
    ----------
    spread : pd.Series
    window : int, optional
        Rolling lookback. If None, uses the full-sample mean and std
        (static normalisation) — this uses the ENTIRE series' mean/std at
        every point, including bars in the future relative to any given
        row, so it must not be used to generate historical trading signals
        for a backtest (look-ahead bias). A rolling window of 20-60 bars is
        typical for live trading signals and backtests.

    Returns
    -------
    pd.Series with the same index as ``spread``.
    """
    # Imported here: `metrics` imports `analysis` at package level, so a
    # module-level import would close a cycle.
    from standard_quant_tools.metrics.risk_metrics import (
        DISPERSION_RTOL,
        has_no_dispersion,
    )

    if window is None:
        mu = spread.mean()
        sigma = spread.std()
        # The library's relative test, not `sigma == 0`. A spread flat at a
        # level like 12.3456 bps has a standard deviation of 7e-15, not 0,
        # and the exact test divided rounding residue by rounding residue:
        # a z-score of 0.99 on a series that never moved, ranked first in a
        # basis scan. A flat spread keeps its 0.0 convention.
        values = spread.dropna().to_numpy(dtype=float)
        if sigma == 0 or (values.size >= 2 and has_no_dispersion(values)):
            return pd.Series(0.0, index=spread.index, name="zscore")
        return ((spread - mu) / sigma).rename("zscore")

    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    # A window with no dispersion (e.g. a constant spread) would otherwise
    # divide by zero or by rounding residue -- NaN out that bar rather than
    # an inf or a residue ratio being mistaken for a real z-score. Relative
    # to the window's own magnitude, as `has_no_dispersion` is.
    window_range = spread.rolling(window).max() - spread.rolling(window).min()
    window_scale = spread.abs().rolling(window).max()
    flat = (rolling_std <= 0) | (window_range <= window_scale * DISPERSION_RTOL)
    safe_std = rolling_std.where(~flat)
    return ((spread - rolling_mean) / safe_std).rename("zscore")


# ── Kalman-filter dynamic hedge ratio ────────────────────────────────────────
#
# cointegration_test's hedge_ratio is a single static OLS coefficient fit
# once over the whole window — the standard starting point, but it can go
# stale as the true relationship drifts. The Kalman filter below treats the
# hedge ratio as a hidden state that follows a random walk and re-estimates
# it every bar via the standard predict/update recursion (see e.g. Chan,
# "Algorithmic Trading", ch. 3, for this exact parametrization). It's a
# diagnostic companion to cointegration_test, not a replacement — and it is
# NOT wired into backtest/pairs.py's run_pair_backtest, which takes a single
# static float hedge ratio for the whole backtest window; feeding it a
# time-varying ratio would be a real follow-up to that engine, not this
# module.
#
# The recursion is inherently sequential (state at t depends on state at
# t-1), so it's numba-@njit'd rather than vectorized — same tool this
# codebase already uses for backtest/strategies.py's state-machine loops.
# Two separate kernels (1-state / 2-state) rather than one branching kernel,
# matching strategies.py's precedent of one njit function per state machine
# instead of a single parametrized one.

_KALMAN_PRIOR_VARIANCE = 1.0e4


@njit
def _kalman_filter_1state(
    y: np.ndarray, x: np.ndarray, delta: float, observation_noise: float
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    n = len(y)
    beta_path = np.empty(n)
    gain_path = np.empty(n)
    innovation_path = np.empty(n)

    vw = delta / (1.0 - delta)
    beta_prev = 0.0
    p_prev = _KALMAN_PRIOR_VARIANCE

    for t in range(n):
        r = p_prev + vw
        y_hat = beta_prev * x[t]
        q = r * x[t] * x[t] + observation_noise
        e = y[t] - y_hat
        k = r * x[t] / q

        beta_t = beta_prev + k * e
        p_t = r - k * x[t] * r

        beta_path[t] = beta_t
        gain_path[t] = k
        innovation_path[t] = e

        beta_prev = beta_t
        p_prev = p_t

    return beta_path, gain_path, innovation_path


@njit
def _kalman_filter_2state(
    y: np.ndarray, x: np.ndarray, delta: float, observation_noise: float
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    n = len(y)
    alpha_path = np.empty(n)
    beta_path = np.empty(n)
    gain_path = np.empty(n)
    innovation_path = np.empty(n)

    vw = delta / (1.0 - delta)
    alpha_prev = 0.0
    beta_prev = 0.0
    p00, p01, p11 = _KALMAN_PRIOR_VARIANCE, 0.0, _KALMAN_PRIOR_VARIANCE

    for t in range(n):
        r00 = p00 + vw
        r01 = p01
        r11 = p11 + vw

        xt = x[t]
        q = r00 + 2.0 * r01 * xt + r11 * xt * xt + observation_noise
        e = y[t] - (alpha_prev + beta_prev * xt)

        rx0 = r00 + r01 * xt
        rx1 = r01 + r11 * xt
        k0 = rx0 / q
        k1 = rx1 / q

        alpha_t = alpha_prev + k0 * e
        beta_t = beta_prev + k1 * e

        p00_t = r00 - k0 * rx0
        p01_t = r01 - k0 * rx1
        p11_t = r11 - k1 * rx1

        alpha_path[t] = alpha_t
        beta_path[t] = beta_t
        gain_path[t] = k1
        innovation_path[t] = e

        alpha_prev, beta_prev = alpha_t, beta_t
        p00, p01, p11 = p00_t, p01_t, p11_t

    return alpha_path, beta_path, gain_path, innovation_path


_BATCH_COINT_COLUMNS = [
    "intercept",
    "hedge_ratio",
    "adf_statistic",
    "optimal_lag",
    "p_value",
    "cv_1pct",
    "cv_5pct",
    "cv_10pct",
    "half_life_days",
    "n_obs",
    "cointegrated",
]

#: What a screen adds to the per-pair test: the verdict with the pair's
#: order reversed, one p-value for both orders, and the multiple-testing
#: correction a screen over many pairs needs.
_SCAN_COLUMNS = _BATCH_COINT_COLUMNS + [
    "p_value_reverse",
    "p_value_both",
    "direction_consistent",
    "p_value_bh",
    "cointegrated_fdr",
]


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """
    Benjamini-Hochberg adjusted p-values (the step-up false discovery rate).

    A NaN p-value is not a test: it is left NaN and not counted in the
    number of tests. Reject at FDR level q where the adjusted value is at
    most q. Valid under independence and positive dependence of the tests;
    Benjamini-Yekutieli (multiply by sum 1/i) is the conservative option
    under arbitrary dependence.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    mask = np.isfinite(p)
    m = int(mask.sum())
    if m == 0:
        return out
    values = p[mask]
    order = np.argsort(values, kind="stable")
    scaled = values[order] * m / np.arange(1, m + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(scaled, 1.0)
    out[mask] = adjusted
    return out


def _degenerate_pairs(
    frame: pd.DataFrame, pair_list: Sequence[Tuple[str, str]]
) -> List[bool]:
    """
    Whether `_degenerate_pair_reason` refuses each pair of a screen.

    The same predicate, with what depends on one series alone -- its
    values, whether it is constant, its largest magnitude and the design
    [1, series_b] -- worked out once per series rather than once per pair,
    and every regression on one BLAS thread. A pair's own work is one
    least-squares fit and the range of its residual.
    """
    columns: Dict[str, np.ndarray] = {}

    def column(name: str) -> np.ndarray:
        if name not in columns:
            columns[name] = frame[name].to_numpy(dtype=float)
        return columns[name]

    constant: Dict[str, bool] = {}
    scales: Dict[str, float] = {}
    designs: Dict[str, np.ndarray] = {}
    flags: List[bool] = []
    with single_threaded_blas():
        for a, b in pair_list:
            a_vals, b_vals = column(a), column(b)
            if a_vals.ndim != 1 or b_vals.shape != a_vals.shape:
                # Not two plain columns (a duplicated name selects a frame):
                # the predicate as it stands, whatever it makes of them.
                flags.append(_degenerate_pair_reason(a_vals, b_vals) is not None)
                continue
            if a_vals.size == 0:
                flags.append(False)
                continue
            if b not in constant:
                constant[b] = _constant_reason(b_vals) is not None
            if constant[b]:
                flags.append(True)
                continue
            if a not in scales:
                scales[a] = float(np.nanmax(np.abs(a_vals)))
            if b not in designs:
                designs[b] = np.column_stack([np.ones(b_vals.size), b_vals])
            flags.append(_affine_reason(a_vals, scales[a], designs[b]) is not None)
    return flags


def scan_cointegrated_pairs(
    prices: Union[pd.DataFrame, Dict[str, pd.Series]],
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    autolag: str = "aic",
    max_lag: int = -1,
    fdr: float = 0.05,
) -> pd.DataFrame:
    """
    Engle-Granger over many pairs in ONE native call.

    A pair screen is O(N^2) in the universe: 2,000 tickers is 1,999,000 pairs.
    Driving that from Python -- ``for a, b in combinations(tickers, 2)``
    calling :func:`cointegration_test` per pair -- pays the pandas round trip
    two million times and uses one core. Measured at 2,000 bars that was 9.8
    hours for one order of each pair; this path tests both orders of every
    pair in 1.7 minutes at 500 bars and 19 at 2,000 on a 16-thread laptop.

    Every series is aligned onto ONE common index before the panel is built,
    which is the one semantic difference from looping :func:`cointegration_test`
    (that aligns each pair against only its own partner). When every series
    already shares an index -- the usual case for same-exchange equities over
    one date range -- the two are identical. When they do not, a common sample
    is arguably the better basis for a screen anyway, because p-values across
    pairs are then comparable; either way it is stated here rather than
    discovered.

    THE VERDICT DEPENDS ON WHICH SERIES IS REGRESSED ON WHICH. Engle-Granger
    is not symmetric: on 24 random walks, 65 of 276 verdicts flipped when
    the columns were swapped. So every pair is also tested reversed, in the
    same native call, and `p_value_both` is the larger of the two p-values
    -- a pair is cointegrated in both orders or it is not cointegrated.
    `direction_consistent` says whether the two orders agree at 5%.

    A SCREEN IS MANY TESTS. At 5%, 276 unrelated pairs produce about 14
    "cointegrated" rows by chance. `p_value_bh` is the Benjamini-Hochberg
    adjustment of `p_value_both` across every answerable pair, and
    `cointegrated_fdr` is it at most `fdr`: the rows that survive a false
    discovery rate of `fdr`. Benjamini-Yekutieli is the conservative
    alternative when the pairs are strongly dependent (they share series);
    `benjamini_hochberg` is exported for recomputing either.

    Args:
        prices: Wide DataFrame (columns = tickers) or dict of ticker -> Series.
        pairs: Which pairs to test. Defaults to every unordered combination.
        autolag: "aic" (default) or "bic".
        max_lag: ADF max lag; -1 for the automatic Schwert rule.
        fdr: False discovery rate for `cointegrated_fdr` (default 0.05).

    Returns:
        DataFrame indexed by a MultiIndex of (symbol_a, symbol_b), with columns
        intercept, hedge_ratio, adf_statistic, optimal_lag, p_value, cv_1pct,
        cv_5pct, cv_10pct, half_life_days, n_obs, cointegrated (all for
        symbol_a regressed on symbol_b), then p_value_reverse, p_value_both,
        direction_consistent, p_value_bh and cointegrated_fdr.

    Raises:
        ValidationError: on an unknown autolag, an fdr outside (0, 1), an
            empty universe, a pair naming a ticker not in `prices`, or fewer
            than 20 aligned bars.
    """
    if autolag.lower() not in ("aic", "bic"):
        raise ValidationError(f"autolag must be 'aic' or 'bic', got {autolag!r}")
    if not (0.0 < float(fdr) < 1.0):
        raise ValidationError(f"fdr must be strictly between 0 and 1, got {fdr!r}")

    frame = prices if isinstance(prices, pd.DataFrame) else pd.DataFrame(prices)
    frame = frame.dropna(how="any")
    tickers = [str(c) for c in frame.columns]
    if len(tickers) < 2:
        raise ValidationError(
            f"scan_cointegrated_pairs: need at least 2 series, got {len(tickers)}"
        )
    if len(frame) < _MIN_COINT_OBS:
        raise ValidationError(
            f"scan_cointegrated_pairs: need at least {_MIN_COINT_OBS} aligned "
            f"bars, got {len(frame)}. Rows with a gap in any series are "
            "dropped before the scan, so one short history shortens them all."
        )

    pos = {t: i for i, t in enumerate(tickers)}
    if pairs is None:
        pair_list = list(combinations(tickers, 2))
    else:
        pair_list = [(str(a), str(b)) for a, b in pairs]
        missing = sorted({t for pr in pair_list for t in pr if t not in pos})
        if missing:
            raise ValidationError(
                f"scan_cointegrated_pairs: pairs reference unknown ticker(s) {missing}"
            )
    if not pair_list:
        return pd.DataFrame(
            columns=_SCAN_COLUMNS,
            index=pd.MultiIndex.from_tuples([], names=["symbol_a", "symbol_b"]),
        )

    index = pd.MultiIndex.from_tuples(pair_list, names=["symbol_a", "symbol_b"])
    use_aic = autolag.lower() != "bic"
    logger.debug(
        "[scan_pairs] universe=%d  pairs=%d  bars=%d  path=%s",
        len(tickers),
        len(pair_list),
        len(frame),
        "C++" if (HAS_CPP and _cpp_core is not None) else "python-loop",
    )

    # A SCAN FLAGS WHAT A SINGLE TEST REFUSES. `cointegration_test` raises on
    # a degenerate pair, which is right for one question and wrong for a
    # hundred: a universe containing one dual listing must not lose the other
    # 4,949 pairs. Same predicate either way, so the scan and the single test
    # never disagree about which pairs are answerable -- and it runs on BOTH
    # backends, because the kernel does not see the guard above.
    degenerate = _degenerate_pairs(frame, pair_list)

    def _blank_row(n_obs: int):
        nan = float("nan")
        return [nan, nan, nan, 0, nan, nan, nan, nan, nan, n_obs, False]

    reversed_pairs = [(b, a) for a, b in pair_list]
    m = len(pair_list)
    if HAS_CPP and _cpp_core is not None:
        # (n_tickers x n_bars), the layout the kernel indexes by row. Both
        # orders of every pair in the one call.
        panel = np.ascontiguousarray(frame.to_numpy(dtype=np.float64).T)
        pair_idx = np.array(
            [(pos[a], pos[b]) for a, b in pair_list + reversed_pairs],
            dtype=np.int32,
        )
        out = np.asarray(
            _cpp_core.batch_engle_granger(panel, pair_idx, max_lag, use_aic)
        )
        df = pd.DataFrame(out[:m], columns=_BATCH_COINT_COLUMNS, index=index)
        p_reverse = out[m:, _BATCH_COINT_COLUMNS.index("p_value")].astype(float)
        for position in range(m):
            if degenerate[position]:
                df.iloc[position] = _blank_row(int(df.iloc[position]["n_obs"]))
    else:
        # Pure-Python fallback: same columns, same order, one pair at a
        # time. The intercept is the OLS identity on the aligned rows (the
        # hedge ratio IS the OLS slope with an intercept, so a - slope * b at
        # the means is that intercept). statsmodels' `coint` does not report
        # the lag it chose, so `optimal_lag` is -1 here: unknown, not zero.
        rows = []
        p_reverse = np.full(m, np.nan)
        for position, (a, b) in enumerate(pair_list):
            if degenerate[position]:
                rows.append(_blank_row(len(frame)))
                continue
            r = cointegration_test(frame[a], frame[b], autolag=autolag)
            p_reverse[position] = cointegration_test(
                frame[b], frame[a], autolag=autolag
            )["p_value"]
            aligned = frame[[a, b]].dropna()
            intercept = float(aligned[a].mean() - r["hedge_ratio"] * aligned[b].mean())
            rows.append(
                [
                    intercept,
                    r["hedge_ratio"],
                    r["adf_statistic"],
                    -1,
                    r["p_value"],
                    r["critical_values"]["1%"],
                    r["critical_values"]["5%"],
                    r["critical_values"]["10%"],
                    r["half_life_days"],
                    r["n_obs"],
                    r["cointegrated"],
                ]
            )
        df = pd.DataFrame(rows, columns=_BATCH_COINT_COLUMNS, index=index)

    df["optimal_lag"] = df["optimal_lag"].astype(int)
    df["n_obs"] = df["n_obs"].astype(int)
    df["cointegrated"] = df["cointegrated"].astype(bool)

    p_forward = df["p_value"].to_numpy(dtype=float)
    p_reverse = np.where(degenerate, np.nan, p_reverse)
    p_both = np.fmax(p_forward, p_reverse)
    p_both = np.where(np.isfinite(p_forward) & np.isfinite(p_reverse), p_both, np.nan)
    answerable = np.isfinite(p_both)
    df["p_value_reverse"] = p_reverse
    df["p_value_both"] = p_both
    df["direction_consistent"] = answerable & ((p_forward < 0.05) == (p_reverse < 0.05))
    p_bh = benjamini_hochberg(p_both)
    df["p_value_bh"] = p_bh
    df["cointegrated_fdr"] = answerable & (np.nan_to_num(p_bh, nan=1.0) <= fdr)
    df["direction_consistent"] = df["direction_consistent"].astype(bool)
    df["cointegrated_fdr"] = df["cointegrated_fdr"].astype(bool)
    return df


def kalman_hedge_ratio(
    series_a: pd.Series,
    series_b: pd.Series,
    delta: float = 1e-4,
    observation_noise: float = 1e-3,
    include_intercept: bool = True,
) -> pd.DataFrame:
    """
    Time-varying hedge ratio between two price series via a Kalman filter.

    Models series_a[t] = intercept[t] + beta[t] * series_b[t] + noise, with
    beta[t] (and intercept[t], if include_intercept) following a random
    walk. Unlike cointegration_test's single static OLS hedge_ratio, this
    re-estimates the ratio every bar — useful as a staleness diagnostic on
    an existing pairs relationship, or to see how much a static ratio would
    have drifted over the window.

    Parameters
    ----------
    series_a, series_b : pd.Series
        Price (or log-price) series, same convention as cointegration_test.
        Index alignment is handled automatically.
    delta : float
        The one tuning knob (standard in the Kalman pairs-trading
        literature): controls how fast the hedge ratio is allowed to drift.
        Smaller = slower-adapting / more stable (closer to a static OLS
        ratio); larger = faster-adapting / noisier. Must be in (0, 1).
    observation_noise : float
        Assumed variance of the observation noise (spread noise). Larger
        values make the filter trust new observations less.
    include_intercept : bool
        If True (default), fits both an intercept and a slope (2-state
        filter). If False, fits slope only (1-state filter, intercept
        forced to 0).

    Returns
    -------
    pd.DataFrame indexed on the common index of series_a/series_b, columns:
        Hedge_Ratio : beta[t]
        Intercept   : intercept[t] (all zero if include_intercept=False)
        Spread      : series_a - Hedge_Ratio*series_b - Intercept
        Kalman_Gain : the slope's Kalman gain at each step (diagnostic —
                      near-zero means the filter has stopped reacting to
                      new observations)
    """
    if not (0.0 < delta < 1.0):
        raise ValidationError(f"delta must be in (0, 1), got {delta}")
    if observation_noise <= 0:
        raise ValidationError(f"observation_noise must be > 0, got {observation_noise}")

    common_idx = series_a.index.intersection(series_b.index)
    a = series_a.loc[common_idx].to_numpy(dtype=float)
    b = series_b.loc[common_idx].to_numpy(dtype=float)
    # Consistent with cointegration_test/compute_spread/half_life above: a
    # NaN/Inf here silently poisons every subsequent state of the sequential
    # filter recursion, with no check anywhere in the numba or native kernel.
    require_finite_array(a, "series_a", "kalman_hedge_ratio")
    require_finite_array(b, "series_b", "kalman_hedge_ratio")
    n = len(a)
    if n < 3:
        raise ValidationError(
            f"kalman_hedge_ratio needs at least 3 aligned observations, got {n}"
        )

    path = "2-state" if include_intercept else "1-state"
    logger.debug(
        "[kalman_hedge_ratio] n_obs=%d  delta=%.2e  observation_noise=%.2e  path=%s",
        n,
        delta,
        observation_noise,
        path,
    )

    if HAS_CPP and _cpp_core is not None:
        if include_intercept:
            r = _cpp_core.kalman_filter_2state(a, b, delta, observation_noise)
            alpha_path, beta_path, gain_path = r["alpha"], r["beta"], r["gain"]
        else:
            r = _cpp_core.kalman_filter_1state(a, b, delta, observation_noise)
            beta_path, gain_path = r["beta"], r["gain"]
            alpha_path = np.zeros(n)
    elif include_intercept:
        alpha_path, beta_path, gain_path, _ = _kalman_filter_2state(
            a, b, delta, observation_noise
        )
    else:
        beta_path, gain_path, _ = _kalman_filter_1state(a, b, delta, observation_noise)
        alpha_path = np.zeros(n)

    spread = a - beta_path * b - alpha_path

    return pd.DataFrame(
        {
            "Hedge_Ratio": beta_path,
            "Intercept": alpha_path,
            "Spread": spread,
            "Kalman_Gain": gain_path,
        },
        index=common_idx,
    )
