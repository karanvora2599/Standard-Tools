import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

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
        n_obs          : int
    """
    common_idx = series_a.index.intersection(series_b.index)
    a = series_a.loc[common_idx]
    b = series_b.loc[common_idx]
    a_vals = a.to_numpy(dtype=float)
    b_vals = b.to_numpy(dtype=float)
    n = len(a_vals)
    path = "C++" if (HAS_CPP and _cpp_core is not None) else "statsmodels"
    logger.debug("[cointegration] n_obs=%d  autolag=%s  path=%s", n, autolag, path)

    # ── C++ fast path ─────────────────────────────────────────────────────────
    if HAS_CPP and _cpp_core is not None:
        use_aic = autolag.lower() != "bic"
        raw = _cpp_core.engle_granger(a_vals, b_vals, -1, use_aic)
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

    spread = pd.Series(a_vals - beta[0] - hedge * b_vals, index=common_idx)
    hl = half_life(spread)

    result = {
        "cointegrated": bool(p_val < 0.05),
        "hedge_ratio": hedge,
        "adf_statistic": float(adf_t),
        "p_value": float(p_val),
        "critical_values": crit,
        "half_life_days": hl,
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

    if hedge_ratio is None:
        if HAS_CPP and _cpp_core is not None:
            r = _cpp_core.ols2(a, b)
            spread_vals = a - r["intercept"] - r["slope"] * b
        else:
            X = np.column_stack([np.ones(len(a)), b])
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

    if len(y) < 3:
        return float("inf")

    if HAS_CPP and _cpp_core is not None:
        r = _cpp_core.ols2(y, x)
        ar_coeff = r["slope"]
    else:
        X = np.column_stack([np.ones(len(y)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        ar_coeff = float(beta[1])

    if ar_coeff >= 0:
        return float("inf")

    return float(-np.log(2) / ar_coeff)


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
    if window is None:
        mu = spread.mean()
        sigma = spread.std()
        if sigma == 0:
            return pd.Series(0.0, index=spread.index, name="zscore")
        return ((spread - mu) / sigma).rename("zscore")

    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    return ((spread - rolling_mean) / rolling_std).rename("zscore")


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
        raise ValidationError(
            f"observation_noise must be > 0, got {observation_noise}"
        )

    common_idx = series_a.index.intersection(series_b.index)
    a = series_a.loc[common_idx].to_numpy(dtype=float)
    b = series_b.loc[common_idx].to_numpy(dtype=float)
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

    if include_intercept:
        alpha_path, beta_path, gain_path, _ = _kalman_filter_2state(
            a, b, delta, observation_noise
        )
    else:
        beta_path, gain_path, _ = _kalman_filter_1state(
            a, b, delta, observation_noise
        )
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
