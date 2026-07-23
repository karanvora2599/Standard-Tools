import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)

# ── C++ extension (optional fast path) ───────────────────────────────────────

_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import _sqt_core as _cpp_core  # type: ignore[attr-defined]
    HAS_CPP = True
except ImportError:
    pass


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
                "1%":  float(raw["cv_1pct"]),
                "5%":  float(raw["cv_5pct"]),
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
        "1%":  float(crit_arr[0]),
        "5%":  float(crit_arr[1]),
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
    logger.debug("[cointegration] cointegrated=%s  p=%.4f  hedge=%.4f  half_life=%.1f days",
                 result["cointegrated"], float(p_val), hedge, hl)
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
