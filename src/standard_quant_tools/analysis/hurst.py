from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

# Regime thresholds — buffer of ±0.05 around the random-walk boundary (0.5)
_TRENDING_THRESHOLD = 0.55
_MEAN_REVERTING_THRESHOLD = 0.45


def _classify(h: float) -> str:
    if h > _TRENDING_THRESHOLD:
        return "trending"
    if h < _MEAN_REVERTING_THRESHOLD:
        return "mean_reverting"
    return "random_walk"


def _log_sizes(min_w: int, max_w: int, n_points: int = 20) -> np.ndarray:
    """Return an array of unique integer window sizes, log-spaced."""
    sizes = np.unique(
        np.logspace(np.log10(min_w), np.log10(max_w), n_points).astype(int)
    )
    return sizes[(sizes >= min_w) & (sizes <= max_w)]


def _dfa(arr: np.ndarray, min_w: int, max_w: int) -> tuple:
    """
    Detrended Fluctuation Analysis.

    Integrates the mean-centred series then measures how RMS residuals
    (after linear detrending within each box) scale with box size.

    Returns (sizes, fluctuations) arrays for the log-log OLS fit.
    """
    y = np.cumsum(arr - arr.mean())
    n = len(y)
    sizes = _log_sizes(min_w, max_w)

    fluctuations, valid = [], []
    for sz in sizes:
        n_chunks = n // sz
        if n_chunks < 2:
            continue
        x = np.arange(sz, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).mean()
        rms_acc = 0.0
        for i in range(n_chunks):
            seg = y[i * sz: (i + 1) * sz]
            seg_mean = seg.mean()
            # Analytic linear detrend (faster than np.polyfit)
            b = ((x - x_mean) * (seg - seg_mean)).mean() / x_var if x_var > 0 else 0.0
            a = seg_mean - b * x_mean
            residuals = seg - (a + b * x)
            rms_acc += (residuals ** 2).mean()
        fluctuations.append(np.sqrt(rms_acc / n_chunks))
        valid.append(sz)

    return np.array(valid, dtype=float), np.array(fluctuations)


def _rs(arr: np.ndarray, min_w: int, max_w: int) -> tuple:
    """
    Classic Rescaled Range (R/S) analysis.

    Known to be biased upward for short series — prefer DFA for most uses.
    Returns (sizes, rs_values) arrays for the log-log OLS fit.
    """
    n = len(arr)
    sizes = _log_sizes(min_w, max_w)

    rs_vals, valid = [], []
    for sz in sizes:
        n_chunks = n // sz
        if n_chunks < 1:
            continue
        rs_acc = 0.0
        count = 0
        for i in range(n_chunks):
            chunk = arr[i * sz: (i + 1) * sz]
            mad = chunk - chunk.mean()
            cum = np.cumsum(mad)
            R = cum.max() - cum.min()
            S = chunk.std(ddof=1)
            if S > 0:
                rs_acc += R / S
                count += 1
        if count > 0:
            rs_vals.append(rs_acc / count)
            valid.append(sz)

    return np.array(valid, dtype=float), np.array(rs_vals)


def _ols_slope_r2(log_n: np.ndarray, log_f: np.ndarray):
    """Return (slope, R²) of a log-log OLS fit."""
    X = np.column_stack([np.ones(len(log_n)), log_n])
    beta, *_ = np.linalg.lstsq(X, log_f, rcond=None)
    slope = float(beta[1])
    y_pred = X @ beta
    ss_res = float(np.sum((log_f - y_pred) ** 2))
    ss_tot = float(np.sum((log_f - log_f.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, r2


def hurst_exponent(
    series: pd.Series,
    method: Literal["dfa", "rs"] = "dfa",
    min_window: int = 10,
    max_window: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate the Hurst exponent of a return series.

    The Hurst exponent H characterises the long-memory scaling of a time series:

    * H > 0.55 — **trending** (persistent): recent direction is likely to continue.
    * H ≈ 0.50 — **random walk**: past provides no predictive signal.
    * H < 0.45 — **mean-reverting** (anti-persistent): price tends to revert after a move.

    Parameters
    ----------
    series : pd.Series
        Return series (first differences or log-returns). **Do not pass price
        levels** — the algorithm operates on the return (difference) series.
    method : {"dfa", "rs"}
        ``"dfa"`` (default) — Detrended Fluctuation Analysis. Less biased than
        R/S for realistic sample sizes (200–2000 bars).
        ``"rs"`` — Classic Rescaled Range. Higher bias but historically familiar.
    min_window : int
        Smallest sub-window size for the scaling analysis (default 10).
    max_window : int, optional
        Largest sub-window (default: ``len(series) // 4`` for DFA,
        ``len(series) // 2`` for R/S).

    Returns
    -------
    dict with keys:
        hurst         : float – estimated H (0 < H < 1)
        regime        : str   – "trending", "random_walk", or "mean_reverting"
        fit_r_squared : float – R² of the log-log scaling fit (closer to 1 is better)
        method        : str   – method used
        n_obs         : int
    """
    arr = series.dropna().to_numpy(dtype=float)
    n = len(arr)

    _nan_result = {
        "hurst": float("nan"),
        "regime": "unknown",
        "fit_r_squared": float("nan"),
        "method": method,
        "n_obs": n,
    }

    default_max = n // 4 if method == "dfa" else n // 2
    max_w = min(max_window if max_window is not None else default_max, default_max)

    if n < min_window * 4 or min_window >= max_w:
        return _nan_result

    if method == "dfa":
        sizes, values = _dfa(arr, min_window, max_w)
    else:
        sizes, values = _rs(arr, min_window, max_w)

    if len(sizes) < 3 or np.any(values <= 0):
        return _nan_result

    h, r2 = _ols_slope_r2(np.log(sizes), np.log(values))
    h = float(np.clip(h, 0.0, 1.5))

    return {
        "hurst": h,
        "regime": _classify(h),
        "fit_r_squared": r2,
        "method": method,
        "n_obs": n,
    }


def rolling_hurst(
    series: pd.Series,
    window: int = 200,
    step: int = 1,
    method: Literal["dfa", "rs"] = "dfa",
    min_window: int = 10,
) -> pd.Series:
    """
    Rolling Hurst exponent over a sliding window.

    Useful for detecting regime shifts (market switching from trending to
    mean-reverting or vice versa).

    Parameters
    ----------
    series : pd.Series
        Return series (not price levels).
    window : int
        Lookback window in bars (default 200). Minimum ~100 for reliable estimates.
    step : int
        Compute every ``step`` bars (default 1 = every bar). Use ``step > 1``
        to speed up computation on long series — intermediate positions are NaN.
    method : {"dfa", "rs"}
        Estimation method passed to ``hurst_exponent``.
    min_window : int
        Smallest sub-window for internal scaling analysis.

    Returns
    -------
    pd.Series  –  H values indexed like ``series``; first ``window - 1`` rows
                  are NaN.
    """
    arr = series.dropna().to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan)

    for i in range(window - 1, n, step):
        chunk = arr[i - window + 1: i + 1]
        result = hurst_exponent(
            pd.Series(chunk), method=method, min_window=min_window
        )
        out[i] = result["hurst"]

    return pd.Series(out, index=series.dropna().index, name="hurst")
