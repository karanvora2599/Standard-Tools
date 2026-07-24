import logging
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

# ── Optional C++ fast path ────────────────────────────────────────────────────
# Falls back to pure Python automatically when the extension hasn't been built.
_cpp = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

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
    Detrended Fluctuation Analysis (Python fallback).
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
            seg = y[i * sz : (i + 1) * sz]
            seg_mean = seg.mean()
            b = ((x - x_mean) * (seg - seg_mean)).mean() / x_var if x_var > 0 else 0.0
            a = seg_mean - b * x_mean
            residuals = seg - (a + b * x)
            rms_acc += (residuals**2).mean()
        fluctuations.append(np.sqrt(rms_acc / n_chunks))
        valid.append(sz)

    return np.array(valid, dtype=float), np.array(fluctuations)


def _rs(arr: np.ndarray, min_w: int, max_w: int) -> tuple:
    """
    Classic Rescaled Range (Python fallback).
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
            chunk = arr[i * sz : (i + 1) * sz]
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


# ── Public API ────────────────────────────────────────────────────────────────


def hurst_exponent(
    series: pd.Series,
    method: Literal["dfa", "rs"] = "dfa",
    min_window: int = 10,
    max_window: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate the Hurst exponent of a return series.

    H > 0.55 → trending (persistent).
    H ≈ 0.50 → random walk.
    H < 0.45 → mean-reverting (anti-persistent).

    Uses the C++ extension when available (20–80× faster than the Python path
    for single calls; the gain is most visible in rolling_hurst).

    Parameters
    ----------
    series     : pd.Series  Return series (NOT price levels).
    method     : "dfa" (default) or "rs".
    min_window : Smallest sub-window (default 10).
    max_window : Largest sub-window; None = auto (n//4 for DFA, n//2 for R/S).

    Returns
    -------
    dict with keys: hurst, regime, fit_r_squared, method, n_obs.
    """
    if method not in ("dfa", "rs"):
        raise ValidationError(
            f"method must be 'dfa' or 'rs', got {method!r} — both the C++ "
            "and Python fallback paths treat anything other than the exact "
            "string 'dfa' as 'rs', so a typo would silently run the wrong "
            "method while echoing the typo'd string back in the result."
        )
    if min_window <= 0:
        raise ValidationError(f"min_window must be > 0, got {min_window}")
    if max_window is not None and max_window <= 0:
        raise ValidationError(f"max_window must be > 0, got {max_window}")

    arr = series.dropna().to_numpy(dtype=float)
    n = len(arr)
    path = "C++" if (HAS_CPP and _cpp is not None) else "python"
    logger.debug(
        "[hurst] method=%s  n_obs=%d  min_w=%d  path=%s", method, n, min_window, path
    )

    _nan_result: Dict[str, Any] = {
        "hurst": float("nan"),
        "regime": "unknown",
        "fit_r_squared": float("nan"),
        "method": method,
        "n_obs": n,
    }

    # ── C++ fast path ─────────────────────────────────────────────────────────
    if HAS_CPP and _cpp is not None:
        max_w = max_window if max_window is not None else -1
        if method == "dfa":
            result = _cpp.hurst_dfa(arr, min_window, max_w)
        else:
            result = _cpp.hurst_rs(arr, min_window, max_w)
        # pybind11 returns a Python dict — convert n_obs back to int
        result["n_obs"] = int(result["n_obs"])
        return result

    # ── Python fallback ───────────────────────────────────────────────────────
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

    result = {
        "hurst": h,
        "regime": _classify(h),
        "fit_r_squared": r2,
        "method": method,
        "n_obs": n,
    }
    logger.debug("[hurst] H=%.4f  regime=%s  R²=%.4f", h, result["regime"], r2)
    return result


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

    Uses the C++ extension when available — the entire rolling computation
    runs in a single C++ pass without re-entering the Python interpreter per
    bar (30–100× faster than the Python fallback for typical window sizes).

    Parameters
    ----------
    series     : pd.Series  Return series (not price levels).
    window     : Lookback window in bars (default 200).
    step       : Compute every `step` bars; intermediate positions are NaN.
    method     : "dfa" (default) or "rs".
    min_window : Smallest sub-window for internal scaling.

    Returns
    -------
    pd.Series indexed like `series`; first (window-1) rows are NaN.
    """
    if method not in ("dfa", "rs"):
        raise ValidationError(
            f"method must be 'dfa' or 'rs', got {method!r} — both the C++ "
            "and Python fallback paths treat anything other than the exact "
            "string 'dfa' as 'rs', so a typo would silently run the wrong "
            "method."
        )
    if window <= 0:
        raise ValidationError(f"window must be > 0, got {window}")
    if step <= 0:
        raise ValidationError(f"step must be > 0, got {step}")
    if min_window <= 0:
        raise ValidationError(f"min_window must be > 0, got {min_window}")

    clean = series.dropna()
    arr = clean.to_numpy(dtype=float)
    n = len(arr)
    path = "C++" if (HAS_CPP and _cpp is not None) else "python"
    n_positions = max(0, (n - window) // step + 1)
    logger.debug(
        "[rolling_hurst] window=%d  step=%d  method=%s  n_obs=%d  positions=%d  path=%s",
        window,
        step,
        method,
        n,
        n_positions,
        path,
    )

    # ── C++ fast path ─────────────────────────────────────────────────────────
    if HAS_CPP and _cpp is not None:
        out = _cpp.rolling_hurst(arr, window, step, method, min_window)
        return pd.Series(out, index=clean.index, name="hurst")

    # ── Python fallback ───────────────────────────────────────────────────────
    out = np.full(n, np.nan)
    for i in range(window - 1, n, step):
        chunk = arr[i - window + 1 : i + 1]
        result = hurst_exponent(pd.Series(chunk), method=method, min_window=min_window)
        out[i] = result["hurst"]

    return pd.Series(out, index=clean.index, name="hurst")
