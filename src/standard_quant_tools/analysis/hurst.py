import functools
import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from standard_quant_tools._compat import to_clean_numpy
from standard_quant_tools.error import ValidationError

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)

# ── Optional C++ fast path ────────────────────────────────────────────────────
# Falls back to pure Python automatically when the extension hasn't been built.
_cpp = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

# ── Regime classification ────────────────────────────────────────────────────
#
# A regime label is a claim that the estimate is further from 0.5 than white
# noise would put it, so the band around 0.5 has to be as wide as the
# estimator's own noise at this length. A fixed +/-0.05 is that width only
# past about 3000 observations: at 256 the DFA estimate of pure white noise
# has a standard deviation of 0.08, and a fixed band labelled 27% of white
# noise "trending" and 30% "mean_reverting". The band here is 1.645 standard
# deviations of the white-noise estimate at the series length (about 5% of
# white noise labelled on each side), and never narrower than 0.05.
_BAND_FLOOR = 0.05
_BAND_Z = 1.645

#: Standard deviation of the estimate on white noise, by series length, at
#: the default scale range (min_window=10, max_window automatic). Measured
#: on 3000 simulated series per length; R/S after the small-sample
#: correction below, which shifts the estimate and leaves its spread alone.
#: `tests/analysis/test_hurst.py` re-measures entries of this table, so it
#: cannot drift away from the estimators it describes.
_NULL_SD_LENGTHS = (
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
)
_NULL_SD = {
    "rs": (
        0.185,
        0.142,
        0.105,
        0.090,
        0.073,
        0.064,
        0.053,
        0.050,
        0.041,
        0.037,
        0.032,
        0.029,
        0.026,
        0.025,
        0.022,
        0.020,
    ),
    "dfa": (
        0.545,
        0.253,
        0.152,
        0.121,
        0.095,
        0.081,
        0.067,
        0.060,
        0.049,
        0.044,
        0.039,
        0.035,
        0.029,
        0.028,
        0.024,
        0.022,
    ),
}


def null_standard_deviation(n_obs: int, method: str) -> float:
    """
    Standard deviation of `hurst_exponent` on white noise of length `n_obs`.

    Interpolated log-log in the measured table above and extrapolated along
    its end segments. Calibrated at the default scale range; a narrower
    range of scales is noisier than this.
    """
    if method not in _NULL_SD:
        raise ValidationError(f"method must be 'dfa' or 'rs', got {method!r}")
    lengths = np.log(np.asarray(_NULL_SD_LENGTHS, dtype=float))
    sds = np.log(np.asarray(_NULL_SD[method], dtype=float))
    x = math.log(max(int(n_obs), 2))
    if x <= lengths[0]:
        i = 0
    elif x >= lengths[-1]:
        i = len(lengths) - 2
    else:
        i = int(np.searchsorted(lengths, x)) - 1
    slope = (sds[i + 1] - sds[i]) / (lengths[i + 1] - lengths[i])
    return float(math.exp(sds[i] + slope * (x - lengths[i])))


def regime_band(n_obs: int, method: str) -> float:
    """
    Half-width of the "random_walk" band around 0.5 at this series length.

    1.645 white-noise standard deviations, floored at 0.05: about 5% of
    white-noise series are labelled on each side, at every length.
    """
    return max(_BAND_FLOOR, _BAND_Z * null_standard_deviation(n_obs, method))


def _classify(h: float, band: float = _BAND_FLOOR) -> str:
    if not np.isfinite(h):
        return "unknown"
    if h > 0.5 + band:
        return "trending"
    if h < 0.5 - band:
        return "mean_reverting"
    return "random_walk"


def classify_regime(h: float, n_obs: int, method: str) -> str:
    """
    The regime label `hurst_exponent` gives an estimate `h` made on `n_obs`
    observations: "trending", "random_walk", "mean_reverting", or "unknown"
    for a NaN. Use this to label `rolling_hurst` values, with `n_obs` the
    rolling window.
    """
    return _classify(h, regime_band(n_obs, method))


# ── R/S small-sample correction ──────────────────────────────────────────────
#
# The rescaled range of a short window is biased upward, and the bias shrinks
# with the window, so the log-log slope -- the Hurst estimate -- comes out
# too steep: +0.09 at 256 observations, +0.07 at 1024, +0.05 at 4096, all on
# pure white noise. Uncorrected, R/S labelled 60-75% of white-noise series
# "trending". Anis and Lloyd (1976), with Peters' (1994) (s - 1/2)/s factor,
# give the expected R/S of a window of s independent observations; the slope
# of its logarithm against log s, over the same window sizes the estimator
# fits, is what white noise produces. Its excess over 0.5 is the bias, and it
# depends only on the window sizes, never on the data.


def _expected_rs(s: int) -> float:
    """Anis-Lloyd-Peters expected rescaled range of s independent values."""
    i = np.arange(1, s, dtype=float)
    total = float(np.sum(np.sqrt((s - i) / i)))
    gamma_ratio = math.exp(math.lgamma((s - 1) / 2.0) - math.lgamma(s / 2.0))
    return (s - 0.5) / s * gamma_ratio / math.sqrt(math.pi) * total


@functools.lru_cache(maxsize=1024)
def rs_bias_correction(min_window: int, max_window: int) -> float:
    """
    The amount the R/S estimate exceeds 0.5 on white noise, for this range.

    The OLS slope of log E[R/S](s) on log s over `_log_sizes(min_window,
    max_window)`, minus 0.5. Subtracted from the raw R/S estimate.
    """
    sizes = _log_sizes(int(min_window), int(max_window))
    sizes = sizes[sizes >= 2]
    if sizes.size < 2:
        return 0.0
    log_s = np.log(sizes.astype(float))
    log_e = np.log([_expected_rs(int(s)) for s in sizes])
    slope, _ = _ols_slope_r2(log_s, log_e)
    return float(slope - 0.5)


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


def _check_windows(method: str, min_window: int) -> None:
    if min_window <= 0:
        raise ValidationError(f"min_window must be > 0, got {min_window}")
    # DFA fits a line inside every box. A box of two points fits exactly, so
    # its fluctuation is floating-point residue (1e-15) and the log-log
    # slope runs to the 1.5 clip: white noise came back H=1.5, "trending",
    # on both backends. Three points leave one residual degree of freedom;
    # four is the smallest box whose fluctuation is a measurement.
    if method == "dfa" and min_window < 4:
        raise ValidationError(
            f"min_window must be at least 4 for DFA, got {min_window}: a "
            "detrended box of two or three points has (almost) no residual "
            "to measure, and the fit runs to the 1.5 clip on white noise. "
            "Use min_window >= 4 (the default is 10)."
        )


def _raw_fit(arr: np.ndarray, method: str, min_window: int, max_w: int):
    """(slope, R^2) of the log-log fit, NaN pair when there is no fit."""
    if HAS_CPP and _cpp is not None:
        if method == "dfa":
            raw = _cpp.hurst_dfa(arr, min_window, max_w)
        else:
            raw = _cpp.hurst_rs(arr, min_window, max_w)
        return float(raw["hurst"]), float(raw["fit_r_squared"])

    if method == "dfa":
        sizes, values = _dfa(arr, min_window, max_w)
    else:
        sizes, values = _rs(arr, min_window, max_w)
    if len(sizes) < 3 or np.any(values <= 0):
        return float("nan"), float("nan")
    h, r2 = _ols_slope_r2(np.log(sizes), np.log(values))
    return float(h), float(r2)


def hurst_exponent(
    series: Union[pd.Series, "pl.Series"],
    method: Literal["dfa", "rs"] = "dfa",
    min_window: int = 10,
    max_window: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate the Hurst exponent of a return series.

    H above 0.5 + band -> "trending" (persistent).
    H within the band  -> "random_walk".
    H below 0.5 - band -> "mean_reverting" (anti-persistent).

    The band (`regime_band`) is 1.645 white-noise standard deviations of the
    estimate at this length, never narrower than 0.05: 0.13 for DFA at 256
    observations, 0.07 at 1024, 0.05 from about 3000. About 5% of white-noise
    series are labelled on each side at every length.

    R/S is corrected for its small-sample bias (Anis-Lloyd-Peters): `hurst`
    is the corrected value, `hurst_raw` the uncorrected slope and
    `bias_correction` the difference. DFA needs no correction and reports
    `bias_correction` 0.0.

    Uses the C++ extension when available (20–80× faster than the Python path
    for single calls; the gain is most visible in rolling_hurst).

    Parameters
    ----------
    series     : pd.Series or polars.Series (see Documentation/14_polars_support.md
                 for the full list of what accepts Polars input today) — Return
                 series (NOT price levels).
    method     : "dfa" (default) or "rs".
    min_window : Smallest sub-window (default 10; at least 4 for DFA).
    max_window : Largest sub-window; None = auto (n//4 for DFA, n//2 for R/S).
                 Must exceed min_window. A value above the automatic maximum
                 is lowered to it and `max_window_used` says so.

    Returns
    -------
    dict with keys: hurst, hurst_raw, bias_correction, regime, regime_band,
    fit_r_squared, method, n_obs, max_window_used, warnings. `hurst` is NaN
    and `regime` "unknown" when the series is too short for the scale range
    or has no scaling to fit (a constant series).
    """
    if method not in ("dfa", "rs"):
        raise ValidationError(
            f"method must be 'dfa' or 'rs', got {method!r} — both the C++ "
            "and Python fallback paths treat anything other than the exact "
            "string 'dfa' as 'rs', so a typo would silently run the wrong "
            "method while echoing the typo'd string back in the result."
        )
    _check_windows(method, min_window)
    if max_window is not None and max_window <= 0:
        raise ValidationError(f"max_window must be > 0, got {max_window}")
    # An inverted or empty scale range is a caller error, not a data
    # shortfall: it used to come back as hurst NaN, which the agent tool
    # then reported as 0.0 -- "strongly mean-reverting" -- for any series.
    if max_window is not None and max_window <= min_window:
        raise ValidationError(
            f"max_window ({max_window}) must be greater than min_window "
            f"({min_window}): the exponent is a slope across window sizes and "
            "needs a range of them. Raise max_window or lower min_window."
        )

    arr = to_clean_numpy(series, dtype=float)
    n = len(arr)
    path = "C++" if (HAS_CPP and _cpp is not None) else "python"
    logger.debug(
        "[hurst] method=%s  n_obs=%d  min_w=%d  path=%s", method, n, min_window, path
    )

    default_max = n // 4 if method == "dfa" else n // 2
    max_w = default_max if max_window is None else min(max_window, default_max)
    band = regime_band(n, method) if n >= 2 else float("nan")
    warnings: List[str] = []
    if max_window is not None and max_window > default_max:
        warnings.append(
            f"max_window {max_window} is above the largest usable window for "
            f"{n} observations ({default_max}, n//{4 if method == 'dfa' else 2}) "
            f"and was lowered to it; max_window_used says what was fitted."
        )

    def _nan_result(reason: str) -> Dict[str, Any]:
        return {
            "hurst": float("nan"),
            "hurst_raw": float("nan"),
            "bias_correction": 0.0,
            "regime": "unknown",
            "regime_band": band,
            "fit_r_squared": float("nan"),
            "method": method,
            "n_obs": n,
            "max_window_used": int(max_w),
            "warnings": warnings + [reason],
        }

    if n < min_window * 4 or min_window >= max_w:
        return _nan_result(
            f"{n} observations are too few for min_window={min_window}: the "
            f"fit needs at least {min_window * 4} and a largest window above "
            "the smallest. No exponent is reported."
        )

    h, r2 = _raw_fit(arr, method, min_window, max_w)
    if not np.isfinite(h):
        return _nan_result(
            "No scaling could be fitted (a constant series, or too few usable "
            "window sizes). No exponent is reported."
        )

    # Both backends clip the raw slope into [0, 1.5]; the correction is
    # applied to the clipped value and the result clipped again, so the two
    # paths and rolling_hurst agree bit for bit.
    h_raw = float(np.clip(h, 0.0, 1.5))
    correction = rs_bias_correction(min_window, max_w) if method == "rs" else 0.0
    h_corrected = float(np.clip(h_raw - correction, 0.0, 1.5))

    result = {
        "hurst": h_corrected,
        "hurst_raw": h_raw,
        "bias_correction": float(correction),
        "regime": _classify(h_corrected, band),
        "regime_band": band,
        "fit_r_squared": r2,
        "method": method,
        "n_obs": n,
        "max_window_used": int(max_w),
        "warnings": warnings,
    }
    logger.debug(
        "[hurst] H=%.4f  raw=%.4f  regime=%s  R²=%.4f",
        h_corrected,
        h_raw,
        result["regime"],
        r2,
    )
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
    mean-reverting or vice versa). Each value equals `hurst_exponent` on
    that window alone, including the R/S small-sample correction (one
    constant for the whole series: the window sizes depend only on `window`
    and `min_window`). To label a value, compare it with
    `regime_band(window, method)`, not with a fixed 0.55/0.45.

    Uses the C++ extension when available — the entire rolling computation
    runs in a single C++ pass without re-entering the Python interpreter per
    bar (30–100× faster than the Python fallback for typical window sizes).

    Parameters
    ----------
    series     : pd.Series  Return series (not price levels).
    window     : Lookback window in bars (default 200).
    step       : Compute every `step` bars; intermediate positions are NaN.
    method     : "dfa" (default) or "rs".
    min_window : Smallest sub-window for internal scaling (at least 4 for DFA).

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
    _check_windows(method, min_window)

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

    # Each window's R/S correction is the same constant: the window sizes
    # depend only on `window` and `min_window`, never on the data.
    max_w = window // 2
    correction = (
        rs_bias_correction(min_window, max_w)
        if method == "rs" and min_window < max_w
        else 0.0
    )

    # ── C++ fast path ─────────────────────────────────────────────────────────
    if HAS_CPP and _cpp is not None:
        out = np.asarray(
            _cpp.rolling_hurst(arr, window, step, method, min_window), dtype=float
        )
        if correction:
            finite = np.isfinite(out)
            out[finite] = np.clip(out[finite] - correction, 0.0, 1.5)
    else:
        # ── Python fallback ───────────────────────────────────────────────────
        # hurst_exponent applies the same correction per window.
        out = np.full(n, np.nan)
        for i in range(window - 1, n, step):
            chunk = arr[i - window + 1 : i + 1]
            result = hurst_exponent(
                pd.Series(chunk), method=method, min_window=min_window
            )
            out[i] = result["hurst"]

    series_out = pd.Series(out, index=clean.index, name="hurst")
    # What a caller needs to read the values: the correction subtracted
    # from each raw R/S slope, and the random-walk band for this window.
    series_out.attrs["bias_correction"] = float(correction)
    series_out.attrs["regime_band"] = float(regime_band(window, method))
    return series_out
