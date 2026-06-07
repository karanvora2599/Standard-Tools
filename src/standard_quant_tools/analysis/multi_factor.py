import math
from typing import Any, Dict

import numpy as np
import pandas as pd

_scipy_stats = None
try:
    from scipy import stats as _scipy_stats  # type: ignore[assignment]
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

_sqrt2 = math.sqrt(2.0)
_math_erf = math.erf


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Exact normal CDF via math.erf — no scipy required."""
    vec = np.vectorize(lambda v: 0.5 * (1.0 + _math_erf(v / _sqrt2)))
    return vec(x).astype(float)


def multi_factor_regression(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> Dict[str, Any]:
    """
    OLS regression of asset_returns on N factors.

    Parameters
    ----------
    asset_returns : pd.Series
        Daily (or periodic) returns of the asset being analysed.
    factor_returns : pd.DataFrame
        One column per factor (e.g. market, SMB, HML). Index must overlap
        with asset_returns — alignment is handled automatically.

    Returns
    -------
    dict with keys:
        alpha         : float  – per-period OLS intercept (raw coefficient; multiply by periods_per_year to annualize)
        loadings      : dict   – {factor_name: coefficient}
        t_stats       : dict   – {factor_name: t-statistic} (includes "alpha")
        p_values      : dict   – {factor_name: two-tailed p-value}
        r_squared     : float
        adj_r_squared : float
        n_obs         : int
    """
    common_idx = asset_returns.index.intersection(factor_returns.index)
    y = asset_returns.loc[common_idx].to_numpy(dtype=float)
    X_f = factor_returns.loc[common_idx].to_numpy(dtype=float)
    factor_names = list(factor_returns.columns)

    n = len(y)
    k = X_f.shape[1] + 1  # +1 for intercept

    _nan = float("nan")
    if n < k + 1:
        nan_factors = {name: _nan for name in factor_names}
        nan_all = {"alpha": _nan, **nan_factors}
        return {
            "alpha": _nan,
            "loadings": nan_factors,
            "t_stats": nan_all,
            "p_values": nan_all,
            "r_squared": _nan,
            "adj_r_squared": _nan,
            "n_obs": n,
        }

    X = np.column_stack([np.ones(n), X_f])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ beta
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof = n - k
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / dof if dof > 0 else _nan

    # Standard errors: SE_i = sqrt(s² * [(X'X)^{-1}]_ii)
    s2 = ss_res / dof if dof > 0 else _nan
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(XtX_inv) * s2, 0.0))
    except np.linalg.LinAlgError:
        se = np.full(k, _nan)

    t_vals = np.where(se > 0, beta / se, np.nan)

    if HAS_SCIPY and _scipy_stats is not None:
        p_vals = 2.0 * _scipy_stats.t.sf(np.abs(t_vals), df=dof)
    else:
        p_vals = 2.0 * (1.0 - _norm_cdf(np.abs(t_vals)))

    all_names = ["alpha"] + factor_names
    t_stats = {name: float(t_vals[i]) for i, name in enumerate(all_names)}
    p_values = {name: float(p_vals[i]) for i, name in enumerate(all_names)}

    return {
        "alpha": float(beta[0]),
        "loadings": {name: float(beta[i + 1]) for i, name in enumerate(factor_names)},
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "n_obs": n,
    }


def rolling_factor_loadings(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Rolling OLS factor loadings over a sliding window.

    Returns a DataFrame indexed like asset_returns with columns
    ["alpha", factor1, factor2, ...]. The first (window-1) rows are NaN.
    """
    common_idx = asset_returns.index.intersection(factor_returns.index)
    y_arr = asset_returns.loc[common_idx].to_numpy(dtype=float)
    X_arr = factor_returns.loc[common_idx].to_numpy(dtype=float)
    factor_names = list(factor_returns.columns)
    col_names = ["alpha"] + factor_names

    n = len(y_arr)
    out = np.full((n, len(col_names)), np.nan)

    for i in range(window - 1, n):
        y_w = y_arr[i - window + 1: i + 1]
        X_w = X_arr[i - window + 1: i + 1]
        X_des = np.column_stack([np.ones(window), X_w])
        beta, *_ = np.linalg.lstsq(X_des, y_w, rcond=None)
        out[i] = beta

    return pd.DataFrame(out, index=common_idx, columns=col_names)
