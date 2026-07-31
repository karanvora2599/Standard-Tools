import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass


def calculate_beta(
    asset_returns: pd.Series, benchmark_returns: pd.Series
) -> Dict[str, float]:
    """
    Calculate static Alpha and Beta using OLS.
    """
    common_index = asset_returns.index.intersection(benchmark_returns.index)
    y = asset_returns.loc[common_index].to_numpy(dtype=np.float64)
    x = benchmark_returns.loc[common_index].to_numpy(dtype=np.float64)
    path = "C++" if (HAS_CPP and _cpp_core is not None) else "numpy"
    logger.debug("[beta] n_obs=%d  path=%s", len(y), path)

    if len(y) < 2:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}

    if HAS_CPP and _cpp_core is not None:
        r = _cpp_core.ols2(y, x)
        result = {
            "alpha": r["intercept"],
            "beta": r["slope"],
            "r_squared": r["r_squared"],
        }
    else:
        X = np.vstack([np.ones(len(x)), x]).T
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha = beta_hat[0]
        beta = beta_hat[1]
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        ss_res = np.sum((y - (alpha + beta * x)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
        result = {"alpha": alpha, "beta": beta, "r_squared": r_squared}

    logger.debug(
        "[beta] alpha=%.6f  beta=%.4f  R²=%.4f",
        result["alpha"],
        result["beta"],
        result["r_squared"],
    )
    return result


def rolling_beta(
    asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60
) -> pd.DataFrame:
    """
    Calculate rolling OLS Beta of asset vs benchmark over a sliding window.

    Uses C++ incremental O(1)-per-step sum updates when available (10-40× faster
    than two sequential pandas rolling operations).  Falls back to pandas otherwise.
    """
    common_index = asset_returns.index.intersection(benchmark_returns.index)
    y = asset_returns.loc[common_index]
    x = benchmark_returns.loc[common_index]
    path = "C++" if (HAS_CPP and _cpp_core is not None) else "pandas"
    logger.debug("[rolling_beta] window=%d  bars=%d  path=%s", window, len(y), path)

    # ── C++ fast path ─────────────────────────────────────────────────────────
    if HAS_CPP and _cpp_core is not None:
        try:
            y_arr = y.to_numpy(dtype=np.float64)
            x_arr = x.to_numpy(dtype=np.float64)
            betas = _cpp_core.rolling_beta(y_arr, x_arr, window)
            return pd.DataFrame({"Rolling_Beta": betas}, index=common_index)
        except Exception as exc:
            logger.warning("[rolling_beta] C++ failed (%s) — using pandas", exc)

    # ── Pandas fallback ───────────────────────────────────────────────────────
    cov = y.rolling(window=window).cov(x)
    var = x.rolling(window=window).var()
    # A window with zero benchmark variance (e.g. a constant benchmark)
    # would otherwise divide by zero -- NaN out that bar rather than a
    # mid-series inf/nan spike being mistaken for a real beta.
    safe_var = var.where(var > 0)
    return pd.DataFrame({"Rolling_Beta": cov / safe_var})
