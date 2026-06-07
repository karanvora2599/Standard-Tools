from typing import Any, Dict

import numpy as np
import pandas as pd

_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import _sqt_core as _cpp_core  # type: ignore[attr-defined]
    HAS_CPP = True
except ImportError:
    pass


def calculate_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    """
    Calculate static Alpha and Beta using OLS.
    """
    common_index = asset_returns.index.intersection(benchmark_returns.index)
    y = asset_returns.loc[common_index].to_numpy(dtype=np.float64)
    x = benchmark_returns.loc[common_index].to_numpy(dtype=np.float64)

    if len(y) < 2:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}

    if HAS_CPP and _cpp_core is not None:
        r = _cpp_core.ols2(y, x)
        return {"alpha": r["intercept"], "beta": r["slope"], "r_squared": r["r_squared"]}

    X = np.vstack([np.ones(len(x)), x]).T
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha = beta_hat[0]
    beta = beta_hat[1]
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - (alpha + beta * x)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return {"alpha": alpha, "beta": beta, "r_squared": r_squared}

def rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60) -> pd.DataFrame:
    """
    Calculate rolling Beta using Pandas rolling cov/var (incremental O(n) algorithm).
    """
    common_index = asset_returns.index.intersection(benchmark_returns.index)
    y = asset_returns.loc[common_index]
    x = benchmark_returns.loc[common_index]

    cov = y.rolling(window=window).cov(x)
    var = x.rolling(window=window).var()

    return pd.DataFrame({'Rolling_Beta': cov / var})
