import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    """
    Calculate static Alpha and Beta using NumPy (faster than statsmodels).
    """
    # Align data
    common_index = asset_returns.index.intersection(benchmark_returns.index)
    y = asset_returns.loc[common_index].values
    x = benchmark_returns.loc[common_index].values
    
    if len(y) < 2:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0}

    # Add constant
    X = np.vstack([np.ones(len(x)), x]).T
    
    # OLS using lstsq
    # solution: [alpha, beta]
    beta_hat, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    alpha = beta_hat[0]
    beta = beta_hat[1]
    
    # R-squared
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    ss_res = np.sum((y - (alpha + beta * x))**2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared
    }

def rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60) -> pd.DataFrame:
    """
    Calculate rolling Beta using vectorized NumPy stride tricks.
    Computes covariance and variance in a single pass vs two separate rolling calls.
    """
    common_index = asset_returns.index.intersection(benchmark_returns.index)
    y = asset_returns.loc[common_index].to_numpy(dtype=float)
    x = benchmark_returns.loc[common_index].to_numpy(dtype=float)
    n = len(x)

    beta_arr = np.full(n, np.nan)
    if n >= window:
        y_w = np.lib.stride_tricks.sliding_window_view(y, window)  # (n-w+1, w)
        x_w = np.lib.stride_tricks.sliding_window_view(x, window)

        y_m = y_w.mean(axis=1, keepdims=True)
        x_m = x_w.mean(axis=1, keepdims=True)

        cov = ((y_w - y_m) * (x_w - x_m)).sum(axis=1) / (window - 1)
        var = ((x_w - x_m) ** 2).sum(axis=1) / (window - 1)
        var[var == 0] = np.nan

        beta_arr[window - 1:] = cov / var

    return pd.DataFrame({"Rolling_Beta": beta_arr}, index=common_index)
