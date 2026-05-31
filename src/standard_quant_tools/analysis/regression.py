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
    Calculate rolling Beta using Pandas rolling cov/var (incremental O(n) algorithm).
    """
    common_index = asset_returns.index.intersection(benchmark_returns.index)
    y = asset_returns.loc[common_index]
    x = benchmark_returns.loc[common_index]

    cov = y.rolling(window=window).cov(x)
    var = x.rolling(window=window).var()

    return pd.DataFrame({'Rolling_Beta': cov / var})
