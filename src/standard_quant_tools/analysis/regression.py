import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Any

def calculate_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    """
    Calculate static Alpha and Beta using OLS.
    """
    # Align data
    common_index = asset_returns.index.intersection(benchmark_returns.index)
    y = asset_returns.loc[common_index]
    x = benchmark_returns.loc[common_index]
    
    if len(y) < 2:
        return {"alpha": 0.0, "beta": 0.0, "r_squared": 0.0, "p_value_alpha": 1.0, "p_value_beta": 1.0}

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    
    return {
        "alpha": model.params[0],
        "beta": model.params[1],
        "r_squared": model.rsquared,
        "p_value_alpha": model.pvalues[0],
        "p_value_beta": model.pvalues[1]
    }

def rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60) -> pd.DataFrame:
    """
    Calculate rolling Beta.
    """
    # Align data
    common_index = asset_returns.index.intersection(benchmark_returns.index)
    y = asset_returns.loc[common_index]
    x = benchmark_returns.loc[common_index]
    
    betas = []
    
    # We can use RollingOLS from statsmodels, but for simplicity and fewer deps if older version, manual loop is robust enough for now
    # Or use pandas rolling covariance / variance
    
    cov = y.rolling(window=window).cov(x)
    var = x.rolling(window=window).var()
    
    rolling_beta = cov / var
    return pd.DataFrame({'Rolling_Beta': rolling_beta})
