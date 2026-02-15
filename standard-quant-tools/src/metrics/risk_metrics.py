import pandas as pd
import numpy as np
from .return_metrics import annualized_volatility, cagr

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe Ratio.
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)

def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino Ratio.
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    downside_dev = downside_returns.std() * np.sqrt(periods_per_year)
    
    if downside_dev == 0:
        return np.inf
        
    return (excess_returns.mean() * periods_per_year) / downside_dev

def max_drawdown(series: pd.Series) -> float:
    """
    Calculate Maximum Drawdown.
    """
    # Assuming series is price data
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min()
