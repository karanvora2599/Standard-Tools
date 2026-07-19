import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def cumulative_return(series: pd.Series) -> float:
    """
    Calculate Cumulative Return.
    """
    if series.empty:
        logger.warning("[cumulative_return] empty series — returning 0.0")
        return 0.0
    return (series.iloc[-1] / series.iloc[0]) - 1

def cagr(series: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    """
    if series.empty:
        logger.warning("[cagr] empty series — returning 0.0")
        return 0.0

    total_ret = cumulative_return(series)
    num_years = len(series) / periods_per_year
    
    if num_years == 0:
        return 0.0
        
    return (1 + total_ret) ** (1 / num_years) - 1

def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Annualized Volatility.
    """
    return returns.std() * np.sqrt(periods_per_year)
