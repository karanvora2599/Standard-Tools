import pandas as pd
import numpy as np
from typing import Dict, Any
from standard_quant_tools.metrics.return_metrics import cumulative_return, cagr, annualized_volatility

def run_strategy(price_data: pd.DataFrame, signal_series: pd.Series, initial_capital: float = 10000.0) -> Dict[str, Any]:
    """
    Run a simple vectorised backtest.
    
    Args:
        price_data: DataFrame with 'Close' column.
        signal_series: Series with 1 (buy), -1 (sell), 0 (hold) aligned with price_data.
        initial_capital: Starting capital.
        
    Returns:
        Dict with 'equity_curve', 'stats'
    """
    # Align
    idx = price_data.index.intersection(signal_series.index)
    prices = price_data.loc[idx, 'Close']
    signals = signal_series.loc[idx]
    
    # Calculate returns
    returns = prices.pct_change().fillna(0)
    
    # Shift signals by 1 to avoid lookahead bias (signal today executes tomorrow/close)
    strategy_returns = signals.shift(1) * returns
    strategy_returns = strategy_returns.fillna(0)
    
    # Calculate Equity Curve
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()
    
    # Calculate Metrics
    total_return = cumulative_return(equity_curve)
    annual_vol = annualized_volatility(strategy_returns)
    
    return {
        "final_equity": equity_curve.iloc[-1] if not equity_curve.empty else initial_capital,
        "total_return": total_return,
        "annualized_volatility": annual_vol,
        "equity_curve": equity_curve
    }
