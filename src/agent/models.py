from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BacktestInput(BaseModel):
    symbol: str = Field(..., description="The ticker symbol to backtest (e.g., 'AAPL').")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format.")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format.")
    strategy_type: str = Field(..., description="The type of strategy (e.g., 'sma_crossover').")
    parameters: Dict[str, Any] = Field({}, description="Strategy parameters (e.g., {'fast_period': 10, 'slow_period': 30}).")
    initial_capital: float = Field(10000.0, description="Initial capital for the backtest.")

class BacktestResult(BaseModel):
    total_return: float
    annualized_volatility: float
    max_drawdown: float
    final_equity: float
    equity_curve: List[float] # Simplified for JSON response usually

class AnalysisInput(BaseModel):
    symbol: str = Field(..., description="Target asset symbol.")
    benchmark: str = Field("SPY", description="Benchmark symbol.")
    period: str = Field("1y", description="Analysis period.")

class AnalysisResult(BaseModel):
    symbol: str
    benchmark: str
    alpha: float
    beta: float
    r_squared: float
    sharpe_ratio: float
