from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────

class BacktestInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol (e.g. 'AAPL').")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    strategy_type: str = Field(
        ...,
        description="Strategy type: 'sma_crossover', 'rsi_mean_reversion', "
                    "'macd_crossover', or 'bollinger_reversion'.",
    )
    parameters: Dict[str, Any] = Field(
        {},
        description=(
            "Strategy parameters. sma_crossover: {fast_period, slow_period}. "
            "rsi_mean_reversion: {period, oversold, overbought}. "
            "macd_crossover: {fast, slow, signal}. "
            "bollinger_reversion: {period, num_std}."
        ),
    )
    initial_capital: float = Field(10_000.0, description="Starting capital.")
    commission_pct: float = Field(0.001, description="Commission per trade (fraction, default 0.1%).")
    slippage_pct: float = Field(0.0005, description="Slippage per trade (fraction, default 0.05%).")


class Trade(BaseModel):
    entry_date: str
    exit_date: str
    direction: str
    entry_price: float
    exit_price: float
    return_pct: float


class BacktestResult(BaseModel):
    total_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return_pct: float
    final_equity: float
    equity_curve: List[float]
    trade_log: Optional[List[Trade]] = None


# ──────────────────────────────────────────────
# Risk Analysis
# ──────────────────────────────────────────────

class AnalysisInput(BaseModel):
    symbol: str = Field(..., description="Target asset symbol.")
    benchmark: str = Field("SPY", description="Benchmark symbol.")
    period: str = Field("1y", description="Analysis period (e.g. '1y', '2y', '6mo').")


class AnalysisResult(BaseModel):
    symbol: str
    benchmark: str
    alpha: float
    beta: float
    r_squared: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    information_ratio: float


# ──────────────────────────────────────────────
# Technical Analysis
# ──────────────────────────────────────────────

class TechnicalInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    indicators: List[str] = Field(
        ['rsi', 'macd', 'bollinger', 'atr'],
        description=(
            "List of indicators to compute. Options: "
            "'sma', 'ema', 'macd', 'rsi', 'stochastic', "
            "'bollinger', 'atr', 'obv', 'vwap', 'adx', 'williams_r'."
        ),
    )


class TechnicalResult(BaseModel):
    symbol: str
    last_close: float
    signals: Dict[str, Any]
    last_values: Dict[str, Any]


# ──────────────────────────────────────────────
# Portfolio Analysis
# ──────────────────────────────────────────────

class PortfolioInput(BaseModel):
    tickers: List[str] = Field(..., description="List of ticker symbols.")
    weights: List[float] = Field(..., description="Portfolio weights (must sum to 1.0).")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    benchmark: str = Field("SPY", description="Benchmark for Information Ratio.")


class PortfolioResult(BaseModel):
    tickers: List[str]
    weights: List[float]
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    information_ratio: float
    total_return: float
    correlation_matrix: Dict[str, Any]


# ──────────────────────────────────────────────
# Screener
# ──────────────────────────────────────────────

class ScreenerInput(BaseModel):
    tickers: List[str] = Field(..., description="Universe of tickers to screen.")
    filters: Dict[str, Any] = Field(
        ...,
        description=(
            "Filter criteria dict. Keys: pe_ratio_max, pb_ratio_max, "
            "debt_equity_max, roe_min, profit_margin_min, div_yield_min, "
            "market_cap_min, rsi_max, rsi_min, price_above_sma (int), "
            "price_below_sma (int), beta_max, beta_min."
        ),
    )
    start_date: Optional[str] = Field(None, description="Historical start for technicals.")
    end_date: Optional[str] = Field(None, description="Historical end for technicals.")
    sort_by: Optional[str] = Field(None, description="Column to sort results by.")
    ascending: bool = Field(True, description="Sort direction.")


class ScreenerResult(BaseModel):
    num_passed: int
    tickers_passed: List[str]
    results: List[Dict[str, Any]]
