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


# ──────────────────────────────────────────────
# Factor Regression
# ──────────────────────────────────────────────

class FactorRegressionInput(BaseModel):
    symbol: str = Field(..., description="Asset to analyse (e.g. 'AAPL').")
    factor_tickers: List[str] = Field(
        ...,
        description=(
            "Ticker proxies for each factor (e.g. ['SPY', 'IWM', 'IWD'] "
            "for market, size, value)."
        ),
    )
    factor_names: Optional[List[str]] = Field(
        None,
        description=(
            "Human-readable names for each factor (e.g. ['mkt', 'smb', 'hml']). "
            "Defaults to factor_tickers when omitted."
        ),
    )
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    rolling_window: Optional[int] = Field(
        None,
        description=(
            "If set, also return the last 20 bars of rolling OLS loadings "
            "over this window (e.g. 60 for a 60-day rolling model)."
        ),
    )


class FactorRegressionResult(BaseModel):
    symbol: str
    factors: List[str]
    alpha: float
    loadings: Dict[str, float]
    t_stats: Dict[str, float]
    p_values: Dict[str, float]
    r_squared: float
    adj_r_squared: float
    n_obs: int
    rolling_alpha_tail: Optional[List[float]] = None
    rolling_loadings_tail: Optional[Dict[str, List[float]]] = None


# ──────────────────────────────────────────────
# Cointegration
# ──────────────────────────────────────────────

class CointegrationInput(BaseModel):
    symbol_a: str = Field(..., description="First asset symbol (the 'long' leg).")
    symbol_b: str = Field(..., description="Second asset symbol (the 'short' leg).")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    zscore_window: int = Field(
        30,
        description="Rolling window (bars) for the spread z-score used to generate a signal.",
    )


class CointegrationResult(BaseModel):
    symbol_a: str
    symbol_b: str
    cointegrated: bool
    p_value: float
    hedge_ratio: float
    adf_statistic: float
    half_life_days: float
    critical_values: Dict[str, float]
    spread_mean: float
    spread_std: float
    current_zscore: float
    signal: str   # "long_a_short_b" | "short_a_long_b" | "neutral"
    n_obs: int


# ──────────────────────────────────────────────
# PCA
# ──────────────────────────────────────────────

class PCAInput(BaseModel):
    tickers: List[str] = Field(..., description="Universe of tickers to decompose.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    n_components: int = Field(3, description="Number of principal components to extract (default 3).")


class PCAResult(BaseModel):
    tickers: List[str]
    n_components: int
    n_obs: int
    explained_variance_ratio: Dict[str, float]       # {"PC1": 0.42, "PC2": 0.12, ...}
    cumulative_variance_ratio: Dict[str, float]      # {"PC1": 0.42, "PC2": 0.54, ...}
    loadings: Dict[str, Dict[str, float]]            # {"PC1": {"AAPL": 0.35, ...}, ...}
    factor_contributions: Dict[str, Dict[str, float]] # {"AAPL": {"PC1": 0.38, ...}, ...}


# ──────────────────────────────────────────────
# Hurst Exponent
# ──────────────────────────────────────────────

class HurstInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    method: str = Field(
        "dfa",
        description="Estimation method: 'dfa' (Detrended Fluctuation Analysis, default) or 'rs' (Rescaled Range).",
    )
    rolling_window: Optional[int] = Field(
        None,
        description=(
            "If set, compute rolling Hurst over this window and include the latest "
            "value plus regime breakdown fractions in the result."
        ),
    )


class HurstResult(BaseModel):
    symbol: str
    hurst: float
    regime: str   # "trending" | "random_walk" | "mean_reverting"
    fit_r_squared: float
    method: str
    n_obs: int
    rolling_current: Optional[float] = None
    rolling_regime_fractions: Optional[Dict[str, float]] = None
