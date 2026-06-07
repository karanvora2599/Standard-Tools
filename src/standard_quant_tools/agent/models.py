from pydantic import BaseModel, Field, field_validator, model_validator
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

    @model_validator(mode="after")
    def _check_weights(self) -> "PortfolioInput":
        if len(self.weights) != len(self.tickers):
            raise ValueError(
                f"len(weights)={len(self.weights)} must equal len(tickers)={len(self.tickers)}"
            )
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total:.8f}")
        return self


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

    @field_validator("n_components")
    @classmethod
    def _check_n_components(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"n_components must be >= 1, got {v}")
        return v


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


# ──────────────────────────────────────────────
# Regime-Adaptive Strategy Selector
# ──────────────────────────────────────────────

class RegimeAdaptiveInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    initial_capital: float = Field(10_000.0, description="Starting capital.")
    commission_pct: float = Field(0.001, description="Commission per trade (fraction).")
    slippage_pct: float = Field(0.0005, description="Slippage per trade (fraction).")
    hurst_method: str = Field("dfa", description="Hurst method: 'dfa' or 'rs'.")
    sma_param_grid: Optional[Dict[str, List[Any]]] = Field(
        None,
        description="Custom param grid for SMA crossover. Default: fast_period=[5,10,20], slow_period=[30,50,100].",
    )
    rsi_param_grid: Optional[Dict[str, List[Any]]] = Field(
        None,
        description="Custom param grid for RSI mean-reversion. Default: period=[7,14,21], oversold=[25,30], overbought=[65,70].",
    )
    macd_param_grid: Optional[Dict[str, List[Any]]] = Field(
        None,
        description="Custom param grid for MACD crossover. Default: fast=[8,12], slow=[21,26], signal=[7,9].",
    )
    bollinger_param_grid: Optional[Dict[str, List[Any]]] = Field(
        None,
        description="Custom param grid for Bollinger reversion. Default: period=[15,20,25], num_std=[1.5,2.0].",
    )
    n_workers: int = Field(1, description="Worker processes for grid search (default 1 for agent use).")


class RegimeAdaptiveResult(BaseModel):
    symbol: str
    regime: str          # "trending" | "random_walk" | "mean_reverting"
    hurst: float
    fit_r_squared: float
    selected_strategy: str
    best_parameters: Dict[str, Any]
    grid_combinations: int
    backtest: BacktestResult


# ──────────────────────────────────────────────
# Cointegration Pair Scanner
# ──────────────────────────────────────────────

class PairScannerInput(BaseModel):
    tickers: List[str] = Field(..., description="Universe of tickers to test for cointegration.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    max_pairs: int = Field(10, description="Maximum number of top pairs to return.")
    min_half_life: float = Field(5.0, description="Minimum mean-reversion half-life in bars.")
    max_half_life: float = Field(126.0, description="Maximum half-life in bars (~6 months).")
    p_value_threshold: float = Field(0.05, description="Maximum cointegration p-value.")
    zscore_window: int = Field(30, description="Rolling window for spread z-score signal.")


class PairResult(BaseModel):
    symbol_a: str
    symbol_b: str
    p_value: float
    hedge_ratio: float
    half_life_days: float
    adf_statistic: float
    current_zscore: float
    signal: str          # "long_a_short_b" | "short_a_long_b" | "neutral"


class PairScannerResult(BaseModel):
    n_pairs_tested: int
    n_pairs_cointegrated: int
    n_pairs_returned: int
    pairs: List[PairResult]


# ──────────────────────────────────────────────
# Walk-Forward Backtest
# ──────────────────────────────────────────────

class WalkForwardInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    strategy: str = Field(
        ...,
        description="Strategy name: 'sma_crossover', 'rsi_mean_reversion', 'macd_crossover', or 'bollinger_reversion'.",
    )
    param_grid: Dict[str, List[Any]] = Field(
        ..., description="Parameter grid, e.g. {'fast_period': [5, 10, 20], 'slow_period': [30, 50]}.",
    )
    train_bars: int = Field(252, description="In-sample window length in bars (default 252 = ~1 year daily).")
    test_bars: int = Field(63, description="Out-of-sample window length in bars (default 63 = ~1 quarter daily).")
    initial_capital: float = Field(10_000.0, description="Starting capital for each window.")
    commission_pct: float = Field(0.001, description="Commission per trade (fraction).")
    slippage_pct: float = Field(0.0005, description="Slippage per trade (fraction).")
    sort_by: str = Field("sharpe_ratio", description="Metric to optimise in-sample (default: 'sharpe_ratio').")


class WalkForwardWindow(BaseModel):
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    out_of_sample_return: float
    out_of_sample_max_drawdown: float


class WalkForwardResult(BaseModel):
    symbol: str
    strategy: str
    n_windows: int
    windows: List[WalkForwardWindow]
    avg_oos_sharpe: float
    avg_oos_return: float
    avg_oos_max_drawdown: float
    pct_windows_profitable: float
    param_stability: Dict[str, Any]  # most common best param per key + frequency


# ──────────────────────────────────────────────
# Portfolio Risk Attribution
# ──────────────────────────────────────────────

class RiskAttributionInput(BaseModel):
    tickers: List[str] = Field(..., description="Portfolio asset symbols.")
    weights: List[float] = Field(..., description="Portfolio weights summing to 1.0.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    benchmark: str = Field("SPY", description="Benchmark symbol for Information Ratio.")
    n_components: int = Field(3, description="Number of PCs for PCA decomposition.")
    factor_tickers: Optional[List[str]] = Field(
        None,
        description="Optional factor proxy tickers (e.g. ['SPY','IWM','IWD']). Enables factor regression on the portfolio.",
    )
    factor_names: Optional[List[str]] = Field(
        None, description="Human-readable factor names. Defaults to factor_tickers."
    )

    @model_validator(mode="after")
    def _check_weights(self) -> "RiskAttributionInput":
        if len(self.weights) != len(self.tickers):
            raise ValueError(
                f"len(weights)={len(self.weights)} must equal len(tickers)={len(self.tickers)}"
            )
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total:.8f}")
        return self


class RiskAttributionResult(BaseModel):
    tickers: List[str]
    weights: List[float]
    # Portfolio-level metrics
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    information_ratio: float
    # Asset risk decomposition
    asset_risk_contributions: Dict[str, float]   # fractional contribution to portfolio vol (sums to 1)
    # PCA decomposition
    pca_variance_explained: Dict[str, float]     # EVR per PC across the asset universe
    portfolio_pc_exposures: Dict[str, float]     # portfolio's loading on each PC
    # Factor model (optional)
    factor_loadings: Optional[Dict[str, float]] = None
    factor_r_squared: Optional[float] = None
    factor_alpha: Optional[float] = None


# ──────────────────────────────────────────────
# ATR-Based Position Sizer
# ──────────────────────────────────────────────

class PositionSizerInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD (for ATR calculation).")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    account_equity: float = Field(..., description="Total account equity in dollars.")
    risk_per_trade_pct: float = Field(
        0.01, description="Fraction of account to risk per trade (default 0.01 = 1%). Must be in (0, 1]."
    )
    atr_period: int = Field(14, description="ATR lookback period (default 14).")
    atr_multiplier: float = Field(
        2.0, description="Stop distance = atr_multiplier × ATR (default 2.0)."
    )
    win_rate: Optional[float] = Field(None, description="Strategy win rate [0,1]. Required for Kelly sizing.")
    avg_win_pct: Optional[float] = Field(None, description="Average winning trade return (e.g. 0.05 = 5%).")
    avg_loss_pct: Optional[float] = Field(None, description="Average losing trade return magnitude (e.g. 0.02 = 2%).")

    @field_validator("risk_per_trade_pct")
    @classmethod
    def _check_risk_pct(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"risk_per_trade_pct must be in (0, 1], got {v}")
        return v


class PositionSizerResult(BaseModel):
    symbol: str
    last_close: float
    atr: float
    atr_pct: float               # ATR as % of price
    stop_distance: float         # atr_multiplier × ATR in $
    # Fixed-risk (ATR-based) sizing
    shares_fixed_risk: int
    position_value_fixed_risk: float
    portfolio_pct_fixed_risk: float
    max_loss_fixed_risk: float   # worst-case $ loss if stop is hit
    # Kelly sizing (populated when win_rate/avg_win/avg_loss are provided)
    kelly_fraction: Optional[float] = None
    shares_half_kelly: Optional[int] = None
    position_value_half_kelly: Optional[float] = None
    portfolio_pct_half_kelly: Optional[float] = None
    # Recommendation
    recommended_sizing: str      # "fixed_risk" | "half_kelly"
    recommended_shares: int
    recommended_position_value: float


# ──────────────────────────────────────────────
# Buy-and-Hold Baseline
# ──────────────────────────────────────────────

class BuyAndHoldInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    initial_capital: float = Field(10_000.0, description="Starting capital.")
    commission_pct: float = Field(0.001, description="One-time buy commission (fraction, default 0.1%).")
    slippage_pct: float = Field(0.0005, description="One-time buy slippage (fraction, default 0.05%).")


# ──────────────────────────────────────────────
# Strategy Comparison
# ──────────────────────────────────────────────

class StrategyComparison(BaseModel):
    strategy: str
    parameters: Dict[str, Any]
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    num_trades: int
    final_equity: float


class CompareStrategiesInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    initial_capital: float = Field(10_000.0, description="Starting capital.")
    commission_pct: float = Field(0.001, description="Commission per trade (fraction).")
    slippage_pct: float = Field(0.0005, description="Slippage per trade (fraction).")
    sort_by: str = Field(
        "sharpe_ratio",
        description=(
            "Metric to rank strategies by. "
            "Options: 'total_return', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown'."
        ),
    )
    sma_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom SMA crossover params. Default: {fast_period: 10, slow_period: 50}.",
    )
    rsi_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom RSI mean-reversion params. Default: {period: 14, oversold: 30, overbought: 70}.",
    )
    macd_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom MACD crossover params. Default: {fast: 12, slow: 26, signal: 9}.",
    )
    bollinger_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom Bollinger reversion params. Default: {period: 20, num_std: 2.0}.",
    )


class CompareStrategiesResult(BaseModel):
    symbol: str
    sort_by: str
    best_strategy: str
    buy_and_hold_return: float
    strategies: List[StrategyComparison]  # sorted by sort_by, best first
