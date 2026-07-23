from enum import Enum
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
    fill_price: str = Field(
        "close",
        description=(
            "'close' (default) — signal known at bar t-1's close is filled at that "
            "same close. 'next_open' — entries/exits/holds are priced off the bar's "
            "own Open where relevant (see run_strategy docstring for the exact "
            "overnight/intraday decomposition); more conservative and realistic."
        ),
    )


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


class PairFailure(BaseModel):
    symbol_a: str
    symbol_b: str
    reason: str


class PairScannerResult(BaseModel):
    n_pairs_tested: int
    n_pairs_cointegrated: int
    n_pairs_returned: int
    pairs: List[PairResult]
    # Explicit failure reporting so an errored pair/ticker is never confused
    # with one that was tested and simply didn't qualify.
    failed_pairs: List[PairFailure] = []
    failed_tickers: Dict[str, str] = {}   # ticker -> fetch-error message


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
    fill_price: str = Field(
        "close",
        description="'close' (default), 'next_open', or 'midpoint' — applied to the out-of-sample leg of every window (see BacktestInput.fill_price).",
    )


class WalkForwardWindow(BaseModel):
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: Dict[str, Any]
    in_sample_sharpe: float
    in_sample_return: float
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
    # Stitched (compounded) out-of-sample metrics — computed from one
    # chronological equity curve across all OOS windows, not an average of
    # independent per-window stats. See backtest/walk_forward.py.
    stitched_oos_return: float
    stitched_oos_sharpe: float
    stitched_oos_sortino: float
    stitched_oos_max_drawdown: float
    stitched_oos_calmar: float
    is_to_oos_sharpe_decay: float   # avg in-sample sharpe minus stitched OOS sharpe
    is_to_oos_return_decay: float   # avg in-sample return minus stitched OOS return
    worst_oos_window: int           # window_index with the lowest out_of_sample_return
    longest_losing_window_streak: int
    parameter_turnover: float       # fraction of consecutive windows whose best_params changed


# ──────────────────────────────────────────────
# Regime-Adaptive Walk-Forward Backtest (leakage-free counterpart to
# RegimeAdaptiveInput/Result — regime detection AND strategy/parameter
# selection happen strictly within each window's training data)
# ──────────────────────────────────────────────

class RegimeAdaptiveWalkForwardInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    train_bars: int = Field(252, description="In-sample window length in bars (default 252 = ~1 year daily).")
    test_bars: int = Field(63, description="Out-of-sample window length in bars (default 63 = ~1 quarter daily).")
    initial_capital: float = Field(10_000.0, description="Starting capital for each window.")
    commission_pct: float = Field(0.001, description="Commission per trade (fraction).")
    slippage_pct: float = Field(0.0005, description="Slippage per trade (fraction).")
    hurst_method: str = Field("dfa", description="Hurst method: 'dfa' or 'rs' — reported as diagnostic context per window, not used to hard-select a strategy family.")
    sma_param_grid: Optional[Dict[str, List[Any]]] = Field(
        None, description="Custom param grid for SMA crossover. Default: fast_period=[5,10,20], slow_period=[30,50,100].",
    )
    rsi_param_grid: Optional[Dict[str, List[Any]]] = Field(
        None, description="Custom param grid for RSI mean-reversion. Default: period=[7,14,21], oversold=[25,30], overbought=[65,70].",
    )
    macd_param_grid: Optional[Dict[str, List[Any]]] = Field(
        None, description="Custom param grid for MACD crossover. Default: fast=[8,12], slow=[21,26], signal=[7,9].",
    )
    bollinger_param_grid: Optional[Dict[str, List[Any]]] = Field(
        None, description="Custom param grid for Bollinger reversion. Default: period=[15,20,25], num_std=[1.5,2.0].",
    )
    sort_by: str = Field("sharpe_ratio", description="Metric to optimise in-sample, across all four strategies (default: 'sharpe_ratio').")
    fill_price: str = Field(
        "close",
        description="'close' (default), 'next_open', or 'midpoint' — applied to the out-of-sample leg of every window (see BacktestInput.fill_price).",
    )


class RegimeAdaptiveWalkForwardWindow(BaseModel):
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    regime: str          # "trending" | "random_walk" | "mean_reverting" | "unknown" — diagnostic context only
    hurst: float
    fit_r_squared: float
    selected_strategy: str
    best_params: Dict[str, Any]
    in_sample_sharpe: float
    in_sample_return: float
    out_of_sample_sharpe: float
    out_of_sample_return: float
    out_of_sample_max_drawdown: float


class RegimeAdaptiveWalkForwardResult(BaseModel):
    symbol: str
    n_windows: int
    windows: List[RegimeAdaptiveWalkForwardWindow]
    avg_oos_sharpe: float
    avg_oos_return: float
    avg_oos_max_drawdown: float
    pct_windows_profitable: float
    strategy_stability: Dict[str, Any]   # most common selected_strategy across windows + frequency
    stitched_oos_return: float
    stitched_oos_sharpe: float
    stitched_oos_sortino: float
    stitched_oos_max_drawdown: float
    stitched_oos_calmar: float
    worst_oos_window: int
    longest_losing_window_streak: int


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
    fill_price: str = Field("close", description="'close' (default) or 'next_open' — see BacktestInput.fill_price.")


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
    fill_price: str = Field("close", description="'close' (default) or 'next_open' — see BacktestInput.fill_price.")


class CompareStrategiesResult(BaseModel):
    symbol: str
    sort_by: str
    best_strategy: str
    buy_and_hold_return: float
    strategies: List[StrategyComparison]  # sorted by sort_by, best first


# ──────────────────────────────────────────────
# Stock Fundamentals
# ──────────────────────────────────────────────

class FundamentalsInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol (e.g. 'AAPL').")


class FundamentalsResult(BaseModel):
    symbol: str
    name: str
    sector: str
    industry: str
    country: Optional[str]
    full_time_employees: Optional[int]
    forward_pe: Optional[float]
    trailing_pe: Optional[float]
    price_to_book: Optional[float]
    debt_to_equity: Optional[float]
    return_on_equity: Optional[float]
    profit_margins: Optional[float]
    dividend_yield: Optional[float]
    market_cap: Optional[int]


# ──────────────────────────────────────────────
# Backtest Optimization
# ──────────────────────────────────────────────

class BacktestOptInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    strategy: str = Field(
        ...,
        description=(
            "Strategy to optimise: 'sma_crossover', 'rsi_mean_reversion', "
            "'macd_crossover', or 'bollinger_reversion'."
        ),
    )
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    param_grid: Dict[str, List[Any]] = Field(
        ...,
        description=(
            "Parameter search space. "
            "sma_crossover: {'fast_period': [5,10,20], 'slow_period': [30,50,100]}. "
            "rsi_mean_reversion: {'period': [7,14,21], 'oversold': [25,30], 'overbought': [65,70]}. "
            "macd_crossover: {'fast': [8,12], 'slow': [21,26], 'signal': [7,9]}. "
            "bollinger_reversion: {'period': [15,20,25], 'num_std': [1.5,2.0,2.5]}."
        ),
    )
    initial_capital: float = Field(10_000.0, description="Starting capital.")
    sort_by: str = Field(
        "sharpe_ratio",
        description=(
            "Metric to optimise. "
            "Options: 'sharpe_ratio', 'total_return', 'calmar_ratio', 'sortino_ratio', 'max_drawdown'."
        ),
    )
    top_n: int = Field(5, description="Number of top parameter combinations to return (default 5, max 20).")
    n_workers: int = Field(1, description="CPU workers for parallel grid search (default 1).")
    fill_price: str = Field("close", description="'close' (default) or 'next_open' — see BacktestInput.fill_price.")


class OptimizationRun(BaseModel):
    rank: int
    parameters: Dict[str, Any]
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    num_trades: int


class BacktestOptResult(BaseModel):
    symbol: str
    strategy: str
    n_combinations: int
    sort_by: str
    best_params: Dict[str, Any]
    best_sharpe: float
    best_return: float
    top_results: List[OptimizationRun]


# ──────────────────────────────────────────────
# Advanced Indicators
# ──────────────────────────────────────────────

class AdvancedIndicatorsInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    mfi_period: int = Field(14, description="Money Flow Index period (default 14).")
    atr_period: int = Field(14, description="Wilder ATR period (default 14).")
    sar_af_start: float = Field(0.02, description="Parabolic SAR initial acceleration factor (default 0.02).")
    sar_af_max: float = Field(0.2, description="Parabolic SAR maximum acceleration factor (default 0.2).")


class AdvancedIndicatorsResult(BaseModel):
    symbol: str
    last_close: float
    sar_value: float
    sar_trend: str       # "bullish" | "bearish"
    sar_signal: str      # "buy" | "sell"
    wilder_atr: float
    wilder_atr_pct: float
    mfi: float
    mfi_signal: str      # "overbought" | "oversold" | "neutral"


# ──────────────────────────────────────────────
# Rolling Beta
# ──────────────────────────────────────────────

class RollingBetaInput(BaseModel):
    symbol: str = Field(..., description="Asset ticker symbol.")
    benchmark: str = Field("SPY", description="Benchmark symbol (default SPY).")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    window: int = Field(60, description="Rolling window in bars (default 60 ≈ 3 months daily).")


class RollingBetaResult(BaseModel):
    symbol: str
    benchmark: str
    window: int
    current_beta: float
    beta_1m_ago: Optional[float]
    beta_3m_ago: Optional[float]
    beta_6m_ago: Optional[float]
    beta_trend: str      # "increasing" | "decreasing" | "stable"
    beta_min: float
    beta_max: float
    beta_mean: float
    n_obs: int


# ──────────────────────────────────────────────
# Extended Risk Metrics
# ──────────────────────────────────────────────

class ExtendedRiskInput(BaseModel):
    symbol: str = Field(..., description="Asset ticker symbol.")
    benchmark: str = Field("SPY", description="Benchmark symbol (default SPY).")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")


class ExtendedRiskResult(BaseModel):
    symbol: str
    benchmark: str
    annualized_return: float
    calmar_ratio: float
    treynor_ratio: float
    var_parametric_95: float
    var_parametric_99: float
    var_historical_99: float
    cvar_99: float
    beta: float


# ──────────────────────────────────────────────
# Custom Signal Backtest (bring-your-own signal)
# ──────────────────────────────────────────────

class SignalType(str, Enum):
    """
    What a custom signal's numeric values mean, and how strictly they're
    validated. Default is SCORE — unrestricted, exactly the behavior every
    caller already gets today (this enum's whole purpose is to make that
    contract explicit and opt into stricter validation, not to change the
    default). run_strategy's math is unchanged either way: it always
    multiplies the (lagged) signal value by the bar's return, regardless
    of signal_type.
    """
    SCORE = "score"                  # unrestricted float — caller owns the scale/leverage semantics
    DIRECTION = "direction"          # must be exactly -1, 0, or 1 (within 1e-9)
    TARGET_WEIGHT = "target_weight"  # must satisfy |value| <= max_abs_weight


def _validate_signal_values(values: Dict[Any, float], signal_type: "SignalType", max_abs_weight: float) -> None:
    if signal_type == SignalType.DIRECTION:
        bad = {k: v for k, v in values.items() if min(abs(v - d) for d in (-1.0, 0.0, 1.0)) > 1e-9}
        if bad:
            sample = dict(list(bad.items())[:5])
            raise ValueError(
                f"signal_type='direction' requires every value to be exactly -1, 0, or 1; "
                f"got out-of-range value(s): {sample}"
            )
    elif signal_type == SignalType.TARGET_WEIGHT:
        bad = {k: v for k, v in values.items() if abs(v) > max_abs_weight}
        if bad:
            sample = dict(list(bad.items())[:5])
            raise ValueError(
                f"signal_type='target_weight' requires |value| <= max_abs_weight={max_abs_weight}; "
                f"got out-of-range value(s): {sample}"
            )
    # SCORE: unrestricted, matches today's permissive behavior exactly.


class CustomSignalBacktestInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    signals: Dict[str, float] = Field(
        ...,
        description=(
            "Map of ISO date (YYYY-MM-DD) -> signal value (1=long, 0=flat, -1=short), "
            "computed entirely outside this library (e.g. your own alpha model). "
            "This tool does not generate the signal logic — it only backtests it. "
            "Whether/how the values are validated is controlled by signal_type. "
            "Dates are matched against the fetched OHLCV index; any dates on "
            "either side with no counterpart are ignored."
        ),
    )
    signal_type: SignalType = Field(
        SignalType.SCORE,
        description=(
            "'score' (default, unrestricted — today's exact behavior) | 'direction' "
            "(every value must be exactly -1, 0, or 1) | 'target_weight' (every "
            "|value| must be <= max_abs_weight)."
        ),
    )
    max_abs_weight: float = Field(
        1.0, description="Bound used only when signal_type='target_weight' (ignored otherwise)."
    )
    initial_capital: float = Field(10_000.0, description="Starting capital.")
    commission_pct: float = Field(0.001, description="Commission per trade (fraction, default 0.1%).")
    slippage_pct: float = Field(0.0005, description="Slippage per trade (fraction, default 0.05%).")
    fill_price: str = Field("close", description="'close' (default) or 'next_open' — see BacktestInput.fill_price.")

    @model_validator(mode="after")
    def _check_signal_values(self) -> "CustomSignalBacktestInput":
        _validate_signal_values(self.signals, self.signal_type, self.max_abs_weight)
        return self


# ──────────────────────────────────────────────
# Signal Panel Backtest (bring-your-own multi-ticker signal matrix)
# ──────────────────────────────────────────────

class SignalPanelBacktestInput(BaseModel):
    tickers: List[str] = Field(..., description="Ticker universe. Must match signal_panel's outer keys.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    signal_panel: Dict[str, Dict[str, float]] = Field(
        ...,
        description=(
            "Per-ticker signal map: {ticker: {date: value}}, value in "
            "{1=long, 0=flat, -1=short}. Computed entirely outside this library "
            "(e.g. a cross-sectional alpha model) — this tool only backtests it "
            "and combines the per-ticker results into portfolio-level metrics."
        ),
    )
    weights: Optional[Dict[str, float]] = Field(
        None,
        description="Per-ticker portfolio weight, must sum to 1.0. Defaults to equal weight across tickers.",
    )
    initial_capital: float = Field(10_000.0, description="Starting capital applied per ticker.")
    commission_pct: float = Field(0.001, description="Commission per trade (fraction).")
    slippage_pct: float = Field(0.0005, description="Slippage per trade (fraction).")
    benchmark: Optional[str] = Field(
        None, description="Optional benchmark ticker — adds information_ratio to portfolio_metrics."
    )
    include_trade_log: bool = Field(False, description="If True, include a per-trade log for each ticker.")
    fill_price: str = Field("close", description="'close' (default) or 'next_open' — see BacktestInput.fill_price.")
    signal_type: SignalType = Field(
        SignalType.SCORE,
        description=(
            "'score' (default, unrestricted — today's exact behavior) | 'direction' "
            "(every value must be exactly -1, 0, or 1) | 'target_weight' (every "
            "|value| must be <= max_abs_weight). Applies uniformly to every ticker's signal map."
        ),
    )
    max_abs_weight: float = Field(
        1.0, description="Bound used only when signal_type='target_weight' (ignored otherwise)."
    )

    @model_validator(mode="after")
    def _check_panel_and_weights(self) -> "SignalPanelBacktestInput":
        missing = [t for t in self.tickers if t not in self.signal_panel]
        if missing:
            raise ValueError(f"signal_panel is missing entries for: {missing}")
        if self.weights is not None:
            if set(self.weights) != set(self.tickers):
                raise ValueError("weights keys must exactly match tickers")
            total = sum(self.weights.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"weights must sum to 1.0, got {total:.6f}")
        for ticker in self.tickers:
            try:
                _validate_signal_values(self.signal_panel[ticker], self.signal_type, self.max_abs_weight)
            except ValueError as exc:
                raise ValueError(f"ticker '{ticker}': {exc}") from exc
        return self


class SignalPanelBacktestResult(BaseModel):
    tickers: List[str]
    per_ticker: Dict[str, BacktestResult]
    portfolio_metrics: Dict[str, Any]


# ──────────────────────────────────────────────
# True Portfolio Simulation (shared cash, rebalancing — the gap
# run_signal_panel_backtest can't close, since it gives every ticker its
# own independent capital and only blends return streams afterward)
# ──────────────────────────────────────────────

_CONSTRUCTION_METHODS = (
    "rank_weighted", "equal_weight_top_bottom", "zscore_normalized", "vol_scaled",
)


class PortfolioSimulationInput(BaseModel):
    tickers: List[str] = Field(..., description="Ticker universe. Must match target_weights' outer keys.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    target_weights: Dict[str, Dict[str, float]] = Field(
        ...,
        description=(
            "Per-ticker map: {ticker: {date: value}}. When signal_type='target_weight' "
            "(default), value = fraction of account equity (negative for short), and "
            "every ticker must share the identical set of rebalance dates — that shared "
            "date set is the rebalance calendar; between rebalance dates, share counts "
            "stay fixed and weights drift with the market, unlike "
            "run_signal_panel_backtest's fixed per-bar blend. When signal_type='score', "
            "value is an arbitrary per-ticker alpha score (unrestricted), converted into "
            "target weights via construction_method before simulation — every date "
            "present becomes a rebalance date."
        ),
    )
    signal_type: SignalType = Field(
        SignalType.TARGET_WEIGHT,
        description=(
            "'target_weight' (default — today's exact behavior: target_weights values "
            "are already portfolio weights) | 'score' (target_weights values are "
            "arbitrary per-ticker alpha scores, converted into weights via "
            "construction_method — see backtest/sizing.py — before simulation)."
        ),
    )
    construction_method: Optional[str] = Field(
        None,
        description=(
            f"Required when signal_type='score'. One of {_CONSTRUCTION_METHODS} "
            "(backtest/sizing.py). Ignored when signal_type='target_weight'."
        ),
    )
    gross_leverage: float = Field(
        1.0, description="Target sum(|weight|) per date when signal_type='score' (ignored otherwise)."
    )
    n_long: Optional[int] = Field(
        None, description="Required when construction_method='equal_weight_top_bottom'."
    )
    n_short: Optional[int] = Field(
        None, description="Required when construction_method='equal_weight_top_bottom'."
    )
    vol_lookback: int = Field(
        20, description="Rolling window (bars) used when construction_method='vol_scaled'."
    )
    make_dollar_neutral: bool = Field(
        False,
        description=(
            "If True, post-process constructed weights so sum(weight)==0 per date "
            "(backtest/sizing.py's dollar_neutral). Only applies when signal_type='score'."
        ),
    )
    initial_capital: float = Field(10_000.0, description="Starting cash for the whole account.")
    commission_pct: float = Field(0.001, description="Commission per trade notional (fraction).")
    slippage_pct: float = Field(0.0005, description="Slippage per trade notional (fraction).")
    max_gross_leverage: float = Field(
        1.0, description="Reject any rebalance date whose sum(|weight|) exceeds this (default 1.0 = fully invested, no leverage)."
    )
    max_position_pct: float = Field(
        1.0, description="Reject any single position whose |weight| exceeds this."
    )
    fill_price: str = Field(
        "close",
        description="'close' (default), 'next_open', or 'midpoint' — see run_strategy's fill_price / the True Portfolio Simulation docs.",
    )
    benchmark: Optional[str] = Field(None, description="Optional benchmark ticker — adds information_ratio.")

    @model_validator(mode="after")
    def _check_weights_panel(self) -> "PortfolioSimulationInput":
        missing = [t for t in self.tickers if t not in self.target_weights]
        if missing:
            raise ValueError(f"target_weights is missing entries for: {missing}")
        date_sets = {t: frozenset(self.target_weights[t]) for t in self.tickers}
        calendars = set(date_sets.values())
        if len(calendars) > 1:
            raise ValueError(
                "every ticker in target_weights must share the identical set of "
                f"rebalance dates (the rebalance calendar); got mismatched sets: {date_sets}"
            )

        if self.signal_type == SignalType.SCORE:
            if self.construction_method not in _CONSTRUCTION_METHODS:
                raise ValueError(
                    f"signal_type='score' requires construction_method to be one of "
                    f"{_CONSTRUCTION_METHODS}, got {self.construction_method!r}"
                )
            if self.construction_method == "equal_weight_top_bottom" and (
                self.n_long is None or self.n_short is None
            ):
                raise ValueError(
                    "construction_method='equal_weight_top_bottom' requires both "
                    "n_long and n_short"
                )
            return self

        # signal_type == TARGET_WEIGHT (default): today's exact validation.
        for date in next(iter(calendars), frozenset()):
            row = {t: self.target_weights[t][date] for t in self.tickers}
            _validate_signal_values(row, SignalType.TARGET_WEIGHT, self.max_position_pct)
            gross = sum(abs(v) for v in row.values())
            if gross > self.max_gross_leverage + 1e-9:
                raise ValueError(
                    f"rebalance date {date}: gross leverage {gross:.4f} exceeds "
                    f"max_gross_leverage={self.max_gross_leverage}"
                )
        return self


class RebalanceEvent(BaseModel):
    date: str
    turnover_pct: float
    gross_leverage_after: float
    n_positions: int


class PortfolioSimulationResult(BaseModel):
    tickers: List[str]
    n_rebalances: int
    rebalance_log: List[RebalanceEvent]
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    information_ratio: Optional[float] = None
    final_equity: float
    final_cash: float
    avg_gross_leverage: float
    max_gross_leverage_used: float
    equity_curve: List[float]
    warnings: List[str] = []


# ──────────────────────────────────────────────
# Pair Trade Backtest (synchronized two-leg execution — reuses
# run_portfolio_simulation so both legs enter/exit on the same rebalance
# event, unlike scan_pairs which only screens for candidates)
# ──────────────────────────────────────────────

class PairTradeBacktestInput(BaseModel):
    symbol_a: str = Field(..., description="First leg ticker.")
    symbol_b: str = Field(..., description="Second leg ticker.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    hedge_ratio: float = Field(
        ..., description="spread = Close_a - hedge_ratio * Close_b — typically the hedge_ratio from run_cointegration_test.",
    )
    entry_z: float = Field(2.0, description="Enter the spread when |z-score| >= entry_z.")
    exit_z: float = Field(0.5, description="Exit to flat once |z-score| <= exit_z.")
    zscore_window: Optional[int] = Field(
        None, description="Rolling window for the spread z-score. None = full-sample static z-score.",
    )
    initial_capital: float = Field(10_000.0, description="Starting cash for the whole account.")
    commission_pct: float = Field(0.001, description="Commission per trade notional (fraction).")
    slippage_pct: float = Field(0.0005, description="Slippage per trade notional (fraction).")
    gross_leverage: float = Field(
        1.0, description="sum(|weight|) while in a position, split between the two legs to match hedge_ratio.",
    )
    fill_price: str = Field(
        "close",
        description="'close' (default), 'next_open', or 'midpoint' — see run_strategy's fill_price.",
    )


class PairTradeBacktestResult(BaseModel):
    symbol_a: str
    symbol_b: str
    hedge_ratio: float
    n_rebalances: int
    n_round_trips: int
    rebalance_log: List[RebalanceEvent]
    entry_spread: Optional[float] = None
    current_spread: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    final_equity: float
    final_cash: float
    equity_curve: List[float]
    warnings: List[str] = []


# ──────────────────────────────────────────────
# Extended Backtest Diagnostics
# ──────────────────────────────────────────────

class BacktestDiagnosticsInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol (e.g. 'AAPL').")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    strategy_type: str = Field(
        ...,
        description="Strategy type: 'sma_crossover', 'rsi_mean_reversion', "
                    "'macd_crossover', or 'bollinger_reversion'.",
    )
    parameters: Dict[str, Any] = Field(
        {}, description="Strategy parameters — same shape as run_sma_backtest / run_rsi_backtest / etc.",
    )
    initial_capital: float = Field(10_000.0, description="Starting capital.")
    commission_pct: float = Field(0.001, description="Commission per trade (fraction, default 0.1%).")
    slippage_pct: float = Field(0.0005, description="Slippage per trade (fraction, default 0.05%).")
    top_n_drawdowns: int = Field(5, description="Number of worst drawdown episodes to return.")
    fill_price: str = Field("close", description="'close' (default) or 'next_open' — see BacktestInput.fill_price.")


class DrawdownEpisode(BaseModel):
    start: str
    trough: str
    end: Optional[str] = None       # None if still underwater at the end of the series
    depth: float                    # negative fraction, e.g. -0.15 = -15%
    duration_bars: int              # peak -> recovery (or peak -> last bar if unrecovered)
    recovery_bars: Optional[int] = None   # trough -> recovery; None if unrecovered


class TradeDiagnostics(BaseModel):
    expectancy_pct: float
    avg_winner_pct: float
    avg_loser_pct: float
    payoff_ratio: float             # can be inf if there are no losing trades
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_mae_pct: float              # average maximum adverse excursion across trades
    avg_mfe_pct: float              # average maximum favorable excursion across trades


class ExposureDiagnostics(BaseModel):
    time_in_market: float
    avg_gross_exposure: float
    avg_net_exposure: float
    pct_long: float
    pct_short: float
    avg_holding_period_bars: Optional[float] = None


class BacktestDiagnosticsResult(BaseModel):
    symbol: str
    strategy_type: str
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    num_trades: int
    top_drawdowns: List[DrawdownEpisode]
    trade_diagnostics: TradeDiagnostics
    exposure: ExposureDiagnostics
