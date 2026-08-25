from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ──────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────


# A parameter grid is evaluated as the CARTESIAN PRODUCT of its axes, so its
# cost is multiplicative in the number of axes and an agent writing a
# reasonable-looking request can ask for an unreasonable amount of work:
# four axes of ten values each is 10,000 full backtests from a dict that fits
# on one line. Estimator complexity was already bounded; the NUMBER of
# estimator invocations was not.
_MAX_GRID_COMBINATIONS = 50_000


def _validate_param_grid(grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Reject an empty axis and a combinatorially oversized grid."""
    if not grid:
        raise ValueError("param_grid must contain at least one parameter")
    combinations = 1
    for name, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"param_grid['{name}'] must be a non-empty list of values to try"
            )
        combinations *= len(values)
    if combinations > _MAX_GRID_COMBINATIONS:
        axes = ", ".join(f"{k}({len(v)})" for k, v in grid.items())
        raise ValueError(
            f"param_grid expands to {combinations:,} combinations "
            f"[{axes}], above the {_MAX_GRID_COMBINATIONS:,} limit. A grid is "
            "evaluated as the cartesian product of its axes, so the cost is "
            "multiplicative — this is a full backtest per combination. Narrow "
            "an axis or search in stages."
        )
    return grid


class BacktestInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol (e.g. 'AAPL').")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    strategy_type: Literal[
        # The eight entries of STRATEGY_REGISTRY...
        "sma_crossover",
        "rsi_mean_reversion",
        "macd_crossover",
        "bollinger_reversion",
        "donchian_breakout",
        "momentum_timeseries",
        "vwap_reversion",
        "adx_trend",
        # ...plus the two synthetic labels that reuse this same input model
        # without going through the registry: buy_and_hold constructs an
        # always-long series directly, and custom_signal carries a
        # caller-supplied one. They are part of the accepted set, so the
        # Literal has to name them or those tools become unreachable.
        "buy_and_hold",
        "custom_signal",
    ] = Field(
        # Was a bare `str` whose description listed only four of the eight
        # registered strategies, so an agent could both pass an unregistered
        # name (failing deep inside dispatch) and fail to discover half the
        # registry from the schema.
        ...,
        description=(
            "Strategy to run. One of: sma_crossover {fast_period, "
            "slow_period}; rsi_mean_reversion {period, oversold, "
            "overbought}; macd_crossover {fast, slow, signal}; "
            "bollinger_reversion {period, num_std}; donchian_breakout "
            "{entry_period, exit_period}; momentum_timeseries {lookback, "
            "threshold}; vwap_reversion {period, entry_threshold}; "
            "adx_trend {adx_period, adx_threshold}. The labels buy_and_hold "
            "and custom_signal are also accepted by the tools that build "
            "their signal series directly rather than from the registry."
        ),
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
    initial_capital: float = Field(
        10_000.0, gt=0, le=1e15, description="Starting capital."
    )
    commission_pct: float = Field(
        0.001, ge=0, le=1, description="Commission per trade (fraction, default 0.1%)."
    )
    slippage_pct: float = Field(
        0.0005, ge=0, le=1, description="Slippage per trade (fraction, default 0.05%)."
    )
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field(
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
    position_size: float = Field(
        1.0,
        description=(
            "The actual executed signal value held during this trade (its sign gives "
            "`direction`) — e.g. 2.5 for a SCORE signal sized at 2.5x leverage, or "
            "exactly 1.0/-1.0 for a DIRECTION signal. return_pct already scales with "
            "this value; it is exposed here so a fractional or leveraged position size "
            "is visible in the trade log itself, not just implicit in return_pct."
        ),
    )
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
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Caveats the engine raised about this simulation — most "
            "importantly the fill_price='close' look-ahead warning, which "
            "says the result may assume a fill at the very close that "
            "produced the signal. run_strategy has always emitted these; "
            "this model used to drop them, so the engine knew the backtest "
            "might contain look-ahead while the agent-facing output did not."
        ),
    )


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
    indicators: List[
        Literal[
            "sma",
            "ema",
            "macd",
            "rsi",
            "stochastic",
            "bollinger",
            "atr",
            "obv",
            "vwap",
            "adx",
            "williams_r",
        ]
    ] = Field(
        ["rsi", "macd", "bollinger", "atr"],
        min_length=1,
        max_length=11,
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
    weights: List[float] = Field(
        ..., description="Portfolio weights (must sum to 1.0)."
    )
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
# Portfolio Optimization (portfolio/optimize.py — Markowitz mean-variance,
# risk parity, and Black-Litterman; produces weights, unlike PortfolioInput/
# Result above which only scores weights already chosen)
# ──────────────────────────────────────────────


class BLViewInput(BaseModel):
    assets: Dict[str, float] = Field(
        ...,
        description=(
            "Ticker -> pick coefficient. {'AAPL': 1.0} is an absolute view "
            "on AAPL; {'AAPL': 1.0, 'MSFT': -1.0} is a relative view "
            "('AAPL will outperform MSFT by view_return')."
        ),
    )
    view_return: float = Field(
        ..., description="Annualized expected return implied by this view."
    )
    confidence: float = Field(
        1.0,
        gt=0,
        le=1,
        description=(
            "1.0 (default) uses the standard He-Litterman view uncertainty. "
            "Lower values widen this view's uncertainty proportionally, "
            "letting it move the posterior less — a documented "
            "simplification of Idzorek's (2005) confidence-scaling method."
        ),
    )


class PortfolioOptimizationInput(BaseModel):
    tickers: List[str] = Field(..., description="Universe of tickers to optimize over.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    method: Literal[
        "max_sharpe",
        "min_volatility",
        "target_return",
        "target_volatility",
        "risk_parity",
        "black_litterman",
    ] = Field("max_sharpe", description="Optimization method.")
    risk_free_rate: float = Field(
        0.0,
        description="Annualized rate — used by max_sharpe and the Sharpe ratio reported for every method.",
    )
    target_return: Optional[float] = Field(
        None, description="Required (annualized) for method='target_return'."
    )
    target_volatility: Optional[float] = Field(
        None, description="Required (annualized) for method='target_volatility'."
    )
    allow_short: bool = Field(
        False, description="Mean-variance methods only: allow negative weights."
    )
    max_weight: Optional[float] = Field(
        None,
        description="Mean-variance methods only: per-asset weight cap. Setting this requires scipy.",
    )
    risk_budget: Optional[Dict[str, float]] = Field(
        None,
        description="risk_parity only: ticker -> target fractional risk contribution, must sum to 1.0. None = equal risk contribution.",
    )
    market_weights: Optional[Dict[str, float]] = Field(
        None,
        description="black_litterman only: ticker -> prior/market-cap weight. None = equal weight.",
    )
    views: Optional[List[BLViewInput]] = Field(
        None,
        description="black_litterman only: at least one view is required for this method.",
    )
    risk_aversion: float = Field(
        2.5,
        gt=0,
        description="black_litterman only: market risk-aversion coefficient (delta).",
    )
    tau: float = Field(
        0.05,
        gt=0,
        description="black_litterman only: confidence in the equilibrium prior (smaller = more confident).",
    )
    periods_per_year: int = Field(
        252,
        gt=0,
        le=31_536_000,
        description="Annualization factor for the fetched return series.",
    )

    @model_validator(mode="after")
    def _check_method_requirements(self) -> "PortfolioOptimizationInput":
        # A repeated ticker silently shrinks the problem: the returns frame
        # is built as {ticker: close}, so duplicate keys collapse and the
        # optimizer sees fewer assets than were asked for. The response then
        # echoed the full requested list as `tickers` while `weights` had one
        # entry per SURVIVING column -- ['AAA','BBB','AAA'] came back with
        # two weights, so the two fields of one result disagreed about the
        # size of the universe. Rejected rather than de-duplicated, since a
        # duplicate means the caller's intent is unclear (double weight? a
        # typo?) and guessing is how the two fields drifted apart to begin
        # with.
        duplicates = sorted({t for t in self.tickers if self.tickers.count(t) > 1})
        if duplicates:
            raise ValueError(
                f"tickers contains duplicate symbols: {duplicates}. Each asset "
                "must appear once — a repeated ticker collapses to a single "
                "column and the returned weights would not line up with the "
                "requested universe."
            )
        if self.method == "target_return" and self.target_return is None:
            raise ValueError("method='target_return' requires target_return")
        if self.method == "target_volatility" and self.target_volatility is None:
            raise ValueError("method='target_volatility' requires target_volatility")
        if self.method == "black_litterman" and not self.views:
            raise ValueError("method='black_litterman' requires at least one view")
        if self.risk_budget is not None:
            missing = [t for t in self.tickers if t not in self.risk_budget]
            if missing:
                raise ValueError(f"risk_budget is missing entries for: {missing}")
        if self.market_weights is not None:
            missing = [t for t in self.tickers if t not in self.market_weights]
            if missing:
                raise ValueError(f"market_weights is missing entries for: {missing}")
        if self.views is not None:
            unknown = sorted(
                {a for v in self.views for a in v.assets if a not in self.tickers}
            )
            if unknown:
                raise ValueError(f"views reference tickers not in tickers: {unknown}")
        return self


class PortfolioOptimizationResult(BaseModel):
    tickers: List[str]
    method: str
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    converged: bool
    risk_contributions: Optional[Dict[str, float]] = Field(
        None,
        description="risk_parity only: fractional contribution to total variance per asset, sums to 1.",
    )
    warnings: List[str] = []


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
    start_date: Optional[str] = Field(
        None, description="Historical start for technicals."
    )
    end_date: Optional[str] = Field(None, description="Historical end for technicals.")
    sort_by: Optional[str] = Field(None, description="Column to sort results by.")
    ascending: bool = Field(True, description="Sort direction.")
    min_beta_obs: int = Field(
        20,
        ge=2,
        le=10_000,
        description=(
            "Minimum bars a ticker must share with the benchmark before a "
            "beta_max/beta_min filter acts on its estimate. Below it the "
            "ticker is reported in failed_tickers rather than given a beta of "
            "0.0 — which would PASS a beta_max screen, reading 'could not be "
            "estimated' as 'very low beta'. Default 20 is a judgment call, not "
            "a mathematical bound: lower it for weekly bars or a deliberate "
            "recent-listing screen. Hard minimum 2, below which the underlying "
            "beta routine returns a sentinel indistinguishable from a real 0.0."
        ),
    )


class ScreenerResult(BaseModel):
    num_passed: int
    tickers_passed: List[str]
    results: List[Dict[str, Any]]
    failed_filters: Dict[str, str] = Field(
        default_factory=dict,
        description="ticker -> the specific filter key it failed (genuine rejection).",
    )
    failed_tickers: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "ticker -> error message for a data-fetch/compute exception — kept "
            "separate from failed_filters so a broken data fetch is never "
            "indistinguishable from a ticker that simply didn't meet the bar."
        ),
    )
    failed_batches: List[str] = Field(
        default_factory=list,
        description=(
            "Error message per worker-process batch that raised before returning "
            "any per-ticker result (n_workers > 1 only) — a crashed batch is never "
            "silently discarded without a trace."
        ),
    )


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
        gt=0,
        le=100_000,
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
    signal: str  # "long_a_short_b" | "short_a_long_b" | "neutral"
    n_obs: int


# ──────────────────────────────────────────────
# Kalman-Filter Dynamic Hedge Ratio
# ──────────────────────────────────────────────


class KalmanHedgeRatioInput(BaseModel):
    symbol_a: str = Field(..., description="First asset symbol (the 'long' leg).")
    symbol_b: str = Field(..., description="Second asset symbol (the 'short' leg).")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    delta: float = Field(
        1e-4,
        gt=0,
        lt=1,
        description=(
            "Controls how fast the hedge ratio is allowed to drift. Smaller "
            "= slower-adapting/more stable (closer to a static OLS ratio); "
            "larger = faster-adapting/noisier."
        ),
    )
    zscore_window: int = Field(
        20,
        le=100_000,
        gt=1,
        description="Rolling window (bars) for the spread z-score used to generate a signal.",
    )


class KalmanHedgeRatioResult(BaseModel):
    symbol_a: str
    symbol_b: str
    current_hedge_ratio: float
    current_intercept: float
    hedge_ratio_std: float
    spread_mean: float
    spread_std: float
    current_zscore: float
    signal: str  # "long_a_short_b" | "short_a_long_b" | "neutral"
    n_obs: int


# ──────────────────────────────────────────────
# PCA
# ──────────────────────────────────────────────


class PCAInput(BaseModel):
    tickers: List[str] = Field(..., description="Universe of tickers to decompose.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    n_components: int = Field(
        3, description="Number of principal components to extract (default 3)."
    )

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
    explained_variance_ratio: Dict[str, float]  # {"PC1": 0.42, "PC2": 0.12, ...}
    cumulative_variance_ratio: Dict[str, float]  # {"PC1": 0.42, "PC2": 0.54, ...}
    loadings: Dict[str, Dict[str, float]]  # {"PC1": {"AAPL": 0.35, ...}, ...}
    factor_contributions: Dict[
        str, Dict[str, float]
    ]  # {"AAPL": {"PC1": 0.38, ...}, ...}


# ──────────────────────────────────────────────
# Correlation & Diversification Analytics
# ──────────────────────────────────────────────


class CorrelationAnalysisInput(BaseModel):
    tickers: List[str] = Field(..., description="Universe of tickers (>= 2).")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    weights: Optional[List[float]] = Field(
        None,
        description=(
            "Portfolio weights for the diversification ratio, same order as "
            "tickers, must sum to 1.0. None (default) uses equal weighting."
        ),
    )

    @field_validator("tickers")
    @classmethod
    def _check_min_tickers(cls, v: List[str]) -> List[str]:
        if len(v) < 2:
            raise ValueError(f"tickers must contain at least 2 symbols, got {len(v)}")
        return v

    @model_validator(mode="after")
    def _check_weights(self) -> "CorrelationAnalysisInput":
        if self.weights is not None:
            if len(self.weights) != len(self.tickers):
                raise ValueError(
                    f"len(weights)={len(self.weights)} must equal len(tickers)={len(self.tickers)}"
                )
            total = sum(self.weights)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"weights must sum to 1.0, got {total:.8f}")
        return self


class CorrelationAnalysisResult(BaseModel):
    tickers: List[str]
    correlation_matrix: Dict[str, Dict[str, float]]
    avg_pairwise_correlation: float
    highest_correlated_pair: Dict[str, Any]
    lowest_correlated_pair: Dict[str, Any]
    diversification_ratio: float


# ──────────────────────────────────────────────
# Hurst Exponent
# ──────────────────────────────────────────────


class HurstInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    method: Literal["dfa", "rs"] = Field(
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
    regime: str  # "trending" | "random_walk" | "mean_reverting"
    fit_r_squared: float
    method: str
    n_obs: int
    rolling_current: Optional[float] = None
    rolling_regime_fractions: Optional[Dict[str, float]] = None


# ──────────────────────────────────────────────
# Rally Detection
# ──────────────────────────────────────────────


class RallyDetectionInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    lookback: int = Field(
        20,
        gt=0,
        description="Trailing-return window in bars (default ~1 trading month).",
    )
    zscore_window: int = Field(
        252,
        le=100_000,
        gt=0,
        description="Historical window the return z-score is measured against (default ~1 trading year).",
    )
    adx_period: int = Field(
        14, le=100_000, gt=0, description="ADX lookback (default 14)."
    )
    adx_threshold: float = Field(
        25.0, gt=0, description="ADX level considered a 'strong' trend."
    )
    breakout_period: int = Field(
        20, gt=0, description="Bars for the new-high breakout check."
    )
    hurst_method: Literal["dfa", "rs"] = Field(
        "dfa", description="Estimation method passed through to the Hurst regime check."
    )
    auto_tune_adx_threshold: bool = Field(
        False,
        description=(
            "If True, ignore adx_threshold and instead use the "
            "auto_tune_percentile-th percentile of this symbol's OWN "
            "trailing ADX history as the 'strong trend' bar -- a "
            "chronically choppy stock and a chronically trending one each "
            "get a threshold calibrated to their own history, rather than "
            "one fixed number for every asset. Default False: unchanged, "
            "exact prior behavior."
        ),
    )
    auto_tune_percentile: float = Field(
        60.0,
        gt=0.0,
        lt=100.0,
        description=(
            "Percentile of this symbol's own historical ADX distribution "
            "used when auto_tune_adx_threshold=True. Default 60 -- "
            "'stronger than most of this symbol's own recent bars,' a "
            "deliberately modest bar. Unused otherwise."
        ),
    )


class RallyDetectionResult(BaseModel):
    symbol: str
    is_rally: bool
    rally_score: float
    trailing_return_pct: float
    return_zscore: float
    adx: float
    di_plus: float
    di_minus: float
    trend_direction: str  # "bullish" | "bearish" | "neutral"
    hurst: float
    regime: str  # "trending" | "random_walk" | "mean_reverting"
    is_new_high: bool
    n_obs: int
    adx_threshold_used: float
    auto_tuned: bool


# ──────────────────────────────────────────────
# Realized Volatility Estimators (Parkinson, Garman-Klass, Yang-Zhang)
# ──────────────────────────────────────────────


class VolatilityEstimatorsInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    period: int = Field(
        20, gt=1, description="Rolling window in bars for every estimator."
    )


class VolatilityEstimatorsResult(BaseModel):
    symbol: str
    period: int
    close_to_close_annualized: float
    parkinson_annualized: float
    garman_klass_annualized: float
    yang_zhang_annualized: float
    yang_zhang_vs_close_to_close_ratio: float


# ──────────────────────────────────────────────
# GARCH(1,1) Conditional Volatility Forecast
# ──────────────────────────────────────────────


class GarchVolatilityForecastInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    forecast_horizon: int = Field(
        10, gt=0, le=252, description="Periods ahead to forecast."
    )


class GarchVolatilityForecastResult(BaseModel):
    symbol: str
    omega: float
    alpha: float
    beta: float
    persistence: float
    converged: bool
    current_annualized_vol: float
    long_run_annualized_vol: float
    forecast_annualized_vol: List[float]
    log_likelihood: float
    aic: float
    bic: float
    n_obs: int


# ──────────────────────────────────────────────
# Regime-Adaptive Strategy Selector
# ──────────────────────────────────────────────


class RegimeAdaptiveInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    initial_capital: float = Field(
        10_000.0, gt=0, le=1e15, description="Starting capital."
    )
    commission_pct: float = Field(
        0.001, ge=0, le=1, description="Commission per trade (fraction)."
    )
    slippage_pct: float = Field(
        0.0005, ge=0, le=1, description="Slippage per trade (fraction)."
    )
    hurst_method: Literal["dfa", "rs"] = Field(
        "dfa", description="Hurst method: 'dfa' or 'rs'."
    )
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
    n_workers: int = Field(
        1,
        ge=1,
        le=256,
        description="Worker processes for grid search (default 1 for agent use).",
    )


class RegimeAdaptiveResult(BaseModel):
    symbol: str
    regime: str  # "trending" | "random_walk" | "mean_reverting"
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
    tickers: List[str] = Field(
        ..., description="Universe of tickers to test for cointegration."
    )
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    max_pairs: int = Field(
        10, gt=0, le=10_000, description="Maximum number of top pairs to return."
    )
    min_half_life: float = Field(
        5.0, gt=0, le=100_000, description="Minimum mean-reversion half-life in bars."
    )
    max_half_life: float = Field(
        126.0, gt=0, le=100_000, description="Maximum half-life in bars (~6 months)."
    )
    p_value_threshold: float = Field(
        0.05, gt=0, le=1, description="Maximum cointegration p-value."
    )
    zscore_window: int = Field(
        30, gt=0, le=100_000, description="Rolling window for spread z-score signal."
    )


class PairResult(BaseModel):
    symbol_a: str
    symbol_b: str
    p_value: float
    hedge_ratio: float
    half_life_days: float
    adf_statistic: float
    current_zscore: float
    signal: str  # "long_a_short_b" | "short_a_long_b" | "neutral"


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
    failed_tickers: Dict[str, str] = {}  # ticker -> fetch-error message


# ──────────────────────────────────────────────
# Walk-Forward Backtest
# ──────────────────────────────────────────────


class WalkForwardInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    strategy: Literal[
        "sma_crossover",
        "rsi_mean_reversion",
        "macd_crossover",
        "bollinger_reversion",
        "donchian_breakout",
        "momentum_timeseries",
        "vwap_reversion",
        "adx_trend",
    ] = Field(
        ...,
        description=(
            "Strategy name: 'sma_crossover', 'rsi_mean_reversion', "
            "'macd_crossover', 'bollinger_reversion', 'donchian_breakout' "
            "(Turtle-style channel breakout), 'momentum_timeseries' "
            "(trailing-return threshold), 'vwap_reversion' (mean reversion "
            "to rolling VWAP — intended for intraday/tick data), or "
            "'adx_trend' (ADX-strength-filtered directional trend)."
        ),
    )
    param_grid: Dict[str, List[Any]] = Field(
        ...,
        description="Parameter grid, e.g. {'fast_period': [5, 10, 20], 'slow_period': [30, 50]}.",
    )
    train_bars: int = Field(
        252,
        gt=0,
        le=100_000,
        description="In-sample window length in bars (default 252 = ~1 year daily).",
    )
    test_bars: int = Field(
        63,
        gt=0,
        le=100_000,
        description="Out-of-sample window length in bars (default 63 = ~1 quarter daily).",
    )
    initial_capital: float = Field(
        10_000.0, gt=0, le=1e15, description="Starting capital for each window."
    )
    commission_pct: float = Field(
        0.001, ge=0, le=1, description="Commission per trade (fraction)."
    )
    slippage_pct: float = Field(
        0.0005, ge=0, le=1, description="Slippage per trade (fraction)."
    )
    sort_by: Literal[
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "total_return",
        "profit_factor",
        "win_rate",
        "max_drawdown",
        "annualized_volatility",
        "avg_trade_return_pct",
        "num_trades",
    ] = Field(
        "sharpe_ratio",
        description="Metric to optimise in-sample (default: 'sharpe_ratio').",
    )
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field(
        "close",
        description="'close' (default), 'next_open', or 'hl2_exploratory' — applied to the out-of-sample leg of every window (see BacktestInput.fill_price).",
    )

    @field_validator("param_grid")
    @classmethod
    def _check_param_grid(cls, v: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        return _validate_param_grid(v)


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
    is_to_oos_sharpe_decay: float  # avg in-sample sharpe minus stitched OOS sharpe
    is_to_oos_return_decay: float  # avg in-sample return minus stitched OOS return
    worst_oos_window: int  # window_index with the lowest out_of_sample_return
    longest_losing_window_streak: int
    parameter_turnover: (
        float  # fraction of consecutive windows whose best_params changed
    )


# ──────────────────────────────────────────────
# Regime-Adaptive Walk-Forward Backtest (leakage-free counterpart to
# RegimeAdaptiveInput/Result — regime detection AND strategy/parameter
# selection happen strictly within each window's training data)
# ──────────────────────────────────────────────


class RegimeAdaptiveWalkForwardInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    train_bars: int = Field(
        252,
        gt=0,
        le=100_000,
        description="In-sample window length in bars (default 252 = ~1 year daily).",
    )
    test_bars: int = Field(
        63,
        gt=0,
        le=100_000,
        description="Out-of-sample window length in bars (default 63 = ~1 quarter daily).",
    )
    initial_capital: float = Field(
        10_000.0, gt=0, le=1e15, description="Starting capital for each window."
    )
    commission_pct: float = Field(
        0.001, ge=0, le=1, description="Commission per trade (fraction)."
    )
    slippage_pct: float = Field(
        0.0005, ge=0, le=1, description="Slippage per trade (fraction)."
    )
    hurst_method: Literal["dfa", "rs"] = Field(
        "dfa",
        description="Hurst method: 'dfa' or 'rs' — reported as diagnostic context per window, not used to hard-select a strategy family.",
    )
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
    sort_by: Literal[
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "total_return",
        "profit_factor",
        "win_rate",
        "max_drawdown",
        "annualized_volatility",
        "avg_trade_return_pct",
        "num_trades",
    ] = Field(
        "sharpe_ratio",
        description="Metric to optimise in-sample, across all four strategies (default: 'sharpe_ratio').",
    )
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field(
        "close",
        description="'close' (default), 'next_open', or 'hl2_exploratory' — applied to the out-of-sample leg of every window (see BacktestInput.fill_price).",
    )


class RegimeAdaptiveWalkForwardWindow(BaseModel):
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    regime: str  # "trending" | "random_walk" | "mean_reverting" | "unknown" — diagnostic context only
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
    strategy_stability: Dict[
        str, Any
    ]  # most common selected_strategy across windows + frequency
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
    asset_risk_contributions: Dict[
        str, float
    ]  # fractional contribution to portfolio vol (sums to 1)
    # PCA decomposition
    pca_variance_explained: Dict[str, float]  # EVR per PC across the asset universe
    portfolio_pc_exposures: Dict[str, float]  # portfolio's loading on each PC
    # Factor model (optional)
    factor_loadings: Optional[Dict[str, float]] = None
    factor_r_squared: Optional[float] = None
    factor_alpha: Optional[float] = None


# ──────────────────────────────────────────────
# Historical Stress-Test Replay
# ──────────────────────────────────────────────


class StressTestInput(BaseModel):
    tickers: List[str] = Field(..., description="Portfolio asset symbols.")
    weights: Optional[List[float]] = Field(
        None,
        description="Portfolio weights, same order as tickers, must sum to 1.0. None (default) uses equal weighting.",
    )
    scenario: Literal[
        "black_monday_1987",
        "dotcom_2000",
        "gfc_2008",
        "volmageddon_2018",
        "covid_2020",
        "rate_shock_2022",
        "custom",
    ] = Field(
        "gfc_2008",
        description="Named historical crash window, or 'custom' to supply your own custom_start_date/custom_end_date.",
    )
    custom_start_date: Optional[str] = Field(
        None, description="Required (YYYY-MM-DD) when scenario='custom'."
    )
    custom_end_date: Optional[str] = Field(
        None, description="Required (YYYY-MM-DD) when scenario='custom'."
    )

    @model_validator(mode="after")
    def _check_custom_dates(self) -> "StressTestInput":
        if self.scenario == "custom" and (
            self.custom_start_date is None or self.custom_end_date is None
        ):
            raise ValueError(
                "custom_start_date and custom_end_date are both required when scenario='custom'"
            )
        if self.weights is not None and len(self.weights) != len(self.tickers):
            raise ValueError(
                f"len(weights)={len(self.weights)} must equal len(tickers)={len(self.tickers)}"
            )
        return self


class StressTestResult(BaseModel):
    scenario: str
    scenario_start_date: str
    scenario_end_date: str
    tickers_used: List[str]
    tickers_missing_data: List[str]
    portfolio_return_pct: float
    max_drawdown_pct: float
    worst_day_return_pct: float
    worst_day_date: str
    best_day_return_pct: float
    best_day_date: str
    n_trading_days: int


# ──────────────────────────────────────────────
# ATR-Based Position Sizer
# ──────────────────────────────────────────────


class PositionSizerInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(
        ..., description="Start date YYYY-MM-DD (for ATR calculation)."
    )
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    account_equity: float = Field(..., description="Total account equity in dollars.")
    risk_per_trade_pct: float = Field(
        0.01,
        description="Fraction of account to risk per trade (default 0.01 = 1%). Must be in (0, 1].",
    )
    atr_period: int = Field(
        14, gt=0, le=100_000, description="ATR lookback period (default 14)."
    )
    atr_multiplier: float = Field(
        2.0, description="Stop distance = atr_multiplier × ATR (default 2.0)."
    )
    win_rate: Optional[float] = Field(
        None, description="Strategy win rate [0,1]. Required for Kelly sizing."
    )
    avg_win_pct: Optional[float] = Field(
        None, description="Average winning trade return (e.g. 0.05 = 5%)."
    )
    avg_loss_pct: Optional[float] = Field(
        None, description="Average losing trade return magnitude (e.g. 0.02 = 2%)."
    )

    @field_validator("risk_per_trade_pct")
    @classmethod
    def _check_risk_pct(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"risk_per_trade_pct must be in (0, 1], got {v}")
        return v

    @field_validator("win_rate")
    @classmethod
    def _check_win_rate(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError(f"win_rate must be in [0, 1], got {v}")
        return v

    @field_validator("avg_win_pct")
    @classmethod
    def _check_avg_win_pct(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0.0:
            raise ValueError(f"avg_win_pct must be >= 0, got {v}")
        return v

    @field_validator("avg_loss_pct")
    @classmethod
    def _check_avg_loss_pct(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0.0:
            raise ValueError(
                f"avg_loss_pct must be > 0 (it's a magnitude, used as a Kelly-formula divisor), got {v}"
            )
        return v


class PositionSizerResult(BaseModel):
    symbol: str
    last_close: float
    atr: float
    atr_pct: float  # ATR as % of price
    stop_distance: float  # atr_multiplier × ATR in $
    # Fixed-risk (ATR-based) sizing
    shares_fixed_risk: int
    position_value_fixed_risk: float
    portfolio_pct_fixed_risk: float
    max_loss_fixed_risk: float  # worst-case $ loss if stop is hit
    # Kelly sizing (populated when win_rate/avg_win/avg_loss are provided)
    kelly_fraction: Optional[float] = None
    shares_half_kelly: Optional[int] = None
    position_value_half_kelly: Optional[float] = None
    portfolio_pct_half_kelly: Optional[float] = None
    # Recommendation
    recommended_sizing: str  # "fixed_risk" | "half_kelly"
    recommended_shares: int
    recommended_position_value: float


# ──────────────────────────────────────────────
# Buy-and-Hold Baseline
# ──────────────────────────────────────────────


class BuyAndHoldInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    initial_capital: float = Field(
        10_000.0, gt=0, le=1e15, description="Starting capital."
    )
    commission_pct: float = Field(
        0.001,
        ge=0,
        le=1,
        description="One-time buy commission (fraction, default 0.1%).",
    )
    slippage_pct: float = Field(
        0.0005,
        ge=0,
        le=1,
        description="One-time buy slippage (fraction, default 0.05%).",
    )
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field(
        "close",
        description="'close' (default) or 'next_open' — see BacktestInput.fill_price.",
    )


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
    initial_capital: float = Field(
        10_000.0, gt=0, le=1e15, description="Starting capital."
    )
    commission_pct: float = Field(
        0.001, ge=0, le=1, description="Commission per trade (fraction)."
    )
    slippage_pct: float = Field(
        0.0005, ge=0, le=1, description="Slippage per trade (fraction)."
    )
    sort_by: Literal[
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "total_return",
        "profit_factor",
        "win_rate",
        "max_drawdown",
        "annualized_volatility",
        "avg_trade_return_pct",
        "num_trades",
    ] = Field(
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
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field(
        "close",
        description="'close' (default) or 'next_open' — see BacktestInput.fill_price.",
    )


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
    strategy: Literal[
        "sma_crossover",
        "rsi_mean_reversion",
        "macd_crossover",
        "bollinger_reversion",
        "donchian_breakout",
        "momentum_timeseries",
        "vwap_reversion",
        "adx_trend",
    ] = Field(
        ...,
        description=(
            "Strategy to optimise: 'sma_crossover', 'rsi_mean_reversion', "
            "'macd_crossover', 'bollinger_reversion', 'donchian_breakout', "
            "'momentum_timeseries', 'vwap_reversion', or 'adx_trend'."
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
    initial_capital: float = Field(
        10_000.0, gt=0, le=1e15, description="Starting capital."
    )
    commission_pct: float = Field(
        0.001, ge=0, le=1, description="Commission per trade (fraction, default 0.1%)."
    )
    slippage_pct: float = Field(
        0.0005, ge=0, le=1, description="Slippage per trade (fraction, default 0.05%)."
    )
    sort_by: Literal[
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "total_return",
        "profit_factor",
        "win_rate",
        "max_drawdown",
        "annualized_volatility",
        "avg_trade_return_pct",
        "num_trades",
    ] = Field(
        "sharpe_ratio",
        description=(
            "Metric to optimise. "
            "Options: 'sharpe_ratio', 'total_return', 'calmar_ratio', 'sortino_ratio', 'max_drawdown'."
        ),
    )
    top_n: int = Field(
        5,
        description="Number of top parameter combinations to return (default 5, max 20).",
    )
    n_workers: int = Field(
        1, ge=1, le=256, description="CPU workers for parallel grid search (default 1)."
    )
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field(
        "close",
        description="'close' (default) or 'next_open' — see BacktestInput.fill_price.",
    )

    @field_validator("param_grid")
    @classmethod
    def _check_param_grid(cls, v: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        return _validate_param_grid(v)


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
    atr_period: int = Field(
        14, gt=0, le=100_000, description="Wilder ATR period (default 14)."
    )
    sar_af_start: float = Field(
        0.02, description="Parabolic SAR initial acceleration factor (default 0.02)."
    )
    sar_af_max: float = Field(
        0.2, description="Parabolic SAR maximum acceleration factor (default 0.2)."
    )


class AdvancedIndicatorsResult(BaseModel):
    symbol: str
    last_close: float
    sar_value: float
    sar_trend: str  # "bullish" | "bearish"
    sar_signal: str  # "buy" | "sell"
    wilder_atr: float
    wilder_atr_pct: float
    mfi: float
    mfi_signal: str  # "overbought" | "oversold" | "neutral"


# ──────────────────────────────────────────────
# Rolling Beta
# ──────────────────────────────────────────────


class RollingBetaInput(BaseModel):
    symbol: str = Field(..., description="Asset ticker symbol.")
    benchmark: str = Field("SPY", description="Benchmark symbol (default SPY).")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    window: int = Field(
        60, description="Rolling window in bars (default 60 ≈ 3 months daily)."
    )


class RollingBetaResult(BaseModel):
    symbol: str
    benchmark: str
    window: int
    current_beta: float
    beta_1m_ago: Optional[float]
    beta_3m_ago: Optional[float]
    beta_6m_ago: Optional[float]
    beta_trend: str  # "increasing" | "decreasing" | "stable"
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
# EVT Tail Risk (Peaks-Over-Threshold)
# ──────────────────────────────────────────────


class TailRiskInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    confidence: float = Field(
        0.99, gt=0.5, lt=1.0, description="VaR/CVaR confidence level."
    )
    tail_fraction: float = Field(
        0.05,
        gt=0.0,
        lt=0.5,
        description="Fraction of observations (by loss) treated as the tail for threshold selection.",
    )
    method: Literal["pwm", "mle"] = Field(
        "pwm",
        description=(
            "'pwm' (default, closed-form, no dependencies) or 'mle' "
            "(maximum likelihood, requires scipy, more statistically "
            "efficient but iterative)."
        ),
    )


class TailRiskResult(BaseModel):
    symbol: str
    confidence: float
    threshold_daily_loss_pct: float
    n_exceedances: int
    n_obs: int
    shape_xi: float
    scale_beta: float
    var_evt: float
    cvar_evt: float
    var_historical_comparison: float
    method: str
    tail_classification: str


# ──────────────────────────────────────────────
# Custom Signal Backtest (bring-your-own signal)
# ──────────────────────────────────────────────


class SignalType(str, Enum):
    """
    What a custom signal's numeric values mean, and how strictly they're
    validated. run_strategy's math is unchanged regardless of signal_type:
    it always multiplies the (lagged) signal value directly into
    strategy_return = lagged_signal * market_return — SCORE's "unrestricted"
    values are not a normalized "confidence score" in that multiplication,
    they are a literal leverage multiplier (a value of 10 means a 10x
    position). CustomSignalBacktestInput (single-asset) defaults to
    DIRECTION for this reason — an LLM-facing tool should not silently
    accept an arbitrary "score" as if it were a bounded confidence value.
    SignalPanelBacktestInput and PortfolioSimulationInput still default to
    SCORE: in both, a SCORE value is converted into a bounded weight via an
    explicit construction_method (backtest/sizing.py) before it ever
    reaches a return calculation, so the same hazard doesn't apply there.
    """

    SCORE = "score"  # unrestricted float — caller owns the scale/leverage semantics
    DIRECTION = "direction"  # must be exactly -1, 0, or 1 (within 1e-9)
    TARGET_WEIGHT = "target_weight"  # must satisfy |value| <= max_abs_weight


def _validate_signal_values(
    values: Dict[Any, float], signal_type: "SignalType", max_abs_weight: float
) -> None:
    if signal_type == SignalType.DIRECTION:
        bad = {
            k: v
            for k, v in values.items()
            if min(abs(v - d) for d in (-1.0, 0.0, 1.0)) > 1e-9
        }
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
        SignalType.DIRECTION,
        description=(
            "'direction' (default — every value must be exactly -1, 0, or 1) | "
            "'target_weight' (every |value| must be <= max_abs_weight) | 'score' "
            "(unrestricted float, multiplied directly into strategy_return = "
            "lagged_signal * market_return — a value of 10 means a 10x position, "
            "not '10x more bullish'; only use this if you have already converted "
            "your own alpha model's output into a leverage multiplier yourself)."
        ),
    )
    max_abs_weight: float = Field(
        1.0,
        description="Bound used only when signal_type='target_weight' (ignored otherwise).",
    )
    signal_fill_policy: Literal["hold", "flat", "error"] = Field(
        "hold",
        description=(
            "How to extend a sparse signal map (e.g. monthly dates) onto the full "
            "daily price calendar before backtesting: 'hold' (default) forward-fills "
            "between submitted dates (flat before the first one) — correct for a "
            "target-position signal meant to persist until changed. 'flat' does not "
            "forward-fill; only the exact submitted dates carry a nonzero signal. "
            "'error' requires every price date to have an explicit signal entry."
        ),
    )
    initial_capital: float = Field(
        10_000.0, gt=0, le=1e15, description="Starting capital."
    )
    commission_pct: float = Field(
        0.001, ge=0, le=1, description="Commission per trade (fraction, default 0.1%)."
    )
    slippage_pct: float = Field(
        0.0005, ge=0, le=1, description="Slippage per trade (fraction, default 0.05%)."
    )
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field(
        "close",
        description="'close' (default) or 'next_open' — see BacktestInput.fill_price.",
    )

    @model_validator(mode="after")
    def _check_signal_values(self) -> "CustomSignalBacktestInput":
        _validate_signal_values(self.signals, self.signal_type, self.max_abs_weight)
        return self


# ──────────────────────────────────────────────
# Signal Panel Backtest (bring-your-own multi-ticker signal matrix)
# ──────────────────────────────────────────────


class SignalPanelBacktestInput(BaseModel):
    tickers: List[str] = Field(
        ..., description="Ticker universe. Must match signal_panel's outer keys."
    )
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
    signal_fill_policy: Literal["hold", "flat", "error"] = Field(
        "hold",
        description=(
            "How to extend each ticker's sparse signal map onto its full daily price "
            "calendar before backtesting: 'hold' (default) forward-fills between "
            "submitted dates (flat before the first one). 'flat' does not "
            "forward-fill; only the exact submitted dates carry a nonzero signal. "
            "'error' requires every price date to have an explicit signal entry."
        ),
    )
    initial_capital: float = Field(
        10_000.0, gt=0, le=1e15, description="Starting capital applied per ticker."
    )
    commission_pct: float = Field(
        0.001, ge=0, le=1, description="Commission per trade (fraction)."
    )
    slippage_pct: float = Field(
        0.0005, ge=0, le=1, description="Slippage per trade (fraction)."
    )
    benchmark: Optional[str] = Field(
        None,
        description="Optional benchmark ticker — adds information_ratio to portfolio_metrics.",
    )
    include_trade_log: bool = Field(
        False, description="If True, include a per-trade log for each ticker."
    )
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field(
        "close",
        description="'close' (default) or 'next_open' — see BacktestInput.fill_price.",
    )
    signal_type: SignalType = Field(
        SignalType.SCORE,
        description=(
            "'score' (default, unrestricted — today's exact behavior) | 'direction' "
            "(every value must be exactly -1, 0, or 1) | 'target_weight' (every "
            "|value| must be <= max_abs_weight). Applies uniformly to every ticker's signal map."
        ),
    )
    max_abs_weight: float = Field(
        1.0,
        description="Bound used only when signal_type='target_weight' (ignored otherwise).",
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
                _validate_signal_values(
                    self.signal_panel[ticker], self.signal_type, self.max_abs_weight
                )
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
    "rank_weighted",
    "equal_weight_top_bottom",
    "zscore_normalized",
    "vol_scaled",
)


class PortfolioSimulationInput(BaseModel):
    tickers: List[str] = Field(
        ..., description="Ticker universe. Must match target_weights' outer keys."
    )
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
        1.0,
        gt=0,
        description="Target sum(|weight|) per date when signal_type='score' (ignored otherwise).",
    )
    n_long: Optional[int] = Field(
        None,
        ge=0,
        description="Required when construction_method='equal_weight_top_bottom'.",
    )
    n_short: Optional[int] = Field(
        None,
        ge=0,
        description="Required when construction_method='equal_weight_top_bottom'.",
    )
    vol_lookback: int = Field(
        20,
        gt=0,
        description="Rolling window (bars) used when construction_method='vol_scaled'.",
    )
    make_dollar_neutral: bool = Field(
        False,
        description=(
            "If True, post-process constructed weights so sum(weight)==0 per date "
            "(backtest/sizing.py's dollar_neutral). Only applies when signal_type='score'."
        ),
    )
    initial_capital: float = Field(
        10_000.0, le=1e15, gt=0, description="Starting cash for the whole account."
    )
    commission_pct: float = Field(
        0.001, le=1, ge=0, description="Commission per trade notional (fraction)."
    )
    sell_commission_pct: Optional[float] = Field(
        None,
        le=1,
        ge=0,
        description=(
            "Separate commission rate for SALES. None (the default) charges "
            "commission_pct on both sides. Real venues are frequently "
            "asymmetric — regulatory fees in several markets are sell-side "
            "only — and a symmetric rate understates the cost of a strategy "
            "that turns over in one direction more than the other."
        ),
    )
    slippage_pct: float = Field(
        0.0005, le=1, ge=0, description="Slippage per trade notional (fraction)."
    )
    max_gross_leverage: float = Field(
        1.0,
        gt=0,
        description=(
            "Reject any rebalance date whose sum(|weight|) exceeds this (default 1.0 = "
            "fully invested, no leverage). Bounds the TARGET weights / sizing basis, not "
            "realized post-cost leverage: transaction costs mechanically inflate the "
            "reported gross_leverage_after (rebalance_log/leverage_curve) above this "
            "limit whenever costs are nonzero, and that is expected, not rejected — see "
            "run_portfolio_simulation's docstring ('Post-trade enforcement') for why."
        ),
    )
    max_position_pct: float = Field(
        1.0,
        gt=0,
        description=(
            "Reject any single position whose |weight| exceeds this. Same target-weight/"
            "sizing-basis scope as max_gross_leverage — see its description."
        ),
    )
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field(
        "close",
        description="'close' (default), 'next_open', or 'hl2_exploratory' — see run_strategy's fill_price / the True Portfolio Simulation docs.",
    )
    commission_model: str = Field(
        "pct",
        description="'pct' (default — commission_pct * notional) or 'per_share' (per_share_rate per share, floored at min_commission).",
    )
    per_share_rate: float = Field(
        0.0,
        ge=0,
        description="Commission per share traded. Only used when commission_model='per_share'.",
    )
    min_commission: float = Field(
        0.0,
        ge=0,
        description="Minimum commission per rebalance leg. Only used when commission_model='per_share'.",
    )
    use_impact_model: bool = Field(
        False,
        description="If True, add a square-root market-impact cost on top of commission + spread (backtest/costs.py).",
    )
    impact_coefficient: float = Field(
        1.0,
        ge=0,
        description="Market-impact model coefficient. Only used when use_impact_model=True.",
    )
    impact_lookback: int = Field(
        20,
        gt=0,
        description="Rolling window (bars) for average dollar volume / volatility used by the impact model.",
    )
    borrow_fee_bps: float = Field(
        0.0,
        ge=0,
        description="Annualized basis-point borrow fee accrued daily on any short position's notional.",
    )
    margin_interest_rate: float = Field(
        0.0,
        ge=0,
        description="Annualized rate accrued daily on negative cash (implied margin borrowing).",
    )
    max_adv_participation: Optional[float] = Field(
        None,
        gt=0,
        description="Reject any rebalance trade whose notional exceeds this fraction of the ticker's own rolling average dollar volume. Requires a 'Volume' column.",
    )
    benchmark: Optional[str] = Field(
        None, description="Optional benchmark ticker — adds information_ratio."
    )

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
            _validate_signal_values(
                row, SignalType.TARGET_WEIGHT, self.max_position_pct
            )
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
        ...,
        description="spread = Close_a - hedge_ratio * Close_b — typically the hedge_ratio from run_cointegration_test.",
    )
    entry_z: float = Field(
        2.0, description="Enter the spread when |z-score| >= entry_z."
    )
    exit_z: float = Field(0.5, description="Exit to flat once |z-score| <= exit_z.")
    zscore_window: Optional[int] = Field(
        30,
        description=(
            "Rolling window (bars) for the spread z-score. Defaults to 30 so signals only use "
            "data available up to each bar. Passing None switches to a full-sample static "
            "z-score computed once over the whole series — this leaks future spread statistics "
            "into every historical signal and should only be used for exploratory/offline "
            "analysis, never to evaluate strategy performance."
        ),
    )
    initial_capital: float = Field(
        10_000.0, le=1e15, gt=0, description="Starting cash for the whole account."
    )
    commission_pct: float = Field(
        0.001, le=1, ge=0, description="Commission per trade notional (fraction)."
    )
    slippage_pct: float = Field(
        0.0005, le=1, ge=0, description="Slippage per trade notional (fraction)."
    )
    gross_leverage: float = Field(
        1.0,
        gt=0,
        description="sum(|weight|) while in a position, split between the two legs to match hedge_ratio.",
    )
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field(
        "next_open",
        description=(
            "'next_open' (default), 'close', or 'hl2_exploratory'. Defaults to 'next_open' "
            "(not 'close') because the z-score signal deciding a transition is itself "
            "computed from that same bar's Close — executing at that same Close would "
            "be look-ahead. Pass 'close' only for explicit same-bar/exploratory analysis."
        ),
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
        description=(
            "Strategy type: 'sma_crossover', 'rsi_mean_reversion', "
            "'macd_crossover', 'bollinger_reversion', 'donchian_breakout', "
            "'momentum_timeseries', 'vwap_reversion', or 'adx_trend'."
        ),
    )
    parameters: Dict[str, Any] = Field(
        {},
        description="Strategy parameters — same shape as run_sma_backtest / run_rsi_backtest / etc.",
    )
    initial_capital: float = Field(
        10_000.0, gt=0, le=1e15, description="Starting capital."
    )
    commission_pct: float = Field(
        0.001, ge=0, le=1, description="Commission per trade (fraction, default 0.1%)."
    )
    slippage_pct: float = Field(
        0.0005, ge=0, le=1, description="Slippage per trade (fraction, default 0.05%)."
    )
    top_n_drawdowns: int = Field(
        5, description="Number of worst drawdown episodes to return."
    )
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field(
        "close",
        description="'close' (default) or 'next_open' — see BacktestInput.fill_price.",
    )


class DrawdownEpisode(BaseModel):
    start: str
    trough: str
    end: Optional[str] = None  # None if still underwater at the end of the series
    depth: float  # negative fraction, e.g. -0.15 = -15%
    duration_bars: int  # peak -> recovery (or peak -> last bar if unrecovered)
    recovery_bars: Optional[int] = None  # trough -> recovery; None if unrecovered


class TradeDiagnostics(BaseModel):
    expectancy_pct: float
    avg_winner_pct: float
    avg_loser_pct: float
    payoff_ratio: float  # can be inf if there are no losing trades
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_mae_pct: float  # average maximum adverse excursion across trades
    avg_mfe_pct: float  # average maximum favorable excursion across trades


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


# ──────────────────────────────────────────────
# Robustness Diagnostics (parameter sensitivity, Deflated Sharpe Ratio,
# block-bootstrap CI — same-sample confidence checks, NOT a substitute for
# run_walk_forward_backtest's out-of-sample validation)
# ──────────────────────────────────────────────


class RobustnessDiagnosticsInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    strategy: Literal[
        "sma_crossover",
        "rsi_mean_reversion",
        "macd_crossover",
        "bollinger_reversion",
        "donchian_breakout",
        "momentum_timeseries",
        "vwap_reversion",
        "adx_trend",
    ] = Field(
        ...,
        description=(
            "Strategy name: 'sma_crossover', 'rsi_mean_reversion', "
            "'macd_crossover', 'bollinger_reversion', 'donchian_breakout', "
            "'momentum_timeseries', 'vwap_reversion', or 'adx_trend'."
        ),
    )
    param_grid: Dict[str, List[Any]] = Field(
        ...,
        description="Parameter grid to search — same shape as run_backtest_optimization.",
    )
    initial_capital: float = Field(
        10_000.0, gt=0, le=1e15, description="Starting capital."
    )
    commission_pct: float = Field(
        0.001, ge=0, le=1, description="Commission per trade (fraction)."
    )
    slippage_pct: float = Field(
        0.0005, ge=0, le=1, description="Slippage per trade (fraction)."
    )
    sort_by: Literal[
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "total_return",
        "profit_factor",
        "win_rate",
        "max_drawdown",
        "annualized_volatility",
        "avg_trade_return_pct",
        "num_trades",
    ] = Field(
        "sharpe_ratio", description="Metric used to pick the best trial from the grid."
    )
    n_bootstrap_iterations: int = Field(
        1000, description="Block-bootstrap resamples for the best trial's Sharpe CI."
    )
    bootstrap_block_size: int = Field(
        20, description="Block length (bars) for the bootstrap."
    )
    bootstrap_confidence: float = Field(
        0.95, description="Two-sided confidence level for the bootstrap CI."
    )
    random_seed: Optional[int] = Field(
        None,
        description="Seed for the block-bootstrap RNG — set for reproducible results (recorded in the audit trail).",
    )
    skew: float = Field(
        0.0,
        description="Return-distribution skew for the Deflated Sharpe Ratio's standard-error formula (0.0 = normal).",
    )
    kurtosis: float = Field(
        3.0,
        description="Return-distribution kurtosis for the Deflated Sharpe Ratio's standard-error formula (3.0 = normal).",
    )

    @field_validator("param_grid")
    @classmethod
    def _check_param_grid(cls, v: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        return _validate_param_grid(v)


class RobustnessDiagnosticsResult(BaseModel):
    symbol: str
    strategy: str
    best_params: Dict[str, Any]
    parameter_sensitivity: Dict[str, Any]
    expected_max_sharpe: float
    deflated_sharpe_ratio: float
    bootstrap_point_estimate: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    bootstrap_confidence: float
    warnings: List[str] = []


# ──────────────────────────────────────────────
# Monte Carlo Forward Simulation (block-bootstrap projection of possible
# future equity paths from a historical return distribution — forward-
# looking, NOT a substitute for run_walk_forward_backtest's out-of-sample
# validation, which tests a strategy's actual historical decisions)
# ──────────────────────────────────────────────


class MonteCarloSimulationInput(BaseModel):
    tickers: List[str] = Field(..., description="Portfolio tickers.")
    weights: Optional[List[float]] = Field(
        None,
        description="Portfolio weights, same order as tickers, must sum to 1.0. None (default) uses equal weighting.",
    )
    start_date: str = Field(
        ...,
        description="Start date YYYY-MM-DD for the historical return window used to estimate the distribution.",
    )
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    horizon_days: int = Field(
        252,
        gt=0,
        le=2520,
        description="Number of forward bars to simulate (default 252 = ~1 year).",
    )
    n_simulations: int = Field(
        1000, ge=100, le=20000, description="Number of independent simulated paths."
    )
    block_size: int = Field(
        20,
        gt=0,
        description="Block length (bars) for the moving-block bootstrap resample.",
    )
    initial_capital: float = Field(
        10_000.0, le=1e15, gt=0, description="Starting capital."
    )
    random_seed: Optional[int] = Field(
        None,
        description="Seed for the resampling RNG — set for reproducible results (recorded in the audit trail).",
    )

    @model_validator(mode="after")
    def _check_weights(self) -> "MonteCarloSimulationInput":
        if self.weights is not None:
            if len(self.weights) != len(self.tickers):
                raise ValueError(
                    f"len(weights)={len(self.weights)} must equal len(tickers)={len(self.tickers)}"
                )
            total = sum(self.weights)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"weights must sum to 1.0, got {total:.8f}")
        return self


class MonteCarloSimulationResult(BaseModel):
    tickers: List[str]
    horizon_days: int
    n_simulations: int
    random_seed: int = Field(
        ...,
        description=(
            "The seed this run actually used. When the request omitted one, a "
            "seed is drawn HERE and passed down, rather than letting the native "
            "kernel derive one from the clock — otherwise the audit record "
            "stored random_seed=None while the numbers came from a value "
            "nobody kept, and the run could never be reproduced. Pass this "
            "back as random_seed to repeat the simulation exactly."
        ),
    )
    terminal_median: float
    terminal_p5: float
    terminal_p95: float
    prob_loss: float
    terminal_var_95: float
    terminal_cvar_95: float
    equity_band_p5: List[float]
    equity_band_p50: List[float]
    equity_band_p95: List[float]


# ──────────────────────────────────────────────
# Capacity Report (liquidity/ADV-based capacity — how much account size a
# target-weight portfolio can support before positions become too large
# relative to each ticker's own trading volume)
# ──────────────────────────────────────────────


class CapacityReportInput(BaseModel):
    tickers: List[str] = Field(
        ..., description="Ticker universe. Must match target_weights' keys."
    )
    start_date: str = Field(
        ...,
        description="Start date YYYY-MM-DD — used to compute each ticker's average dollar volume.",
    )
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    target_weights: Dict[str, float] = Field(
        ...,
        description="{ticker: target fraction of account equity} — a single snapshot, not a rebalance panel.",
    )
    max_participation: float = Field(
        0.1,
        description="Max fraction of a ticker's own average dollar volume a position may represent.",
    )
    adv_lookback: int = Field(
        20,
        description="Rolling window (bars, trailing from the end of the requested range) for average dollar/share volume.",
    )
    include_sector_exposure: bool = Field(
        True,
        description="If True, fetch each ticker's sector via the data provider's get_ticker_info and report exposure by sector (best-effort — 'Unknown' when unavailable).",
    )

    @model_validator(mode="after")
    def _check_weights_match_tickers(self) -> "CapacityReportInput":
        missing = [t for t in self.tickers if t not in self.target_weights]
        if missing:
            raise ValueError(f"target_weights is missing entries for: {missing}")
        return self


class CapacityReportResult(BaseModel):
    tickers: List[str]
    per_ticker_max_account_size: Dict[
        str, Optional[float]
    ]  # None = unbounded (zero target weight)
    binding_ticker: Optional[str] = None
    max_account_size: Optional[float] = None  # None = unbounded (every weight is zero)
    days_to_liquidate_at_capacity: Dict[str, float]
    sector_exposure: Optional[Dict[str, float]] = None
    warnings: List[str] = []


# ──────────────────────────────────────────────
# Liquidity / Microstructure Proxies (Amihud illiquidity, Corwin-Schultz
# spread estimator — academic proxies for bid/ask spread and market depth
# from OHLCV alone, since no real bid/ask data exists in this library)
# ──────────────────────────────────────────────


class LiquidityAnalysisInput(BaseModel):
    tickers: List[str] = Field(..., description="Tickers to analyze.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    window: int = Field(
        20, gt=0, description="Rolling window (bars) for both liquidity proxies."
    )


class LiquidityAnalysisResult(BaseModel):
    tickers: List[str]
    per_ticker: Dict[str, Dict[str, float]]
    least_liquid_ticker: str
    most_liquid_ticker: str


# ──────────────────────────────────────────────
# Data Quality Report (dataset provenance + missing-bar/stale-price/
# price-jump detection — data/metadata.py + data/quality.py)
# ──────────────────────────────────────────────


class DataQualityReportInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    stale_run_length: int = Field(
        3,
        description="Minimum consecutive-identical-Close run length to flag as stale.",
    )
    jump_threshold: float = Field(
        0.15,
        description="Fractional single-bar Close-to-Close move to flag as a jump (default 0.15 = 15%).",
    )


class MissingBar(BaseModel):
    date: str
    weekday: str


class StalePriceRun(BaseModel):
    start: str
    end: str
    price: float
    run_length: int


class PriceJump(BaseModel):
    date: str
    pct_change: float


class DataQualityReportResult(BaseModel):
    symbol: str
    metadata: Dict[str, Any]
    missing_bars: List[MissingBar] = Field(
        ...,
        description=(
            "Weekday gaps in the price history. WARNING: detected with a weekday-only "
            "heuristic, not a real market-holiday calendar — every U.S. market holiday "
            "(Thanksgiving, Christmas, etc.) in the requested range will appear here as a "
            "false positive. Treat entries as leads to investigate, not confirmed data gaps."
        ),
    )
    stale_price_runs: List[StalePriceRun]
    price_jumps: List[PriceJump]


# ──────────────────────────────────────────────
# Compact Backtest Result (BacktestResultV2 — summary/risk/exposure/cost
# sub-reports plus artifact URIs instead of embedding the full equity
# curve/trade log inline, unlike the plain BacktestResult)
# ──────────────────────────────────────────────


class PerformanceSummary(BaseModel):
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float


class RiskSummary(BaseModel):
    max_drawdown: float
    var_95: float
    cvar_95: float


class ExposureSummary(BaseModel):
    time_in_market: float
    avg_gross_exposure: float
    avg_net_exposure: float
    pct_long: float
    pct_short: float
    avg_holding_period_bars: Optional[float] = None


class CostSummary(BaseModel):
    total_commission_pct: (
        float  # sum of commission drag across all bars, as a fraction of capital
    )
    total_slippage_pct: float  # sum of slippage drag, same units
    total_cost_pct: float
    num_trades: int


class BacktestCompactInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    strategy_type: str = Field(
        ...,
        description=(
            "Strategy type: 'sma_crossover', 'rsi_mean_reversion', "
            "'macd_crossover', 'bollinger_reversion', 'donchian_breakout', "
            "'momentum_timeseries', 'vwap_reversion', or 'adx_trend'."
        ),
    )
    parameters: Dict[str, Any] = Field(
        {}, description="Strategy parameters — same shape as BacktestInput."
    )
    initial_capital: float = Field(
        10_000.0, gt=0, le=1e15, description="Starting capital."
    )
    commission_pct: float = Field(
        0.001, ge=0, le=1, description="Commission per trade (fraction)."
    )
    slippage_pct: float = Field(
        0.0005, ge=0, le=1, description="Slippage per trade (fraction)."
    )
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field(
        "close",
        description="'close' (default), 'next_open', or 'hl2_exploratory' — see BacktestInput.fill_price.",
    )
    run_id: Optional[str] = Field(
        None,
        description="Identifier for the saved artifacts. Auto-generated (a UUID) when not supplied.",
    )


class BacktestResultV2(BaseModel):
    run_id: str
    strategy_name: str
    summary: PerformanceSummary
    risk: RiskSummary
    exposure: ExposureSummary
    costs: CostSummary
    equity_curve_uri: str
    trades_uri: Optional[str] = None  # None when the strategy never traded
    warnings: List[str] = []
    validation_status: str  # "ok" | "warning"


# ──────────────────────────────────────────────
# Options Pricing, Greeks & Implied Volatility (analysis/options.py —
# Black-Scholes-Merton, European options only)
# ──────────────────────────────────────────────


class OptionPricingInput(BaseModel):
    spot: float = Field(..., gt=0, description="Current underlying price.")
    strike: float = Field(..., gt=0, description="Option strike price.")
    time_to_expiry: float = Field(
        ..., gt=0, description="Time to expiry in years (e.g. 0.25 = 3 months)."
    )
    risk_free_rate: float = Field(
        ...,
        description="Annualized continuously-compounded risk-free rate (e.g. 0.05 = 5%).",
    )
    volatility: float = Field(
        ..., gt=0, description="Annualized volatility (e.g. 0.20 = 20%)."
    )
    option_type: Literal["call", "put"] = Field("call", description="Option type.")
    dividend_yield: float = Field(
        0.0,
        ge=0,
        description="Continuous dividend yield (Merton extension); 0.0 = plain Black-Scholes.",
    )


class OptionGreeks(BaseModel):
    delta: float
    gamma: float
    vega: float = Field(
        ...,
        description="Price change per 1.0 (100 percentage points) of volatility — divide by 100 for the conventional 'per vol point' quote.",
    )
    theta: float = Field(
        ...,
        description="Price change per YEAR (raw) — divide by 365 for the conventional 'per calendar day' quote.",
    )
    rho: float


class OptionPricingResult(BaseModel):
    option_type: str
    price: float
    greeks: OptionGreeks
    d1: float
    d2: float


class ImpliedVolatilityInput(BaseModel):
    option_price: float = Field(
        ..., gt=0, description="Observed market price of the option."
    )
    spot: float = Field(..., gt=0, description="Current underlying price.")
    strike: float = Field(..., gt=0, description="Option strike price.")
    time_to_expiry: float = Field(..., gt=0, description="Time to expiry in years.")
    risk_free_rate: float = Field(
        ..., description="Annualized continuously-compounded risk-free rate."
    )
    option_type: Literal["call", "put"] = Field("call", description="Option type.")
    dividend_yield: float = Field(0.0, ge=0, description="Continuous dividend yield.")


class ImpliedVolatilityResult(BaseModel):
    implied_volatility: float
    converged: bool
    iterations: int
    method: str  # "newton" | "bisection"


# ──────────────────────────────────────────────
# Discovery — what this library can do, asked rather than assumed
#
# The modeling runtime has had `list_features` and
# `list_modeling_capabilities` since it shipped; this 46-tool surface had
# nothing equivalent. A caller learned the strategy vocabulary from prose
# inside a Field description, the stress-scenario names from a sentence in
# a tool description, and whether a tick feed existed by calling something
# that raised NotImplementedError. Each of those is a contract the library
# already holds in a data structure — STRATEGY_PARAM_SCHEMA, _SCENARIOS,
# the provider classes — and prose is a lossy copy of a data structure that
# drifts from it silently.
# ──────────────────────────────────────────────


class StrategyParameter(BaseModel):
    """One strategy parameter's declared contract, from
    backtest/strategy_params.py's STRATEGY_PARAM_SCHEMA."""

    name: str
    kind: Literal["window", "number"] = Field(
        ...,
        description=(
            "'window' is a positive whole number of BARS (rejected below 1: "
            "pandas reads a negative period as a forward window, which is "
            "look-ahead by construction). 'number' is any finite float "
            "within the bounds below."
        ),
    )
    default: Any = Field(..., description="Value used when the caller omits this.")
    minimum: Optional[float] = Field(None, description="Inclusive lower bound, if any.")
    maximum: Optional[float] = Field(None, description="Inclusive upper bound, if any.")


class StrategyRelation(BaseModel):
    """A constraint BETWEEN two parameters. Each value can be individually
    valid while the pair is nonsense, so these are checked separately."""

    left: str
    right: str
    requirement: str = Field(..., description="Always of the form 'left < right'.")
    why: str = Field(
        ..., description="What breaks when the relation is violated, in plain terms."
    )


class StrategyDescriptor(BaseModel):
    name: str
    parameters: List[StrategyParameter]
    relations: List[StrategyRelation] = Field(
        default_factory=list,
        description="Empty for strategies whose parameters are independent.",
    )


class ListStrategiesInput(BaseModel):
    strategy_type: Optional[str] = Field(
        None,
        description=(
            "Return only this strategy's contract. None (the default) "
            "returns all eight."
        ),
    )


class ListStrategiesResult(BaseModel):
    strategies: List[StrategyDescriptor]
    max_window_bars: int = Field(
        ...,
        description=(
            "Upper bound on any 'window' parameter. A longer window is more "
            "likely a units mix-up (days vs minutes) than an intent."
        ),
    )
    synthetic_labels: List[str] = Field(
        ...,
        description=(
            "Accepted strategy_type values that are NOT in the registry and "
            "take no parameters: 'buy_and_hold' constructs an always-long "
            "series directly and 'custom_signal' carries a caller-supplied "
            "one."
        ),
    )


class ListStressScenariosInput(BaseModel):
    """No arguments — the scenario table is a fixed, offline constant."""


class StressScenario(BaseModel):
    name: str
    start: str
    end: str
    calendar_days: int = Field(
        ...,
        description=(
            "Length of the window in calendar days, not trading days — this "
            "is computed from the dates alone and involves no market data."
        ),
    )


class ListStressScenariosResult(BaseModel):
    scenarios: List[StressScenario]


class DataCapabilitiesInput(BaseModel):
    source: str = Field(
        "yfinance",
        description=(
            "Provider to describe: 'yfinance', 'polygon', or 'bloomberg'. "
            "Describing a provider does NOT fetch any market data."
        ),
    )


class DataCapabilitiesResult(BaseModel):
    provider: str
    available: bool = Field(
        ...,
        description=(
            "False when the provider could not even be constructed — a "
            "missing API key, an uninstalled SDK. Everything below is then "
            "the class's declared capability, not a working connection."
        ),
    )
    unavailable_reason: Optional[str] = Field(
        None, description="Why construction failed, verbatim, when available is False."
    )
    ohlcv: bool
    ohlcv_async: bool
    ticker_info: bool
    financial_ratios: bool
    trades: bool = Field(
        ...,
        description=(
            "Tick-level trades. False means the microstructure tools cannot "
            "run on this provider AT ALL — bar data is not a substitute, and "
            "nothing here synthesizes one."
        ),
    )
    quotes: bool = Field(
        ..., description="Top-of-book bid/offer. No shipped provider offers depth."
    )
    supported_intervals: Optional[List[str]] = Field(
        None, description="Bar intervals this provider accepts, if it declares a set."
    )
    guarantees: Dict[str, bool] = Field(
        ...,
        description=(
            "adjusted / survivorship_free / point_in_time, as the provider "
            "itself reports them — what it actually promises, not what would "
            "be ideal."
        ),
    )
    cache_dir: str = Field(..., description="Where the persistent OHLCV cache lives.")
    notes: List[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Transaction costs — priced on their own, and swept
#
# backtest/costs.py is ten pure functions, and until now every one of them
# was reachable only by running a whole portfolio simulation that happened
# to compose the subset you wanted. Two of them (maker_taker_cost,
# pct_of_range_spread) were reachable from no tool at all.
#
# The question a cost model actually gets asked is "does this strategy
# survive it", and that question was previously answered by running the
# same backtest N times with different commission_pct and comparing by
# hand. compare_cost_models does the sweep in one call on one fetch, and
# solves for the rate at which the edge disappears -- which is the number
# the N-call version was groping toward.
# ──────────────────────────────────────────────


class TradeCostLeg(BaseModel):
    """One priced component of a trade's cost."""

    component: Literal["commission", "spread", "impact", "borrow", "margin_interest"]
    model: str = Field(
        ..., description="Which backtest/costs.py function priced this leg."
    )
    cost: float = Field(..., description="Currency units.")
    bps_of_notional: float


class EstimateTradeCostInput(BaseModel):
    notional: float = Field(
        ...,
        gt=0,
        description="Trade size in currency units. Cost is charged on |notional|.",
    )
    side: Literal["buy", "sell"] = Field(
        "buy",
        description=(
            "Only matters for commission_model='directional' (separate "
            "buy/sell rates) — every other model is side-agnostic."
        ),
    )
    commission_model: Literal[
        "pct", "per_share", "directional", "maker_taker", "none"
    ] = Field(
        "pct",
        description=(
            "'pct' rate x notional | 'per_share' rate x shares floored at a "
            "minimum | 'directional' separate buy and sell rates | "
            "'maker_taker' where the maker rate MAY be a rebate (negative) | "
            "'none' to price the other components alone."
        ),
    )
    commission_pct: float = Field(
        0.001, ge=0, le=1, description="commission_model='pct': fraction of notional."
    )
    shares: Optional[float] = Field(
        None,
        gt=0,
        description="commission_model='per_share': share count. Required for that model.",
    )
    per_share_rate: float = Field(
        0.005, ge=0, description="commission_model='per_share': currency per share."
    )
    min_commission: float = Field(
        1.0, ge=0, description="commission_model='per_share': floor per trade."
    )
    buy_rate: float = Field(
        0.001,
        ge=0,
        description="commission_model='directional': fraction charged on buys.",
    )
    sell_rate: float = Field(
        0.001,
        ge=0,
        description=(
            "commission_model='directional': fraction charged on sells. Often "
            "the higher of the two — regulatory fees are typically sell-side."
        ),
    )
    taker_rate: float = Field(
        0.0005,
        ge=0,
        description="commission_model='maker_taker': fraction taken when crossing.",
    )
    maker_rate: float = Field(
        -0.0001,
        description=(
            "commission_model='maker_taker': fraction when providing "
            "liquidity. MAY be negative — that is a rebate, and it is the "
            "one cost input here that is allowed below zero."
        ),
    )
    is_maker: bool = Field(
        False,
        description="commission_model='maker_taker': did this order provide liquidity?",
    )
    spread_model: Literal["fixed_bps", "pct_of_range", "none"] = Field(
        "fixed_bps",
        description=(
            "'fixed_bps' a flat basis-point haircut | 'pct_of_range' a "
            "fraction of the bar's own High-Low range, which widens the "
            "estimate on volatile bars | 'none'."
        ),
    )
    spread_bps: float = Field(
        1.0, ge=0, description="spread_model='fixed_bps': basis points of notional."
    )
    bar_high: Optional[float] = Field(
        None, gt=0, description="spread_model='pct_of_range': the bar's High."
    )
    bar_low: Optional[float] = Field(
        None, gt=0, description="spread_model='pct_of_range': the bar's Low."
    )
    bar_close: Optional[float] = Field(
        None, gt=0, description="spread_model='pct_of_range': the bar's Close."
    )
    range_pct: float = Field(
        0.1,
        ge=0,
        description="spread_model='pct_of_range': fraction of the High-Low range to charge.",
    )
    avg_dollar_volume: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Supply this together with `volatility` to add a square-root "
            "market-impact leg. Omit either one and impact is not priced — "
            "impact is never guessed from notional alone."
        ),
    )
    volatility: Optional[float] = Field(
        None, ge=0, description="Per-period return volatility for the impact model."
    )
    impact_coefficient: float = Field(
        1.0, ge=0, description="Impact model coefficient."
    )
    short_borrow_bps: float = Field(
        0.0,
        ge=0,
        description="Annualized basis points on a short's notional, accrued over holding_days.",
    )
    holding_days: float = Field(
        1.0,
        ge=0,
        description="Days the position is held, for borrow and margin accrual.",
    )
    margin_cash: float = Field(
        0.0,
        description=(
            "Account cash. Only a NEGATIVE value accrues margin interest — "
            "there is nothing borrowed to charge on a positive balance."
        ),
    )
    margin_annual_rate: float = Field(
        0.0, ge=0, description="Annualized rate on negative cash."
    )

    @model_validator(mode="after")
    def _check_model_inputs(self) -> "EstimateTradeCostInput":
        if self.commission_model == "per_share" and self.shares is None:
            raise ValueError(
                "commission_model='per_share' needs `shares` — a per-share "
                "rate cannot be derived from notional without a price."
            )
        if self.spread_model == "pct_of_range":
            missing = [
                name
                for name, value in (
                    ("bar_high", self.bar_high),
                    ("bar_low", self.bar_low),
                    ("bar_close", self.bar_close),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    f"spread_model='pct_of_range' needs {missing} — the "
                    "estimate is a fraction of that bar's own range."
                )
            if self.bar_high is not None and self.bar_low is not None:
                if self.bar_high < self.bar_low:
                    raise ValueError(
                        f"bar_high ({self.bar_high}) is below bar_low "
                        f"({self.bar_low})"
                    )
        impact_inputs = (self.avg_dollar_volume, self.volatility)
        if any(v is not None for v in impact_inputs) and not all(
            v is not None for v in impact_inputs
        ):
            raise ValueError(
                "the impact model needs BOTH avg_dollar_volume and "
                "volatility. Pricing impact from notional alone would be a "
                "number with no model behind it."
            )
        return self


class EstimateTradeCostResult(BaseModel):
    notional: float
    side: str
    legs: List[TradeCostLeg]
    total_cost: float
    total_bps: float = Field(
        ..., description="Total one-way cost in basis points of notional."
    )
    breakeven_move_bps: float = Field(
        ...,
        description=(
            "How far the price must move in your favour to cover a ROUND "
            "TRIP at this cost — two times total_bps. The one-way figure "
            "understates what an entry actually has to earn."
        ),
    )
    notes: List[str] = Field(default_factory=list)


class CostScenario(BaseModel):
    """One point in the cost sweep."""

    label: str = Field(..., min_length=1, max_length=64)
    commission_pct: float = Field(..., ge=0, le=1)
    slippage_pct: float = Field(0.0, ge=0, le=1)


class CostScenarioResult(BaseModel):
    label: str
    commission_pct: float
    slippage_pct: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    n_trades: int
    cost_drag_vs_gross: float = Field(
        ...,
        description=(
            "total_return under this scenario minus the zero-cost "
            "total_return — always <= 0, and the amount costs took."
        ),
    )


class CompareCostModelsInput(BaseModel):
    symbol: str = Field(..., description="Ticker symbol.")
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    strategy_type: str = Field(
        ...,
        description=(
            "One of the eight registry strategies — call list_strategies for "
            "the names and their parameters."
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Strategy parameters, validated as usual."
    )
    scenarios: List[CostScenario] = Field(
        ...,
        min_length=1,
        max_length=12,
        description=(
            "Cost assumptions to price the SAME signal series under. The "
            "signal is computed once, so these differ only in what the "
            "trading cost."
        ),
    )
    initial_capital: float = Field(10_000.0, gt=0, le=1e15)
    fill_price: Literal["close", "next_open", "hl2_exploratory"] = Field("close")
    solve_breakeven: bool = Field(
        True,
        description=(
            "Solve for the commission rate at which total return reaches "
            "zero. Costs are monotone in the rate for a fixed signal series, "
            "so this is a bisection, not a search."
        ),
    )

    @field_validator("scenarios")
    @classmethod
    def _unique_labels(cls, scenarios: List[CostScenario]) -> List[CostScenario]:
        labels = [s.label for s in scenarios]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(
                f"scenario labels must be unique; got duplicates {duplicates}. "
                "Two rows with the same label are indistinguishable in the result."
            )
        return scenarios


class CompareCostModelsResult(BaseModel):
    symbol: str
    strategy_type: str
    n_bars: int
    gross_total_return: float = Field(
        ...,
        description=(
            "Total return with ZERO costs — the ceiling every scenario is "
            "measured against. Always computed, never one of the submitted "
            "scenarios."
        ),
    )
    gross_sharpe_ratio: float
    scenarios: List[CostScenarioResult]
    breakeven_commission_pct: Optional[float] = Field(
        None,
        description=(
            "Commission rate at which total return crosses zero, holding "
            "slippage at the first scenario's value. None when the strategy "
            "loses money even at zero cost (nothing to break even from) or "
            "still profits at a 100% commission (no crossing exists)."
        ),
    )
    survives_all_scenarios: bool = Field(
        ..., description="True when every submitted scenario keeps total_return > 0."
    )
    notes: List[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Panel indicators, and follow-ups on a run that already happened
#
# Two costs this section removes. The first is per-ticker round trips:
# get_technical_analysis answers for one symbol, so a 50-name screen was 50
# calls, while indicators/panel.py already computes the whole universe in
# one native call and was reachable from no tool.
#
# The second is recomputation. run_backtest_compact persists its equity
# curve and trade log and hands back URIs, but every follow-up tool took a
# symbol and a strategy and RE-RAN the backtest to answer a question about
# a run that was already on disk. That is slower, and it is not the same
# run -- a provider revision between the two calls silently diagnoses
# something other than what was reported.
# ──────────────────────────────────────────────


class TechnicalPanelInput(BaseModel):
    tickers: List[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description=(
            "Universe to compute across. One fetch and one native call for "
            "the whole set, so this is much cheaper than a "
            "get_technical_analysis call per ticker."
        ),
    )
    start_date: str = Field(..., description="Start date YYYY-MM-DD.")
    end_date: str = Field(..., description="End date YYYY-MM-DD.")
    indicators: List[
        Literal["rsi", "adx", "atr", "bollinger_bands", "stochastic_oscillator"]
    ] = Field(
        ["rsi"],
        min_length=1,
        description=(
            "Any of rsi, adx, atr, bollinger_bands, stochastic_oscillator — "
            "the set indicators/panel.py computes natively."
        ),
    )
    rsi_period: int = Field(14, gt=0, le=1000)
    adx_period: int = Field(14, gt=0, le=1000)
    atr_period: int = Field(14, gt=0, le=1000)
    bollinger_period: int = Field(20, gt=0, le=1000)
    bollinger_num_std: float = Field(2.0, gt=0, le=100)
    stoch_k_period: int = Field(14, gt=0, le=1000)
    stoch_d_period: int = Field(3, gt=0, le=1000)
    persist_run_id: Optional[str] = Field(
        None,
        description=(
            "Persist the FULL panel (every bar, not just the latest) as "
            "Parquet artifacts under this run id and return their URIs. "
            "Letters, digits, '_' and '-' only. Omit to get the latest-bar "
            "snapshot alone — the full panel is far too large to return "
            "inline for any real universe."
        ),
    )

    @field_validator("tickers")
    @classmethod
    def _no_duplicates(cls, tickers: List[str]) -> List[str]:
        duplicates = sorted({t for t in tickers if tickers.count(t) > 1})
        if duplicates:
            raise ValueError(
                f"tickers contains duplicates {duplicates}; the panel is keyed "
                "by ticker, so a repeat collapses and the result would report "
                "fewer columns than were asked for."
            )
        return tickers


class TechnicalPanelResult(BaseModel):
    tickers: List[str]
    indicators: List[str]
    as_of: str = Field(..., description="Date of the latest bar in the panel.")
    n_bars: int
    latest: Dict[str, Dict[str, float]] = Field(
        ...,
        description=(
            "ticker -> {field: value} at the latest bar. Field names match "
            "the per-ticker tools exactly (RSI, ADX, DI_Plus, BB_Upper, "
            "Stoch_K, ...), so nothing new has to be learned to read this."
        ),
    )
    incomplete_tickers: List[str] = Field(
        default_factory=list,
        description=(
            "Tickers whose latest value is NaN for at least one requested "
            "indicator — usually too few bars for the lookback. Reported "
            "rather than dropped: a silently missing ticker looks like a "
            "screen that legitimately excluded it."
        ),
    )
    calendar_start: str = Field(
        ...,
        description=(
            "First bar of the SHARED calendar the panel was computed on — "
            "the intersection of every ticker's history, not the requested "
            "start date. They differ whenever one ticker is younger than "
            "the window."
        ),
    )
    calendar_limited_by: List[str] = Field(
        default_factory=list,
        description=(
            "Tickers whose own first bar is later than the earliest "
            "available in the universe, and which therefore truncate the "
            "shared calendar for EVERY ticker. A recent listing here can "
            "collapse a multi-year request to a handful of bars and turn "
            "every other ticker's indicator to NaN — drop it, or shorten "
            "the window deliberately."
        ),
    )
    notes: List[str] = Field(default_factory=list)
    artifact_uris: Dict[str, str] = Field(
        default_factory=dict,
        description="indicator -> Parquet URI, when persist_run_id was given.",
    )
    execution_path: str = Field(
        ..., description="'C++' or 'per-ticker' — which panel path actually ran."
    )


class DescribeArtifactInput(BaseModel):
    uri: str = Field(
        ...,
        description=(
            "An artifact URI returned by a tool (equity_curve_uri, "
            "trades_uri, ...). Must resolve inside SQT_RUNS_DIR."
        ),
    )
    preview_rows: int = Field(
        5,
        ge=0,
        le=50,
        description=(
            "Rows to include from each end. The middle is never returned — "
            "an equity curve has thousands of bars and this tool exists to "
            "describe one, not to move it into the conversation."
        ),
    )


class DescribeArtifactResult(BaseModel):
    uri: str
    rows: int
    columns: List[str]
    index_name: Optional[str] = None
    index_start: Optional[str] = None
    index_end: Optional[str] = None
    content_hash: str = Field(
        ...,
        description=(
            "SHA-256 of the file's bytes. Two tools reading the same URI "
            "can confirm they saw the same artifact, and a re-run that "
            "changed it is visible without diffing the contents."
        ),
    )
    head: List[Dict[str, Any]] = Field(default_factory=list)
    tail: List[Dict[str, Any]] = Field(default_factory=list)
    column_summary: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Per numeric column: min, max, mean, and the NaN count.",
    )


class DrawdownTableInput(BaseModel):
    equity_curve_uri: str = Field(
        ...,
        description=(
            "URI of a persisted equity curve — run_backtest_compact's "
            "equity_curve_uri. Reading the run that happened is both cheaper "
            "and more honest than re-running the backtest to describe it."
        ),
    )
    min_depth: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Drop episodes shallower than this fraction (0.05 = 5%). 0 keeps "
            "every episode, including the one-bar noise that dominates a "
            "long curve by count while contributing nothing to risk."
        ),
    )
    max_episodes: int = Field(
        50,
        gt=0,
        le=500,
        description="Cap on returned episodes, deepest first.",
    )


class DrawdownTableResult(BaseModel):
    equity_curve_uri: str
    n_bars: int
    n_episodes_total: int = Field(
        ..., description="Episodes found before min_depth and max_episodes filtering."
    )
    n_episodes_returned: int
    max_drawdown: float
    episodes: List[DrawdownEpisode]
    currently_underwater: bool = Field(
        ...,
        description=(
            "True when the curve ends inside a drawdown that never "
            "recovered. That last episode's recovery_bars is null, and its "
            "duration is a floor rather than a measurement."
        ),
    )
    time_underwater_pct: float = Field(
        ...,
        description="Fraction of bars spent below a prior peak, across all episodes.",
    )


# ──────────────────────────────────────────────
# Provenance — reading and verifying the decision log
#
# Every dispatch() call already writes a tamper-evident record: the tool,
# its inputs, content hashes of the market data it read, which execution
# path ran, and the output hash, chained so that editing a past line breaks
# every line after it. Thirteen CLI commands operate on that log and no
# tool did, which meant the one participant who could not check its own
# work was the agent whose work it was.
#
# READ AND VERIFY ONLY. Retention is deliberately absent: `gc`, `seal`,
# `hold`, `release-hold` and `keygen` stay CLI-only, because handing the
# agent whose decisions are logged the power to seal, hold or delete them
# defeats the reason the log exists. Nothing here can alter a record.
# export_audit_bundle writes a new zip and touches no existing file.
# ──────────────────────────────────────────────


class ExplainDecisionInput(BaseModel):
    request_id: str = Field(
        ...,
        min_length=1,
        description=(
            "The request id of a recorded tool call. Every dispatch() writes "
            "one; it appears in the audit log and in log records correlated "
            "by RequestIdFilter."
        ),
    )


class DataSourceRef(BaseModel):
    """One market-data input a recorded call actually read."""

    symbol: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    rows: Optional[int] = None
    content_hash: Optional[str] = Field(
        None,
        description=(
            "Hash of the data as it was AT THE TIME. A later fetch that "
            "disagrees is what distinguishes a revised dataset from a code "
            "change."
        ),
    )


class ExplainDecisionResult(BaseModel):
    request_id: str
    timestamp_utc: str
    tool_name: str
    status: str
    input: Dict[str, Any]
    data_sources: List[DataSourceRef] = Field(default_factory=list)
    duration_ms: float
    execution_path: str = Field(
        ...,
        description=(
            "'C++' or 'Python/Numba' — which implementation actually ran. "
            "The fallback chain is transparent at call time and is exactly "
            "the kind of thing that is impossible to reconstruct afterwards "
            "without a record."
        ),
    )
    output_hash: Optional[str] = None
    git_commit_sha: Optional[str] = None
    package_version: Optional[str] = None
    random_seed: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    record_hash: Optional[str] = None


class ReplayDecisionInput(BaseModel):
    request_id: str = Field(
        ..., min_length=1, description="The recorded call to re-run and compare."
    )


class DataSourceMatch(BaseModel):
    symbol: Optional[str] = None
    matches: Optional[bool] = Field(
        None,
        description=(
            "True when re-fetching that input reproduces the recorded hash. "
            "False means the DATA changed underneath the decision. None "
            "means it could not be checked."
        ),
    )
    detail: Optional[str] = None


class ReplayDecisionResult(BaseModel):
    request_id: str
    tool_name: str
    output_match: Optional[bool] = Field(
        None,
        description=(
            "True when re-running reproduces the recorded output hash. "
            "None when the record predates comparable hashing, which is "
            "reported as 'not comparable' rather than as a mismatch."
        ),
    )
    data_source_matches: List[DataSourceMatch] = Field(default_factory=list)
    verdict: Literal[
        "reproduced", "data_changed", "code_changed", "not_comparable", "failed"
    ] = Field(
        ...,
        description=(
            "'reproduced' output and data both match. 'data_changed' the "
            "inputs no longer hash the same, so a different output is "
            "EXPECTED and says nothing about the code. 'code_changed' the "
            "data still matches but the output does not — the only "
            "combination that implicates the library. 'not_comparable' the "
            "record cannot be checked. 'failed' the replay itself errored."
        ),
    )
    notes: List[str] = Field(default_factory=list)


class CompareDecisionsInput(BaseModel):
    request_id_a: str = Field(..., min_length=1)
    request_id_b: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _distinct(self) -> "CompareDecisionsInput":
        if self.request_id_a == self.request_id_b:
            raise ValueError(
                "request_id_a and request_id_b are the same record; a diff "
                "against itself is always empty."
            )
        return self


class CompareDecisionsResult(BaseModel):
    request_id_a: str
    request_id_b: str
    same_tool: bool
    same_input: bool
    same_output: bool
    diff: str = Field(
        ..., description="Unified diff of the two records, as the CLI renders it."
    )
    summary: List[str] = Field(
        default_factory=list,
        description="Plain-language statement of what differs and what that implies.",
    )


class VerifyAuditIntegrityInput(BaseModel):
    date: Optional[str] = Field(
        None,
        description=(
            "YYYY-MM-DD to verify one day's file in isolation. None (the "
            "default) verifies the whole trail, which also catches a missing "
            "day that a per-file check cannot see."
        ),
    )
    public_key_path: Optional[str] = Field(
        None,
        description=(
            "Verify the Ed25519-signed checkpoint for `date` as well as the "
            "hash chain. Requires `date`. The chain alone detects partial "
            "tampering; only a signature detects a wholesale rewrite."
        ),
    )

    @model_validator(mode="after")
    def _key_needs_a_date(self) -> "VerifyAuditIntegrityInput":
        if self.public_key_path is not None and self.date is None:
            raise ValueError(
                "public_key_path needs a date — checkpoints are signed per "
                "calendar day, so there is no trail-wide signature to check."
            )
        return self


class VerifyAuditIntegrityResult(BaseModel):
    scope: str = Field(..., description="'trail' or the single date verified.")
    intact: bool
    problems: List[str] = Field(
        default_factory=list,
        description="Every broken link found, in the order encountered.",
    )
    checkpoint_signature_valid: Optional[bool] = Field(
        None, description="None when no public key was supplied."
    )
    notes: List[str] = Field(default_factory=list)


class ExportAuditBundleInput(BaseModel):
    start_date: str = Field(..., description="YYYY-MM-DD, inclusive.")
    end_date: str = Field(..., description="YYYY-MM-DD, inclusive.")
    out_path: str = Field(
        ...,
        description=(
            "Destination .zip path. This tool WRITES a new file; it never "
            "modifies or removes anything in the audit log."
        ),
    )

    @model_validator(mode="after")
    def _ordered_range(self) -> "ExportAuditBundleInput":
        if self.end_date < self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) precedes start_date "
                f"({self.start_date})"
            )
        return self


class ExportAuditBundleResult(BaseModel):
    out_path: str
    start_date: str
    end_date: str
    size_bytes: int
    notes: List[str] = Field(default_factory=list)
