"""
Agent-callable tool functions — designed for LLM function calling.
All inputs/outputs use Pydantic models for clean JSON serialization.
"""

import datetime
import logging
import math
import numbers
import time
import uuid
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from standard_quant_tools import audit
from standard_quant_tools._jsonsafe import sanitize_for_json
from standard_quant_tools.agent.models import (
    AdvancedIndicatorsInput,
    AdvancedIndicatorsResult,
    AnalysisInput,
    AnalysisResult,
    BacktestCompactInput,
    BacktestDiagnosticsInput,
    BacktestDiagnosticsResult,
    BacktestInput,
    BacktestOptInput,
    BacktestOptResult,
    BacktestResult,
    BacktestResultV2,
    BuyAndHoldInput,
    CapacityReportInput,
    CapacityReportResult,
    CointegrationInput,
    CointegrationResult,
    CompareStrategiesInput,
    CompareStrategiesResult,
    CorrelationAnalysisInput,
    CorrelationAnalysisResult,
    CostSummary,
    CustomSignalBacktestInput,
    DataQualityReportInput,
    DataQualityReportResult,
    DrawdownEpisode,
    ExposureDiagnostics,
    ExposureSummary,
    ExtendedRiskInput,
    ExtendedRiskResult,
    FactorRegressionInput,
    FactorRegressionResult,
    FundamentalsInput,
    FundamentalsResult,
    GarchVolatilityForecastInput,
    GarchVolatilityForecastResult,
    HurstInput,
    HurstResult,
    ImpliedVolatilityInput,
    ImpliedVolatilityResult,
    KalmanHedgeRatioInput,
    KalmanHedgeRatioResult,
    LiquidityAnalysisInput,
    LiquidityAnalysisResult,
    MissingBar,
    MonteCarloSimulationInput,
    MonteCarloSimulationResult,
    OptimizationRun,
    OptionGreeks,
    OptionPricingInput,
    OptionPricingResult,
    PairFailure,
    PairResult,
    PairScannerInput,
    PairScannerResult,
    PairTradeBacktestInput,
    PairTradeBacktestResult,
    PCAInput,
    PCAResult,
    PerformanceSummary,
    PortfolioInput,
    PortfolioOptimizationInput,
    PortfolioOptimizationResult,
    PortfolioResult,
    PortfolioSimulationInput,
    PortfolioSimulationResult,
    PositionSizerInput,
    PositionSizerResult,
    PriceJump,
    RallyDetectionInput,
    RallyDetectionResult,
    RebalanceEvent,
    RegimeAdaptiveInput,
    RegimeAdaptiveResult,
    RegimeAdaptiveWalkForwardInput,
    RegimeAdaptiveWalkForwardResult,
    RegimeAdaptiveWalkForwardWindow,
    RiskAttributionInput,
    RiskAttributionResult,
    RiskSummary,
    RobustnessDiagnosticsInput,
    RobustnessDiagnosticsResult,
    RollingBetaInput,
    RollingBetaResult,
    ScreenerInput,
    ScreenerResult,
    SignalPanelBacktestInput,
    SignalPanelBacktestResult,
    SignalType,
    StalePriceRun,
    StrategyComparison,
    StressTestInput,
    StressTestResult,
    TailRiskInput,
    TailRiskResult,
    TechnicalInput,
    TechnicalResult,
    Trade,
    TradeDiagnostics,
    VolatilityEstimatorsInput,
    VolatilityEstimatorsResult,
    WalkForwardInput,
    WalkForwardResult,
    WalkForwardWindow,
)
from standard_quant_tools.analysis.cointegration import (
    cointegration_test,
    compute_spread,
    kalman_hedge_ratio,
    spread_zscore,
)
from standard_quant_tools.analysis.correlation import (
    diversification_ratio,
    pairwise_correlation_summary,
)
from standard_quant_tools.analysis.garch import garch_volatility_forecast
from standard_quant_tools.analysis.hurst import hurst_exponent, rolling_hurst
from standard_quant_tools.analysis.multi_factor import (
    multi_factor_regression,
    rolling_factor_loadings,
)
from standard_quant_tools.analysis.options import (
    black_scholes_greeks,
    black_scholes_price,
)
from standard_quant_tools.analysis.options import (
    implied_volatility as _implied_volatility,
)
from standard_quant_tools.analysis.pca import factor_contributions, pca_returns
from standard_quant_tools.analysis.rally import detect_rally
from standard_quant_tools.analysis.regression import calculate_beta, rolling_beta
from standard_quant_tools.backtest.artifacts import save_artifact
from standard_quant_tools.backtest.constraints import (
    capacity_report as _capacity_report,
)
from standard_quant_tools.backtest.constraints import (
    days_to_liquidate as _days_to_liquidate,
)
from standard_quant_tools.backtest.constraints import (
    sector_exposure as _sector_exposure,
)
from standard_quant_tools.backtest.engine import backtest_grid, run_strategy
from standard_quant_tools.backtest.liquidity import (
    amihud_illiquidity,
    corwin_schultz_spread,
)
from standard_quant_tools.backtest.monte_carlo import simulate_forward_paths
from standard_quant_tools.backtest.pairs import run_pair_backtest as _pair_backtest_run
from standard_quant_tools.backtest.panel import (
    run_signal_panel_backtest as _signal_panel_backtest,
)
from standard_quant_tools.backtest.portfolio_engine import (
    run_portfolio_simulation as _portfolio_engine_run,
)
from standard_quant_tools.backtest.robustness import (
    block_bootstrap_ci as _block_bootstrap_ci,
)
from standard_quant_tools.backtest.robustness import (
    deflated_sharpe_ratio as _deflated_sharpe_ratio,
)
from standard_quant_tools.backtest.robustness import (
    parameter_sensitivity as _parameter_sensitivity,
)
from standard_quant_tools.backtest.sizing import (
    dollar_neutral,
    equal_weight_top_bottom,
    rank_weighted,
    vol_scaled,
    zscore_normalized,
)
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY
from standard_quant_tools.backtest.stress_test import (
    replay_stress_scenario,
    scenario_dates,
)
from standard_quant_tools.backtest.walk_forward import (
    longest_losing_streak,
    parameter_turnover,
)
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.data.quality import (
    detect_missing_bars,
    detect_price_jumps,
    detect_stale_prices,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.momentum import rsi, stochastic_oscillator
from standard_quant_tools.indicators.trend import (
    adx,
    ema,
    macd,
    parabolic_sar,
    sma,
    williams_r,
)
from standard_quant_tools.indicators.volatility import atr, bollinger_bands, wilder_atr
from standard_quant_tools.indicators.volume import mfi, obv, vwap
from standard_quant_tools.metrics.diagnostics import (
    exposure_stats,
    top_n_drawdowns,
    trade_excursions,
    trade_expectancy,
)
from standard_quant_tools.metrics.return_metrics import annualized_volatility, cagr
from standard_quant_tools.metrics.risk_metrics import (
    calmar_ratio,
    cvar,
    evt_tail_risk,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    treynor_ratio,
    var_historical,
    var_parametric,
)
from standard_quant_tools.metrics.volatility_estimators import (
    garman_klass_volatility,
    parkinson_volatility,
    yang_zhang_volatility,
)
from standard_quant_tools.portfolio.optimize import (
    _check_covariance_estimable,
    _small_sample_warnings,
    black_litterman,
    build_bl_views,
    mean_variance_optimize,
    risk_parity_weights,
)
from standard_quant_tools.portfolio.portfolio import (
    build_portfolio,
    fetch_ohlcv_panel_sync,
    fetch_returns_sync,
    portfolio_metrics,
)
from standard_quant_tools.screener.screener import screen_stocks

_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass

# Indicators technical_indicators() can compute in one native call. "atr" is
# deliberately excluded: the tool's plain atr() uses a simple rolling mean,
# while the fused call's ATR field is Wilder-smoothed -- a different
# algorithm, not just a faster path to the same numbers.
_FUSABLE_INDICATORS = {"rsi", "adx", "bollinger", "stochastic"}


def _parse_period(period: str) -> datetime.datetime:
    """Convert period string ('1y', '6mo', '2y') to a start datetime."""
    now = datetime.datetime.now()
    unit = period[-2:] if period.endswith("mo") else period[-1]
    num = int(period[: -2 if unit == "mo" else -1])
    if unit == "mo":
        return now - datetime.timedelta(days=num * 30)
    if unit == "y":
        return now - datetime.timedelta(days=num * 365)
    if unit == "d":
        return now - datetime.timedelta(days=num)
    return now - datetime.timedelta(days=365)


def _apply_signal_fill_policy(
    signal_series: pd.Series,
    price_index: pd.Index,
    policy: str,
) -> pd.Series:
    """
    A caller-submitted signal map only covers the dates it explicitly
    listed (e.g. a monthly rebalance signal against a daily price index).
    Reindexing onto the full price calendar HERE — not downstream, where
    run_strategy intersects the price and signal indices — is what keeps
    the backtest on the real daily bar time scale instead of silently
    collapsing to whatever sparse dates were submitted (which would also
    corrupt annualization, since every metric assumes periods_per_year=252
    daily bars).

    "hold" (default): forward-fill between submitted dates, flat (0.0)
        before the first one — correct for a target-position signal that's
        meant to persist until explicitly changed.
    "flat": no forward-fill — only the exact submitted dates carry a
        nonzero signal; every other bar is flat.
    "error": every price-calendar date must have an explicit entry.
    """
    if policy == "error":
        missing = price_index.difference(signal_series.index)
        if len(missing) > 0:
            raise ValidationError(
                f"signal_fill_policy='error': {len(missing)} price date(s) have no "
                f"corresponding signal entry (e.g. {[str(d.date()) for d in missing[:5]]})"
            )
        return signal_series.reindex(price_index)
    reindexed = signal_series.reindex(price_index)
    if policy == "hold":
        return reindexed.ffill().fillna(0.0)
    return reindexed.fillna(0.0)  # "flat"


# ──────────────────────────────────────────────────────────────────
# Backtesting Tools
# ──────────────────────────────────────────────────────────────────


def _run_backtest(
    input_data: BacktestInput,
    df: pd.DataFrame,
    signal_series: pd.Series,
) -> BacktestResult:
    """Shared backtest execution used by all strategy-specific tools."""
    logger.debug(
        "[backtest] %s  %s  %s → %s  capital=%.0f",
        input_data.strategy_type,
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.initial_capital,
    )
    results = run_strategy(
        df,
        signal_series,
        input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        include_trade_log=True,
        fill_price=input_data.fill_price,
    )

    trade_log_raw = results.get("trade_log", pd.DataFrame())
    trades = None
    if isinstance(trade_log_raw, pd.DataFrame) and not trade_log_raw.empty:
        trades = [
            Trade(
                entry_date=str(r["entry_date"]),
                exit_date=str(r["exit_date"]),
                direction=str(r["direction"]),
                entry_price=float(r["entry_price"]),
                exit_price=float(r["exit_price"]),
                position_size=float(r.get("position_size", 1.0)),
                return_pct=float(r["return_pct"]),
            )
            for r in trade_log_raw.to_dict(orient="records")
        ]

    bt = BacktestResult(
        total_return=results["total_return"],
        annualized_volatility=results["annualized_volatility"],
        sharpe_ratio=results["sharpe_ratio"],
        sortino_ratio=results["sortino_ratio"],
        max_drawdown=results["max_drawdown"],
        calmar_ratio=results["calmar_ratio"],
        win_rate=results["win_rate"],
        profit_factor=results["profit_factor"],
        num_trades=results["num_trades"],
        avg_trade_return_pct=results["avg_trade_return_pct"],
        final_equity=results["final_equity"],
        equity_curve=results["equity_curve"].tolist(),
        trade_log=trades,
        # run_strategy emits a look-ahead caveat for fill_price="close" (a
        # signal derived from bar t's own Close cannot realistically be
        # filled at that same Close). Rebuilding the result here without it
        # meant the engine knew the simulation might contain look-ahead
        # while the agent-facing output said nothing -- exactly the silent
        # behaviour this library exists to prevent.
        warnings=list(results.get("warnings", [])),
    )
    logger.debug(
        "[backtest] result  return=%.2f%%  sharpe=%.3f  maxdd=%.2f%%  trades=%d  win=%.0f%%",
        bt.total_return * 100,
        bt.sharpe_ratio,
        bt.max_drawdown * 100,
        bt.num_trades,
        bt.win_rate * 100,
    )
    return bt


def run_sma_backtest(input_data: BacktestInput) -> BacktestResult:
    """SMA crossover backtest: long when fast SMA > slow SMA."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY["sma_crossover"](df, **input_data.parameters)
    return _run_backtest(input_data, df, signals)


def run_rsi_backtest(input_data: BacktestInput) -> BacktestResult:
    """RSI mean-reversion backtest: enter long at oversold, exit at overbought."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY["rsi_mean_reversion"](df, **input_data.parameters)
    return _run_backtest(input_data, df, signals)


def run_macd_backtest(input_data: BacktestInput) -> BacktestResult:
    """MACD crossover backtest: long when MACD line crosses above signal line."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY["macd_crossover"](df, **input_data.parameters)
    return _run_backtest(input_data, df, signals)


def run_bollinger_backtest(input_data: BacktestInput) -> BacktestResult:
    """Bollinger Band mean-reversion: enter at lower band, exit at middle band."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY["bollinger_reversion"](df, **input_data.parameters)
    return _run_backtest(input_data, df, signals)


def run_buy_and_hold(input_data: BuyAndHoldInput) -> BacktestResult:
    """Buy-and-hold baseline: long the full period. Use to compare against active strategies."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = pd.Series(1.0, index=df.index)
    bt_input = BacktestInput(
        symbol=input_data.symbol,
        start_date=input_data.start_date,
        end_date=input_data.end_date,
        strategy_type="buy_and_hold",
        parameters={},
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        fill_price=input_data.fill_price,
    )
    return _run_backtest(bt_input, df, signals)


def compare_strategies(input_data: CompareStrategiesInput) -> CompareStrategiesResult:
    """
    Run all four strategies on the same symbol/period with a buy-and-hold baseline.
    Returns results sorted by sort_by (best first).
    Use this instead of calling the four backtest tools individually.
    """
    logger.debug(
        "[compare_strategies] %s  %s → %s  sort_by=%s",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.sort_by,
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )

    # Buy-and-hold baseline (long all bars, one trade)
    bh_signals = pd.Series(1.0, index=df.index)
    bh_input = BacktestInput(
        symbol=input_data.symbol,
        start_date=input_data.start_date,
        end_date=input_data.end_date,
        strategy_type="buy_and_hold",
        parameters={},
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        fill_price=input_data.fill_price,
    )
    bh = _run_backtest(bh_input, df, bh_signals)

    strategy_params: Dict[str, Dict[str, Any]] = {
        "sma_crossover": input_data.sma_parameters or _DEFAULT_PARAMS["sma_crossover"],
        "rsi_mean_reversion": input_data.rsi_parameters
        or _DEFAULT_PARAMS["rsi_mean_reversion"],
        "macd_crossover": input_data.macd_parameters
        or _DEFAULT_PARAMS["macd_crossover"],
        "bollinger_reversion": input_data.bollinger_parameters
        or _DEFAULT_PARAMS["bollinger_reversion"],
    }

    comparisons: List[StrategyComparison] = []
    for strat_name, params in strategy_params.items():
        signals = STRATEGY_REGISTRY[strat_name](df, **params)
        bt_input = BacktestInput(
            symbol=input_data.symbol,
            start_date=input_data.start_date,
            end_date=input_data.end_date,
            strategy_type=strat_name,
            parameters=params,
            initial_capital=input_data.initial_capital,
            commission_pct=input_data.commission_pct,
            slippage_pct=input_data.slippage_pct,
            fill_price=input_data.fill_price,
        )
        bt = _run_backtest(bt_input, df, signals)
        comparisons.append(
            StrategyComparison(
                strategy=strat_name,
                parameters=params,
                total_return=bt.total_return,
                sharpe_ratio=bt.sharpe_ratio,
                sortino_ratio=bt.sortino_ratio,
                max_drawdown=bt.max_drawdown,
                calmar_ratio=bt.calmar_ratio,
                win_rate=bt.win_rate,
                num_trades=bt.num_trades,
                final_equity=bt.final_equity,
            )
        )

    # Higher is always better for all supported metrics (max_drawdown: -0.10 > -0.30)
    comparisons.sort(
        key=lambda c: getattr(c, input_data.sort_by, 0.0),
        reverse=True,
    )
    logger.debug(
        "[compare_strategies] winner=%s  sharpe=%.3f  return=%.2f%%  vs B&H=%.2f%%",
        comparisons[0].strategy,
        comparisons[0].sharpe_ratio,
        comparisons[0].total_return * 100,
        bh.total_return * 100,
    )

    return CompareStrategiesResult(
        symbol=input_data.symbol,
        sort_by=input_data.sort_by,
        best_strategy=comparisons[0].strategy,
        buy_and_hold_return=bh.total_return,
        strategies=comparisons,
    )


# ──────────────────────────────────────────────────────────────────
# Risk Analysis Tool
# ──────────────────────────────────────────────────────────────────


def analyze_stock_risk(input_data: AnalysisInput) -> AnalysisResult:
    """Full risk profile: alpha, beta, Sharpe, Sortino, VaR, CVaR, Information Ratio."""
    logger.debug(
        "[analyze_risk] %s  vs %s  period=%s",
        input_data.symbol,
        input_data.benchmark,
        input_data.period,
    )
    provider = DataFactory.get_provider()
    end = datetime.datetime.now()
    start = _parse_period(input_data.period)

    asset_df = provider.get_ohlcv(input_data.symbol, start, end)
    bench_df = provider.get_ohlcv(input_data.benchmark, start, end)

    asset_ret = asset_df["Close"].pct_change().dropna()
    bench_ret = bench_df["Close"].pct_change().dropna()

    beta_metrics = calculate_beta(asset_ret, bench_ret)
    equity_curve = (1 + asset_ret).cumprod()

    result = AnalysisResult(
        symbol=input_data.symbol,
        benchmark=input_data.benchmark,
        alpha=round(beta_metrics["alpha"], 6),
        beta=round(beta_metrics["beta"], 4),
        r_squared=round(beta_metrics["r_squared"], 4),
        sharpe_ratio=round(sharpe_ratio(asset_ret), 4),
        sortino_ratio=round(sortino_ratio(asset_ret), 4),
        max_drawdown=round(max_drawdown(equity_curve), 6),
        var_95=round(var_historical(asset_ret, 0.95), 6),
        cvar_95=round(cvar(asset_ret, 0.95), 6),
        information_ratio=round(information_ratio(asset_ret, bench_ret), 4),
    )
    logger.debug(
        "[analyze_risk] beta=%.4f  alpha=%.6f  sharpe=%.3f  VaR95=%.3f%%  maxdd=%.2f%%",
        result.beta,
        result.alpha,
        result.sharpe_ratio,
        result.var_95 * 100,
        result.max_drawdown * 100,
    )
    return result


# ──────────────────────────────────────────────────────────────────
# Technical Analysis Tool
# ──────────────────────────────────────────────────────────────────


def get_technical_analysis(input_data: TechnicalInput) -> TechnicalResult:
    """
    Run a configurable set of indicators and return the last bar's values
    plus simple directional signals.
    """
    logger.debug(
        "[tech_analysis] %s  %s → %s  indicators=%s",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.indicators,
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    last_close = float(close.iloc[-1])

    last_vals: Dict[str, Any] = {"close": round(last_close, 4)}
    signals: Dict[str, Any] = {}
    requested = [ind.lower() for ind in input_data.indicators]

    # Fused fast path: when 2+ of {rsi, adx, bollinger, stochastic} are
    # requested, compute them in one native call instead of one per
    # indicator. Additive only -- the individual wrappers below are
    # unchanged and still used standalone whenever the fast path doesn't
    # apply (fewer than 2 fusable indicators, or C++ unavailable).
    fused: Optional[Dict[str, Any]] = None
    fusable_requested = _FUSABLE_INDICATORS & set(requested)
    if HAS_CPP and _cpp_core is not None and len(fusable_requested) >= 2:
        try:
            fused = _cpp_core.technical_indicators(
                high.to_numpy(dtype=np.float64),
                low.to_numpy(dtype=np.float64),
                close.to_numpy(dtype=np.float64),
                compute_rsi="rsi" in fusable_requested,
                compute_adx="adx" in fusable_requested,
                compute_bollinger="bollinger" in fusable_requested,
                compute_stochastic="stochastic" in fusable_requested,
            )
            logger.debug(
                "[tech_analysis] fused technical_indicators path for %s",
                sorted(fusable_requested),
            )
        except Exception as exc:
            logger.warning(
                "[tech_analysis] fused technical_indicators failed (%s) — "
                "falling back to per-indicator calls",
                exc,
            )
            fused = None

    if "sma" in requested:
        for p in (20, 50, 200):
            s = sma(close, p).dropna()
            if not s.empty:
                v = float(s.iloc[-1])
                last_vals[f"sma_{p}"] = round(v, 4)
                signals[f"price_above_sma_{p}"] = last_close > v

    if "ema" in requested:
        for p in (12, 26):
            e = ema(close, p).dropna()
            if not e.empty:
                last_vals[f"ema_{p}"] = round(float(e.iloc[-1]), 4)

    if "macd" in requested:
        m = macd(close)
        last_macd = float(m["MACD"].dropna().iloc[-1])
        last_sig = float(m["Signal"].dropna().iloc[-1])
        last_hist = float(m["Histogram"].dropna().iloc[-1])
        last_vals["macd"] = round(last_macd, 4)
        last_vals["macd_signal"] = round(last_sig, 4)
        last_vals["macd_histogram"] = round(last_hist, 4)
        signals["macd_bullish"] = last_macd > last_sig

    if "rsi" in requested:
        if fused is not None and "rsi" in fused:
            r = pd.Series(fused["rsi"], index=close.index).dropna()
        else:
            r = rsi(close, 14).dropna()
        if not r.empty:
            last_rsi = float(r.iloc[-1])
            last_vals["rsi_14"] = round(last_rsi, 2)
            signals["rsi_oversold"] = last_rsi < 30
            signals["rsi_overbought"] = last_rsi > 70

    if "stochastic" in requested:
        if fused is not None and "stochastic_oscillator" in fused:
            arr = fused["stochastic_oscillator"]
            stoch = pd.DataFrame(
                {"Stoch_K": arr[:, 0], "Stoch_D": arr[:, 1]}, index=close.index
            )
        else:
            stoch = stochastic_oscillator(high, low, close)
        k = float(stoch["Stoch_K"].dropna().iloc[-1])
        d = float(stoch["Stoch_D"].dropna().iloc[-1])
        last_vals["stoch_k"] = round(k, 2)
        last_vals["stoch_d"] = round(d, 2)
        signals["stoch_oversold"] = k < 20 and d < 20

    if "bollinger" in requested:
        if fused is not None and "bollinger_bands" in fused:
            arr = fused["bollinger_bands"]
            bb = pd.DataFrame(
                {"BB_Upper": arr[:, 0], "BB_Middle": arr[:, 1], "BB_Lower": arr[:, 2]},
                index=close.index,
            )
        else:
            bb = bollinger_bands(close)
        upper = float(bb["BB_Upper"].dropna().iloc[-1])
        middle = float(bb["BB_Middle"].dropna().iloc[-1])
        lower = float(bb["BB_Lower"].dropna().iloc[-1])
        last_vals["bb_upper"] = round(upper, 4)
        last_vals["bb_middle"] = round(middle, 4)
        last_vals["bb_lower"] = round(lower, 4)
        signals["price_near_lower_band"] = last_close <= lower * 1.01
        signals["price_near_upper_band"] = last_close >= upper * 0.99

    if "atr" in requested:
        a = atr(high, low, close).dropna()
        if not a.empty:
            last_vals["atr_14"] = round(float(a.iloc[-1]), 4)

    if "obv" in requested:
        o = obv(close, volume).dropna()
        if len(o) >= 2:
            last_vals["obv"] = int(float(o.iloc[-1]))
            signals["obv_rising"] = float(o.iloc[-1]) > float(o.iloc[-2])

    if "vwap" in requested:
        v = vwap(high, low, close, volume)
        if not v.dropna().empty:
            last_vwap = float(v.dropna().iloc[-1])
            last_vals["vwap"] = round(last_vwap, 4)
            signals["price_above_vwap"] = last_close > last_vwap

    if "adx" in requested:
        if fused is not None and "adx" in fused:
            arr = fused["adx"]
            adx_df = pd.DataFrame(
                {"DI_Plus": arr[:, 0], "DI_Minus": arr[:, 1], "ADX": arr[:, 2]},
                index=close.index,
            ).dropna()
        else:
            adx_df = adx(high, low, close).dropna()
        if not adx_df.empty:
            last_vals["adx"] = round(float(adx_df["ADX"].iloc[-1]), 2)
            last_vals["di_plus"] = round(float(adx_df["DI_Plus"].iloc[-1]), 2)
            last_vals["di_minus"] = round(float(adx_df["DI_Minus"].iloc[-1]), 2)
            signals["strong_trend"] = float(adx_df["ADX"].iloc[-1]) > 25
            signals["bullish_di"] = float(adx_df["DI_Plus"].iloc[-1]) > float(
                adx_df["DI_Minus"].iloc[-1]
            )

    if "williams_r" in requested:
        wr = williams_r(high, low, close).dropna()
        if not wr.empty:
            last_wr = float(wr.iloc[-1])
            last_vals["williams_r"] = round(last_wr, 2)
            signals["williams_r_oversold"] = last_wr < -80
            signals["williams_r_overbought"] = last_wr > -20

    return TechnicalResult(
        symbol=input_data.symbol,
        last_close=round(last_close, 4),
        signals=signals,
        last_values=last_vals,
    )


# ──────────────────────────────────────────────────────────────────
# Portfolio Analysis Tool
# ──────────────────────────────────────────────────────────────────


def get_portfolio_analysis(input_data: PortfolioInput) -> PortfolioResult:
    """Compute portfolio-level metrics with async data fetching."""
    logger.debug(
        "[portfolio_analysis] tickers=%s  weights=%s  vs %s  %s → %s",
        input_data.tickers,
        [round(w, 4) for w in input_data.weights],
        input_data.benchmark,
        input_data.start_date,
        input_data.end_date,
    )
    returns_df = fetch_returns_sync(
        input_data.tickers, input_data.start_date, input_data.end_date
    )
    bench_df_raw = DataFactory.get_provider().get_ohlcv(
        input_data.benchmark, input_data.start_date, input_data.end_date
    )
    bench_returns = bench_df_raw["Close"].pct_change().dropna()

    common_idx = returns_df.index.intersection(bench_returns.index)
    aligned_returns = returns_df.loc[common_idx]
    aligned_bench = bench_returns.loc[common_idx]

    metrics = portfolio_metrics(
        aligned_returns, input_data.weights, benchmark_returns=aligned_bench
    )

    return PortfolioResult(
        tickers=input_data.tickers,
        weights=input_data.weights,
        annualized_return=metrics["annualized_return"],
        annualized_volatility=metrics["annualized_volatility"],
        sharpe_ratio=metrics["sharpe_ratio"],
        sortino_ratio=metrics["sortino_ratio"],
        max_drawdown=metrics["max_drawdown"],
        calmar_ratio=metrics["calmar_ratio"],
        var_95=metrics["var_95"],
        cvar_95=metrics["cvar_95"],
        information_ratio=metrics.get("information_ratio", 0.0),
        total_return=metrics["total_return"],
        correlation_matrix=correlation_matrix_to_dict(aligned_returns),
    )


# ──────────────────────────────────────────────────────────────────
# Portfolio Optimization (produces weights, unlike get_portfolio_analysis
# above which only scores weights already chosen)
# ──────────────────────────────────────────────────────────────────


def run_portfolio_optimization(
    input_data: PortfolioOptimizationInput,
) -> PortfolioOptimizationResult:
    """
    Produce portfolio weights via Markowitz mean-variance (max_sharpe,
    min_volatility, target_return, target_volatility), risk parity, or
    Black-Litterman — unlike get_portfolio_analysis, which only scores
    weights the caller already picked.
    """
    logger.debug(
        "[portfolio_optimization] tickers=%s  method=%s  %s → %s",
        input_data.tickers,
        input_data.method,
        input_data.start_date,
        input_data.end_date,
    )
    returns_df = fetch_returns_sync(
        input_data.tickers, input_data.start_date, input_data.end_date
    )
    warnings: List[str] = []

    # Label weights from the columns the covariance was actually built from,
    # not from the requested list. PortfolioOptimizationInput now rejects
    # duplicates, which was the way the two could differ, but the invariant
    # worth holding is that a weight is named by the series it was computed
    # from -- so these stay coupled by construction rather than by a
    # validator elsewhere continuing to hold.
    solved_tickers = list(returns_df.columns)
    warnings.extend(_small_sample_warnings(returns_df.shape[0], len(solved_tickers)))

    if input_data.method == "risk_parity":
        # risk_parity and black_litterman bypass mean_variance_optimize, so
        # they need the same gate it applies -- otherwise a covariance that
        # is singular by construction reaches them too.
        _check_covariance_estimable(
            returns_df.shape[0], len(solved_tickers), returns_df.cov().to_numpy()
        )
        mu, cov = _mean_cov_for_tools(returns_df, input_data.periods_per_year)
        budget = (
            np.array([input_data.risk_budget[t] for t in input_data.tickers])
            if input_data.risk_budget is not None
            else None
        )
        rp = risk_parity_weights(cov, risk_budget=budget)
        if not rp["converged"]:
            warnings.append(
                "risk parity iteration did not converge within max_iterations "
                "— weights are the best achieved, not exact equal risk contribution"
            )
        w = rp["weights"]
        exp_ret = float(w @ mu)
        exp_vol = float(np.sqrt(w @ cov @ w))
        sharpe = (
            (exp_ret - input_data.risk_free_rate) / exp_vol if exp_vol > 1e-12 else 0.0
        )
        return PortfolioOptimizationResult(
            tickers=input_data.tickers,
            method=input_data.method,
            weights={t: round(float(wi), 6) for t, wi in zip(solved_tickers, w)},
            expected_return=round(exp_ret, 6),
            expected_volatility=round(exp_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            converged=rp["converged"],
            risk_contributions={
                t: round(float(c), 6)
                for t, c in zip(solved_tickers, rp["risk_contributions"])
            },
            warnings=warnings,
        )

    if input_data.method == "black_litterman":
        _check_covariance_estimable(
            returns_df.shape[0], len(solved_tickers), returns_df.cov().to_numpy()
        )
        mu, cov = _mean_cov_for_tools(returns_df, input_data.periods_per_year)
        n = len(input_data.tickers)
        mkt_w = (
            np.array([input_data.market_weights[t] for t in input_data.tickers])
            if input_data.market_weights is not None
            else np.full(n, 1.0 / n)
        )
        assert input_data.views  # validated non-empty by the Pydantic model
        P, Q, omega = build_bl_views(
            input_data.tickers,
            [v.model_dump() for v in input_data.views],
            cov,
            tau=input_data.tau,
        )
        bl = black_litterman(
            cov,
            mkt_w,
            P,
            Q,
            risk_aversion=input_data.risk_aversion,
            tau=input_data.tau,
            omega=omega,
        )
        w = bl["implied_weights"]
        exp_ret = float(w @ bl["posterior_returns"])
        exp_vol = float(np.sqrt(w @ bl["posterior_cov"] @ w))
        sharpe = (
            (exp_ret - input_data.risk_free_rate) / exp_vol if exp_vol > 1e-12 else 0.0
        )
        return PortfolioOptimizationResult(
            tickers=input_data.tickers,
            method=input_data.method,
            weights={t: round(float(wi), 6) for t, wi in zip(solved_tickers, w)},
            expected_return=round(exp_ret, 6),
            expected_volatility=round(exp_vol, 6),
            sharpe_ratio=round(sharpe, 4),
            converged=True,
            warnings=warnings,
        )

    result = mean_variance_optimize(
        returns_df,
        objective=input_data.method,
        risk_free_rate=input_data.risk_free_rate,
        target_return=input_data.target_return,
        target_volatility=input_data.target_volatility,
        allow_short=input_data.allow_short,
        max_weight=input_data.max_weight,
        periods_per_year=input_data.periods_per_year,
    )
    # The optimizer's own caveats (currently the small-sample covariance
    # warning) travel with the result rather than being dropped at this
    # boundary -- a 22x volatility understatement from too short a window is
    # exactly what an agent consuming this needs told.
    warnings.extend(result.get("warnings", []))
    if not result["converged"]:
        warnings.append(
            "optimizer did not converge — constraints may be infeasible; "
            "treat weights as approximate"
        )
    return PortfolioOptimizationResult(
        tickers=result["tickers"],
        method=input_data.method,
        weights={t: round(float(wi), 6) for t, wi in result["weights"].items()},
        expected_return=round(result["expected_return"], 6),
        expected_volatility=round(result["expected_volatility"], 6),
        sharpe_ratio=round(result["sharpe_ratio"], 4),
        converged=result["converged"],
        warnings=warnings,
    )


def _mean_cov_for_tools(
    returns_df: pd.DataFrame, periods_per_year: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Same annualized (mean, cov) computation portfolio.optimize's private
    _mean_cov does — reimplemented here rather than imported since it's a
    two-line pandas call and importing a leading-underscore helper across
    module boundaries would couple this file to that module's internals."""
    mu = returns_df.mean().to_numpy(dtype=float) * periods_per_year
    cov = returns_df.cov().to_numpy(dtype=float) * periods_per_year
    return mu, cov


def correlation_matrix_to_dict(returns_df: pd.DataFrame) -> Dict[str, Any]:
    from standard_quant_tools.portfolio.portfolio import correlation_matrix

    corr = correlation_matrix(returns_df)
    return corr.round(4).to_dict()


# ──────────────────────────────────────────────────────────────────
# Screener Tool
# ──────────────────────────────────────────────────────────────────


def run_screener(input_data: ScreenerInput) -> ScreenerResult:
    """Screen a universe of tickers against fundamental and technical filters."""
    logger.debug(
        "[screener_tool] universe=%d  filters=%s  sort_by=%s",
        len(input_data.tickers),
        list(input_data.filters.keys()),
        input_data.sort_by,
    )
    result_df = screen_stocks(
        input_data.tickers,
        input_data.filters,
        start_date=input_data.start_date,
        end_date=input_data.end_date,
        sort_by=input_data.sort_by,
        ascending=input_data.ascending,
        min_beta_obs=input_data.min_beta_obs,
    )
    failed_filters = dict(result_df.attrs.get("failed_filters", {}))
    failed_tickers = dict(result_df.attrs.get("failed_tickers", {}))
    failed_batches = list(result_df.attrs.get("failed_batches", []))

    if result_df.empty:
        return ScreenerResult(
            num_passed=0,
            tickers_passed=[],
            results=[],
            failed_filters=failed_filters,
            failed_tickers=failed_tickers,
            failed_batches=failed_batches,
        )

    records = result_df.reset_index().to_dict(orient="records")
    return ScreenerResult(
        num_passed=len(result_df),
        tickers_passed=list(result_df.index),
        results=records,
        failed_filters=failed_filters,
        failed_tickers=failed_tickers,
        failed_batches=failed_batches,
    )


# ──────────────────────────────────────────────────────────────────
# Factor Regression Tool
# ──────────────────────────────────────────────────────────────────


def run_factor_regression(input_data: FactorRegressionInput) -> FactorRegressionResult:
    """OLS multi-factor regression: alpha, loadings, t-stats, p-values, R²."""
    logger.debug(
        "[factor_regression] %s  factors=%s  %s → %s",
        input_data.symbol,
        input_data.factor_tickers,
        input_data.start_date,
        input_data.end_date,
    )
    provider = DataFactory.get_provider()
    asset_df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    asset_rets = asset_df["Close"].pct_change().dropna()

    names = input_data.factor_names or input_data.factor_tickers
    factor_series = {}
    for ticker, name in zip(input_data.factor_tickers, names):
        df = provider.get_ohlcv(ticker, input_data.start_date, input_data.end_date)
        factor_series[name] = df["Close"].pct_change().dropna()

    factors = pd.DataFrame(factor_series)
    result = multi_factor_regression(asset_rets, factors)

    rolling_alpha_tail = None
    rolling_loadings_tail = None
    if input_data.rolling_window:
        rolling = rolling_factor_loadings(
            asset_rets, factors, window=input_data.rolling_window
        )
        tail = rolling.dropna().tail(20)
        if not tail.empty:
            rolling_alpha_tail = [round(float(v), 6) for v in tail["alpha"].tolist()]
            rolling_loadings_tail = {
                col: [round(float(v), 6) for v in tail[col].tolist()]
                for col in tail.columns
                if col != "alpha"
            }

    return FactorRegressionResult(
        symbol=input_data.symbol,
        factors=names,
        alpha=round(float(result["alpha"]), 6),
        loadings={k: round(float(v), 6) for k, v in result["loadings"].items()},
        t_stats={
            k: round(float(v), 4) if not (v != v) else 0.0
            for k, v in result["t_stats"].items()
        },
        p_values={
            k: round(float(v), 4) if not (v != v) else 1.0
            for k, v in result["p_values"].items()
        },
        r_squared=round(float(result["r_squared"]), 4),
        adj_r_squared=(
            round(float(result["adj_r_squared"]), 4)
            if not (result["adj_r_squared"] != result["adj_r_squared"])
            else 0.0
        ),
        n_obs=result["n_obs"],
        rolling_alpha_tail=rolling_alpha_tail,
        rolling_loadings_tail=rolling_loadings_tail,
    )


# ──────────────────────────────────────────────────────────────────
# Cointegration Tool
# ──────────────────────────────────────────────────────────────────


def run_cointegration_test(input_data: CointegrationInput) -> CointegrationResult:
    """Engle-Granger cointegration test with hedge ratio, half-life, and a z-score signal."""
    logger.debug(
        "[cointegration] %s vs %s  %s → %s  z_window=%d",
        input_data.symbol_a,
        input_data.symbol_b,
        input_data.start_date,
        input_data.end_date,
        input_data.zscore_window,
    )
    provider = DataFactory.get_provider()
    prices_a = provider.get_ohlcv(
        input_data.symbol_a, input_data.start_date, input_data.end_date
    )["Close"]
    prices_b = provider.get_ohlcv(
        input_data.symbol_b, input_data.start_date, input_data.end_date
    )["Close"]

    result = cointegration_test(prices_a, prices_b)
    spread = compute_spread(prices_a, prices_b, hedge_ratio=result["hedge_ratio"])
    z = spread_zscore(spread, window=input_data.zscore_window)
    valid_z = z.dropna()
    current_z = round(float(valid_z.iloc[-1]), 4) if not valid_z.empty else 0.0

    if current_z < -2.0:
        signal = "long_a_short_b"
    elif current_z > 2.0:
        signal = "short_a_long_b"
    else:
        signal = "neutral"

    # Guard against inf half-life (non-mean-reverting spread)
    hl = result["half_life_days"]
    hl_safe = round(min(hl, 9999.0), 1) if hl != float("inf") else 9999.0

    return CointegrationResult(
        symbol_a=input_data.symbol_a,
        symbol_b=input_data.symbol_b,
        cointegrated=result["cointegrated"],
        p_value=round(float(result["p_value"]), 4),
        hedge_ratio=round(float(result["hedge_ratio"]), 4),
        adf_statistic=round(float(result["adf_statistic"]), 4),
        half_life_days=hl_safe,
        critical_values={
            k: round(float(v), 4) for k, v in result["critical_values"].items()
        },
        spread_mean=round(float(spread.mean()), 6),
        spread_std=round(float(spread.std()), 6),
        current_zscore=current_z,
        signal=signal,
        n_obs=result["n_obs"],
    )


# ──────────────────────────────────────────────────────────────────
# Kalman-Filter Dynamic Hedge Ratio Tool
# ──────────────────────────────────────────────────────────────────


def run_kalman_hedge_ratio(input_data: KalmanHedgeRatioInput) -> KalmanHedgeRatioResult:
    """
    Time-varying hedge ratio via a Kalman filter — a diagnostic companion
    to run_cointegration_test's static OLS hedge_ratio, useful for checking
    whether a static ratio has gone stale. NOT wired into run_pair_backtest,
    which still takes a single static hedge ratio for the whole window.
    """
    logger.debug(
        "[kalman_hedge_ratio] %s vs %s  %s → %s  delta=%.2e",
        input_data.symbol_a,
        input_data.symbol_b,
        input_data.start_date,
        input_data.end_date,
        input_data.delta,
    )
    provider = DataFactory.get_provider()
    prices_a = provider.get_ohlcv(
        input_data.symbol_a, input_data.start_date, input_data.end_date
    )["Close"]
    prices_b = provider.get_ohlcv(
        input_data.symbol_b, input_data.start_date, input_data.end_date
    )["Close"]

    kf = kalman_hedge_ratio(prices_a, prices_b, delta=input_data.delta)
    z = spread_zscore(kf["Spread"], window=input_data.zscore_window)
    valid_z = z.dropna()
    current_z = round(float(valid_z.iloc[-1]), 4) if not valid_z.empty else 0.0

    if current_z < -2.0:
        signal = "long_a_short_b"
    elif current_z > 2.0:
        signal = "short_a_long_b"
    else:
        signal = "neutral"

    beta_series = kf["Hedge_Ratio"]

    return KalmanHedgeRatioResult(
        symbol_a=input_data.symbol_a,
        symbol_b=input_data.symbol_b,
        current_hedge_ratio=round(float(beta_series.iloc[-1]), 4),
        current_intercept=round(float(kf["Intercept"].iloc[-1]), 4),
        hedge_ratio_std=round(float(beta_series.std()), 4),
        spread_mean=round(float(kf["Spread"].mean()), 6),
        spread_std=round(float(kf["Spread"].std()), 6),
        current_zscore=current_z,
        signal=signal,
        n_obs=len(kf),
    )


# ──────────────────────────────────────────────────────────────────
# PCA Tool
# ──────────────────────────────────────────────────────────────────


def run_pca_analysis(input_data: PCAInput) -> PCAResult:
    """PCA on multi-asset returns: explained variance, loadings, per-asset factor contributions."""
    logger.debug(
        "[pca_analysis] tickers=%s  n_components=%s  %s → %s",
        input_data.tickers,
        input_data.n_components,
        input_data.start_date,
        input_data.end_date,
    )
    provider = DataFactory.get_provider()
    returns = pd.DataFrame(
        {
            t: provider.get_ohlcv(t, input_data.start_date, input_data.end_date)[
                "Close"
            ].pct_change()
            for t in input_data.tickers
        }
    ).dropna()

    result = pca_returns(returns, n_components=input_data.n_components)
    contrib = factor_contributions(returns, n_components=input_data.n_components)

    evr = {k: round(float(v), 4) for k, v in result["explained_variance_ratio"].items()}
    cumvar = {
        k: round(float(v), 4) for k, v in result["cumulative_variance_ratio"].items()
    }

    loadings_dict = {
        pc: {
            t: round(float(result["loadings"].loc[t, pc]), 4)
            for t in input_data.tickers
        }
        for pc in result["loadings"].columns
    }

    contrib_dict = {
        t: {pc: round(float(contrib.loc[t, pc]), 4) for pc in contrib.columns}
        for t in contrib.index
    }

    return PCAResult(
        tickers=input_data.tickers,
        n_components=result["n_components"],
        n_obs=result["n_obs"],
        explained_variance_ratio=evr,
        cumulative_variance_ratio=cumvar,
        loadings=loadings_dict,
        factor_contributions=contrib_dict,
    )


def get_correlation_analysis(
    input_data: CorrelationAnalysisInput,
) -> CorrelationAnalysisResult:
    """
    Correlation matrix, avg pairwise correlation, most/least correlated
    pair, and the Choueifaty-Coignard diversification ratio for a universe.
    """
    logger.debug(
        "[correlation_analysis] tickers=%s  %s → %s  weighted=%s",
        input_data.tickers,
        input_data.start_date,
        input_data.end_date,
        input_data.weights is not None,
    )
    returns_df = fetch_returns_sync(
        input_data.tickers, input_data.start_date, input_data.end_date
    )

    summary = pairwise_correlation_summary(returns_df)
    dr = diversification_ratio(returns_df, weights=input_data.weights)

    corr_dict = {
        row_ticker: {
            col_ticker: round(float(value), 4) for col_ticker, value in row.items()
        }
        for row_ticker, row in summary["correlation_matrix"].iterrows()
    }

    return CorrelationAnalysisResult(
        tickers=input_data.tickers,
        correlation_matrix=corr_dict,
        avg_pairwise_correlation=round(summary["avg_pairwise_correlation"], 4),
        highest_correlated_pair={
            **summary["highest_correlated_pair"],
            "correlation": round(summary["highest_correlated_pair"]["correlation"], 4),
        },
        lowest_correlated_pair={
            **summary["lowest_correlated_pair"],
            "correlation": round(summary["lowest_correlated_pair"]["correlation"], 4),
        },
        diversification_ratio=round(dr, 4) if dr == dr else 0.0,
    )


# ──────────────────────────────────────────────────────────────────
# Hurst Tool
# ──────────────────────────────────────────────────────────────────


def run_hurst_analysis(input_data: HurstInput) -> HurstResult:
    """Hurst exponent via DFA or R/S. Optionally includes rolling regime breakdown."""
    logger.debug(
        "[hurst] %s  %s → %s  method=%s  rolling=%s",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.method,
        input_data.rolling_window,
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    returns = df["Close"].pct_change().dropna()

    result = hurst_exponent(returns, method=input_data.method)

    rolling_current = None
    rolling_regime_fractions = None
    if input_data.rolling_window:
        rolling = rolling_hurst(
            returns, window=input_data.rolling_window, method=input_data.method
        )
        valid = rolling.dropna()
        if not valid.empty:
            rolling_current = round(float(valid.iloc[-1]), 4)
            total = len(valid)
            rolling_regime_fractions = {
                "trending": round(float((valid > 0.55).sum() / total), 3),
                "random_walk": round(
                    float(((valid >= 0.45) & (valid <= 0.55)).sum() / total), 3
                ),
                "mean_reverting": round(float((valid < 0.45).sum() / total), 3),
            }

    h = result["hurst"]
    r2 = result["fit_r_squared"]

    return HurstResult(
        symbol=input_data.symbol,
        hurst=round(float(h), 4) if not (h != h) else 0.0,
        regime=result["regime"],
        fit_r_squared=round(float(r2), 4) if not (r2 != r2) else 0.0,
        method=result["method"],
        n_obs=result["n_obs"],
        rolling_current=rolling_current,
        rolling_regime_fractions=rolling_regime_fractions,
    )


def get_rally_signal(input_data: RallyDetectionInput) -> RallyDetectionResult:
    """
    Detect whether a symbol is currently rallying via 5 independent
    confirming signals (unusual positive return, ADX trend strength,
    bullish DI+/DI- direction, Hurst trending regime, new-high breakout)
    rather than trusting any single indicator alone. Optionally auto-tunes
    the ADX "strong trend" threshold to this symbol's own trailing ADX
    history instead of a fixed default (see auto_tune_adx_threshold).
    """
    logger.debug(
        "[rally] %s  %s → %s  lookback=%d  adx_threshold=%.1f  auto_tune=%s",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.lookback,
        input_data.adx_threshold,
        input_data.auto_tune_adx_threshold,
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )

    result = detect_rally(
        df,
        lookback=input_data.lookback,
        zscore_window=input_data.zscore_window,
        adx_period=input_data.adx_period,
        adx_threshold=input_data.adx_threshold,
        breakout_period=input_data.breakout_period,
        hurst_method=input_data.hurst_method,
        auto_tune_adx_threshold=input_data.auto_tune_adx_threshold,
        auto_tune_percentile=input_data.auto_tune_percentile,
    )

    return RallyDetectionResult(
        symbol=input_data.symbol,
        is_rally=result["is_rally"],
        rally_score=round(result["rally_score"], 4),
        trailing_return_pct=round(result["trailing_return_pct"], 6),
        return_zscore=round(result["return_zscore"], 4),
        adx=round(result["adx"], 4),
        di_plus=round(result["di_plus"], 4),
        di_minus=round(result["di_minus"], 4),
        trend_direction=result["trend_direction"],
        hurst=round(result["hurst"], 4),
        regime=result["regime"],
        is_new_high=result["is_new_high"],
        n_obs=result["n_obs"],
        adx_threshold_used=round(result["adx_threshold_used"], 4),
        auto_tuned=result["auto_tuned"],
    )


def get_volatility_estimators(
    input_data: VolatilityEstimatorsInput,
) -> VolatilityEstimatorsResult:
    """
    Realized volatility via Parkinson, Garman-Klass, and Yang-Zhang
    estimators, alongside plain close-to-close volatility for comparison.
    """
    logger.debug(
        "[volatility_estimators] %s  %s → %s  period=%d",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.period,
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    period = input_data.period

    close_returns = df["Close"].pct_change().dropna()
    close_to_close = annualized_volatility(close_returns)

    parkinson = parkinson_volatility(df["High"], df["Low"], period=period)
    garman_klass = garman_klass_volatility(
        df["Open"], df["High"], df["Low"], df["Close"], period=period
    )
    yang_zhang = yang_zhang_volatility(
        df["Open"], df["High"], df["Low"], df["Close"], period=period
    )

    def _latest(series: pd.Series) -> float:
        valid = series.dropna()
        return float(valid.iloc[-1]) if not valid.empty else float("nan")

    yz_val = _latest(yang_zhang)
    ctc_val = float(close_to_close)
    ratio = (yz_val / ctc_val) if ctc_val > 0 else float("nan")

    return VolatilityEstimatorsResult(
        symbol=input_data.symbol,
        period=period,
        close_to_close_annualized=round(ctc_val, 6),
        parkinson_annualized=round(_latest(parkinson), 6),
        garman_klass_annualized=round(_latest(garman_klass), 6),
        yang_zhang_annualized=round(yz_val, 6),
        yang_zhang_vs_close_to_close_ratio=round(ratio, 4),
    )


def run_garch_volatility_forecast(
    input_data: GarchVolatilityForecastInput,
) -> GarchVolatilityForecastResult:
    """
    GARCH(1,1) conditional volatility: fits how variance itself evolves
    (today's variance depends on yesterday's shock and yesterday's
    variance) and forecasts it forward — unlike get_volatility_estimators,
    which only describes past realized variance.
    """
    logger.debug(
        "[garch_volatility_forecast] %s  %s → %s  horizon=%d",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.forecast_horizon,
    )
    provider = DataFactory.get_provider()
    close = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )["Close"]
    returns = close.pct_change().dropna()

    result = garch_volatility_forecast(
        returns, forecast_horizon=input_data.forecast_horizon
    )

    return GarchVolatilityForecastResult(
        symbol=input_data.symbol,
        omega=result["omega"],
        alpha=round(result["alpha"], 6),
        beta=round(result["beta"], 6),
        persistence=round(result["persistence"], 6),
        converged=result["converged"],
        current_annualized_vol=round(result["current_annualized_vol"], 6),
        long_run_annualized_vol=round(result["long_run_annualized_vol"], 6),
        forecast_annualized_vol=[
            round(v, 6) for v in result["forecast_annualized_vol"]
        ],
        log_likelihood=round(result["log_likelihood"], 4),
        aic=round(result["aic"], 4),
        bic=round(result["bic"], 4),
        n_obs=result["n_obs"],
    )


# ──────────────────────────────────────────────────────────────────
# Default parameter grids for regime-adaptive strategy selection
# ──────────────────────────────────────────────────────────────────

_DEFAULT_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "sma_crossover": {
        "fast_period": [5, 10, 20],
        "slow_period": [30, 50, 100],
    },
    "rsi_mean_reversion": {
        "period": [7, 14, 21],
        "oversold": [25, 30],
        "overbought": [65, 70],
    },
    "macd_crossover": {
        "fast": [8, 12],
        "slow": [21, 26],
        "signal": [7, 9],
    },
    "bollinger_reversion": {
        "period": [15, 20, 25],
        "num_std": [1.5, 2.0],
    },
    "donchian_breakout": {
        "entry_period": [10, 20, 40],
        "exit_period": [5, 10],
    },
    "momentum_timeseries": {
        "lookback": [30, 60, 90],
        "threshold": [0.0, 0.05],
    },
    "vwap_reversion": {
        "period": [10, 20, 40],
        "entry_threshold": [0.01, 0.02, 0.03],
    },
    "adx_trend": {
        "adx_period": [10, 14, 21],
        "adx_threshold": [20.0, 25.0, 30.0],
    },
}

_REGIME_STRATEGY_MAP: Dict[str, str] = {
    "trending": "sma_crossover",
    "mean_reverting": "rsi_mean_reversion",
    "random_walk": "macd_crossover",
}

# Single canonical default parameters for each strategy (used by compare_strategies)
_DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "sma_crossover": {"fast_period": 10, "slow_period": 50},
    "rsi_mean_reversion": {"period": 14, "oversold": 30, "overbought": 70},
    "macd_crossover": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger_reversion": {"period": 20, "num_std": 2.0},
}


# ──────────────────────────────────────────────────────────────────
# Feature 1: Regime-Adaptive Strategy Selector
# ──────────────────────────────────────────────────────────────────


def run_regime_adaptive_backtest(
    input_data: RegimeAdaptiveInput,
) -> RegimeAdaptiveResult:
    """
    Classify the market regime via Hurst exponent, then automatically select
    and optimise the most appropriate strategy via parameter grid search.

    Regime → Strategy mapping:
      trending      → sma_crossover
      mean_reverting → rsi_mean_reversion
      random_walk   → macd_crossover
    """
    from standard_quant_tools.analysis.hurst import hurst_exponent as _hurst

    logger.debug(
        "[regime_adaptive] %s  %s → %s  hurst_method=%s",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.hurst_method,
    )

    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    returns = df["Close"].pct_change().dropna()

    hurst_result = _hurst(returns, method=input_data.hurst_method)
    h = hurst_result["hurst"]
    regime = hurst_result["regime"]
    fit_r2 = hurst_result["fit_r_squared"]

    # "unknown" is returned when Hurst is NaN (insufficient data); default to MACD
    strategy_name = _REGIME_STRATEGY_MAP.get(regime, "macd_crossover")

    grid_map = {
        "sma_crossover": input_data.sma_param_grid,
        "rsi_mean_reversion": input_data.rsi_param_grid,
        "macd_crossover": input_data.macd_param_grid,
        "bollinger_reversion": input_data.bollinger_param_grid,
    }
    param_grid = grid_map[strategy_name] or _DEFAULT_PARAM_GRIDS[strategy_name]

    grid_df = backtest_grid(
        df,
        strategy=strategy_name,
        param_grid=param_grid,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        sort_by="sharpe_ratio",
        ascending=False,
        n_workers=input_data.n_workers,
    )

    best_row = grid_df.iloc[0]
    param_keys = list(param_grid.keys())
    best_params: Dict[str, Any] = {
        k: (
            int(best_row[k])
            if isinstance(param_grid[k][0], int)
            else float(best_row[k])
        )
        for k in param_keys
    }

    signals = STRATEGY_REGISTRY[strategy_name](df, **best_params)
    dummy_input = BacktestInput(
        symbol=input_data.symbol,
        start_date=input_data.start_date,
        end_date=input_data.end_date,
        strategy_type=strategy_name,
        parameters=best_params,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
    )
    bt_result = _run_backtest(dummy_input, df, signals)

    n_combos = 1
    for vals in param_grid.values():
        n_combos *= len(vals)

    logger.debug(
        "[regime_adaptive] H=%.4f  regime=%s  strategy=%s  best_params=%s  combos=%d",
        float(h) if not math.isnan(float(h)) else 0.0,
        regime,
        strategy_name,
        best_params,
        n_combos,
    )
    return RegimeAdaptiveResult(
        symbol=input_data.symbol,
        regime=regime,
        hurst=round(float(h) if not math.isnan(float(h)) else 0.0, 4),
        fit_r_squared=round(float(fit_r2) if not math.isnan(float(fit_r2)) else 0.0, 4),
        selected_strategy=strategy_name,
        best_parameters=best_params,
        grid_combinations=n_combos,
        backtest=bt_result,
    )


# ──────────────────────────────────────────────────────────────────
# Feature 1b: Regime-Adaptive Walk-Forward Backtest (leakage-free)
# ──────────────────────────────────────────────────────────────────


def run_regime_adaptive_walkforward_backtest(
    input_data: RegimeAdaptiveWalkForwardInput,
) -> RegimeAdaptiveWalkForwardResult:
    """
    Leakage-free counterpart to run_regime_adaptive_backtest. That tool
    computes Hurst, optimises parameters, and backtests all on the same
    full requested range — useful as a quick exploratory check, but not
    out-of-sample validated. This tool instead walks forward exactly like
    run_walk_forward_backtest: at each non-overlapping window, regime
    detection AND strategy/parameter selection happen strictly on
    train_df, then the frozen (strategy, params) choice is evaluated
    strictly on the following test_df.

    Unlike run_regime_adaptive_backtest's hardcoded regime -> strategy map,
    every window here grid-searches all four registered strategies
    in-sample and keeps whichever wins by sort_by — the regime/Hurst value
    is still computed and reported per window, but purely as diagnostic
    context, not as a hard selector. Reuses backtest/walk_forward.py's
    stitching helpers for the aggregate OOS metrics — same math as
    run_walk_forward_backtest, no new stitching logic.
    """
    from standard_quant_tools.analysis.hurst import hurst_exponent as _hurst

    logger.debug(
        "[regime_adaptive_wf] %s  %s → %s  train=%d  test=%d  hurst_method=%s  sort_by=%s",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.train_bars,
        input_data.test_bars,
        input_data.hurst_method,
        input_data.sort_by,
    )

    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    n = len(df)

    train_bars = input_data.train_bars
    test_bars = input_data.test_bars
    if n < train_bars + test_bars:
        raise ValueError(
            f"Not enough data for regime-adaptive walk-forward: need at least "
            f"{train_bars + test_bars} bars, got {n}."
        )

    # Only the original 4 strategies have dedicated override fields on
    # RegimeAdaptiveWalkForwardInput; the 4 newer STRATEGY_REGISTRY entries
    # always use _DEFAULT_PARAM_GRIDS below (.get(...) -> None -> falls
    # through to the default), same as any future registry addition would
    # without a matching Pydantic field added here.
    grid_overrides: Dict[str, Optional[Dict[str, List[Any]]]] = {
        "sma_crossover": input_data.sma_param_grid,
        "rsi_mean_reversion": input_data.rsi_param_grid,
        "macd_crossover": input_data.macd_param_grid,
        "bollinger_reversion": input_data.bollinger_param_grid,
    }

    windows: List[RegimeAdaptiveWalkForwardWindow] = []
    oos_signal_tails: List[pd.Series] = []
    cursor = 0
    first_test_start = train_bars

    while cursor + train_bars + test_bars <= n:
        train_df = df.iloc[cursor : cursor + train_bars]
        test_df = df.iloc[cursor + train_bars : cursor + train_bars + test_bars]
        full_slice = df.iloc[cursor : cursor + train_bars + test_bars]

        train_returns = train_df["Close"].pct_change().dropna()
        hurst_result = _hurst(train_returns, method=input_data.hurst_method)
        h = hurst_result["hurst"]
        regime = hurst_result["regime"]
        fit_r2 = hurst_result["fit_r_squared"]

        best_overall: Optional[Dict[str, Any]] = None
        for strat_name in STRATEGY_REGISTRY:
            param_grid = (
                grid_overrides.get(strat_name) or _DEFAULT_PARAM_GRIDS[strat_name]
            )
            grid_df = backtest_grid(
                train_df,
                strategy=strat_name,
                param_grid=param_grid,
                initial_capital=input_data.initial_capital,
                commission_pct=input_data.commission_pct,
                slippage_pct=input_data.slippage_pct,
                sort_by=input_data.sort_by,
                ascending=False,
                n_workers=1,
            )
            best_row = grid_df.iloc[0]
            metric_val = float(best_row.get(input_data.sort_by, float("-inf")))
            if best_overall is None or metric_val > best_overall["metric_val"]:
                param_keys = list(param_grid.keys())
                best_params: Dict[str, Any] = {
                    k: (
                        int(best_row[k])
                        if isinstance(param_grid[k][0], int)
                        else float(best_row[k])
                    )
                    for k in param_keys
                }
                best_overall = {
                    "strategy": strat_name,
                    "params": best_params,
                    "metric_val": metric_val,
                    "sharpe": float(best_row.get("sharpe_ratio", 0.0)),
                    "return": float(best_row.get("total_return", 0.0)),
                }

        assert best_overall is not None  # STRATEGY_REGISTRY is never empty
        strategy_name = best_overall["strategy"]
        best_params = best_overall["params"]

        # Warm-up-aware, same as run_walk_forward_backtest: signals generated
        # over train+test together, keeping only the test-bars tail.
        oos_signals_full = STRATEGY_REGISTRY[strategy_name](full_slice, **best_params)
        oos_signals = oos_signals_full.iloc[train_bars:]
        oos_signal_tails.append(oos_signals)

        # Window-scoped diagnostic only — see stitched_oos_* below for the
        # single continuous OOS backtest.
        oos = run_strategy(
            test_df,
            oos_signals,
            initial_capital=input_data.initial_capital,
            commission_pct=input_data.commission_pct,
            slippage_pct=input_data.slippage_pct,
            fill_price=input_data.fill_price,
        )

        windows.append(
            RegimeAdaptiveWalkForwardWindow(
                window_index=len(windows),
                train_start=str(train_df.index[0].date()),
                train_end=str(train_df.index[-1].date()),
                test_start=str(test_df.index[0].date()),
                test_end=str(test_df.index[-1].date()),
                regime=regime,
                hurst=round(float(h) if not math.isnan(float(h)) else 0.0, 4),
                fit_r_squared=round(
                    float(fit_r2) if not math.isnan(float(fit_r2)) else 0.0, 4
                ),
                selected_strategy=strategy_name,
                best_params=best_params,
                in_sample_sharpe=round(best_overall["sharpe"], 4),
                in_sample_return=round(best_overall["return"], 6),
                out_of_sample_sharpe=round(float(oos["sharpe_ratio"]), 4),
                out_of_sample_return=round(float(oos["total_return"]), 4),
                out_of_sample_max_drawdown=round(float(oos["max_drawdown"]), 4),
            )
        )
        cursor += test_bars

    oos_sharpes = [w.out_of_sample_sharpe for w in windows]
    oos_returns = [w.out_of_sample_return for w in windows]
    oos_mdd = [w.out_of_sample_max_drawdown for w in windows]
    pct_profitable = sum(1 for r in oos_returns if r > 0) / len(windows)

    strat_counts = Counter(w.selected_strategy for w in windows)
    most_common_strat, most_common_n = strat_counts.most_common(1)[0]
    strategy_stability = {
        "most_common": most_common_strat,
        "frequency": round(most_common_n / len(windows), 3),
    }

    stitched_signals = pd.concat(oos_signal_tails)
    full_oos_df = df.iloc[first_test_start : cursor + train_bars]
    stitched = run_strategy(
        full_oos_df,
        stitched_signals,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        fill_price=input_data.fill_price,
    )
    worst_window = min(windows, key=lambda w: w.out_of_sample_return)

    result = RegimeAdaptiveWalkForwardResult(
        symbol=input_data.symbol,
        n_windows=len(windows),
        windows=windows,
        avg_oos_sharpe=round(float(np.mean(oos_sharpes)), 4),
        avg_oos_return=round(float(np.mean(oos_returns)), 4),
        avg_oos_max_drawdown=round(float(np.mean(oos_mdd)), 4),
        pct_windows_profitable=round(pct_profitable, 4),
        strategy_stability=strategy_stability,
        stitched_oos_return=round(stitched["total_return"], 6),
        stitched_oos_sharpe=round(stitched["sharpe_ratio"], 4),
        stitched_oos_sortino=round(stitched["sortino_ratio"], 4),
        stitched_oos_max_drawdown=round(stitched["max_drawdown"], 6),
        stitched_oos_calmar=round(stitched["calmar_ratio"], 4),
        worst_oos_window=worst_window.window_index,
        longest_losing_window_streak=longest_losing_streak(oos_returns),
    )
    logger.debug(
        "[regime_adaptive_wf] windows=%d  stitched_sharpe=%.3f  stitched_return=%.2f%%  strategy_stability=%s",
        result.n_windows,
        result.stitched_oos_sharpe,
        result.stitched_oos_return * 100,
        strategy_stability,
    )
    return result


# ──────────────────────────────────────────────────────────────────
# Feature 2: Cointegration Pair Scanner
# ──────────────────────────────────────────────────────────────────


def scan_pairs(input_data: PairScannerInput) -> PairScannerResult:
    """
    Test all ticker combinations for cointegration and return the top pairs
    ranked by half-life (shortest first = fastest mean-reversion = most tradeable).
    Fetches each ticker's prices once, then evaluates all O(n²/2) combinations.
    """
    from standard_quant_tools.analysis.cointegration import cointegration_test as _coint
    from standard_quant_tools.analysis.cointegration import compute_spread as _spread
    from standard_quant_tools.analysis.cointegration import spread_zscore as _zscore

    n_t = len(input_data.tickers)
    logger.debug(
        "[scan_pairs] universe=%d  combinations=%d  p_threshold=%.2f  hl_range=[%.0f, %.0f]",
        n_t,
        n_t * (n_t - 1) // 2,
        input_data.p_value_threshold,
        input_data.min_half_life,
        input_data.max_half_life,
    )

    provider = DataFactory.get_provider()

    prices: Dict[str, Optional[pd.Series]] = {}
    failed_tickers: Dict[str, str] = {}
    for ticker in input_data.tickers:
        try:
            df = provider.get_ohlcv(ticker, input_data.start_date, input_data.end_date)
            prices[ticker] = df["Close"]
        except Exception as exc:
            prices[ticker] = None
            failed_tickers[ticker] = str(exc)

    valid_tickers = [t for t, p in prices.items() if p is not None]
    all_pairs = list(combinations(valid_tickers, 2))
    n_tested = 0
    passing: List[PairResult] = []
    failed_pairs: List[PairFailure] = []

    for a, b in all_pairs:
        try:
            result = _coint(prices[a], prices[b])  # type: ignore[arg-type]
            n_tested += 1

            if (
                not result["cointegrated"]
                or result["p_value"] > input_data.p_value_threshold
            ):
                continue

            hl = result["half_life_days"]
            if (
                not math.isfinite(hl)
                or hl < input_data.min_half_life
                or hl > input_data.max_half_life
            ):
                continue

            spread = _spread(prices[a], prices[b], hedge_ratio=result["hedge_ratio"])  # type: ignore[arg-type]
            z = _zscore(spread, window=input_data.zscore_window).dropna()
            current_z = round(float(z.iloc[-1]), 4) if not z.empty else 0.0

            signal = (
                "long_a_short_b"
                if current_z < -2.0
                else "short_a_long_b" if current_z > 2.0 else "neutral"
            )

            passing.append(
                PairResult(
                    symbol_a=a,
                    symbol_b=b,
                    p_value=round(float(result["p_value"]), 4),
                    hedge_ratio=round(float(result["hedge_ratio"]), 4),
                    half_life_days=round(float(hl), 2),
                    adf_statistic=round(float(result["adf_statistic"]), 4),
                    current_zscore=current_z,
                    signal=signal,
                )
            )
        except Exception as exc:
            n_tested += 1
            failed_pairs.append(PairFailure(symbol_a=a, symbol_b=b, reason=str(exc)))

    passing.sort(key=lambda p: p.half_life_days)
    top = passing[: input_data.max_pairs]
    logger.debug(
        "[scan_pairs] tested=%d  cointegrated=%d (%.0f%%)  returning=%d  failed_pairs=%d  failed_tickers=%d",
        n_tested,
        len(passing),
        100 * len(passing) / max(n_tested, 1),
        len(top),
        len(failed_pairs),
        len(failed_tickers),
    )

    return PairScannerResult(
        n_pairs_tested=n_tested,
        n_pairs_cointegrated=len(passing),
        n_pairs_returned=len(top),
        pairs=top,
        failed_pairs=failed_pairs,
        failed_tickers=failed_tickers,
    )


# ──────────────────────────────────────────────────────────────────
# Feature 3: Walk-Forward Backtest
# ──────────────────────────────────────────────────────────────────


def run_walk_forward_backtest(input_data: WalkForwardInput) -> WalkForwardResult:
    """
    Walk-forward validation: repeatedly optimise parameters on an in-sample
    window (backtest_grid), then evaluate the best parameters on the next
    out-of-sample window. Returns per-window stats and aggregate OOS metrics.

    The OOS windows are non-overlapping; the training window slides forward
    by test_bars each step. Each window's OOS signals are generated from
    train_df + test_df together (so indicators get train_bars of warm-up
    before the OOS region starts, instead of computing over test_df alone —
    wrong for any indicator needing more history than test_bars provides),
    keeping only the test-bars tail. Those tails are then stitched
    chronologically into ONE continuous signal series and run through a
    SINGLE run_strategy call spanning the whole OOS region — one capital
    base, one compounding stream, real transaction costs at every actual
    transition including window boundaries (now just an ordinary bar, not a
    reset) — rather than resetting capital and re-running run_strategy
    independently per window. windows[i].out_of_sample_* stay window-scoped
    diagnostics (their own independent run_strategy call, still useful to
    see per-window) — stitched_oos_* are the economically correct aggregate,
    computed from the single continuous backtest above.
    """
    logger.debug(
        "[walk_forward] %s  strategy=%s  %s → %s  train=%d  test=%d  sort_by=%s",
        input_data.symbol,
        input_data.strategy,
        input_data.start_date,
        input_data.end_date,
        input_data.train_bars,
        input_data.test_bars,
        input_data.sort_by,
    )
    if input_data.strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{input_data.strategy}'. "
            f"Available: {list(STRATEGY_REGISTRY)}"
        )

    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    n = len(df)

    train_bars = input_data.train_bars
    test_bars = input_data.test_bars
    if n < train_bars + test_bars:
        raise ValueError(
            f"Not enough data for walk-forward: need at least "
            f"{train_bars + test_bars} bars, got {n}."
        )

    windows: List[WalkForwardWindow] = []
    oos_signal_tails: List[pd.Series] = []
    cursor = 0
    first_test_start = train_bars

    while cursor + train_bars + test_bars <= n:
        train_df = df.iloc[cursor : cursor + train_bars]
        test_df = df.iloc[cursor + train_bars : cursor + train_bars + test_bars]
        full_slice = df.iloc[cursor : cursor + train_bars + test_bars]

        grid_df = backtest_grid(
            train_df,
            strategy=input_data.strategy,
            param_grid=input_data.param_grid,
            initial_capital=input_data.initial_capital,
            commission_pct=input_data.commission_pct,
            slippage_pct=input_data.slippage_pct,
            sort_by=input_data.sort_by,
            ascending=False,
            n_workers=1,
        )

        best_row = grid_df.iloc[0]
        param_keys = list(input_data.param_grid.keys())
        # Coerce param types: DataFrame stores everything as float; restore original type.
        best_params: Dict[str, Any] = {
            k: (
                int(best_row[k])
                if isinstance(input_data.param_grid[k][0], int)
                else float(best_row[k])
            )
            for k in param_keys
        }
        is_sharpe = float(best_row.get("sharpe_ratio", 0.0))
        is_return = float(best_row.get("total_return", 0.0))

        # Warm-up-aware: generate signals over train+test together, keep only
        # the test-bars tail — a strategy needing more lookback than
        # test_bars alone provides now gets it, from this window's own
        # train_df (not reaching further back than cursor).
        oos_signals_full = STRATEGY_REGISTRY[input_data.strategy](
            full_slice, **best_params
        )
        oos_signals = oos_signals_full.iloc[train_bars:]
        oos_signal_tails.append(oos_signals)

        # Window-scoped diagnostic only — independently capitalized/reset,
        # not the aggregate (see stitched_oos_* below for the continuous one).
        oos = run_strategy(
            test_df,
            oos_signals,
            initial_capital=input_data.initial_capital,
            commission_pct=input_data.commission_pct,
            slippage_pct=input_data.slippage_pct,
            fill_price=input_data.fill_price,
        )

        windows.append(
            WalkForwardWindow(
                window_index=len(windows),
                train_start=str(train_df.index[0].date()),
                train_end=str(train_df.index[-1].date()),
                test_start=str(test_df.index[0].date()),
                test_end=str(test_df.index[-1].date()),
                best_params=best_params,
                in_sample_sharpe=round(is_sharpe, 4),
                in_sample_return=round(is_return, 6),
                out_of_sample_sharpe=round(float(oos["sharpe_ratio"]), 4),
                out_of_sample_return=round(float(oos["total_return"]), 4),
                out_of_sample_max_drawdown=round(float(oos["max_drawdown"]), 4),
            )
        )
        cursor += test_bars

    oos_sharpes = [w.out_of_sample_sharpe for w in windows]
    oos_returns = [w.out_of_sample_return for w in windows]
    oos_mdd = [w.out_of_sample_max_drawdown for w in windows]
    pct_profitable = sum(1 for r in oos_returns if r > 0) / len(windows)

    param_stability: Dict[str, Any] = {}
    for key in input_data.param_grid:
        counts = Counter(str(w.best_params[key]) for w in windows)
        most_common_val, most_common_n = counts.most_common(1)[0]
        param_stability[key] = {
            "most_common": most_common_val,
            "frequency": round(most_common_n / len(windows), 3),
        }

    # Continuous OOS backtest: one signal series spanning the whole OOS
    # region (windows are contiguous by construction, cursor += test_bars
    # each step), one run_strategy call, one capital base.
    stitched_signals = pd.concat(oos_signal_tails)
    full_oos_df = df.iloc[first_test_start : cursor + train_bars]
    stitched = run_strategy(
        full_oos_df,
        stitched_signals,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        fill_price=input_data.fill_price,
    )
    avg_is_sharpe = float(np.mean([w.in_sample_sharpe for w in windows]))
    avg_is_return = float(np.mean([w.in_sample_return for w in windows]))
    worst_window = min(windows, key=lambda w: w.out_of_sample_return)

    result_wf = WalkForwardResult(
        symbol=input_data.symbol,
        strategy=input_data.strategy,
        n_windows=len(windows),
        windows=windows,
        avg_oos_sharpe=round(float(np.mean(oos_sharpes)), 4),
        avg_oos_return=round(float(np.mean(oos_returns)), 4),
        avg_oos_max_drawdown=round(float(np.mean(oos_mdd)), 4),
        pct_windows_profitable=round(pct_profitable, 4),
        param_stability=param_stability,
        stitched_oos_return=round(stitched["total_return"], 6),
        stitched_oos_sharpe=round(stitched["sharpe_ratio"], 4),
        stitched_oos_sortino=round(stitched["sortino_ratio"], 4),
        stitched_oos_max_drawdown=round(stitched["max_drawdown"], 6),
        stitched_oos_calmar=round(stitched["calmar_ratio"], 4),
        is_to_oos_sharpe_decay=round(avg_is_sharpe - stitched["sharpe_ratio"], 4),
        is_to_oos_return_decay=round(avg_is_return - stitched["total_return"], 6),
        worst_oos_window=worst_window.window_index,
        longest_losing_window_streak=longest_losing_streak(oos_returns),
        parameter_turnover=parameter_turnover([w.best_params for w in windows]),
    )
    logger.debug(
        "[walk_forward] windows=%d  avg_OOS_sharpe=%.3f  avg_OOS_return=%.2f%%  profitable=%.0f%%",
        result_wf.n_windows,
        result_wf.avg_oos_sharpe,
        result_wf.avg_oos_return * 100,
        result_wf.pct_windows_profitable * 100,
    )
    return result_wf


# ──────────────────────────────────────────────────────────────────
# Feature 4: Portfolio Risk Attribution
# ──────────────────────────────────────────────────────────────────


def get_portfolio_risk_attribution(
    input_data: RiskAttributionInput,
) -> RiskAttributionResult:
    """
    Deep portfolio risk decomposition: portfolio-level metrics, per-asset
    marginal risk contributions (fractional, summing to 1), PCA decomposition
    of the asset universe, and an optional multi-factor regression on the
    aggregate portfolio returns.
    """
    from standard_quant_tools.analysis.multi_factor import (
        multi_factor_regression as _mfr,
    )
    from standard_quant_tools.analysis.pca import pca_returns as _pca

    logger.debug(
        "[risk_attribution] tickers=%s  weights=%s  vs %s  %s → %s",
        input_data.tickers,
        [round(w, 4) for w in input_data.weights],
        input_data.benchmark,
        input_data.start_date,
        input_data.end_date,
    )

    provider = DataFactory.get_provider()
    weights = np.array(input_data.weights, dtype=float)

    returns_df = pd.DataFrame(
        {
            t: provider.get_ohlcv(t, input_data.start_date, input_data.end_date)[
                "Close"
            ].pct_change()
            for t in input_data.tickers
        }
    ).dropna()

    bench_ret = (
        provider.get_ohlcv(
            input_data.benchmark, input_data.start_date, input_data.end_date
        )["Close"]
        .pct_change()
        .dropna()
    )

    port_arr: np.ndarray = returns_df.values @ weights
    port_ret = pd.Series(port_arr, index=returns_df.index)
    port_equity = (1 + port_ret).cumprod()

    common = port_ret.index.intersection(bench_ret.index)
    bench_aligned = bench_ret.loc[common]
    port_aligned = port_ret.loc[common]

    # ── Portfolio-level metrics ────────────────────────────────────
    ann_ret = float(cagr(port_equity))
    ann_vol = float(annualized_volatility(port_ret))
    sr = float(sharpe_ratio(port_ret))
    sort_r = float(sortino_ratio(port_ret))
    mdd = float(max_drawdown(port_equity))
    v95 = float(var_historical(port_ret, 0.95))
    cv95 = float(cvar(port_ret, 0.95))
    ir = float(information_ratio(port_aligned, bench_aligned))

    # ── Marginal Risk Contribution (fraction of portfolio variance) ─
    cov_ann = returns_df.cov().values * 252
    port_var = float(weights @ cov_ann @ weights)
    mcr = (
        (cov_ann @ weights) * weights / port_var
        if port_var > 0
        else np.zeros(len(weights))
    )
    asset_risk_contribs = {
        t: round(float(mcr[i]), 6) for i, t in enumerate(input_data.tickers)
    }

    # ── PCA decomposition ─────────────────────────────────────────
    n_comp = min(input_data.n_components, len(input_data.tickers))
    pca_res = _pca(returns_df, n_components=n_comp)
    evr = {
        k: round(float(v), 4) for k, v in pca_res["explained_variance_ratio"].items()
    }
    loadings_mat = pca_res["loadings"].values  # (n_assets, n_comp)
    port_exposures = weights @ loadings_mat  # (n_comp,)
    pc_names = list(pca_res["explained_variance_ratio"].index)
    port_pc_exposures = {
        pc_names[i]: round(float(port_exposures[i]), 4) for i in range(n_comp)
    }

    # ── Optional factor regression on portfolio returns ───────────
    factor_loadings: Optional[Dict[str, float]] = None
    factor_r2: Optional[float] = None
    factor_alpha: Optional[float] = None

    if input_data.factor_tickers:
        names = input_data.factor_names or input_data.factor_tickers
        factor_df = pd.DataFrame(
            {
                name: provider.get_ohlcv(
                    tick, input_data.start_date, input_data.end_date
                )["Close"].pct_change()
                for name, tick in zip(names, input_data.factor_tickers)
            }
        ).dropna()

        mfr = _mfr(port_ret, factor_df)
        factor_loadings = {k: round(float(v), 4) for k, v in mfr["loadings"].items()}
        factor_r2 = round(float(mfr["r_squared"]), 4)
        factor_alpha = round(float(mfr["alpha"]), 6)

    return RiskAttributionResult(
        tickers=input_data.tickers,
        weights=list(input_data.weights),
        annualized_return=round(ann_ret, 4),
        annualized_volatility=round(ann_vol, 4),
        sharpe_ratio=round(sr, 4),
        sortino_ratio=round(sort_r, 4),
        max_drawdown=round(mdd, 6),
        var_95=round(v95, 6),
        cvar_95=round(cv95, 6),
        information_ratio=round(ir, 4),
        asset_risk_contributions=asset_risk_contribs,
        pca_variance_explained=evr,
        portfolio_pc_exposures=port_pc_exposures,
        factor_loadings=factor_loadings,
        factor_r_squared=factor_r2,
        factor_alpha=factor_alpha,
    )


def run_stress_test(input_data: StressTestInput) -> StressTestResult:
    """
    Replay a portfolio's current weights against a named historical crash
    window (or a custom date range) using each ticker's own real historical
    returns for that window. A ticker with no data that far back (e.g. it
    hadn't listed yet) is isolated into tickers_missing_data rather than
    failing the whole call — the replay proceeds on whatever subset of
    tickers actually has data.
    """
    if input_data.scenario == "custom":
        # StressTestInput._check_custom_dates already guarantees both are
        # set when scenario == "custom" (raises otherwise) — asserted here
        # too so this stays a str, not Optional[str], for every caller below.
        assert input_data.custom_start_date is not None
        assert input_data.custom_end_date is not None
        scenario_start = input_data.custom_start_date
        scenario_end = input_data.custom_end_date
    else:
        scenario_start, scenario_end = scenario_dates(input_data.scenario)

    logger.debug(
        "[stress_test] tickers=%s  scenario=%s  %s → %s",
        input_data.tickers,
        input_data.scenario,
        scenario_start,
        scenario_end,
    )

    provider = DataFactory.get_provider()
    returns_by_ticker: Dict[str, pd.Series] = {}
    tickers_missing_data: List[str] = []
    for t in input_data.tickers:
        try:
            close = provider.get_ohlcv(t, scenario_start, scenario_end)["Close"]
            returns = close.pct_change().dropna()
            if returns.empty:
                raise ValueError("no trading days with data in this window")
            returns_by_ticker[t] = returns
        except Exception as exc:
            logger.debug(
                "[stress_test] %s has no data for %s → %s (%s)",
                t,
                scenario_start,
                scenario_end,
                exc,
            )
            tickers_missing_data.append(t)

    if not returns_by_ticker:
        raise ValidationError(
            f"None of {input_data.tickers} have any data for {scenario_start} → "
            f"{scenario_end} — cannot replay this scenario."
        )

    tickers_used = list(returns_by_ticker.keys())
    returns_df = pd.DataFrame(returns_by_ticker).dropna()

    if input_data.weights is None:
        weights = np.full(len(input_data.tickers), 1.0 / len(input_data.tickers))
    else:
        weights = np.array(input_data.weights, dtype=float)
    # Re-normalize weights over just the tickers that actually have data,
    # preserving their relative proportions, rather than silently treating
    # a missing ticker's weight as if it had earned a 0% return.
    weight_by_ticker = dict(zip(input_data.tickers, weights))
    used_weights = np.array([weight_by_ticker[t] for t in tickers_used])
    used_weights_sum = used_weights.sum()
    if used_weights_sum <= 0:
        raise ValidationError(
            "the tickers with available data for this scenario carry zero "
            "total weight — cannot replay."
        )
    used_weights = used_weights / used_weights_sum

    result = replay_stress_scenario(returns_df, used_weights)

    return StressTestResult(
        scenario=input_data.scenario,
        scenario_start_date=scenario_start,
        scenario_end_date=scenario_end,
        tickers_used=tickers_used,
        tickers_missing_data=tickers_missing_data,
        portfolio_return_pct=round(result["portfolio_return_pct"], 6),
        max_drawdown_pct=round(result["max_drawdown_pct"], 6),
        worst_day_return_pct=round(result["worst_day_return_pct"], 6),
        worst_day_date=result["worst_day_date"],
        best_day_return_pct=round(result["best_day_return_pct"], 6),
        best_day_date=result["best_day_date"],
        n_trading_days=result["n_trading_days"],
    )


# ──────────────────────────────────────────────────────────────────
# Feature 5: ATR-Based Position Sizer
# ──────────────────────────────────────────────────────────────────


def get_position_size(input_data: PositionSizerInput) -> PositionSizerResult:
    """
    Compute risk-adjusted position size using ATR-based stop-loss sizing
    and optionally Kelly criterion when strategy statistics are provided.

    Fixed-risk sizing: shares = (account × risk_pct) / (atr_multiplier × ATR)
    Kelly sizing:      f = (b×p − q) / b  where b = avg_win/avg_loss,
                       p = win_rate, q = 1−win_rate. Half-Kelly is recommended.
    """
    logger.debug(
        "[position_size] %s  equity=%.0f  risk_pct=%.2f%%  atr_period=%d  atr_mult=%.1f",
        input_data.symbol,
        input_data.account_equity,
        input_data.risk_per_trade_pct * 100,
        input_data.atr_period,
        input_data.atr_multiplier,
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )

    last_close = float(df["Close"].iloc[-1])
    atr_series = atr(
        df["High"], df["Low"], df["Close"], period=input_data.atr_period
    ).dropna()
    last_atr = float(atr_series.iloc[-1])

    stop_distance = last_atr * input_data.atr_multiplier
    dollar_risk = input_data.account_equity * input_data.risk_per_trade_pct

    shares_fr = max(int(dollar_risk / stop_distance), 0) if stop_distance > 0 else 0
    pos_val_fr = shares_fr * last_close
    port_pct_fr = (
        pos_val_fr / input_data.account_equity if input_data.account_equity > 0 else 0.0
    )

    kelly_fraction: Optional[float] = None
    shares_hk: Optional[int] = None
    pos_val_hk: Optional[float] = None
    port_pct_hk: Optional[float] = None

    has_kelly_inputs = (
        input_data.win_rate is not None
        and input_data.avg_win_pct is not None
        and input_data.avg_loss_pct is not None
    )

    if has_kelly_inputs:
        assert input_data.win_rate is not None
        assert input_data.avg_win_pct is not None
        assert input_data.avg_loss_pct is not None
        wr, aw, al = (
            input_data.win_rate,
            input_data.avg_win_pct,
            input_data.avg_loss_pct,
        )
        b = aw / al if al > 0 else 0.0
        raw_kelly = (b * wr - (1.0 - wr)) / b if b > 0 else 0.0
        kelly_fraction = round(max(raw_kelly, 0.0), 4)

        half_kelly_equity = input_data.account_equity * kelly_fraction * 0.5
        shares_hk = max(int(half_kelly_equity / last_close), 0) if last_close > 0 else 0
        pos_val_hk = shares_hk * last_close
        port_pct_hk = (
            pos_val_hk / input_data.account_equity
            if input_data.account_equity > 0
            else 0.0
        )

    use_kelly = (
        has_kelly_inputs
        and kelly_fraction is not None
        and kelly_fraction > 0
        and (shares_hk or 0) > 0
    )
    recommended_sizing = "half_kelly" if use_kelly else "fixed_risk"
    _rec_shares: int = (shares_hk or 0) if use_kelly else shares_fr  # type: ignore[assignment]
    recommended_value = _rec_shares * last_close
    logger.debug(
        "[position_size] close=%.4f  ATR=%.4f  stop=%.4f  sizing=%s  shares=%d  value=%.2f  kelly=%s",
        last_close,
        last_atr,
        stop_distance,
        recommended_sizing,
        _rec_shares,
        recommended_value,
        str(kelly_fraction),
    )

    return PositionSizerResult(
        symbol=input_data.symbol,
        last_close=round(last_close, 4),
        atr=round(last_atr, 4),
        atr_pct=round(last_atr / last_close * 100, 4) if last_close > 0 else 0.0,
        stop_distance=round(stop_distance, 4),
        shares_fixed_risk=shares_fr,
        position_value_fixed_risk=round(pos_val_fr, 2),
        portfolio_pct_fixed_risk=round(port_pct_fr, 4),
        max_loss_fixed_risk=round(shares_fr * stop_distance, 2),
        kelly_fraction=kelly_fraction,
        shares_half_kelly=shares_hk,
        position_value_half_kelly=(
            round(pos_val_hk, 2) if pos_val_hk is not None else None
        ),
        portfolio_pct_half_kelly=(
            round(port_pct_hk, 4) if port_pct_hk is not None else None
        ),
        recommended_sizing=recommended_sizing,
        recommended_shares=_rec_shares,
        recommended_position_value=round(recommended_value, 2),
    )


# ──────────────────────────────────────────────────────────────────
# Tool Registry for LLM Function Calling
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# Tool Registry for LLM Function Calling
# ──────────────────────────────────────────────────────────────────
# Stock Fundamentals Tool
# ──────────────────────────────────────────────────────────────────


def get_stock_fundamentals(input_data: FundamentalsInput) -> FundamentalsResult:
    """Fetch company metadata and key financial ratios for a single ticker."""
    logger.debug("[fundamentals] %s", input_data.symbol)
    provider = DataFactory.get_provider()
    info = provider.get_ticker_info(input_data.symbol)
    ratios = provider.get_financial_ratios(input_data.symbol)
    logger.debug(
        "[fundamentals] %s  sector=%s  fwd_pe=%s  mktcap=%s",
        info.name,
        info.sector,
        ratios.forward_pe,
        f"${ratios.market_cap/1e9:.1f}B" if ratios.market_cap else "N/A",
    )
    return FundamentalsResult(
        symbol=input_data.symbol,
        name=info.name,
        sector=info.sector,
        industry=info.industry,
        country=info.country,
        full_time_employees=info.full_time_employees,
        forward_pe=ratios.forward_pe,
        trailing_pe=ratios.trailing_pe,
        price_to_book=ratios.price_to_book,
        debt_to_equity=ratios.debt_to_equity,
        return_on_equity=ratios.return_on_equity,
        profit_margins=ratios.profit_margins,
        dividend_yield=ratios.dividend_yield,
        market_cap=ratios.market_cap,
    )


# ──────────────────────────────────────────────────────────────────
# Backtest Optimization Tool
# ──────────────────────────────────────────────────────────────────


def run_backtest_optimization(input_data: BacktestOptInput) -> BacktestOptResult:
    """
    Run a full parameter grid search for one strategy and return the top N combinations.
    Use this to find the best parameters before committing to a single backtest.
    """
    logger.debug(
        "[backtest_opt] %s  %s  %s → %s  sort_by=%s",
        input_data.symbol,
        input_data.strategy,
        input_data.start_date,
        input_data.end_date,
        input_data.sort_by,
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )

    grid_df = backtest_grid(
        price_data=df,
        strategy=input_data.strategy,
        param_grid=input_data.param_grid,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        sort_by=input_data.sort_by,
        ascending=False,
        n_workers=input_data.n_workers,
        fill_price=input_data.fill_price,
    )

    n_combinations = len(grid_df)
    top_n = min(input_data.top_n, 20, n_combinations)
    top_df = grid_df.head(top_n)

    metric_cols = {
        "total_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "win_rate",
        "profit_factor",
        "num_trades",
        "avg_trade_return_pct",
        "final_equity",
    }
    param_cols = [c for c in grid_df.columns if c not in metric_cols]

    top_results = [
        OptimizationRun(
            rank=rank,
            parameters={col: row[col] for col in param_cols if col in row},
            total_return=round(float(row.get("total_return", 0.0)), 6),
            sharpe_ratio=round(float(row.get("sharpe_ratio", 0.0)), 4),
            sortino_ratio=round(float(row.get("sortino_ratio", 0.0)), 4),
            calmar_ratio=round(float(row.get("calmar_ratio", 0.0)), 4),
            max_drawdown=round(float(row.get("max_drawdown", 0.0)), 6),
            num_trades=int(row.get("num_trades", 0)),
        )
        for rank, (_, row) in enumerate(top_df.iterrows(), start=1)
    ]

    best_row = top_df.iloc[0]
    best_params = {col: best_row[col] for col in param_cols if col in best_row}
    logger.debug(
        "[backtest_opt] n=%d  best_params=%s  %s=%.4f",
        n_combinations,
        best_params,
        input_data.sort_by,
        float(best_row.get(input_data.sort_by, 0.0)),
    )
    return BacktestOptResult(
        symbol=input_data.symbol,
        strategy=input_data.strategy,
        n_combinations=n_combinations,
        sort_by=input_data.sort_by,
        best_params=best_params,
        best_sharpe=round(float(best_row.get("sharpe_ratio", 0.0)), 4),
        best_return=round(float(best_row.get("total_return", 0.0)), 6),
        top_results=top_results,
    )


# ──────────────────────────────────────────────────────────────────
# Advanced Indicators Tool
# ──────────────────────────────────────────────────────────────────


def get_advanced_indicators(
    input_data: AdvancedIndicatorsInput,
) -> AdvancedIndicatorsResult:
    """
    Compute Parabolic SAR (trend), Wilder ATR (volatility), and MFI (volume-flow).
    Complements get_technical_analysis with indicators not included there.
    """
    logger.debug(
        "[advanced_indicators] %s  %s → %s",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]
    last_close = float(close.iloc[-1])

    sar_df = parabolic_sar(
        high, low, af_start=input_data.sar_af_start, af_max=input_data.sar_af_max
    )
    sar_val = float(sar_df["SAR"].iloc[-1])
    sar_trend_int = int(sar_df["Trend"].iloc[-1])
    sar_trend = "bullish" if sar_trend_int == 1 else "bearish"
    sar_signal = "buy" if sar_trend_int == 1 else "sell"

    watr_series = wilder_atr(high, low, close, period=input_data.atr_period).dropna()
    watr_val = float(watr_series.iloc[-1])
    watr_pct = watr_val / last_close if last_close > 0 else 0.0

    mfi_series = mfi(high, low, close, volume, period=input_data.mfi_period).dropna()
    mfi_val = float(mfi_series.iloc[-1])
    if mfi_val >= 80:
        mfi_signal = "overbought"
    elif mfi_val <= 20:
        mfi_signal = "oversold"
    else:
        mfi_signal = "neutral"

    logger.debug(
        "[advanced_indicators] SAR=%.4f (%s)  ATR=%.4f (%.2f%%)  MFI=%.1f (%s)",
        sar_val,
        sar_trend,
        watr_val,
        watr_pct * 100,
        mfi_val,
        mfi_signal,
    )
    return AdvancedIndicatorsResult(
        symbol=input_data.symbol,
        last_close=round(last_close, 4),
        sar_value=round(sar_val, 4),
        sar_trend=sar_trend,
        sar_signal=sar_signal,
        wilder_atr=round(watr_val, 4),
        wilder_atr_pct=round(watr_pct, 6),
        mfi=round(mfi_val, 2),
        mfi_signal=mfi_signal,
    )


# ──────────────────────────────────────────────────────────────────
# Rolling Beta Tool
# ──────────────────────────────────────────────────────────────────


def get_rolling_beta(input_data: RollingBetaInput) -> RollingBetaResult:
    """
    Compute rolling OLS beta to detect beta drift over time.
    Use alongside analyze_stock_risk (static beta) for a fuller market-sensitivity picture.
    """
    logger.debug(
        "[rolling_beta] %s vs %s  %s → %s  window=%d",
        input_data.symbol,
        input_data.benchmark,
        input_data.start_date,
        input_data.end_date,
        input_data.window,
    )
    provider = DataFactory.get_provider()
    asset_df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    bench_df = provider.get_ohlcv(
        input_data.benchmark, input_data.start_date, input_data.end_date
    )

    asset_ret = asset_df["Close"].pct_change().dropna()
    bench_ret = bench_df["Close"].pct_change().dropna()

    rb = rolling_beta(asset_ret, bench_ret, window=input_data.window)[
        "Rolling_Beta"
    ].dropna()
    if rb.empty:
        raise ValueError(
            f"Not enough data for rolling beta with window={input_data.window}"
        )

    current = float(rb.iloc[-1])
    b_1m = float(rb.iloc[-22]) if len(rb) >= 22 else None
    b_3m = float(rb.iloc[-63]) if len(rb) >= 63 else None
    b_6m = float(rb.iloc[-126]) if len(rb) >= 126 else None

    if len(rb) >= 22:
        delta = current - float(rb.iloc[-22])
        trend = (
            "increasing"
            if delta > 0.1
            else ("decreasing" if delta < -0.1 else "stable")
        )
    else:
        trend = "stable"

    logger.debug(
        "[rolling_beta] current=%.4f  trend=%s  min=%.4f  max=%.4f  n=%d",
        current,
        trend,
        float(rb.min()),
        float(rb.max()),
        len(rb),
    )
    return RollingBetaResult(
        symbol=input_data.symbol,
        benchmark=input_data.benchmark,
        window=input_data.window,
        current_beta=round(current, 4),
        beta_1m_ago=round(b_1m, 4) if b_1m is not None else None,
        beta_3m_ago=round(b_3m, 4) if b_3m is not None else None,
        beta_6m_ago=round(b_6m, 4) if b_6m is not None else None,
        beta_trend=trend,
        beta_min=round(float(rb.min()), 4),
        beta_max=round(float(rb.max()), 4),
        beta_mean=round(float(rb.mean()), 4),
        n_obs=len(rb),
    )


# ──────────────────────────────────────────────────────────────────
# Extended Risk Metrics Tool
# ──────────────────────────────────────────────────────────────────


def get_extended_risk_metrics(input_data: ExtendedRiskInput) -> ExtendedRiskResult:
    """
    Extended risk metrics not in analyze_stock_risk: Calmar ratio, Treynor ratio,
    parametric VaR at 95%/99%, historical VaR at 99%, CVaR at 99%, and CAGR.
    Pair with analyze_stock_risk for a complete risk picture.
    """
    logger.debug(
        "[extended_risk] %s vs %s  %s → %s",
        input_data.symbol,
        input_data.benchmark,
        input_data.start_date,
        input_data.end_date,
    )
    provider = DataFactory.get_provider()
    asset_df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    bench_df = provider.get_ohlcv(
        input_data.benchmark, input_data.start_date, input_data.end_date
    )

    asset_ret = asset_df["Close"].pct_change().dropna()
    bench_ret = bench_df["Close"].pct_change().dropna()
    equity_curve = (1 + asset_ret).cumprod()

    ann_ret = cagr(equity_curve)
    cal = calmar_ratio(equity_curve)
    beta_val = calculate_beta(asset_ret, bench_ret)["beta"]
    treynor = treynor_ratio(asset_ret, bench_ret)
    vp95 = var_parametric(asset_ret, confidence=0.95)
    vp99 = var_parametric(asset_ret, confidence=0.99)
    vh99 = var_historical(asset_ret, 0.99)
    cv99 = cvar(asset_ret, 0.99)

    logger.debug(
        "[extended_risk] calmar=%.4f  treynor=%.4f  VaR_p95=%.4f  VaR_p99=%.4f",
        cal,
        treynor,
        vp95,
        vp99,
    )
    return ExtendedRiskResult(
        symbol=input_data.symbol,
        benchmark=input_data.benchmark,
        annualized_return=round(ann_ret, 6),
        calmar_ratio=round(cal, 4),
        treynor_ratio=round(treynor, 6),
        var_parametric_95=round(vp95, 6),
        var_parametric_99=round(vp99, 6),
        var_historical_99=round(vh99, 6),
        cvar_99=round(cv99, 6),
        beta=round(beta_val, 4),
    )


# ──────────────────────────────────────────────────────────────────
# EVT Tail Risk Tool
# ──────────────────────────────────────────────────────────────────


def get_tail_risk_metrics(input_data: TailRiskInput) -> TailRiskResult:
    """
    Extreme Value Theory tail risk via Peaks-Over-Threshold: fits a
    Generalized Pareto Distribution to the worst tail of daily losses and
    extrapolates VaR/CVaR from that fitted tail, rather than the raw
    empirical quantile get_extended_risk_metrics' var_historical_99 uses —
    var_historical_comparison reports that side by side for contrast.
    """
    logger.debug(
        "[tail_risk] %s  %s → %s  confidence=%.3f  tail_fraction=%.3f  method=%s",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.confidence,
        input_data.tail_fraction,
        input_data.method,
    )
    provider = DataFactory.get_provider()
    close = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )["Close"]
    returns = close.pct_change().dropna()

    result = evt_tail_risk(
        returns,
        confidence=input_data.confidence,
        tail_fraction=input_data.tail_fraction,
        method=input_data.method,
    )
    hist_comparison = var_historical(returns, confidence=input_data.confidence)

    return TailRiskResult(
        symbol=input_data.symbol,
        confidence=result["confidence"],
        threshold_daily_loss_pct=round(result["threshold"], 6),
        n_exceedances=result["n_exceedances"],
        n_obs=result["n_obs"],
        shape_xi=round(result["shape_xi"], 4),
        scale_beta=round(result["scale_beta"], 6),
        var_evt=round(result["var_evt"], 6),
        cvar_evt=(
            round(result["cvar_evt"], 6)
            if result["cvar_evt"] != float("inf")
            else float("inf")
        ),
        var_historical_comparison=round(hist_comparison, 6),
        method=result["method"],
        tail_classification=result["tail_classification"],
    )


# ──────────────────────────────────────────────────────────────────
# Custom Signal Backtest (bring-your-own signal)
# ──────────────────────────────────────────────────────────────────


def run_custom_signal_backtest(input_data: CustomSignalBacktestInput) -> BacktestResult:
    """
    Backtest a signal computed entirely outside this library (e.g. your own
    alpha model) on a single symbol. Unlike run_sma_backtest / run_rsi_backtest /
    run_macd_backtest / run_bollinger_backtest, this tool does not generate the
    signal — you supply it, and it reuses the same fast backtest engine.
    """
    logger.debug(
        "[custom_signal_backtest] %s  %s → %s  n_signal_points=%d",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        len(input_data.signals),
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )

    signal_series = pd.Series(
        {pd.Timestamp(d): v for d, v in input_data.signals.items()}
    ).sort_index()
    signal_series = _apply_signal_fill_policy(
        signal_series,
        df.index,
        input_data.signal_fill_policy,
    )

    bt_input = BacktestInput(
        symbol=input_data.symbol,
        start_date=input_data.start_date,
        end_date=input_data.end_date,
        strategy_type="custom_signal",
        parameters={},
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        fill_price=input_data.fill_price,
    )
    return _run_backtest(bt_input, df, signal_series)


# ──────────────────────────────────────────────────────────────────
# Signal Panel Backtest (bring-your-own multi-ticker signal matrix)
# ──────────────────────────────────────────────────────────────────


def run_signal_panel_backtest(
    input_data: SignalPanelBacktestInput,
) -> SignalPanelBacktestResult:
    """
    Backtest a pre-computed signal panel (e.g. your own cross-sectional alpha
    model) across a ticker universe, combined into portfolio-level metrics.
    Fetches OHLCV internally and reuses run_strategy per ticker plus the
    existing portfolio module for the combination — no new backtest math.
    """
    logger.debug(
        "[signal_panel_backtest] tickers=%d  %s → %s",
        len(input_data.tickers),
        input_data.start_date,
        input_data.end_date,
    )
    # Concurrent fetch (bounded by the default executor's thread pool, ~32
    # in flight) rather than one blocking get_ohlcv() call per ticker in a
    # loop — for a large universe (e.g. the full S&P 500), the sequential
    # form used to mean minutes of pure network wait before anything else
    # even started. Needs the full OHLCV panel (Volume/High/Low, not just
    # Close-derived returns), hence fetch_ohlcv_panel_sync rather than the
    # cheaper fetch_returns_sync other multi-ticker tools in this module use.
    price_data = fetch_ohlcv_panel_sync(
        input_data.tickers, input_data.start_date, input_data.end_date
    )

    signal_panel = pd.DataFrame(
        {
            t: _apply_signal_fill_policy(
                pd.Series(
                    {pd.Timestamp(d): v for d, v in input_data.signal_panel[t].items()}
                ).sort_index(),
                price_data[t].index,
                input_data.signal_fill_policy,
            )
            for t in input_data.tickers
        }
    ).sort_index()

    bench_returns = None
    if input_data.benchmark:
        bench_df = DataFactory.get_provider().get_ohlcv(
            input_data.benchmark, input_data.start_date, input_data.end_date
        )
        bench_returns = bench_df["Close"].pct_change().dropna()

    weights_arg: Any = input_data.weights if input_data.weights is not None else None

    raw = _signal_panel_backtest(
        price_data,
        signal_panel,
        weights=weights_arg,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        benchmark_returns=bench_returns,
        include_trade_log=input_data.include_trade_log,
        fill_price=input_data.fill_price,
    )

    per_ticker: Dict[str, BacktestResult] = {}
    for ticker, res in raw["per_ticker"].items():
        trade_log_raw = res.get("trade_log", pd.DataFrame())
        trades = None
        if isinstance(trade_log_raw, pd.DataFrame) and not trade_log_raw.empty:
            trades = [
                Trade(
                    entry_date=str(r["entry_date"]),
                    exit_date=str(r["exit_date"]),
                    direction=str(r["direction"]),
                    entry_price=float(r["entry_price"]),
                    exit_price=float(r["exit_price"]),
                    position_size=float(r.get("position_size", 1.0)),
                    return_pct=float(r["return_pct"]),
                )
                for r in trade_log_raw.to_dict(orient="records")
            ]
        per_ticker[ticker] = BacktestResult(
            total_return=res["total_return"],
            annualized_volatility=res["annualized_volatility"],
            sharpe_ratio=res["sharpe_ratio"],
            sortino_ratio=res["sortino_ratio"],
            max_drawdown=res["max_drawdown"],
            calmar_ratio=res["calmar_ratio"],
            win_rate=res["win_rate"],
            profit_factor=res["profit_factor"],
            num_trades=res["num_trades"],
            avg_trade_return_pct=res["avg_trade_return_pct"],
            final_equity=res["final_equity"],
            equity_curve=res["equity_curve"].tolist(),
            trade_log=trades,
        )

    portfolio_metrics_out = dict(raw["portfolio_metrics"])
    logger.debug(
        "[signal_panel_backtest] portfolio  sharpe=%.3f  return=%.2f%%",
        portfolio_metrics_out.get("sharpe_ratio", float("nan")),
        portfolio_metrics_out.get("annualized_return", 0.0) * 100,
    )

    return SignalPanelBacktestResult(
        tickers=input_data.tickers,
        per_ticker=per_ticker,
        portfolio_metrics=portfolio_metrics_out,
    )


# ──────────────────────────────────────────────────────────────────
# True Portfolio Simulation (shared cash, rebalancing)
# ──────────────────────────────────────────────────────────────────


def _metrics_with_day0_cost(
    equity_curve: pd.Series,
    initial_capital: float,
) -> Tuple[pd.Series, float, float, pd.Series]:
    """
    Returns (returns, total_return, annualized_return, equity_with_start)
    accounting for a same-bar-0 rebalance cost that would otherwise be
    invisible: equity_curve.iloc[0] from run_portfolio_simulation/
    run_pair_backtest is already net of that rebalance's costs, so naively
    using cumulative_return/cagr (baseline = equity_curve.iloc[0]) and
    equity_curve.pct_change() (first return = NaN -> filled 0.0) silently
    drops the day-0 cost's entire effect on every downstream metric.
    Prepending a synthetic pre-trade observation at initial_capital makes
    the first real return capture it instead.

    equity_with_start is also returned (not just consumed internally) so
    callers can compute max_drawdown/calmar_ratio from the same complete
    curve: max_drawdown(equity_curve) alone would treat the already-
    post-cost equity_curve.iloc[0] as the peak, so a portfolio that lost
    5% entirely to day-0 rebalance costs and then stayed flat would
    otherwise report max_drawdown=0.0 (and calmar_ratio=inf) despite
    total_return correctly reporting -5%. calmar_ratio should still be
    computed as annualized_return (above, already using the correct
    initial_capital baseline and undistorted bar count) divided by
    max_drawdown(equity_with_start) -- not via the standalone calmar_ratio()
    metrics function, which would redundantly recompute cagr from
    equity_with_start's own iloc[0] baseline using len(equity_with_start)
    (one bar too many, from the synthetic point) as its bar count.
    """
    if equity_curve.empty:
        return equity_curve, 0.0, 0.0, equity_curve
    synthetic_index = equity_curve.index[0] - pd.Timedelta(days=1)
    equity_with_start = pd.concat(
        [
            pd.Series([initial_capital], index=[synthetic_index]),
            equity_curve,
        ]
    )
    returns = equity_with_start.pct_change().dropna()
    total_return = float(equity_curve.iloc[-1]) / initial_capital - 1.0
    num_years = len(equity_curve) / 252
    annualized_return = (
        (1.0 + total_return) ** (1.0 / num_years) - 1.0 if num_years > 0 else 0.0
    )
    return returns, total_return, annualized_return, equity_with_start


def run_portfolio_simulation(
    input_data: PortfolioSimulationInput,
) -> PortfolioSimulationResult:
    """
    True shared-cash portfolio simulation: one account, position sizing
    relative to current equity, and rebalancing at the dates in
    target_weights — unlike run_signal_panel_backtest, which gives every
    ticker its own independent capital and only blends per-ticker return
    streams afterward. Reuses backtest/portfolio_engine.py for the
    simulation and the existing metrics functions for the summary; no new
    metric math.

    When signal_type='score', target_weights holds arbitrary per-ticker
    alpha scores instead of weights — converted via construction_method
    (backtest/sizing.py: rank_weighted, equal_weight_top_bottom,
    zscore_normalized, vol_scaled) before simulation.
    """
    logger.debug(
        "[portfolio_simulation] tickers=%d  %s → %s  max_gross_leverage=%.2f",
        len(input_data.tickers),
        input_data.start_date,
        input_data.end_date,
        input_data.max_gross_leverage,
    )
    # Concurrent fetch (bounded by the default executor's thread pool, ~32
    # in flight) rather than one blocking get_ohlcv() call per ticker in a
    # loop — for a large universe (e.g. the full S&P 500), the sequential
    # form used to mean minutes of pure network wait before anything else
    # even started. Needs the full OHLCV panel (Volume/High/Low, not just
    # Close-derived returns), hence fetch_ohlcv_panel_sync rather than the
    # cheaper fetch_returns_sync other multi-ticker tools in this module use.
    price_data = fetch_ohlcv_panel_sync(
        input_data.tickers, input_data.start_date, input_data.end_date
    )

    values_panel = pd.DataFrame(
        {
            t: pd.Series(
                {pd.Timestamp(d): v for d, v in input_data.target_weights[t].items()}
            )
            for t in input_data.tickers
        }
    ).sort_index()

    if input_data.signal_type == SignalType.SCORE:
        method = input_data.construction_method
        if method == "rank_weighted":
            target_weights = rank_weighted(
                values_panel, gross_leverage=input_data.gross_leverage
            )
        elif method == "equal_weight_top_bottom":
            # n_long/n_short non-None is enforced by PortfolioSimulationInput's
            # model validator when construction_method == "equal_weight_top_bottom".
            assert input_data.n_long is not None and input_data.n_short is not None
            target_weights = equal_weight_top_bottom(
                values_panel,
                n_long=input_data.n_long,
                n_short=input_data.n_short,
                gross_leverage=input_data.gross_leverage,
            )
        elif method == "zscore_normalized":
            target_weights = zscore_normalized(
                values_panel, gross_leverage=input_data.gross_leverage
            )
        else:  # vol_scaled — validated to be one of _CONSTRUCTION_METHODS by the input model
            returns_df = pd.DataFrame(
                {t: price_data[t]["Close"].pct_change() for t in input_data.tickers}
            )
            target_weights = vol_scaled(
                values_panel,
                returns_df,
                lookback=input_data.vol_lookback,
                gross_leverage=input_data.gross_leverage,
            )
        if input_data.make_dollar_neutral:
            target_weights = dollar_neutral(target_weights)
        logger.debug(
            "[portfolio_simulation] converted SCORE signals via construction_method=%s  gross_leverage=%.2f",
            method,
            input_data.gross_leverage,
        )
    else:
        target_weights = values_panel

    raw = _portfolio_engine_run(
        price_data,
        target_weights,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        max_gross_leverage=input_data.max_gross_leverage,
        max_position_pct=input_data.max_position_pct,
        fill_price=input_data.fill_price,
        commission_model=input_data.commission_model,
        per_share_rate=input_data.per_share_rate,
        min_commission=input_data.min_commission,
        use_impact_model=input_data.use_impact_model,
        impact_coefficient=input_data.impact_coefficient,
        impact_lookback=input_data.impact_lookback,
        borrow_fee_bps=input_data.borrow_fee_bps,
        margin_interest_rate=input_data.margin_interest_rate,
        max_adv_participation=input_data.max_adv_participation,
    )

    equity_curve = raw["equity_curve"]
    returns, total_return, annualized_return, equity_with_start = (
        _metrics_with_day0_cost(
            equity_curve,
            input_data.initial_capital,
        )
    )
    day0_max_dd = float(max_drawdown(equity_with_start))
    day0_calmar = (
        annualized_return / abs(day0_max_dd) if day0_max_dd != 0.0 else float("inf")
    )

    ir: Optional[float] = None
    if input_data.benchmark:
        bench_df = DataFactory.get_provider().get_ohlcv(
            input_data.benchmark, input_data.start_date, input_data.end_date
        )
        bench_returns = bench_df["Close"].pct_change().dropna()
        common = returns.index.intersection(bench_returns.index)
        if len(common) > 1:
            ir = round(
                float(
                    information_ratio(returns.loc[common], bench_returns.loc[common])
                ),
                4,
            )

    rebalance_events = [
        RebalanceEvent(
            date=str(r["date"]),
            turnover_pct=float(r["turnover_pct"]),
            gross_leverage_after=float(r["gross_leverage_after"]),
            n_positions=int(r["n_positions"]),
        )
        for r in raw["rebalance_log"].to_dict(orient="records")
    ]

    leverage_series = raw["leverage_curve"]
    avg_gross_leverage = (
        round(float(leverage_series.mean()), 4) if not leverage_series.empty else 0.0
    )
    max_gross_leverage_used = (
        round(float(leverage_series.max()), 4) if not leverage_series.empty else 0.0
    )

    logger.debug(
        "[portfolio_simulation] rebalances=%d  final_equity=%.2f  sharpe=%.3f  warnings=%s",
        len(rebalance_events),
        raw["final_equity"],
        float(sharpe_ratio(returns)),
        raw["warnings"],
    )

    return PortfolioSimulationResult(
        tickers=input_data.tickers,
        n_rebalances=len(rebalance_events),
        rebalance_log=rebalance_events,
        total_return=round(total_return, 6),
        annualized_return=round(annualized_return, 6),
        annualized_volatility=round(float(annualized_volatility(returns)), 6),
        sharpe_ratio=round(float(sharpe_ratio(returns)), 4),
        sortino_ratio=round(float(sortino_ratio(returns)), 4),
        max_drawdown=round(day0_max_dd, 6),
        calmar_ratio=round(day0_calmar, 4),
        var_95=round(float(var_historical(returns, 0.95)), 6),
        cvar_95=round(float(cvar(returns, 0.95)), 6),
        information_ratio=ir,
        final_equity=round(float(raw["final_equity"]), 2),
        final_cash=round(float(raw["final_cash"]), 2),
        avg_gross_leverage=avg_gross_leverage,
        max_gross_leverage_used=max_gross_leverage_used,
        equity_curve=equity_curve.tolist(),
        warnings=list(raw["warnings"]),
    )


# ──────────────────────────────────────────────────────────────────
# Pair Trade Backtest (synchronized two-leg execution)
# ──────────────────────────────────────────────────────────────────


def run_pair_trade_backtest(
    input_data: PairTradeBacktestInput,
) -> PairTradeBacktestResult:
    """
    Backtest a cointegrated pair as one synchronized two-leg trade — unlike
    scan_pairs, which only screens for cointegration candidates and reports
    a current z-score signal. Reuses backtest/pairs.py, which itself reuses
    run_portfolio_simulation: both legs are columns of the same
    target_weights row, so they enter/exit together by construction and
    share one cash account.
    """
    logger.debug(
        "[pair_trade_backtest] %s/%s  hedge_ratio=%.4f  entry_z=%.2f  exit_z=%.2f",
        input_data.symbol_a,
        input_data.symbol_b,
        input_data.hedge_ratio,
        input_data.entry_z,
        input_data.exit_z,
    )
    provider = DataFactory.get_provider()
    price_data = {
        input_data.symbol_a: provider.get_ohlcv(
            input_data.symbol_a, input_data.start_date, input_data.end_date
        ),
        input_data.symbol_b: provider.get_ohlcv(
            input_data.symbol_b, input_data.start_date, input_data.end_date
        ),
    }

    raw = _pair_backtest_run(
        price_data,
        symbol_a=input_data.symbol_a,
        symbol_b=input_data.symbol_b,
        hedge_ratio=input_data.hedge_ratio,
        entry_z=input_data.entry_z,
        exit_z=input_data.exit_z,
        zscore_window=input_data.zscore_window,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        gross_leverage=input_data.gross_leverage,
        fill_price=input_data.fill_price,
    )

    equity_curve = raw["equity_curve"]
    returns, total_return, annualized_return, equity_with_start = (
        _metrics_with_day0_cost(
            equity_curve,
            input_data.initial_capital,
        )
    )
    day0_max_dd = float(max_drawdown(equity_with_start))
    day0_calmar = (
        annualized_return / abs(day0_max_dd) if day0_max_dd != 0.0 else float("inf")
    )
    rebalance_events = [
        RebalanceEvent(
            date=str(r["date"]),
            turnover_pct=float(r["turnover_pct"]),
            gross_leverage_after=float(r["gross_leverage_after"]),
            n_positions=int(r["n_positions"]),
        )
        for r in raw["rebalance_log"].to_dict(orient="records")
    ]

    logger.debug(
        "[pair_trade_backtest] rebalances=%d  round_trips=%d  final_equity=%.2f  sharpe=%.3f",
        len(rebalance_events),
        raw["n_round_trips"],
        raw["final_equity"],
        float(sharpe_ratio(returns)),
    )

    return PairTradeBacktestResult(
        symbol_a=input_data.symbol_a,
        symbol_b=input_data.symbol_b,
        hedge_ratio=raw["hedge_ratio"],
        n_rebalances=len(rebalance_events),
        n_round_trips=raw["n_round_trips"],
        rebalance_log=rebalance_events,
        entry_spread=raw["entry_spread"],
        current_spread=round(float(raw["current_spread"]), 6),
        total_return=round(total_return, 6),
        annualized_return=round(annualized_return, 6),
        annualized_volatility=round(float(annualized_volatility(returns)), 6),
        sharpe_ratio=round(float(sharpe_ratio(returns)), 4),
        sortino_ratio=round(float(sortino_ratio(returns)), 4),
        max_drawdown=round(day0_max_dd, 6),
        calmar_ratio=round(day0_calmar, 4),
        final_equity=round(float(raw["final_equity"]), 2),
        final_cash=round(float(raw["final_cash"]), 2),
        equity_curve=equity_curve.tolist(),
        warnings=list(raw["warnings"]),
    )


# ──────────────────────────────────────────────────────────────────
# Robustness Diagnostics (parameter sensitivity, Deflated Sharpe Ratio,
# block-bootstrap CI)
# ──────────────────────────────────────────────────────────────────


def get_robustness_diagnostics(
    input_data: RobustnessDiagnosticsInput,
) -> RobustnessDiagnosticsResult:
    """
    Same-sample robustness checks for a grid search: how much better is the
    best trial than the pack (parameter_sensitivity), does the best trial's
    Sharpe survive correcting for having been selected as the max of
    n_trials attempts (deflated_sharpe_ratio), and a block-bootstrap
    confidence interval on the best trial's Sharpe ratio. This is NOT a
    substitute for run_walk_forward_backtest's out-of-sample validation —
    it quantifies confidence in a same-sample estimate, a different and
    complementary question ("how sure am I this number is real" vs. "would
    it have held up on unseen data").
    """
    logger.debug(
        "[robustness_diagnostics] %s  strategy=%s  sort_by=%s  n_combos=%s",
        input_data.symbol,
        input_data.strategy,
        input_data.sort_by,
        {k: len(v) for k, v in input_data.param_grid.items()},
    )
    if input_data.strategy not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{input_data.strategy}'. "
            f"Available: {list(STRATEGY_REGISTRY)}"
        )

    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )

    grid_df = backtest_grid(
        df,
        strategy=input_data.strategy,
        param_grid=input_data.param_grid,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        sort_by=input_data.sort_by,
        ascending=False,
    )
    sensitivity = _parameter_sensitivity(grid_df, metric_col=input_data.sort_by)

    best_row = grid_df.iloc[0]
    param_keys = list(input_data.param_grid.keys())
    best_params: Dict[str, Any] = {
        k: (
            int(best_row[k])
            if isinstance(input_data.param_grid[k][0], int)
            else float(best_row[k])
        )
        for k in param_keys
    }
    best_sharpe = float(best_row["sharpe_ratio"])
    sharpe_trials_std = (
        float(grid_df["sharpe_ratio"].std()) if len(grid_df) > 1 else 0.0
    )

    # grid_df's sharpe_ratio column is annualized (run_strategy -> sharpe_ratio's
    # default periods_per_year=252), but deflated_sharpe_ratio's formula requires
    # the non-annualized, per-period Sharpe (its z-score already scales by
    # sqrt(n_obs - 1) itself). De-annualize before the DSR call, then re-annualize
    # expected_max_sharpe for reporting so it's on the same scale as best_sharpe.
    _ANNUALIZATION = np.sqrt(252.0)
    dsr = _deflated_sharpe_ratio(
        observed_sharpe=best_sharpe / _ANNUALIZATION,
        sharpe_trials_std=sharpe_trials_std / _ANNUALIZATION,
        n_trials=len(grid_df),
        n_obs=len(df),
        skew=input_data.skew,
        kurtosis=input_data.kurtosis,
    )
    dsr["expected_max_sharpe"] = round(dsr["expected_max_sharpe"] * _ANNUALIZATION, 6)

    best_signals = STRATEGY_REGISTRY[input_data.strategy](df, **best_params)
    best_result = run_strategy(
        df,
        best_signals,
        initial_capital=input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
    )
    best_returns = best_result["equity_curve"].pct_change().fillna(0.0)

    bootstrap = _block_bootstrap_ci(
        best_returns,
        sharpe_ratio,
        n_iterations=input_data.n_bootstrap_iterations,
        block_size=input_data.bootstrap_block_size,
        confidence=input_data.bootstrap_confidence,
        seed=input_data.random_seed,
    )

    warnings: List[str] = []
    if len(grid_df) < 5:
        warnings.append(
            f"Only {len(grid_df)} trial(s) searched — parameter sensitivity and "
            "deflated Sharpe estimates are noisy with this few combinations."
        )

    logger.debug(
        "[robustness_diagnostics] trials=%d  best_sharpe=%.3f  expected_max=%.3f  dsr=%.3f",
        len(grid_df),
        best_sharpe,
        dsr["expected_max_sharpe"],
        dsr["deflated_sharpe_ratio"],
    )

    return RobustnessDiagnosticsResult(
        symbol=input_data.symbol,
        strategy=input_data.strategy,
        best_params=best_params,
        parameter_sensitivity=sensitivity,
        expected_max_sharpe=dsr["expected_max_sharpe"],
        deflated_sharpe_ratio=dsr["deflated_sharpe_ratio"],
        bootstrap_point_estimate=round(bootstrap["point_estimate"], 4),
        bootstrap_ci_lower=round(bootstrap["ci_lower"], 4),
        bootstrap_ci_upper=round(bootstrap["ci_upper"], 4),
        bootstrap_confidence=input_data.bootstrap_confidence,
        warnings=warnings,
    )


def run_monte_carlo_simulation(
    input_data: MonteCarloSimulationInput,
) -> MonteCarloSimulationResult:
    """
    Monte Carlo forward simulation: projects possible future equity paths
    from a portfolio's historical return distribution via moving-block
    bootstrap resampling. Forward-looking, unlike get_robustness_diagnostics
    (a same-sample confidence check) or run_walk_forward_backtest (which
    tests actual historical decisions) — this is a projection from
    historical statistics, not a prediction or a validation of any
    strategy's decisions.
    """
    logger.debug(
        "[monte_carlo] tickers=%s  %s → %s  horizon=%d  n_sim=%d",
        input_data.tickers,
        input_data.start_date,
        input_data.end_date,
        input_data.horizon_days,
        input_data.n_simulations,
    )
    returns_df = fetch_returns_sync(
        input_data.tickers, input_data.start_date, input_data.end_date
    )
    n_assets = returns_df.shape[1]
    weights = (
        np.full(n_assets, 1.0 / n_assets)
        if input_data.weights is None
        else input_data.weights
    )
    portfolio_returns = build_portfolio(returns_df, weights)

    result = simulate_forward_paths(
        portfolio_returns,
        horizon_days=input_data.horizon_days,
        n_simulations=input_data.n_simulations,
        block_size=input_data.block_size,
        initial_capital=input_data.initial_capital,
        seed=input_data.random_seed,
    )

    return MonteCarloSimulationResult(
        tickers=input_data.tickers,
        horizon_days=input_data.horizon_days,
        n_simulations=input_data.n_simulations,
        terminal_median=round(result["terminal_median"], 2),
        terminal_p5=round(result["terminal_p5"], 2),
        terminal_p95=round(result["terminal_p95"], 2),
        prob_loss=round(result["prob_loss"], 4),
        terminal_var_95=round(result["terminal_var_95"], 6),
        terminal_cvar_95=round(result["terminal_cvar_95"], 6),
        equity_band_p5=[round(v, 2) for v in result["equity_band_p5"]],
        equity_band_p50=[round(v, 2) for v in result["equity_band_p50"]],
        equity_band_p95=[round(v, 2) for v in result["equity_band_p95"]],
    )


# ──────────────────────────────────────────────────────────────────
# Capacity Report (liquidity/ADV-based capacity)
# ──────────────────────────────────────────────────────────────────


def get_capacity_report(input_data: CapacityReportInput) -> CapacityReportResult:
    """
    How much account size a target-weight portfolio can support before
    positions become too large relative to each ticker's own trading
    volume. Reuses backtest/constraints.py's capacity_report/
    days_to_liquidate — no new math, just data fetching and formatting for
    JSON tool-calling.
    """
    logger.debug(
        "[capacity_report] tickers=%d  max_participation=%.3f",
        len(input_data.tickers),
        input_data.max_participation,
    )
    provider = DataFactory.get_provider()

    avg_dollar_volumes: Dict[str, float] = {}
    avg_share_volumes: Dict[str, float] = {}
    last_close: Dict[str, float] = {}
    for t in input_data.tickers:
        df = provider.get_ohlcv(t, input_data.start_date, input_data.end_date)
        if "Volume" not in df.columns:
            raise ValueError(f"OHLCV for {t!r} is missing a 'Volume' column")
        dollar_vol = (
            (df["Close"] * df["Volume"])
            .rolling(input_data.adv_lookback, min_periods=1)
            .mean()
        )
        share_vol = df["Volume"].rolling(input_data.adv_lookback, min_periods=1).mean()
        avg_dollar_volumes[t] = float(dollar_vol.iloc[-1])
        avg_share_volumes[t] = float(share_vol.iloc[-1])
        last_close[t] = float(df["Close"].iloc[-1])

    raw = _capacity_report(
        input_data.tickers,
        avg_dollar_volumes,
        input_data.target_weights,
        input_data.max_participation,
    )

    per_ticker_out: Dict[str, Optional[float]] = {
        t: (None if v == float("inf") else round(v, 2))
        for t, v in raw["per_ticker"].items()
    }
    max_account_size = raw["max_account_size"]
    max_account_size_out: Optional[float] = (
        None if max_account_size == float("inf") else round(max_account_size, 2)
    )

    days_to_liquidate_out: Dict[str, float] = {}
    for t in input_data.tickers:
        weight = abs(input_data.target_weights[t])
        if weight <= 0 or max_account_size_out is None or avg_share_volumes[t] <= 0:
            days_to_liquidate_out[t] = 0.0
            continue
        notional_at_capacity = max_account_size_out * weight
        shares_at_capacity = (
            notional_at_capacity / last_close[t] if last_close[t] > 0 else 0.0
        )
        days_to_liquidate_out[t] = round(
            _days_to_liquidate(
                shares_at_capacity, avg_share_volumes[t], input_data.max_participation
            ),
            2,
        )

    warnings: List[str] = []
    sector_exp: Optional[Dict[str, float]] = None
    if input_data.include_sector_exposure:
        sectors: Dict[str, str] = {}
        for t in input_data.tickers:
            try:
                info = provider.get_ticker_info(t)
                sectors[t] = info.sector
            except Exception as exc:
                sectors[t] = "Unknown"
                warnings.append(f"could not fetch sector for {t!r}: {exc}")
        sector_exp = _sector_exposure(input_data.target_weights, sectors)

    logger.debug(
        "[capacity_report] binding=%s  max_account_size=%s",
        raw["binding_ticker"],
        max_account_size_out,
    )

    return CapacityReportResult(
        tickers=input_data.tickers,
        per_ticker_max_account_size=per_ticker_out,
        binding_ticker=raw["binding_ticker"],
        max_account_size=max_account_size_out,
        days_to_liquidate_at_capacity=days_to_liquidate_out,
        sector_exposure=sector_exp,
        warnings=warnings,
    )


def get_liquidity_metrics(
    input_data: LiquidityAnalysisInput,
) -> LiquidityAnalysisResult:
    """
    Amihud illiquidity ratio and Corwin-Schultz spread estimator per
    ticker — academic proxies for market depth and bid/ask spread derived
    purely from OHLCV, since no real bid/ask data exists in this library.
    """
    logger.debug(
        "[liquidity_metrics] tickers=%s  window=%d",
        input_data.tickers,
        input_data.window,
    )
    provider = DataFactory.get_provider()
    window = input_data.window

    per_ticker: Dict[str, Dict[str, float]] = {}
    for t in input_data.tickers:
        df = provider.get_ohlcv(t, input_data.start_date, input_data.end_date)
        if "Volume" not in df.columns:
            raise ValueError(f"OHLCV for {t!r} is missing a 'Volume' column")

        returns = df["Close"].pct_change()
        dollar_volume = df["Close"] * df["Volume"]
        avg_dollar_volume = dollar_volume.rolling(window, min_periods=1).mean()

        amihud = amihud_illiquidity(returns, dollar_volume, window=window)
        cs_spread = corwin_schultz_spread(df["High"], df["Low"], window=window)

        amihud_valid = amihud.dropna()
        cs_valid = cs_spread.dropna()

        per_ticker[t] = {
            "avg_dollar_volume": round(float(avg_dollar_volume.iloc[-1]), 2),
            "amihud_illiquidity": (
                round(float(amihud_valid.iloc[-1]), 6)
                if not amihud_valid.empty
                else 0.0
            ),
            "corwin_schultz_spread_bps": (
                round(float(cs_valid.iloc[-1]) * 10_000.0, 4)
                if not cs_valid.empty
                else 0.0
            ),
        }

    least_liquid = max(per_ticker, key=lambda t: per_ticker[t]["amihud_illiquidity"])
    most_liquid = min(per_ticker, key=lambda t: per_ticker[t]["amihud_illiquidity"])

    return LiquidityAnalysisResult(
        tickers=input_data.tickers,
        per_ticker=per_ticker,
        least_liquid_ticker=least_liquid,
        most_liquid_ticker=most_liquid,
    )


# ──────────────────────────────────────────────────────────────────
# Data Quality Report (dataset provenance + missing-bar/stale-price/
# price-jump detection)
# ──────────────────────────────────────────────────────────────────


def get_data_quality_report(
    input_data: DataQualityReportInput,
) -> DataQualityReportResult:
    """
    Dataset provenance (data/metadata.py — what this provider does and
    doesn't guarantee) plus missing-bar/stale-price/price-jump detection
    (data/quality.py) on the fetched OHLCV. All checks are heuristics on
    data already fetched, not a new data source — see each function's
    docstring for known false-positive modes. In particular, missing_bars
    has no market-holiday calendar: every U.S. market holiday in the
    requested range will be reported as a gap, not just genuine missing
    data — treat entries as leads to investigate, not confirmed defects.
    """
    logger.debug(
        "[data_quality_report] %s  %s → %s",
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
    )
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    metadata = provider.get_metadata(input_data.symbol)

    missing = [MissingBar(**m) for m in detect_missing_bars(df)]
    stale = [
        StalePriceRun(**s)
        for s in detect_stale_prices(df, n=input_data.stale_run_length)
    ]
    jumps = [
        PriceJump(**j)
        for j in detect_price_jumps(df, threshold=input_data.jump_threshold)
    ]

    logger.debug(
        "[data_quality_report] missing_bars=%d  stale_runs=%d  price_jumps=%d",
        len(missing),
        len(stale),
        len(jumps),
    )

    return DataQualityReportResult(
        symbol=input_data.symbol,
        metadata=metadata.model_dump(),
        missing_bars=missing,
        stale_price_runs=stale,
        price_jumps=jumps,
    )


# ──────────────────────────────────────────────────────────────────
# Compact Backtest Result (BacktestResultV2)
# ──────────────────────────────────────────────────────────────────


def run_backtest_compact(input_data: BacktestCompactInput) -> BacktestResultV2:
    """
    Compact counterpart to run_sma_backtest/run_rsi_backtest/run_macd_backtest/
    run_bollinger_backtest: instead of embedding the full equity_curve/
    trade_log inline (BacktestResult), saves them via backtest/artifacts.py
    and returns summary/risk/exposure/cost sub-reports plus URIs — closes
    the "agent tool result can contain the complete equity curve" gap for
    callers who opt into this shape. Reuses run_strategy and
    metrics/diagnostics.py's exposure_stats; no new backtest or metric math.
    """
    if input_data.strategy_type not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{input_data.strategy_type}'. "
            f"Available: {list(STRATEGY_REGISTRY)}"
        )
    run_id = input_data.run_id or uuid.uuid4().hex
    logger.debug(
        "[backtest_compact] run_id=%s  %s  %s",
        run_id,
        input_data.symbol,
        input_data.strategy_type,
    )

    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY[input_data.strategy_type](df, **input_data.parameters)

    results = run_strategy(
        df,
        signals,
        input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        include_trade_log=True,
        fill_price=input_data.fill_price,
    )
    equity_curve = results["equity_curve"]
    returns = equity_curve.pct_change().fillna(0.0)
    trade_log = results.get("trade_log", pd.DataFrame())
    executed = signals.shift(1).fillna(0.0)

    exposure = exposure_stats(executed, trade_log)

    pos_diff = executed.diff().fillna(executed.iloc[0])
    turnover = float(pos_diff.abs().sum())
    total_commission_pct = turnover * input_data.commission_pct
    total_slippage_pct = turnover * input_data.slippage_pct

    equity_curve_uri = save_artifact(equity_curve, run_id, "equity_curve")
    trades_uri = (
        save_artifact(trade_log, run_id, "trades") if not trade_log.empty else None
    )

    warnings: List[str] = []
    validation_status = "ok"
    if int(results["num_trades"]) < 5:
        warnings.append(
            f"Only {results['num_trades']} trade(s) — too few to draw reliable conclusions."
        )
        validation_status = "warning"

    logger.debug(
        "[backtest_compact] run_id=%s  trades=%d  sharpe=%.3f  status=%s",
        run_id,
        results["num_trades"],
        results["sharpe_ratio"],
        validation_status,
    )

    return BacktestResultV2(
        run_id=run_id,
        strategy_name=input_data.strategy_type,
        summary=PerformanceSummary(
            total_return=round(float(results["total_return"]), 6),
            annualized_return=round(float(cagr(equity_curve)), 6),
            annualized_volatility=round(float(results["annualized_volatility"]), 6),
            sharpe_ratio=round(float(results["sharpe_ratio"]), 4),
            sortino_ratio=round(float(results["sortino_ratio"]), 4),
            calmar_ratio=round(float(results["calmar_ratio"]), 4),
        ),
        risk=RiskSummary(
            max_drawdown=round(float(results["max_drawdown"]), 6),
            var_95=round(float(var_historical(returns, 0.95)), 6),
            cvar_95=round(float(cvar(returns, 0.95)), 6),
        ),
        exposure=ExposureSummary(**exposure),
        costs=CostSummary(
            total_commission_pct=round(total_commission_pct, 6),
            total_slippage_pct=round(total_slippage_pct, 6),
            total_cost_pct=round(total_commission_pct + total_slippage_pct, 6),
            num_trades=int(results["num_trades"]),
        ),
        equity_curve_uri=equity_curve_uri,
        trades_uri=trades_uri,
        warnings=warnings,
        validation_status=validation_status,
    )


# ──────────────────────────────────────────────────────────────────
# Options Pricing, Greeks & Implied Volatility
# ──────────────────────────────────────────────────────────────────


def get_option_pricing(input_data: OptionPricingInput) -> OptionPricingResult:
    """Black-Scholes-Merton price and Greeks for a European option (European exercise only)."""
    logger.debug(
        "[option_pricing] %s  S=%.4f K=%.4f T=%.4f r=%.4f sigma=%.4f q=%.4f",
        input_data.option_type,
        input_data.spot,
        input_data.strike,
        input_data.time_to_expiry,
        input_data.risk_free_rate,
        input_data.volatility,
        input_data.dividend_yield,
    )
    price = black_scholes_price(
        input_data.spot,
        input_data.strike,
        input_data.time_to_expiry,
        input_data.risk_free_rate,
        input_data.volatility,
        option_type=input_data.option_type,
        dividend_yield=input_data.dividend_yield,
    )
    greeks = black_scholes_greeks(
        input_data.spot,
        input_data.strike,
        input_data.time_to_expiry,
        input_data.risk_free_rate,
        input_data.volatility,
        option_type=input_data.option_type,
        dividend_yield=input_data.dividend_yield,
    )
    return OptionPricingResult(
        option_type=input_data.option_type,
        price=round(price, 6),
        greeks=OptionGreeks(
            delta=round(greeks["delta"], 6),
            gamma=round(greeks["gamma"], 6),
            vega=round(greeks["vega"], 6),
            theta=round(greeks["theta"], 6),
            rho=round(greeks["rho"], 6),
        ),
        d1=round(greeks["d1"], 6),
        d2=round(greeks["d2"], 6),
    )


def get_implied_volatility(
    input_data: ImpliedVolatilityInput,
) -> ImpliedVolatilityResult:
    """Solve for Black-Scholes-Merton implied volatility from an observed European option price."""
    logger.debug(
        "[implied_volatility] %s  price=%.4f  S=%.4f K=%.4f T=%.4f r=%.4f",
        input_data.option_type,
        input_data.option_price,
        input_data.spot,
        input_data.strike,
        input_data.time_to_expiry,
        input_data.risk_free_rate,
    )
    result = _implied_volatility(
        input_data.option_price,
        input_data.spot,
        input_data.strike,
        input_data.time_to_expiry,
        input_data.risk_free_rate,
        option_type=input_data.option_type,
        dividend_yield=input_data.dividend_yield,
    )
    return ImpliedVolatilityResult(
        implied_volatility=round(result["implied_volatility"], 6),
        converged=result["converged"],
        iterations=result["iterations"],
        method=result["method"],
    )


# ──────────────────────────────────────────────────────────────────
# Extended Backtest Diagnostics
# ──────────────────────────────────────────────────────────────────


def get_backtest_diagnostics(
    input_data: BacktestDiagnosticsInput,
) -> BacktestDiagnosticsResult:
    """
    Extended diagnostics for one of the library's built-in strategies:
    top drawdown episodes (with recovery time), trade expectancy/payoff/
    streaks with MAE/MFE, and exposure statistics — the detail Sharpe and
    total return alone don't surface. Reuses run_strategy for the backtest
    itself and the pure functions in metrics/diagnostics.py for everything
    else; no new backtest math.
    """
    logger.debug(
        "[backtest_diagnostics] %s  %s  %s → %s",
        input_data.symbol,
        input_data.strategy_type,
        input_data.start_date,
        input_data.end_date,
    )
    if input_data.strategy_type not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{input_data.strategy_type}'. "
            f"Available: {list(STRATEGY_REGISTRY)}"
        )

    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(
        input_data.symbol, input_data.start_date, input_data.end_date
    )
    signals = STRATEGY_REGISTRY[input_data.strategy_type](df, **input_data.parameters)

    results = run_strategy(
        df,
        signals,
        input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        include_trade_log=True,
        fill_price=input_data.fill_price,
    )
    equity_curve = results["equity_curve"]
    trade_log = results.get("trade_log", pd.DataFrame())
    # Same one-bar-lag convention run_strategy applies internally
    # (signals.shift(1)) — needed here to report the *held* position,
    # not the raw pre-lag signal.
    executed = signals.shift(1).fillna(0.0)

    top_dd = top_n_drawdowns(equity_curve, n=input_data.top_n_drawdowns)
    drawdown_episodes = [
        DrawdownEpisode(
            start=str(row["start"].date()),
            trough=str(row["trough"].date()),
            end=None if pd.isna(row["end"]) else str(row["end"].date()),
            depth=float(row["depth"]),
            duration_bars=int(row["duration_bars"]),
            recovery_bars=(
                None if pd.isna(row["recovery_bars"]) else int(row["recovery_bars"])
            ),
        )
        for _, row in top_dd.iterrows()
    ]

    expectancy = trade_expectancy(trade_log)
    excursions = trade_excursions(trade_log, df)
    avg_mae = float(excursions["mae_pct"].mean()) if not excursions.empty else 0.0
    avg_mfe = float(excursions["mfe_pct"].mean()) if not excursions.empty else 0.0

    exposure = exposure_stats(executed, trade_log)

    logger.debug(
        "[backtest_diagnostics] sharpe=%.3f  drawdowns=%d  expectancy=%.2f%%  time_in_market=%.0f%%",
        results["sharpe_ratio"],
        len(drawdown_episodes),
        expectancy["expectancy_pct"],
        exposure["time_in_market"] * 100,
    )

    return BacktestDiagnosticsResult(
        symbol=input_data.symbol,
        strategy_type=input_data.strategy_type,
        total_return=round(float(results["total_return"]), 6),
        sharpe_ratio=round(float(results["sharpe_ratio"]), 4),
        sortino_ratio=round(float(results["sortino_ratio"]), 4),
        max_drawdown=round(float(results["max_drawdown"]), 6),
        calmar_ratio=round(float(results["calmar_ratio"]), 4),
        num_trades=int(results["num_trades"]),
        top_drawdowns=drawdown_episodes,
        trade_diagnostics=TradeDiagnostics(
            expectancy_pct=expectancy["expectancy_pct"],
            avg_winner_pct=expectancy["avg_winner_pct"],
            avg_loser_pct=expectancy["avg_loser_pct"],
            payoff_ratio=expectancy["payoff_ratio"],
            max_consecutive_wins=expectancy["max_consecutive_wins"],
            max_consecutive_losses=expectancy["max_consecutive_losses"],
            avg_mae_pct=round(avg_mae, 4),
            avg_mfe_pct=round(avg_mfe, 4),
        ),
        exposure=ExposureDiagnostics(**exposure),
    )


# ──────────────────────────────────────────────────────────────────


def get_agent_tools(
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns tool definitions formatted for OpenAI / Anthropic function calling.
    All tools have Pydantic-derived schemas — no manual JSON authoring required.

    Args:
        categories: Optional list of `TOOL_CATEGORY` values (see below) to
            filter to — e.g. `["screener"]` returns only the 2 screener
            tools. `None` (the default) returns every tool, identical to
            this function's behavior before this parameter existed; every
            existing caller (`dispatch()`, single-agent scripts that haven't
            adopted a router yet) keeps working unchanged. An unknown
            category name is silently ignored rather than raising, since a
            router's job is to narrow *when confident*, not to be a strict
            validator — see `agent/router.py`.
    """
    tool_defs = [
        ("run_sma_backtest", "SMA crossover backtest.", BacktestInput),
        ("run_rsi_backtest", "RSI mean-reversion backtest.", BacktestInput),
        ("run_macd_backtest", "MACD crossover backtest.", BacktestInput),
        (
            "run_bollinger_backtest",
            "Bollinger Band mean-reversion backtest.",
            BacktestInput,
        ),
        (
            "run_buy_and_hold",
            "Buy-and-hold baseline: long the full period. Use as a passive benchmark.",
            BuyAndHoldInput,
        ),
        (
            "compare_strategies",
            "Run all four strategies on the same symbol and return ranked results vs buy-and-hold.",
            CompareStrategiesInput,
        ),
        (
            "analyze_stock_risk",
            "Full risk analysis: alpha, beta, Sharpe, VaR, CVaR.",
            AnalysisInput,
        ),
        (
            "get_technical_analysis",
            "Compute configurable technical indicators.",
            TechnicalInput,
        ),
        ("get_portfolio_analysis", "Multi-asset portfolio metrics.", PortfolioInput),
        (
            "run_portfolio_optimization",
            "Produce portfolio weights via Markowitz mean-variance (max_sharpe/min_volatility/target_return/target_volatility), risk parity, or Black-Litterman — unlike get_portfolio_analysis, which only scores weights already chosen.",
            PortfolioOptimizationInput,
        ),
        (
            "run_screener",
            "Filter a stock universe by fundamental and technical criteria.",
            ScreenerInput,
        ),
        (
            "run_factor_regression",
            "Multi-factor OLS regression: alpha, loadings, t-stats, p-values, R².",
            FactorRegressionInput,
        ),
        (
            "run_cointegration_test",
            "Engle-Granger cointegration: hedge ratio, half-life, spread z-score signal.",
            CointegrationInput,
        ),
        (
            "run_kalman_hedge_ratio",
            "Time-varying hedge ratio via a Kalman filter — a staleness diagnostic companion to run_cointegration_test's static OLS hedge ratio.",
            KalmanHedgeRatioInput,
        ),
        (
            "run_pca_analysis",
            "PCA on multi-asset returns: explained variance, loadings, factor contributions.",
            PCAInput,
        ),
        (
            "get_correlation_analysis",
            "Correlation matrix, avg pairwise correlation, most/least correlated pair, and diversification ratio for a universe.",
            CorrelationAnalysisInput,
        ),
        (
            "run_hurst_analysis",
            "Hurst exponent (DFA/R-S): regime classification and optional rolling breakdown.",
            HurstInput,
        ),
        (
            "get_rally_signal",
            "Detect a rally via 5 confirming signals: return z-score, ADX trend strength, DI+/DI- direction, Hurst trending regime, and new-high breakout.",
            RallyDetectionInput,
        ),
        (
            "get_volatility_estimators",
            "Realized volatility via Parkinson, Garman-Klass, and Yang-Zhang estimators vs. plain close-to-close.",
            VolatilityEstimatorsInput,
        ),
        (
            "run_garch_volatility_forecast",
            "GARCH(1,1) conditional volatility: fits how variance evolves over time and forecasts it forward, unlike get_volatility_estimators' backward-looking realized estimates.",
            GarchVolatilityForecastInput,
        ),
        (
            "run_regime_adaptive_backtest",
            "Classify market regime via Hurst, auto-select and optimise the best strategy.",
            RegimeAdaptiveInput,
        ),
        (
            "run_regime_adaptive_walkforward_backtest",
            "Leakage-free regime-adaptive backtest: regime/strategy/parameter selection per walk-forward window, evaluated strictly out-of-sample.",
            RegimeAdaptiveWalkForwardInput,
        ),
        (
            "scan_pairs",
            "Scan a ticker universe for cointegrated pairs, ranked by half-life.",
            PairScannerInput,
        ),
        (
            "run_walk_forward_backtest",
            "Walk-forward validation: optimise in-sample, evaluate out-of-sample, return OOS stats.",
            WalkForwardInput,
        ),
        (
            "get_portfolio_risk_attribution",
            "Deep portfolio risk decomposition: MCR per asset, PCA attribution, optional factor model.",
            RiskAttributionInput,
        ),
        (
            "run_stress_test",
            "Replay a portfolio's weights against a named historical crash window (or custom date range) using real historical returns.",
            StressTestInput,
        ),
        (
            "get_position_size",
            "ATR-based position sizing with optional Kelly criterion.",
            PositionSizerInput,
        ),
        (
            "get_stock_fundamentals",
            "Fetch company metadata and key financial ratios (PE, P/B, debt/equity, ROE, market cap).",
            FundamentalsInput,
        ),
        (
            "run_backtest_optimization",
            "Grid-search strategy parameters and return the top N combinations ranked by a chosen metric.",
            BacktestOptInput,
        ),
        (
            "get_advanced_indicators",
            "Compute Parabolic SAR (trend), Wilder ATR (volatility), and MFI (volume-flow oscillator).",
            AdvancedIndicatorsInput,
        ),
        (
            "get_rolling_beta",
            "Compute rolling OLS beta to detect beta drift over time vs a benchmark.",
            RollingBetaInput,
        ),
        (
            "get_extended_risk_metrics",
            "Extended risk: Calmar ratio, Treynor ratio, parametric VaR 95/99, historical VaR 99, CVaR 99.",
            ExtendedRiskInput,
        ),
        (
            "get_tail_risk_metrics",
            "Extreme Value Theory tail risk (Peaks-Over-Threshold GPD fit): VaR/CVaR extrapolated from the fitted tail, compared against the naive historical quantile.",
            TailRiskInput,
        ),
        (
            "run_custom_signal_backtest",
            "Backtest a signal computed outside this library (your own alpha model) on one symbol.",
            CustomSignalBacktestInput,
        ),
        (
            "run_signal_panel_backtest",
            "Backtest a pre-computed signal panel across a ticker universe, combined into portfolio metrics.",
            SignalPanelBacktestInput,
        ),
        (
            "get_backtest_diagnostics",
            "Extended diagnostics for a built-in strategy: top drawdown episodes, trade expectancy/payoff/streaks with MAE/MFE, and exposure stats.",
            BacktestDiagnosticsInput,
        ),
        (
            "run_portfolio_simulation",
            "True shared-cash portfolio simulation with rebalancing at target-weight dates — unlike run_signal_panel_backtest, positions share one account instead of each ticker getting its own capital.",
            PortfolioSimulationInput,
        ),
        (
            "run_pair_trade_backtest",
            "Backtest a cointegrated pair as one synchronized two-leg trade — both legs enter/exit together and share one cash account, unlike scan_pairs which only screens candidates.",
            PairTradeBacktestInput,
        ),
        (
            "get_robustness_diagnostics",
            "Same-sample robustness checks for a grid search: parameter sensitivity, Deflated Sharpe Ratio, and a block-bootstrap confidence interval on the best trial's Sharpe ratio.",
            RobustnessDiagnosticsInput,
        ),
        (
            "run_monte_carlo_simulation",
            "Monte Carlo forward simulation of a portfolio's future equity paths via moving-block bootstrap of its historical returns.",
            MonteCarloSimulationInput,
        ),
        (
            "get_capacity_report",
            "How much account size a target-weight portfolio can support before positions become too large relative to each ticker's own trading volume, plus days-to-liquidate and sector exposure.",
            CapacityReportInput,
        ),
        (
            "get_liquidity_metrics",
            "Amihud illiquidity ratio and Corwin-Schultz spread estimator per ticker — OHLCV-derived proxies for market depth and bid/ask spread.",
            LiquidityAnalysisInput,
        ),
        (
            "get_data_quality_report",
            "Dataset provenance (adjusted/survivorship-free/point-in-time guarantees) plus missing-bar/stale-price/price-jump detection on a symbol's OHLCV.",
            DataQualityReportInput,
        ),
        (
            "run_backtest_compact",
            "Compact backtest result: summary/risk/exposure/cost sub-reports plus equity-curve/trade-log artifact URIs, instead of embedding the full data inline like run_sma_backtest etc.",
            BacktestCompactInput,
        ),
        (
            "get_option_pricing",
            "Black-Scholes-Merton price and Greeks (delta, gamma, vega, theta, rho) for a European option.",
            OptionPricingInput,
        ),
        (
            "get_implied_volatility",
            "Solve for Black-Scholes-Merton implied volatility from an observed European option price.",
            ImpliedVolatilityInput,
        ),
    ]

    if categories is not None:
        allowed = set(categories)
        tool_defs = [t for t in tool_defs if TOOL_CATEGORY.get(t[0]) in allowed]

    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": model.model_json_schema(),
            },
        }
        for name, desc, model in tool_defs
    ]


# ──────────────────────────────────────────────────────────────────
# Dispatch — route LLM tool calls by name
# ──────────────────────────────────────────────────────────────────

_TOOL_DISPATCH: Dict[str, Any] = {
    "run_sma_backtest": (run_sma_backtest, BacktestInput),
    "run_rsi_backtest": (run_rsi_backtest, BacktestInput),
    "run_macd_backtest": (run_macd_backtest, BacktestInput),
    "run_bollinger_backtest": (run_bollinger_backtest, BacktestInput),
    "run_buy_and_hold": (run_buy_and_hold, BuyAndHoldInput),
    "compare_strategies": (compare_strategies, CompareStrategiesInput),
    "analyze_stock_risk": (analyze_stock_risk, AnalysisInput),
    "get_technical_analysis": (get_technical_analysis, TechnicalInput),
    "get_portfolio_analysis": (get_portfolio_analysis, PortfolioInput),
    "run_portfolio_optimization": (
        run_portfolio_optimization,
        PortfolioOptimizationInput,
    ),
    "run_screener": (run_screener, ScreenerInput),
    "run_factor_regression": (run_factor_regression, FactorRegressionInput),
    "run_cointegration_test": (run_cointegration_test, CointegrationInput),
    "run_kalman_hedge_ratio": (run_kalman_hedge_ratio, KalmanHedgeRatioInput),
    "run_pca_analysis": (run_pca_analysis, PCAInput),
    "get_correlation_analysis": (get_correlation_analysis, CorrelationAnalysisInput),
    "run_hurst_analysis": (run_hurst_analysis, HurstInput),
    "get_rally_signal": (get_rally_signal, RallyDetectionInput),
    "get_volatility_estimators": (get_volatility_estimators, VolatilityEstimatorsInput),
    "run_garch_volatility_forecast": (
        run_garch_volatility_forecast,
        GarchVolatilityForecastInput,
    ),
    "run_regime_adaptive_backtest": (run_regime_adaptive_backtest, RegimeAdaptiveInput),
    "run_regime_adaptive_walkforward_backtest": (
        run_regime_adaptive_walkforward_backtest,
        RegimeAdaptiveWalkForwardInput,
    ),
    "scan_pairs": (scan_pairs, PairScannerInput),
    "run_walk_forward_backtest": (run_walk_forward_backtest, WalkForwardInput),
    "get_portfolio_risk_attribution": (
        get_portfolio_risk_attribution,
        RiskAttributionInput,
    ),
    "run_stress_test": (run_stress_test, StressTestInput),
    "get_position_size": (get_position_size, PositionSizerInput),
    "get_stock_fundamentals": (get_stock_fundamentals, FundamentalsInput),
    "run_backtest_optimization": (run_backtest_optimization, BacktestOptInput),
    "get_advanced_indicators": (get_advanced_indicators, AdvancedIndicatorsInput),
    "get_rolling_beta": (get_rolling_beta, RollingBetaInput),
    "get_extended_risk_metrics": (get_extended_risk_metrics, ExtendedRiskInput),
    "get_tail_risk_metrics": (get_tail_risk_metrics, TailRiskInput),
    "run_custom_signal_backtest": (
        run_custom_signal_backtest,
        CustomSignalBacktestInput,
    ),
    "run_signal_panel_backtest": (run_signal_panel_backtest, SignalPanelBacktestInput),
    "get_backtest_diagnostics": (get_backtest_diagnostics, BacktestDiagnosticsInput),
    "run_portfolio_simulation": (run_portfolio_simulation, PortfolioSimulationInput),
    "run_pair_trade_backtest": (run_pair_trade_backtest, PairTradeBacktestInput),
    "get_robustness_diagnostics": (
        get_robustness_diagnostics,
        RobustnessDiagnosticsInput,
    ),
    "run_monte_carlo_simulation": (
        run_monte_carlo_simulation,
        MonteCarloSimulationInput,
    ),
    "get_capacity_report": (get_capacity_report, CapacityReportInput),
    "get_liquidity_metrics": (get_liquidity_metrics, LiquidityAnalysisInput),
    "get_data_quality_report": (get_data_quality_report, DataQualityReportInput),
    "run_backtest_compact": (run_backtest_compact, BacktestCompactInput),
    "get_option_pricing": (get_option_pricing, OptionPricingInput),
    "get_implied_volatility": (get_implied_volatility, ImpliedVolatilityInput),
}


# ──────────────────────────────────────────────────────────────────
# Tool categories — single source of truth for anything that needs to
# group/filter/narrow the tool set: get_agent_tools(categories=...),
# agent/router.py's classification prompt, and
# Multi_Agent_Implementation/worker_agents.py's WORKER_AGENTS (each worker's
# "tools" list is *derived* from this dict, not hand-duplicated). Every key
# in _TOOL_DISPATCH must appear here exactly once — enforced by
# tests/test_agent_tools.py::TestToolCategoryCoverage.
# ──────────────────────────────────────────────────────────────────

TOOL_CATEGORY: Dict[str, str] = {
    # screener — filter a universe, fetch fundamentals
    "run_screener": "screener",
    "get_stock_fundamentals": "screener",
    # analysis — single-asset risk/technical/volatility/option profiling,
    # multi-asset portfolio metrics, data quality
    "analyze_stock_risk": "analysis",
    "get_technical_analysis": "analysis",
    "get_advanced_indicators": "analysis",
    "get_rally_signal": "analysis",
    "get_rolling_beta": "analysis",
    "get_extended_risk_metrics": "analysis",
    "get_tail_risk_metrics": "analysis",
    "get_portfolio_analysis": "analysis",
    "get_data_quality_report": "analysis",
    "get_volatility_estimators": "analysis",
    "run_garch_volatility_forecast": "analysis",
    "get_option_pricing": "analysis",
    "get_implied_volatility": "analysis",
    # quant_research — factor/cointegration/PCA/Hurst/correlation structure
    "run_factor_regression": "quant_research",
    "run_cointegration_test": "quant_research",
    "run_kalman_hedge_ratio": "quant_research",
    "run_pca_analysis": "quant_research",
    "run_hurst_analysis": "quant_research",
    "scan_pairs": "quant_research",
    "get_correlation_analysis": "quant_research",
    # backtest_execution — run a built-in strategy / portfolio / pair once
    "run_sma_backtest": "backtest_execution",
    "run_rsi_backtest": "backtest_execution",
    "run_macd_backtest": "backtest_execution",
    "run_bollinger_backtest": "backtest_execution",
    "run_buy_and_hold": "backtest_execution",
    "compare_strategies": "backtest_execution",
    "run_backtest_compact": "backtest_execution",
    "run_portfolio_simulation": "backtest_execution",
    "run_pair_trade_backtest": "backtest_execution",
    # backtest_validation — optimize/validate/diagnose a built-in strategy
    "run_backtest_optimization": "backtest_validation",
    "run_regime_adaptive_backtest": "backtest_validation",
    "run_regime_adaptive_walkforward_backtest": "backtest_validation",
    "run_walk_forward_backtest": "backtest_validation",
    "get_backtest_diagnostics": "backtest_validation",
    "get_robustness_diagnostics": "backtest_validation",
    "run_monte_carlo_simulation": "backtest_validation",
    # custom_signal — backtest a signal computed outside this library
    "run_custom_signal_backtest": "custom_signal",
    "run_signal_panel_backtest": "custom_signal",
    # portfolio_risk — risk decomposition, sizing, capacity, stress, liquidity
    "get_portfolio_risk_attribution": "portfolio_risk",
    "get_position_size": "portfolio_risk",
    "get_capacity_report": "portfolio_risk",
    "run_stress_test": "portfolio_risk",
    "get_liquidity_metrics": "portfolio_risk",
    "run_portfolio_optimization": "portfolio_risk",
}


def dispatch(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Route an LLM tool call to the correct tool function and return a JSON-ready dict.

    Replaces the manual TOOL_FN / INPUT_MODEL lookup pattern. Pass the tool name
    and parsed arguments from the LLM response; get back a plain dict from
    result.model_dump() ready to send back to the model.

    Args:
        tool_name:  Function name as returned by the LLM (e.g. "analyze_stock_risk").
        arguments:  Parsed tool arguments dict from the LLM tool call.

    Returns:
        result.model_dump() — a plain dict, JSON-serializable.

    Raises:
        ValueError: Unknown tool name.
        pydantic.ValidationError: Arguments don't match the tool's input schema.

    Example (OpenAI)::

        import json
        from standard_quant_tools.agent import get_agent_tools, dispatch

        for tc in msg.tool_calls:
            result = dispatch(tc.function.name, json.loads(tc.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    Example (Anthropic)::

        for block in response.content:
            if block.type == "tool_use":
                result = dispatch(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
    """
    if tool_name not in _TOOL_DISPATCH:
        raise ValueError(
            f"Unknown tool '{tool_name}'. " f"Available: {sorted(_TOOL_DISPATCH)}"
        )
    fn, model_cls = _TOOL_DISPATCH[tool_name]
    logger.debug("[dispatch] → %s  args=%s", tool_name, list(arguments.keys()))
    t0 = time.perf_counter()
    try:
        result = audit._run_and_record(tool_name, fn, model_cls(**arguments))
    except Exception as exc:
        logger.error("[dispatch] ✗ %s  error=%s", tool_name, exc)
        raise
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.debug("[dispatch] ✓ %s  completed in %.0fms", tool_name, elapsed_ms)
    return _sanitize_for_json(result)


# Re-exported under its original private name so existing imports of
# agent.tools._sanitize_for_json keep working. The implementation now lives
# in standard_quant_tools._jsonsafe because BOTH agent surfaces need it and
# neither should import the other — the modeling runtime is deliberately
# independent of this 46-tool registry. See that module for why non-finite
# metrics are real here and why None is the right JSON representation.
_sanitize_for_json = sanitize_for_json
