"""
The `research` runtime: describe an asset or a universe.

Screening, single-asset risk and technical profiling, and the statistical
structure of a universe -- factors, cointegration, PCA, Hurst, correlation.
Everything here answers "what is this thing like"; nothing here runs a
strategy, and a backtest tool called against this runtime is refused by
name rather than executed.
"""

import datetime
import logging
import math
import re
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from standard_quant_tools.agent.models import (
    AdvancedIndicatorsInput,
    AdvancedIndicatorsResult,
    AnalysisInput,
    AnalysisResult,
    ChangePoint,
    ChangePointInput,
    ChangePointResult,
    CointegrationInput,
    CointegrationResult,
    CorrelationAnalysisInput,
    CorrelationAnalysisResult,
    DataQualityReportInput,
    DataQualityReportResult,
    ExtendedRiskInput,
    ExtendedRiskResult,
    FactorRegressionInput,
    FactorRegressionResult,
    FundamentalsInput,
    FundamentalsResult,
    GarchVolatilityForecastInput,
    GarchVolatilityForecastResult,
    GrangerInput,
    GrangerLag,
    GrangerResult,
    HurstInput,
    HurstResult,
    KalmanHedgeRatioInput,
    KalmanHedgeRatioResult,
    MissingBar,
    PairFailure,
    PairResult,
    PairScannerInput,
    PairScannerResult,
    PartialCorrelationInput,
    PartialCorrelationResult,
    PCAInput,
    PCAResult,
    PortfolioInput,
    PortfolioResult,
    PriceJump,
    RallyDetectionInput,
    RallyDetectionResult,
    Regime,
    RegimeDetectionInput,
    RegimeDetectionResult,
    RegimeSegment,
    RollingBetaInput,
    RollingBetaResult,
    ScreenerInput,
    ScreenerResult,
    StalePriceRun,
    StationarityInput,
    StationarityResult,
    TailDependenceInput,
    TailDependenceResult,
    TailRiskInput,
    TailRiskResult,
    TechnicalInput,
    TechnicalPanelInput,
    TechnicalPanelResult,
    TechnicalResult,
    VarianceRatio,
    VolatilityEstimatorsInput,
    VolatilityEstimatorsResult,
)
from standard_quant_tools.agent.runtimes._shared import (
    HAS_CPP,
    _cpp_core,
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
from standard_quant_tools.analysis.pca import factor_contributions, pca_returns
from standard_quant_tools.analysis.rally import detect_rally
from standard_quant_tools.analysis.regression import calculate_beta, rolling_beta
from standard_quant_tools.backtest.artifacts import save_artifact
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.data.quality import (
    detect_missing_bars,
    detect_price_jumps,
    detect_stale_prices,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.momentum import rsi, stochastic_oscillator
from standard_quant_tools.indicators.panel import (
    technical_indicators_panel as _technical_indicators_panel,
)
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
from standard_quant_tools.portfolio.portfolio import (
    fetch_ohlcv_panel_sync,
    fetch_returns_sync,
    portfolio_metrics,
)
from standard_quant_tools.screener.screener import screen_stocks
from standard_quant_tools.validation import last_finite

# Indicators technical_indicators() can compute in one native call. "atr" is
# deliberately excluded: the tool's plain atr() uses a simple rolling mean,
# while the fused call's ATR field is Wilder-smoothed -- a different
# algorithm, not just a faster path to the same numbers.
_FUSABLE_INDICATORS = {"rsi", "adx", "bollinger", "stochastic"}
_PERIOD_PATTERN = re.compile(r"^(\d+)(d|w|mo|y)$")
_PERIOD_DAYS = {"d": 1, "w": 7, "mo": 30, "y": 365}


def _parse_period(period: str) -> datetime.datetime:
    """
    Convert a period string ('1y', '6mo', '2y', '30d', '4w') to a start
    datetime.

    An unrecognized unit used to fall through to `now - 365 days`, so a
    malformed request did not fail — it silently became a DIFFERENT valid
    request. '6m' (a plausible typo for '6mo'), '1yr', 'ytd' and '' all
    quietly returned one year of data, and every downstream number was then
    computed over a window the caller never asked for. A wrong window is not
    detectable from the result, which is what makes the silent fallback worse
    than an error.

    Parsed strictly, with the accepted forms named in the message.
    """
    if not isinstance(period, str):
        raise ValidationError(
            f"period must be a string like '1y', '6mo', '30d', got "
            f"{type(period).__name__}"
        )
    match = _PERIOD_PATTERN.match(period.strip().lower())
    if match is None:
        raise ValidationError(
            f"period={period!r} is not a recognized window. Use <number><unit> "
            "where unit is d (days), w (weeks), mo (months) or y (years) — for "
            "example '30d', '4w', '6mo', '2y'. An unrecognized value used to "
            "fall back to one year silently, turning a malformed request into a "
            "different valid one."
        )
    amount = int(match.group(1))
    if amount < 1:
        raise ValidationError(f"period={period!r} must cover at least one unit")
    days = amount * _PERIOD_DAYS[match.group(2)]
    return datetime.datetime.now() - datetime.timedelta(days=days)


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

    asset_ret = asset_df["Close"].pct_change(fill_method=None).dropna()
    bench_ret = bench_df["Close"].pct_change(fill_method=None).dropna()

    beta_metrics = calculate_beta(asset_ret, bench_ret)
    equity_curve = (1 + asset_ret).cumprod()

    result = AnalysisResult(
        symbol=input_data.symbol,
        benchmark=input_data.benchmark,
        alpha=round(beta_metrics["alpha"], 6),
        beta=round(beta_metrics["beta"], 4),
        r_squared=round(beta_metrics["r_squared"], 4),
        sharpe_ratio=round(sharpe_ratio(asset_ret, input_data.risk_free_rate), 4),
        sortino_ratio=round(sortino_ratio(asset_ret, input_data.risk_free_rate), 4),
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
        last_macd = last_finite(m["MACD"], "MACD")
        last_sig = last_finite(m["Signal"], "Signal")
        last_hist = last_finite(m["Histogram"], "Histogram")
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
        k = last_finite(stoch["Stoch_K"], "Stoch_K")
        d = last_finite(stoch["Stoch_D"], "Stoch_D")
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
        upper = last_finite(bb["BB_Upper"], "BB_Upper")
        middle = last_finite(bb["BB_Middle"], "BB_Middle")
        lower = last_finite(bb["BB_Lower"], "BB_Lower")
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
            last_vwap = last_finite(v, "v")
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
    bench_returns = bench_df_raw["Close"].pct_change(fill_method=None).dropna()

    common_idx = returns_df.index.intersection(bench_returns.index)
    aligned_returns = returns_df.loc[common_idx]
    aligned_bench = bench_returns.loc[common_idx]

    metrics = portfolio_metrics(
        aligned_returns,
        input_data.weights,
        risk_free_rate=input_data.risk_free_rate,
        benchmark_returns=aligned_bench,
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


def correlation_matrix_to_dict(returns_df: pd.DataFrame) -> Dict[str, Any]:
    from standard_quant_tools.portfolio.portfolio import correlation_matrix

    corr = correlation_matrix(returns_df)
    return corr.round(4).to_dict()


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
    asset_rets = asset_df["Close"].pct_change(fill_method=None).dropna()

    names = input_data.factor_names or input_data.factor_tickers
    factor_series = {}
    for ticker, name in zip(input_data.factor_tickers, names):
        df = provider.get_ohlcv(ticker, input_data.start_date, input_data.end_date)
        factor_series[name] = df["Close"].pct_change(fill_method=None).dropna()

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

    kf = kalman_hedge_ratio(
        prices_a,
        prices_b,
        delta=input_data.delta,
        observation_noise=input_data.observation_noise,
        include_intercept=input_data.include_intercept,
    )
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
            ].pct_change(fill_method=None)
            for t in input_data.tickers
        }
    ).dropna()

    result = pca_returns(
        returns,
        n_components=input_data.n_components,
        standardize=input_data.standardize,
        method=input_data.method,
    )
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
    returns = df["Close"].pct_change(fill_method=None).dropna()

    result = hurst_exponent(
        returns,
        method=input_data.method,
        min_window=input_data.min_window,
        max_window=input_data.max_window,
    )

    rolling_current = None
    rolling_regime_fractions = None
    if input_data.rolling_window:
        rolling = rolling_hurst(
            returns,
            window=input_data.rolling_window,
            method=input_data.method,
            min_window=input_data.min_window,
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

    close_returns = df["Close"].pct_change(fill_method=None).dropna()
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
    returns = close.pct_change(fill_method=None).dropna()

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


def scan_pairs(input_data: PairScannerInput) -> PairScannerResult:
    """
    Test all ticker combinations for cointegration and return the top pairs
    ranked by half-life (shortest first = fastest mean-reversion = most tradeable).
    Fetches each ticker's prices once, then evaluates all O(n²/2) combinations.
    """
    from standard_quant_tools.analysis.cointegration import cointegration_test as _coint
    from standard_quant_tools.analysis.cointegration import compute_spread as _spread
    from standard_quant_tools.analysis.cointegration import (
        scan_cointegrated_pairs as _scan_pairs_batch,
    )
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

    # ── Batch fast path ──────────────────────────────────────────────────
    # The loop below is O(N^2) in the universe -- 2,000 tickers is 1,999,000
    # iterations, each paying a full pandas round trip into the extension and
    # none of them parallel. Measured at 2,000 bars that is 9.8 hours;
    # scan_cointegrated_pairs does the same work in about 5 minutes.
    #
    # Engaged only when every series shares an IDENTICAL index. The batch
    # path aligns the whole universe onto one common sample, while this loop
    # aligns each pair against only its own partner -- the same thing when
    # the indexes already match (the usual case for one date range from one
    # provider) and NOT the same thing when they do not. Rather than silently
    # change which bars a pair is tested on, fall back to the loop.
    batch: Dict[Tuple[str, str], Any] = {}
    if len(valid_tickers) >= 2:
        first = prices[valid_tickers[0]].index  # type: ignore[union-attr]
        if all(prices[t].index.equals(first) for t in valid_tickers[1:]):  # type: ignore[union-attr]
            try:
                frame = pd.DataFrame({t: prices[t] for t in valid_tickers})
                scanned = _scan_pairs_batch(frame, all_pairs)
                batch = {(a, b): row for (a, b), row in scanned.iterrows()}
                logger.debug(
                    "[scan_pairs] batch path: %d pairs in one native call",
                    len(batch),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "[scan_pairs] batch path failed (%s) - using the per-pair loop",
                    exc,
                )
                batch = {}

    for a, b in all_pairs:
        try:
            if batch:
                row = batch[(a, b)]
                if not math.isfinite(float(row["adf_statistic"])):
                    raise ValueError(
                        "Engle-Granger produced no statistic for this pair "
                        "(degenerate or perfectly collinear series)"
                    )
                result = {
                    "cointegrated": bool(row["cointegrated"]),
                    "p_value": float(row["p_value"]),
                    "hedge_ratio": float(row["hedge_ratio"]),
                    "adf_statistic": float(row["adf_statistic"]),
                    "half_life_days": float(row["half_life_days"]),
                }
            else:
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
        high,
        low,
        af_start=input_data.sar_af_start,
        af_step=input_data.sar_af_step,
        af_max=input_data.sar_af_max,
    )
    # Each of these takes the latest value of an indicator whose window can
    # exceed the data supplied. `.iloc[-1]` on the resulting empty series
    # raises a pandas IndexError that names neither the tool nor the
    # shortfall; `last_finite` names both.
    sar_val = last_finite(sar_df["SAR"], "SAR")
    sar_trend_int = int(last_finite(sar_df["Trend"], "SAR trend"))
    sar_trend = "bullish" if sar_trend_int == 1 else "bearish"
    sar_signal = "buy" if sar_trend_int == 1 else "sell"

    watr_series = wilder_atr(high, low, close, period=input_data.atr_period).dropna()
    watr_val = last_finite(watr_series, "Wilder ATR")
    watr_pct = watr_val / last_close if last_close > 0 else 0.0

    mfi_series = mfi(high, low, close, volume, period=input_data.mfi_period).dropna()
    mfi_val = last_finite(mfi_series, "MFI")
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

    asset_ret = asset_df["Close"].pct_change(fill_method=None).dropna()
    bench_ret = bench_df["Close"].pct_change(fill_method=None).dropna()

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

    asset_ret = asset_df["Close"].pct_change(fill_method=None).dropna()
    bench_ret = bench_df["Close"].pct_change(fill_method=None).dropna()
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
    returns = close.pct_change(fill_method=None).dropna()

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


def get_technical_panel(input_data: TechnicalPanelInput) -> TechnicalPanelResult:
    """
    Indicators for a whole universe in one call, reported at the latest bar.

    `get_technical_analysis` answers for one symbol, so screening fifty
    names cost fifty round trips while `indicators/panel.py` was already
    computing the entire universe in a single native call and was reachable
    from no tool. The arithmetic is identical to looping the per-ticker
    functions — the panel path feeds the same kernels — so this is the same
    answer, not an approximation of it.

    Only the LATEST bar is returned inline. A universe times a history is a
    matrix, and returning it would put megabytes into a conversation that
    then carries them for every subsequent turn; pass `persist_run_id` to
    write the full panel to Parquet and get URIs back instead.

    Tickers whose latest value is NaN for a requested indicator are listed
    in `incomplete_tickers` rather than dropped. Dropping them would make a
    ticker with too little history look identical to one a screen
    legitimately excluded.
    """
    logger.debug(
        "[technical_panel] tickers=%d indicators=%s",
        len(input_data.tickers),
        input_data.indicators,
    )
    panel = fetch_ohlcv_panel_sync(
        input_data.tickers, input_data.start_date, input_data.end_date
    )
    missing = [t for t in input_data.tickers if t not in panel or panel[t].empty]
    if missing:
        raise ValidationError(
            f"no OHLCV returned for {missing}. The panel needs bars for every "
            "requested ticker — a partial panel would silently change which "
            "universe the indicators describe."
        )

    frames = _technical_indicators_panel(
        panel,
        input_data.indicators,
        rsi_period=input_data.rsi_period,
        adx_period=input_data.adx_period,
        atr_period=input_data.atr_period,
        bollinger_period=input_data.bollinger_period,
        bollinger_num_std=input_data.bollinger_num_std,
        stoch_k_period=input_data.stoch_k_period,
        stoch_d_period=input_data.stoch_d_period,
    )

    # The panel is computed on the bars every ticker SHARES, so one young
    # ticker silently shortens the window for all of them. Work out who
    # bounds it before the indicators come back as unexplained NaNs.
    first_bars = {t: panel[t].index[0] for t in input_data.tickers}
    earliest = min(first_bars.values())
    shared_start = max(first_bars.values())
    limited_by = sorted(t for t, first in first_bars.items() if first > earliest)
    notes: List[str] = []

    latest: Dict[str, Dict[str, float]] = {t: {} for t in input_data.tickers}
    incomplete: set = set()
    n_bars = 0
    as_of = ""
    for name, frame in frames.items():
        if frame.empty:
            continue
        n_bars = max(n_bars, int(len(frame)))
        as_of = (
            str(frame.index[-1].date())
            if hasattr(frame.index[-1], "date")
            else str(frame.index[-1])
        )
        row = frame.iloc[-1]
        if isinstance(frame.columns, pd.MultiIndex):
            for ticker, field in frame.columns:
                value = row[(ticker, field)]
                if pd.isna(value):
                    incomplete.add(str(ticker))
                    continue
                latest.setdefault(str(ticker), {})[str(field)] = round(float(value), 6)
        else:
            # Single-column indicators are one column per ticker; label the
            # field with the indicator's own uppercase name so "rsi" reads
            # as RSI, matching get_technical_analysis's vocabulary.
            field = name.upper()
            for ticker in frame.columns:
                value = row[ticker]
                if pd.isna(value):
                    incomplete.add(str(ticker))
                    continue
                latest.setdefault(str(ticker), {})[field] = round(float(value), 6)

    artifact_uris: Dict[str, str] = {}
    if input_data.persist_run_id is not None:
        for name, frame in frames.items():
            if frame.empty:
                continue
            to_save = frame
            if isinstance(frame.columns, pd.MultiIndex):
                # Parquet has no MultiIndex column concept; flatten to
                # "TICKER::FIELD" so the round trip is lossless and the
                # separator cannot collide with a ticker or a field name.
                to_save = frame.copy()
                to_save.columns = [f"{t}::{f}" for t, f in frame.columns]
            artifact_uris[name] = save_artifact(
                to_save, input_data.persist_run_id, f"panel_{name}", overwrite=True
            )

    from standard_quant_tools.indicators import panel as _panel_module

    execution_path = (
        "C++"
        if (_panel_module.HAS_CPP and _panel_module._cpp_core is not None)
        else "per-ticker"
    )
    longest_lookback = max(
        (
            input_data.rsi_period if "rsi" in input_data.indicators else 0,
            input_data.adx_period if "adx" in input_data.indicators else 0,
            input_data.atr_period if "atr" in input_data.indicators else 0,
            (
                input_data.bollinger_period
                if "bollinger_bands" in input_data.indicators
                else 0
            ),
            (
                input_data.stoch_k_period + input_data.stoch_d_period
                if "stochastic_oscillator" in input_data.indicators
                else 0
            ),
        )
    )
    if limited_by:
        notes.append(
            f"The shared calendar starts {str(shared_start.date())}, later "
            f"than the earliest bar available in this universe, because "
            f"{limited_by} have shorter histories. Indicators are computed "
            "on the intersection, so those tickers shorten the window for "
            "every other one too."
        )
    if n_bars <= longest_lookback:
        notes.append(
            f"The shared calendar is {n_bars} bars but the longest requested "
            f"lookback is {longest_lookback}, so no indicator can have "
            "warmed up. Every value here is NaN for a reason that is about "
            "the calendar, not the tickers."
        )

    return TechnicalPanelResult(
        tickers=input_data.tickers,
        indicators=list(input_data.indicators),
        as_of=as_of,
        n_bars=n_bars,
        latest=latest,
        incomplete_tickers=sorted(incomplete),
        calendar_start=(
            str(shared_start.date())
            if hasattr(shared_start, "date")
            else str(shared_start)
        ),
        calendar_limited_by=limited_by,
        notes=notes,
        artifact_uris=artifact_uris,
        execution_path=execution_path,
    )


def _price_series(symbol: str, start_date: str, end_date: str, on: str = "price"):
    """Close prices, or their returns, for one symbol."""
    from standard_quant_tools.data.factory import DataFactory

    frame = DataFactory.get_provider().get_ohlcv(symbol, start_date, end_date)
    close = frame["Close"] if "Close" in frame.columns else frame["close"]
    return close.pct_change(fill_method=None).dropna() if on == "returns" else close


def detect_change_points(input_data: ChangePointInput) -> ChangePointResult:
    """
    When the process generating this series CHANGED — not what kind of
    process it is.

    `run_hurst_analysis` answers "is this trending or mean-reverting". This
    answers "and when did it stop being that", which the first cannot: a
    single Hurst exponent over a sample containing a regime break describes
    neither regime.

    Uses binary segmentation, which finds the strongest break and recurses
    either side. It can miss two breaks that cancel — a step up followed by
    an equal step down — and that is the known cost rather than a surprise.
    Read `gain` on each break: it is how much the split actually bought, so
    a marginal call looks marginal instead of looking like a boundary.
    """
    from standard_quant_tools.analysis.structure import detect_change_points as _detect

    logger.debug("[detect_change_points] %s", input_data.symbol)
    series = _price_series(
        input_data.symbol, input_data.start_date, input_data.end_date, input_data.on
    )
    result = _detect(
        series,
        max_breaks=input_data.max_breaks,
        min_segment=input_data.min_segment,
        penalty=input_data.penalty,
    )
    return ChangePointResult(
        symbol=input_data.symbol,
        n_observations=result["n_observations"],
        n_breaks=result["n_breaks"],
        breaks=[ChangePoint(**b) for b in result["breaks"]],
        segments=[RegimeSegment(**s) for s in result["segments"]],
        warnings=result["warnings"],
    )


def get_partial_correlation(
    input_data: PartialCorrelationInput,
) -> PartialCorrelationResult:
    """
    The correlation between two assets once the common drivers are removed
    from both.

    Two stocks in the same sector correlate at 0.7 and it says almost
    nothing about their relationship — remove the market and the sector and
    what is left is the part that is actually about those two companies.
    That residual is what a pair trade lives on, and the raw correlation
    systematically overstates it.
    """
    from standard_quant_tools.analysis.structure import partial_correlation as _partial

    logger.debug("[get_partial_correlation] %s vs %s", input_data.x, input_data.y)
    names = [input_data.x, input_data.y] + list(input_data.controlling_for)
    frame = pd.concat(
        {
            name: _price_series(
                name, input_data.start_date, input_data.end_date, "returns"
            )
            for name in dict.fromkeys(names)
        },
        axis=1,
    )
    result = _partial(frame, input_data.x, input_data.y, input_data.controlling_for)
    return PartialCorrelationResult(**result)


def test_granger_causality(input_data: GrangerInput) -> GrangerResult:
    """
    Does one series help predict another beyond that series' own past?

    NOT CAUSALITY, whatever the name says. A common driver produces this. A
    faster-updating proxy for the same information produces this. What it
    establishes is temporal precedence in a linear model — necessary for a
    tradeable lead and nowhere near sufficient.

    Every lag up to `max_lag` is tested and the smallest p-value reported,
    which is a multiple comparison. Treat the number as a screen rather than
    a test result.
    """
    from standard_quant_tools.analysis.structure import granger_causality as _granger

    logger.debug(
        "[test_granger_causality] %s -> %s", input_data.cause, input_data.effect
    )
    cause = _price_series(
        input_data.cause, input_data.start_date, input_data.end_date, "returns"
    )
    effect = _price_series(
        input_data.effect, input_data.start_date, input_data.end_date, "returns"
    )
    result = _granger(cause, effect, max_lag=input_data.max_lag)
    return GrangerResult(
        cause=input_data.cause,
        effect=input_data.effect,
        best_lag=result["best_lag"],
        p_value=result["p_value"],
        uncorrected_p_value=result.get("uncorrected_p_value"),
        n_tests=result.get("n_tests", 1),
        significant_at_05=result["significant_at_05"],
        by_lag=[GrangerLag(**r) for r in result["by_lag"]],
        warnings=result["warnings"],
    )


def analyze_tail_dependence(
    input_data: TailDependenceInput,
) -> TailDependenceResult:
    """
    Whether two assets move together IN THE TAIL, which is the only regime a
    diversification claim has to survive.

    A full-sample correlation of 0.3 is perfectly compatible with two assets
    that are independent day to day and fall together every time it matters.
    This measures the conditional probability directly.

    Read `n_tail_observations` alongside the estimate. At a 1% quantile on a
    year of data that is two or three points, and an estimate from three
    points has a confidence interval covering most of [0, 1].
    """
    from standard_quant_tools.analysis.structure import tail_dependence as _tail

    logger.debug("[analyze_tail_dependence] %s vs %s", input_data.x, input_data.y)
    x = _price_series(
        input_data.x, input_data.start_date, input_data.end_date, "returns"
    )
    y = _price_series(
        input_data.y, input_data.start_date, input_data.end_date, "returns"
    )
    result = _tail(x, y, quantile=input_data.quantile)
    return TailDependenceResult(x=input_data.x, y=input_data.y, **result)


def run_stationarity_tests(input_data: StationarityInput) -> StationarityResult:
    """
    ADF, KPSS and the variance ratio, with the four-way verdict spelled out.

    ADF and KPSS have OPPOSITE nulls, which is the whole reason to run both.
    Failing to reject ADF is not evidence of a unit root — it is a failure
    to find evidence against one, and on a few hundred observations that
    happens to genuinely mean-reverting series routinely. The verdict
    separates "the data says non-stationary" from "the data says nothing",
    and `inconclusive` is a statement about the sample size.

    `contradictory` — both rejecting — usually means a structural break or
    changing volatility rather than either answer. Run detect_change_points
    before trusting either.
    """
    from standard_quant_tools.analysis.stationarity import (
        run_stationarity_tests as _tests,
    )

    logger.debug("[run_stationarity_tests] %s", input_data.symbol)
    series = _price_series(
        input_data.symbol, input_data.start_date, input_data.end_date, input_data.on
    )
    result = _tests(series, lags=input_data.lags)
    return StationarityResult(
        symbol=input_data.symbol,
        n_observations=result["n_observations"],
        adf_statistic=result["adf_statistic"],
        adf_critical_5pct=result["adf_critical_5pct"],
        adf_rejects_unit_root=result["adf_rejects_unit_root"],
        kpss_statistic=result["kpss_statistic"],
        kpss_critical_5pct=result["kpss_critical_5pct"],
        kpss_rejects_stationarity=result["kpss_rejects_stationarity"],
        variance_ratios=[VarianceRatio(**v) for v in result["variance_ratios"]],
        verdict=result["verdict"],
        detail=result["detail"],
        warnings=result["warnings"],
    )


def detect_regimes(input_data: RegimeDetectionInput) -> RegimeDetectionResult:
    """
    Label each observation with a volatility regime.

    A Gaussian MIXTURE rather than a hidden Markov model, and the difference
    matters: a mixture treats observations as independent draws, so it will
    flip between regimes on a single observation where an HMM's transition
    matrix would smooth. `persistence` reports how often it does — below
    about 0.8 the labels are describing noise rather than regimes.

    Regimes come back sorted by volatility, so regime 0 is always the calm
    one. Without that the labels permute between runs and every downstream
    comparison is meaningless.
    """
    from standard_quant_tools.analysis.stationarity import detect_regimes as _regimes

    logger.debug("[detect_regimes] %s", input_data.symbol)
    series = _price_series(
        input_data.symbol, input_data.start_date, input_data.end_date, "returns"
    )
    result = _regimes(series, n_regimes=input_data.n_regimes, seed=input_data.seed)
    return RegimeDetectionResult(
        symbol=input_data.symbol,
        n_regimes=result["n_regimes"],
        regimes=[Regime(**g) for g in result["regimes"]],
        current_regime=result["current_regime"],
        persistence=result["persistence"],
        n_switches=result["n_switches"],
        warnings=result["warnings"],
    )
