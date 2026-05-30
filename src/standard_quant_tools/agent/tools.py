"""
Agent-callable tool functions — designed for LLM function calling.
All inputs/outputs use Pydantic models for clean JSON serialization.
"""

import datetime
from typing import Any, Dict, List

import pandas as pd

from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.indicators.trend import sma, ema, macd, adx, williams_r
from standard_quant_tools.indicators.momentum import rsi, stochastic_oscillator
from standard_quant_tools.indicators.volatility import bollinger_bands, atr
from standard_quant_tools.indicators.volume import obv, vwap
from standard_quant_tools.backtest.engine import run_strategy
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY
from standard_quant_tools.analysis.regression import calculate_beta
from standard_quant_tools.analysis.multi_factor import multi_factor_regression, rolling_factor_loadings
from standard_quant_tools.analysis.cointegration import cointegration_test, compute_spread, spread_zscore
from standard_quant_tools.analysis.pca import pca_returns, factor_contributions
from standard_quant_tools.analysis.hurst import hurst_exponent, rolling_hurst
from standard_quant_tools.metrics.risk_metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown,
    var_historical, cvar, information_ratio,
)
from standard_quant_tools.portfolio.portfolio import portfolio_metrics, fetch_returns_sync
from standard_quant_tools.screener.screener import screen_stocks
from standard_quant_tools.agent.models import (
    BacktestInput, BacktestResult, Trade,
    AnalysisInput, AnalysisResult,
    TechnicalInput, TechnicalResult,
    PortfolioInput, PortfolioResult,
    ScreenerInput, ScreenerResult,
    FactorRegressionInput, FactorRegressionResult,
    CointegrationInput, CointegrationResult,
    PCAInput, PCAResult,
    HurstInput, HurstResult,
)


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


# ──────────────────────────────────────────────────────────────────
# Backtesting Tools
# ──────────────────────────────────────────────────────────────────

def _run_backtest(
    input_data: BacktestInput,
    df: pd.DataFrame,
    signal_series: pd.Series,
) -> BacktestResult:
    """Shared backtest execution used by all strategy-specific tools."""
    results = run_strategy(
        df, signal_series, input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        include_trade_log=True,
    )

    trade_log_raw = results.get("trade_log", pd.DataFrame())
    trades = None
    if isinstance(trade_log_raw, pd.DataFrame) and not trade_log_raw.empty:
        trades = [
            Trade(
                entry_date=str(row["entry_date"]),
                exit_date=str(row["exit_date"]),
                direction=str(row["direction"]),
                entry_price=float(row["entry_price"]),
                exit_price=float(row["exit_price"]),
                return_pct=float(row["return_pct"]),
            )
            for _, row in trade_log_raw.iterrows()
        ]

    return BacktestResult(
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
    )


def run_sma_backtest(input_data: BacktestInput) -> BacktestResult:
    """SMA crossover backtest: long when fast SMA > slow SMA."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)
    signals = STRATEGY_REGISTRY["sma_crossover"](df, **input_data.parameters)
    return _run_backtest(input_data, df, signals)


def run_rsi_backtest(input_data: BacktestInput) -> BacktestResult:
    """RSI mean-reversion backtest: enter long at oversold, exit at overbought."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)
    signals = STRATEGY_REGISTRY["rsi_mean_reversion"](df, **input_data.parameters)
    return _run_backtest(input_data, df, signals)


def run_macd_backtest(input_data: BacktestInput) -> BacktestResult:
    """MACD crossover backtest: long when MACD line crosses above signal line."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)
    signals = STRATEGY_REGISTRY["macd_crossover"](df, **input_data.parameters)
    return _run_backtest(input_data, df, signals)


def run_bollinger_backtest(input_data: BacktestInput) -> BacktestResult:
    """Bollinger Band mean-reversion: enter at lower band, exit at middle band."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)
    signals = STRATEGY_REGISTRY["bollinger_reversion"](df, **input_data.parameters)
    return _run_backtest(input_data, df, signals)


# ──────────────────────────────────────────────────────────────────
# Risk Analysis Tool
# ──────────────────────────────────────────────────────────────────

def analyze_stock_risk(input_data: AnalysisInput) -> AnalysisResult:
    """Full risk profile: alpha, beta, Sharpe, Sortino, VaR, CVaR, Information Ratio."""
    provider = DataFactory.get_provider()
    end = datetime.datetime.now()
    start = _parse_period(input_data.period)

    asset_df = provider.get_ohlcv(input_data.symbol, start, end)
    bench_df = provider.get_ohlcv(input_data.benchmark, start, end)

    asset_ret = asset_df["Close"].pct_change().dropna()
    bench_ret = bench_df["Close"].pct_change().dropna()

    beta_metrics = calculate_beta(asset_ret, bench_ret)
    equity_curve = (1 + asset_ret).cumprod()

    return AnalysisResult(
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


# ──────────────────────────────────────────────────────────────────
# Technical Analysis Tool
# ──────────────────────────────────────────────────────────────────

def get_technical_analysis(input_data: TechnicalInput) -> TechnicalResult:
    """
    Run a configurable set of indicators and return the last bar's values
    plus simple directional signals.
    """
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    last_close = float(close.iloc[-1])

    last_vals: Dict[str, Any] = {"close": round(last_close, 4)}
    signals: Dict[str, Any] = {}
    requested = [ind.lower() for ind in input_data.indicators]

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
        r = rsi(close, 14).dropna()
        if not r.empty:
            last_rsi = float(r.iloc[-1])
            last_vals["rsi_14"] = round(last_rsi, 2)
            signals["rsi_oversold"] = last_rsi < 30
            signals["rsi_overbought"] = last_rsi > 70

    if "stochastic" in requested:
        stoch = stochastic_oscillator(high, low, close)
        k = float(stoch["Stoch_K"].dropna().iloc[-1])
        d = float(stoch["Stoch_D"].dropna().iloc[-1])
        last_vals["stoch_k"] = round(k, 2)
        last_vals["stoch_d"] = round(d, 2)
        signals["stoch_oversold"] = k < 20 and d < 20

    if "bollinger" in requested:
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
        adx_df = adx(high, low, close).dropna()
        if not adx_df.empty:
            last_vals["adx"] = round(float(adx_df["ADX"].iloc[-1]), 2)
            last_vals["di_plus"] = round(float(adx_df["DI_Plus"].iloc[-1]), 2)
            last_vals["di_minus"] = round(float(adx_df["DI_Minus"].iloc[-1]), 2)
            signals["strong_trend"] = float(adx_df["ADX"].iloc[-1]) > 25
            signals["bullish_di"] = (
                float(adx_df["DI_Plus"].iloc[-1]) > float(adx_df["DI_Minus"].iloc[-1])
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

    metrics = portfolio_metrics(aligned_returns, input_data.weights, benchmark_returns=aligned_bench)

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


# ──────────────────────────────────────────────────────────────────
# Screener Tool
# ──────────────────────────────────────────────────────────────────

def run_screener(input_data: ScreenerInput) -> ScreenerResult:
    """Screen a universe of tickers against fundamental and technical filters."""
    result_df = screen_stocks(
        input_data.tickers,
        input_data.filters,
        start_date=input_data.start_date,
        end_date=input_data.end_date,
        sort_by=input_data.sort_by,
        ascending=input_data.ascending,
    )

    if result_df.empty:
        return ScreenerResult(num_passed=0, tickers_passed=[], results=[])

    records = result_df.reset_index().to_dict(orient="records")
    return ScreenerResult(
        num_passed=len(result_df),
        tickers_passed=list(result_df.index),
        results=records,
    )


# ──────────────────────────────────────────────────────────────────
# Factor Regression Tool
# ──────────────────────────────────────────────────────────────────

def run_factor_regression(input_data: FactorRegressionInput) -> FactorRegressionResult:
    """OLS multi-factor regression: alpha, loadings, t-stats, p-values, R²."""
    provider = DataFactory.get_provider()
    asset_df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)
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
        rolling = rolling_factor_loadings(asset_rets, factors, window=input_data.rolling_window)
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
        t_stats={k: round(float(v), 4) if not (v != v) else 0.0
                 for k, v in result["t_stats"].items()},
        p_values={k: round(float(v), 4) if not (v != v) else 1.0
                  for k, v in result["p_values"].items()},
        r_squared=round(float(result["r_squared"]), 4),
        adj_r_squared=round(float(result["adj_r_squared"]), 4)
        if not (result["adj_r_squared"] != result["adj_r_squared"]) else 0.0,
        n_obs=result["n_obs"],
        rolling_alpha_tail=rolling_alpha_tail,
        rolling_loadings_tail=rolling_loadings_tail,
    )


# ──────────────────────────────────────────────────────────────────
# Cointegration Tool
# ──────────────────────────────────────────────────────────────────

def run_cointegration_test(input_data: CointegrationInput) -> CointegrationResult:
    """Engle-Granger cointegration test with hedge ratio, half-life, and a z-score signal."""
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
        critical_values={k: round(float(v), 4) for k, v in result["critical_values"].items()},
        spread_mean=round(float(spread.mean()), 6),
        spread_std=round(float(spread.std()), 6),
        current_zscore=current_z,
        signal=signal,
        n_obs=result["n_obs"],
    )


# ──────────────────────────────────────────────────────────────────
# PCA Tool
# ──────────────────────────────────────────────────────────────────

def run_pca_analysis(input_data: PCAInput) -> PCAResult:
    """PCA on multi-asset returns: explained variance, loadings, per-asset factor contributions."""
    provider = DataFactory.get_provider()
    returns = pd.DataFrame({
        t: provider.get_ohlcv(t, input_data.start_date, input_data.end_date)["Close"].pct_change()
        for t in input_data.tickers
    }).dropna()

    result = pca_returns(returns, n_components=input_data.n_components)
    contrib = factor_contributions(returns, n_components=input_data.n_components)

    evr = {k: round(float(v), 4) for k, v in result["explained_variance_ratio"].items()}
    cumvar = {k: round(float(v), 4) for k, v in result["cumulative_variance_ratio"].items()}

    loadings_dict = {
        pc: {t: round(float(result["loadings"].loc[t, pc]), 4) for t in input_data.tickers}
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


# ──────────────────────────────────────────────────────────────────
# Hurst Tool
# ──────────────────────────────────────────────────────────────────

def run_hurst_analysis(input_data: HurstInput) -> HurstResult:
    """Hurst exponent via DFA or R/S. Optionally includes rolling regime breakdown."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)
    returns = df["Close"].pct_change().dropna()

    result = hurst_exponent(returns, method=input_data.method)

    rolling_current = None
    rolling_regime_fractions = None
    if input_data.rolling_window:
        rolling = rolling_hurst(returns, window=input_data.rolling_window, method=input_data.method)
        valid = rolling.dropna()
        if not valid.empty:
            rolling_current = round(float(valid.iloc[-1]), 4)
            total = len(valid)
            rolling_regime_fractions = {
                "trending": round(float((valid > 0.55).sum() / total), 3),
                "random_walk": round(float(((valid >= 0.45) & (valid <= 0.55)).sum() / total), 3),
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


# ──────────────────────────────────────────────────────────────────
# Tool Registry for LLM Function Calling
# ──────────────────────────────────────────────────────────────────

def get_agent_tools() -> List[Dict[str, Any]]:
    """
    Returns tool definitions formatted for OpenAI / Anthropic function calling.
    All tools have Pydantic-derived schemas — no manual JSON authoring required.
    """
    tool_defs = [
        ("run_sma_backtest", "SMA crossover backtest.", BacktestInput),
        ("run_rsi_backtest", "RSI mean-reversion backtest.", BacktestInput),
        ("run_macd_backtest", "MACD crossover backtest.", BacktestInput),
        ("run_bollinger_backtest", "Bollinger Band mean-reversion backtest.", BacktestInput),
        ("analyze_stock_risk", "Full risk analysis: alpha, beta, Sharpe, VaR, CVaR.", AnalysisInput),
        ("get_technical_analysis", "Compute configurable technical indicators.", TechnicalInput),
        ("get_portfolio_analysis", "Multi-asset portfolio metrics.", PortfolioInput),
        ("run_screener", "Filter a stock universe by fundamental and technical criteria.", ScreenerInput),
        ("run_factor_regression", "Multi-factor OLS regression: alpha, loadings, t-stats, p-values, R².", FactorRegressionInput),
        ("run_cointegration_test", "Engle-Granger cointegration: hedge ratio, half-life, spread z-score signal.", CointegrationInput),
        ("run_pca_analysis", "PCA on multi-asset returns: explained variance, loadings, factor contributions.", PCAInput),
        ("run_hurst_analysis", "Hurst exponent (DFA/R-S): regime classification and optional rolling breakdown.", HurstInput),
    ]

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
