"""
Agent-callable tool functions — designed for LLM function calling.
All inputs/outputs use Pydantic models for clean JSON serialization.
"""

import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.indicators.trend import sma, ema, macd, adx, williams_r
from standard_quant_tools.indicators.momentum import rsi, stochastic_oscillator
from standard_quant_tools.indicators.volatility import bollinger_bands, atr
from standard_quant_tools.indicators.volume import obv, vwap
from standard_quant_tools.backtest.engine import run_strategy
from standard_quant_tools.analysis.regression import calculate_beta
from standard_quant_tools.metrics.risk_metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown,
    var_historical, cvar, information_ratio
)
from standard_quant_tools.portfolio.portfolio import portfolio_metrics, fetch_returns_sync
from standard_quant_tools.screener.screener import screen_stocks
from standard_quant_tools.agent.models import (
    BacktestInput, BacktestResult, Trade,
    AnalysisInput, AnalysisResult,
    TechnicalInput, TechnicalResult,
    PortfolioInput, PortfolioResult,
    ScreenerInput, ScreenerResult,
)


def _parse_period(period: str) -> datetime.datetime:
    """Convert period string ('1y', '6mo', '2y') to a start datetime."""
    now = datetime.datetime.now()
    unit = period[-2:] if period.endswith('mo') else period[-1]
    num = int(period[: -2 if unit == 'mo' else -1])
    if unit == 'mo':
        return now - datetime.timedelta(days=num * 30)
    if unit == 'y':
        return now - datetime.timedelta(days=num * 365)
    if unit == 'd':
        return now - datetime.timedelta(days=num)
    return now - datetime.timedelta(days=365)


# ──────────────────────────────────────────────────────────────────
# Backtesting Tools
# ──────────────────────────────────────────────────────────────────

def _run_backtest(input_data: BacktestInput, signal_series: pd.Series) -> BacktestResult:
    """Shared backtest execution used by all strategy-specific tools."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)

    results = run_strategy(
        df, signal_series, input_data.initial_capital,
        commission_pct=input_data.commission_pct,
        slippage_pct=input_data.slippage_pct,
        include_trade_log=True,
    )

    trade_log_raw = results.get('trade_log', pd.DataFrame())
    trades = None
    if isinstance(trade_log_raw, pd.DataFrame) and not trade_log_raw.empty:
        trades = [
            Trade(
                entry_date=str(row['entry_date']),
                exit_date=str(row['exit_date']),
                direction=str(row['direction']),
                entry_price=float(row['entry_price']),
                exit_price=float(row['exit_price']),
                return_pct=float(row['return_pct']),
            )
            for _, row in trade_log_raw.iterrows()
        ]

    return BacktestResult(
        total_return=results['total_return'],
        annualized_volatility=results['annualized_volatility'],
        sharpe_ratio=results['sharpe_ratio'],
        sortino_ratio=results['sortino_ratio'],
        max_drawdown=results['max_drawdown'],
        calmar_ratio=results['calmar_ratio'],
        win_rate=results['win_rate'],
        profit_factor=results['profit_factor'],
        num_trades=results['num_trades'],
        avg_trade_return_pct=results['avg_trade_return_pct'],
        final_equity=results['final_equity'],
        equity_curve=results['equity_curve'].tolist(),
        trade_log=trades,
    )


def run_sma_backtest(input_data: BacktestInput) -> BacktestResult:
    """SMA crossover backtest: long when fast SMA > slow SMA."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)
    fast = int(input_data.parameters.get('fast_period', 10))
    slow = int(input_data.parameters.get('slow_period', 30))
    signals = pd.Series(
        np.where(sma(df['Close'], fast) > sma(df['Close'], slow), 1, 0),
        index=df.index,
    )
    return _run_backtest(input_data, signals)


def run_rsi_backtest(input_data: BacktestInput) -> BacktestResult:
    """
    RSI mean-reversion backtest.
    Enter long when RSI < oversold; exit (flat) when RSI > overbought.
    """
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)
    period = int(input_data.parameters.get('period', 14))
    oversold = float(input_data.parameters.get('oversold', 30))
    overbought = float(input_data.parameters.get('overbought', 70))

    rsi_vals = rsi(df['Close'], period)

    # Vectorized state machine using cumulative max trick
    raw_signal = pd.Series(0, index=df.index, dtype=float)
    raw_signal[rsi_vals < oversold] = 1
    raw_signal[rsi_vals > overbought] = 0

    # Forward-fill the long signal between entry and exit
    in_position = False
    values = raw_signal.to_numpy(dtype=float)
    rsi_arr = rsi_vals.to_numpy(dtype=float)
    for i in range(len(values)):
        if np.isnan(rsi_arr[i]):
            continue
        if not in_position and rsi_arr[i] < oversold:
            in_position = True
        elif in_position and rsi_arr[i] > overbought:
            in_position = False
        values[i] = 1.0 if in_position else 0.0

    signals = pd.Series(values, index=df.index)
    return _run_backtest(input_data, signals)


def run_macd_backtest(input_data: BacktestInput) -> BacktestResult:
    """MACD crossover backtest: long when MACD line crosses above signal line."""
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)
    fast = int(input_data.parameters.get('fast', 12))
    slow = int(input_data.parameters.get('slow', 26))
    signal_period = int(input_data.parameters.get('signal', 9))

    macd_df = macd(df['Close'], fast, slow, signal_period)
    signals = pd.Series(
        np.where(macd_df['MACD'] > macd_df['Signal'], 1, 0),
        index=df.index,
    )
    return _run_backtest(input_data, signals)


def run_bollinger_backtest(input_data: BacktestInput) -> BacktestResult:
    """
    Bollinger Band mean-reversion backtest.
    Enter long when price touches lower band; exit when it reaches the middle band.
    """
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)
    period = int(input_data.parameters.get('period', 20))
    num_std = float(input_data.parameters.get('num_std', 2.0))

    bb = bollinger_bands(df['Close'], period, num_std)
    close = df['Close']

    in_position = False
    values = np.zeros(len(close))
    close_arr = close.to_numpy(dtype=float)
    lower_arr = bb['BB_Lower'].to_numpy(dtype=float)
    middle_arr = bb['BB_Middle'].to_numpy(dtype=float)

    for i in range(len(close_arr)):
        if np.isnan(lower_arr[i]):
            continue
        if not in_position and close_arr[i] <= lower_arr[i]:
            in_position = True
        elif in_position and close_arr[i] >= middle_arr[i]:
            in_position = False
        values[i] = 1.0 if in_position else 0.0

    signals = pd.Series(values, index=df.index)
    return _run_backtest(input_data, signals)


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

    asset_ret = asset_df['Close'].pct_change().dropna()
    bench_ret = bench_df['Close'].pct_change().dropna()

    beta_metrics = calculate_beta(asset_ret, bench_ret)
    equity_curve = (1 + asset_ret).cumprod()

    return AnalysisResult(
        symbol=input_data.symbol,
        benchmark=input_data.benchmark,
        alpha=round(beta_metrics['alpha'], 6),
        beta=round(beta_metrics['beta'], 4),
        r_squared=round(beta_metrics['r_squared'], 4),
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
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    last_close = float(close.iloc[-1])

    last_vals: Dict[str, Any] = {'close': round(last_close, 4)}
    signals: Dict[str, Any] = {}
    requested = [ind.lower() for ind in input_data.indicators]

    if 'sma' in requested:
        for p in (20, 50, 200):
            s = sma(close, p).dropna()
            if not s.empty:
                v = float(s.iloc[-1])
                last_vals[f'sma_{p}'] = round(v, 4)
                signals[f'price_above_sma_{p}'] = last_close > v

    if 'ema' in requested:
        for p in (12, 26):
            e = ema(close, p).dropna()
            if not e.empty:
                last_vals[f'ema_{p}'] = round(float(e.iloc[-1]), 4)

    if 'macd' in requested:
        m = macd(close)
        last_macd = float(m['MACD'].dropna().iloc[-1])
        last_sig = float(m['Signal'].dropna().iloc[-1])
        last_hist = float(m['Histogram'].dropna().iloc[-1])
        last_vals['macd'] = round(last_macd, 4)
        last_vals['macd_signal'] = round(last_sig, 4)
        last_vals['macd_histogram'] = round(last_hist, 4)
        signals['macd_bullish'] = last_macd > last_sig

    if 'rsi' in requested:
        r = rsi(close, 14).dropna()
        if not r.empty:
            last_rsi = float(r.iloc[-1])
            last_vals['rsi_14'] = round(last_rsi, 2)
            signals['rsi_oversold'] = last_rsi < 30
            signals['rsi_overbought'] = last_rsi > 70

    if 'stochastic' in requested:
        stoch = stochastic_oscillator(high, low, close)
        k = float(stoch['Stoch_K'].dropna().iloc[-1])
        d = float(stoch['Stoch_D'].dropna().iloc[-1])
        last_vals['stoch_k'] = round(k, 2)
        last_vals['stoch_d'] = round(d, 2)
        signals['stoch_oversold'] = k < 20 and d < 20

    if 'bollinger' in requested:
        bb = bollinger_bands(close)
        upper = float(bb['BB_Upper'].dropna().iloc[-1])
        middle = float(bb['BB_Middle'].dropna().iloc[-1])
        lower = float(bb['BB_Lower'].dropna().iloc[-1])
        last_vals['bb_upper'] = round(upper, 4)
        last_vals['bb_middle'] = round(middle, 4)
        last_vals['bb_lower'] = round(lower, 4)
        signals['price_near_lower_band'] = last_close <= lower * 1.01
        signals['price_near_upper_band'] = last_close >= upper * 0.99

    if 'atr' in requested:
        a = atr(high, low, close).dropna()
        if not a.empty:
            last_vals['atr_14'] = round(float(a.iloc[-1]), 4)

    if 'obv' in requested:
        o = obv(close, volume).dropna()
        if len(o) >= 2:
            last_vals['obv'] = int(float(o.iloc[-1]))
            signals['obv_rising'] = float(o.iloc[-1]) > float(o.iloc[-2])

    if 'vwap' in requested:
        v = vwap(high, low, close, volume)
        if not v.dropna().empty:
            last_vwap = float(v.dropna().iloc[-1])
            last_vals['vwap'] = round(last_vwap, 4)
            signals['price_above_vwap'] = last_close > last_vwap

    if 'adx' in requested:
        adx_df = adx(high, low, close).dropna()
        if not adx_df.empty:
            last_vals['adx'] = round(float(adx_df['ADX'].iloc[-1]), 2)
            last_vals['di_plus'] = round(float(adx_df['DI_Plus'].iloc[-1]), 2)
            last_vals['di_minus'] = round(float(adx_df['DI_Minus'].iloc[-1]), 2)
            signals['strong_trend'] = float(adx_df['ADX'].iloc[-1]) > 25
            signals['bullish_di'] = float(adx_df['DI_Plus'].iloc[-1]) > float(adx_df['DI_Minus'].iloc[-1])

    if 'williams_r' in requested:
        wr = williams_r(high, low, close).dropna()
        if not wr.empty:
            last_wr = float(wr.iloc[-1])
            last_vals['williams_r'] = round(last_wr, 2)
            signals['williams_r_oversold'] = last_wr < -80
            signals['williams_r_overbought'] = last_wr > -20

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
    bench_returns = bench_df_raw['Close'].pct_change().dropna()

    # Align returns_df with benchmark
    common_idx = returns_df.index.intersection(bench_returns.index)
    aligned_returns = returns_df.loc[common_idx]
    aligned_bench = bench_returns.loc[common_idx]

    metrics = portfolio_metrics(aligned_returns, input_data.weights, benchmark_returns=aligned_bench)

    return PortfolioResult(
        tickers=input_data.tickers,
        weights=input_data.weights,
        annualized_return=metrics['annualized_return'],
        annualized_volatility=metrics['annualized_volatility'],
        sharpe_ratio=metrics['sharpe_ratio'],
        sortino_ratio=metrics['sortino_ratio'],
        max_drawdown=metrics['max_drawdown'],
        calmar_ratio=metrics['calmar_ratio'],
        var_95=metrics['var_95'],
        cvar_95=metrics['cvar_95'],
        information_ratio=metrics.get('information_ratio', 0.0),
        total_return=metrics['total_return'],
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

    records = result_df.reset_index().to_dict(orient='records')
    return ScreenerResult(
        num_passed=len(result_df),
        tickers_passed=list(result_df.index),
        results=records,
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
        ('run_sma_backtest', 'SMA crossover backtest.', BacktestInput),
        ('run_rsi_backtest', 'RSI mean-reversion backtest.', BacktestInput),
        ('run_macd_backtest', 'MACD crossover backtest.', BacktestInput),
        ('run_bollinger_backtest', 'Bollinger Band mean-reversion backtest.', BacktestInput),
        ('analyze_stock_risk', 'Full risk analysis: alpha, beta, Sharpe, VaR, CVaR.', AnalysisInput),
        ('get_technical_analysis', 'Compute configurable technical indicators.', TechnicalInput),
        ('get_portfolio_analysis', 'Multi-asset portfolio metrics.', PortfolioInput),
        ('run_screener', 'Filter a stock universe by fundamental and technical criteria.', ScreenerInput),
    ]

    return [
        {
            'type': 'function',
            'function': {
                'name': name,
                'description': desc,
                'parameters': model.model_json_schema(),
            },
        }
        for name, desc, model in tool_defs
    ]
