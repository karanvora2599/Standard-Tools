from typing import List, Dict, Any
from data.factory import DataFactory
from indicators.trend import sma
from backtest.engine import run_strategy
from analysis.regression import calculate_beta
from metrics.risk_metrics import sharpe_ratio, max_drawdown
from agent.models import BacktestInput, BacktestResult, AnalysisInput, AnalysisResult
import pandas as pd
import numpy as np

def run_sma_backtest(input_data: BacktestInput) -> BacktestResult:
    """
    Agent tool to run a simple SMA crossover backtest.
    """
    provider = DataFactory.get_provider()
    df = provider.get_ohlcv(input_data.symbol, input_data.start_date, input_data.end_date)
    
    fast_period = input_data.parameters.get('fast_period', 10)
    slow_period = input_data.parameters.get('slow_period', 30)
    
    # Calculate indicators
    fast_ma = sma(df['Close'], fast_period)
    slow_ma = sma(df['Close'], slow_period)
    
    # Generate signals: 1 where Fast > Slow
    signals = pd.Series(np.where(fast_ma > slow_ma, 1, 0), index=df.index)
    
    # Run backtest
    results = run_strategy(df, signals, input_data.initial_capital)
    
    # Calculate Max Drawdown from equity curve
    mdd = max_drawdown(results['equity_curve'])
    
    return BacktestResult(
        total_return=results['total_return'],
        annualized_volatility=results['annualized_volatility'],
        max_drawdown=mdd,
        final_equity=results['final_equity'],
        equity_curve=results['equity_curve'].tolist()
    )

def analyze_stock_risk(input_data: AnalysisInput) -> AnalysisResult:
    """
    Agent tool to analyze stock risk against a benchmark.
    """
    provider = DataFactory.get_provider()
    
    # Fetch data (defaulting to last 1 year usually if period handled simpler)
    # For now, let's just fetch 252 days
    import datetime
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=365)
    
    asset_df = provider.get_ohlcv(input_data.symbol, start, end)
    bench_df = provider.get_ohlcv(input_data.benchmark, start, end)
    
    asset_ret = asset_df['Close'].pct_change().dropna()
    bench_ret = bench_df['Close'].pct_change().dropna()
    
    beta_metrics = calculate_beta(asset_ret, bench_ret)
    sr = sharpe_ratio(asset_ret)
    
    return AnalysisResult(
        symbol=input_data.symbol,
        benchmark=input_data.benchmark,
        alpha=beta_metrics['alpha'],
        beta=beta_metrics['beta'],
        r_squared=beta_metrics['r_squared'],
        sharpe_ratio=sr
    )

def get_agent_tools() -> List[Dict[str, Any]]:
    """
    Returns a list of tools formatted for OpenAI function calling (or similar).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "run_sma_backtest",
                "description": "Run a simple SMA crossover backtest on a stock.",
                "parameters": BacktestInput.model_json_schema()
            }
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_stock_risk",
                "description": "Calculate Alpha, Beta, and Sharpe Ratio for a stock against a benchmark.",
                "parameters": AnalysisInput.model_json_schema()
            }
        }
    ]
