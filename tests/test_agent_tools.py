"""Tests for all 8 agent tools (mocked data provider)."""

import pandas as pd
import pytest

from standard_quant_tools.agent.models import (
    AnalysisInput, BacktestInput, PortfolioInput,
    ScreenerInput, TechnicalInput,
)
from standard_quant_tools.agent.tools import (
    analyze_stock_risk,
    get_agent_tools,
    get_portfolio_analysis,
    get_technical_analysis,
    run_bollinger_backtest,
    run_macd_backtest,
    run_rsi_backtest,
    run_screener,
    run_sma_backtest,
)


START, END = '2023-01-01', '2024-01-01'


class TestGetAgentTools:
    def test_returns_list_of_eight_tools(self):
        tools = get_agent_tools()
        assert len(tools) == 8

    def test_all_tools_have_correct_schema_keys(self):
        for tool in get_agent_tools():
            assert tool['type'] == 'function'
            assert 'name' in tool['function']
            assert 'description' in tool['function']
            assert 'parameters' in tool['function']

    def test_tool_names_are_correct(self):
        names = {t['function']['name'] for t in get_agent_tools()}
        expected = {
            'run_sma_backtest', 'run_rsi_backtest', 'run_macd_backtest',
            'run_bollinger_backtest', 'analyze_stock_risk',
            'get_technical_analysis', 'get_portfolio_analysis', 'run_screener',
        }
        assert names == expected

    def test_parameters_are_valid_json_schema(self):
        for tool in get_agent_tools():
            schema = tool['function']['parameters']
            assert schema.get('type') == 'object'
            assert 'properties' in schema


class TestSMABacktest:
    def test_returns_backtest_result(self, patched_factory):
        inp = BacktestInput(
            symbol='AAPL', start_date=START, end_date=END,
            strategy_type='sma_crossover',
            parameters={'fast_period': 10, 'slow_period': 30},
        )
        result = run_sma_backtest(inp)
        assert result.total_return is not None
        assert result.num_trades >= 0
        assert 0.0 <= result.win_rate <= 1.0

    def test_equity_curve_has_values(self, patched_factory):
        inp = BacktestInput(
            symbol='AAPL', start_date=START, end_date=END,
            strategy_type='sma_crossover',
        )
        result = run_sma_backtest(inp)
        assert len(result.equity_curve) > 0

    def test_all_required_fields_populated(self, patched_factory):
        inp = BacktestInput(
            symbol='AAPL', start_date=START, end_date=END,
            strategy_type='sma_crossover',
        )
        result = run_sma_backtest(inp)
        assert result.final_equity > 0
        assert result.max_drawdown <= 0
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.calmar_ratio, float)


class TestRSIBacktest:
    def test_returns_backtest_result(self, patched_factory):
        inp = BacktestInput(
            symbol='AAPL', start_date=START, end_date=END,
            strategy_type='rsi_mean_reversion',
            parameters={'period': 14, 'oversold': 30, 'overbought': 70},
        )
        result = run_rsi_backtest(inp)
        assert isinstance(result.total_return, float)

    def test_win_rate_bounded(self, patched_factory):
        inp = BacktestInput(
            symbol='AAPL', start_date=START, end_date=END,
            strategy_type='rsi_mean_reversion',
        )
        result = run_rsi_backtest(inp)
        assert 0.0 <= result.win_rate <= 1.0


class TestMACDBacktest:
    def test_returns_backtest_result(self, patched_factory):
        inp = BacktestInput(
            symbol='AAPL', start_date=START, end_date=END,
            strategy_type='macd_crossover',
            parameters={'fast': 12, 'slow': 26, 'signal': 9},
        )
        result = run_macd_backtest(inp)
        assert isinstance(result.total_return, float)


class TestBollingerBacktest:
    def test_returns_backtest_result(self, patched_factory):
        inp = BacktestInput(
            symbol='AAPL', start_date=START, end_date=END,
            strategy_type='bollinger_reversion',
            parameters={'period': 20, 'num_std': 2.0},
        )
        result = run_bollinger_backtest(inp)
        assert isinstance(result.total_return, float)


class TestAnalyzeStockRisk:
    def test_returns_analysis_result(self, patched_factory):
        inp = AnalysisInput(symbol='AAPL', benchmark='SPY', period='1y')
        result = analyze_stock_risk(inp)
        assert result.symbol == 'AAPL'
        assert result.benchmark == 'SPY'

    def test_all_fields_are_floats(self, patched_factory):
        inp = AnalysisInput(symbol='AAPL', benchmark='SPY', period='1y')
        result = analyze_stock_risk(inp)
        for field in ('alpha', 'beta', 'r_squared', 'sharpe_ratio',
                      'sortino_ratio', 'max_drawdown', 'var_95', 'cvar_95',
                      'information_ratio'):
            assert isinstance(getattr(result, field), float), f"{field} is not float"

    def test_r_squared_bounded(self, patched_factory):
        inp = AnalysisInput(symbol='AAPL', benchmark='SPY', period='1y')
        result = analyze_stock_risk(inp)
        assert 0.0 <= result.r_squared <= 1.0

    def test_var_less_than_cvar(self, patched_factory):
        inp = AnalysisInput(symbol='AAPL', benchmark='SPY', period='1y')
        result = analyze_stock_risk(inp)
        assert result.cvar_95 >= result.var_95


class TestGetTechnicalAnalysis:
    def test_returns_technical_result(self, patched_factory):
        inp = TechnicalInput(
            symbol='AAPL', start_date=START, end_date=END,
            indicators=['rsi', 'macd', 'bollinger', 'sma'],
        )
        result = get_technical_analysis(inp)
        assert result.symbol == 'AAPL'
        assert result.last_close > 0

    def test_rsi_indicator_in_last_values(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['rsi'])
        result = get_technical_analysis(inp)
        assert 'rsi_14' in result.last_values
        assert 0 <= result.last_values['rsi_14'] <= 100

    def test_macd_signal_in_signals(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['macd'])
        result = get_technical_analysis(inp)
        assert 'macd_bullish' in result.signals
        assert isinstance(result.signals['macd_bullish'], bool)

    def test_bollinger_values_in_last_values(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['bollinger'])
        result = get_technical_analysis(inp)
        assert 'bb_upper' in result.last_values
        assert result.last_values['bb_upper'] > result.last_values['bb_lower']

    def test_adx_signal_in_signals(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['adx'])
        result = get_technical_analysis(inp)
        assert 'strong_trend' in result.signals
        assert 'adx' in result.last_values

    def test_obv_in_last_values(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['obv'])
        result = get_technical_analysis(inp)
        assert 'obv' in result.last_values


class TestGetPortfolioAnalysis:
    def test_returns_portfolio_result(self, patched_factory):
        inp = PortfolioInput(
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            weights=[1/3, 1/3, 1/3],
            start_date=START, end_date=END,
        )
        result = get_portfolio_analysis(inp)
        assert result.tickers == ['AAPL', 'MSFT', 'GOOGL']

    def test_all_metric_fields_populated(self, patched_factory):
        inp = PortfolioInput(
            tickers=['AAPL', 'MSFT'],
            weights=[0.6, 0.4],
            start_date=START, end_date=END,
        )
        result = get_portfolio_analysis(inp)
        for field in ('annualized_return', 'annualized_volatility', 'sharpe_ratio',
                      'max_drawdown', 'var_95', 'cvar_95'):
            assert isinstance(getattr(result, field), float), f"{field} not float"

    def test_correlation_matrix_in_result(self, patched_factory):
        inp = PortfolioInput(
            tickers=['AAPL', 'MSFT'],
            weights=[0.5, 0.5],
            start_date=START, end_date=END,
        )
        result = get_portfolio_analysis(inp)
        assert isinstance(result.correlation_matrix, dict)


class TestRunScreener:
    def test_returns_screener_result(self, patched_factory):
        inp = ScreenerInput(tickers=['AAPL', 'MSFT'], filters={'pe_ratio_max': 35.0})
        result = run_screener(inp)
        assert result.num_passed >= 0
        assert isinstance(result.tickers_passed, list)
        assert isinstance(result.results, list)

    def test_num_passed_matches_tickers_passed(self, patched_factory):
        inp = ScreenerInput(tickers=['AAPL', 'MSFT', 'GOOGL'], filters={})
        result = run_screener(inp)
        assert result.num_passed == len(result.tickers_passed)
