"""Tests for the original 12 agent tools (mocked data provider)."""

import pandas as pd
import pytest

from standard_quant_tools.agent.models import (
    AnalysisInput, BacktestInput, CointegrationInput,
    FactorRegressionInput, HurstInput, PCAInput,
    PortfolioInput, ScreenerInput, TechnicalInput,
)
from standard_quant_tools.agent.tools import (
    analyze_stock_risk,
    get_agent_tools,
    get_portfolio_analysis,
    get_technical_analysis,
    run_bollinger_backtest,
    run_cointegration_test,
    run_factor_regression,
    run_hurst_analysis,
    run_macd_backtest,
    run_pca_analysis,
    run_rsi_backtest,
    run_screener,
    run_sma_backtest,
)


START, END = '2023-01-01', '2024-01-01'


class TestGetAgentTools:
    def test_returns_list_of_twenty_six_tools(self):
        tools = get_agent_tools()
        assert len(tools) == 26

    def test_all_tools_have_correct_schema_keys(self):
        for tool in get_agent_tools():
            assert tool['type'] == 'function'
            assert 'name' in tool['function']
            assert 'description' in tool['function']
            assert 'parameters' in tool['function']

    def test_original_tool_names_present(self):
        names = {t['function']['name'] for t in get_agent_tools()}
        original = {
            'run_sma_backtest', 'run_rsi_backtest', 'run_macd_backtest',
            'run_bollinger_backtest', 'run_buy_and_hold', 'compare_strategies',
            'analyze_stock_risk', 'get_technical_analysis', 'get_portfolio_analysis',
            'run_screener', 'run_factor_regression', 'run_cointegration_test',
            'run_pca_analysis', 'run_hurst_analysis',
        }
        assert original.issubset(names)

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


class TestRunFactorRegression:
    def test_returns_factor_regression_result(self, patched_factory):
        inp = FactorRegressionInput(
            symbol='AAPL',
            factor_tickers=['SPY', 'IWM'],
            factor_names=['mkt', 'smb'],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert result.symbol == 'AAPL'
        assert result.factors == ['mkt', 'smb']

    def test_defaults_factor_names_to_tickers(self, patched_factory):
        inp = FactorRegressionInput(
            symbol='AAPL',
            factor_tickers=['SPY', 'IWM'],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert result.factors == ['SPY', 'IWM']

    def test_alpha_is_float(self, patched_factory):
        inp = FactorRegressionInput(
            symbol='AAPL',
            factor_tickers=['SPY'],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert isinstance(result.alpha, float)

    def test_loadings_keys_match_factor_names(self, patched_factory):
        inp = FactorRegressionInput(
            symbol='AAPL',
            factor_tickers=['SPY', 'IWM'],
            factor_names=['mkt', 'smb'],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert set(result.loadings.keys()) == {'mkt', 'smb'}

    def test_t_stats_and_p_values_have_alpha_key(self, patched_factory):
        inp = FactorRegressionInput(
            symbol='AAPL',
            factor_tickers=['SPY'],
            factor_names=['mkt'],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert 'alpha' in result.t_stats
        assert 'alpha' in result.p_values

    def test_r_squared_bounded(self, patched_factory):
        inp = FactorRegressionInput(
            symbol='AAPL',
            factor_tickers=['SPY'],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert 0.0 <= result.r_squared <= 1.0

    def test_n_obs_positive(self, patched_factory):
        inp = FactorRegressionInput(
            symbol='AAPL',
            factor_tickers=['SPY'],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert result.n_obs > 0

    def test_rolling_tail_none_when_not_requested(self, patched_factory):
        inp = FactorRegressionInput(
            symbol='AAPL',
            factor_tickers=['SPY'],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert result.rolling_alpha_tail is None
        assert result.rolling_loadings_tail is None

    def test_rolling_tail_populated_when_requested(self, patched_factory):
        inp = FactorRegressionInput(
            symbol='AAPL',
            factor_tickers=['SPY'],
            factor_names=['mkt'],
            start_date=START,
            end_date=END,
            rolling_window=60,
        )
        result = run_factor_regression(inp)
        assert result.rolling_alpha_tail is not None
        assert result.rolling_loadings_tail is not None
        assert 'mkt' in result.rolling_loadings_tail


class TestRunCointegrationTest:
    def test_returns_cointegration_result(self, patched_factory):
        inp = CointegrationInput(
            symbol_a='KO', symbol_b='PEP',
            start_date=START, end_date=END,
        )
        result = run_cointegration_test(inp)
        assert result.symbol_a == 'KO'
        assert result.symbol_b == 'PEP'

    def test_cointegrated_is_bool(self, patched_factory):
        inp = CointegrationInput(
            symbol_a='KO', symbol_b='PEP',
            start_date=START, end_date=END,
        )
        result = run_cointegration_test(inp)
        assert isinstance(result.cointegrated, bool)

    def test_p_value_bounded(self, patched_factory):
        inp = CointegrationInput(
            symbol_a='KO', symbol_b='PEP',
            start_date=START, end_date=END,
        )
        result = run_cointegration_test(inp)
        assert 0.0 <= result.p_value <= 1.0

    def test_signal_is_valid_string(self, patched_factory):
        inp = CointegrationInput(
            symbol_a='KO', symbol_b='PEP',
            start_date=START, end_date=END,
        )
        result = run_cointegration_test(inp)
        assert result.signal in {'long_a_short_b', 'short_a_long_b', 'neutral'}

    def test_half_life_is_finite_float(self, patched_factory):
        inp = CointegrationInput(
            symbol_a='KO', symbol_b='PEP',
            start_date=START, end_date=END,
        )
        result = run_cointegration_test(inp)
        assert isinstance(result.half_life_days, float)
        assert result.half_life_days <= 9999.0

    def test_critical_values_have_three_levels(self, patched_factory):
        inp = CointegrationInput(
            symbol_a='KO', symbol_b='PEP',
            start_date=START, end_date=END,
        )
        result = run_cointegration_test(inp)
        assert set(result.critical_values.keys()) == {'1%', '5%', '10%'}

    def test_n_obs_positive(self, patched_factory):
        inp = CointegrationInput(
            symbol_a='KO', symbol_b='PEP',
            start_date=START, end_date=END,
        )
        result = run_cointegration_test(inp)
        assert result.n_obs > 0

    def test_hedge_ratio_is_float(self, patched_factory):
        inp = CointegrationInput(
            symbol_a='KO', symbol_b='PEP',
            start_date=START, end_date=END,
        )
        result = run_cointegration_test(inp)
        assert isinstance(result.hedge_ratio, float)


class TestRunPCAAnalysis:
    def test_returns_pca_result(self, patched_factory):
        inp = PCAInput(
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            start_date=START, end_date=END,
            n_components=2,
        )
        result = run_pca_analysis(inp)
        assert result.tickers == ['AAPL', 'MSFT', 'GOOGL']
        assert result.n_components == 2

    def test_explained_variance_ratio_keys(self, patched_factory):
        inp = PCAInput(
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            start_date=START, end_date=END,
            n_components=3,
        )
        result = run_pca_analysis(inp)
        assert set(result.explained_variance_ratio.keys()) == {'PC1', 'PC2', 'PC3'}

    def test_evr_values_sum_to_one(self, patched_factory):
        inp = PCAInput(
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            start_date=START, end_date=END,
            n_components=3,
        )
        result = run_pca_analysis(inp)
        total = sum(result.explained_variance_ratio.values())
        assert abs(total - 1.0) < 0.01

    def test_cumulative_variance_ends_near_one(self, patched_factory):
        inp = PCAInput(
            tickers=['AAPL', 'MSFT', 'GOOGL'],
            start_date=START, end_date=END,
            n_components=3,
        )
        result = run_pca_analysis(inp)
        last_pc = f"PC{inp.n_components}"
        assert abs(result.cumulative_variance_ratio[last_pc] - 1.0) < 0.01

    def test_loadings_structure(self, patched_factory):
        inp = PCAInput(
            tickers=['AAPL', 'MSFT'],
            start_date=START, end_date=END,
            n_components=2,
        )
        result = run_pca_analysis(inp)
        assert 'PC1' in result.loadings
        assert set(result.loadings['PC1'].keys()) == {'AAPL', 'MSFT'}

    def test_factor_contributions_structure(self, patched_factory):
        inp = PCAInput(
            tickers=['AAPL', 'MSFT'],
            start_date=START, end_date=END,
            n_components=2,
        )
        result = run_pca_analysis(inp)
        assert 'AAPL' in result.factor_contributions
        assert 'PC1' in result.factor_contributions['AAPL']

    def test_n_obs_positive(self, patched_factory):
        inp = PCAInput(
            tickers=['AAPL', 'MSFT'],
            start_date=START, end_date=END,
            n_components=2,
        )
        result = run_pca_analysis(inp)
        assert result.n_obs > 0


class TestRunHurstAnalysis:
    def test_returns_hurst_result(self, patched_factory):
        inp = HurstInput(
            symbol='AAPL', start_date=START, end_date=END,
        )
        result = run_hurst_analysis(inp)
        assert result.symbol == 'AAPL'

    def test_hurst_bounded(self, patched_factory):
        inp = HurstInput(symbol='AAPL', start_date=START, end_date=END)
        result = run_hurst_analysis(inp)
        assert 0.0 <= result.hurst <= 1.5

    def test_regime_is_valid_string(self, patched_factory):
        inp = HurstInput(symbol='AAPL', start_date=START, end_date=END)
        result = run_hurst_analysis(inp)
        assert result.regime in {'trending', 'random_walk', 'mean_reverting', 'unknown'}

    def test_fit_r_squared_bounded(self, patched_factory):
        inp = HurstInput(symbol='AAPL', start_date=START, end_date=END)
        result = run_hurst_analysis(inp)
        assert 0.0 <= result.fit_r_squared <= 1.0

    def test_method_dfa_is_default(self, patched_factory):
        inp = HurstInput(symbol='AAPL', start_date=START, end_date=END)
        result = run_hurst_analysis(inp)
        assert result.method == 'dfa'

    def test_method_rs_respected(self, patched_factory):
        inp = HurstInput(symbol='AAPL', start_date=START, end_date=END, method='rs')
        result = run_hurst_analysis(inp)
        assert result.method == 'rs'

    def test_n_obs_positive(self, patched_factory):
        inp = HurstInput(symbol='AAPL', start_date=START, end_date=END)
        result = run_hurst_analysis(inp)
        assert result.n_obs > 0

    def test_rolling_fields_none_when_not_requested(self, patched_factory):
        inp = HurstInput(symbol='AAPL', start_date=START, end_date=END)
        result = run_hurst_analysis(inp)
        assert result.rolling_current is None
        assert result.rolling_regime_fractions is None

    def test_rolling_fields_populated_when_requested(self, patched_factory):
        inp = HurstInput(
            symbol='AAPL', start_date=START, end_date=END,
            rolling_window=100,
        )
        result = run_hurst_analysis(inp)
        assert result.rolling_current is not None
        assert result.rolling_regime_fractions is not None
        fracs = result.rolling_regime_fractions
        assert set(fracs.keys()) == {'trending', 'random_walk', 'mean_reverting'}
        assert abs(sum(fracs.values()) - 1.0) < 0.01


class TestGetTechnicalAnalysisExtended:
    """Coverage for indicators added after the initial 12-tool set."""

    def test_stochastic_in_last_values(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['stochastic'])
        result = get_technical_analysis(inp)
        assert 'stoch_k' in result.last_values
        assert 'stoch_d' in result.last_values

    def test_stochastic_k_bounded(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['stochastic'])
        result = get_technical_analysis(inp)
        assert 0.0 <= result.last_values['stoch_k'] <= 100.0

    def test_stochastic_oversold_signal_is_bool(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['stochastic'])
        result = get_technical_analysis(inp)
        assert isinstance(result.signals['stoch_oversold'], bool)

    def test_vwap_in_last_values(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['vwap'])
        result = get_technical_analysis(inp)
        assert 'vwap' in result.last_values
        assert result.last_values['vwap'] > 0

    def test_vwap_signal_is_bool(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['vwap'])
        result = get_technical_analysis(inp)
        assert isinstance(result.signals['price_above_vwap'], bool)

    def test_williams_r_in_last_values(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['williams_r'])
        result = get_technical_analysis(inp)
        assert 'williams_r' in result.last_values

    def test_williams_r_bounded(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['williams_r'])
        result = get_technical_analysis(inp)
        assert -100.0 <= result.last_values['williams_r'] <= 0.0

    def test_williams_r_signals_present(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['williams_r'])
        result = get_technical_analysis(inp)
        assert 'williams_r_oversold' in result.signals
        assert 'williams_r_overbought' in result.signals
        assert isinstance(result.signals['williams_r_oversold'], bool)
        assert isinstance(result.signals['williams_r_overbought'], bool)

    def test_ema_in_last_values(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['ema'])
        result = get_technical_analysis(inp)
        assert 'ema_12' in result.last_values
        assert 'ema_26' in result.last_values

    def test_ema_values_positive(self, patched_factory):
        inp = TechnicalInput(symbol='AAPL', start_date=START, end_date=END, indicators=['ema'])
        result = get_technical_analysis(inp)
        assert result.last_values['ema_12'] > 0
        assert result.last_values['ema_26'] > 0

    def test_all_indicators_at_once(self, patched_factory):
        inp = TechnicalInput(
            symbol='AAPL', start_date=START, end_date=END,
            indicators=['rsi', 'macd', 'bollinger', 'sma', 'ema', 'stochastic', 'vwap', 'williams_r', 'adx', 'obv'],
        )
        result = get_technical_analysis(inp)
        for key in ('rsi_14', 'bb_upper', 'bb_lower', 'ema_12', 'ema_26',
                    'stoch_k', 'stoch_d', 'vwap', 'williams_r', 'adx', 'obv'):
            assert key in result.last_values, f"missing key: {key}"
