"""Tests for the original 12 agent tools (mocked data provider)."""

import pandas as pd
import pytest

from standard_quant_tools.agent import tools as tools_module
from standard_quant_tools.agent.models import (
    AnalysisInput,
    BacktestInput,
    BacktestOptInput,
    BuyAndHoldInput,
    CointegrationInput,
    CompareStrategiesInput,
    CorrelationAnalysisInput,
    FactorRegressionInput,
    GarchVolatilityForecastInput,
    HurstInput,
    KalmanHedgeRatioInput,
    LiquidityAnalysisInput,
    MonteCarloSimulationInput,
    PCAInput,
    PortfolioInput,
    RallyDetectionInput,
    ScreenerInput,
    StressTestInput,
    TailRiskInput,
    TechnicalInput,
    VolatilityEstimatorsInput,
)
from standard_quant_tools.agent.tools import (
    _TOOL_DISPATCH,
    TOOL_CATEGORY,
    analyze_stock_risk,
    compare_strategies,
    get_agent_tools,
    get_correlation_analysis,
    get_liquidity_metrics,
    get_portfolio_analysis,
    get_rally_signal,
    get_tail_risk_metrics,
    get_technical_analysis,
    get_volatility_estimators,
    run_backtest_optimization,
    run_bollinger_backtest,
    run_buy_and_hold,
    run_cointegration_test,
    run_factor_regression,
    run_garch_volatility_forecast,
    run_hurst_analysis,
    run_kalman_hedge_ratio,
    run_macd_backtest,
    run_monte_carlo_simulation,
    run_pca_analysis,
    run_rsi_backtest,
    run_screener,
    run_sma_backtest,
    run_stress_test,
)
from standard_quant_tools.data.factory import DataFactory

START, END = "2023-01-01", "2024-01-01"


class TestSanitizeForJson:
    """
    Regression tests (operational item B): sortino_ratio/calmar_ratio can
    legitimately return float('inf') (no downside deviation / no drawdown
    at all) -- valid math, but not valid JSON per RFC 8259. dispatch()'s
    own docstring recommends json.dumps(result) for sending its output to
    an LLM, and Python's json.dumps emits the non-standard Infinity/NaN
    tokens by default, which many strict JSON parsers reject. inf/-inf/nan
    must be sanitized to None before dispatch() returns.
    """

    def test_sanitizes_top_level_inf(self):
        from standard_quant_tools.agent.tools import _sanitize_for_json

        result = _sanitize_for_json({"calmar_ratio": float("inf"), "sharpe_ratio": 1.5})
        assert result == {"calmar_ratio": None, "sharpe_ratio": 1.5}

    def test_sanitizes_negative_inf_and_nan(self):
        from standard_quant_tools.agent.tools import _sanitize_for_json

        result = _sanitize_for_json({"a": float("-inf"), "b": float("nan"), "c": 0.0})
        assert result["a"] is None
        assert result["b"] is None
        assert result["c"] == 0.0

    def test_sanitizes_nested_dicts_and_lists(self):
        from standard_quant_tools.agent.tools import _sanitize_for_json

        result = _sanitize_for_json(
            {"windows": [{"sharpe": float("inf")}, {"sharpe": 0.5}]}
        )
        assert result == {"windows": [{"sharpe": None}, {"sharpe": 0.5}]}

    def test_leaves_finite_values_and_non_float_types_untouched(self):
        from standard_quant_tools.agent.tools import _sanitize_for_json

        result = _sanitize_for_json(
            {"n": 5, "s": "text", "b": True, "f": 1.23, "none": None}
        )
        assert result == {"n": 5, "s": "text", "b": True, "f": 1.23, "none": None}

    def test_json_dumps_succeeds_without_allow_nan_after_sanitizing(self):
        """The whole point: standard-compliant json.dumps(..., allow_nan=False)
        must succeed on the sanitized output where it would have raised on
        the raw inf value."""
        import json

        from standard_quant_tools.agent.tools import _sanitize_for_json

        raw = {"calmar_ratio": float("inf")}
        with pytest.raises(ValueError):
            json.dumps(raw, allow_nan=False)
        sanitized = _sanitize_for_json(raw)
        assert json.dumps(sanitized, allow_nan=False) == '{"calmar_ratio": null}'


class TestGetAgentTools:
    def test_returns_one_tool_per_registered_dispatch_entry(self):
        """Derived from _TOOL_DISPATCH rather than a hardcoded count -- a
        magic number here is exactly the kind of drift this repo has
        accumulated before (README/comments variously said 34/42/45)."""
        tools = get_agent_tools()
        assert len(tools) == len(_TOOL_DISPATCH)

    def test_all_tools_have_correct_schema_keys(self):
        for tool in get_agent_tools():
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_original_tool_names_present(self):
        names = {t["function"]["name"] for t in get_agent_tools()}
        original = {
            "run_sma_backtest",
            "run_rsi_backtest",
            "run_macd_backtest",
            "run_bollinger_backtest",
            "run_buy_and_hold",
            "compare_strategies",
            "analyze_stock_risk",
            "get_technical_analysis",
            "get_portfolio_analysis",
            "run_screener",
            "run_factor_regression",
            "run_cointegration_test",
            "run_pca_analysis",
            "run_hurst_analysis",
        }
        assert original.issubset(names)

    def test_parameters_are_valid_json_schema(self):
        for tool in get_agent_tools():
            schema = tool["function"]["parameters"]
            assert schema.get("type") == "object"
            assert "properties" in schema

    def test_no_categories_arg_matches_default_output_exactly(self):
        """Backward-compat guard: get_agent_tools(categories=None) must be
        byte-for-byte identical to get_agent_tools() -- every existing
        caller (dispatch(), pre-router single-agent scripts) relies on
        this."""
        assert get_agent_tools() == get_agent_tools(categories=None)

    def test_single_category_filter_returns_only_that_categorys_tools(self):
        names = {
            t["function"]["name"] for t in get_agent_tools(categories=["screener"])
        }
        assert names == {
            name for name, cat in TOOL_CATEGORY.items() if cat == "screener"
        }

    def test_multi_category_filter_unions_categories(self):
        names = {
            t["function"]["name"]
            for t in get_agent_tools(categories=["screener", "custom_signal"])
        }
        expected = {
            name
            for name, cat in TOOL_CATEGORY.items()
            if cat in ("screener", "custom_signal")
        }
        assert names == expected
        assert len(names) < len(_TOOL_DISPATCH)  # confirms it actually narrowed

    def test_unknown_category_returns_empty_not_an_error(self):
        """An unknown category name is silently ignored (see
        get_agent_tools's docstring) -- a router isn't a strict validator,
        it narrows when confident and shouldn't raise on a typo'd key."""
        assert get_agent_tools(categories=["not_a_real_category"]) == []


class TestToolCategoryCoverage:
    """The drift-proofing test: TOOL_CATEGORY is the single source of truth
    every other categorization (get_agent_tools(categories=...), the router,
    Multi_Agent_Implementation's WORKER_AGENTS) derives from. If a new tool
    is added to _TOOL_DISPATCH without a TOOL_CATEGORY entry, this fails
    immediately instead of silently drifting like the __all__/README/
    worker-list counts did before."""

    def test_every_dispatched_tool_has_exactly_one_category(self):
        assert set(TOOL_CATEGORY) == set(_TOOL_DISPATCH)

    def test_every_category_value_is_a_known_key(self):
        known_categories = {
            "data",
            "screener",
            "analysis",
            "quant_research",
            "backtest_execution",
            "backtest_validation",
            "custom_signal",
            "portfolio_risk",
            "discovery",
            "provenance",
            "microstructure",
            "derivatives",
        }
        assert set(TOOL_CATEGORY.values()) <= known_categories

    def test_backtest_execution_and_validation_are_disjoint_and_cover_backtest(self):
        """Regression guard for the specific split this repo made: every
        tool that used to be in one 16-tool 'backtest' bucket is now in
        exactly one of the two narrower categories, not both and not
        neither."""
        execution = {n for n, c in TOOL_CATEGORY.items() if c == "backtest_execution"}
        validation = {n for n, c in TOOL_CATEGORY.items() if c == "backtest_validation"}
        assert execution.isdisjoint(validation)
        # Derived rather than two magic numbers: the counts had to be edited
        # by hand on every addition, which makes the guard read as a
        # tripwire for growth rather than for a re-merge. What the split
        # actually promises is that every backtest_* tool lands in exactly
        # one of the two, and that neither side is empty.
        both = {n for n, c in TOOL_CATEGORY.items() if c.startswith("backtest_")}
        assert execution | validation == both
        assert execution and validation

    def test_run_backtest_optimization_and_run_sma_backtest_are_separated(self):
        """The exact kind of confusable-tool pair this category split exists
        to keep apart: 'run a strategy' vs 'optimize a strategy's params'."""
        assert TOOL_CATEGORY["run_sma_backtest"] == "backtest_execution"
        assert TOOL_CATEGORY["run_backtest_optimization"] == "backtest_validation"


class TestAgentModelExports:
    """Drift-proofing for agent/__init__.py: every Pydantic model defined in
    models.py must be re-exported from the agent package, or callers doing
    `from standard_quant_tools.agent import SomeInput` silently break even
    though the model itself works fine. This is the same class of bug
    TestToolCategoryCoverage guards against, one layer up."""

    def test_every_model_defined_in_models_py_is_exported(self):
        import inspect

        from pydantic import BaseModel

        import standard_quant_tools.agent as agent_pkg
        from standard_quant_tools.agent import models as models_module

        defined_models = {
            name
            for name, obj in vars(models_module).items()
            if inspect.isclass(obj)
            and issubclass(obj, BaseModel)
            and obj.__module__ == models_module.__name__
        }
        missing = defined_models - set(agent_pkg.__all__)
        assert (
            not missing
        ), f"models.py classes missing from agent.__all__: {sorted(missing)}"

    def test_every_exported_model_name_is_importable_from_the_package(self):
        import standard_quant_tools.agent as agent_pkg
        from standard_quant_tools.agent import models as models_module

        model_names_in_all = {
            name for name in agent_pkg.__all__ if hasattr(models_module, name)
        }
        for name in model_names_in_all:
            assert hasattr(agent_pkg, name), f"{name} is in __all__ but not importable"


class TestSMABacktest:
    def test_returns_backtest_result(self, patched_factory):
        inp = BacktestInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="sma_crossover",
            parameters={"fast_period": 10, "slow_period": 30},
        )
        result = run_sma_backtest(inp)
        assert result.total_return is not None
        assert result.num_trades >= 0
        assert 0.0 <= result.win_rate <= 1.0

    def test_equity_curve_has_values(self, patched_factory):
        inp = BacktestInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="sma_crossover",
        )
        result = run_sma_backtest(inp)
        assert len(result.equity_curve) > 0

    def test_all_required_fields_populated(self, patched_factory):
        inp = BacktestInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="sma_crossover",
        )
        result = run_sma_backtest(inp)
        assert result.final_equity > 0
        assert result.max_drawdown <= 0
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.calmar_ratio, float)


class TestRSIBacktest:
    def test_returns_backtest_result(self, patched_factory):
        inp = BacktestInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="rsi_mean_reversion",
            parameters={"period": 14, "oversold": 30, "overbought": 70},
        )
        result = run_rsi_backtest(inp)
        assert isinstance(result.total_return, float)

    def test_win_rate_bounded(self, patched_factory):
        inp = BacktestInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="rsi_mean_reversion",
        )
        result = run_rsi_backtest(inp)
        assert 0.0 <= result.win_rate <= 1.0


class TestMACDBacktest:
    def test_returns_backtest_result(self, patched_factory):
        inp = BacktestInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="macd_crossover",
            parameters={"fast": 12, "slow": 26, "signal": 9},
        )
        result = run_macd_backtest(inp)
        assert isinstance(result.total_return, float)


class TestBollingerBacktest:
    def test_returns_backtest_result(self, patched_factory):
        inp = BacktestInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="bollinger_reversion",
            parameters={"period": 20, "num_std": 2.0},
        )
        result = run_bollinger_backtest(inp)
        assert isinstance(result.total_return, float)


class TestAnalyzeStockRisk:
    def test_returns_analysis_result(self, patched_factory):
        inp = AnalysisInput(symbol="AAPL", benchmark="SPY", period="1y")
        result = analyze_stock_risk(inp)
        assert result.symbol == "AAPL"
        assert result.benchmark == "SPY"

    def test_all_fields_are_floats(self, patched_factory):
        inp = AnalysisInput(symbol="AAPL", benchmark="SPY", period="1y")
        result = analyze_stock_risk(inp)
        for field in (
            "alpha",
            "beta",
            "r_squared",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "var_95",
            "cvar_95",
            "information_ratio",
        ):
            assert isinstance(getattr(result, field), float), f"{field} is not float"

    def test_r_squared_bounded(self, patched_factory):
        inp = AnalysisInput(symbol="AAPL", benchmark="SPY", period="1y")
        result = analyze_stock_risk(inp)
        assert 0.0 <= result.r_squared <= 1.0

    def test_var_less_than_cvar(self, patched_factory):
        inp = AnalysisInput(symbol="AAPL", benchmark="SPY", period="1y")
        result = analyze_stock_risk(inp)
        assert result.cvar_95 >= result.var_95


class TestGetTechnicalAnalysis:
    def test_returns_technical_result(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            indicators=["rsi", "macd", "bollinger", "sma"],
        )
        result = get_technical_analysis(inp)
        assert result.symbol == "AAPL"
        assert result.last_close > 0

    def test_rsi_indicator_in_last_values(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["rsi"]
        )
        result = get_technical_analysis(inp)
        assert "rsi_14" in result.last_values
        assert 0 <= result.last_values["rsi_14"] <= 100

    def test_macd_signal_in_signals(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["macd"]
        )
        result = get_technical_analysis(inp)
        assert "macd_bullish" in result.signals
        assert isinstance(result.signals["macd_bullish"], bool)

    def test_bollinger_values_in_last_values(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["bollinger"]
        )
        result = get_technical_analysis(inp)
        assert "bb_upper" in result.last_values
        assert result.last_values["bb_upper"] > result.last_values["bb_lower"]

    def test_adx_signal_in_signals(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["adx"]
        )
        result = get_technical_analysis(inp)
        assert "strong_trend" in result.signals
        assert "adx" in result.last_values

    def test_obv_in_last_values(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["obv"]
        )
        result = get_technical_analysis(inp)
        assert "obv" in result.last_values

    def test_fused_path_matches_per_indicator_fallback(self, patched_factory):
        """When 2+ of {rsi, adx, bollinger, stochastic} are requested, the
        fused technical_indicators() native call is used instead of one C++
        round trip per indicator. Its numeric output must be identical to
        the per-indicator fallback path (forced here by disabling the fused
        path), since it's the same underlying kernels -- just one call."""
        inp = TechnicalInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            indicators=["rsi", "adx", "bollinger", "stochastic"],
        )
        fused_result = get_technical_analysis(inp)

        # Patched on the RESEARCH runtime, which is where
        # get_technical_analysis now lives. `from _shared import HAS_CPP`
        # binds a copy, so flipping it on the facade or on _shared would
        # leave this module's own reference untouched and the test would
        # silently compare the fused path against itself.
        from standard_quant_tools.agent.runtimes.research import tools as research_tools

        original_has_cpp = research_tools.HAS_CPP
        research_tools.HAS_CPP = False
        try:
            fallback_result = get_technical_analysis(inp)
        finally:
            research_tools.HAS_CPP = original_has_cpp

        assert fused_result.last_values == fallback_result.last_values
        assert fused_result.signals == fallback_result.signals


class TestGetPortfolioAnalysis:
    def test_returns_portfolio_result(self, patched_factory):
        inp = PortfolioInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            weights=[1 / 3, 1 / 3, 1 / 3],
            start_date=START,
            end_date=END,
        )
        result = get_portfolio_analysis(inp)
        assert result.tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_all_metric_fields_populated(self, patched_factory):
        inp = PortfolioInput(
            tickers=["AAPL", "MSFT"],
            weights=[0.6, 0.4],
            start_date=START,
            end_date=END,
        )
        result = get_portfolio_analysis(inp)
        for field in (
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "var_95",
            "cvar_95",
        ):
            assert isinstance(getattr(result, field), float), f"{field} not float"

    def test_correlation_matrix_in_result(self, patched_factory):
        inp = PortfolioInput(
            tickers=["AAPL", "MSFT"],
            weights=[0.5, 0.5],
            start_date=START,
            end_date=END,
        )
        result = get_portfolio_analysis(inp)
        assert isinstance(result.correlation_matrix, dict)


class TestRunScreener:
    def test_returns_screener_result(self, patched_factory):
        inp = ScreenerInput(tickers=["AAPL", "MSFT"], filters={"pe_ratio_max": 35.0})
        result = run_screener(inp)
        assert result.num_passed >= 0
        assert isinstance(result.tickers_passed, list)
        assert isinstance(result.results, list)

    def test_num_passed_matches_tickers_passed(self, patched_factory):
        inp = ScreenerInput(tickers=["AAPL", "MSFT", "GOOGL"], filters={})
        result = run_screener(inp)
        assert result.num_passed == len(result.tickers_passed)


class TestRunFactorRegression:
    def test_returns_factor_regression_result(self, patched_factory):
        inp = FactorRegressionInput(
            symbol="AAPL",
            factor_tickers=["SPY", "IWM"],
            factor_names=["mkt", "smb"],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert result.symbol == "AAPL"
        assert result.factors == ["mkt", "smb"]

    def test_defaults_factor_names_to_tickers(self, patched_factory):
        inp = FactorRegressionInput(
            symbol="AAPL",
            factor_tickers=["SPY", "IWM"],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert result.factors == ["SPY", "IWM"]

    def test_alpha_is_float(self, patched_factory):
        inp = FactorRegressionInput(
            symbol="AAPL",
            factor_tickers=["SPY"],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert isinstance(result.alpha, float)

    def test_loadings_keys_match_factor_names(self, patched_factory):
        inp = FactorRegressionInput(
            symbol="AAPL",
            factor_tickers=["SPY", "IWM"],
            factor_names=["mkt", "smb"],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert set(result.loadings.keys()) == {"mkt", "smb"}

    def test_t_stats_and_p_values_have_alpha_key(self, patched_factory):
        inp = FactorRegressionInput(
            symbol="AAPL",
            factor_tickers=["SPY"],
            factor_names=["mkt"],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert "alpha" in result.t_stats
        assert "alpha" in result.p_values

    def test_r_squared_bounded(self, patched_factory):
        inp = FactorRegressionInput(
            symbol="AAPL",
            factor_tickers=["SPY"],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert 0.0 <= result.r_squared <= 1.0

    def test_n_obs_positive(self, patched_factory):
        inp = FactorRegressionInput(
            symbol="AAPL",
            factor_tickers=["SPY"],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert result.n_obs > 0

    def test_rolling_tail_none_when_not_requested(self, patched_factory):
        inp = FactorRegressionInput(
            symbol="AAPL",
            factor_tickers=["SPY"],
            start_date=START,
            end_date=END,
        )
        result = run_factor_regression(inp)
        assert result.rolling_alpha_tail is None
        assert result.rolling_loadings_tail is None

    def test_rolling_tail_populated_when_requested(self, patched_factory):
        inp = FactorRegressionInput(
            symbol="AAPL",
            factor_tickers=["SPY"],
            factor_names=["mkt"],
            start_date=START,
            end_date=END,
            rolling_window=60,
        )
        result = run_factor_regression(inp)
        assert result.rolling_alpha_tail is not None
        assert result.rolling_loadings_tail is not None
        assert "mkt" in result.rolling_loadings_tail


class TestRunCointegrationTest:
    def test_returns_cointegration_result(self, patched_factory):
        inp = CointegrationInput(
            symbol_a="KO",
            symbol_b="PEP",
            start_date=START,
            end_date=END,
        )
        result = run_cointegration_test(inp)
        assert result.symbol_a == "KO"
        assert result.symbol_b == "PEP"

    def test_cointegrated_is_bool(self, patched_factory):
        inp = CointegrationInput(
            symbol_a="KO",
            symbol_b="PEP",
            start_date=START,
            end_date=END,
        )
        result = run_cointegration_test(inp)
        assert isinstance(result.cointegrated, bool)

    def test_p_value_bounded(self, patched_factory):
        inp = CointegrationInput(
            symbol_a="KO",
            symbol_b="PEP",
            start_date=START,
            end_date=END,
        )
        result = run_cointegration_test(inp)
        assert 0.0 <= result.p_value <= 1.0

    def test_signal_is_valid_string(self, patched_factory):
        inp = CointegrationInput(
            symbol_a="KO",
            symbol_b="PEP",
            start_date=START,
            end_date=END,
        )
        result = run_cointegration_test(inp)
        assert result.signal in {"long_a_short_b", "short_a_long_b", "neutral"}

    def test_half_life_is_finite_float(self, patched_factory):
        inp = CointegrationInput(
            symbol_a="KO",
            symbol_b="PEP",
            start_date=START,
            end_date=END,
        )
        result = run_cointegration_test(inp)
        assert isinstance(result.half_life_days, float)
        assert result.half_life_days <= 9999.0

    def test_critical_values_have_three_levels(self, patched_factory):
        inp = CointegrationInput(
            symbol_a="KO",
            symbol_b="PEP",
            start_date=START,
            end_date=END,
        )
        result = run_cointegration_test(inp)
        assert set(result.critical_values.keys()) == {"1%", "5%", "10%"}

    def test_n_obs_positive(self, patched_factory):
        inp = CointegrationInput(
            symbol_a="KO",
            symbol_b="PEP",
            start_date=START,
            end_date=END,
        )
        result = run_cointegration_test(inp)
        assert result.n_obs > 0

    def test_hedge_ratio_is_float(self, patched_factory):
        inp = CointegrationInput(
            symbol_a="KO",
            symbol_b="PEP",
            start_date=START,
            end_date=END,
        )
        result = run_cointegration_test(inp)
        assert isinstance(result.hedge_ratio, float)


class TestRunKalmanHedgeRatio:
    def test_returns_result_for_symbols(self, patched_factory):
        inp = KalmanHedgeRatioInput(
            symbol_a="KO", symbol_b="PEP", start_date=START, end_date=END
        )
        result = run_kalman_hedge_ratio(inp)
        assert result.symbol_a == "KO"
        assert result.symbol_b == "PEP"

    def test_signal_is_valid_string(self, patched_factory):
        inp = KalmanHedgeRatioInput(
            symbol_a="KO", symbol_b="PEP", start_date=START, end_date=END
        )
        result = run_kalman_hedge_ratio(inp)
        assert result.signal in {"long_a_short_b", "short_a_long_b", "neutral"}

    def test_hedge_ratio_and_intercept_are_float(self, patched_factory):
        inp = KalmanHedgeRatioInput(
            symbol_a="KO", symbol_b="PEP", start_date=START, end_date=END
        )
        result = run_kalman_hedge_ratio(inp)
        assert isinstance(result.current_hedge_ratio, float)
        assert isinstance(result.current_intercept, float)

    def test_hedge_ratio_std_nonnegative(self, patched_factory):
        inp = KalmanHedgeRatioInput(
            symbol_a="KO", symbol_b="PEP", start_date=START, end_date=END
        )
        result = run_kalman_hedge_ratio(inp)
        assert result.hedge_ratio_std >= 0.0

    def test_n_obs_positive(self, patched_factory):
        inp = KalmanHedgeRatioInput(
            symbol_a="KO", symbol_b="PEP", start_date=START, end_date=END
        )
        result = run_kalman_hedge_ratio(inp)
        assert result.n_obs > 0

    def test_delta_out_of_bounds_rejected_by_pydantic(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            KalmanHedgeRatioInput(
                symbol_a="KO", symbol_b="PEP", start_date=START, end_date=END, delta=1.5
            )


class TestRunPCAAnalysis:
    def test_returns_pca_result(self, patched_factory):
        inp = PCAInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START,
            end_date=END,
            n_components=2,
        )
        result = run_pca_analysis(inp)
        assert result.tickers == ["AAPL", "MSFT", "GOOGL"]
        assert result.n_components == 2

    def test_explained_variance_ratio_keys(self, patched_factory):
        inp = PCAInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START,
            end_date=END,
            n_components=3,
        )
        result = run_pca_analysis(inp)
        assert set(result.explained_variance_ratio.keys()) == {"PC1", "PC2", "PC3"}

    def test_evr_values_sum_to_one(self, patched_factory):
        inp = PCAInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START,
            end_date=END,
            n_components=3,
        )
        result = run_pca_analysis(inp)
        total = sum(result.explained_variance_ratio.values())
        assert abs(total - 1.0) < 0.01

    def test_cumulative_variance_ends_near_one(self, patched_factory):
        inp = PCAInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START,
            end_date=END,
            n_components=3,
        )
        result = run_pca_analysis(inp)
        last_pc = f"PC{inp.n_components}"
        assert abs(result.cumulative_variance_ratio[last_pc] - 1.0) < 0.01

    def test_loadings_structure(self, patched_factory):
        inp = PCAInput(
            tickers=["AAPL", "MSFT"],
            start_date=START,
            end_date=END,
            n_components=2,
        )
        result = run_pca_analysis(inp)
        assert "PC1" in result.loadings
        assert set(result.loadings["PC1"].keys()) == {"AAPL", "MSFT"}

    def test_factor_contributions_structure(self, patched_factory):
        inp = PCAInput(
            tickers=["AAPL", "MSFT"],
            start_date=START,
            end_date=END,
            n_components=2,
        )
        result = run_pca_analysis(inp)
        assert "AAPL" in result.factor_contributions
        assert "PC1" in result.factor_contributions["AAPL"]

    def test_n_obs_positive(self, patched_factory):
        inp = PCAInput(
            tickers=["AAPL", "MSFT"],
            start_date=START,
            end_date=END,
            n_components=2,
        )
        result = run_pca_analysis(inp)
        assert result.n_obs > 0


class TestRunHurstAnalysis:
    def test_returns_hurst_result(self, patched_factory):
        inp = HurstInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
        )
        result = run_hurst_analysis(inp)
        assert result.symbol == "AAPL"

    def test_hurst_bounded(self, patched_factory):
        inp = HurstInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_hurst_analysis(inp)
        assert 0.0 <= result.hurst <= 1.5

    def test_regime_is_valid_string(self, patched_factory):
        inp = HurstInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_hurst_analysis(inp)
        assert result.regime in {"trending", "random_walk", "mean_reverting", "unknown"}

    def test_fit_r_squared_bounded(self, patched_factory):
        inp = HurstInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_hurst_analysis(inp)
        assert 0.0 <= result.fit_r_squared <= 1.0

    def test_method_dfa_is_default(self, patched_factory):
        inp = HurstInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_hurst_analysis(inp)
        assert result.method == "dfa"

    def test_method_rs_respected(self, patched_factory):
        inp = HurstInput(symbol="AAPL", start_date=START, end_date=END, method="rs")
        result = run_hurst_analysis(inp)
        assert result.method == "rs"

    def test_invalid_method_rejected_by_pydantic(self, patched_factory):
        """
        HurstInput.method is a Literal["dfa", "rs"] specifically so a typo
        (e.g. "DFA") is rejected at the API boundary rather than silently
        running R/S analysis while echoing the typo'd string back in the
        result — see analysis/hurst.py's own runtime guard for the same
        regression, defended here at the Pydantic layer too.
        """
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            HurstInput(symbol="AAPL", start_date=START, end_date=END, method="DFA")

    def test_n_obs_positive(self, patched_factory):
        inp = HurstInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_hurst_analysis(inp)
        assert result.n_obs > 0

    def test_rolling_fields_none_when_not_requested(self, patched_factory):
        inp = HurstInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_hurst_analysis(inp)
        assert result.rolling_current is None
        assert result.rolling_regime_fractions is None

    def test_rolling_fields_populated_when_requested(self, patched_factory):
        inp = HurstInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            rolling_window=100,
        )
        result = run_hurst_analysis(inp)
        assert result.rolling_current is not None
        assert result.rolling_regime_fractions is not None
        fracs = result.rolling_regime_fractions
        assert set(fracs.keys()) == {"trending", "random_walk", "mean_reverting"}
        assert abs(sum(fracs.values()) - 1.0) < 0.01


class TestGetRallySignal:
    def test_returns_result_for_symbol(self, patched_factory):
        inp = RallyDetectionInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_rally_signal(inp)
        assert result.symbol == "AAPL"

    def test_rally_score_is_fraction_of_five_signals(self, patched_factory):
        inp = RallyDetectionInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_rally_signal(inp)
        assert result.rally_score in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}

    def test_is_rally_matches_score_threshold(self, patched_factory):
        inp = RallyDetectionInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_rally_signal(inp)
        assert result.is_rally == (result.rally_score >= 0.6)

    def test_trend_direction_is_valid_string(self, patched_factory):
        inp = RallyDetectionInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_rally_signal(inp)
        assert result.trend_direction in {"bullish", "bearish", "neutral"}

    def test_regime_is_valid_string(self, patched_factory):
        inp = RallyDetectionInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_rally_signal(inp)
        assert result.regime in {"trending", "random_walk", "mean_reverting", "unknown"}

    def test_adx_is_nonnegative(self, patched_factory):
        inp = RallyDetectionInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_rally_signal(inp)
        assert result.adx >= 0.0

    def test_n_obs_positive(self, patched_factory):
        inp = RallyDetectionInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_rally_signal(inp)
        assert result.n_obs > 0

    def test_default_reports_fixed_adx_threshold_not_auto_tuned(self, patched_factory):
        inp = RallyDetectionInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_rally_signal(inp)
        assert result.auto_tuned is False
        assert result.adx_threshold_used == 25.0

    def test_auto_tune_reports_a_different_threshold(self, patched_factory):
        inp = RallyDetectionInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            auto_tune_adx_threshold=True,
            auto_tune_percentile=60.0,
        )
        result = get_rally_signal(inp)
        assert result.auto_tuned is True
        assert isinstance(result.adx_threshold_used, float)

    def test_auto_tune_percentile_out_of_range_rejected_by_pydantic(
        self, patched_factory
    ):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            RallyDetectionInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                auto_tune_percentile=0.0,
            )

    def test_custom_thresholds_respected(self, patched_factory):
        inp = RallyDetectionInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            lookback=10,
            adx_threshold=30.0,
            breakout_period=15,
        )
        result = get_rally_signal(inp)
        assert result.symbol == "AAPL"

    def test_invalid_hurst_method_rejected_by_pydantic(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            RallyDetectionInput(
                symbol="AAPL", start_date=START, end_date=END, hurst_method="DFA"
            )

    def test_non_positive_lookback_rejected_by_pydantic(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            RallyDetectionInput(
                symbol="AAPL", start_date=START, end_date=END, lookback=0
            )


class TestGetVolatilityEstimators:
    def test_returns_result_for_symbol(self, patched_factory):
        inp = VolatilityEstimatorsInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_volatility_estimators(inp)
        assert result.symbol == "AAPL"
        assert result.period == 20

    def test_all_volatility_fields_nonnegative(self, patched_factory):
        inp = VolatilityEstimatorsInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_volatility_estimators(inp)
        assert result.close_to_close_annualized >= 0.0
        assert result.parkinson_annualized >= 0.0
        assert result.garman_klass_annualized >= 0.0
        assert result.yang_zhang_annualized >= 0.0

    def test_ratio_is_yang_zhang_over_close_to_close(self, patched_factory):
        """
        The ratio is computed from the raw (unrounded) values inside the
        tool, not by dividing the two already-rounded (6dp) output fields --
        so recomputing it from those rounded fields only agrees to within
        rounding-composition error, not exactly.
        """
        inp = VolatilityEstimatorsInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_volatility_estimators(inp)
        expected = result.yang_zhang_annualized / result.close_to_close_annualized
        assert result.yang_zhang_vs_close_to_close_ratio == pytest.approx(
            expected, abs=1e-3
        )

    def test_custom_period_respected(self, patched_factory):
        inp = VolatilityEstimatorsInput(
            symbol="AAPL", start_date=START, end_date=END, period=10
        )
        result = get_volatility_estimators(inp)
        assert result.period == 10

    def test_invalid_period_rejected_by_pydantic(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            VolatilityEstimatorsInput(
                symbol="AAPL", start_date=START, end_date=END, period=1
            )


class TestRunGarchVolatilityForecast:
    def test_returns_result_for_symbol(self, patched_factory):
        inp = GarchVolatilityForecastInput(
            symbol="AAPL", start_date=START, end_date=END
        )
        result = run_garch_volatility_forecast(inp)
        assert result.symbol == "AAPL"

    def test_persistence_equals_alpha_plus_beta(self, patched_factory):
        inp = GarchVolatilityForecastInput(
            symbol="AAPL", start_date=START, end_date=END
        )
        result = run_garch_volatility_forecast(inp)
        assert result.persistence == pytest.approx(
            round(result.alpha + result.beta, 6), abs=1e-4
        )

    def test_forecast_length_matches_horizon(self, patched_factory):
        inp = GarchVolatilityForecastInput(
            symbol="AAPL", start_date=START, end_date=END, forecast_horizon=5
        )
        result = run_garch_volatility_forecast(inp)
        assert len(result.forecast_annualized_vol) == 5

    def test_vols_are_nonnegative(self, patched_factory):
        inp = GarchVolatilityForecastInput(
            symbol="AAPL", start_date=START, end_date=END
        )
        result = run_garch_volatility_forecast(inp)
        assert result.current_annualized_vol >= 0.0
        assert result.long_run_annualized_vol >= 0.0
        assert all(v >= 0.0 for v in result.forecast_annualized_vol)

    def test_invalid_horizon_rejected_by_pydantic(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            GarchVolatilityForecastInput(
                symbol="AAPL", start_date=START, end_date=END, forecast_horizon=0
            )


class TestGetCorrelationAnalysis:
    """
    patched_factory's mock provider returns the identical sample_ohlcv for
    every ticker regardless of symbol -- so every pair is perfectly
    correlated here (avg/highest/lowest all == 1.0, diversification_ratio
    == 1.0). That's expected given the shared fixture, not a bug; these
    tests assert on that known structure rather than assuming variation.
    """

    def test_returns_result_for_tickers(self, patched_factory):
        inp = CorrelationAnalysisInput(
            tickers=["AAPL", "MSFT", "GOOGL"], start_date=START, end_date=END
        )
        result = get_correlation_analysis(inp)
        assert result.tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_correlation_matrix_has_all_tickers_as_keys(self, patched_factory):
        inp = CorrelationAnalysisInput(
            tickers=["AAPL", "MSFT"], start_date=START, end_date=END
        )
        result = get_correlation_analysis(inp)
        assert set(result.correlation_matrix.keys()) == {"AAPL", "MSFT"}
        assert set(result.correlation_matrix["AAPL"].keys()) == {"AAPL", "MSFT"}

    def test_diagonal_is_one(self, patched_factory):
        inp = CorrelationAnalysisInput(
            tickers=["AAPL", "MSFT"], start_date=START, end_date=END
        )
        result = get_correlation_analysis(inp)
        assert result.correlation_matrix["AAPL"]["AAPL"] == pytest.approx(1.0)
        assert result.correlation_matrix["MSFT"]["MSFT"] == pytest.approx(1.0)

    def test_identical_mock_data_yields_full_correlation(self, patched_factory):
        inp = CorrelationAnalysisInput(
            tickers=["AAPL", "MSFT"], start_date=START, end_date=END
        )
        result = get_correlation_analysis(inp)
        assert result.avg_pairwise_correlation == pytest.approx(1.0)
        assert result.highest_correlated_pair["correlation"] == pytest.approx(1.0)
        assert result.lowest_correlated_pair["correlation"] == pytest.approx(1.0)
        assert result.diversification_ratio == pytest.approx(1.0)

    def test_highest_and_lowest_pair_ticker_membership(self, patched_factory):
        inp = CorrelationAnalysisInput(
            tickers=["AAPL", "MSFT", "GOOGL"], start_date=START, end_date=END
        )
        result = get_correlation_analysis(inp)
        assert {
            result.highest_correlated_pair["a"],
            result.highest_correlated_pair["b"],
        } <= {"AAPL", "MSFT", "GOOGL"}

    def test_too_few_tickers_rejected_by_pydantic(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            CorrelationAnalysisInput(tickers=["AAPL"], start_date=START, end_date=END)

    def test_custom_weights_accepted(self, patched_factory):
        inp = CorrelationAnalysisInput(
            tickers=["AAPL", "MSFT"],
            start_date=START,
            end_date=END,
            weights=[0.3, 0.7],
        )
        result = get_correlation_analysis(inp)
        assert result.diversification_ratio == pytest.approx(1.0)

    def test_weights_not_summing_to_one_rejected_by_pydantic(self):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            CorrelationAnalysisInput(
                tickers=["AAPL", "MSFT"],
                start_date=START,
                end_date=END,
                weights=[0.3, 0.3],
            )

    def test_weights_length_mismatch_rejected_by_pydantic(self):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            CorrelationAnalysisInput(
                tickers=["AAPL", "MSFT", "GOOGL"],
                start_date=START,
                end_date=END,
                weights=[0.5, 0.5],
            )


class TestRunMonteCarloSimulation:
    def test_returns_result_for_tickers(self, patched_factory):
        inp = MonteCarloSimulationInput(
            tickers=["AAPL", "MSFT"],
            start_date=START,
            end_date=END,
            horizon_days=30,
            n_simulations=200,
            random_seed=1,
        )
        result = run_monte_carlo_simulation(inp)
        assert result.tickers == ["AAPL", "MSFT"]
        assert result.horizon_days == 30
        assert result.n_simulations == 200

    def test_equity_bands_have_horizon_length(self, patched_factory):
        inp = MonteCarloSimulationInput(
            tickers=["AAPL"],
            start_date=START,
            end_date=END,
            horizon_days=45,
            n_simulations=200,
            random_seed=2,
        )
        result = run_monte_carlo_simulation(inp)
        assert len(result.equity_band_p5) == 45
        assert len(result.equity_band_p50) == 45
        assert len(result.equity_band_p95) == 45

    def test_bands_ordered(self, patched_factory):
        inp = MonteCarloSimulationInput(
            tickers=["AAPL"],
            start_date=START,
            end_date=END,
            horizon_days=30,
            n_simulations=500,
            random_seed=3,
        )
        result = run_monte_carlo_simulation(inp)
        for p5, p50, p95 in zip(
            result.equity_band_p5, result.equity_band_p50, result.equity_band_p95
        ):
            assert p5 <= p50 <= p95

    def test_prob_loss_bounded(self, patched_factory):
        inp = MonteCarloSimulationInput(
            tickers=["AAPL"],
            start_date=START,
            end_date=END,
            horizon_days=30,
            n_simulations=200,
            random_seed=4,
        )
        result = run_monte_carlo_simulation(inp)
        assert 0.0 <= result.prob_loss <= 1.0

    def test_reproducible_with_same_seed(self, patched_factory):
        kwargs = dict(
            tickers=["AAPL"],
            start_date=START,
            end_date=END,
            horizon_days=30,
            n_simulations=200,
            random_seed=42,
        )
        r1 = run_monte_carlo_simulation(MonteCarloSimulationInput(**kwargs))
        r2 = run_monte_carlo_simulation(MonteCarloSimulationInput(**kwargs))
        assert r1.terminal_median == r2.terminal_median
        assert r1.equity_band_p50 == r2.equity_band_p50

    def test_invalid_horizon_rejected_by_pydantic(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            MonteCarloSimulationInput(
                tickers=["AAPL"], start_date=START, end_date=END, horizon_days=0
            )

    def test_too_few_simulations_rejected_by_pydantic(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            MonteCarloSimulationInput(
                tickers=["AAPL"], start_date=START, end_date=END, n_simulations=10
            )

    def test_custom_weights_accepted(self, patched_factory):
        inp = MonteCarloSimulationInput(
            tickers=["AAPL", "MSFT"],
            start_date=START,
            end_date=END,
            weights=[0.3, 0.7],
            horizon_days=30,
            n_simulations=200,
            random_seed=5,
        )
        result = run_monte_carlo_simulation(inp)
        assert result.tickers == ["AAPL", "MSFT"]

    def test_weights_not_summing_to_one_rejected_by_pydantic(self):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            MonteCarloSimulationInput(
                tickers=["AAPL", "MSFT"],
                start_date=START,
                end_date=END,
                weights=[0.3, 0.3],
            )

    def test_weights_length_mismatch_rejected_by_pydantic(self):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            MonteCarloSimulationInput(
                tickers=["AAPL", "MSFT", "GOOGL"],
                start_date=START,
                end_date=END,
                weights=[0.5, 0.5],
            )


class TestRunStressTest:
    """
    patched_factory's mock provider returns sample_ohlcv for every ticker
    regardless of symbol/date range, so every requested ticker "has data"
    for every scenario here -- tickers_missing_data is expected to be empty
    under this fixture; a real provider would populate it for tickers that
    didn't exist yet during an old scenario window.
    """

    def test_default_scenario_is_gfc_2008(self, patched_factory):
        inp = StressTestInput(tickers=["AAPL", "MSFT"])
        result = run_stress_test(inp)
        assert result.scenario == "gfc_2008"
        assert result.scenario_start_date == "2008-09-01"
        assert result.scenario_end_date == "2009-03-09"

    def test_named_scenario_dates_reported(self, patched_factory):
        inp = StressTestInput(tickers=["AAPL"], scenario="covid_2020")
        result = run_stress_test(inp)
        assert result.scenario_start_date == "2020-02-19"
        assert result.scenario_end_date == "2020-03-23"

    def test_all_tickers_used_under_mock_data(self, patched_factory):
        inp = StressTestInput(tickers=["AAPL", "MSFT", "GOOGL"])
        result = run_stress_test(inp)
        assert set(result.tickers_used) == {"AAPL", "MSFT", "GOOGL"}
        assert result.tickers_missing_data == []

    def test_max_drawdown_nonpositive(self, patched_factory):
        inp = StressTestInput(tickers=["AAPL"], scenario="dotcom_2000")
        result = run_stress_test(inp)
        assert result.max_drawdown_pct <= 0.0

    def test_worst_day_return_le_best_day_return(self, patched_factory):
        inp = StressTestInput(tickers=["AAPL"], scenario="volmageddon_2018")
        result = run_stress_test(inp)
        assert result.worst_day_return_pct <= result.best_day_return_pct

    def test_custom_scenario_requires_custom_dates(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            StressTestInput(tickers=["AAPL"], scenario="custom")

    def test_custom_scenario_with_dates_accepted(self, patched_factory):
        inp = StressTestInput(
            tickers=["AAPL"],
            scenario="custom",
            custom_start_date="2015-01-01",
            custom_end_date="2015-03-01",
        )
        result = run_stress_test(inp)
        assert result.scenario_start_date == "2015-01-01"
        assert result.scenario_end_date == "2015-03-01"

    def test_unknown_named_scenario_rejected_by_pydantic(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            StressTestInput(tickers=["AAPL"], scenario="not_a_real_crash")

    def test_mismatched_weights_length_rejected(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            StressTestInput(tickers=["AAPL", "MSFT"], weights=[1.0])


class TestGetLiquidityMetrics:
    def test_returns_result_for_tickers(self, patched_factory):
        inp = LiquidityAnalysisInput(
            tickers=["AAPL", "MSFT"], start_date=START, end_date=END
        )
        result = get_liquidity_metrics(inp)
        assert result.tickers == ["AAPL", "MSFT"]
        assert set(result.per_ticker.keys()) == {"AAPL", "MSFT"}

    def test_per_ticker_fields_present(self, patched_factory):
        inp = LiquidityAnalysisInput(tickers=["AAPL"], start_date=START, end_date=END)
        result = get_liquidity_metrics(inp)
        fields = result.per_ticker["AAPL"]
        assert set(fields.keys()) == {
            "avg_dollar_volume",
            "amihud_illiquidity",
            "corwin_schultz_spread_bps",
        }

    def test_amihud_illiquidity_nonnegative(self, patched_factory):
        inp = LiquidityAnalysisInput(tickers=["AAPL"], start_date=START, end_date=END)
        result = get_liquidity_metrics(inp)
        assert result.per_ticker["AAPL"]["amihud_illiquidity"] >= 0.0

    def test_corwin_schultz_spread_nonnegative(self, patched_factory):
        inp = LiquidityAnalysisInput(tickers=["AAPL"], start_date=START, end_date=END)
        result = get_liquidity_metrics(inp)
        assert result.per_ticker["AAPL"]["corwin_schultz_spread_bps"] >= 0.0

    def test_least_and_most_liquid_are_valid_tickers(self, patched_factory):
        inp = LiquidityAnalysisInput(
            tickers=["AAPL", "MSFT", "GOOGL"], start_date=START, end_date=END
        )
        result = get_liquidity_metrics(inp)
        assert result.least_liquid_ticker in {"AAPL", "MSFT", "GOOGL"}
        assert result.most_liquid_ticker in {"AAPL", "MSFT", "GOOGL"}

    def test_custom_window_respected(self, patched_factory):
        inp = LiquidityAnalysisInput(
            tickers=["AAPL"], start_date=START, end_date=END, window=10
        )
        result = get_liquidity_metrics(inp)
        assert result.tickers == ["AAPL"]

    def test_invalid_window_rejected_by_pydantic(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            LiquidityAnalysisInput(
                tickers=["AAPL"], start_date=START, end_date=END, window=0
            )


class TestGetTailRiskMetrics:
    def test_returns_result_for_symbol(self, patched_factory):
        inp = TailRiskInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_tail_risk_metrics(inp)
        assert result.symbol == "AAPL"

    def test_default_method_is_pwm(self, patched_factory):
        inp = TailRiskInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_tail_risk_metrics(inp)
        assert result.method == "pwm"

    def test_var_evt_positive_and_cvar_not_below_var(self, patched_factory):
        inp = TailRiskInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_tail_risk_metrics(inp)
        assert result.var_evt > 0.0
        assert result.cvar_evt >= result.var_evt

    def test_tail_classification_is_valid(self, patched_factory):
        inp = TailRiskInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_tail_risk_metrics(inp)
        assert result.tail_classification in {
            "heavy_tailed",
            "light_tailed",
            "near_exponential",
        }

    def test_var_historical_comparison_is_float(self, patched_factory):
        inp = TailRiskInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_tail_risk_metrics(inp)
        assert isinstance(result.var_historical_comparison, float)

    def test_invalid_tail_fraction_rejected_by_pydantic(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            TailRiskInput(
                symbol="AAPL", start_date=START, end_date=END, tail_fraction=0.6
            )

    def test_invalid_method_rejected_by_pydantic(self, patched_factory):
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            TailRiskInput(symbol="AAPL", start_date=START, end_date=END, method="bogus")


class TestGetTechnicalAnalysisExtended:
    """Coverage for indicators added after the initial 12-tool set."""

    def test_stochastic_in_last_values(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["stochastic"]
        )
        result = get_technical_analysis(inp)
        assert "stoch_k" in result.last_values
        assert "stoch_d" in result.last_values

    def test_stochastic_k_bounded(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["stochastic"]
        )
        result = get_technical_analysis(inp)
        assert 0.0 <= result.last_values["stoch_k"] <= 100.0

    def test_stochastic_oversold_signal_is_bool(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["stochastic"]
        )
        result = get_technical_analysis(inp)
        assert isinstance(result.signals["stoch_oversold"], bool)

    def test_vwap_in_last_values(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["vwap"]
        )
        result = get_technical_analysis(inp)
        assert "vwap" in result.last_values
        assert result.last_values["vwap"] > 0

    def test_vwap_signal_is_bool(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["vwap"]
        )
        result = get_technical_analysis(inp)
        assert isinstance(result.signals["price_above_vwap"], bool)

    def test_williams_r_in_last_values(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["williams_r"]
        )
        result = get_technical_analysis(inp)
        assert "williams_r" in result.last_values

    def test_williams_r_bounded(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["williams_r"]
        )
        result = get_technical_analysis(inp)
        assert -100.0 <= result.last_values["williams_r"] <= 0.0

    def test_williams_r_signals_present(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["williams_r"]
        )
        result = get_technical_analysis(inp)
        assert "williams_r_oversold" in result.signals
        assert "williams_r_overbought" in result.signals
        assert isinstance(result.signals["williams_r_oversold"], bool)
        assert isinstance(result.signals["williams_r_overbought"], bool)

    def test_ema_in_last_values(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["ema"]
        )
        result = get_technical_analysis(inp)
        assert "ema_12" in result.last_values
        assert "ema_26" in result.last_values

    def test_ema_values_positive(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL", start_date=START, end_date=END, indicators=["ema"]
        )
        result = get_technical_analysis(inp)
        assert result.last_values["ema_12"] > 0
        assert result.last_values["ema_26"] > 0

    def test_all_indicators_at_once(self, patched_factory):
        inp = TechnicalInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            indicators=[
                "rsi",
                "macd",
                "bollinger",
                "sma",
                "ema",
                "stochastic",
                "vwap",
                "williams_r",
                "adx",
                "obv",
            ],
        )
        result = get_technical_analysis(inp)
        for key in (
            "rsi_14",
            "bb_upper",
            "bb_lower",
            "ema_12",
            "ema_26",
            "stoch_k",
            "stoch_d",
            "vwap",
            "williams_r",
            "adx",
            "obv",
        ):
            assert key in result.last_values, f"missing key: {key}"


# ── fill_price integration smoke tests ────────────────────────────────────────
# One per directly-threaded tool: confirms fill_price="next_open" is actually
# wired through to run_strategy/backtest_grid (changes the result) and runs
# without error. The exact next_open economics are hand-verified separately
# in test_fill_price.py — these just prove the plumbing.
#
# patched_factory's Open is constructed as Close.shift(1) exactly (no
# overnight gap by design), under which "next_open" mathematically collapses
# to "close" (a correctness property of the engine, not a test bug — see
# test_fill_price.py). These tests need a fixture with genuine gaps instead.


@pytest.fixture
def gapped_factory(sample_close, monkeypatch: pytest.MonkeyPatch):
    """Like patched_factory, but Open has a genuine intraday gap from the
    prior Close (Open = 0.999 * that same bar's Close), so fill_price
    actually matters."""
    from unittest.mock import MagicMock

    close = sample_close
    spread = pd.Series(0.5, index=close.index)
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close + spread,
            "Low": close - spread,
            "Close": close,
            "Volume": pd.Series(1_000_000.0, index=close.index),
        }
    )
    provider = MagicMock()
    provider.get_ohlcv.return_value = df
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return provider


class TestFillPriceIntegration:
    def test_sma_backtest_next_open_differs_from_default(self, gapped_factory):
        base = run_sma_backtest(
            BacktestInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                strategy_type="sma_crossover",
                parameters={"fast_period": 5, "slow_period": 20},
            )
        )
        next_open = run_sma_backtest(
            BacktestInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                strategy_type="sma_crossover",
                parameters={"fast_period": 5, "slow_period": 20},
                fill_price="next_open",
            )
        )
        assert base.final_equity != pytest.approx(next_open.final_equity, abs=1e-6)

    def test_buy_and_hold_next_open_runs_and_differs(self, gapped_factory):
        base = run_buy_and_hold(
            BuyAndHoldInput(symbol="AAPL", start_date=START, end_date=END)
        )
        next_open = run_buy_and_hold(
            BuyAndHoldInput(
                symbol="AAPL", start_date=START, end_date=END, fill_price="next_open"
            )
        )
        assert base.final_equity != pytest.approx(next_open.final_equity, abs=1e-6)

    def test_compare_strategies_next_open_runs_and_differs(self, gapped_factory):
        base = compare_strategies(
            CompareStrategiesInput(symbol="AAPL", start_date=START, end_date=END)
        )
        next_open = compare_strategies(
            CompareStrategiesInput(
                symbol="AAPL", start_date=START, end_date=END, fill_price="next_open"
            )
        )
        base_returns = {c.strategy: c.total_return for c in base.strategies}
        next_open_returns = {c.strategy: c.total_return for c in next_open.strategies}
        assert base_returns != next_open_returns

    def test_backtest_optimization_next_open_runs_and_differs(self, gapped_factory):
        grid = {"fast_period": [5, 10], "slow_period": [30, 50]}
        base = run_backtest_optimization(
            BacktestOptInput(
                symbol="AAPL",
                strategy="sma_crossover",
                start_date=START,
                end_date=END,
                param_grid=grid,
            )
        )
        next_open = run_backtest_optimization(
            BacktestOptInput(
                symbol="AAPL",
                strategy="sma_crossover",
                start_date=START,
                end_date=END,
                param_grid=grid,
                fill_price="next_open",
            )
        )
        assert base.best_sharpe != pytest.approx(next_open.best_sharpe, abs=1e-9)

    def test_backtest_optimization_respects_requested_costs(self, gapped_factory):
        """
        Regression test (high-severity item 1): run_backtest_optimization
        must pass commission_pct/slippage_pct through to backtest_grid
        instead of silently using backtest_grid's own hardcoded defaults
        (0.001/0.0005) regardless of what the caller requested. A zero-cost
        request must produce materially different (better) results than
        the (nonzero-cost) default.
        """
        grid = {"fast_period": [5, 10], "slow_period": [30, 50]}
        default_cost = run_backtest_optimization(
            BacktestOptInput(
                symbol="AAPL",
                strategy="sma_crossover",
                start_date=START,
                end_date=END,
                param_grid=grid,
            )
        )
        zero_cost = run_backtest_optimization(
            BacktestOptInput(
                symbol="AAPL",
                strategy="sma_crossover",
                start_date=START,
                end_date=END,
                param_grid=grid,
                commission_pct=0.0,
                slippage_pct=0.0,
            )
        )
        assert zero_cost.best_sharpe != pytest.approx(
            default_cost.best_sharpe, abs=1e-9
        )
        assert zero_cost.best_return != pytest.approx(
            default_cost.best_return, abs=1e-9
        )
