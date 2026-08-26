"""
Agent-callable tool functions — designed for LLM function calling.

THIS MODULE IS THE UNION, NOT THE SOURCE. Every tool now lives in the
runtime that owns it (`agent/runtimes/research`, `/backtest`, `/portfolio`,
`/meta`), each with its own dispatch table that refuses what it does not
own. What stays here is the library-wide view those runtimes add up to:
`TOOL_CATEGORY`, `_TOOL_DISPATCH`, `get_agent_tools()` and `dispatch()`.

It stays because a great deal already depends on it — the thirty-three
example scripts, the multi-agent orchestrator, the MCP catalog and several
hundred tests all import from here — and because a union view is genuinely
useful for a caller that has decided it wants the whole surface.

But `dispatch()` here is UNSCOPED by construction: it will run any of the
62 tools regardless of what was advertised to the model. That is the exact
gap the runtimes exist to close, so an agent that is meant to be scoped
should be handed a runtime rather than this module:

    from standard_quant_tools.agent.runtimes import resolve
    research = resolve("research")
    research.dispatch("run_sma_backtest", {...})   # refused, by name

All inputs/outputs use Pydantic models for clean JSON serialization.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from standard_quant_tools import audit
from standard_quant_tools._jsonsafe import sanitize_for_json
from standard_quant_tools.agent.runtimes import backtest as _backtest
from standard_quant_tools.agent.runtimes import meta as _meta
from standard_quant_tools.agent.runtimes import portfolio as _portfolio
from standard_quant_tools.agent.runtimes import research as _research
from standard_quant_tools.agent.runtimes.backtest import (
    compare_cost_models,
    compare_strategies,
    get_backtest_diagnostics,
    get_drawdown_table,
    get_robustness_diagnostics,
    run_backtest_compact,
    run_backtest_optimization,
    run_bollinger_backtest,
    run_buy_and_hold,
    run_custom_signal_backtest,
    run_macd_backtest,
    run_monte_carlo_simulation,
    run_pair_trade_backtest,
    run_portfolio_simulation,
    run_regime_adaptive_backtest,
    run_regime_adaptive_walkforward_backtest,
    run_rsi_backtest,
    run_signal_panel_backtest,
    run_sma_backtest,
    run_strategy_matrix,
    run_walk_forward_backtest,
)
from standard_quant_tools.agent.runtimes.meta import (
    compare_decisions,
    convert_reference,
    describe_artifact,
    describe_data_capabilities,
    describe_temporal_contract,
    describe_reference,
    describe_tool,
    explain_decision,
    export_audit_bundle,
    list_reference_kinds,
    list_strategies,
    list_stress_scenarios,
    replay_decision,
    validate_tool_call,
    verify_audit_integrity,
)
from standard_quant_tools.agent.runtimes.portfolio import (
    check_spread_proxy,
    estimate_trade_cost,
    get_capacity_report,
    get_liquidity_metrics,
    get_microstructure_metrics,
    get_portfolio_risk_attribution,
    get_position_size,
    get_trade_profile,
    run_portfolio_optimization,
    run_stress_test,
)
from standard_quant_tools.agent.runtimes.research import (
    analyze_stock_risk,
    get_advanced_indicators,
    get_correlation_analysis,
    get_data_quality_report,
    get_extended_risk_metrics,
    get_implied_volatility,
    get_option_pricing,
    get_portfolio_analysis,
    get_rally_signal,
    get_rolling_beta,
    get_stock_fundamentals,
    get_tail_risk_metrics,
    get_technical_analysis,
    get_technical_panel,
    get_volatility_estimators,
    run_cointegration_test,
    run_factor_regression,
    run_garch_volatility_forecast,
    run_hurst_analysis,
    run_kalman_hedge_ratio,
    run_pca_analysis,
    run_screener,
    scan_pairs,
)

logger = logging.getLogger(__name__)

_RUNTIME_MODULES = [
    _research,
    _backtest,
    _portfolio,
    _meta,
]

#: Every tool in the library, from each runtime's own declaration —
#: assembled rather than restated, so this cannot drift from them.
TOOL_DEFS = [d for module in _RUNTIME_MODULES for d in module.TOOL_DEFS]

#: The routing taxonomy, unioned from the runtimes. Unchanged in
#: meaning: a category still says which tools suit a request, while a
#: runtime says which tools a caller may execute.
TOOL_CATEGORY: Dict[str, str] = {
    name: category
    for module in _RUNTIME_MODULES
    for name, category in module.TOOL_CATEGORY.items()
}

_TOOL_DISPATCH: Dict[str, Any] = {
    name: entry
    for module in _RUNTIME_MODULES
    for name, entry in module.TOOL_DISPATCH.items()
}


def get_agent_tools(
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns tool definitions formatted for OpenAI / Anthropic function calling.
    All tools have Pydantic-derived schemas — no manual JSON authoring required.

    Args:
        categories: Optional list of `TOOL_CATEGORY` values to filter to —
            e.g. `["screener"]` returns only the 2 screener tools. `None`
            (the default) returns every tool in every runtime. An unknown
            category name is silently ignored rather than raising, since a
            router's job is to narrow *when confident*, not to be a strict
            validator — see `agent/router.py`.

    NOTE: this narrows the ADVERTISED list only. `dispatch()` below will
    still execute anything in the library, which is why an agent that must
    be scoped should be given a runtime (`agent.runtimes.resolve`) whose
    own `get_tools()` and `dispatch()` agree with each other.
    """
    tool_defs = TOOL_DEFS
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

# An explicit __all__ is what makes the imports above RE-exports rather
# than unused ones. Without it a linter prunes all 62 names and this
# module silently stops being a facade -- which is exactly what happened
# the first time this file was generated.
__all__ = [
    "run_strategy_matrix",
    "describe_tool",
    "validate_tool_call",
    "convert_reference",
    "describe_reference",
    "list_reference_kinds",
    "TOOL_CATEGORY",
    "TOOL_DEFS",
    "dispatch",
    "get_agent_tools",
    "analyze_stock_risk",
    "check_spread_proxy",
    "compare_cost_models",
    "compare_decisions",
    "compare_strategies",
    "describe_artifact",
    "describe_data_capabilities",
    "describe_temporal_contract",
    "estimate_trade_cost",
    "explain_decision",
    "export_audit_bundle",
    "get_advanced_indicators",
    "get_backtest_diagnostics",
    "get_capacity_report",
    "get_correlation_analysis",
    "get_data_quality_report",
    "get_drawdown_table",
    "get_extended_risk_metrics",
    "get_implied_volatility",
    "get_liquidity_metrics",
    "get_microstructure_metrics",
    "get_option_pricing",
    "get_portfolio_analysis",
    "get_portfolio_risk_attribution",
    "get_position_size",
    "get_rally_signal",
    "get_robustness_diagnostics",
    "get_rolling_beta",
    "get_stock_fundamentals",
    "get_tail_risk_metrics",
    "get_technical_analysis",
    "get_technical_panel",
    "get_trade_profile",
    "get_volatility_estimators",
    "list_strategies",
    "list_stress_scenarios",
    "replay_decision",
    "run_backtest_compact",
    "run_backtest_optimization",
    "run_bollinger_backtest",
    "run_buy_and_hold",
    "run_cointegration_test",
    "run_custom_signal_backtest",
    "run_factor_regression",
    "run_garch_volatility_forecast",
    "run_hurst_analysis",
    "run_kalman_hedge_ratio",
    "run_macd_backtest",
    "run_monte_carlo_simulation",
    "run_pair_trade_backtest",
    "run_pca_analysis",
    "run_portfolio_optimization",
    "run_portfolio_simulation",
    "run_regime_adaptive_backtest",
    "run_regime_adaptive_walkforward_backtest",
    "run_rsi_backtest",
    "run_screener",
    "run_signal_panel_backtest",
    "run_sma_backtest",
    "run_stress_test",
    "run_walk_forward_backtest",
    "scan_pairs",
    "verify_audit_integrity",
]
