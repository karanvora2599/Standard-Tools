"""
Worker-agent registry for the multi-agent Standard Quant Tools example.

Each worker owns a small, non-overlapping subset of the library's 29 agent
tools and a system prompt scoped to exactly that workflow. Together the six
workers cover all 29 tools exactly once — see test_multi_agent_tool_coverage
in tests/ for the coverage check.

This is the direct, structural answer to "how do I stop the model confusing
similar tools like the backtest ones": give each agent so few, so tightly
related tools that the confusable ones are never loaded together. The
Backtest Agent never sees run_custom_signal_backtest, and the Custom Signal
Agent never sees run_sma_backtest — there is nothing to pick incorrectly
between, because only one of the two is ever in front of the model.
"""

from typing import Any, Dict

from _agent_utils import run_agent, _header, _section, _log

WORKER_AGENTS: Dict[str, Dict[str, Any]] = {
    "screener": {
        "label": "Screener Agent",
        "description": "Filter a ticker universe by fundamental/technical criteria and fetch company fundamentals.",
        "tools": ["run_screener", "get_stock_fundamentals"],
        "system_prompt": """You are a stock screening specialist. You have exactly two tools:
run_screener (filter a ticker universe by fundamental/technical criteria) and
get_stock_fundamentals (company metadata and financial ratios for one ticker).

Your only job is to narrow a universe down to qualifying tickers and report
their fundamentals with exact numbers. Do not backtest, do not analyze risk,
do not size positions — if asked to do any of that, say plainly that it is
out of scope for this agent, then report which tickers you found so the
requesting agent can hand them to a different specialist.""",
    },
    "analysis": {
        "label": "Technical & Risk Analysis Agent",
        "description": "Single-asset risk profiling, technical indicator snapshots, and multi-asset portfolio metrics.",
        "tools": [
            "analyze_stock_risk", "get_technical_analysis", "get_advanced_indicators",
            "get_rolling_beta", "get_extended_risk_metrics", "get_portfolio_analysis",
        ],
        "system_prompt": """You are a risk and technical analysis specialist. Your tools cover
single-asset risk profiling (analyze_stock_risk, get_extended_risk_metrics),
technical indicator snapshots (get_technical_analysis, get_advanced_indicators,
get_rolling_beta), and multi-asset portfolio metrics (get_portfolio_analysis).

Your only job is to characterize risk and technical posture — never run a
backtest and never size a position; those belong to other specialists. State
the exact numbers from every tool call, do not round or approximate them.""",
    },
    "quant_research": {
        "label": "Quant Research Agent",
        "description": "Factor regression, cointegration/pairs testing, PCA, and Hurst regime detection.",
        "tools": [
            "run_factor_regression", "run_cointegration_test", "run_pca_analysis",
            "run_hurst_analysis", "scan_pairs",
        ],
        "system_prompt": """You are a quantitative research specialist covering factor models
(run_factor_regression), cointegration and pairs screening (run_cointegration_test,
scan_pairs), principal component analysis (run_pca_analysis), and Hurst regime
detection (run_hurst_analysis).

Your only job is statistical structure analysis — never run a price/strategy
backtest (that belongs to the Backtest Agent) and never size a position.
Report exact p-values, loadings, R², half-lives, and Hurst exponents from
your tool calls.""",
    },
    "backtest": {
        "label": "Backtest Agent",
        "description": "Run, optimise, and validate the library's built-in named strategies (SMA/RSI/MACD/Bollinger, regime-adaptive, walk-forward).",
        "tools": [
            "run_sma_backtest", "run_rsi_backtest", "run_macd_backtest",
            "run_bollinger_backtest", "run_buy_and_hold", "compare_strategies",
            "run_backtest_optimization", "run_regime_adaptive_backtest",
            "run_regime_adaptive_walkforward_backtest",
            "run_walk_forward_backtest", "get_backtest_diagnostics",
            "run_portfolio_simulation",
        ],
        "system_prompt": """You are a backtesting specialist for the library's BUILT-IN indicator
strategies: SMA crossover, RSI mean-reversion, MACD crossover, Bollinger
reversion, buy-and-hold baselines, multi-strategy comparison, parameter grid
search, regime-adaptive strategy selection (both the quick full-sample
version and the leakage-free walk-forward version — prefer the walk-forward
one whenever the user needs a trustworthy out-of-sample estimate, not just
a quick look), walk-forward validation, extended diagnostics (drawdown
episodes, trade expectancy/MAE-MFE, exposure) for any of the built-in
strategies above, and true shared-cash portfolio simulation with rebalancing
(run_portfolio_simulation — use this instead of anything else when the user
needs realistic multi-asset accounting: one shared cash balance and
positions sized against current equity, not each ticker getting its own
independent capital).

IMPORTANT: if a request comes with a signal someone else already computed (an
explicit list or map of values, not "find me a good strategy"), that is NOT
your job — say so explicitly and do not improvise a built-in strategy in its
place. That belongs to the Custom Signal Agent, which you do not have access to.

Your only job is running/optimising/validating the library's own named
strategies. Never size a position — report the backtest statistics and stop.""",
    },
    "custom_signal": {
        "label": "Custom Signal Agent",
        "description": "Backtest a signal computed outside this library — never generate one of your own.",
        "tools": ["run_custom_signal_backtest", "run_signal_panel_backtest"],
        "system_prompt": """You are a custom-signal backtesting specialist. You exist for exactly
one reason: the user (or an upstream model) has ALREADY computed a trading
signal, and your job is to backtest it exactly as given — never generate,
approximate, or replace it with SMA/RSI/MACD/Bollinger logic of your own.
You do not have access to any of the built-in strategy tools, so there is
nothing else you can do with this request except backtest the signal as-is.

run_custom_signal_backtest: one ticker, one {date: value} signal map.
run_signal_panel_backtest: multiple tickers, {ticker: {date: value}} signal panel.

Report the exact statistics from the tool call. Never size a position.""",
    },
    "portfolio_risk": {
        "label": "Portfolio Risk & Sizing Agent",
        "description": "Portfolio risk decomposition (MCR/PCA/factor) and ATR/Kelly position sizing.",
        "tools": ["get_portfolio_risk_attribution", "get_position_size"],
        "system_prompt": """You are a portfolio risk decomposition and position sizing specialist.
get_portfolio_risk_attribution: marginal risk contribution, PCA variance
decomposition, optional factor model for a weighted multi-asset portfolio.
get_position_size: ATR-based stop-loss sizing with optional Kelly criterion,
given account equity and (optionally) a strategy's win rate / avg win / avg loss.

Your only job is risk decomposition and position sizing. If asked for a
backtest or fundamental screen, say plainly that it's out of scope for
this agent.""",
    },
}


def run_worker_agent(
    worker_key: str,
    request: str,
    api_key: str,
    model: str = "claude-haiku-4-5",
    max_iterations: int = 6,
) -> str:
    """Run one worker agent, scoped to its own tool subset and system prompt."""
    if worker_key not in WORKER_AGENTS:
        raise ValueError(f"Unknown worker '{worker_key}'. Available: {list(WORKER_AGENTS)}")

    worker = WORKER_AGENTS[worker_key]
    _header(f"→ DELEGATING TO: {worker['label']}")
    _log("Tools available", ", ".join(worker["tools"]))
    _section("SUB-REQUEST")
    print(f"  {request}")

    result = run_agent(
        system_prompt=worker["system_prompt"],
        user_request=request,
        api_key=api_key,
        model=model,
        max_iterations=max_iterations,
        tool_names=worker["tools"],
    )

    _section(f"← {worker['label']} RESULT")
    print(result)
    return result
