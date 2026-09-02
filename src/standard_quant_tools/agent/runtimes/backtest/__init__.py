"""The `backtest` runtime's registry: what it advertises and what it
can execute. The two are built from one list, so a tool cannot be
advertised without being dispatchable or the reverse."""

from standard_quant_tools.agent.models import (
    BacktestCompactInput,
    BacktestDiagnosticsInput,
    BacktestInput,
    BacktestOptInput,
    BuyAndHoldInput,
    CompareCostModelsInput,
    CompareStrategiesInput,
    CustomSignalBacktestInput,
    DrawdownTableInput,
    MatrixCell,
    MonteCarloSimulationInput,
    PairTradeBacktestInput,
    PortfolioSimulationInput,
    RegimeAdaptiveInput,
    RegimeAdaptiveWalkForwardInput,
    RobustnessDiagnosticsInput,
    SignalPanelBacktestInput,
    StrategyMatrixInput,
    StrategyMatrixResult,
    WalkForwardInput,
)

from .futures_tools import (  # noqa: F401
    FUTURES_TOOL_CATEGORY,
    FUTURES_TOOL_DEFS,
    FUTURES_TOOL_DISPATCH,
    run_futures_backtest,
    run_futures_hedge_backtest,
)
from .terminal_mc_tools import (  # noqa: F401
    TerminalMonteCarloInput,
    run_terminal_monte_carlo,
)
from .tools import (
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
from .trade_tools import (  # noqa: F401
    TRADE_TOOL_DEFS,
    TRADE_TOOL_DISPATCH,
    analyze_trade_clustering,
    compare_against_random,
    estimate_break_even_cost,
    get_exposure_attribution,
    run_monte_carlo_trade_paths,
)
from .validation_tools import (  # noqa: F401
    VALIDATION_TOOL_DEFS,
    VALIDATION_TOOL_DISPATCH,
    analyze_parameter_decay,
    build_purged_cv_splits,
    estimate_backtest_overfitting,
    get_deflated_sharpe_ratio,
    get_regime_stratified_performance,
    run_reality_check,
)

#: (name, description, input model) — the single source for both
#: the advertised schema and the dispatch table below.
TOOL_DEFS = [
    (
        "run_terminal_monte_carlo",
        "Monte Carlo that keeps only where the paths ENDED, so the "
        "simulation count is capped by wall-clock rather than by memory -- "
        "a million paths over a year is about 2 GB of path matrix for a "
        "handful of terminal quantiles, and this avoids allocating it. Same "
        "block bootstrap as run_monte_carlo_simulation. It says nothing "
        "about the journey: worst drawdown along the way and time "
        "underwater are not in here, and a benign distribution of endpoints "
        "can be reached by paths nobody could hold.",
        TerminalMonteCarloInput,
    ),
    (
        "run_strategy_matrix",
        "Every requested strategy against every requested ticker in one "
        "call, ranked. Fetches once per ticker and reuses the bars across "
        "strategies, so every cell is priced on identical data — which N "
        "separate calls cannot promise.",
        StrategyMatrixInput,
    ),
    (
        "run_sma_backtest",
        "Backtest a moving-average crossover: long when the fast average "
        "crosses above the slow one, flat or short when it crosses back. "
        "The simplest trend-following rule there is, which makes it the "
        "right BASELINE -- a more elaborate strategy that cannot beat it "
        "has not earned its complexity. One run at one parameter pair; "
        "run_backtest_optimization searches, and run_walk_forward_backtest "
        "checks the search survived out of sample.",
        BacktestInput,
    ),
    (
        "run_rsi_backtest",
        "Backtest an RSI mean-reversion rule: buy oversold, sell overbought. "
        "The counterpart to the crossover strategies -- it profits when "
        "prices revert and loses in a trend, so comparing the two on the "
        "same period says more about the period than either does alone. "
        "run_stationarity_tests and run_hurst_analysis say in advance "
        "which regime the sample is in.",
        BacktestInput,
    ),
    (
        "run_macd_backtest",
        "Backtest a MACD signal-line crossover. Trend-following like the SMA "
        "version but on the difference of two exponential averages, so it "
        "turns faster and trades more -- which makes it the one most "
        "sensitive to transaction costs. Run estimate_break_even_cost on "
        "the result: a MACD strategy that breaks even near its assumed "
        "cost is a cost assumption rather than an edge.",
        BacktestInput,
    ),
    (
        "run_bollinger_backtest",
        "Backtest a Bollinger Band reversion rule: buy the lower band, sell "
        "the upper. Mean-reverting like the RSI version but with a "
        "volatility-scaled threshold, so it trades less in calm markets "
        "and more in volatile ones. That scaling is the reason to prefer "
        "it to a fixed threshold, and the reason its trade count varies "
        "so much across periods.",
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
        "run_walk_forward_backtest",
        "Walk-forward validation: optimise in-sample, evaluate out-of-sample, return OOS stats.",
        WalkForwardInput,
    ),
    (
        "run_backtest_optimization",
        "Grid-search strategy parameters and return the top N combinations ranked by a chosen metric.",
        BacktestOptInput,
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
        "run_backtest_compact",
        "Compact backtest result: summary/risk/exposure/cost sub-reports plus equity-curve/trade-log artifact URIs, instead of embedding the full data inline like run_sma_backtest etc.",
        BacktestCompactInput,
    ),
    (
        "get_drawdown_table",
        "Every drawdown episode in a persisted equity curve (peak, trough, recovery, depth, duration), deepest first, from an equity_curve_uri rather than by re-running the backtest.",
        DrawdownTableInput,
    ),
    (
        "compare_cost_models",
        "Run one strategy under several cost assumptions on a single fetched signal series, and solve for the commission rate at which its total return reaches zero. Answers 'does this survive costs' in one call.",
        CompareCostModelsInput,
    ),
]

# The backtest_validation tools declared in validation_tools.py,
# concatenated rather than pasted so the group stays readable as a
# unit and cannot half-register.
TOOL_DEFS = TOOL_DEFS + VALIDATION_TOOL_DEFS

# The backtest_validation tools declared in trade_tools.py,
# concatenated rather than pasted so the group stays readable as a
# unit and cannot half-register.
TOOL_DEFS = TOOL_DEFS + TRADE_TOOL_DEFS

TOOL_DISPATCH = {name: (globals()[name], model) for name, _d, model in TOOL_DEFS}

#: This runtime's slice of the library-wide routing taxonomy.
TOOL_CATEGORY = {
    # Validation, not execution: it asks how a strategy could have turned
    # out, which is the question that category answers.
    "run_terminal_monte_carlo": "backtest_validation",
    "run_strategy_matrix": "backtest_execution",
    "run_sma_backtest": "backtest_execution",
    "run_rsi_backtest": "backtest_execution",
    "run_macd_backtest": "backtest_execution",
    "run_bollinger_backtest": "backtest_execution",
    "run_buy_and_hold": "backtest_execution",
    "compare_strategies": "backtest_execution",
    "run_regime_adaptive_backtest": "backtest_validation",
    "run_regime_adaptive_walkforward_backtest": "backtest_validation",
    "run_walk_forward_backtest": "backtest_validation",
    "run_backtest_optimization": "backtest_validation",
    "run_custom_signal_backtest": "custom_signal",
    "run_signal_panel_backtest": "custom_signal",
    "get_backtest_diagnostics": "backtest_validation",
    "run_portfolio_simulation": "backtest_execution",
    "run_pair_trade_backtest": "backtest_execution",
    "get_robustness_diagnostics": "backtest_validation",
    "run_monte_carlo_simulation": "backtest_validation",
    "run_backtest_compact": "backtest_execution",
    "get_drawdown_table": "backtest_validation",
    "compare_cost_models": "backtest_validation",
}

TOOL_DISPATCH.update(VALIDATION_TOOL_DISPATCH)
TOOL_CATEGORY.update({name: "backtest_validation" for name in VALIDATION_TOOL_DISPATCH})

# Its own file and its own category entry: run_futures_backtest is an
# EXECUTION tool like the others, not a validation one.
TOOL_DEFS = TOOL_DEFS + FUTURES_TOOL_DEFS
TOOL_DISPATCH.update(FUTURES_TOOL_DISPATCH)
TOOL_CATEGORY.update(FUTURES_TOOL_CATEGORY)

TOOL_DISPATCH.update(TRADE_TOOL_DISPATCH)
TOOL_CATEGORY.update({name: "backtest_validation" for name in TRADE_TOOL_DISPATCH})

__all__ = [
    "estimate_break_even_cost",
    "run_monte_carlo_trade_paths",
    "analyze_trade_clustering",
    "compare_against_random",
    "get_exposure_attribution",
    "get_deflated_sharpe_ratio",
    "estimate_backtest_overfitting",
    "build_purged_cv_splits",
    "run_reality_check",
    "get_regime_stratified_performance",
    "analyze_parameter_decay",
    "run_strategy_matrix",
    "TOOL_CATEGORY",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "compare_cost_models",
    "compare_strategies",
    "get_backtest_diagnostics",
    "get_drawdown_table",
    "get_robustness_diagnostics",
    "run_backtest_compact",
    "run_backtest_optimization",
    "run_bollinger_backtest",
    "run_buy_and_hold",
    "run_custom_signal_backtest",
    "run_futures_backtest",
    "run_futures_hedge_backtest",
    "run_macd_backtest",
    "run_monte_carlo_simulation",
    "run_pair_trade_backtest",
    "run_portfolio_simulation",
    "run_regime_adaptive_backtest",
    "run_regime_adaptive_walkforward_backtest",
    "run_rsi_backtest",
    "run_signal_panel_backtest",
    "run_sma_backtest",
    "run_walk_forward_backtest",
]
