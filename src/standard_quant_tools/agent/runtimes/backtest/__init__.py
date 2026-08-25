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
    MonteCarloSimulationInput,
    PairTradeBacktestInput,
    PortfolioSimulationInput,
    RegimeAdaptiveInput,
    RegimeAdaptiveWalkForwardInput,
    RobustnessDiagnosticsInput,
    SignalPanelBacktestInput,
    WalkForwardInput,
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
    run_walk_forward_backtest,
)

#: (name, description, input model) — the single source for both
#: the advertised schema and the dispatch table below.
TOOL_DEFS = [
    (
        "run_sma_backtest",
        "SMA crossover backtest.",
        BacktestInput,
    ),
    (
        "run_rsi_backtest",
        "RSI mean-reversion backtest.",
        BacktestInput,
    ),
    (
        "run_macd_backtest",
        "MACD crossover backtest.",
        BacktestInput,
    ),
    (
        "run_bollinger_backtest",
        "Bollinger Band mean-reversion backtest.",
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

TOOL_DISPATCH = {name: (globals()[name], model) for name, _d, model in TOOL_DEFS}

#: This runtime's slice of the library-wide routing taxonomy.
TOOL_CATEGORY = {
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

__all__ = [
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
