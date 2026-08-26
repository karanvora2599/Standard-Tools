"""The `portfolio` runtime's registry: what it advertises and what it
can execute. The two are built from one list, so a tool cannot be
advertised without being dispatchable or the reverse."""

from standard_quant_tools.agent.models import (
    CapacityReportInput,
    EstimateCovarianceInput,
    EstimateTradeCostInput,
    LiquidityAnalysisInput,
    LiquidityEventsInput,
    MicrostructureInput,
    PlanRebalanceInput,
    PortfolioOptimizationInput,
    PositionSizerInput,
    RiskAttributionInput,
    SpreadProxyCheckInput,
    StressTestInput,
    TradeProfileInput,
)

from .construction_tools import (  # noqa: F401
    CONSTRUCTION_TOOL_DEFS,
    CONSTRUCTION_TOOL_DISPATCH,
    analyze_concentration,
    get_factor_exposure_budget,
    get_liquidity_adjusted_var,
    get_marginal_risk_contribution,
    optimize_hierarchical_risk_parity,
    optimize_max_diversification,
    optimize_risk_parity,
    run_portfolio_scenarios,
)
from .tools import (
    check_spread_proxy,
    detect_liquidity_events,
    estimate_covariance,
    estimate_trade_cost,
    get_capacity_report,
    get_liquidity_metrics,
    get_microstructure_metrics,
    get_portfolio_risk_attribution,
    get_position_size,
    get_trade_profile,
    plan_rebalance,
    run_portfolio_optimization,
    run_stress_test,
)

#: (name, description, input model) — the single source for both
#: the advertised schema and the dispatch table below.
TOOL_DEFS = [
    (
        "plan_rebalance",
        "A day-by-day path from the weights you hold to the weights you want. Every optimizer here returns a target vector and implicitly assumes you arrive instantly and for free; trading fast costs market impact and trading slow means holding the portfolio you were trying to leave, so this returns the SCHEDULE and both costs rather than one number. Surfaces what nothing else does: a target weight the market cannot supply, with the number of days it would really take.",
        PlanRebalanceInput,
    ),
    (
        "estimate_covariance",
        "A covariance matrix plus the diagnostics that say whether to trust it. Shrinkage is the ANSWER to the conditioning warnings the optimizer already emits rather than a caveat about them: a covariance over N assets has N(N+1)/2 parameters, and mean-variance optimization is an error-maximizer over its worst-estimated directions. Read observations_per_parameter and condition_number before the matrix. Returns it annualized.",
        EstimateCovarianceInput,
    ),
    (
        "run_portfolio_optimization",
        "Produce portfolio weights via Markowitz mean-variance (max_sharpe/min_volatility/target_return/target_volatility), risk parity, or Black-Litterman — unlike get_portfolio_analysis, which only scores weights already chosen.",
        PortfolioOptimizationInput,
    ),
    (
        "get_portfolio_risk_attribution",
        "Deep portfolio risk decomposition: MCR per asset, PCA attribution, optional factor model.",
        RiskAttributionInput,
    ),
    (
        "run_stress_test",
        "Replay a portfolio's weights against a named historical crash window (or custom date range) using real historical returns.",
        StressTestInput,
    ),
    (
        "get_position_size",
        "How large to trade, from a stop distance measured in ATR and an "
        "account risk budget. Answers the question a signal does not: a "
        "correct direction sized wrong loses money. Kelly is optional and "
        "is a CEILING rather than a target -- full Kelly maximizes "
        "long-run growth and produces drawdowns almost nobody holds "
        "through, which is why half-Kelly is the common practice.",
        PositionSizerInput,
    ),
    (
        "get_capacity_report",
        "How much account size a target-weight portfolio can support before positions become too large relative to each ticker's own trading volume, plus days-to-liquidate and sector exposure.",
        CapacityReportInput,
    ),
    (
        "get_liquidity_metrics",
        "Amihud illiquidity ratio and Corwin-Schultz spread estimator per ticker — OHLCV-derived proxies for market depth and bid/ask spread.",
        LiquidityAnalysisInput,
    ),
    (
        "estimate_trade_cost",
        "Itemized cost of one hypothetical trade under a composed cost model: commission (pct/per-share/directional/maker-taker), spread (fixed bps or a fraction of the bar's range), square-root market impact, short borrow and margin interest. No market data needed.",
        EstimateTradeCostInput,
    ),
]

# The portfolio_risk tools declared in construction_tools.py,
# concatenated rather than pasted so the group stays readable as a
# unit and cannot half-register.
TOOL_DEFS = TOOL_DEFS + CONSTRUCTION_TOOL_DEFS

TOOL_DISPATCH = {name: (globals()[name], model) for name, _d, model in TOOL_DEFS}

#: This runtime's slice of the library-wide routing taxonomy.
TOOL_CATEGORY = {
    "run_portfolio_optimization": "portfolio_risk",
    "plan_rebalance": "portfolio_risk",
    "estimate_covariance": "portfolio_risk",
    "get_portfolio_risk_attribution": "portfolio_risk",
    "run_stress_test": "portfolio_risk",
    "get_position_size": "portfolio_risk",
    "get_capacity_report": "portfolio_risk",
    "get_liquidity_metrics": "portfolio_risk",
    "estimate_trade_cost": "portfolio_risk",
}

TOOL_DISPATCH.update(CONSTRUCTION_TOOL_DISPATCH)
TOOL_CATEGORY.update({name: "portfolio_risk" for name in CONSTRUCTION_TOOL_DISPATCH})

__all__ = [
    "optimize_max_diversification",
    "get_marginal_risk_contribution",
    "run_portfolio_scenarios",
    "optimize_risk_parity",
    "optimize_hierarchical_risk_parity",
    "get_factor_exposure_budget",
    "analyze_concentration",
    "get_liquidity_adjusted_var",
    "TOOL_CATEGORY",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "check_spread_proxy",
    "estimate_trade_cost",
    "get_capacity_report",
    "get_liquidity_metrics",
    "get_microstructure_metrics",
    "get_portfolio_risk_attribution",
    "get_position_size",
    "get_trade_profile",
    "run_portfolio_optimization",
    "run_stress_test",
]
