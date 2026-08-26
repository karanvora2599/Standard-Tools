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
    optimize_hierarchical_risk_parity,
    optimize_risk_parity,
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
        "detect_liquidity_events",
        "Which part of the market changed, not merely that it did. Runs a CUSUM change detector across several channels — spread, effective spread, signed volume, trade intensity, realized volatility, mid return — and reports which broke and how badly. Price is the channel that moves LAST, so a report where the spread and flow fired while the mid did not is the ordinary sequence rather than a contradiction. Channels needing an order book are declared and REFUSED by name rather than dropped, because a missing row reads as a quiet channel. Needs a tick-capable provider.",
        LiquidityEventsInput,
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
        "ATR-based position sizing with optional Kelly criterion.",
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
        "get_microstructure_metrics",
        "Measured (not estimated) spreads from tick data: quoted and effective spread, the realized/impact decomposition, signed order flow and quote imbalance. Requires a provider with a tick feed — call describe_data_capabilities first.",
        MicrostructureInput,
    ),
    (
        "get_trade_profile",
        "How a symbol's volume is distributed across trade sizes (quantile buckets) and times of day. Distinguishes a book that trades in blocks from one that trades in odd lots at the same ADV. Requires a tick feed.",
        TradeProfileInput,
    ),
    (
        "check_spread_proxy",
        "Measure the spread from ticks, compute get_liquidity_metrics' OHLCV proxy for the same name, and report which way the proxy errs — understating it means backtests priced from it have been charging too little. Requires a tick feed.",
        SpreadProxyCheckInput,
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
    "get_microstructure_metrics": "microstructure",
    "get_trade_profile": "microstructure",
    "detect_liquidity_events": "microstructure",
    "check_spread_proxy": "microstructure",
    "estimate_trade_cost": "portfolio_risk",
}

TOOL_DISPATCH.update(CONSTRUCTION_TOOL_DISPATCH)
TOOL_CATEGORY.update({name: "portfolio_risk" for name in CONSTRUCTION_TOOL_DISPATCH})

__all__ = [
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
