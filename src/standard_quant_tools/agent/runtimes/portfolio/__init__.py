"""The `portfolio` runtime's registry: what it advertises and what it
can execute. The two are built from one list, so a tool cannot be
advertised without being dispatchable or the reverse."""

from standard_quant_tools.agent.models import (
    CapacityReportInput,
    EstimateTradeCostInput,
    LiquidityAnalysisInput,
    MicrostructureInput,
    PortfolioOptimizationInput,
    PositionSizerInput,
    RiskAttributionInput,
    SpreadProxyCheckInput,
    StressTestInput,
    TradeProfileInput,
)

from .tools import (
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

#: (name, description, input model) — the single source for both
#: the advertised schema and the dispatch table below.
TOOL_DEFS = [
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

TOOL_DISPATCH = {name: (globals()[name], model) for name, _d, model in TOOL_DEFS}

#: This runtime's slice of the library-wide routing taxonomy.
TOOL_CATEGORY = {
    "run_portfolio_optimization": "portfolio_risk",
    "get_portfolio_risk_attribution": "portfolio_risk",
    "run_stress_test": "portfolio_risk",
    "get_position_size": "portfolio_risk",
    "get_capacity_report": "portfolio_risk",
    "get_liquidity_metrics": "portfolio_risk",
    "get_microstructure_metrics": "microstructure",
    "get_trade_profile": "microstructure",
    "check_spread_proxy": "microstructure",
    "estimate_trade_cost": "portfolio_risk",
}

__all__ = [
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
