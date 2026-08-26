"""The `microstructure` runtime: what the market will charge you to trade.

Eleven tools in two halves that answer the same question at two data
fidelities. Four MEASURE spreads and order flow from ticks and refuse to run
without a tick feed. Seven ESTIMATE the same quantities from OHLCV, which is
the normal case, and each says what it is a proxy for and how it fails.

WHY IT LEFT `portfolio`. Four tools is not a runtime, it is overhead, so
these sat inside `portfolio` while there were four of them. At eleven the
grouping is worth its own boundary: `portfolio` turns a view into a
position, while these price the road to it -- and an agent sizing a position
rarely wants a tick-feed refusal in its listing. `portfolio_risk` keeps
fourteen tools, so both sides clear the floor of eight.

The four tick tools MOVED here, which breaks anything scoped to
`portfolio`. MOVED_FROM turns that into an instruction rather than an
"unknown tool" that reads like a hallucination.
"""

from standard_quant_tools.agent.models import (
    LiquidityEventsInput,
    MicrostructureInput,
    SpreadProxyCheckInput,
    TradeProfileInput,
)
from standard_quant_tools.agent.runtimes.portfolio.tools import (
    check_spread_proxy,
    detect_liquidity_events,
    get_microstructure_metrics,
    get_trade_profile,
)

from .estimator_tools import (  # noqa: F401
    ESTIMATOR_TOOL_DEFS,
    ESTIMATOR_TOOL_DISPATCH,
    estimate_corwin_schultz_spread,
    estimate_kyle_lambda,
    estimate_roll_spread,
    estimate_vpin,
    get_amihud_illiquidity,
    get_intraday_volume_profile,
    get_order_flow_imbalance,
)

#: The four that need a tick feed. Their descriptions are unchanged from
#: when they lived in `portfolio` -- the boundary moved, not the tools.
TICK_TOOL_DEFS = [
    (
        "get_microstructure_metrics",
        "Quoted and effective spread MEASURED from trades and quotes, with the effective spread split into the realized (liquidity-provider) and impact (price-move) halves, plus Lee-Ready signed order flow. Needs a tick feed and refuses without one rather than approximating from bars.",
        MicrostructureInput,
    ),
    (
        "get_trade_profile",
        "How volume distributes across trade sizes and times of day, from tick data. Answers whether the liquidity is in a few large prints or many small ones, which decides whether a large order can hide.",
        TradeProfileInput,
    ),
    (
        "detect_liquidity_events",
        "CUSUM change detection over tick-derived liquidity channels -- when the spread, depth, trade intensity or signed flow regime CHANGED, rather than what it is on average. Declares every channel it knows about, including the ones this feed cannot supply.",
        LiquidityEventsInput,
    ),
    (
        "check_spread_proxy",
        "Measures the spread from ticks and compares it against the OHLCV estimate, so the proxy's error on THIS name is a number rather than an assumption. The Corwin-Schultz estimate is what a bar-only session would have used.",
        SpreadProxyCheckInput,
    ),
]

TOOL_DEFS = TICK_TOOL_DEFS + ESTIMATOR_TOOL_DEFS

TOOL_DISPATCH = {
    "get_microstructure_metrics": (get_microstructure_metrics, MicrostructureInput),
    "get_trade_profile": (get_trade_profile, TradeProfileInput),
    "detect_liquidity_events": (detect_liquidity_events, LiquidityEventsInput),
    "check_spread_proxy": (check_spread_proxy, SpreadProxyCheckInput),
}
TOOL_DISPATCH.update(ESTIMATOR_TOOL_DISPATCH)

TOOL_CATEGORY = {name: "microstructure" for name in TOOL_DISPATCH}

__all__ = [
    "TOOL_CATEGORY",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "check_spread_proxy",
    "detect_liquidity_events",
    "estimate_corwin_schultz_spread",
    "estimate_kyle_lambda",
    "estimate_roll_spread",
    "estimate_vpin",
    "get_amihud_illiquidity",
    "get_intraday_volume_profile",
    "get_microstructure_metrics",
    "get_order_flow_imbalance",
    "get_trade_profile",
]
