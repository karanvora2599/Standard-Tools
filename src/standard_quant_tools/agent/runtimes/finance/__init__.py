"""The `finance` runtime's registry: what it advertises and what it can
execute, assembled from each tool module's own list so a tool cannot be
advertised without being dispatchable or the reverse.

WHY THIS IS ITS OWN RUNTIME. These are the questions about what holding a
book COSTS and what a stream of cash is WORTH: the financing, borrow and
cash drag on a set of positions, the margin a broker must hold against
them and how long a stress path leaves before the first call, the
currency exposure and carry of a multi-currency book, and the time value
of dated cash flows -- NPV, IRR, duration, a cost of capital and a
discounted valuation. They take POSITIONS or CASH FLOWS rather than
tickers or return streams, which is what separates them from `portfolio`
and `allocation`.

EVERY RATE IS THE CALLER'S. This library has no rates or FX provider, so
a benchmark rate, a spread, an FX rate or a discount rate arrives as an
argument with its day count, and a tool never substitutes one it does not
have. Positions use the nested model in `standard_quant_tools.finance`.

NOTHING MOVED HERE. Every tool is new; no donor runtime was depleted and
no MOVED_FROM entry was needed.

SENTENCE ONE IS THE WHOLE DESCRIPTION under `--tool-detail auto`, which
thins everything after it, so each tool module's first sentence stands on
its own and names the deliverable.
"""

from . import cash_flow_tools, currency_tools, margin_tools
from .cash_flow_tools import (
    analyze_cash_flows,
    estimate_cost_of_capital,
    value_cash_flows,
)
from .currency_tools import (
    analyze_book_carry,
    get_currency_exposure,
    price_fx_forward_hedge,
)
from .margin_tools import (  # noqa: F401
    analyze_margin_runway,
    compute_margin_requirement,
    estimate_financing_cost,
)

#: One module per area, each declaring its own TOOL_DEFS (name, description,
#: input model) and TOOL_DISPATCH (name -> (function, input model)). A new
#: area is one import and one entry here.
_MODULES: tuple = (cash_flow_tools, margin_tools, currency_tools)

TOOL_DEFS = [d for module in _MODULES for d in module.TOOL_DEFS]

TOOL_DISPATCH = {
    name: entry for module in _MODULES for name, entry in module.TOOL_DISPATCH.items()
}

#: Every tool here belongs to the one category this runtime owns.
TOOL_CATEGORY = {name: "finance" for name in TOOL_DISPATCH}

__all__ = [
    "TOOL_CATEGORY",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "analyze_cash_flows",
    "estimate_cost_of_capital",
    "value_cash_flows",
    "analyze_margin_runway",
    "compute_margin_requirement",
    "estimate_financing_cost",
    "analyze_book_carry",
    "get_currency_exposure",
    "price_fx_forward_hedge",
]
