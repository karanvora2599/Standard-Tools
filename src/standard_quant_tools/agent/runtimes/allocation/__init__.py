"""The `allocation` runtime's registry: what it advertises and what it can
execute, assembled from each tool module's own list so a tool cannot be
advertised without being dispatchable or the reverse.

WHY THIS IS ITS OWN RUNTIME. `portfolio` builds a book out of ASSETS: it
takes tickers or an asset covariance and answers with weights per name.
The questions here take RETURN STREAMS -- a strategy's, a sleeve's, a
model's -- and ask how much capital each deserves, whether a rule for
choosing that would have worked out of sample, how hard to lever it, what
an overlay does to it, how much money it can take before impact eats it,
and what a book with deposits and withdrawals actually returned. None of
them takes a ticker, so none of them fits `portfolio`'s grouping, and in
`portfolio` they would have thinned its flagship tools out of the MCP
listing.

NOTHING MOVED HERE. Every tool is new; no donor runtime was depleted and
no MOVED_FROM entry was needed. Streams come in as references or dated
values (`agent/runtimes/streams.py`) and go out as references, so a result
from `backtest` or `modeling` crosses into this runtime by value.

SENTENCE ONE IS THE WHOLE DESCRIPTION under `--tool-detail auto`, which
thins everything after it, so each tool module's first sentence stands on
its own and names the deliverable.
"""

from . import (
    attribution_tools,
    book_return_tools,
    leverage_tools,
    model_sleeve_tools,
    selection_tools,
    sleeve_tools,
)
from .attribution_tools import attribute_returns_brinson
from .book_return_tools import measure_book_returns
from .leverage_tools import (
    apply_risk_overlay,
    get_kelly_report,
    get_strategy_capacity_curve,
)
from .model_sleeve_tools import build_model_sleeves
from .selection_tools import select_allocation_rule
from .sleeve_tools import (
    allocate_across_sleeves,
    combine_return_streams,
    evaluate_allocation_rule,
)

#: One module per area, each declaring its own TOOL_DEFS (name, description,
#: input model) and TOOL_DISPATCH (name -> (function, input model)). A new
#: area is one import and one entry here.
_MODULES: tuple = (
    sleeve_tools,
    leverage_tools,
    attribution_tools,
    book_return_tools,
    model_sleeve_tools,
    selection_tools,
)

TOOL_DEFS = [d for module in _MODULES for d in module.TOOL_DEFS]

TOOL_DISPATCH = {
    name: entry for module in _MODULES for name, entry in module.TOOL_DISPATCH.items()
}

#: Every tool here belongs to the one category this runtime owns.
TOOL_CATEGORY = {name: "allocation" for name in TOOL_DISPATCH}

__all__ = [
    "TOOL_CATEGORY",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "allocate_across_sleeves",
    "apply_risk_overlay",
    "attribute_returns_brinson",
    "build_model_sleeves",
    "combine_return_streams",
    "evaluate_allocation_rule",
    "get_kelly_report",
    "get_strategy_capacity_curve",
    "measure_book_returns",
    "select_allocation_rule",
]
