"""
Parallel tool runtimes: the execution boundary an agent is scoped to.

WHY A BOUNDARY AND NOT A FILTER. `get_agent_tools(categories=...)` has
always been able to narrow the SCHEMA list handed to a model, but
`dispatch()` knew every tool regardless. An agent given two screener tools
that hallucinated `run_walk_forward_backtest` therefore got a successful
result rather than an error -- the narrowing was advisory at the schema
layer and absent at the execution layer, so a wrong guess was rewarded.
(The MCP server did enforce it; nothing else did, which meant every
`Implementation/*` script and the multi-agent orchestrator ran with a fake
boundary.)

A runtime closes that. Each one holds its OWN dispatch table containing
only its own tools, so a name from another runtime is not merely
discouraged -- it is unroutable. This is the same guarantee the modeling
runtime has had since it shipped, generalized to the rest of the surface.

RUNTIME IS NOT CATEGORY. `TOOL_CATEGORY` survives unchanged as the routing
taxonomy: it still drives `router.py`, the MCP `--categories` flag, and the
twelve workers in `Multi_Agent_Implementation`. Categories are a hint about
which tools suit a request; runtimes are a hard statement about which tools
a caller may execute. Several categories live inside one runtime, and the
grouping is deliberately coarse -- a runtime holding two tools is overhead
rather than isolation, so nothing here has fewer than eight.

THE RUNTIMES PARTITION THE SURFACE. Every tool belongs to exactly one, and
a test pins it. Duplicating a convenient tool into a second runtime would
dissolve the boundary at exactly the points where it matters most, so a
caller needing tools from two runtimes composes them explicitly with
`combine()` and can see in the code that it has widened its own scope.

THE ERROR HAS TO BE RECOVERABLE. When a caller asks a runtime for a tool
that exists elsewhere, the failure names the runtime that actually owns it.
"Unknown tool" alone would leave a model unable to tell a hallucination
from a scoping mistake, and it would guess again.

RUNTIMES ISOLATE EXECUTION, NOT DATA. This is the distinction that keeps
the boundary usable. Every real workflow spans runtimes -- screen in
`research`, backtest in `backtest`, size in `portfolio`, hand a model's
predictions from `modeling` to a backtest -- and a boundary that also
blocked RESULTS would make the multi-agent orchestrator impossible.

So results cross by VALUE, never by shared dispatch table: an artifact URI
written by one runtime and read by another, an identifier (dataset_id,
model_id, request_id), or the plain JSON dict every tool already returns.
That is strictly better than sharing a table, for three reasons. A value is
serializable, so it survives the process boundary between two agents in the
orchestrator. It is auditable, because the handoff shows up in the decision
log as an input to the second call. And it cannot smuggle execution rights:
holding an equity-curve URI lets you DESCRIBE that curve from `meta`, and
still does not let you run `get_drawdown_table`, which lives in `backtest`.

An agent that genuinely needs two runtimes uses `combine()`. The widening
is then visible in the code that asked for it, rather than being the
silent default it used to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from standard_quant_tools._jsonsafe import sanitize_for_json
from standard_quant_tools.audit.dispatch import _run_and_record

#: runtime name -> the TOOL_CATEGORY values it owns. The grouping rule is
#: "could one agent plausibly be scoped to this for a whole session", which
#: is why screening sits with analysis (you screen in order to analyze) and
#: why the OHLCV liquidity proxies sit with the tick measurements (they are
#: the same question at two data fidelities).
RUNTIME_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "research": ("screener", "analysis", "quant_research"),
    "backtest": ("backtest_execution", "backtest_validation", "custom_signal"),
    "portfolio": ("portfolio_risk", "microstructure"),
    "meta": ("discovery", "provenance"),
}

RUNTIME_LABELS: Dict[str, str] = {
    "research": "Research",
    "backtest": "Backtest",
    "portfolio": "Portfolio & Execution",
    "meta": "Discovery & Provenance",
}

RUNTIME_DESCRIPTIONS: Dict[str, str] = {
    "research": (
        "Describe an asset or a universe: screen it, profile its risk and "
        "technicals, and analyze its statistical structure (factors, "
        "cointegration, PCA, Hurst, correlation). Does not run strategies."
    ),
    "backtest": (
        "Run, optimize, validate and diagnose a trading strategy — the "
        "library's built-in ones or a signal the caller computed "
        "themselves. Does not construct portfolios or size positions."
    ),
    "portfolio": (
        "Turn a view into a position and price what it costs: optimal "
        "weights, risk attribution, sizing, stress tests, capacity, and "
        "liquidity measured from bars or from ticks."
    ),
    "meta": (
        "Questions about the library and the session rather than about a "
        "market: what this library accepts and what the data provider can "
        "serve, and what a past tool call did and whether it still "
        "reproduces."
    ),
}

#: The modeling runtime lives in `modeling/agent` and is referenced by name
#: rather than rebuilt here. It predates this module and its separation is
#: the precedent this one generalizes.
MODELING_RUNTIME = "modeling"


@dataclass(frozen=True)
class Runtime:
    """One execution boundary: a name, the categories it owns, and the only
    tools it can run."""

    name: str
    label: str
    description: str
    categories: Tuple[str, ...]
    dispatch_table: Mapping[str, Tuple[Callable[..., Any], type]]
    tool_defs: Tuple[Tuple[str, str, type], ...]

    @property
    def tool_names(self) -> List[str]:
        return sorted(self.dispatch_table)

    def __len__(self) -> int:
        return len(self.dispatch_table)

    def __contains__(self, tool_name: object) -> bool:
        return tool_name in self.dispatch_table

    def get_tools(
        self, categories: Optional[Sequence[str]] = None
    ) -> List[Dict[str, Any]]:
        """This runtime's tool schemas, in the usual OpenAI-style envelope.

        `categories` narrows further WITHIN the runtime, for a caller that
        wants a still-smaller list. It cannot widen: a category this
        runtime does not own contributes nothing rather than reaching into
        another runtime.
        """
        from standard_quant_tools.agent.tools import TOOL_CATEGORY

        wanted = (
            set(categories) & set(self.categories)
            if categories
            else set(self.categories)
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": model.model_json_schema(),
                },
            }
            for name, description, model in self.tool_defs
            if TOOL_CATEGORY.get(name) in wanted
        ]

    def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Route a tool call, refusing anything this runtime does not own.

        Mirrors `agent.tools.dispatch` exactly -- same audit record, same
        JSON-safety boundary -- over a table that holds only this runtime's
        tools.
        """
        if tool_name not in self.dispatch_table:
            raise ValueError(self._out_of_scope_message(tool_name))
        fn, model_cls = self.dispatch_table[tool_name]
        return sanitize_for_json(_run_and_record(tool_name, fn, model_cls(**arguments)))

    __call__ = dispatch

    def _out_of_scope_message(self, tool_name: str) -> str:
        """Say where the tool actually lives, not merely that it is absent.

        A bare "unknown tool" cannot be told apart from a hallucinated
        name, so a model receiving one guesses again. Naming the owning
        runtime turns a dead end into a recoverable scoping error.
        """
        owner = owner_of(tool_name)
        if owner is None:
            return (
                f"Unknown tool {tool_name!r}. The {self.name!r} runtime "
                f"provides: {self.tool_names}. No runtime provides a tool by "
                "that name — it does not exist in this library."
            )
        return (
            f"{tool_name!r} exists but belongs to the {owner!r} runtime, not "
            f"to {self.name!r}. This caller is scoped to {self.name!r}, which "
            f"provides: {self.tool_names}. Either use one of those, or "
            f"construct the {owner!r} runtime deliberately — widening scope "
            "is a decision, not a fallback."
        )


def _build() -> Dict[str, Runtime]:
    """Assemble each Runtime from its own package's declaration.

    Imported here rather than at module scope because the packages import
    this module for nothing -- but the FACADE (agent/tools.py) imports the
    packages, and building at import time would order those two against
    each other.
    """
    import importlib

    runtimes: Dict[str, Runtime] = {}
    for name, categories in RUNTIME_CATEGORIES.items():
        package = importlib.import_module(f"standard_quant_tools.agent.runtimes.{name}")
        runtimes[name] = Runtime(
            name=name,
            label=RUNTIME_LABELS[name],
            description=RUNTIME_DESCRIPTIONS[name],
            categories=categories,
            dispatch_table=package.TOOL_DISPATCH,
            tool_defs=tuple(package.TOOL_DEFS),
        )
    return runtimes


_RUNTIMES: Optional[Dict[str, Runtime]] = None


def all_runtimes() -> Dict[str, Runtime]:
    """Every non-modeling runtime, built once.

    Built lazily rather than at import: this module is imported BY the
    package that defines the tools it indexes, and doing the work at import
    time would be a cycle.
    """
    global _RUNTIMES
    if _RUNTIMES is None:
        _RUNTIMES = _build()
    return _RUNTIMES


def resolve(name: str) -> Runtime:
    """One runtime by name."""
    runtimes = all_runtimes()
    if name not in runtimes:
        available = sorted(runtimes) + [MODELING_RUNTIME]
        raise ValueError(
            f"unknown runtime {name!r}; expected one of {available}. The "
            f"{MODELING_RUNTIME!r} runtime lives in "
            "standard_quant_tools.modeling.agent and is imported from there."
        )
    return runtimes[name]


def owner_of(tool_name: str) -> Optional[str]:
    """Which runtime owns a tool, including the modeling one, or None."""
    for name, runtime in all_runtimes().items():
        if tool_name in runtime:
            return name
    from standard_quant_tools.modeling.agent import MODELING_TOOL_DISPATCH

    if tool_name in MODELING_TOOL_DISPATCH:
        return MODELING_RUNTIME
    return None


def combine(names: Sequence[str], label: Optional[str] = None) -> Runtime:
    """One runtime spanning several, for a caller that genuinely needs them.

    The composition is explicit on purpose. A workflow that screens, then
    backtests, then sizes does cross three runtimes, and pretending
    otherwise would just push callers back to the unscoped dispatch. What
    this avoids is the *silent* version: the widening appears in the code
    that asked for it, so the scope an agent runs with is readable.
    """
    if not names:
        raise ValueError("combine() needs at least one runtime name")
    parts = [resolve(name) for name in names]
    table: Dict[str, Tuple[Callable[..., Any], type]] = {}
    categories: List[str] = []
    defs: List[Tuple[str, str, type]] = []
    for part in parts:
        table.update(part.dispatch_table)
        categories.extend(part.categories)
        defs.extend(part.tool_defs)
    joined = "+".join(part.name for part in parts)
    return Runtime(
        name=joined,
        label=label or " + ".join(part.label for part in parts),
        description=" ".join(part.description for part in parts),
        categories=tuple(categories),
        dispatch_table=table,
        tool_defs=tuple(defs),
    )


__all__ = [
    "MODELING_RUNTIME",
    "RUNTIME_CATEGORIES",
    "RUNTIME_DESCRIPTIONS",
    "RUNTIME_LABELS",
    "Runtime",
    "all_runtimes",
    "combine",
    "owner_of",
    "resolve",
]
