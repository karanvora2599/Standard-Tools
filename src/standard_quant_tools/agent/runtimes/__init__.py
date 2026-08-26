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
#: Modeling is absent here on purpose: it has no category taxonomy to own,
#: being one ordered pipeline rather than a set of interchangeable tools.
#: It is still a runtime, and `all_runtimes()` includes it -- it just has
#: nothing to contribute to a mapping FROM categories.
RUNTIME_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "research": ("screener", "analysis", "quant_research"),
    "backtest": ("backtest_execution", "backtest_validation", "custom_signal"),
    "portfolio": ("portfolio_risk", "microstructure"),
    "meta": ("discovery", "provenance"),
    "derivatives": ("derivatives",),
}

RUNTIME_LABELS: Dict[str, str] = {
    "research": "Research",
    "backtest": "Backtest",
    "portfolio": "Portfolio & Execution",
    "meta": "Discovery & Provenance",
    "derivatives": "Derivatives",
    "modeling": "Modeling",
    "feature_lab": "Feature Lab",
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
    "derivatives": (
        "Price an option and understand what holding it does to you: the "
        "second-order greeks, multi-leg payoffs, the consistency of a quoted "
        "surface, what the market is pricing as a move, and what a delta "
        "hedge costs to run. Takes quotes as arguments rather than fetching "
        "a chain -- this library has no options data provider, and a tool "
        "that pretended to would compute a chain that does not exist."
    ),
    "modeling": (
        "Build, validate and score a statistical model from this library's "
        "own features: dataset construction, leakage-purged walk-forward "
        "fitting, the model registry, and evaluation of out-of-sample "
        "predictions. One ordered pipeline rather than a set of "
        "interchangeable tools."
    ),
    "feature_lab": (
        "Interrogate the FEATURES of a built dataset, before and "
        "independently of fitting anything: what each one measures and "
        "predicts, which are restatements of one another, whether one has "
        "drifted or only worked in one regime, whether its IC is larger than "
        "what this panel's noise produces, and what each is worth to a "
        "fitted model. Exploratory and repeatable, where `modeling` is one "
        "ordered pipeline."
    ),
}

#: Tools that used to live somewhere else, and where they were.
#:
#: A split is a breaking change: an agent scoped to the donor loses the tool,
#: and a bare "belongs to feature_lab" reads as though it had hallucinated
#: the name. Saying where it USED to live turns a break into an instruction,
#: and it costs one dict.
#:
#: Retired one minor version after the move. A moved-from record nobody
#: cleans up becomes a changelog nobody reads, embedded in an error message
#: everybody does.
MOVED_FROM: Dict[str, str] = {
    # Left `research` when `derivatives` reached twelve tools and became its
    # own execution boundary. Same arguments, same behaviour, new scope.
    "get_option_pricing": "research",
    "get_implied_volatility": "research",
    **{
        name: "modeling"
        for name in (
            "analyze_feature",
            "compare_feature_sets",
            "get_feature_drift",
            "get_feature_ic_decay",
            "get_feature_redundancy",
            "get_feature_regime_stability",
            "run_feature_ablation",
            "run_feature_permutation_test",
            "select_features",
        )
    },
}

#: The modeling runtime lives in `modeling/agent`. It predates this module
#: and its separation is the precedent this one generalizes, so it is
#: wrapped BY REFERENCE below rather than rebuilt -- the Runtime holds
#: MODELING_TOOL_DISPATCH itself, and this module never becomes a second
#: place where the modeling surface is described.
MODELING_RUNTIME = "modeling"

#: `feature_lab` is wrapped the same way and for the same reason: its
#: dispatch table lives in modeling/agent/feature_tools.py and is held here
#: by reference, so this module never becomes a second description of it.
FEATURE_LAB_RUNTIME = "feature_lab"


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

        def _category_of(name: str) -> str:
            # The modeling runtime has no entry in TOOL_CATEGORY -- it is
            # one pipeline, not a taxonomy -- so its tools answer with the
            # runtime's own name, which is the category it declares.
            return TOOL_CATEGORY.get(name, self.name)

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
            if _category_of(name) in wanted
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
            f"to {self.name!r}.{_moved_note(tool_name, self.name)} This "
            f"caller is scoped to {self.name!r}, which "
            f"provides: {self.tool_names}. Either use one of those, or "
            f"construct the {owner!r} runtime deliberately — widening scope "
            "is a decision, not a fallback."
        )


def _moved_note(tool: str, scope: str) -> str:
    """
    " It used to be in 'modeling'." -- when that is why the caller is here.

    Added only when the caller is scoped to the runtime the tool LEFT.
    Anyone else never had it, so telling them it moved explains a history
    they were not part of and makes the message longer for nobody's benefit.
    """
    previous = MOVED_FROM.get(tool)
    if previous is None or previous != scope:
        return ""
    return (
        f" It used to be in {previous!r} and moved; the BOUNDARY was renamed, "
        "not the tool -- its arguments and behaviour are unchanged."
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

    # The modeling runtime, wrapped rather than rebuilt. Its dispatch table
    # is the SAME object modeling_dispatch routes through, so resolving it
    # here cannot drift from calling it there.
    from standard_quant_tools.modeling.agent import (
        MODELING_TOOL_DISPATCH,
        get_modeling_tools,
    )

    runtimes[MODELING_RUNTIME] = Runtime(
        name=MODELING_RUNTIME,
        label=RUNTIME_LABELS[MODELING_RUNTIME],
        description=RUNTIME_DESCRIPTIONS[MODELING_RUNTIME],
        categories=(MODELING_RUNTIME,),
        dispatch_table=MODELING_TOOL_DISPATCH,
        # Keyed by NAME, never zipped: get_modeling_tools() iterates its
        # own ordered def list while MODELING_TOOL_DISPATCH is a separate
        # dict, and the two orders do not match. Zipping them paired each
        # description with another tool's input model -- schemas that
        # looked plausible and described the wrong tool.
        tool_defs=tuple(
            (
                tool["function"]["name"],
                tool["function"]["description"],
                MODELING_TOOL_DISPATCH[tool["function"]["name"]][1],
            )
            for tool in get_modeling_tools()
        ),
    )
    # The feature_lab runtime, wrapped by reference exactly like modeling.
    # Both live under modeling/agent because that is where the analysis they
    # call lives; the RUNTIME boundary and the package layout answer
    # different questions and do not have to agree.
    from standard_quant_tools.modeling.agent.feature_tools import (
        FEATURE_TOOL_DISPATCH,
        get_feature_tools,
    )

    runtimes[FEATURE_LAB_RUNTIME] = Runtime(
        name=FEATURE_LAB_RUNTIME,
        label=RUNTIME_LABELS[FEATURE_LAB_RUNTIME],
        description=RUNTIME_DESCRIPTIONS[FEATURE_LAB_RUNTIME],
        categories=(FEATURE_LAB_RUNTIME,),
        dispatch_table=FEATURE_TOOL_DISPATCH,
        # Keyed by NAME rather than zipped, for the reason recorded above.
        tool_defs=tuple(
            (
                tool["function"]["name"],
                tool["function"]["description"],
                FEATURE_TOOL_DISPATCH[tool["function"]["name"]][1],
            )
            for tool in get_feature_tools()
        ),
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
        raise ValueError(
            f"unknown runtime {name!r}; expected one of {sorted(runtimes)}."
        )
    return runtimes[name]


def owner_of(tool_name: str) -> Optional[str]:
    """Which runtime owns a tool, including the modeling one, or None."""
    for name, runtime in all_runtimes().items():
        if tool_name in runtime:
            return name
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
    "FEATURE_LAB_RUNTIME",
    "MODELING_RUNTIME",
    "MOVED_FROM",
    "RUNTIME_CATEGORIES",
    "RUNTIME_DESCRIPTIONS",
    "RUNTIME_LABELS",
    "Runtime",
    "all_runtimes",
    "combine",
    "owner_of",
    "resolve",
]
