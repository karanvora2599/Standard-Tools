"""
The tool catalog: which tools this server exposes, and what each one costs.

WHY EXPOSURE IS A POLICY AND NOT A LIST. The two registries hold 57 tools
whose input schemas and descriptions total about 103 KB. An MCP client
fetches the tool list once at connect and carries it for the whole session,
so exposing everything spends roughly 26,000 tokens of every conversation
before the user has asked anything.

That is the constraint this module exists to manage. Tools are selected by
CATEGORY, using the same `TOOL_CATEGORY` taxonomy that already drives
`agent/router.py` and the nine workers in `Multi_Agent_Implementation/` --
reused rather than re-invented, so a tool's categorization stays correct in
exactly one place.

Tool count and cost turn out to be almost unrelated, which is why the
selection is by measured size rather than by intuition: `analysis` carries
13 tools in 11.7 KB while `custom_signal` carries 2 tools in 6.0 KB, and
`backtest_execution` alone is a quarter of the whole surface. At the other
end, `discovery` is 3 tools in 1.2 KB -- cheaper than any single tool in
`backtest_execution`, which is why it is on by default despite being the
newest category. Run `sqt-mcp --print-budget` for the current table;
`category_costs()` computes it and the budget test pins the ceiling.

THE RUNTIMES STAY APART. Each entry records which RUNTIME it came from, and
`dispatch_for()` returns that runtime's dispatch function. The names happen
not to collide (70 tools, 70 unique names), so one flat lookup would work --
and would be exactly the merge the library declined to make.

There were two registries when this module was written and there are five
runtimes now (research, backtest, portfolio, meta, modeling). Nothing here
changed in kind: a runtime is a dispatch table that refuses what it does not
own, which is what the modeling registry always was. The server pairs each
tool with its owning runtime's dispatcher for the same reason it always
did -- the dispatchers have identical signatures, so nothing structural
stops a caller pairing a tool from one with the dispatcher of another, and
that failure reads like the model chose badly rather than like the wiring
is wrong.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from standard_quant_tools.agent.router import TOOL_CATEGORY
from standard_quant_tools.agent.runtimes import RUNTIME_CATEGORIES
from standard_quant_tools.agent.runtimes import resolve as resolve_runtime
from standard_quant_tools.agent.tools import get_agent_tools
from standard_quant_tools.mcp.schemas import (
    dereference,
    property_names,
    schema_bytes,
)
from standard_quant_tools.modeling.agent import (
    MODELING_TOOL_DISPATCH,
    get_modeling_tools,
    modeling_dispatch,
)

#: Retained under their original names because `ToolEntry.registry` is a
#: public field and the MCP tests, the server and downstream callers all
#: read it. ANALYSIS_REGISTRY is no longer a single dispatch table -- it is
#: the four non-modeling runtimes -- so it survives only as the answer to
#: "is this tool from the modeling side or the rest of the library".
ANALYSIS_REGISTRY = "analysis"
MODELING_REGISTRY = "modeling"

#: category -> owning runtime, inverted from the runtimes' own declaration
#: so this module never holds a second copy of the grouping.
_CATEGORY_RUNTIME = {
    category: runtime
    for runtime, categories in RUNTIME_CATEGORIES.items()
    for category in categories
}

#: The modeling runtime has no category taxonomy -- it is one ordered
#: pipeline -- so it is a single category whose name matches its registry.
MODELING_CATEGORY = "modeling"

#: Every selectable category, analysis ones first in cost order at runtime.
ALL_CATEGORIES: Tuple[str, ...] = tuple(
    sorted(set(TOOL_CATEGORY.values())) + [MODELING_CATEGORY]
)

#: Every selectable runtime, in the order they are declared. `modeling`
#: is appended rather than read from RUNTIME_CATEGORIES because it has no
#: category taxonomy of its own -- see MODELING_CATEGORY.
ALL_RUNTIMES: Tuple[str, ...] = tuple(RUNTIME_CATEGORIES) + (MODELING_REGISTRY,)

#: runtime -> the categories it owns. The inverse of `_CATEGORY_RUNTIME`,
#: and the mapping `--runtime` is resolved through.
RUNTIME_CATEGORY_MAP: Dict[str, Tuple[str, ...]] = {
    **{runtime: tuple(cats) for runtime, cats in RUNTIME_CATEGORIES.items()},
    MODELING_REGISTRY: (MODELING_CATEGORY,),
}

#: The default. Measured at 25 tools / 21.2 KB / ~5k tokens: screening, risk
#: and technical snapshots, the factor/cointegration/Hurst research path,
#: and discovery. It deliberately omits `backtest_execution` (23.7 KB) and
#: `modeling` (26.0 KB) -- the two heaviest categories, both better switched
#: on for a session that needs them than paid for by every session that does
#: not.
#:
#: `discovery` earns its place by being the only category that makes the
#: OTHERS cheaper to use: it is 1.2 KB, and the questions it answers --
#: which parameters a strategy takes, which stress windows exist, whether
#: this provider has ticks -- were otherwise answered by a failed call and
#: an error round trip, which costs more than the category does.
DEFAULT_CATEGORIES: Tuple[str, ...] = (
    "screener",
    "analysis",
    "quant_research",
    "discovery",
)

#: Property names that mean "this tool will go and fetch market data".
#: Matched as substrings against every property anywhere in the input
#: schema, because `build_model_dataset` hides its universe two levels down
#: inside a DatasetSpec and a top-level scan would call it offline.
_NETWORK_MARKERS = ("symbol", "ticker", "universe")


@dataclass(frozen=True)
class ToolEntry:
    """One exposed tool, with everything the MCP layer needs to serve it."""

    name: str
    category: str
    registry: str
    runtime: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]]
    reads_market_data: bool
    persists_artifact: bool

    @property
    def idempotent(self) -> bool:
        """
        False when a call writes a new artifact.

        MCP's `idempotentHint` asks whether repeating a call with the same
        arguments has any additional effect. Four tools persist a Parquet
        artifact per call and so genuinely do; the other fifty do not.
        """
        return not self.persists_artifact

    def cost_bytes(self, include_output_schema: bool = False) -> int:
        """
        Serialized schema size -- what this tool costs a client's context.

        Output schemas are EXCLUDED by default because the server omits them
        by default: declaring all 54 adds about 74 KB, a 77% increase, and
        `structuredContent` is returned either way. Counting them here
        regardless would report a cost no client actually pays and make the
        category budget useless for choosing a `--categories` value.
        """
        total = schema_bytes(self.input_schema) + len(self.description)
        if include_output_schema and self.output_schema is not None:
            total += schema_bytes(self.output_schema)
        return total


def _output_schema(fn: Callable[..., Any]) -> Optional[Dict[str, Any]]:
    """
    The tool's result model as a flat JSON Schema, or None if it has no
    usable return annotation.

    Every one of the 54 tools has a typed Pydantic return today (verified by
    the test suite), so this returns a schema for all of them -- which is
    what lets the server declare `outputSchema` and send
    `structuredContent` rather than untyped text.
    """
    try:
        annotation = typing.get_type_hints(fn).get("return")
    except Exception:  # pragma: no cover - exotic annotations
        return None
    dump = getattr(annotation, "model_json_schema", None)
    if dump is None:
        return None
    return dereference(dump())


def _reads_market_data(input_schema: Dict[str, Any]) -> bool:
    names = [p.lower() for p in property_names(input_schema)]
    return any(marker in name for name in names for marker in _NETWORK_MARKERS)


def _entries_for_registry(
    schemas: Sequence[Dict[str, Any]],
    dispatch_table: Dict[str, Any],
    registry: str,
    category_of: Callable[[str], str],
) -> Iterable[ToolEntry]:
    for tool in schemas:
        fn_schema = tool["function"]
        name = fn_schema["name"]
        entry = dispatch_table[name]
        fn = entry[0] if isinstance(entry, tuple) else entry
        input_schema = dereference(fn_schema["parameters"])
        output_schema = _output_schema(fn)
        result_fields = set()
        annotation = typing.get_type_hints(fn).get("return")
        if annotation is not None:
            result_fields = set(getattr(annotation, "model_fields", {}) or {})
        category = category_of(name)
        yield ToolEntry(
            name=name,
            category=category,
            registry=registry,
            runtime=_CATEGORY_RUNTIME.get(category, MODELING_REGISTRY),
            description=fn_schema["description"],
            input_schema=input_schema,
            output_schema=output_schema,
            reads_market_data=_reads_market_data(input_schema),
            persists_artifact=any(f.endswith("_uri") for f in result_fields),
        )


def build_catalog() -> Dict[str, ToolEntry]:
    """Every tool in both registries, keyed by name."""
    from standard_quant_tools.agent import tools as analysis_tools

    catalog: Dict[str, ToolEntry] = {}
    for entry in _entries_for_registry(
        get_agent_tools(),
        analysis_tools._TOOL_DISPATCH,
        ANALYSIS_REGISTRY,
        lambda name: TOOL_CATEGORY[name],
    ):
        catalog[entry.name] = entry
    for entry in _entries_for_registry(
        get_modeling_tools(),
        MODELING_TOOL_DISPATCH,
        MODELING_REGISTRY,
        lambda _name: MODELING_CATEGORY,
    ):
        if entry.name in catalog:  # pragma: no cover - pinned by a test
            raise RuntimeError(
                f"tool name {entry.name!r} exists in both registries; the MCP "
                "server routes by name and cannot disambiguate it"
            )
        catalog[entry.name] = entry
    return catalog


def select(catalog: Dict[str, ToolEntry], categories: Sequence[str]) -> List[ToolEntry]:
    """The catalog narrowed to the requested categories, name-sorted."""
    unknown = [c for c in categories if c not in ALL_CATEGORIES]
    if unknown:
        raise ValueError(
            f"unknown categor{'y' if len(unknown) == 1 else 'ies'} {unknown}; "
            f"expected some of {list(ALL_CATEGORIES)}"
        )
    wanted = set(categories)
    return sorted(
        (e for e in catalog.values() if e.category in wanted), key=lambda e: e.name
    )


def _split_runtimes(raw: str) -> List[str]:
    """
    Parse a --runtime value. Accepts `research`, `research+meta`,
    `research,meta` and `all`.

    `+` is accepted because that is how `combine()` names a joined runtime,
    and a flag that spells the same thing differently from the library
    would be one more thing to remember wrongly.
    """
    if raw.strip().lower() == "all":
        return list(ALL_RUNTIMES)
    parts = [p.strip() for p in raw.replace("+", ",").split(",")]
    return [p for p in parts if p]


def categories_for_runtimes(runtimes: Sequence[str]) -> Tuple[str, ...]:
    """
    Every category owned by the named runtimes, in ALL_CATEGORIES order.

    This is what makes `--runtime research` mean *all of research* rather
    than research narrowed by whatever `--categories` happens to default
    to. Ordering follows ALL_CATEGORIES so the reported selection is
    stable regardless of the order the runtimes were named in.
    """
    unknown = [r for r in runtimes if r not in RUNTIME_CATEGORY_MAP]
    if unknown:
        raise ValueError(
            f"unknown runtime{'s' if len(unknown) > 1 else ''} {unknown}; "
            f"expected some of {list(ALL_RUNTIMES)}"
        )
    wanted = {c for r in runtimes for c in RUNTIME_CATEGORY_MAP[r]}
    return tuple(c for c in ALL_CATEGORIES if c in wanted)


def select_runtimes(
    catalog: Dict[str, ToolEntry], runtimes: Sequence[str]
) -> List[ToolEntry]:
    """The catalog narrowed to the named runtimes, name-sorted."""
    wanted = set(runtimes)
    unknown = [r for r in wanted if r not in RUNTIME_CATEGORY_MAP]
    if unknown:
        raise ValueError(
            f"unknown runtime{'s' if len(unknown) > 1 else ''} {unknown}; "
            f"expected some of {list(ALL_RUNTIMES)}"
        )
    return sorted(
        (e for e in catalog.values() if e.runtime in wanted), key=lambda e: e.name
    )


def runtime_costs(
    catalog: Optional[Dict[str, ToolEntry]] = None,
    include_output_schemas: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """runtime -> (tool count, schema bytes), for the budget report.

    The number a client actually pays once exposure is runtime-scoped,
    where `category_costs` reports the number it pays per filter WITHIN
    that scope.
    """
    catalog = catalog or build_catalog()
    costs: Dict[str, Tuple[int, int]] = {}
    for entry in catalog.values():
        count, size = costs.get(entry.runtime, (0, 0))
        costs[entry.runtime] = (
            count + 1,
            size + entry.cost_bytes(include_output_schemas),
        )
    return costs


def category_costs(
    catalog: Optional[Dict[str, ToolEntry]] = None,
    include_output_schemas: bool = False,
) -> Dict[str, Tuple[int, int]]:
    """category -> (tool count, schema bytes), for the budget report."""
    catalog = catalog or build_catalog()
    costs: Dict[str, Tuple[int, int]] = {}
    for entry in catalog.values():
        count, size = costs.get(entry.category, (0, 0))
        costs[entry.category] = (
            count + 1,
            size + entry.cost_bytes(include_output_schemas),
        )
    return costs


def dispatch_for(entry: ToolEntry) -> Callable[[str, Dict[str, Any]], Dict[str, Any]]:
    """
    The dispatch function belonging to this tool's RUNTIME.

    Paired with the tool rather than chosen separately, for the same reason
    `_agent_utils.py` pairs them: every dispatcher has the same signature,
    so nothing structural stops a caller pairing a tool from one runtime
    with the dispatcher of another. That fails at the call with an error
    naming the model's choice, which reads like the model picked badly
    rather than like the wiring is wrong.

    Returning the RUNTIME's dispatcher rather than one union dispatcher is
    what makes the server's scoping real all the way down: a tool served
    from the research runtime is executed by a table that holds only
    research tools.
    """
    if entry.registry == MODELING_REGISTRY:
        return modeling_dispatch
    return resolve_runtime(entry.runtime).dispatch
