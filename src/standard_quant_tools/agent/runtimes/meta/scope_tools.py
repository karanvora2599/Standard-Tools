"""
Questions about the tool surface itself.

The library has 198 tools across ten runtimes, and no single session
should hold all of them. Choosing a scope is therefore a real decision with
a real cost, and until now it was one a caller had to make blind: the byte
cost of a runtime existed as a number inside `mcp/catalog.py` and was
reachable only by reading the source.

WHY THIS IS A TOOL AND NOT DOCUMENTATION. Documentation goes stale the day a
tool is added. These read the live registry, so the answer is always the
current one -- and the cost figures in particular move every time a schema
changes, which is often.

`compare_artifacts` is the odd one out and belongs here for the same reason
`describe_reference` does: it is a question about what the session PRODUCED
rather than about a market. Two stored results, one question -- did anything
change, and what.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(default_factory=list)


class ToolCostInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtimes: Optional[List[str]] = Field(
        None,
        description="Runtimes to price. Omit for every runtime, which is "
        "what makes the comparison useful.",
    )
    include_output_schemas: bool = Field(
        False,
        description="Output schemas are omitted by the server by default and "
        "add roughly 77%. Counting them when the server does not would "
        "report a cost nobody pays.",
    )


class DescribeRuntimeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: Optional[str] = Field(None, description="Omit to describe every runtime.")
    include_tool_names: bool = Field(True)


class CompareArtifactsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    a: Dict[str, Any] = Field(..., description="The first result object.")
    b: Dict[str, Any] = Field(..., description="The second.")
    tolerance: float = Field(
        1e-9,
        ge=0,
        description="Relative tolerance below which two numbers count as "
        "unchanged. The default is machine-precision; raise it to ignore "
        "differences you consider noise.",
    )
    label_a: str = Field(
        "a",
        description="What to call the first side when reporting. 'baseline' "
        "and 'candidate' read better than 'a' and 'b' in a report a human "
        "will see, and the difference entries key on 'a'/'b' regardless, so "
        "this is naming rather than structure.",
    )
    label_b: str = Field("b", description="What to call the second side.")


class RuntimeCost(BaseModel):
    model_config = ConfigDict(extra="allow")

    runtime: str = ""
    n_tools: int = 0
    bytes: int = 0
    approx_tokens: int = 0
    share_of_total: Stat = None


class ToolCostResult(_Result):
    n_runtimes: int = 0
    total_bytes: int = 0
    total_approx_tokens: int = 0
    by_runtime: List[RuntimeCost] = Field(default_factory=list)
    cheapest: str = ""
    most_expensive: str = ""


class RuntimeDescription(BaseModel):
    model_config = ConfigDict(extra="allow")

    runtime: str = ""
    label: str = ""
    description: str = ""
    categories: List[str] = Field(default_factory=list)
    n_tools: int = 0
    tool_names: List[str] = Field(default_factory=list)


class DescribeRuntimeResult(_Result):
    n_runtimes: int = 0
    runtimes: List[RuntimeDescription] = Field(default_factory=list)


class FieldDifference(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str = ""
    kind: str = Field(
        "", description="'changed', 'only_in_a', 'only_in_b' or 'type_changed'."
    )
    a: Optional[Any] = None
    b: Optional[Any] = None
    relative_change: Stat = None


class CompareArtifactsResult(_Result):
    # Echoed back because the difference entries key on 'a' and 'b'
    # whatever the caller called them -- without this the labels would name
    # the sides in the prose and nothing would say which key is which.
    label_a: str = "a"
    label_b: str = "b"
    identical: bool = False
    n_differences: int = 0
    n_fields_compared: int = 0
    differences: List[FieldDifference] = Field(default_factory=list)
    largest_relative_change: Stat = None


def estimate_tool_cost(input_data: ToolCostInput) -> ToolCostResult:
    from standard_quant_tools.mcp.catalog import (
        ALL_RUNTIMES,
        build_catalog,
        select_runtimes,
    )

    catalog = build_catalog()
    wanted = list(input_data.runtimes) if input_data.runtimes else list(ALL_RUNTIMES)
    unknown = [r for r in wanted if r not in ALL_RUNTIMES]
    if unknown:
        raise ValidationError(
            f"estimate_tool_cost: unknown runtime(s) {unknown}. "
            f"Available: {list(ALL_RUNTIMES)}."
        )

    rows: List[Dict[str, Any]] = []
    for runtime in wanted:
        entries = select_runtimes(catalog, [runtime])
        size = sum(
            e.cost_bytes(include_output_schema=input_data.include_output_schemas)
            for e in entries
        )
        rows.append(
            {
                "runtime": runtime,
                "n_tools": len(entries),
                "bytes": int(size),
                "approx_tokens": int(size // 4),
            }
        )
    total = sum(r["bytes"] for r in rows)
    for row in rows:
        row["share_of_total"] = float(row["bytes"] / total) if total else None
    rows.sort(key=lambda r: r["bytes"], reverse=True)

    warnings: List[str] = [
        "Bytes are the SERIALIZED SCHEMA cost -- what a client pays to be "
        "told the tool exists, before any call. Tokens are bytes/4, which is "
        "a rule of thumb and not a tokenizer.",
        "This is the cost at FULL detail. The server defaults to "
        "--tool-detail auto, which thins any runtime over the budget and "
        "injects describe_tool so the full schema stays one call away -- so "
        "the heaviest runtimes cost materially less in practice.",
    ]
    if not input_data.include_output_schemas:
        warnings.append(
            "Output schemas are EXCLUDED, matching the server's default. "
            "Declaring them adds roughly 77%, which is why it is a flag."
        )
    return ToolCostResult(
        n_runtimes=len(rows),
        total_bytes=int(total),
        total_approx_tokens=int(total // 4),
        by_runtime=[RuntimeCost(**r) for r in rows],
        cheapest=rows[-1]["runtime"] if rows else "",
        most_expensive=rows[0]["runtime"] if rows else "",
        warnings=warnings,
    )


def describe_runtime(input_data: DescribeRuntimeInput) -> DescribeRuntimeResult:
    from standard_quant_tools.agent.runtimes import all_runtimes

    registry = all_runtimes()
    if input_data.runtime:
        if input_data.runtime not in registry:
            raise ValidationError(
                f"describe_runtime: unknown runtime "
                f"{input_data.runtime!r}. Available: {sorted(registry)}."
            )
        wanted = [input_data.runtime]
    else:
        wanted = list(registry)

    rows = [
        RuntimeDescription(
            runtime=name,
            label=registry[name].label,
            description=registry[name].description,
            categories=list(registry[name].categories),
            n_tools=len(registry[name]),
            tool_names=(
                registry[name].tool_names if input_data.include_tool_names else []
            ),
        )
        for name in wanted
    ]
    return DescribeRuntimeResult(
        n_runtimes=len(rows),
        runtimes=rows,
        warnings=[
            "A runtime is an EXECUTION boundary, not a hint. A tool from "
            "another runtime is unroutable rather than discouraged, and the "
            "refusal names the runtime that owns it.",
            "Results cross runtimes freely -- by artifact URI, by identifier, "
            "or as plain JSON. It is EXECUTION that is scoped, not data, "
            "which is what keeps multi-runtime workflows possible.",
        ],
    )


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Nested structure into path -> leaf value."""
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            out.update(_flatten(value, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def compare_artifacts(input_data: CompareArtifactsInput) -> CompareArtifactsResult:
    flat_a = _flatten(input_data.a)
    flat_b = _flatten(input_data.b)
    keys = sorted(set(flat_a) | set(flat_b))

    differences: List[Dict[str, Any]] = []
    largest: Optional[float] = None
    for key in keys:
        in_a, in_b = key in flat_a, key in flat_b
        if not in_b:
            differences.append({"path": key, "kind": "only_in_a", "a": flat_a[key]})
            continue
        if not in_a:
            differences.append({"path": key, "kind": "only_in_b", "b": flat_b[key]})
            continue
        left, right = flat_a[key], flat_b[key]
        if isinstance(left, bool) != isinstance(right, bool):
            differences.append(
                {"path": key, "kind": "type_changed", "a": left, "b": right}
            )
            continue
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not (math.isfinite(float(left)) and math.isfinite(float(right))):
                if repr(left) != repr(right):
                    differences.append(
                        {"path": key, "kind": "changed", "a": left, "b": right}
                    )
                continue
            scale = max(abs(float(left)), abs(float(right)), 1e-300)
            relative = abs(float(left) - float(right)) / scale
            if relative > input_data.tolerance:
                differences.append(
                    {
                        "path": key,
                        "kind": "changed",
                        "a": left,
                        "b": right,
                        "relative_change": float(relative),
                    }
                )
                largest = relative if largest is None else max(largest, relative)
            continue
        if left != right:
            differences.append({"path": key, "kind": "changed", "a": left, "b": right})
    differences.sort(key=lambda d: (d.get("relative_change") or 0.0), reverse=True)

    label_a, label_b = input_data.label_a, input_data.label_b
    warnings: List[str] = []
    if not differences:
        warnings.append(
            f"{label_a} and {label_b} are identical to the given tolerance. "
            "That is evidence the computation reproduces, not proof -- an "
            "identical result from an identical cached input says nothing "
            "about the computation at all."
        )
    else:
        changed = [d for d in differences if d["kind"] == "changed"]
        structural = [d for d in differences if d["kind"] != "changed"]
        if structural:
            warnings.append(
                f"{len(structural)} STRUCTURAL difference(s) -- fields "
                f"present in {label_a} and not {label_b} (or the reverse), "
                "or with a changed type. Those usually mean a version change "
                "rather than a numerical one, and they are worth resolving "
                "before reading the value differences. In `differences`, "
                f"`a` is {label_a} and `b` is {label_b}."
            )
        if changed:
            warnings.append(
                f"{len(changed)} value(s) differ by more than "
                f"{input_data.tolerance:g}. The list is ordered by relative "
                "change, so the largest movers are first."
            )
    return CompareArtifactsResult(
        label_a=label_a,
        label_b=label_b,
        identical=not differences,
        n_differences=len(differences),
        n_fields_compared=len(keys),
        differences=[FieldDifference(**d) for d in differences[:100]],
        largest_relative_change=largest,
        warnings=warnings,
    )


SCOPE_TOOL_DEFS = [
    (
        "estimate_tool_cost",
        "What each runtime costs a client's context, in bytes and approximate "
        "tokens, before any call is made. Choosing a scope is a real decision "
        "with a real cost and this makes it visible -- the numbers come from "
        "the live registry rather than from documentation, so they are "
        "current whenever a schema changes. Output schemas are excluded by "
        "default because the server omits them by default; counting them "
        "would report a cost nobody pays.",
        ToolCostInput,
    ),
    (
        "describe_runtime",
        "What each runtime is for, which categories it owns, and which tools "
        "it holds. A runtime is an EXECUTION boundary rather than a hint: a "
        "tool from another runtime is unroutable, not merely discouraged. Use "
        "this before scoping a session, and after a refusal that named a "
        "runtime you did not expect.",
        DescribeRuntimeInput,
    ),
    (
        "compare_artifacts",
        "A field-by-field diff of two result objects, ordered by the size of "
        "the change. Answers the question that follows every re-run: did "
        "anything move, and what. Separates STRUCTURAL differences -- a field "
        "present in one and not the other -- from numerical ones, because the "
        "first usually means a version change and should be resolved before "
        "the second is read. An identical result is evidence of "
        "reproducibility and not proof of it: identical output from an "
        "identical cached input says nothing about the computation.",
        CompareArtifactsInput,
    ),
]

SCOPE_TOOL_DISPATCH = {
    "estimate_tool_cost": (estimate_tool_cost, ToolCostInput),
    "describe_runtime": (describe_runtime, DescribeRuntimeInput),
    "compare_artifacts": (compare_artifacts, CompareArtifactsInput),
}

__all__ = [
    "SCOPE_TOOL_DEFS",
    "SCOPE_TOOL_DISPATCH",
    "compare_artifacts",
    "describe_runtime",
    "estimate_tool_cost",
]
