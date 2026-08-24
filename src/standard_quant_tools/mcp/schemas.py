"""
JSON Schema shaping for the MCP surface.

WHY THIS EXISTS. Seven of the fifty-four tools carry `$ref`/`$defs` in their
input schemas -- the ones whose inputs are nested spec models
(`run_portfolio_optimization`, `run_portfolio_simulation`,
`build_model_dataset`, `run_model_experiment`, and three more). Those are
also the seven most complex tools in the library, so they are exactly the
ones where a client that resolves `$ref` poorly costs the most.

The references are resolvable in principle -- pydantic emits `$defs`
alongside them. In practice "resolvable in principle" has to be the same
thing as "resolved", so this module inlines them and `tests/mcp/` asserts
that nothing reaching a client still contains a `$ref`.

Inlining duplicates any definition used more than once, so it is not free.
`schema_bytes()` exists so the cost is measured rather than assumed, and the
budget test reports it.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Set

#: Where pydantic v2 puts its definitions, and the prefix its refs use.
_DEFS_KEY = "$defs"
_REF_KEY = "$ref"
_REF_PREFIX = "#/$defs/"


class CircularSchemaError(ValueError):
    """A schema refers to itself, so it cannot be fully inlined.

    Raised rather than silently leaving a `$ref` behind: a partially
    dereferenced schema would pass a naive "does it contain $ref" check in
    some places and fail it in others, which is worse than a clear error at
    build time. No shipped tool triggers this today -- the spec models are
    trees -- and the test suite pins that.
    """


def dereference(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return `schema` with every `$ref` replaced by its definition inline.

    The `$defs` block is removed once nothing points at it. Definitions used
    more than once are duplicated, which is the price of not requiring the
    client to resolve references.
    """
    defs = schema.get(_DEFS_KEY) or {}
    if not defs:
        return copy.deepcopy(schema)

    def resolve(node: Any, stack: Set[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, stack) for item in node]
        if not isinstance(node, dict):
            return node

        ref = node.get(_REF_KEY)
        if isinstance(ref, str):
            if not ref.startswith(_REF_PREFIX):
                # An external or non-$defs pointer. Nothing in this library
                # produces one; leaving it untouched would smuggle a $ref
                # past the test, so refuse instead.
                raise CircularSchemaError(
                    f"unsupported reference {ref!r}: only {_REF_PREFIX}* is "
                    "resolvable here"
                )
            name = ref[len(_REF_PREFIX) :]
            if name in stack:
                raise CircularSchemaError(
                    f"definition {name!r} refers to itself, so the schema "
                    "cannot be inlined"
                )
            if name not in defs:
                raise CircularSchemaError(f"dangling reference {ref!r}")
            resolved = resolve(defs[name], stack | {name})
            # Sibling keys alongside a $ref (title, description, default)
            # are overrides, and JSON Schema says they win over the target.
            siblings = {k: v for k, v in node.items() if k != _REF_KEY}
            if siblings:
                merged = dict(resolved)
                merged.update(resolve(siblings, stack))
                return merged
            return copy.deepcopy(resolved)

        return {k: resolve(v, stack) for k, v in node.items() if k != _DEFS_KEY}

    out = resolve(schema, set())
    assert isinstance(out, dict)
    out.pop(_DEFS_KEY, None)
    return out


def contains_ref(schema: Any) -> bool:
    """True if any `$ref` survives anywhere in the structure."""
    if isinstance(schema, dict):
        return _REF_KEY in schema or any(contains_ref(v) for v in schema.values())
    if isinstance(schema, list):
        return any(contains_ref(v) for v in schema)
    return False


def property_names(schema: Any) -> List[str]:
    """
    Every property name anywhere in a schema, nested objects included.

    Used to derive the `openWorldHint` annotation: a tool that names a
    ticker, symbol or universe anywhere in its input is one that will go
    and fetch market data. A shallow scan of the top level misses it --
    `build_model_dataset` hides its universe two levels down inside a
    DatasetSpec -- which is why this recurses.
    """
    found: List[str] = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "properties" and isinstance(value, dict):
                found.extend(value.keys())
            found.extend(property_names(value))
    elif isinstance(schema, list):
        for item in schema:
            found.extend(property_names(item))
    return found


def schema_bytes(schema: Dict[str, Any]) -> int:
    """Serialized size, for the context-budget accounting in the tests."""
    return len(json.dumps(schema, separators=(",", ":")))
