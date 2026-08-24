"""
The resource surface: what this server can hand back by URI rather than by
stuffing into a tool result.

WHY IT MATTERS MORE HERE THAN IN A FUNCTION-CALLING LOOP. 83 list- or
dict-valued fields live across 28 result models. `BacktestResult` carries an
equity curve and a trade log; `PortfolioResult` carries a whole correlation
matrix. In the single-agent scripts that is survivable -- the model sees the
JSON once and `_agent_utils.py` truncates the console echo. Over MCP the
result enters the client's context and STAYS there for the rest of the
session, so a five-year backtest taxes every subsequent turn.

So a result over `--inline-limit` bytes is stored whole and returned as a
compact summary plus a `sqt://result/...` link. The summary names every
field it left out and how large it was: this must never be a silent
truncation, because a model that cannot tell a small result from a
summarized one will report the summary as if it were the whole thing.

EVERYTHING IS SANDBOXED. `modeling.artifacts.run_dir()` resolves and
confirms a path inside SQT_RUNS_DIR before any read or write. A URI arriving
from a client is untrusted input and is the one place in this server where a
traversal bug would be reachable from outside, so every path here goes
through that function rather than around it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from standard_quant_tools import cli as _cli
from standard_quant_tools._jsonsafe import sanitize_for_json
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts

SCHEME = "sqt"

#: Static resources -- always listed, no arguments.
CATALOG_FEATURES = "sqt://catalog/features"
CATALOG_CAPABILITIES = "sqt://catalog/capabilities"
CATALOG_CATEGORIES = "sqt://catalog/categories"

#: Templated resources -- listed as templates, read by substituting an id.
TEMPLATES: Tuple[Tuple[str, str, str], ...] = (
    (
        "sqt://result/{result_id}",
        "Full tool result",
        "A tool result too large to inline, stored whole. The id comes from "
        "the `result_uri` of a summarized tool response.",
    ),
    (
        "sqt://artifact/{run_id}/{name}",
        "Run artifact",
        "A Parquet artifact written by a tool -- equity curves, trade logs, "
        "out-of-sample predictions, target weights -- as JSON records.",
    ),
    (
        "sqt://model/{model_id}",
        "Registered model manifest",
        "A registered model's manifest: spec, data and spec hashes, "
        "out-of-sample metrics, feature importance, and lineage.",
    ),
    (
        "sqt://dataset/{dataset_id}",
        "Modeling dataset metadata",
        "A built dataset's metadata: universe, features, target, row counts, "
        "warnings, and the drop attribution from alignment.",
    ),
    (
        "sqt://audit/{request_id}",
        "Audit decision record",
        "The hash-chained decision record for one tool call: inputs, the "
        "market data it pulled with content hashes, which execution path "
        "ran, and a hash of its output. Replayable with `sqt replay`.",
    ),
)

_RESULT_PREFIX = "mcpres-"
#: save_json validates this as a bare identifier and appends ".json"
#: itself, so it is stored without an extension and read back with one.
_RESULT_NAME = "result"


# ── storing an oversized result ──────────────────────────────────────


def _summarize(payload: Dict[str, Any], limit: int) -> Tuple[Dict[str, Any], List[str]]:
    """
    Keep every scalar, drop the bulk, and say exactly what was dropped.

    The rule is all-or-nothing per field rather than truncating inside one:
    half a trade log looks like a whole trade log to a model reading it, and
    there is no honest way to signal "this list continues" inside the value
    itself.
    """
    kept: Dict[str, Any] = {}
    omitted: List[str] = []
    for key, value in payload.items():
        encoded = len(json.dumps(value, default=str))
        if isinstance(value, (list, dict)) and encoded > limit // 4:
            size = len(value)
            omitted.append(
                f"{key} ({'rows' if isinstance(value, list) else 'keys'}={size}, "
                f"{encoded:,} bytes)"
            )
        else:
            kept[key] = value
    return kept, omitted


def store_result(
    tool_name: str, payload: Dict[str, Any], limit: int
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Return `(payload_to_send, result_uri)`.

    Under the limit, the payload is returned untouched and `result_uri` is
    None -- the common case, and it costs nothing. Over it, the whole result
    is persisted and a summary is returned alongside the URI that serves the
    original.
    """
    encoded = json.dumps(payload, default=str)
    if limit <= 0 or len(encoded) <= limit:
        return payload, None

    digest = hashlib.sha256(f"{tool_name}:{encoded}".encode("utf-8")).hexdigest()[:32]
    result_id = f"{_RESULT_PREFIX}{digest}"
    directory = _artifacts.run_dir(result_id)
    _artifacts.save_json(
        directory, _RESULT_NAME, {"tool": tool_name, "result": payload}
    )

    kept, omitted = _summarize(payload, limit)
    uri = f"sqt://result/{result_id}"
    kept["_truncated"] = {
        "reason": (
            f"the full result is {len(encoded):,} bytes, over the "
            f"{limit:,}-byte inline limit"
        ),
        "omitted_fields": omitted,
        "result_uri": uri,
        "note": (
            "The fields listed in omitted_fields are NOT in this response. "
            "Read the result_uri resource to get them. Do not describe them "
            "as absent or empty -- they exist and were withheld for size."
        ),
    }
    return kept, uri


def load_result(result_id: str) -> Dict[str, Any]:
    directory = _artifacts.run_dir(result_id)
    path = directory / f"{_RESULT_NAME}.json"
    if not path.exists():
        raise ValidationError(
            f"no stored result {result_id!r}. Stored results live under "
            "SQT_RUNS_DIR; if the server was restarted with a different "
            "SQT_RUNS_DIR, earlier result links will not resolve."
        )
    return _artifacts.load_json(str(path))


# ── reading ──────────────────────────────────────────────────────────


def _catalog_categories() -> Dict[str, Any]:
    from standard_quant_tools.mcp.catalog import build_catalog, category_costs

    catalog = build_catalog()
    costs = category_costs(catalog)
    return {
        "note": (
            "Context cost of each tool category at connect. Tool count and "
            "cost are only loosely related -- pick by bytes, not by count."
        ),
        "categories": [
            {
                "category": name,
                "tools": count,
                "schema_bytes": size,
                "approx_tokens": size // 4,
                "tool_names": sorted(
                    e.name for e in catalog.values() if e.category == name
                ),
            }
            for name, (count, size) in sorted(costs.items(), key=lambda kv: -kv[1][1])
        ],
    }


def read(uri: str) -> Dict[str, Any]:
    """
    Resolve a `sqt://` URI to a JSON-ready payload.

    Raises ValidationError for anything unknown, malformed, or outside the
    sandbox -- the caller turns that into an MCP error rather than leaking
    a traceback to the client.
    """
    if not uri.startswith(f"{SCHEME}://"):
        raise ValidationError(f"not a {SCHEME}:// URI: {uri!r}")
    rest = uri[len(SCHEME) + 3 :]
    parts = [p for p in rest.split("/") if p]
    if not parts:
        raise ValidationError(f"empty {SCHEME}:// URI")

    kind, args = parts[0], parts[1:]

    if kind == "catalog":
        return _read_catalog(args, uri)
    if kind == "result":
        _require(args, 1, uri, "sqt://result/{result_id}")
        return load_result(args[0])
    if kind == "artifact":
        _require(args, 2, uri, "sqt://artifact/{run_id}/{name}")
        return _read_artifact(args[0], args[1])
    if kind == "model":
        _require(args, 1, uri, "sqt://model/{model_id}")
        return _read_model(args[0])
    if kind == "dataset":
        _require(args, 1, uri, "sqt://dataset/{dataset_id}")
        return _read_dataset(args[0])
    if kind == "audit":
        _require(args, 1, uri, "sqt://audit/{request_id}")
        return _read_audit(args[0])
    raise ValidationError(
        f"unknown resource kind {kind!r} in {uri!r}; expected one of "
        "catalog, result, artifact, model, dataset, audit"
    )


def _require(args: List[str], n: int, uri: str, shape: str) -> None:
    if len(args) != n:
        raise ValidationError(f"{uri!r} does not match {shape}")


def _read_catalog(args: List[str], uri: str) -> Dict[str, Any]:
    _require(args, 1, uri, "sqt://catalog/{features|capabilities|categories}")
    which = args[0]
    if which == "features":
        from standard_quant_tools.modeling.agent.models import ListFeaturesInput
        from standard_quant_tools.modeling.agent.tools import list_features

        return sanitize_for_json(
            list_features(ListFeaturesInput(category=None)).model_dump()
        )
    if which == "capabilities":
        from standard_quant_tools.modeling.capabilities import modeling_capabilities

        return sanitize_for_json(modeling_capabilities())
    if which == "categories":
        return _catalog_categories()
    raise ValidationError(
        f"unknown catalog {which!r}; expected features, capabilities or categories"
    )


def _read_artifact(run_id: str, name: str) -> Dict[str, Any]:
    directory = _artifacts.run_dir(run_id)
    if not name.endswith(".parquet"):
        name = f"{name}.parquet"
    frame = _artifacts.load_artifact(str(directory / name))
    return {
        "run_id": run_id,
        "artifact": name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "records": sanitize_for_json(frame.reset_index().to_dict(orient="records")),
    }


def _read_model(model_id: str) -> Dict[str, Any]:
    from standard_quant_tools.modeling.registry.model_registry import load_manifest

    return sanitize_for_json(load_manifest(model_id).model_dump())


def _read_dataset(dataset_id: str) -> Dict[str, Any]:
    from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

    _, meta, _ = _load_dataset_panel(dataset_id)
    return sanitize_for_json(meta)


def _read_audit(request_id: str) -> Dict[str, Any]:
    try:
        return sanitize_for_json(_cli.find_record(request_id))
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
