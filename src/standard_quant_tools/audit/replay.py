"""verify_replay(): re-run a recorded tool call and compare data/output
hashes against what was originally stored.

Covers BOTH agent surfaces. The tool registry used to be hardcoded to
`agent.tools._TOOL_DISPATCH`, so a `run_model_experiment` record — which
modeling_dispatch had faithfully written to the audit log — could not be
replayed at all: it failed with "Unknown tool". The modeling runtime is
deliberately independent of the 46-tool registry, so replay resolves
against each in turn rather than either importing the other.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from .context import _data_sources_var
from .hashing import hash_payload
from .models import ReplayResult

# Identifiers minted fresh on every modeling run: `ds_` + 12 hex for a
# dataset, `mdl_` + 12 hex for a model (see modeling.artifacts).
_VOLATILE_ID_RE = re.compile(r"\b(?:ds|mdl)_[0-9a-f]{12}\b")


def _resolve_tool(tool_name: str) -> Tuple[Any, Any, str]:
    """
    Find `tool_name` in either agent surface.

    Local imports: both tool packages import this one, so importing them
    back at module load time would be circular.
    """
    from standard_quant_tools.agent.tools import _TOOL_DISPATCH

    if tool_name in _TOOL_DISPATCH:
        fn, model_cls = _TOOL_DISPATCH[tool_name]
        return fn, model_cls, "agent"

    from standard_quant_tools.modeling.agent.tools import MODELING_TOOL_DISPATCH

    if tool_name in MODELING_TOOL_DISPATCH:
        fn, model_cls = MODELING_TOOL_DISPATCH[tool_name]
        return fn, model_cls, "modeling"

    raise ValueError(
        f"Unknown tool {tool_name!r} in decision record — not found in the agent "
        f"tool registry or the modeling tool registry."
    )


def _has_volatile_identifiers(obj: Any) -> bool:
    """True when any string in `obj` carries a run-specific dataset/model id."""
    if isinstance(obj, str):
        return bool(_VOLATILE_ID_RE.search(obj))
    if isinstance(obj, dict):
        return any(_has_volatile_identifiers(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_volatile_identifiers(v) for v in obj)
    return False


def normalize_identifiers(obj: Any) -> Any:
    """
    Replace run-specific dataset/model ids with a stable placeholder.

    Applied to BOTH the recorded and the replayed output before comparison,
    so the comparison asks whether the substance reproduced rather than
    whether two UUIDs happened to match. Deliberately narrow: only the
    `ds_`/`mdl_` identifier pattern is rewritten, including where it appears
    inside an artifact path — a genuine change to any metric, feature list
    or fold count still shows up as a mismatch.
    """
    if isinstance(obj, str):
        return _VOLATILE_ID_RE.sub("<run_id>", obj)
    if isinstance(obj, dict):
        return {k: normalize_identifiers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_identifiers(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(normalize_identifiers(v) for v in obj)
    return obj


# Internal alias kept so call sites read consistently with the other
# underscore-prefixed helpers in this module.
_normalize_identifiers = normalize_identifiers


def verify_replay(record: Dict[str, Any]) -> ReplayResult:
    """
    Re-run a recorded tool call and compare data + output hashes against
    what was stored. A data-source mismatch with a matching output usually
    means the provider revised historical data; an output mismatch with
    matching data sources means the code/logic changed since the record
    was written.
    """
    fn, model_cls, surface = _resolve_tool(record["tool_name"])
    tool_name = record["tool_name"]

    token_data = _data_sources_var.set([])
    try:
        result_obj = fn(model_cls(**record["input"]))
        new_output = result_obj.model_dump()
        new_sources = list(_data_sources_var.get() or [])
    finally:
        _data_sources_var.reset(token_data)

    stored_output_hash = record.get("output_hash")
    new_output_hash = hash_payload(new_output)
    output_match: Optional[bool] = (
        new_output_hash == stored_output_hash
        if stored_output_hash is not None
        else None
    )

    notes: List[str] = []

    # ── Semantic comparison for surfaces with non-deterministic ids ──────
    # Modeling mints a fresh UUID-based dataset_id/model_id on every run and
    # embeds it in artifact paths, so a byte-identical re-run NEVER matches
    # literally -- every modeling replay would report a false mismatch, which
    # is worse than no replay support at all because it looks like evidence
    # of drift. Re-compare with those identifiers normalized away, so the
    # question becomes "did the SUBSTANCE reproduce" rather than "were the
    # random ids the same".
    if output_match is False and _has_volatile_identifiers(new_output):
        normalized_hash = hash_payload(_normalize_identifiers(new_output))
        stored_normalized = record.get("output_hash_normalized")
        if stored_normalized is not None:
            output_match = normalized_hash == stored_normalized
            notes.append(
                "Compared with run-specific identifiers (dataset_id/model_id and the "
                "artifact paths containing them) normalized away — these are freshly "
                "minted per run and never reproduce literally."
            )
        else:
            # Recorded before normalized hashing existed: the literal
            # mismatch cannot be distinguished from a real one.
            output_match = None
            notes.append(
                "This record predates normalized output hashing, and its output "
                "contains run-specific identifiers that never reproduce literally — "
                "so a literal mismatch here is not evidence of drift either way. "
                "Re-record to get a comparable hash."
            )

    old_by_key = {
        (s["symbol"], s["start"], s["end"], s["interval"]): s["content_hash"]
        for s in record.get("data_sources", [])
    }
    new_by_key = {
        (s["symbol"], s["start"], s["end"], s["interval"]): s["content_hash"]
        for s in new_sources
    }
    # Iterate the union of old and new keys, not just new_sources — a key
    # present in the original record but absent from the replay (e.g. the
    # tool no longer fetches a symbol/range it used to) must still be
    # reported, not silently dropped just because the loop only walked
    # what the replay happened to touch.
    data_matches: List[Dict[str, Any]] = []
    for key in sorted(set(old_by_key) | set(new_by_key)):
        symbol, start, end, interval = key
        old_hash = old_by_key.get(key)
        new_hash = new_by_key.get(key)
        data_matches.append(
            {
                "symbol": symbol,
                "start": start,
                "end": end,
                "interval": interval,
                "old_hash": old_hash,
                "new_hash": new_hash,
                "match": old_hash == new_hash,
            }
        )

    data_all_match = all(m["match"] for m in data_matches) if data_matches else True
    if not data_all_match:
        if output_match is False:
            notes.append(
                "Underlying data changed and the output changed accordingly — "
                "the provider likely revised historical values."
            )
        else:
            notes.append(
                "Underlying data changed but the output is unaffected "
                "(e.g. a scale- or shift-invariant metric) — worth a closer look."
            )
    elif output_match is False:
        notes.append(
            "Output changed even though input data is identical — "
            "code/logic likely changed since the record was written."
        )

    return ReplayResult(
        request_id=record["request_id"],
        tool_name=tool_name,
        output_match=output_match,
        data_source_matches=data_matches,
        notes=notes,
    )
