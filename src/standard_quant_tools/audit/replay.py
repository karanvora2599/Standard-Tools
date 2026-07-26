"""verify_replay(): re-run a recorded tool call and compare data/output
hashes against what was originally stored."""

from typing import Any, Dict, List, Optional

from .context import _data_sources_var
from .hashing import hash_payload
from .models import ReplayResult


def verify_replay(record: Dict[str, Any]) -> ReplayResult:
    """
    Re-run a recorded tool call and compare data + output hashes against
    what was stored. A data-source mismatch with a matching output usually
    means the provider revised historical data; an output mismatch with
    matching data sources means the code/logic changed since the record
    was written.
    """
    # Local import: agent.tools imports this package, so importing it back
    # at module load time would create a circular import.
    from standard_quant_tools.agent.tools import _TOOL_DISPATCH

    tool_name = record["tool_name"]
    if tool_name not in _TOOL_DISPATCH:
        raise ValueError(f"Unknown tool '{tool_name}' in decision record.")
    fn, model_cls = _TOOL_DISPATCH[tool_name]

    token_data = _data_sources_var.set([])
    try:
        result_obj = fn(model_cls(**record["input"]))
        new_output = result_obj.model_dump()
        new_sources = list(_data_sources_var.get() or [])
    finally:
        _data_sources_var.reset(token_data)

    new_output_hash = hash_payload(new_output)
    stored_output_hash = record.get("output_hash")
    output_match: Optional[bool] = (
        new_output_hash == stored_output_hash
        if stored_output_hash is not None
        else None
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

    notes: List[str] = []
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
