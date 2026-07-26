"""Optional field redaction (`SQT_AUDIT_REDACT_FIELDS`) applied to a tool
call's `input` before its decision record is written."""

import copy
import os
from typing import Any, Dict, List

from .hashing import hash_payload


def _redact_fields() -> List[str]:
    """Dotted field paths to redact from `input`, from `SQT_AUDIT_REDACT_FIELDS`
    (comma-separated, e.g. "account_id,client.ssn"). Empty/unset = redact
    nothing, the default."""
    raw = os.environ.get("SQT_AUDIT_REDACT_FIELDS", "")
    return [f.strip() for f in raw.split(",") if f.strip()]


def _redact_path(node: Any, parts: List[str]) -> None:
    if not isinstance(node, dict) or not parts:
        return
    key = parts[0]
    if key not in node:
        return
    if len(parts) == 1:
        node[key] = f"<redacted:{hash_payload(node[key])[:8]}>"
    else:
        _redact_path(node[key], parts[1:])


def _redact(input_dict: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """
    Replace each dotted-path field (e.g. "account_id" or "client.ssn") in a
    copy of `input_dict` with a short, non-reversible content-hash
    placeholder (`<redacted:xxxxxxxx>`) — two records that redacted the same
    underlying value still compare equal on that field without the raw
    value ever touching disk. A dotted path that doesn't match anything in
    this particular record is silently skipped, since not every tool's
    input has every configured field.
    """
    if not fields:
        return input_dict
    result = copy.deepcopy(input_dict)
    for dotted in fields:
        _redact_path(result, dotted.split("."))
    return result
