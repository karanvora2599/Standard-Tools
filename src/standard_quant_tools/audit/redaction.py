"""Optional field redaction (`SQT_AUDIT_REDACT_FIELDS`) applied to a tool
call's `input` before its decision record is written, plus best-effort
redaction of the same values if they leak into `error_message`."""

import copy
import logging
import os
from typing import Any, Dict, List, Optional

from standard_quant_tools.config import load_env

from .hashing import hash_payload

logger = logging.getLogger(__name__)

_warned_no_salt = False


def _redact_fields() -> List[str]:
    """Dotted field paths to redact from `input`, from `SQT_AUDIT_REDACT_FIELDS`
    (comma-separated, e.g. "account_id,client.ssn"). Empty/unset = redact
    nothing, the default."""
    raw = os.environ.get("SQT_AUDIT_REDACT_FIELDS", "")
    return [f.strip() for f in raw.split(",") if f.strip()]


def _placeholder_for(value: Any) -> str:
    """
    The single source of truth for turning a raw value into its redacted
    placeholder — used both when scrubbing `input` (`_redact_path` below)
    and when scrubbing `error_message` (`redact_text` below), so the two
    can never disagree on what a given value's placeholder is.

    A plain, unsalted SHA-256 truncated to 8 hex chars (32 bits) is
    brute-forceable offline for any field with a small/guessable value
    space (SSNs, PINs, short account IDs) — exactly the population reading
    the audit log is supposed to be kept from recovering the real value.
    Set `SQT_AUDIT_REDACT_SALT` (via a local `.env` file or a real
    environment variable — loaded through `config.load_env()`, the same
    convention every other `SQT_*` secret in this package uses) to mix a
    secret into the hash and close that gap. The salt must stay stable for
    "two records that redacted the same value compare equal on that field"
    (this module's own long-standing guarantee) to keep holding across
    process restarts — a fresh random salt per process would break that
    property, so this deliberately reads a configured, persistent salt
    rather than generating one.
    """
    load_env()
    salt = os.environ.get("SQT_AUDIT_REDACT_SALT")
    if salt:
        digest = hash_payload({"salt": salt, "value": value})
    else:
        global _warned_no_salt
        if not _warned_no_salt:
            _warned_no_salt = True
            logger.warning(
                "[audit] SQT_AUDIT_REDACT_SALT is not set — redaction "
                "placeholders are unsalted and brute-forceable offline for "
                "small value spaces (SSNs, PINs, short IDs). Set "
                "SQT_AUDIT_REDACT_SALT to close this gap. (Logged once per process.)"
            )
        digest = hash_payload(value)
    return f"<redacted:{digest[:8]}>"


def _redact_path(node: Any, parts: List[str]) -> None:
    if not isinstance(node, dict) or not parts:
        return
    key = parts[0]
    if key not in node:
        return
    if len(parts) == 1:
        node[key] = _placeholder_for(node[key])
    else:
        _redact_path(node[key], parts[1:])


def _extract_path(node: Any, parts: List[str]) -> Optional[Any]:
    """Read (without mutating) the raw value at a dotted path, mirroring
    `_redact_path`'s traversal. Returns None if the path doesn't resolve —
    including the case where the value stored there genuinely is None,
    since there's nothing to redact from `error_message` either way."""
    if not isinstance(node, dict) or not parts:
        return None
    key = parts[0]
    if key not in node:
        return None
    if len(parts) == 1:
        return node[key]
    return _extract_path(node[key], parts[1:])


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


def redact_text(text: str, raw_input: Dict[str, Any], fields: List[str]) -> str:
    """
    Best-effort companion to `_redact`: scrub `error_message` the same way
    `input` is scrubbed. `input` redaction alone isn't enough — a tool
    exception whose message echoes a redacted value back (a common Python
    pattern, e.g. `ValueError(f"Unknown account: {account_id}")`) would
    otherwise leak it unredacted in the same record where `input` is masked.

    Necessarily literal-substring-match only: each redacted field's raw
    value (read from `raw_input`, before it was redacted) is stringified
    and replaced with the same placeholder `_redact` used for it in
    `input`, wherever that exact string appears in `text`. A message that
    reformats the value (different precision, repr, etc.) won't match —
    a documented limitation, not a claim of exhaustive coverage.
    """
    if not fields:
        return text
    for dotted in fields:
        value = _extract_path(raw_input, dotted.split("."))
        if value is not None:
            text = text.replace(str(value), _placeholder_for(value))
    return text
