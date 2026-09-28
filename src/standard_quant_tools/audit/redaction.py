"""Optional field redaction (`SQT_AUDIT_REDACT_FIELDS`) applied to a tool
call's `input` before its decision record is written, plus best-effort
redaction of the same values if they leak into `error_message`."""

import copy
import logging
import os
import re
import warnings
from typing import Any, Dict, Iterator, List, Tuple

from standard_quant_tools._env import env_str
from standard_quant_tools.config import load_env

from .hashing import hash_payload

logger = logging.getLogger(__name__)

_warned_no_salt = False

#: A path segment ending in this fans out over the elements of the list it
#: names: `positions[].symbol` is the `symbol` of every position.
_EACH = "[]"


class UnsaltedRedactionWarning(UserWarning):
    """Redaction placeholders are being made without a salt."""


def _redact_fields() -> List[str]:
    """Dotted field paths to redact from `input`, from `SQT_AUDIT_REDACT_FIELDS`
    (comma-separated, e.g. "account_id,client.ssn,positions[].symbol").
    Empty/unset = redact nothing, the default."""
    raw = env_str("SQT_AUDIT_REDACT_FIELDS") or ""
    return [f.strip() for f in raw.split(",") if f.strip()]


def _warn_unsalted_once() -> None:
    """Say, once per process, that placeholders are unsalted.

    It was a `logger.warning` to a logger the package gives only a
    NullHandler, so a plain script, the `sqt` CLI and the MCP server never
    showed it and the gap it describes stayed silent. `warnings.warn` reaches
    stderr by default and pytest's warnings summary; the log line is kept for
    hosts that read their logs. See the CHANGELOG entry of 2026-09-28.
    """
    global _warned_no_salt
    if _warned_no_salt:
        return
    _warned_no_salt = True
    message = (
        "SQT_AUDIT_REDACT_SALT is not set — redaction placeholders are "
        "unsalted and brute-forceable offline for small value spaces (SSNs, "
        "PINs, short IDs). Set SQT_AUDIT_REDACT_SALT to a long random secret "
        "and keep it stable. (Said once per process.)"
    )
    logger.warning("[audit] %s", message)
    # stacklevel 4: past this helper, _placeholder_for and the redaction
    # function that asked for a placeholder.
    warnings.warn(message, UnsaltedRedactionWarning, stacklevel=4)


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
        _warn_unsalted_once()
        digest = hash_payload(value)
    return f"<redacted:{digest[:8]}>"


def _segments(dotted: str) -> List[Tuple[str, bool]]:
    """`"positions[].symbol"` -> `[("positions", True), ("symbol", False)]`:
    each key, and whether the path fans out over the list it names."""
    parts: List[Tuple[str, bool]] = []
    for segment in dotted.split("."):
        each = segment.endswith(_EACH)
        parts.append((segment[: -len(_EACH)] if each else segment, each))
    return parts


def _redact_path(node: Any, parts: List[Tuple[str, bool]]) -> None:
    """Replace every value `parts` reaches in `node` with its placeholder.

    A list met where the path continues is walked element by element, so a
    field inside a list of records is reached whether or not the path says
    `[]`. That used to stop at the list: `positions.symbol` and
    `positions[].symbol` both redacted nothing, and a policy covering
    nothing looked exactly like one that matched nothing.
    """
    if isinstance(node, list):
        for element in node:
            _redact_path(element, parts)
        return
    if not isinstance(node, dict) or not parts:
        return
    (key, each), rest = parts[0], parts[1:]
    if key not in node:
        return
    if each and isinstance(node[key], list):
        if rest:
            for element in node[key]:
                _redact_path(element, rest)
        else:
            node[key] = [_placeholder_for(element) for element in node[key]]
        return
    if not rest:
        node[key] = _placeholder_for(node[key])
    else:
        _redact_path(node[key], rest)


def _extract_path(node: Any, parts: List[Tuple[str, bool]]) -> Iterator[Any]:
    """Every raw value `parts` reaches in `node`, by the same traversal as
    `_redact_path` and without mutating anything."""
    if isinstance(node, list):
        for element in node:
            yield from _extract_path(element, parts)
        return
    if not isinstance(node, dict) or not parts:
        return
    (key, each), rest = parts[0], parts[1:]
    if key not in node:
        return
    if each and isinstance(node[key], list):
        for element in node[key]:
            if rest:
                yield from _extract_path(element, rest)
            else:
                yield element
        return
    if not rest:
        yield node[key]
    else:
        yield from _extract_path(node[key], rest)


def _redact(input_dict: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """
    Replace each dotted-path field (e.g. "account_id" or "client.ssn") in a
    copy of `input_dict` with a short, non-reversible content-hash
    placeholder (`<redacted:xxxxxxxx>`) — two records that redacted the same
    underlying value still compare equal on that field without the raw
    value ever touching disk. A dotted path that doesn't match anything in
    this particular record is silently skipped, since not every tool's
    input has every configured field.

    A path reaches into lists: a list met where the path continues is
    walked element by element (`positions.symbol`), and a segment ending in
    `[]` says so explicitly (`positions[].symbol`, `nested.deep[].ssn`). A
    path ending in `[]` redacts each element of that list separately.
    """
    if not fields:
        return input_dict
    result = copy.deepcopy(input_dict)
    for dotted in fields:
        _redact_path(result, _segments(dotted))
    return result


#: Characters that continue a token: a value found next to one of these is
#: part of a longer token (`5` inside `123-45-6789`, `1.5` or `x5`) and is
#: left alone. A period only continues a token when a word character follows
#: or precedes it, so a value at the end of a sentence is still found.
_TOKEN_START = r"(?<![\w-])(?<!\w\.)"
_TOKEN_END = r"(?![\w-])(?!\.\w)"


def redact_text(text: str, raw_input: Dict[str, Any], fields: List[str]) -> str:
    """
    Best-effort companion to `_redact`: scrub `error_message` the same way
    `input` is scrubbed. `input` redaction alone isn't enough — a tool
    exception whose message echoes a redacted value back (a common Python
    pattern, e.g. `ValueError(f"Unknown account: {account_id}")`) would
    otherwise leak it unredacted in the same record where `input` is masked.

    Each redacted field's raw value (read from `raw_input`, before it was
    redacted, through the same path rules as `_redact`) is stringified and
    replaced with the same placeholder `_redact` used for it in `input`,
    wherever it appears in `text` AS A WHOLE TOKEN. It used to replace every
    substring, so redacting a quantity of 5 rewrote every 5 in the message
    -- `123-45-6789` became `123-4<redacted:...>-6789` -- and could corrupt a
    placeholder already written whose hex happened to contain the value.
    Longer values are matched first, in one pass, so no replacement is
    re-scanned. `None` and booleans are never searched for. A message that
    reformats the value (different precision, repr, etc.) won't match — a
    documented limitation, not a claim of exhaustive coverage.
    """
    if not fields:
        return text
    replacements: Dict[str, str] = {}
    for dotted in fields:
        for value in _extract_path(raw_input, _segments(dotted)):
            if value is None or isinstance(value, bool):
                continue
            spelled = str(value)
            if spelled and spelled not in replacements:
                replacements[spelled] = _placeholder_for(value)
    if not replacements:
        return text
    ordered = sorted(replacements, key=len, reverse=True)
    pattern = re.compile(
        _TOKEN_START
        + "(?:"
        + "|".join(re.escape(s) for s in ordered)
        + ")"
        + _TOKEN_END
    )
    return pattern.sub(lambda match: replacements[match.group(0)], text)
