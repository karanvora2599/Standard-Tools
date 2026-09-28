"""Optional field redaction (`SQT_AUDIT_REDACT_FIELDS`) applied to a tool
call's `input` before its decision record is written, plus best-effort
redaction of the same values if they leak into `error_message`."""

import copy
import logging
import os
import re
import warnings
from typing import Any, Dict, Iterator, List, NamedTuple

from standard_quant_tools._env import env_str
from standard_quant_tools.config import load_env

from .hashing import hash_payload

logger = logging.getLogger(__name__)

_warned_no_salt = False

#: A path segment ending in this fans out over the elements of the list it
#: names: `positions[].symbol` is the `symbol` of every position.
_EACH = "[]"

#: A path segment ending in this names a mapping whose KEYS are redacted:
#: `positions{}` hides every key of `positions` and keeps its values. A
#: mapping keyed by account or by symbol carries the sensitive value in the
#: key, where a value path cannot reach it.
_KEYS = "{}"


class _Segment(NamedTuple):
    """One dotted segment of a redaction path."""

    key: str
    #: The segment ended in `[]`: fan out over the list it names.
    each: bool
    #: The segment ended in `{}`: redact the keys of the mapping it names.
    keys: bool


class UnsaltedRedactionWarning(UserWarning):
    """Redaction placeholders are being made without a salt."""


def _redact_fields() -> List[str]:
    """Dotted field paths to redact from `input`, from `SQT_AUDIT_REDACT_FIELDS`
    (comma-separated, e.g. "account_id,client.ssn,positions[].symbol" or,
    for the keys of a mapping, "positions{}").
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


def _segments(dotted: str) -> List[_Segment]:
    """`"positions[].symbol"` -> `[("positions", True, False), ("symbol",
    False, False)]`, and `"book{}"` -> `[("book", False, True)]`: each key,
    whether the path fans out over the list it names, and whether it
    redacts the keys of the mapping it names."""
    parts: List[_Segment] = []
    for segment in dotted.split("."):
        keys = segment.endswith(_KEYS)
        if keys:
            segment = segment[: -len(_KEYS)]
        each = segment.endswith(_EACH)
        if each:
            segment = segment[: -len(_EACH)]
        parts.append(_Segment(segment, each, keys))
    return parts


def _mappings(node: Any) -> Iterator[Dict[Any, Any]]:
    """Every mapping `node` is or holds through lists -- what a `{}`
    segment redacts the keys of. A list of mappings is walked for the
    reason a list met mid-path is: the path continues into it."""
    if isinstance(node, dict):
        yield node
    elif isinstance(node, list):
        for element in node:
            yield from _mappings(element)


def _redact_path(
    node: Any, parts: List[_Segment], key_maps: Dict[int, Dict[Any, Any]]
) -> None:
    """Replace every value `parts` reaches in `node` with its placeholder,
    and collect into `key_maps` every mapping a `{}` segment names.

    A list met where the path continues is walked element by element, so a
    field inside a list of records is reached whether or not the path says
    `[]`. That used to stop at the list: `positions.symbol` and
    `positions[].symbol` both redacted nothing, and a policy covering
    nothing looked exactly like one that matched nothing.

    Keys are collected, not renamed here. Every path is walked over the raw
    keys first and each collected mapping is renamed once afterwards (see
    `_redact`), so the order paths are listed in cannot decide whether
    `book{}` hides a key that `book.ACC-1.ssn` still has to find, and a
    mapping two paths both name is not hashed twice. A `{}` segment that is
    not the last goes on into every value of that mapping.
    """
    if isinstance(node, list):
        for element in node:
            _redact_path(element, parts, key_maps)
        return
    if not isinstance(node, dict) or not parts:
        return
    segment, rest = parts[0], parts[1:]
    if segment.key not in node:
        return
    if segment.keys:
        for mapping in _mappings(node[segment.key]):
            key_maps[id(mapping)] = mapping
            if rest:
                for value in mapping.values():
                    _redact_path(value, rest, key_maps)
        return
    if segment.each and isinstance(node[segment.key], list):
        if rest:
            for element in node[segment.key]:
                _redact_path(element, rest, key_maps)
        else:
            node[segment.key] = [
                _placeholder_for(element) for element in node[segment.key]
            ]
        return
    if not rest:
        node[segment.key] = _placeholder_for(node[segment.key])
    else:
        _redact_path(node[segment.key], rest, key_maps)


def _redact_keys(mapping: Dict[Any, Any]) -> None:
    """Rename every key of `mapping` to its placeholder, in place and in
    order, keeping each value under it.

    The placeholder is the one the same text gets as a value, salted the
    same way, so a symbol redacted as a key and as a value reads the same in
    both places. Two keys whose 32-bit placeholders collide keep both
    entries -- the later one gains a `~2` -- because a record that dropped
    one would misstate the input's shape rather than hide it.
    """
    items = list(mapping.items())
    mapping.clear()
    for key, value in items:
        placeholder = _placeholder_for(key)
        name, n = placeholder, 2
        while name in mapping:
            name, n = f"{placeholder}~{n}", n + 1
        mapping[name] = value


def _extract_path(node: Any, parts: List[_Segment]) -> Iterator[Any]:
    """Every raw value -- and, for a `{}` segment, every raw key -- `parts`
    reaches in `node`, by the same traversal as `_redact_path` and without
    mutating anything."""
    if isinstance(node, list):
        for element in node:
            yield from _extract_path(element, parts)
        return
    if not isinstance(node, dict) or not parts:
        return
    segment, rest = parts[0], parts[1:]
    if segment.key not in node:
        return
    if segment.keys:
        for mapping in _mappings(node[segment.key]):
            yield from mapping.keys()
            if rest:
                for value in mapping.values():
                    yield from _extract_path(value, rest)
        return
    if segment.each and isinstance(node[segment.key], list):
        for element in node[segment.key]:
            if rest:
                yield from _extract_path(element, rest)
            else:
                yield element
        return
    if not rest:
        yield node[segment.key]
    else:
        yield from _extract_path(node[segment.key], rest)


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

    A segment ending in `{}` redacts the KEYS of the mapping it names
    (`positions{}` for positions keyed by symbol, `book.accounts{}`), each
    replaced by the placeholder that text gets as a value; the values and
    the mapping's size are kept. Followed by more path (`accounts{}.ssn`),
    it also goes on into every value of that mapping. Only a mapping a path
    names has its keys touched.
    """
    if not fields:
        return input_dict
    result = copy.deepcopy(input_dict)
    key_maps: Dict[int, Dict[Any, Any]] = {}
    for dotted in dict.fromkeys(fields):
        _redact_path(result, _segments(dotted), key_maps)
    for mapping in key_maps.values():
        _redact_keys(mapping)
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
    wherever it appears in `text` AS A WHOLE TOKEN. A redacted mapping key
    is scrubbed the same way, with the placeholder the key got in `input`:
    `KeyError('ACC-1')` from positions keyed by account would otherwise put
    back the account the keys were hidden to protect. It used to replace every
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
