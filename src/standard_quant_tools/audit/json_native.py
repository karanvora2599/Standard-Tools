"""
What a decision record's `input` is made of by the time it is hashed and
written: JSON-native values only, so the line on disk parses back into
exactly the object its `record_hash` was taken over.

A tool's input reaches the audit package as `model_dump()` -- live Python
objects. The record hash used to be taken over those objects while the line
was written by pydantic's JSON serialiser, and wherever the two disagreed
the stored hash could never be reproduced from the line. A NaN was hashed as
`NaN` and written as `null`; a timestamp was hashed with a space and written
with a `T`; a set was hashed sorted and written in iteration order; an
integer-keyed mapping sorted its keys as integers when hashed and as strings
when re-read. Each of those left a day that verified as tampered for ever,
on a call nobody had touched -- usually one the tool had correctly refused.
A numpy value in a free-form input was worse: pydantic could not serialise
it at all, the write failed open, and the call left no record.

This is deliberately NOT `_jsonsafe.sanitize_for_json`, the boundary on the
OUTPUT side. That one turns a non-finite metric into `None`, the JSON way of
saying "undefined". An input has to survive a replay instead: `verify_replay`
rebuilds the call with `model_cls(**record["input"])`, and a NaN recorded as
`null` would replay as a different call. So non-finite floats become the
tokens `"NaN"`, `"Infinity"` and `"-Infinity"`, which pydantic's float
validation reads straight back as the values they stand for. The two
boundaries answer different questions and must not be merged.

Standard library only, and idempotent: a value already made of JSON-native
types (every record written before this existed) comes back unchanged, so
the hash of such a record is bit-identical under either rule. See the
CHANGELOG entry of 2026-09-27.
"""

import datetime as _dt
import enum
import json
import math
from typing import Any

#: The tokens a non-finite float is recorded as. They are Python's own
#: `json` spellings, and pydantic's float validation accepts each of them
#: as the value it names, which is what lets a replay restore the call.
NAN_TOKEN = "NaN"
POSITIVE_INFINITY_TOKEN = "Infinity"
NEGATIVE_INFINITY_TOKEN = "-Infinity"


def _finite_or_token(value: float) -> Any:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return NAN_TOKEN
    return POSITIVE_INFINITY_TOKEN if value > 0 else NEGATIVE_INFINITY_TOKEN


def _text(value: str) -> str:
    """A plain `str` that UTF-8 can encode. A lone surrogate is legal in a
    Python string and fatal to the JSON writer, which would drop the whole
    record; escaping it keeps the record and shows what was there."""
    plain = str.__str__(value)
    try:
        plain.encode("utf-8")
    except UnicodeEncodeError:
        return plain.encode("utf-8", "backslashreplace").decode("utf-8")
    return plain


def _key(key: Any) -> str:
    """A mapping key as JSON will spell it once the line is re-read.

    JSON keys are strings. Leaving an integer key as an integer made the
    hash sort `2` before `10` while the re-read line sorts `"10"` before
    `"2"`, so the same record hashed two ways.
    """
    if isinstance(key, str) and not isinstance(key, enum.Enum):
        return _text(key)
    native = to_json_native(key)
    if isinstance(native, str):
        return native
    if native is None:
        return "null"
    if isinstance(native, bool):
        return "true" if native else "false"
    return json.dumps(native, sort_keys=True)


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def to_json_native(obj: Any) -> Any:
    """
    `obj` rebuilt from JSON-native values only: dicts with string keys,
    lists, strings, integers, finite floats, booleans and None.

    - a non-finite float becomes "NaN", "Infinity" or "-Infinity";
    - mapping keys become strings, spelled as JSON spells them;
    - a set becomes a list in a canonical order, a tuple a list;
    - bytes become their hex string (`hash_payload`'s own convention);
    - a date, time or timestamp becomes its ISO 8601 text;
    - an enum member becomes its value, as pydantic writes it;
    - a numpy scalar or array (anything with `tolist`) becomes the plain
      value or nested list; numpy datetimes become their ISO text;
    - anything else becomes `str(obj)`.
    """
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, enum.Enum):
        return to_json_native(obj.value)
    if isinstance(obj, str):
        return _text(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return _finite_or_token(float(obj))
    if isinstance(obj, dict):
        return {_key(k): to_json_native(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted((to_json_native(v) for v in obj), key=_sort_key)
    if isinstance(obj, (list, tuple)):
        return [to_json_native(v) for v in obj]
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return bytes(obj).hex()
    dtype = getattr(obj, "dtype", None)
    if getattr(dtype, "kind", None) in ("M", "m"):
        # numpy datetimes and durations: `tolist` turns a nanosecond one into
        # a bare integer count, which no reader would recognise as a time.
        try:
            return to_json_native(obj.astype(str).tolist())
        except Exception:  # noqa: BLE001 - fall through to the text form
            return _text(str(obj))
    isoformat = getattr(obj, "isoformat", None)
    if isinstance(obj, (_dt.date, _dt.time)) or (
        callable(isoformat) and not isinstance(obj, type)
    ):
        try:
            return _text(str(isoformat()))
        except Exception:  # noqa: BLE001 - e.g. a NaT without a usable form
            return _text(str(obj))
    tolist = getattr(obj, "tolist", None)
    if callable(tolist) and not isinstance(obj, type):
        return to_json_native(tolist())
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict) and not isinstance(obj, type):
        try:
            return to_json_native(to_dict())
        except Exception:  # noqa: BLE001 - fall through to the text form
            pass
    if isinstance(obj, _dt.timedelta):
        return obj.total_seconds()
    return _text(str(obj))


__all__ = [
    "NAN_TOKEN",
    "NEGATIVE_INFINITY_TOKEN",
    "POSITIVE_INFINITY_TOKEN",
    "to_json_native",
]
