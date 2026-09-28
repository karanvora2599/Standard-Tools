"""
The one place a non-finite number is turned into a null.

This helper was written fourteen times, once per runtime module, each copy
private and each used exactly once. Nothing was dead and nothing was wrong,
which is why it survived: fourteen correct three-line functions look like
tidy local style rather than a problem.

They had already drifted into four spellings. One tested `isinstance(value,
float)`, another added an `int`/`bool` branch that cannot change the answer
because an integer is always finite, and two carried the docstring
explaining WHY any of it is necessary while twelve did not. That is the
whole cost of the duplication: the reasoning lived in two copies out of
fourteen, so twelve readers met a bare `isinstance` check with no way to
know what it was defending against, and the next person to touch one of them
had a one-in-seven chance of finding the explanation.

`modeling.agent.feature_models` keeps its own, deliberately. It looks like a
fifteenth copy and is not the same function: it coerces with `float()` and
swallows `TypeError`/`ValueError`, so a string reaches it and leaves as
`None`, where this one passes it straight through. Merging them would have
been a silent behaviour change in whichever direction it went.

A NULL THAT SAYS NOTHING IS HALF A FIX. `ExplainsNulls` below is the other
half: a result model built on it writes one warning per null number saying
why it is null -- a Sortino that is 0/0, a profit factor with no trade, a
correlation of a series that never moved -- so "undefined here" and "not
computed" stop looking the same on the wire.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, ClassVar, Dict, Iterator, Optional, Tuple, Union

from pydantic import BaseModel, BeforeValidator, PrivateAttr, model_validator


def finite_or_none(value: Any) -> Any:
    """
    Non-finite in, null out.

    Applied before validation so a NaN never reaches the serializer. The
    alternative -- letting it through and relying on the JSON encoder --
    produces `NaN` in the payload, which is not valid JSON and which several
    MCP clients reject at the transport layer rather than at the tool. A
    rejection there is much worse than a null: it fails the whole response
    rather than the one field that could not be computed, and it fails it
    somewhere the agent cannot see or act on.

    Any real number that is not an integer is checked, not only a Python
    float: a numpy float32 is not a `float` subclass, and it carried a NaN
    straight past the old `isinstance(value, float)` test. Integers are
    always finite, and a value that is not a number is not this function's
    to reinterpret.
    """
    if (
        isinstance(value, numbers.Real)
        and not isinstance(value, numbers.Integral)
        and not math.isfinite(value)
    ):
        return None
    return value


#: How a number came to be null. It arrived as NaN (undefined), as an
#: infinity (unbounded), or already null in a required number whose only
#: meaning for null is "could not be computed" -- a row copied from a result
#: that had already nulled it.
NAN, INF, NULL = "nan", "inf", "null"

#: Why a field can be null: one sentence for every case, or a pair --
#: (why it is undefined, why it is unbounded) -- when the two differ, as
#: they do for a ratio that is 0/0 on one input and x/0 on another.
Reason = Union[str, Tuple[str, str]]

#: What a field with no reason of its own says. True of every null and
#: specific to none, which is why the fields that can be null on a legal
#: input each carry their own.
GENERIC_REASON: Tuple[str, str] = (
    "it is not a number for this input",
    "it is infinite for this input",
)


class _Key(str):
    """A mapping key inside a path, told apart from a field name so that a
    per-name mapping (weights, a correlation row) is counted rather than
    listed name by name."""


def _kind(value: Any) -> Optional[str]:
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        number = float(value)
        if math.isnan(number):
            return NAN
        if math.isinf(number):
            return INF
    return None


def _scan(value: Any, path: Tuple[Any, ...], out: Dict[Tuple[Any, ...], str]) -> None:
    """Every non-finite number in `value`, by path. A nested result model is
    skipped: it explains its own."""
    kind = _kind(value)
    if kind is not None:
        out[path] = kind
    elif isinstance(value, BaseModel):
        return
    elif isinstance(value, dict):
        for key, item in value.items():
            _scan(item, path + (_Key(key),), out)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan(item, path + (index,), out)
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            _scan(value.tolist(), path, out)
        except Exception:  # noqa: BLE001 - not a container after all
            return


def _nulled(value: Any) -> Any:
    """`value` with every non-finite number replaced by None."""
    if _kind(value) is not None:
        return None
    if isinstance(value, dict):
        return {key: _nulled(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nulled(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_nulled(item) for item in value)
    return value


_ABSENT = object()


def _value_at(root: Any, path: Tuple[Any, ...]) -> Any:
    current = root
    for token in path:
        if isinstance(current, BaseModel):
            extra = current.__pydantic_extra__ or {}
            if isinstance(token, str) and token in type(current).model_fields:
                current = getattr(current, token)
            elif token in extra:
                current = extra[token]
            else:
                return _ABSENT
        elif isinstance(current, dict):
            if token not in current:
                return _ABSENT
            current = current[token]
        elif isinstance(current, (list, tuple)):
            if not isinstance(token, int) or not 0 <= token < len(current):
                return _ABSENT
            current = current[token]
        else:
            return _ABSENT
    return current


def _is_stat(info: Any) -> bool:
    return any(
        isinstance(item, BeforeValidator) and item.func is finite_or_none
        for item in info.metadata
    )


def _dotted(path: Tuple[Any, ...]) -> str:
    text = ""
    for token in path:
        if isinstance(token, int) or token == "[*]":
            text += f"[{token}]" if isinstance(token, int) else token
        else:
            text += f".{token}" if text else str(token)
    return text


def _children(value: Any) -> Iterator[Tuple[Tuple[Any, ...], "ExplainsNulls"]]:
    if isinstance(value, ExplainsNulls):
        yield (), value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, ExplainsNulls):
                yield (_Key(key),), item
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            if isinstance(item, ExplainsNulls):
                yield (index,), item


class ExplainsNulls(BaseModel):
    """
    A result that says why each null number in it is null.

    `Stat` turns a NaN or an infinity into null, which keeps the payload
    valid JSON -- and on its own it made "this ratio is 0/0 here" and "this
    was never computed" indistinguishable, because both arrive as null with
    nothing beside them. A model built on this base writes one line into its
    own `warnings` (or `notes`) for every null number, naming the field and
    the reason from `null_reasons`, so a caller reading a null reads why.

    It covers what `Stat` cannot see from a single field:

    - numbers inside mappings and lists (`Dict[str, Stat]` metrics, weight
      maps, an equity curve), counted rather than listed when there are many;
    - a nested row model with no warnings of its own, whose nulls are
      reported by the result that holds it;
    - a required number that arrives already null, as it does when a row is
      copied from a result that had nulled it -- a comparison row built from
      a backtest's own Sortino;
    - undeclared keys on a model that allows them, which never pass through
      a field validator: they are nulled here, and said to be.
    """

    #: Field name (or mapping key) -> why it can be null; "*" for any key
    #: the model cannot name in advance. Merged along the class hierarchy,
    #: so a base can carry the reasons its subclasses share.
    null_reasons: ClassVar[Dict[str, Reason]] = {}

    _nulls: Dict[Tuple[Any, ...], Tuple[str, str]] = PrivateAttr(default_factory=dict)

    @classmethod
    def _reasons(cls) -> Dict[str, Reason]:
        reasons: Dict[str, Reason] = {}
        for klass in reversed(cls.__mro__):
            reasons.update(vars(klass).get("null_reasons", {}))
        return reasons

    @classmethod
    def _reason_text(cls, path: Tuple[Any, ...], kind: str) -> str:
        reasons = cls._reasons()
        names = [token for token in path if isinstance(token, str)]
        # "*" is the model's own default, for keys it cannot name in advance.
        reason: Reason = reasons.get("*", GENERIC_REASON)
        for candidate in (names[-1:] + names[:1]) if names else []:
            if candidate in reasons:
                reason = reasons[candidate]
                break
        undefined, unbounded = (reason, reason) if isinstance(reason, str) else reason
        if kind == NAN:
            return undefined
        if kind == INF:
            return unbounded
        return undefined if undefined == unbounded else f"{undefined}; or {unbounded}"

    @classmethod
    def _notes_field(cls) -> Optional[str]:
        for name in ("warnings", "notes"):
            if name in cls.model_fields:
                return name
        return None

    @model_validator(mode="wrap")
    @classmethod
    def _explain_nulls(cls, data: Any, handler: Any) -> Any:
        if isinstance(data, cls) or not isinstance(data, dict):
            return handler(data)
        found: Dict[Tuple[Any, ...], str] = {}
        for name, value in data.items():
            _scan(value, (name,), found)
        model = handler(data)
        extra = model.__pydantic_extra__
        if extra:
            for key in list(extra):
                extra[key] = _nulled(extra[key])
        nulls: Dict[Tuple[Any, ...], Tuple[str, str]] = {
            path: (kind, cls._reason_text(path, kind))
            for path, kind in found.items()
            if _value_at(model, path) is None
        }
        # A number that arrives already null: required, or given explicitly
        # to a field that declares why it can be null. Either way the null
        # was not this model's choice, and it would otherwise go unexplained.
        reasons = cls._reasons()
        for name, info in cls.model_fields.items():
            if (
                (name,) not in nulls
                and _is_stat(info)
                and getattr(model, name) is None
                and (info.is_required() or (name in data and name in reasons))
            ):
                nulls[(name,)] = (NULL, cls._reason_text((name,), NULL))
        for name in cls.model_fields:
            for prefix, child in _children(getattr(model, name)):
                if type(child)._notes_field() is not None:
                    continue  # it says so itself
                for path, entry in child._nulls.items():
                    full = (name,) + prefix + path
                    nulls.pop(full, None)
                    nulls[full] = entry
        model._nulls = nulls
        field = cls._notes_field()
        if nulls and field is not None:
            notes = getattr(model, field)
            named = {name for note in notes for name in _named_in(note)}
            # A number that arrived already null was explained where it was
            # nulled; a result rebuilt from its own dump must not say it twice.
            fresh = {
                path: entry
                for path, entry in nulls.items()
                if entry[0] != NULL or _dotted(path) not in named
            }
            for line in _null_lines(fresh):
                if line not in notes:
                    notes.append(line)
        return model


def _named_in(note: str) -> Iterator[str]:
    """The paths a line written by `_null_lines` names, from its label."""
    label = note.split(":", 1)[0]
    for part in label.split(", "):
        yield part.split(" ", 1)[0]


def _null_lines(nulls: Dict[Tuple[Any, ...], Tuple[str, str]]) -> Iterator[str]:
    """One line per (field pattern, reason). Three or fewer places are named;
    more are counted, so a flat equity curve is one line and not a thousand."""
    groups: Dict[Tuple[Tuple[Any, ...], str], list] = {}
    for path, (_kind_of, text) in nulls.items():
        pattern = tuple(
            (
                "[*]"
                if isinstance(token, int)
                else "*" if isinstance(token, _Key) else token
            )
            for token in path
        )
        groups.setdefault((pattern, text), []).append(path)
    for (pattern, text), paths in groups.items():
        if len(paths) <= 3:
            names = ", ".join(_dotted(path) for path in paths)
            verb = "is" if len(paths) == 1 else "are"
            yield f"{names} {verb} null: {text}."
        else:
            yield f"{_dotted(pattern)} is null in {len(paths)} places: {text}."
