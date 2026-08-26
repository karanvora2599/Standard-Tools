"""
Synthesize a plausible input for any tool, from its Pydantic schema alone.

WHY THIS EXISTS. The adversarial and contract tests need to call every tool
in the library, and hand-writing 157 sets of arguments would mean the tools
added after those tests were written are the tools nobody fuzzes -- which is
exactly backwards, since new code is where the bugs are.

Reading the schema instead means a tool added tomorrow is fuzzed tomorrow,
with no test file touched. The cost is that the generated inputs are
plausible rather than meaningful: a return series here is a random walk, not
a strategy's actual P&L. That is fine for the questions these tests ask,
which are "does it crash" and "is the output valid JSON" rather than "is the
number right" -- correctness is what the per-module test files are for.

WHAT IT DELIBERATELY REFUSES TO GUESS. A field it cannot synthesize makes
the tool UNSYNTHESIZABLE rather than producing a `None` and hoping. A tool
skipped for a missing field is visible in the skip list; a tool called with
a wrong-typed guess produces a ValidationError that looks like a finding and
is not.
"""

from __future__ import annotations

import math
import typing
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

import numpy as np
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

#: A seeded generator, so a fuzz failure is reproducible from the test name
#: alone rather than only from a log line nobody kept.
_RNG = np.random.default_rng(20260826)

#: Names that look like tickers, for symbol-shaped fields.
SYMBOLS = ["AAPL", "MSFT", "XOM", "JPM", "PG", "NVDA"]

#: Business dates covering a period long enough for every minimum-length
#: constraint in the library.
_DATES = [
    d.strftime("%Y-%m-%d")
    for d in __import__("pandas").bdate_range("2019-01-02", periods=2000)
]


class Unsynthesizable(Exception):
    """Raised when a field's type carries no way to invent a value."""


def _returns(n: int) -> List[float]:
    """A random walk's returns: the shape most list-of-float fields want."""
    return [float(x) for x in _RNG.normal(0.0004, 0.012, max(n, 1))]


def _prices(n: int) -> List[float]:
    values = 100.0 * np.exp(np.cumsum(_RNG.normal(0.0004, 0.012, max(n, 1))))
    return [float(x) for x in values]


def _volumes(n: int) -> List[float]:
    return [float(x) for x in _RNG.uniform(1e5, 9e5, max(n, 1))]


def _covariance(n: int) -> List[List[float]]:
    """A positive-definite matrix, because a random one is not."""
    root = _RNG.normal(0, 1, (n, n * 3))
    matrix = (root @ root.T) / (n * 3) * 0.04
    return [[float(v) for v in row] for row in matrix]


def _minimum_length(info: Any) -> int:
    """The `min_length` a field declares, or a workable default."""
    for meta in getattr(info, "metadata", []):
        value = getattr(meta, "min_length", None)
        if value is not None:
            return int(value)
    return 0


def _numeric_bounds(info: Any) -> Tuple[Optional[float], Optional[float]]:
    low = high = None
    for meta in getattr(info, "metadata", []):
        for attribute, target in (
            ("gt", "low"),
            ("ge", "low"),
            ("lt", "high"),
            ("le", "high"),
        ):
            value = getattr(meta, attribute, None)
            if value is None:
                continue
            if target == "low":
                low = float(value) if low is None else max(low, float(value))
            else:
                high = float(value) if high is None else min(high, float(value))
    return low, high


def _scalar(annotation: Any, info: Any, name: str) -> Any:
    low, high = _numeric_bounds(info)
    if annotation is bool:
        return False
    if annotation is int:
        if low is not None and high is not None:
            return int(max(low + 1, min(high - 1, (low + high) / 2)))
        if low is not None:
            return int(low) + 5
        if high is not None:
            return max(1, int(high) - 5)
        return 5
    if annotation is float:
        if low is not None and high is not None:
            return float((low + high) / 2)
        if low is not None:
            # `gt=0` and `ge=0` both land here; a small positive works for both.
            return float(low) + (0.25 if low == 0 else abs(low) * 0.1 + 0.25)
        if high is not None:
            return float(high) - 0.25
        # Rates, prices and volatilities all read as plain floats.
        if any(k in name for k in ("rate", "yield", "cost", "tolerance")):
            return 0.03
        if any(k in name for k in ("price", "spot", "strike", "forward")):
            return 100.0
        return 0.25
    if annotation is str:
        if "date" in name:
            return _DATES[0]
        if "symbol" in name or "ticker" in name:
            return SYMBOLS[0]
        return "a"
    raise Unsynthesizable(f"{name}: no rule for scalar {annotation!r}")


def _value(annotation: Any, info: Any, name: str, length: int) -> Any:
    origin = get_origin(annotation)

    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        options = [a for a in get_args(annotation) if a is not type(None)]
        if not options:
            raise Unsynthesizable(f"{name}: Optional[None]")
        return _value(options[0], info, name, length)

    if origin is typing.Literal:
        return get_args(annotation)[0]

    if origin in (list, List):
        (inner,) = get_args(annotation) or (float,)
        count = max(length, _minimum_length(info))
        if inner is float:
            if "price" in name or "high" in name or "low" in name or "close" in name:
                return _prices(count)
            if "volume" in name:
                return _volumes(count)
            if "strike" in name:
                return [float(80 + 5 * i) for i in range(count)]
            if "vol" in name:
                return [float(0.2 + 0.01 * i) for i in range(count)]
            return _returns(count)
        if inner is int:
            return list(range(1, count + 1))
        if inner is str:
            if "date" in name or "timestamp" in name:
                return _DATES[:count]
            if "regime" in name:
                return [("calm" if i < count // 2 else "storm") for i in range(count)]
            return SYMBOLS[: max(count, 2)]
        if get_origin(inner) in (list, List):
            return _covariance(max(count, 3))
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return [build(inner, length=length) for _ in range(max(count, 1))]
        raise Unsynthesizable(f"{name}: list of {inner!r}")

    if origin in (dict, Dict):
        key_type, value_type = get_args(annotation) or (str, float)
        keys = SYMBOLS[:3]
        if get_origin(value_type) in (list, List):
            return {k: _returns(length) for k in keys}
        if get_origin(value_type) in (dict, Dict):
            return {"shock": {k: -0.05 for k in keys}}
        if value_type is float:
            # Term structures and vol maps are keyed by a stringified number.
            if "expiry" in name or "implied" in name:
                return {"0.0833": 0.25, "0.25": 0.27, "0.5": 0.29}
            return {k: 1.0 / len(keys) for k in keys}
        if value_type is Any:
            return {"a": 1.0, "b": 2.0}
        raise Unsynthesizable(f"{name}: dict of {value_type!r}")

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return build(annotation, length=length)

    return _scalar(annotation, info, name)


def build(model: type[BaseModel], *, length: int = 300) -> Any:
    """
    A valid instance of `model`, or raise `Unsynthesizable`.

    `length` sets the size of every generated series. It is a parameter
    because several tools have minimums in the hundreds and several others
    are O(n^2), so one constant cannot serve both.
    """
    kwargs: Dict[str, Any] = {}
    for name, info in model.model_fields.items():
        if not info.is_required() and info.default is not PydanticUndefined:
            # Optional fields are left at their defaults: a tool's default
            # path is the one most callers take and the one worth fuzzing.
            continue
        kwargs[name] = _value(info.annotation, info, name.lower(), length)
    return model(**kwargs)


def build_arguments(model: type[BaseModel], *, length: int = 300) -> Dict[str, Any]:
    """The same, as a plain dict ready for `dispatch()`."""
    return build(model, length=length).model_dump(exclude_none=True)


def is_finite_json(value: Any) -> bool:
    """True when nothing in the structure is NaN or infinite."""
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(is_finite_json(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(is_finite_json(v) for v in value)
    return True
