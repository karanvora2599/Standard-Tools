"""
Synthesize a plausible input for any tool, from its Pydantic schema alone.

WHY THIS EXISTS. The adversarial and contract tests need to call every tool
in the library, and hand-writing one set of arguments per tool would mean
the tools
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

import atexit
import math
import re
import shutil
import tempfile
import typing
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    get_args,
    get_origin,
)

import numpy as np
from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

#: A seeded generator, so a fuzz failure is reproducible from the test name
#: alone rather than only from a log line nobody kept.
_RNG = np.random.default_rng(20260826)

#: Names that look like tickers, for symbol-shaped fields.
SYMBOLS = ["AAPL", "MSFT", "XOM", "JPM", "PG", "NVDA"]

#: A throwaway directory for fields that name a file on disk.
#:
#: WHY THIS IS NOT JUST ANOTHER STRING. `export_audit_bundle` takes an
#: `out_path` and WRITES a zip to it, resolved against the working
#: directory. Synthesizing the generic placeholder for it meant every
#: surface run dropped a file named `a` in whatever directory pytest was
#: started from -- which for this repo is the repo root, where it was
#: eventually committed. A test that mutates the tree it is testing is a
#: test that can put its own output into a release.
#:
#: Routing these fields here keeps the tool FUZZED rather than skipped:
#: the call still happens, the write still happens, and it lands
#: somewhere the suite owns. The path is stable for the life of the
#: process so that calling one tool twice -- which the determinism layer
#: does -- passes the same arguments both times.
_SCRATCH = Path(tempfile.mkdtemp(prefix="sqt-synth-"))
atexit.register(shutil.rmtree, _SCRATCH, True)

#: A field naming a file on disk. Matched on the SUFFIX rather than by
#: substring, because `out`, `dir` and `file` appear inside plenty of
#: fields that hold no path at all.
#:
#: Public because `test_determinism.py` needs the same answer: a tool
#: naming a path is not a function of its arguments alone -- the file is
#: the other input, read or written -- and two definitions of
#: "names a path" would drift.
PATH_FIELD = re.compile(r"(^|_)(path|file|filename|dir|directory)$")

#: A field holding a DATE, by name. `date` and `timestamp` were the only two
#: recognized, so `as_of` got the generic `"a"` placeholder and
#: `score_model` refused its own input before the fuzzer reached the tool.
DATE_FIELD = re.compile(
    r"(^|_)(date|timestamp|as_of|asof|since|until|at|on|start|end)$"
)

#: A field whose length must MATCH a companion list of tickers. Generating
#: 300 weights beside 6 tickers is what made three portfolio tools
#: unsynthesizable; generating 300 of BOTH would be worse, because each
#: ticker is a fetch.
ALIGNED_TO_UNIVERSE = re.compile(
    r"(^|_)(weights|allocations|holdings|positions|quantities|shares)$"
)

#: Two id fields on one model must not collide. `compare_decisions` refused
#: because `request_id_a` and `request_id_b` both got `"a"` and a diff
#: against itself is not a diff.
DISTINCT_FIELD = re.compile(r"(_a|_b|_one|_two|id)$")


def names_a_path(model: Any) -> bool:
    """Whether any of this model's fields names a file on disk."""
    return any(PATH_FIELD.search(f) for f in model.model_fields)


#: Business dates covering a period long enough for every minimum-length
#: constraint in the library.
_DATES = [
    d.strftime("%Y-%m-%d")
    for d in __import__("pandas").bdate_range("2019-01-02", periods=2000)
]


#: Business days between a synthesized start and end. Long enough that
#: a 252-day lookback has room and short enough that a fuzz case that
#: actually fetches stays cheap.
_WINDOW = 400


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


def _maximum_length(info: Any) -> Optional[int]:
    """The `max_length` a field declares, if it declares one."""
    for meta in getattr(info, "metadata", []):
        value = getattr(meta, "max_length", None)
        if value is not None:
            return int(value)
    return None


def _count_for(info: Any, length: int) -> int:
    """
    How many items to generate, respecting BOTH bounds.

    `max_length` used to be ignored, which is why `compare_cost_models`
    (12 scenarios) and `scan_basis_dislocations` (100 pairs) were handed 300
    and dropped out of the fuzz set entirely -- silently, because an
    unbuildable model was skipped rather than reported.
    """
    count = max(length, _minimum_length(info))
    ceiling = _maximum_length(info)
    if ceiling is not None:
        count = min(count, ceiling)
    return max(count, 1)


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


def _scalar(annotation: Any, info: Any, name: str, salt: int = 0) -> Any:
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
        if "date" in name or DATE_FIELD.search(name):
            # A RANGE, not one date twice. Both ends used to resolve to
            # _DATES[0], so every windowed tool in the fuzz set was handed a
            # zero-length window and only ever exercised its empty-range
            # path -- and `DatasetSpec`, which refuses start == end
            # outright, could not be built at all.
            # Bounded: `salt` counts repeated sub-models and has no
            # ceiling of its own, and running off `_DATES` would surface as
            # an unbuildable tool rather than as an error in here.
            if "end" in name or "until" in name or "stop" in name:
                return _DATES[(_WINDOW + salt) % len(_DATES)]
            return _DATES[salt % (len(_DATES) - _WINDOW)]
        if "symbol" in name or "ticker" in name:
            return SYMBOLS[0]
        if PATH_FIELD.search(name):
            return str(_SCRATCH / name)
        if DISTINCT_FIELD.search(name):
            # Distinct, not merely plausible: two id fields that collide
            # make a tool refuse its own synthesized input. The salt covers
            # the other collision -- N copies of one nested model, each
            # carrying the same `id`, which several tools reject as
            # duplicate columns before the fuzzer reaches them.
            return name if not salt else f"{name}{salt}"
        # `salt` distinguishes REPEATS of one model. A list of scenarios or
        # of feature specs is built by calling `build` N times, and several
        # tools refuse duplicate labels or duplicate feature ids -- which is
        # correct of them, and made those tools unsynthesizable.
        return "a" if not salt else f"a{salt}"
    if annotation is dict:
        # A bare `dict` annotation, which carries no value type to read.
        return _mapping_for(name)
    raise Unsynthesizable(f"{name}: no rule for scalar {annotation!r}")


def _mapping_for(name: str) -> Dict[str, Any]:
    """A plausible untyped mapping, chosen by what the field is called."""
    if "ratio" in name:
        # `validate_financial_ratios` and `compare_ratio_frames` check these
        # for values implausible on their face, so they have to look like
        # ratios rather than like {"a": 1.0}.
        return {
            "price_to_earnings": 18.4,
            "price_to_book": 3.1,
            "dividend_yield": 0.012,
            "debt_to_equity": 0.64,
        }
    if "param" in name:
        return {}
    return {"a": 1.0, "b": 2.0}


def _records_for(name: str, count: int) -> List[Dict[str, Any]]:
    """A list of row-shaped dicts, chosen by what the field is called."""
    if "snapshot" in name or "book" in name:
        rows = []
        for index in range(count):
            mid = 100.0 + index * 0.01
            row: Dict[str, Any] = {}
            for level in range(3):
                offset = 0.01 * level
                row[f"bid_price_{level}"] = round(mid - 0.01 - offset, 4)
                row[f"ask_price_{level}"] = round(mid + 0.01 + offset, 4)
                row[f"bid_size_{level}"] = 500.0 + level * 100
                row[f"ask_size_{level}"] = 500.0 + level * 100
            rows.append(row)
        return rows
    if "record" in name or "pit" in name:
        # The point-in-time contract: two DIFFERENT timestamps, with
        # available_time after event_time, which is the rule that matters.
        return [
            {
                "entity": SYMBOLS[index % len(SYMBOLS)],
                "event_time": _DATES[index],
                "available_time": _DATES[index + 20],
                "value": float(index),
            }
            for index in range(count)
        ]
    return [{"a": float(index), "b": float(index) * 2} for index in range(count)]


def _unwrap_annotated(annotation: Any) -> Tuple[Any, Optional[float], Optional[float]]:
    """
    (base type, minimum, maximum) for an `Annotated[int, Field(ge=, le=)]`.

    An item constraint is the only place some bounds can live: pydantic puts
    `List[Annotated[int, Field(ge=1, le=60)]]` bounds on the ITEM, not the
    field, so `info.metadata` never sees them. Generating 1..n regardless
    produced `lags=[61]` for `build_model_dataset` -- refused by its own
    model, which dropped the tool out of the fuzz set entirely.
    """
    if get_origin(annotation) is not Annotated:
        return annotation, None, None
    base, *metadata = get_args(annotation)
    minimum = maximum = None
    for item in metadata:
        # Bounds arrive either directly (annotated_types.Ge/Le) or nested
        # inside a FieldInfo's own metadata, depending on how the field was
        # written; both spellings are the same constraint.
        for candidate in [item, *(getattr(item, "metadata", None) or [])]:
            if getattr(candidate, "ge", None) is not None:
                minimum = candidate.ge
            if getattr(candidate, "le", None) is not None:
                maximum = candidate.le
    return base, minimum, maximum


def _value(annotation: Any, info: Any, name: str, length: int, salt: int = 0) -> Any:
    annotation, _, _ = _unwrap_annotated(annotation)
    origin = get_origin(annotation)

    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        options = [a for a in get_args(annotation) if a is not type(None)]
        if not options:
            raise Unsynthesizable(f"{name}: Optional[None]")
        return _value(options[0], info, name, length, salt)

    if origin is typing.Literal:
        return get_args(annotation)[0]

    if origin in (list, List):
        (inner,) = get_args(annotation) or (float,)
        inner, item_min, item_max = _unwrap_annotated(inner)
        count = _count_for(info, length)
        if get_origin(inner) is typing.Literal:
            # `run_strategy_matrix` takes a list of strategy names. Every
            # option is valid, so the whole set is the most useful input.
            options = list(get_args(inner))
            return options[: max(1, min(count, len(options)))]
        if get_origin(inner) in (dict, Dict):
            return _records_for(name, min(count, 200))
        if inner is float:
            if ALIGNED_TO_UNIVERSE.search(name):
                # Must match the companion ticker list EXACTLY, and both are
                # capped at the symbol set because every ticker is a fetch.
                share = 1.0 / len(SYMBOLS)
                return [share] * len(SYMBOLS)
            if "signal" in name:
                # `run_custom_signal_backtest` refuses anything but -1/0/1
                # when signal_type is 'direction', which is the first
                # Literal option and therefore the one synthesized.
                return [float((index % 3) - 1) for index in range(count)]
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
            # Start at the item minimum and stop at its maximum, rather
            # than always 1..n: a bounded list (lags, ge=1 le=60) would
            # otherwise be handed a value its own model refuses.
            first = int(item_min) if item_min is not None else 1
            values = [first + index for index in range(count)]
            if item_max is not None:
                values = [v for v in values if v <= int(item_max)] or [first]
            return values
        if inner is str:
            if "date" in name or "timestamp" in name:
                return _DATES[:count]
            if "regime" in name:
                return [("calm" if i < count // 2 else "storm") for i in range(count)]
            # Capped at the real symbol set on purpose. Padding it out to
            # `count` would make every universe-shaped tool fetch hundreds
            # of tickers per fuzz case.
            return SYMBOLS[: max(min(count, len(SYMBOLS)), 2)]
        if get_origin(inner) in (list, List):
            return _covariance(max(count, 3))
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return [
                build(inner, length=length, salt=index + 1)
                for index in range(max(count, 1))
            ]
        raise Unsynthesizable(f"{name}: list of {inner!r}")

    if origin in (dict, Dict):
        key_type, value_type = get_args(annotation) or (str, float)
        # The WHOLE symbol set, not the first three. A tool taking both
        # `tickers` and a ticker-keyed mapping validates that they agree,
        # and three keys beside six tickers is a refusal rather than a test.
        keys = SYMBOLS
        if "grid" in name:
            # A parameter grid is a CROSS PRODUCT. Three keys of 300 values
            # is 27 million combinations, which every optimizer here refuses
            # by design -- so the old generic branch made four backtest
            # tools unsynthesizable rather than fuzzing them.
            return {"fast_period": [5, 10], "slow_period": [20, 30]}
        if get_origin(value_type) in (list, List):
            return {k: _returns(length) for k in keys}
        if get_origin(value_type) in (dict, Dict):
            if "weight" in name or "panel" in name or "score" in name:
                # {ticker: {date: value}} -- the panel shape, which is
                # validated against the companion `tickers` list. The old
                # {"shock": ...} shape is a stress scenario and is only
                # right for stress-shaped fields.
                share = 1.0 / len(keys)
                return {
                    ticker: {date: share for date in _DATES[: min(length, 60)]}
                    for ticker in keys
                }
            return {"shock": {k: -0.05 for k in keys}}
        if value_type is float:
            # Term structures and vol maps are keyed by a stringified number.
            if "expiry" in name or "implied" in name:
                return {"0.0833": 0.25, "0.25": 0.27, "0.5": 0.29}
            if "signal" in name:
                # A date-keyed signal series. `run_custom_signal_backtest`
                # refuses anything but -1/0/1 under the default
                # signal_type, and a weight-shaped 0.167 is not that.
                return {
                    date: float((index % 3) - 1)
                    for index, date in enumerate(_DATES[:length])
                }
            return {k: 1.0 / len(keys) for k in keys}
        if value_type is Any or value_type is object:
            # `object` is what `Dict[str, object]` degrades to, and it is
            # how every modeling spec carries estimator parameters. An
            # empty mapping is the valid "use the defaults" case.
            return _mapping_for(name)
        raise Unsynthesizable(f"{name}: dict of {value_type!r}")

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return build(annotation, length=length, salt=salt)

    return _scalar(annotation, info, name, salt)


def _optional_fields(model: type[BaseModel]) -> List[Tuple[str, Any]]:
    return [
        (name, info)
        for name, info in model.model_fields.items()
        if not info.is_required() and info.default is not PydanticUndefined
    ]


def build(model: type[BaseModel], *, length: int = 300, salt: int = 0) -> Any:
    """
    A valid instance of `model`, or raise `Unsynthesizable`.

    `length` sets the size of every generated series. It is a parameter
    because several tools have minimums in the hundreds and several others
    are O(n^2), so one constant cannot serve both.

    REQUIRED FIELDS FIRST, because a tool's default path is the one most
    callers take and the one worth fuzzing. But "required" is a per-field
    idea and several models here have CROSS-FIELD rules that no single
    field declares:

        exactly one of `symbol` / `ref` / `values`
        exactly one of `snapshots` / `ref`
        method='walk_forward' also needs train_window and test_window

    Filling only the required fields satisfies none of those, so the model
    refused its own synthesized input and the tool dropped out of the fuzz
    set -- silently, which is the part that mattered. So: try the required
    set, then the required set plus each optional in turn, then plus all of
    them. The first that constructs wins, and the ORIGINAL refusal is
    re-raised if none does, because that message names the real rule.
    """
    required: Dict[str, Any] = {}
    for name, info in model.model_fields.items():
        if not info.is_required() and info.default is not PydanticUndefined:
            continue
        required[name] = _value(info.annotation, info, name.lower(), length, salt)
    # Bound to its OWN name inside the block: Python deletes an
    # `except ... as` target when the block exits, so re-raising it later
    # needs a separate binding.
    first_refusal: Optional[ValidationError] = None
    try:
        return model(**required)
    except ValidationError as exc:
        first_refusal = exc

    optional = _optional_fields(model)
    extras: Dict[str, Any] = {}
    for name, info in optional:
        try:
            extras[name] = _value(info.annotation, info, name.lower(), length, salt)
        except Unsynthesizable:
            continue

    for name, value in extras.items():
        try:
            return model(**required, **{name: value})
        except ValidationError:
            continue

    try:
        return model(**required, **extras)
    except ValidationError:
        raise first_refusal


def build_arguments(model: type[BaseModel], *, length: int = 300) -> Dict[str, Any]:
    """The same, as a plain dict ready for `dispatch()`."""
    return build(model, length=length).model_dump(exclude_none=True)


def synthesize(
    model: type[BaseModel], *, length: int = 300
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Arguments for this model, or the REASON there are none.

    The reason is the point. Both fuzz layers used to wrap `build_arguments`
    in `except Exception: continue`, so a tool the synthesizer could not
    build silently left the fuzz set -- and this module's own docstring
    claimed the opposite ("a tool skipped for a missing field is visible in
    the skip list"). There was no skip list. Twenty-five tools were outside
    every adversarial and determinism check, including four that take a
    polymorphic data source and one that had just been given a new input
    field.
    """
    try:
        return build_arguments(model, length=length), None
    except Unsynthesizable as exc:
        return None, f"no rule for a field type -- {exc}"
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "(cross-field)"
        return (
            None,
            f"the model refuses its own synthesized input at {where}: {first['msg']}",
        )
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return None, f"{type(exc).__name__}: {exc}"


def synthesize_surface(
    *, length: int = 300
) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[Tuple[str, str, str]]]:
    """Every tool with a baseline, and every tool without one plus why."""
    from standard_quant_tools.agent.runtimes import all_runtimes

    built: List[Tuple[str, str, Dict[str, Any]]] = []
    skipped: List[Tuple[str, str, str]] = []
    for runtime_name, runtime in all_runtimes().items():
        for tool_name, _description, model in runtime.tool_defs:
            arguments, reason = synthesize(model, length=length)
            if arguments is None:
                skipped.append((runtime_name, tool_name, reason or "unknown"))
            else:
                built.append((runtime_name, tool_name, arguments))
    return built, skipped


def is_finite_json(value: Any) -> bool:
    """True when nothing in the structure is NaN or infinite."""
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(is_finite_json(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(is_finite_json(v) for v in value)
    return True
