"""
Every tool, against inputs designed to break it.

THE CONTRACT BEING TESTED is narrow and absolute: for any input at all, a
tool either

  1. raises a `QuantError` or a Pydantic `ValidationError` from its input
     model — a refusal that names what was wrong and what to do instead, or
  2. returns a result whose own model holds no NaN and no infinity, before
     anything downstream cleans it.

Anything else is a defect. An `IndexError` from inside pandas, a
`ZeroDivisionError`, a `KeyError` on a column, an `AttributeError` on a
None — each of those crosses the tool boundary as a message naming no tool,
no argument and no remedy, and reads to a caller like a library bug rather
than a request the data could not support.

A BARE `ValueError` IS NOT A REFUSAL. It used to count as one, and that hid
most of what was wrong: numpy's "expected non-negative integer" for a seed of
-1, pandas' `DateParseError` for a date key that is not a date (a
`ValueError` subclass), sklearn's "Input contains NaN", a rolling window of
-1. Each is a `ValueError`, each names no argument, and `except QuantError`
misses every one of them. Only the library's own errors and the input
model's refusal qualify now. A Pydantic `ValidationError` raised by a
RESULT model is not a refusal either: it is the tool building an answer its
own schema rejects.

THE NON-FINITE CHECK READS THE RAW RESULT. It used to read what `dispatch()`
returned, which `sanitize_for_json` had already cleaned, so it could never
fail -- and a hundred result fields carried NaN or infinity past it, reaching
the wire as a null with no reason beside it. The check now reads the result
model's own dump, captured on its way into the audit record. A `Stat` field
is already null there, so anything non-finite it finds is a field that is
not typed to say "undefined", and the failure names the path.

INPUTS ARE SYNTHESIZED FROM THE SCHEMA, not hand-written. A hand-written
fixture list would cover the tools that existed when it was written, which
means the newest tools — where the bugs are — would be the ones never
fuzzed. See `synth.py`. The synthesized input is then given every published
fixture it can name (`hermetic.published_baseline`), so a probe reaches the
computation instead of re-testing the argument lookup; and the probes
themselves are generated from each input schema: a negative seed, an
unparseable date or date key, a duplicated list entry, NaN and infinity as
scalars and inside lists and mappings, empty and single-element lists, a
window longer than the data, a swapped min/max pair.

WHAT IS NOT TESTED HERE: whether the answers are right. These tests ask
"does it fail cleanly", and correctness is what the per-module test files
are for. Nor are the sizes that make a tool run for minutes (a hundred
thousand components or bootstrap draws): bounding those is a schema
question, and a probe that could hang has no place in the default run.
"""

from __future__ import annotations

import copy
import json
import math
import re
import typing
import warnings
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple

import pydantic
import pytest

#: About a minute and a half for some 5,000 calls, most of which now reach
#: the computation; the fake market that replaced live yfinance and the
#: capped resampling counts are what keep it there. `25_testing.md` records
#: the figure. Marked `slow` so `-m "not slow"`
#: skips it while iterating; the default run includes it, because a fuzzing
#: suite nobody runs by default is a fuzzing suite nobody runs.
pytestmark = pytest.mark.slow

from standard_quant_tools.error import QuantError

from . import hermetic, synth
from .synth import is_finite_json, synthesize_surface

#: A refusal. Anything outside this set is a defect -- a bare ValueError
#: included, for the reason in the module docstring.
CLEAN_REFUSAL = (QuantError, pydantic.ValidationError)

#: Exceptions that are never acceptable, listed so the failure message can
#: say WHY rather than only that something was raised.
NEVER = (
    IndexError,
    KeyError,
    AttributeError,
    TypeError,
    ZeroDivisionError,
    RecursionError,
    UnboundLocalError,
    OverflowError,
)


def _every_tool() -> List[Tuple[str, str, type]]:
    from standard_quant_tools.agent.runtimes import all_runtimes

    return [
        (runtime_name, tool_name, model)
        for runtime_name, runtime in all_runtimes().items()
        for tool_name, _description, model in runtime.tool_defs
    ]


TOOLS, SKIPPED = synthesize_surface()
TOOL_IDS = [f"{r}:{t}" for r, t, _ in TOOLS]

#: Tools with no synthesized baseline, and why. DECLARED, because the
#: alternative was what this file did for its whole life: swallow the
#: failure and carry on with a smaller parametrization that looks exactly
#: like a full one.
#:
#: An entry here is a real gap, not a pardon.
#:
#: EMPTY, and it has been non-empty exactly once. `run_feature_ablation`
#: sat here because its `spec` field was annotated `typing.Any` -- which
#: also meant the MCP server advertised no schema for it, so the fuzzer's
#: inability to invent one was a symptom rather than the problem. Typing it
#: as the `ModelSpec` the tool already constructs fixed both, and the
#: stale-exemption guard below is what said so.
EXPECTED_UNSYNTHESIZABLE: dict[str, str] = {}

#: Tools whose published baseline still refuses, and why. Every other tool
#: must RETURN on its baseline, not merely fail cleanly: a baseline that
#: refuses turns every probe of that tool into a test of the refusal. These
#: need a feed, a key or a store this machine does not have, a model kind
#: the fixtures do not train, or a precondition their own first call
#: consumes. An entry that starts returning fails below, so the list cannot
#: outlive its reasons.
EXPECTED_BASELINE_REFUSAL: dict[str, str] = {
    "fetch_tick_tape": "needs a tick feed; yfinance serves none",
    "fetch_quote_panel": "needs a quote feed; yfinance serves none",
    "fetch_order_book": "needs an L2 depth feed; yfinance serves none",
    "fetch_order_events": "needs an order-by-order feed; yfinance serves none",
    "preflight_vendor_request": "yfinance routes no named vendor datasets",
    "get_microstructure_metrics": "reads from Polygon, which needs a key",
    "get_trade_profile": "reads from Polygon, which needs a key",
    "check_spread_proxy": "reads from Polygon, which needs a key",
    "list_remote_models": "needs a remote model store",
    "pull_model_package": "needs a remote model store",
    "predict_survival_curve": "needs a survival model; the fixtures train ridge",
    "score_prediction_intervals": "needs interval predictions; the fixtures make points",
    "monitor_model": "outcomes_ref is read as a path, not a published reference",
}


def _dispatch_tables() -> Dict[str, Tuple[Any, type]]:
    from standard_quant_tools.agent.runtimes import all_runtimes

    return {
        tool: entry
        for runtime in all_runtimes().values()
        for tool, entry in runtime.dispatch_table.items()
    }


def _models_in(annotation: Any, found: Set[type]) -> None:
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        if annotation in found:
            return
        found.add(annotation)
        for info in annotation.model_fields.values():
            _models_in(info.annotation, found)
        return
    for argument in typing.get_args(annotation):
        _models_in(argument, found)


def _result_only_models() -> Set[str]:
    """Names of models that appear in some tool's RESULT and in no input."""
    outputs: Set[type] = set()
    inputs: Set[type] = set()
    for fn, model in _dispatch_tables().values():
        try:
            hints = typing.get_type_hints(fn)
        except Exception:  # noqa: BLE001 - an unresolvable hint names no model
            hints = {}
        _models_in(hints.get("return"), outputs)
        _models_in(model, inputs)
    return {m.__name__ for m in outputs} - {m.__name__ for m in inputs}


RESULT_ONLY_MODELS = _result_only_models()

#: The raw result of the most recent call, captured on its way into the
#: audit record and before `sanitize_for_json` touched it.
_RAW: Dict[str, Any] = {}


@pytest.fixture(scope="module", autouse=True)
def _capture_raw_results():
    """
    Read every result as the tool's own model dumped it.

    `Runtime.dispatch` passes the dump from `_run_and_record` straight to
    `sanitize_for_json`; wrapping the first is how this layer sees what the
    second would otherwise hide. The record is still written by the real
    function, so the audit assertion in conftest covers these calls too.
    """
    import standard_quant_tools.agent.runtimes as runtimes

    real = runtimes._run_and_record

    def capturing(tool_name: str, fn: Any, model_instance: Any) -> Dict[str, Any]:
        result = real(tool_name, fn, model_instance)
        _RAW["result"] = result
        return result

    patch = pytest.MonkeyPatch()
    patch.setattr(runtimes, "_run_and_record", capturing)
    try:
        yield
    finally:
        patch.undo()


def _call(runtime_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
    from standard_quant_tools.agent.runtimes import resolve

    _RAW.clear()
    with warnings.catch_warnings():
        # numpy warns on empty slices and zero division; a warning is not a
        # defect, and the assertion below is about what the tool RETURNS.
        warnings.simplefilter("ignore")
        return resolve(runtime_name).dispatch(tool_name, arguments)


def _nonfinite_paths(value: Any, path: str = "") -> List[str]:
    if isinstance(value, float):
        return [] if math.isfinite(value) else [f"{path or '<result>'}={value!r}"]
    if isinstance(value, dict):
        return [
            p
            for k, v in value.items()
            for p in _nonfinite_paths(v, f"{path}.{k}" if path else str(k))
        ]
    if isinstance(value, (list, tuple)):
        return [
            p for i, v in enumerate(value) for p in _nonfinite_paths(v, f"{path}[{i}]")
        ]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _nonfinite_paths(value.tolist(), path)
        except Exception:  # noqa: BLE001 - not a container after all
            return []
    return []


def _problem(
    runtime_name: str, tool_name: str, arguments: Dict[str, Any], mutation: str
) -> Optional[str]:
    """None when the call refused cleanly or returned a finite result;
    otherwise one line saying what went wrong."""
    try:
        result = _call(runtime_name, tool_name, arguments)
    except pydantic.ValidationError as refusal:
        if refusal.title in RESULT_ONLY_MODELS:
            return (
                f"{mutation}: the tool built a {refusal.title} its own result "
                f"model refuses -- {str(refusal)[:200]}"
            )
        if not str(refusal).strip():
            return f"{mutation}: refused with an EMPTY message"
        return None
    except QuantError as refusal:
        if not str(refusal).strip():
            return f"{mutation}: refused with an EMPTY message"
        return None
    except NEVER as exc:
        return (
            f"{mutation}: unhandled {type(exc).__name__}: {str(exc)[:200]} -- "
            "a tool must refuse by name or return a result"
        )
    except ValueError as exc:
        return (
            f"{mutation}: bare {type(exc).__module__}.{type(exc).__name__}: "
            f"{str(exc)[:200]} -- not a refusal; raise the library "
            "ValidationError naming the argument, or refuse it in the schema"
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad
        return f"{mutation}: unexpected {type(exc).__name__}: {str(exc)[:200]}"
    raw = _RAW.get("result")
    if raw is None:
        return (
            f"{mutation}: the raw result was not captured, so the non-finite "
            "check below would pass vacuously -- Runtime.dispatch no longer "
            "goes through agent.runtimes._run_and_record"
        )
    bad = _nonfinite_paths(raw)
    if bad:
        return (
            f"{mutation}: non-finite result field(s) {bad[:4]} before "
            "sanitisation. Type the field Stat (or a finite-or-None element "
            "type) and say why it is undefined, or refuse the input that "
            "produces it"
        )
    try:
        json.dumps(result, allow_nan=False, default=str)
    except ValueError as exc:
        return f"{mutation}: the payload is not strict JSON: {exc}"
    if not is_finite_json(result):
        return f"{mutation}: a non-finite float reached the payload"
    return None


def _assert_clean(
    runtime_name: str, tool_name: str, arguments: Dict[str, Any], mutation: str
) -> None:
    """Either a named refusal, or a finite result. Nothing else."""
    problem = _problem(runtime_name, tool_name, arguments, mutation)
    if problem is not None:
        pytest.fail(f"{tool_name}: {problem}")


def _assert_all_clean(
    runtime_name: str, tool_name: str, cases: List[Tuple[str, Dict[str, Any]]]
) -> None:
    """Every case, and every failure reported together: a tool with three
    distinct defects says so in one run rather than one per fix."""
    problems = []
    for description, arguments in cases:
        problem = _problem(runtime_name, tool_name, arguments, description)
        if problem is not None:
            problems.append(problem)
    if problems:
        pytest.fail(
            f"{tool_name}: {len(problems)} of {len(cases)} probes failed:\n  "
            + "\n  ".join(problems[:25])
        )


def _uniquely(model: type, arguments: Dict[str, Any], tag: str) -> Dict[str, Any]:
    """A run_id and output file of its own for every publishing call, so
    'already published' never answers in place of the path under test."""
    out = dict(arguments)
    stamp = f"{zlib.crc32(tag.encode()):08x}"
    if "run_id" in model.model_fields and (
        "run_id" in out or model.model_fields["run_id"].is_required()
    ):
        out["run_id"] = f"probe_{stamp}"
    for name in model.model_fields:
        value = out.get(name)
        if (
            name.startswith("out")
            and synth.PATH_FIELD.search(name)
            and isinstance(value, str)
            and value
        ):
            head, _sep, tail = value.replace("\\", "/").rpartition("/")
            out[name] = f"{head}/{stamp}_{tail}" if head else f"{stamp}_{tail}"
    return out


_MODELS = {tool: model for _runtime, tool, model in _every_tool()}


class TestTheBaselineHolds:
    """A valid input must produce a valid result. If this fails the
    mutations below are testing nothing."""

    @pytest.mark.parametrize("runtime,tool,arguments", TOOLS, ids=TOOL_IDS)
    def test_a_synthesized_input_is_handled_cleanly(self, runtime, tool, arguments):
        _assert_clean(runtime, tool, arguments, "a valid synthesized input")

    @pytest.mark.parametrize("runtime,tool,arguments", TOOLS, ids=TOOL_IDS)
    def test_a_published_baseline_returns_a_result(
        self, runtime, tool, arguments, published
    ):
        """
        Not merely cleanly: it RETURNS, so every probe of it reaches the
        computation. The declared exceptions must still refuse -- one that
        starts returning is taken off the list rather than left to exempt a
        tool that no longer needs it.
        """
        model = _MODELS[tool]
        args = _uniquely(
            model,
            hermetic.published_baseline(tool, model, arguments, published),
            f"{tool}:base",
        )
        problem = _problem(runtime, tool, args, "its published baseline")
        assert problem is None, f"{tool}: {problem}"
        returned = "result" in _RAW
        if tool in EXPECTED_BASELINE_REFUSAL:
            assert not returned, (
                f"{tool} returns on its published baseline now; remove it from "
                "EXPECTED_BASELINE_REFUSAL so its probes are held to the result"
            )
        else:
            assert returned, (
                f"{tool} refused its published baseline, so every probe of it "
                "tests only that refusal. Teach hermetic.published_baseline "
                "the input, or declare the tool in EXPECTED_BASELINE_REFUSAL "
                "with the reason."
            )

    def test_every_expected_refusal_names_a_real_tool(self):
        live = {tool for _runtime, tool, _args in TOOLS}
        assert set(EXPECTED_BASELINE_REFUSAL) <= live

    def test_every_tool_without_a_baseline_is_declared(self):
        """
        THE GUARD THAT WAS MISSING, and the reason 25 tools sat outside
        every check in this file without anyone noticing.

        Collection swallowed `Exception` and continued, so a tool the
        synthesizer could not build produced a SMALLER parametrization —
        which is indistinguishable from a full one in the output. The floor
        that was supposed to catch this asked for 100 synthesizable tools
        out of a surface that had 178, leaving 78 tools of headroom for the
        gap to grow in silently.

        Now every absence is named. Adding a tool the synthesizer cannot
        build fails HERE, at the tool, rather than reducing the coverage of
        every other test in the file.
        """
        undeclared = {
            name: reason
            for _runtime, name, reason in SKIPPED
            if name not in EXPECTED_UNSYNTHESIZABLE
        }
        assert not undeclared, (
            "these tools have no synthesized baseline and are therefore in "
            "NO adversarial or determinism check, which the suite will not "
            f"otherwise tell you: {undeclared}. Either teach synth.py the "
            "shape, or add the tool to EXPECTED_UNSYNTHESIZABLE with the "
            "reason."
        )

    def test_no_declared_gap_has_quietly_been_fixed(self):
        """
        The other direction. A tool that becomes synthesizable should leave
        the list, or the list becomes a place where exemptions accumulate
        and outlive their reason.
        """
        live = {name for _runtime, name, _reason in SKIPPED}
        stale = sorted(set(EXPECTED_UNSYNTHESIZABLE) - live)
        assert not stale, (
            f"{stale} can be synthesized now and should be removed from "
            "EXPECTED_UNSYNTHESIZABLE so the list keeps meaning something"
        )

    def test_the_synthesizer_covers_most_of_the_surface(self):
        """
        A guard on the guard. If a refactor breaks the synthesizer, every
        test in this file silently passes on an empty parametrization —
        which looks exactly like success.

        Expressed against the LIVE surface rather than a constant: 100 was
        chosen when the library had far fewer tools, and a fixed floor turns
        into slack the moment the surface grows past it.
        """
        total = len(_every_tool())
        assert len(TOOLS) >= total - len(EXPECTED_UNSYNTHESIZABLE), (
            f"{len(TOOLS)} of {total} tools synthesized, but only "
            f"{len(EXPECTED_UNSYNTHESIZABLE)} are declared unsynthesizable"
        )


def _mutations(arguments: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Hostile variants of a valid input, by the shape of each value.

    Each targets a specific way numeric code fails: an empty sequence
    reaching a mean, a constant series reaching a division by its own
    standard deviation, a NaN propagating silently to the output, a
    magnitude that overflows an exponential.
    """
    out: List[Tuple[str, Dict[str, Any]]] = []

    for key, value in arguments.items():
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], (int, float))
            and not isinstance(value[0], bool)
        ):
            out.append((f"{key}=[] (empty)", {**arguments, key: []}))
            out.append((f"{key}=[x] (single)", {**arguments, key: value[:1]}))
            out.append(
                (f"{key} all-identical", {**arguments, key: [value[0]] * len(value)})
            )
            out.append((f"{key} all-zero", {**arguments, key: [0.0] * len(value)}))
            with_nan = list(value)
            with_nan[len(with_nan) // 2] = float("nan")
            out.append((f"{key} contains NaN", {**arguments, key: with_nan}))
            with_inf = list(value)
            with_inf[0] = float("inf")
            out.append((f"{key} contains inf", {**arguments, key: with_inf}))
            out.append((f"{key} huge", {**arguments, key: [v * 1e300 for v in value]}))
            out.append((f"{key} tiny", {**arguments, key: [v * 1e-300 for v in value]}))
            out.append((f"{key} negated", {**arguments, key: [-v for v in value]}))
            out.append(
                (f"{key} truncated", {**arguments, key: value[: len(value) // 3]})
            )
        elif isinstance(value, float):
            out.append((f"{key}=0.0", {**arguments, key: 0.0}))
            out.append((f"{key} negative", {**arguments, key: -abs(value) - 1.0}))
            out.append((f"{key} huge", {**arguments, key: 1e300}))
            out.append((f"{key} tiny", {**arguments, key: 1e-300}))
            out.append((f"{key}=NaN", {**arguments, key: float("nan")}))
        elif isinstance(value, dict) and value:
            out.append((f"{key}={{}} (empty)", {**arguments, key: {}}))
            first = next(iter(value))
            out.append(
                (f"{key} single entry", {**arguments, key: {first: value[first]}})
            )
    return out


# ── probes from the input schema ────────────────────────────────────────

_NAN, _INF = float("nan"), float("inf")

#: A seed, by name. numpy refuses a negative one with a bare ValueError and
#: the extension with a RuntimeError from its argument cast.
_SEED = re.compile(r"(^|_)seed$")

#: A length measured in bars, which can be longer than the data. A COUNT of
#: draws or permutations is left out: 5000 of those is only slow, and
#: nothing about it is longer than anything.
_WINDOW = re.compile(
    r"(window|period|lookback|span|lag|horizon|length|components|top_n|bars$)"
)
_COUNT = re.compile(
    r"(permutation|bootstrap|path|simulation|iteration|trial|draw|sample|"
    r"observation|fits|resample|points|seed)"
)
_DATE = re.compile(
    r"(^|_)(date|timestamp|as_of|asof|since|until|start|end|expiry|valuation)($|_)"
)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")

#: (low, high) name patterns whose values can be swapped into a
#: contradiction the schema should refuse or the tool should survive.
_PAIRS = [
    (r"^min_(.+)$", r"max_\1"),
    (r"^(.+)_min$", r"\1_max"),
    (r"^(.*)low(.*)$", r"\1high\2"),
    (r"^(.*)lower(.*)$", r"\1upper\2"),
    (r"^fast(.*)$", r"slow\1"),
    (r"^short(.*)$", r"long\1"),
    (r"^(.*)start(.*)$", r"\1end\2"),
    (r"^train_(.+)$", r"test_\1"),
]


def _unwrap(annotation: Any) -> Any:
    while True:
        origin = typing.get_origin(annotation)
        if origin is typing.Annotated:
            annotation = typing.get_args(annotation)[0]
            continue
        if origin is typing.Union:
            options = [a for a in typing.get_args(annotation) if a is not type(None)]
            if len(options) == 1:
                annotation = options[0]
                continue
        return annotation


def _shape(annotation: Any) -> str:
    """'int', 'float', 'str', 'list[str]', 'dict[float]', 'model', ..."""
    ann = _unwrap(annotation)
    origin = typing.get_origin(ann)
    if ann is bool or origin is typing.Literal:
        return "fixed"
    if isinstance(ann, type) and issubclass(ann, pydantic.BaseModel):
        return "model"
    for scalar in (int, float, str):
        if ann is scalar:
            return scalar.__name__
    if origin in (list, typing.List):
        args = typing.get_args(ann)
        return f"list[{_shape(args[0]) if args else 'any'}]"
    if origin in (dict, typing.Dict):
        args = typing.get_args(ann)
        return f"dict[{_shape(args[1]) if len(args) == 2 else 'any'}]"
    if ann is dict:
        return "dict[any]"
    return "any"


def _pairs(model: type, values: Dict[str, Any]) -> List[Tuple[str, str]]:
    names = [n for n in model.model_fields if values.get(n) is not None]
    found = []
    for low in names:
        for pattern, replacement in _PAIRS:
            if re.match(pattern, low):
                high = re.sub(pattern, replacement, low)
                if high in names and high != low:
                    found.append((low, high))
    return found


def _scalar_probes(name: str, shape: str, present: bool) -> List[Tuple[str, Any]]:
    if shape == "int":
        probes = [
            ("=0", 0),
            ("=-1" + (" (negative seed)" if _SEED.search(name) else ""), -1),
        ]
        if _WINDOW.search(name) and not _COUNT.search(name):
            probes.append(("=5000 (longer than the data)", 5000))
        return probes
    if shape == "float":
        probes = [("=+inf", _INF), ("=-inf", -_INF)]
        return probes if present else [("=NaN", _NAN)] + probes
    if shape == "str" and not synth.PATH_FIELD.search(name):
        probes = [("='' (empty)", "")]
        if _DATE.search(name):
            probes += [("='not-a-date'", "not-a-date"), ("='2019-13-45'", "2019-13-45")]
        return probes
    return []


def _nested_probes(
    prefix: str, model: type, value: Dict[str, Any]
) -> List[Tuple[str, Dict[str, Any]]]:
    """Scalar probes one level inside a nested model's value."""
    out = []
    for name, info in model.model_fields.items():
        if name not in value:
            continue
        shape = _shape(info.annotation)
        if shape == "float":
            cases = [("=NaN", _NAN), ("=+inf", _INF)]
        elif shape == "int":
            cases = [("=-1", -1)]
        elif shape == "str" and _DATE.search(name):
            cases = [("='not-a-date'", "not-a-date")]
        else:
            continue
        for label, probe in cases:
            out.append((f"{prefix}.{name}{label}", {**value, name: probe}))
    for low, high in _pairs(model, value):
        out.append(
            (
                f"{prefix}: {low}<->{high} swapped",
                {**value, low: value[high], high: value[low]},
            )
        )
    return out


def _schema_probes(
    model: type, arguments: Dict[str, Any]
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Hostile variants of a valid input, by the TYPE each field declares.

    `_mutations` reads the values it was given, so it never touched an
    integer, a string, a list of names, a mapping's values or a nested
    model -- and those are where a negative seed, an unparseable date, a
    duplicated ticker and a NaN inside a mapping got through.
    """
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name, info in model.model_fields.items():
        shape = _shape(info.annotation)
        value = arguments.get(name)
        present = name in arguments
        for label, probe in _scalar_probes(name, shape, present):
            out.append((f"{name}{label}", {**arguments, name: probe}))
        if shape == "list[str]":
            out.append((f"{name}=[] (empty)", {**arguments, name: []}))
            if isinstance(value, list) and value:
                if len(value) > 1:
                    out.append((f"{name}=[x] (single)", {**arguments, name: value[:1]}))
                out.append(
                    (
                        f"{name} duplicated",
                        {**arguments, name: [value[0]] * max(2, len(value))},
                    )
                )
        elif shape.startswith("dict[") and isinstance(value, dict) and value:
            first = next(iter(value))
            if shape in ("dict[float]", "dict[int]", "dict[any]"):
                out.append(
                    (f"{name} value NaN", {**arguments, name: {**value, first: _NAN}})
                )
                out.append(
                    (f"{name} value +inf", {**arguments, name: {**value, first: _INF}})
                )
            if _ISO_DATE.match(str(first)):
                relabelled = dict(value)
                relabelled["not-a-date"] = relabelled.pop(first)
                out.append(
                    (f"{name} one key not a date", {**arguments, name: relabelled})
                )
                out.append(
                    (
                        f"{name} keyed by tickers instead of dates",
                        {
                            **arguments,
                            name: dict(zip(hermetic.SYMBOLS, value.values())),
                        },
                    )
                )
        elif shape == "model" and isinstance(value, dict):
            inner = _unwrap(info.annotation)
            for label, nested in _nested_probes(name, inner, value):
                out.append((label, {**arguments, name: nested}))
        elif (
            shape == "list[model]"
            and isinstance(value, list)
            and value
            and isinstance(value[0], dict)
        ):
            inner = _unwrap(typing.get_args(_unwrap(info.annotation))[0])
            out.append((f"{name}=[] (empty)", {**arguments, name: []}))
            if len(value) > 1:
                out.append((f"{name}=[first] (single)", {**arguments, name: value[:1]}))
            for label, nested in _nested_probes(f"{name}[0]", inner, value[0]):
                out.append((label, {**arguments, name: [nested] + value[1:]}))
    for low, high in _pairs(model, arguments):
        swapped = {**arguments, low: arguments[high], high: arguments[low]}
        out.append((f"{low}<->{high} swapped", swapped))
        if isinstance(arguments[low], str) and isinstance(arguments[high], str):
            out.append(
                (f"{low}=={high} (zero-length)", {**arguments, high: arguments[low]})
            )
    lists = [
        f
        for f, info in model.model_fields.items()
        if _shape(info.annotation) in ("list[float]", "list[int]")
        and isinstance(arguments.get(f), list)
        and len(arguments[f]) > 3
    ]
    if len(lists) >= 2:
        a, b = lists[0], lists[1]
        out.append(
            (
                f"{b} half the length of {a}",
                {**arguments, b: arguments[b][: len(arguments[b]) // 2]},
            )
        )
    return out


#: Built once. Parametrizing per mutation would produce ~20,000 test IDs and
#: a collection phase longer than the run; the loop inside each test keeps
#: the surface identical and the report readable.
MUTATION_COUNTS = {tool: len(_mutations(args)) for _r, tool, args in TOOLS}
SCHEMA_PROBE_COUNTS = {
    tool: len(_schema_probes(_MODELS[tool], args)) for _r, tool, args in TOOLS
}


class TestHostileInputs:
    @pytest.mark.parametrize("runtime,tool,arguments", TOOLS, ids=TOOL_IDS)
    def test_no_mutation_produces_an_unhandled_exception(
        self, runtime, tool, arguments, published
    ):
        """
        The core of the regime. Ten mutation families per numeric argument,
        every one of which must produce a refusal or a finite result -- on
        the published baseline, so a mutation reaches the computation.
        """
        model = _MODELS[tool]
        base = hermetic.published_baseline(tool, model, arguments, published)
        _assert_all_clean(
            runtime,
            tool,
            [
                (d, _uniquely(model, m, f"{tool}:m{i}"))
                for i, (d, m) in enumerate(_mutations(base))
            ],
        )

    @pytest.mark.parametrize("runtime,tool,arguments", TOOLS, ids=TOOL_IDS)
    def test_no_schema_probe_produces_an_unhandled_exception(
        self, runtime, tool, arguments, published
    ):
        """
        The probes generated from the input schema: a negative seed, an
        unparseable date and date key, duplicated and empty lists, NaN and
        infinity as scalars and inside mappings and nested models, a window
        longer than the data, swapped and equal min/max pairs.
        """
        model = _MODELS[tool]
        base = hermetic.published_baseline(tool, model, arguments, published)
        _assert_all_clean(
            runtime,
            tool,
            [
                (d, _uniquely(model, p, f"{tool}:p{i}"))
                for i, (d, p) in enumerate(_schema_probes(model, base))
            ],
        )

    def test_the_mutation_set_is_not_empty(self):
        """Another guard on a guard: a synthesizer returning only scalars
        would make the loop above a no-op."""
        total = sum(MUTATION_COUNTS.values())
        assert total >= 500, (
            f"only {total} mutations across {len(TOOLS)} tools; the fuzzer "
            "is not exercising anything"
        )

    def test_every_probe_family_is_generated(self):
        """
        The probes that found real defects, counted across the surface, so a
        refactor of the generator cannot quietly drop one: each family must
        reach at least a few tools.
        """
        labels = [
            label
            for _r, tool, args in TOOLS
            for label, _args in _schema_probes(_MODELS[tool], args)
        ]
        families = {
            "negative seed": "(negative seed)",
            "unparseable date": "='not-a-date'",
            "impossible date": "='2019-13-45'",
            "duplicated list entry": " duplicated",
            "infinite scalar": "=+inf",
            "NaN inside a mapping": " value NaN",
            "window longer than the data": "(longer than the data)",
            "swapped pair": " swapped",
            "zero-length range": "(zero-length)",
            "empty list": "=[] (empty)",
        }
        counts = {
            name: sum(marker in label for label in labels)
            for name, marker in families.items()
        }
        thin = {name: n for name, n in counts.items() if n < 3}
        assert not thin, f"probe families reaching fewer than three tools: {thin}"
        assert sum(SCHEMA_PROBE_COUNTS.values()) >= 2000


class TestRefusalsAreActionable:
    """
    A refusal that says "invalid input" is barely better than a crash. The
    library's position is that an error should be self-correcting, and
    these are the properties that make one so.
    """

    @pytest.mark.parametrize("runtime,tool,arguments", TOOLS, ids=TOOL_IDS)
    def test_an_empty_series_refusal_names_a_number(self, runtime, tool, arguments):
        """
        "Not enough data" is unactionable; "12 observations, needs 30" tells
        the caller exactly what to change.
        """
        for key, value in arguments.items():
            if not (isinstance(value, list) and len(value) > 2):
                continue
            if not isinstance(value[0], (int, float)):
                continue
            try:
                _call(runtime, tool, {**arguments, key: value[:1]})
            except CLEAN_REFUSAL as refusal:
                message = str(refusal)
                assert any(char.isdigit() for char in message), (
                    f"{tool} refused a 1-element {key} without naming any "
                    f"number: {message[:140]}"
                )
            except Exception:
                pass  # covered by the unhandled-exception test above
            break

    def test_no_refusal_is_a_bare_exception_class_name(self):
        """A message that is just the type name carries no information."""
        bare = []
        for runtime, tool, arguments in TOOLS:
            for key, value in arguments.items():
                if not (isinstance(value, list) and value):
                    continue
                try:
                    _call(runtime, tool, {**arguments, key: []})
                except CLEAN_REFUSAL as refusal:
                    if len(str(refusal)) < 20:
                        bare.append(f"{tool}: {refusal!r}")
                except Exception:
                    pass
                break
        assert not bare, f"refusals too terse to act on: {bare}"


class TestTheCheckItselfCanFail:
    """The two halves of the contract that used to pass on anything."""

    def test_a_non_finite_result_field_is_caught_before_sanitisation(self, monkeypatch):
        """A result model with a plain float holding NaN: the payload says
        null, and this layer still sees the NaN and names its path."""
        from standard_quant_tools.agent.runtimes import resolve

        class _Plain(pydantic.BaseModel):
            value: float

        runtime = resolve("meta")
        _fn, model = runtime.dispatch_table["list_strategies"]
        monkeypatch.setitem(
            runtime.dispatch_table,
            "list_strategies",
            (lambda _input: _Plain(value=float("nan")), model),
        )
        problem = _problem("meta", "list_strategies", {}, "a planted NaN")
        assert problem is not None and "value=nan" in problem

    def test_a_bare_value_error_is_not_a_refusal(self, monkeypatch):
        from standard_quant_tools.agent.runtimes import resolve

        def raise_date_parse_error(_input):
            import pandas as pd

            pd.Timestamp("not-a-date")

        runtime = resolve("meta")
        _fn, model = runtime.dispatch_table["list_strategies"]
        monkeypatch.setitem(
            runtime.dispatch_table, "list_strategies", (raise_date_parse_error, model)
        )
        problem = _problem("meta", "list_strategies", {}, "a planted parse error")
        assert problem is not None and "DateParseError" in problem

    def test_a_result_model_refusing_its_own_tool_is_not_a_refusal(self, monkeypatch):
        """A Pydantic error is a refusal only when the INPUT model raised it.
        One from a result model -- an undeclared key, a wrong type the tool
        produced -- is the tool failing to describe its own answer."""
        from standard_quant_tools.agent.models import BacktestResult
        from standard_quant_tools.agent.runtimes import resolve

        assert "BacktestResult" in RESULT_ONLY_MODELS
        runtime = resolve("meta")
        _fn, model = runtime.dispatch_table["list_strategies"]
        monkeypatch.setitem(
            runtime.dispatch_table,
            "list_strategies",
            (lambda _input: BacktestResult(total_return="not a number"), model),
        )
        problem = _problem("meta", "list_strategies", {}, "a planted result error")
        assert problem is not None and "result model refuses" in problem

    def test_a_clean_result_and_a_typed_refusal_pass(self):
        """The null case: the check does not fire on what is correct."""
        assert _problem("meta", "list_strategies", {}, "a valid call") is None
        assert (
            _problem(
                "meta",
                "describe_tool",
                {"tool_name": "no_such_tool"},
                "an unknown name",
            )
            is None
        )
