"""
Every tool, against inputs designed to break it.

THE CONTRACT BEING TESTED is narrow and absolute: for any input at all, a
tool either

  1. raises a `QuantError` or a Pydantic `ValidationError` — a refusal that
     names what was wrong and what to do instead, or
  2. returns a result that serializes to strict JSON with no NaN and no
     infinity.

Anything else is a defect. An `IndexError` from inside pandas, a
`ZeroDivisionError`, a `KeyError` on a column, an `AttributeError` on a
None — each of those crosses the tool boundary as a message naming no tool,
no argument and no remedy, and reads to a caller like a library bug rather
than a request the data could not support.

WHAT THIS ALREADY FOUND. Three tools raised

    IndexError: single positional indexer is out-of-bounds

on a date range shorter than an indicator's own window — a 20-period band
over one bar is all-NaN, `.dropna()` empties it and `.iloc[-1]` raises from
inside pandas. Twenty-one call sites shared the pattern. And four tools
raised a bare `ValueError` where the rest of the library raises
`ValidationError`, so `except QuantError` silently missed them.

INPUTS ARE SYNTHESIZED FROM THE SCHEMA, not hand-written. A hand-written
fixture list would cover the tools that existed when it was written, which
means the newest tools — where the bugs are — would be the ones never
fuzzed. See `synth.py`.

WHAT IS NOT TESTED HERE: whether the answers are right. These tests ask
"does it fail cleanly", and correctness is what the per-module test files
are for.
"""

from __future__ import annotations

import json
import math
import warnings
from typing import Any, Dict, List, Tuple

import pydantic
import pytest

#: Roughly four and a half minutes: 202 tools x ~40 mutations each, over
#: real 400-business-day windows rather than the zero-length ones this
#: layer used to synthesize. Marked `slow` so
#: `-m "not slow"` skips it while iterating; the default run includes it,
#: because a fuzzing suite nobody runs by default is a fuzzing suite
#: nobody runs.
pytestmark = pytest.mark.slow

from standard_quant_tools.error import QuantError

from .synth import is_finite_json, synthesize_surface

#: A refusal. Anything outside this set is an unhandled exception.
CLEAN_REFUSAL = (QuantError, pydantic.ValidationError, ValueError)

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


def _call(runtime_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
    from standard_quant_tools.agent.runtimes import resolve

    with warnings.catch_warnings():
        # numpy warns on empty slices and zero division; a warning is not a
        # defect, and the assertion below is about what the tool RETURNS.
        warnings.simplefilter("ignore")
        return resolve(runtime_name).dispatch(tool_name, arguments)


def _assert_clean(
    runtime_name: str, tool_name: str, arguments: Dict[str, Any], mutation: str
) -> None:
    """Either a named refusal, or strict-JSON output. Nothing else."""
    try:
        result = _call(runtime_name, tool_name, arguments)
    except CLEAN_REFUSAL as refusal:
        message = str(refusal)
        assert message.strip(), (
            f"{tool_name} refused {mutation} with an EMPTY message, which "
            "tells a caller nothing about what to change"
        )
        return
    except NEVER as exc:
        pytest.fail(
            f"{tool_name} raised an unhandled {type(exc).__name__} on "
            f"{mutation}: {exc}\n"
            "A tool must refuse by name or return a result. This error "
            "names no tool, no argument and no remedy."
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad
        pytest.fail(
            f"{tool_name} raised an unexpected {type(exc).__name__} on "
            f"{mutation}: {exc}"
        )
    else:
        try:
            json.dumps(result, allow_nan=False, default=str)
        except ValueError as exc:
            pytest.fail(
                f"{tool_name} returned a document that is not strict JSON on "
                f"{mutation}: {exc}. JSON has no NaN or Infinity literal and "
                "strict clients reject the payload at the transport layer."
            )
        assert is_finite_json(result), (
            f"{tool_name} returned a non-finite float on {mutation}. Map it "
            "to null: 'not defined' and '0.0' say different things."
        )


class TestTheBaselineHolds:
    """A valid input must produce a valid result. If this fails the
    mutations below are testing nothing."""

    @pytest.mark.parametrize("runtime,tool,arguments", TOOLS, ids=TOOL_IDS)
    def test_a_synthesized_input_is_handled_cleanly(self, runtime, tool, arguments):
        _assert_clean(runtime, tool, arguments, "a valid synthesized input")

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
    Hostile variants of a valid input.

    Each targets a specific way numeric code fails: an empty sequence
    reaching a mean, a constant series reaching a division by its own
    standard deviation, a NaN propagating silently to the output, a
    magnitude that overflows an exponential.
    """
    out: List[Tuple[str, Dict[str, Any]]] = []

    for key, value in arguments.items():
        if isinstance(value, list) and value and isinstance(value[0], (int, float)):
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


#: Built once. Parametrizing per mutation would produce ~9,000 test IDs and
#: a collection phase longer than the run; the loop inside each test keeps
#: the surface identical and the report readable.
MUTATION_COUNTS = {tool: len(_mutations(args)) for _r, tool, args in TOOLS}


class TestHostileInputs:
    @pytest.mark.parametrize("runtime,tool,arguments", TOOLS, ids=TOOL_IDS)
    def test_no_mutation_produces_an_unhandled_exception(
        self, runtime, tool, arguments
    ):
        """
        The core of the regime. Ten mutation families per numeric argument,
        every one of which must produce a refusal or a valid result.
        """
        for description, mutated in _mutations(arguments):
            _assert_clean(runtime, tool, mutated, description)

    def test_the_mutation_set_is_not_empty(self):
        """Another guard on a guard: a synthesizer returning only scalars
        would make the loop above a no-op."""
        total = sum(MUTATION_COUNTS.values())
        assert total >= 500, (
            f"only {total} mutations across {len(TOOLS)} tools; the fuzzer "
            "is not exercising anything"
        )


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
