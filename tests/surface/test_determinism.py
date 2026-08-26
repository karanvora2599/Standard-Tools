"""
Same input, same answer. Different seed, different answer.

TWO PROPERTIES, and the second is the one people forget. A tool that
accepts a `seed` and ignores it passes every reproducibility test ever
written — its output is perfectly stable — while quietly making the seed a
lie. A caller who changes it to check robustness gets the identical number
back and concludes the result is robust.

So both directions are tested:

  - **Reproducible.** The same arguments produce byte-identical output.
    Without this an audit record cannot be replayed and a backtest cannot
    be defended.
  - **Seed-sensitive.** A different seed produces a different answer, for
    every tool that advertises one.

THE THIRD PROPERTY IS PURITY. Calling a tool must not mutate the arguments
it was given. A tool that sorts its input in place changes the caller's
data, and the second call in a loop then operates on something the caller
never passed.

Nothing here needs a network: the tools with a `seed` are the simulation
and bootstrap tools, which all take inline data.
"""

from __future__ import annotations

import copy
import json
import warnings
from typing import Any, Dict, List, Tuple

import pytest

#: Roughly seven minutes: every offline tool is called four times (twice
#: for reproducibility, twice more for purity) and several are
#: simulations. Marked `slow` for the reason recorded in
#: test_adversarial_inputs.py.
pytestmark = pytest.mark.slow

from .synth import Unsynthesizable, build_arguments, names_a_path


def _seeded_tools() -> List[Tuple[str, str, Dict[str, Any]]]:
    """Every synthesizable tool that advertises a `seed`."""
    from standard_quant_tools.agent.runtimes import all_runtimes

    out = []
    for runtime_name, runtime in all_runtimes().items():
        for tool_name, _description, model in runtime.tool_defs:
            if "seed" not in model.model_fields:
                continue
            try:
                arguments = build_arguments(model)
            except (Unsynthesizable, Exception):
                continue
            out.append((runtime_name, tool_name, arguments))
    return out


#: Fields naming an external data source. A tool that takes one does not
#: have its output determined by its arguments -- the market is the other
#: input, and it moves between two calls. Reproducibility for those is a
#: different property, and the audit trail's `verify_replay` is what tests
#: it: it pins the DATA as well as the arguments.
FETCHES_DATA = {"symbol", "symbols", "ticker", "tickers", "benchmark", "universe"}


def _offline_tools() -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Tools whose arguments fully determine their output.

    Excluding the fetching tools is not a convenience. Two calls a second
    apart can legitimately see different data — a provider updating a bar,
    a partially warm cache — and a test that called that non-determinism
    would be reporting the market rather than the code.

    A tool naming a PATH is excluded for the same reason: the file is the
    other input, whether the tool reads it or writes it.
    `verify_audit_integrity` reads a public key and `export_audit_bundle`
    writes a bundle, and neither is a function of its arguments alone.

    `export_audit_bundle` is worth naming, because it is what this rule
    was written for. Called twice with the same arguments it correctly
    returns DIFFERENT notes — the second call overwrote the file the
    first one wrote, and saying so is the honest answer. That went
    unnoticed for as long as it did because the synthesized path used to
    be the bare string `a`, so the target always already existed in the
    repo root and both calls reported the overwrite. Making the
    synthesizer write somewhere the suite owns is what surfaced it.

    Reproducibility for a writer is a real property. It is a different
    one — about the bytes on disk rather than the returned dict — and
    `tests/audit/` is where it is checked.
    """
    from standard_quant_tools.agent.runtimes import all_runtimes

    out = []
    for runtime_name, runtime in all_runtimes().items():
        for tool_name, _description, model in runtime.tool_defs:
            if FETCHES_DATA & set(model.model_fields):
                continue
            if names_a_path(model):
                continue
            try:
                arguments = build_arguments(model)
            except (Unsynthesizable, Exception):
                continue
            out.append((runtime_name, tool_name, arguments))
    return out


def _without_timestamps(value: Any) -> Any:
    """
    Strip fields that record WHEN rather than WHAT.

    `get_data_quality_report` stamps its metadata with the fetch time and
    `export_audit_bundle` stamps the bundle. Both are correct — a
    provenance record without a time is not a provenance record — and
    neither is a source of non-determinism in the computation.
    """
    stamped = ("timestamp", "generated", "fetched", "as_of_utc", "created", "_at")
    if isinstance(value, dict):
        return {
            k: _without_timestamps(v)
            for k, v in value.items()
            if not any(marker in k.lower() for marker in stamped)
        }
    if isinstance(value, list):
        return [_without_timestamps(v) for v in value]
    return value


SEEDED = _seeded_tools()
SEEDED_IDS = [f"{r}:{t}" for r, t, _ in SEEDED]
OFFLINE = _offline_tools()
OFFLINE_IDS = [f"{r}:{t}" for r, t, _ in OFFLINE]


def _call(runtime_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
    from standard_quant_tools.agent.runtimes import resolve

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return resolve(runtime_name).dispatch(tool_name, arguments)


def _serialize(value: Any) -> str:
    """A stable string for comparison, with timestamps removed."""
    return json.dumps(_without_timestamps(value), sort_keys=True, default=str)


def _quiet(runtime_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Call, returning None when the tool refuses — a refusal is not the
    subject of these tests."""
    try:
        return _call(runtime_name, tool_name, arguments)
    except Exception:  # noqa: BLE001 - covered by test_adversarial_inputs
        return None


class TestSeedsAreHonoured:
    def test_there_are_seeded_tools_to_check(self):
        """A guard on the guard: an empty parametrization passes silently
        and looks exactly like success."""
        assert len(SEEDED) >= 5, (
            f"only {len(SEEDED)} seeded tools found; either the synthesizer "
            "broke or `seed` was renamed"
        )

    @pytest.mark.parametrize("runtime,tool,arguments", SEEDED, ids=SEEDED_IDS)
    def test_the_same_seed_reproduces_the_same_answer(self, runtime, tool, arguments):
        """
        Without this an audit record cannot be replayed, and a simulated
        result cannot be defended to anyone who reruns it.
        """
        first = _quiet(runtime, tool, {**arguments, "seed": 7})
        second = _quiet(runtime, tool, {**arguments, "seed": 7})
        if first is None:
            pytest.skip(f"{tool} refused the synthesized input")
        assert _serialize(first) == _serialize(
            second
        ), f"{tool} is not reproducible at a fixed seed"

    @pytest.mark.parametrize("runtime,tool,arguments", SEEDED, ids=SEEDED_IDS)
    def test_a_different_seed_changes_the_answer(self, runtime, tool, arguments):
        """
        THE ONE PEOPLE FORGET. A tool that accepts a seed and ignores it
        passes every reproducibility test ever written, because its output
        is perfectly stable — and a caller who varies the seed to check
        robustness gets the same number back and concludes it is robust.
        """
        first = _quiet(runtime, tool, {**arguments, "seed": 1})
        second = _quiet(runtime, tool, {**arguments, "seed": 999_983})
        if first is None or second is None:
            pytest.skip(f"{tool} refused the synthesized input")
        assert _serialize(first) != _serialize(second), (
            f"{tool} advertises a `seed` and returns an identical result for "
            "1 and 999983. Either the seed is not reaching the generator, or "
            "the tool is deterministic and should not advertise one."
        )


class TestOfflineToolsAreReproducibleAndPure:
    @pytest.mark.parametrize("runtime,tool,arguments", OFFLINE, ids=OFFLINE_IDS)
    def test_calling_twice_gives_the_same_answer(self, runtime, tool, arguments):
        """
        Every tool, seeded or not. An unseeded tool that varies between
        calls has an unlogged source of randomness, which makes every
        result it produced unreproducible.
        """
        first = _quiet(runtime, tool, arguments)
        second = _quiet(runtime, tool, arguments)
        if first is None:
            pytest.skip(f"{tool} refused the synthesized input")
        assert _serialize(first) == _serialize(second), (
            f"{tool} returned different results for identical arguments. If "
            "it is stochastic it needs a `seed`; if it is not, something is "
            "reading a clock or an unseeded generator."
        )

    @pytest.mark.parametrize("runtime,tool,arguments", OFFLINE, ids=OFFLINE_IDS)
    def test_the_arguments_are_not_mutated(self, runtime, tool, arguments):
        """
        A tool that sorts or clips its input IN PLACE changes the caller's
        own data. The second iteration of a loop then runs on something the
        caller never passed, and the bug surfaces far from its cause.
        """
        before = copy.deepcopy(arguments)
        _quiet(runtime, tool, arguments)
        assert arguments == before, (
            f"{tool} mutated the arguments it was given. The caller's data "
            "changed underneath it."
        )
