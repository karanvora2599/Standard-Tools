"""
A worker's prompt is checked against the tools it can actually run.

WHY THIS EXISTS. `test_multi_agent_tool_coverage.py` already pins the layer
below this one: every registered tool belongs to exactly one worker, and no
registry is left without one. That is necessary and it is not sufficient,
because it says nothing about the PROSE each worker is given.

The prose is what a model actually reads. It establishes a prior over what
kind of work the agent believes it does, and it goes stale the moment a
runtime changes underneath it -- silently, because no assertion anywhere
compared the two. What that cost, measured on `main` before this file
existed:

- The microstructure worker opened with "you are working from TICK data --
  not from bars" and "every one of your tools needs a provider with a tick
  feed". Eight bar-based estimators had landed in that runtime by then.
  The agent would decline "estimate Kyle lambda from this close and volume
  series" while holding `estimate_kyle_lambda`.
- The model-research worker taught eight feature tools in detail. All eight
  had moved to the `feature_lab` runtime and none was loaded for it.
- The analysis worker taught `get_option_pricing` and
  `get_implied_volatility`. Both had moved to `derivatives`.

Eighteen references in total named a tool the worker did not hold, and
sixteen of those named a tool in ANOTHER RUNTIME. That second number is the
serious one. `run_worker_agent` passes `registry=worker["runtime"]`, so
`Runtime.dispatch` refuses a cross-runtime name outright -- the prompt was
walking the model into a wall it had been told to walk into, which is worse
than saying nothing.

THE FIX THIS FILE GUARDS is a split between two kinds of text. The
hand-written half teaches JUDGEMENT: when to reach for which tool, what the
numbers mean, what not to claim. The generated half -- `_scope_block()` --
is the INVENTORY, derived from the runtime's own `tool_defs`, so it cannot
disagree with the dispatch table. These tests pin both halves.

WHAT IS DELIBERATELY ALLOWED. Naming another agent's tool is often the most
useful thing a prompt can do: "that is the Derivatives Agent's job" routes
a request correctly, where silence leaves the model to guess. So a
cross-scope mention passes when the surrounding text marks it as a handoff,
and fails when it reads as an instruction. That distinction is the whole
point of the file, and it is why this cannot be a plain substring check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from standard_quant_tools.agent.runtimes import all_runtimes

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "Multi_Agent_Implementation"))

from worker_agents import WORKER_AGENTS  # noqa: E402


@pytest.fixture(scope="module")
def registry():
    return {t for rt in all_runtimes().values() for t in rt.dispatch_table}


@pytest.fixture(scope="module")
def owner():
    return {t: n for n, rt in all_runtimes().items() for t in rt.dispatch_table}


#: Phrases that mark a sentence as routing work AWAY rather than
#: instructing the agent to act. Kept deliberately short and specific: a
#: loose list would let a genuine instruction pass as a handoff, which is
#: the failure this file exists to catch.
_HANDOFF = (
    "agent",
    "belongs to",
    "belong to",
    "hand it",
    "hand off",
    "handoff",
    "hand the request",
    "not loaded for you",
    "not yours",
    "are not yours",
    "refused by name",
    "defer to",
    "owns that",
    "own that",
    "is not a tool of yours",
    "out of scope",
    "another runtime",
    "that agent",
)

_TOKEN = re.compile(r"\b([a-z][a-z0-9_]{4,})\b")

#: Where the generated inventory starts. Everything from here down is
#: derived, and it is deliberately EXCLUDED from the handoff judgement
#: below.
#:
#: This is not tidiness; it was a hole in this file for one commit. The
#: block's preamble ends "...belongs to another agent", and its bullet
#: lines carry no sentence-ending punctuation, so the whole block parses as
#: one enormous sentence containing a handoff marker. Any instruction
#: sitting near it inherited that marker and passed as "routed".
#: Mutation-tested: appending "Always call get_option_pricing to value the
#: hedge" to a prompt was NOT caught until this split existed -- exactly the
#: vacuous pass this suite is supposed to be immune to.
_SCOPE_MARKER = "TOOLS IN YOUR CURRENT SCOPE"


def _prose(prompt: str) -> str:
    """The hand-written half: everything before the generated block."""
    head, _, _ = prompt.partition(_SCOPE_MARKER)
    return head


def _mentions(prompt: str, registry: set[str]) -> set[str]:
    return {t for t in _TOKEN.findall(prompt) if t in registry}


def _windows_naming(prompt: str, tool: str) -> list[str]:
    """
    Every sentence-ish window that names `tool`.

    A window is the sentence containing the mention plus the one before
    it, because the routing instruction is regularly the preceding
    sentence: "Options are not yours. Use the Derivatives Agent for
    get_option_pricing."
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n\n", prompt)
    out = []
    for i, sentence in enumerate(sentences):
        if re.search(rf"\b{re.escape(tool)}\b", sentence):
            start = max(0, i - 1)
            out.append(" ".join(sentences[start : i + 1]).lower())
    return out


def _is_handoff(prompt: str, tool: str) -> bool:
    windows = _windows_naming(prompt, tool)
    if not windows:
        return False
    return all(any(marker in w for marker in _HANDOFF) for w in windows)


WORKERS = sorted(WORKER_AGENTS)


class TestNoWorkerIsToldToCallWhatItCannotRun:
    @pytest.mark.parametrize("worker", WORKERS)
    def test_out_of_scope_mentions_are_marked_as_handoffs(
        self, worker, registry, owner
    ):
        """
        The invariant that would have caught all three P0s.

        A worker may NAME another runtime's tool, but only to route the
        request. Anything else is an instruction the dispatch table will
        refuse.
        """
        spec = WORKER_AGENTS[worker]
        prose = _prose(spec["system_prompt"])
        mine = set(spec["tools"])
        offenders = [
            f"{tool} (owned by {owner[tool]!r})"
            for tool in sorted(_mentions(prose, registry) - mine)
            if not _is_handoff(prose, tool)
        ]
        assert not offenders, (
            f"the {worker!r} worker's prompt names tools it cannot run, "
            f"without marking them as a handoff: {offenders}. Either drop "
            "the reference or say which agent owns it -- a cross-runtime "
            "name is refused by Runtime.dispatch, so instructing the model "
            "to use one guarantees a failed call."
        )


class TestEveryToolAWorkerHoldsIsVisibleToIt:
    @pytest.mark.parametrize("worker", WORKERS)
    def test_every_tool_in_scope_is_named_in_the_prompt(self, worker, registry):
        """
        A tool a worker holds but is never told about is a tool it will not
        reach for. This was 19 of 26 for `quant_research`, 13 of 20 for
        `backtest_validation` and 9 of 12 for `microstructure` -- all of
        them registered, schema-loaded, and semantically invisible.

        `_scope_block()` makes this automatic. The test pins the outcome
        rather than the mechanism, so removing the generated block fails
        here even if the prose is rewritten by hand.
        """
        spec = WORKER_AGENTS[worker]
        missing = sorted(
            set(spec["tools"]) - _mentions(spec["system_prompt"], registry)
        )
        assert not missing, (
            f"the {worker!r} worker holds {len(missing)} tool(s) its prompt "
            f"never names: {missing}. The generated scope block should cover "
            "every tool in `tools`; if it has been removed or the tool list "
            "is built some other way, the agent cannot reach these."
        )


class TestTheGeneratedScopeBlockIsPresentAndHonest:
    @pytest.mark.parametrize("worker", WORKERS)
    def test_the_prompt_carries_a_generated_scope_listing(self, worker):
        assert (
            "TOOLS IN YOUR CURRENT SCOPE" in WORKER_AGENTS[worker]["system_prompt"]
        ), (
            f"{worker!r} has no generated scope block. That block is what "
            "keeps the inventory from drifting away from the dispatch table."
        )

    @pytest.mark.parametrize("worker", WORKERS)
    def test_the_scope_listing_names_neither_more_nor_less_than_the_scope(self, worker):
        """The block is generated, so it should be exactly the tool list --
        a listing that has drifted is worse than none, because it reads as
        authoritative."""
        spec = WORKER_AGENTS[worker]
        prompt = spec["system_prompt"]
        block = prompt[prompt.index("TOOLS IN YOUR CURRENT SCOPE") :]
        listed = set(re.findall(r"^- ([a-z][a-z0-9_]+):?", block, re.MULTILINE))
        assert listed == set(spec["tools"]), (
            f"{worker!r}'s scope listing disagrees with its tool list. "
            f"only in listing: {sorted(listed - set(spec['tools']))}; "
            f"only in tools: {sorted(set(spec['tools']) - listed)}"
        )


class TestTheRuntimeAWorkerDispatchesThroughOwnsItsTools:
    @pytest.mark.parametrize("worker", WORKERS)
    def test_every_tool_is_owned_by_the_workers_own_runtime(self, worker, owner):
        """
        The layer beneath the prompt. A worker whose tool list contains a
        name its runtime does not own would fail at the first call, and the
        error would read as a bad model choice rather than as wiring.
        """
        spec = WORKER_AGENTS[worker]
        runtime = spec["runtime"]
        wrong = [
            f"{t} -> {owner.get(t)}" for t in spec["tools"] if owner.get(t) != runtime
        ]
        assert not wrong, (
            f"{worker!r} dispatches through {runtime!r} but holds tools that "
            f"runtime does not own: {wrong}"
        )
