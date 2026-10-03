"""
How the native kernels decide to go parallel, held at the source.

Since the CHANGELOG entry of 2026-10-02 a region goes parallel on its
estimated serial time -- units of work times the kernel's measured cost per
unit -- and every call that does is counted while it runs; the pooled rank's
parallel sort divides the threads by that count, so pooled correlations run
from several Python threads at once share the cores. Both properties live
at the call sites, where a new kernel could quietly skip them: a site that
counts units without a cost falls back to the old 50,000-unit rule, and a
region that decides inline in its `if` clause is never counted, so the
sorts running beside it would not know it is there. Neither is visible in a
result -- that is the point of both -- so they are checked here, in the
source the build compiles.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "src" / "standard_quant_tools" / "_cpp"
SOURCES = sorted((CPP / "src").glob("*.cpp"))
POLICY = CPP / "include" / "sqt" / "omp_policy.hpp"


CALLS = {
    "worth_parallel": r"\bworth_parallel\s*\(",
    # the guard is constructed, not called: `parallel_call omp_call(...)`
    "parallel_call": r"\bparallel_call\s+\w+\s*\(",
}


def _calls(text: str, name: str) -> list[str]:
    """The argument lists of every use of `name`, parentheses balanced."""
    found = []
    for match in re.finditer(CALLS[name], text):
        depth, start = 1, match.end()
        i = start
        while depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        found.append(text[start : i - 1])
    return found


def _top_level_commas(args: str) -> int:
    depth = commas = 0
    for ch in args:
        depth += {"(": 1, "{": 1, ")": -1, "}": -1}.get(ch, 0)
        commas += ch == "," and depth == 0
    return commas


def _pragmas(text: str) -> list[str]:
    """Every `#pragma omp parallel` directive, continuation lines joined."""
    joined = re.sub(r"\\\s*\n", " ", text)
    return [
        line
        for line in joined.splitlines()
        if re.match(r"\s*#\s*pragma\s+omp\s+parallel\b", line)
    ]


class TestEveryCallSiteStatesItsCost:
    def test_there_are_call_sites_to_check(self):
        counted = sum(
            len(_calls(p.read_text(encoding="utf-8"), "parallel_call")) for p in SOURCES
        )
        assert counted >= 15, counted

    def test_no_call_site_counts_units_without_a_cost(self):
        """worth_parallel and parallel_call take (tasks, units, cost); the
        two-argument worth_parallel is the old unit rule, kept only so an
        unconverted call still compiles."""
        bare = []
        for path in SOURCES:
            text = path.read_text(encoding="utf-8")
            for name in ("worth_parallel", "parallel_call"):
                for args in _calls(text, name):
                    commas = _top_level_commas(args)
                    # parallel_call(bool) is a decision already made with a cost
                    if name == "worth_parallel" and commas != 2:
                        bare.append(f"{path.name}: {name}({args.strip()})")
                    if name == "parallel_call" and commas not in (0, 2):
                        bare.append(f"{path.name}: {name}({args.strip()})")
        assert not bare, bare

    def test_no_region_decides_inline_where_it_would_go_uncounted(self):
        """A region's `if` clause reads a counted call's decision; calling
        worth_parallel in the clause itself would leave the call out of the
        count that concurrent calls share."""
        inline = []
        for path in SOURCES:
            for pragma in _pragmas(path.read_text(encoding="utf-8")):
                if "worth_parallel" in pragma:
                    inline.append(f"{path.name}: {pragma.strip()[:120]}")
        assert not inline, inline

    def test_the_policy_names_its_threshold_and_both_controls(self):
        header = POLICY.read_text(encoding="utf-8")
        assert "kMinSerialNs = 150'000.0" in header
        assert "SQT_NUM_THREADS" in header and "SQT_OMP_MIN_WORK" in header
        assert "class parallel_call" in header

    def test_only_the_pooled_sort_divides_the_threads(self):
        """Shared the same way, the per-date and per-pair loops measured
        slower: a share fixed when a region starts leaves cores idle as the
        calls around it finish. Only the pooled sort, whose run count
        follows its thread count, asks for a share."""
        users = {
            path.name: path.read_text(encoding="utf-8").count("shared_threads()")
            for path in SOURCES
        }
        assert {name: n for name, n in users.items() if n} == {"panel_stats.cpp": 1}
        text = (CPP / "src" / "panel_stats.cpp").read_text(encoding="utf-8")
        body = text[text.index("void average_ranks(") :]
        assert "shared_threads()" in body[: body.index("\n}\n")]


class TestThePermutationNullIsGuided:
    def test_the_permutation_loop_rebalances(self):
        """Every draw is the same work, but the threads are not the same
        speed; static's equal split waited on the slowest one."""
        text = (CPP / "src" / "panel_stats.cpp").read_text(encoding="utf-8")
        body = text[text.index("bool permutation_null_ic(") :]
        body = body[: body.index("\n}\n")]
        (pragma,) = _pragmas(body)
        assert "schedule(guided)" in pragma, pragma
