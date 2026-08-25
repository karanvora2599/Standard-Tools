"""
The runtime boundary.

The behaviour being pinned here is the one that did not exist before:
`get_agent_tools(categories=...)` narrowed the schema list while
`dispatch()` still knew every tool, so an agent scoped to two screener
tools that hallucinated a backtest tool got a SUCCESSFUL result. A wrong
guess was rewarded, which is the worst possible feedback to give a model.

So the tests that matter most are the negative ones: a runtime must refuse
a tool it does not own, and the refusal must say where that tool actually
lives. "Unknown tool" alone cannot be told apart from a hallucinated name,
and a model receiving it guesses again.
"""

import pytest

from standard_quant_tools.agent.runtimes import (
    MODELING_RUNTIME,
    RUNTIME_CATEGORIES,
    all_runtimes,
    combine,
    owner_of,
    resolve,
)
from standard_quant_tools.agent.tools import _TOOL_DISPATCH, TOOL_CATEGORY, dispatch


class TestPartition:
    def test_the_runtimes_cover_every_tool_exactly_once(self):
        """A tool duplicated into a second runtime would dissolve the
        boundary at exactly the points where it matters most."""
        seen: dict = {}
        for name, runtime in all_runtimes().items():
            for tool in runtime.tool_names:
                assert tool not in seen, f"{tool} is in both {seen[tool]} and {name}"
                seen[tool] = name
        assert set(seen) == set(_TOOL_DISPATCH)

    def test_every_category_belongs_to_exactly_one_runtime(self):
        owners: dict = {}
        for name, categories in RUNTIME_CATEGORIES.items():
            for category in categories:
                assert category not in owners
                owners[category] = name
        assert set(owners) == set(TOOL_CATEGORY.values())

    def test_no_runtime_is_too_small_to_be_worth_a_boundary(self):
        """A runtime holding two tools is overhead, not isolation. The
        grouping is deliberately coarse and should stay that way."""
        for name, runtime in all_runtimes().items():
            assert len(runtime) >= 8, f"{name} has only {len(runtime)} tools"

    def test_the_modeling_runtime_is_not_rebuilt_here(self):
        """It predates this module and lives in modeling/agent. Rebuilding
        it would create a second definition of a boundary that already
        exists."""
        assert MODELING_RUNTIME not in all_runtimes()
        assert owner_of("run_model_experiment") == MODELING_RUNTIME


class TestScopeIsEnforced:
    def test_a_tool_from_another_runtime_is_refused(self):
        research = resolve("research")
        with pytest.raises(ValueError) as exc:
            research.dispatch("run_sma_backtest", {})
        assert "belongs to the 'backtest' runtime" in str(exc.value)

    def test_the_refusal_lists_what_is_actually_in_scope(self):
        """So a model can recover on the next turn instead of guessing."""
        meta = resolve("meta")
        with pytest.raises(ValueError) as exc:
            meta.dispatch("run_screener", {})
        message = str(exc.value)
        assert "list_strategies" in message
        assert "widening scope is a decision" in message.lower()

    def test_widening_is_named_as_a_decision_not_offered_as_a_fallback(self):
        research = resolve("research")
        with pytest.raises(ValueError) as exc:
            research.dispatch("get_position_size", {})
        assert "construct the 'portfolio' runtime deliberately" in str(exc.value)

    def test_a_name_no_runtime_has_says_so_plainly(self):
        """Distinguishable from a scoping error: nothing to widen to."""
        with pytest.raises(ValueError) as exc:
            resolve("research").dispatch("run_the_jewels", {})
        assert "does not exist in this library" in str(exc.value)

    def test_a_modeling_tool_is_refused_by_name_from_an_analysis_runtime(self):
        with pytest.raises(ValueError) as exc:
            resolve("backtest").dispatch("run_model_experiment", {})
        assert "'modeling' runtime" in str(exc.value)

    def test_an_unknown_runtime_name_lists_the_real_ones(self):
        with pytest.raises(ValueError) as exc:
            resolve("quant_stuff")
        assert "modeling" in str(exc.value)


class TestSchemas:
    def test_a_runtime_only_advertises_its_own_tools(self):
        for runtime in all_runtimes().values():
            advertised = {t["function"]["name"] for t in runtime.get_tools()}
            assert advertised == set(runtime.tool_names)

    def test_narrowing_by_category_cannot_widen_past_the_runtime(self):
        """Asking the meta runtime for backtest tools yields nothing, not
        backtest tools — a filter argument must never be a way out."""
        meta = resolve("meta")
        assert meta.get_tools(categories=["backtest_execution"]) == []

    def test_narrowing_within_a_runtime_works(self):
        research = resolve("research")
        screener_only = research.get_tools(categories=["screener"])
        assert {t["function"]["name"] for t in screener_only} == {
            "run_screener",
            "get_stock_fundamentals",
        }

    def test_every_runtime_schema_is_the_same_one_the_union_advertises(self):
        from standard_quant_tools.agent.tools import get_agent_tools

        union = {t["function"]["name"]: t for t in get_agent_tools()}
        for runtime in all_runtimes().values():
            for tool in runtime.get_tools():
                assert tool == union[tool["function"]["name"]]


class TestExecution:
    def test_an_in_scope_call_matches_the_unscoped_dispatch(self):
        """The boundary changes what may run, never what running produces."""
        scoped = resolve("meta").dispatch("list_stress_scenarios", {})
        unscoped = dispatch("list_stress_scenarios", {})
        assert scoped == unscoped

    def test_a_runtime_is_callable(self):
        meta = resolve("meta")
        assert meta("list_stress_scenarios", {}) == meta.dispatch(
            "list_stress_scenarios", {}
        )

    def test_calls_are_still_audited(self, tmp_path, monkeypatch):
        """The boundary must not route around the decision log."""
        from standard_quant_tools.audit.paths import _audit_dir, _iter_day_files

        monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
        resolve("meta").dispatch("list_stress_scenarios", {})
        assert _iter_day_files(_audit_dir()), "a scoped call wrote no audit record"

    def test_bad_arguments_still_fail_validation(self):
        with pytest.raises(Exception) as exc:
            resolve("meta").dispatch("list_strategies", {"strategy_type": "nope"})
        assert "sma_crossover" in str(exc.value)


class TestCombine:
    def test_combining_unions_the_tables(self):
        wide = combine(["research", "backtest"])
        assert len(wide) == len(resolve("research")) + len(resolve("backtest"))
        assert "run_screener" in wide and "run_sma_backtest" in wide

    def test_a_combined_runtime_still_refuses_what_neither_owns(self):
        wide = combine(["research", "backtest"])
        with pytest.raises(ValueError) as exc:
            wide.dispatch("get_position_size", {})
        assert "'portfolio' runtime" in str(exc.value)

    def test_combining_names_itself_after_its_parts(self):
        """So a log line or an error says what scope was actually granted."""
        assert combine(["meta", "portfolio"]).name == "meta+portfolio"

    def test_combining_nothing_is_an_error(self):
        with pytest.raises(ValueError):
            combine([])
