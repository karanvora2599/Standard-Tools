"""
Coverage check for Multi_Agent_Implementation/worker_agents.py.

Pure data validation — no Anthropic API key or network access required.
Fails fast if a new agent tool is ever added to the library without being
assigned to exactly one worker agent, which would otherwise go unnoticed
until someone actually tried to use it through the multi-agent example.
"""

import sys
from pathlib import Path

import pytest

from standard_quant_tools.agent import get_agent_tools

MULTI_AGENT_DIR = Path(__file__).resolve().parent.parent / "Multi_Agent_Implementation"


@pytest.fixture(scope="module")
def worker_agents():
    sys.path.insert(0, str(MULTI_AGENT_DIR))
    from worker_agents import WORKER_AGENTS

    return WORKER_AGENTS


class TestWorkerToolCoverage:
    def test_every_worker_has_tools_and_a_system_prompt(self, worker_agents):
        for key, worker in worker_agents.items():
            assert worker["tools"], f"{key} has no tools assigned"
            assert worker["system_prompt"].strip(), f"{key} has an empty system prompt"
            assert worker["label"], f"{key} has no label"
            assert worker["description"], f"{key} has no description"

    def test_union_of_worker_tools_equals_all_library_tools(self, worker_agents):
        library_tools = {t["function"]["name"] for t in get_agent_tools()}
        assigned = set()
        for worker in worker_agents.values():
            assigned.update(worker["tools"])
        missing = library_tools - assigned
        extra = assigned - library_tools
        assert not missing, f"Tools not assigned to any worker: {sorted(missing)}"
        assert not extra, f"Workers reference tools that don't exist: {sorted(extra)}"

    def test_no_tool_is_assigned_to_more_than_one_worker(self, worker_agents):
        seen: dict = {}
        duplicates = []
        for key, worker in worker_agents.items():
            for tool in worker["tools"]:
                if tool in seen:
                    duplicates.append((tool, seen[tool], key))
                seen[tool] = key
        assert (
            not duplicates
        ), f"Tools assigned to more than one worker (confusion risk): {duplicates}"

    def test_confusable_backtest_tools_are_split_across_workers(self, worker_agents):
        # The whole point of the split: built-in strategy tools and
        # bring-your-own-signal tools must never be loaded into the same
        # worker, or the disambiguation problem they were split to fix
        # would simply reappear.
        backtest_tools = set(worker_agents["backtest_execution"]["tools"]) | set(
            worker_agents["backtest_validation"]["tools"]
        )
        custom_signal_tools = set(worker_agents["custom_signal"]["tools"])
        assert "run_sma_backtest" in backtest_tools
        assert "run_custom_signal_backtest" in custom_signal_tools
        assert backtest_tools.isdisjoint(custom_signal_tools)

    def test_backtest_execution_and_validation_tools_are_split_across_workers(
        self, worker_agents
    ):
        # The same confusion-pair guarantee, one level narrower: "run this
        # exact strategy" (backtest_execution) and "find/validate the best
        # parameters" (backtest_validation) must never share a worker either,
        # or a worker could pick run_sma_backtest when the request actually
        # needed run_backtest_optimization (or vice versa).
        execution_tools = set(worker_agents["backtest_execution"]["tools"])
        validation_tools = set(worker_agents["backtest_validation"]["tools"])
        assert "run_sma_backtest" in execution_tools
        assert "run_backtest_optimization" in validation_tools
        assert execution_tools.isdisjoint(validation_tools)

    def test_there_are_seven_workers(self, worker_agents):
        """Regression guard for the backtest_execution/backtest_validation
        split -- catches an accidental re-merge or an accidental further
        split just as easily as a magic number would, but derived from the
        actual registry rather than repeating a number that could itself
        drift."""
        assert set(worker_agents) == {
            "screener",
            "analysis",
            "quant_research",
            "backtest_execution",
            "backtest_validation",
            "custom_signal",
            "portfolio_risk",
        }
