"""
Coverage check for Multi_Agent_Implementation/worker_agents.py.

Pure data validation — no Anthropic API key or network access required.
Fails fast if a new agent tool is ever added to the library without being
assigned to exactly one worker agent, which would otherwise go unnoticed
until someone actually tried to use it through the multi-agent example.

The check is PER REGISTRY. There are two — the 46-tool analysis surface and
the separate 8-tool modeling runtime — and the library never merges them.
Checking their union against a merged tool list would pass while a worker
quietly listed a modeling tool in an analysis worker, which fails at the
first tool call because the two dispatch functions do not know each other's
names. So each registry is required to be covered exactly once by the
workers that declare it, and the two sets are required to be disjoint.
"""

import sys
from pathlib import Path

import pytest

from standard_quant_tools.agent import get_agent_tools
from standard_quant_tools.modeling.agent import get_modeling_tools
from standard_quant_tools.modeling.agent.feature_tools import get_feature_tools

from .. import REPO_ROOT

MULTI_AGENT_DIR = REPO_ROOT / "Multi_Agent_Implementation"


#: registry name -> the tool names that registry actually exposes.
REGISTRY_TOOLS = {
    "analysis": lambda: {t["function"]["name"] for t in get_agent_tools()},
    "modeling": lambda: {t["function"]["name"] for t in get_modeling_tools()},
    "feature_lab": lambda: {t["function"]["name"] for t in get_feature_tools()},
}


@pytest.fixture(scope="module")
def worker_agents():
    sys.path.insert(0, str(MULTI_AGENT_DIR))
    from worker_agents import WORKER_AGENTS

    return WORKER_AGENTS


def _by_registry(worker_agents):
    """Tool names each worker registry claims, keyed by registry name."""
    claimed: dict = {}
    for worker in worker_agents.values():
        claimed.setdefault(worker["registry"], set()).update(worker["tools"])
    return claimed


class TestWorkerToolCoverage:
    def test_every_worker_has_tools_a_prompt_and_a_known_registry(self, worker_agents):
        for key, worker in worker_agents.items():
            assert worker["tools"], f"{key} has no tools assigned"
            assert worker["system_prompt"].strip(), f"{key} has an empty system prompt"
            assert worker["label"], f"{key} has no label"
            assert worker["description"], f"{key} has no description"
            assert (
                worker["registry"] in REGISTRY_TOOLS
            ), f"{key} declares unknown registry {worker['registry']!r}"

    @pytest.mark.parametrize("registry", sorted(REGISTRY_TOOLS))
    def test_workers_cover_each_registry_exactly(self, worker_agents, registry):
        available = REGISTRY_TOOLS[registry]()
        claimed = _by_registry(worker_agents).get(registry, set())
        missing = available - claimed
        extra = claimed - available
        assert not missing, f"{registry}: tools in no worker: {sorted(missing)}"
        assert not extra, (
            f"{registry}: workers claim tool(s) this registry does not have: "
            f"{sorted(extra)} — most likely listed under the wrong registry"
        )

    def test_the_two_registries_share_no_tool_name(self, worker_agents):
        # If a name ever existed in both registries, "which dispatch function
        # runs it" would depend on which worker happened to be asked, and the
        # per-registry coverage check above would still pass.
        analysis = REGISTRY_TOOLS["analysis"]()
        modeling = REGISTRY_TOOLS["modeling"]()
        assert analysis.isdisjoint(
            modeling
        ), f"name collision across registries: {sorted(analysis & modeling)}"

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

    def test_the_modeling_pipeline_is_split_at_the_dataset(self, worker_agents):
        # model_research ends by producing a dataset_id; model_builder needs
        # one and has no tool that can make it. If build_model_dataset ever
        # drifted into the builder, the handoff this split exists to force
        # would silently disappear -- the builder could do everything itself
        # and the research step would become optional.
        research = set(worker_agents["model_research"]["tools"])
        builder = set(worker_agents["model_builder"]["tools"])
        assert "build_model_dataset" in research
        assert "analyze_features" in research
        assert "run_model_experiment" in builder
        assert "evaluate_model_portfolio" in builder
        assert research.isdisjoint(builder)

    def test_the_worker_set_is_exactly_the_declared_one(self, worker_agents):
        """Regression guard for the backtest_execution/backtest_validation
        split and the model_research/model_builder split -- catches an
        accidental re-merge or an accidental further split just as easily as
        a magic number would, but derived from the actual registry rather
        than repeating a number that could itself drift."""
        assert set(worker_agents) == {
            "screener",
            "analysis",
            "quant_research",
            "backtest_execution",
            "backtest_validation",
            "custom_signal",
            "derivatives",
            "portfolio_risk",
            "discovery",
            "provenance",
            "microstructure",
            "model_research",
            "model_builder",
            # Split out of model_research when the nine feature tools moved
            # to their own runtime. Feature work runs repeatedly before any
            # model exists, which is a different job from assembling one
            # dataset and handing off its id.
            "feature_lab",
        }

    def test_every_analysis_category_has_exactly_one_worker(self, worker_agents):
        """The analysis workers ARE the TOOL_CATEGORY taxonomy, one worker
        per category. A new category with no worker means its tools are
        unreachable through the multi-agent example; a worker with no
        category means it lists tools by hand."""
        from standard_quant_tools.agent.tools import TOOL_CATEGORY

        categories = set(TOOL_CATEGORY.values())
        workers = {
            key
            for key, worker in worker_agents.items()
            if worker["registry"] == "analysis"
        }
        assert workers == categories


def test_every_registry_has_at_least_one_worker(worker_agents):
    """
    A registry whose workers all disappeared is reachable by nobody.

    This is not hypothetical: splitting the nine feature tools out of
    `modeling` into `feature_lab` left every one of them claimed by no
    worker, and every existing test passed -- because the coverage check
    iterates the registries that HAVE workers, so a registry with none is
    invisible to it. The gap was in the test, not only in the data.
    """
    from standard_quant_tools.agent.tools import _TOOL_DISPATCH
    from standard_quant_tools.modeling.agent import MODELING_TOOL_DISPATCH
    from standard_quant_tools.modeling.agent.feature_tools import (
        FEATURE_TOOL_DISPATCH,
    )

    registries = {
        "analysis": _TOOL_DISPATCH,
        "modeling": MODELING_TOOL_DISPATCH,
        "feature_lab": FEATURE_TOOL_DISPATCH,
    }
    covered = {w["registry"] for w in worker_agents.values()}
    missing = sorted(set(registries) - covered)
    assert not missing, (
        f"registries with no worker at all: {missing}. Every tool in them is "
        "unreachable from the multi-agent implementation, and the per-registry "
        "coverage check cannot see it."
    )
