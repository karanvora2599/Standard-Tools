"""
Tests for the provider-agnostic tool-category router
(src/standard_quant_tools/agent/router.py). `parse_router_response` and
`build_router_prompt` are pure functions -- no network calls -- so most of
this file runs fully offline. The routing-*accuracy* eval at the bottom is
the exception: it costs real API calls and is gated behind
`@pytest.mark.integration`, matching this repo's existing
`-m "not integration"` default-CI convention.
"""

import pytest

from standard_quant_tools.agent.router import (
    TOOL_CATEGORIES,
    build_router_prompt,
    parse_router_response,
)
from standard_quant_tools.agent.tools import TOOL_CATEGORY

from .. import REPO_ROOT

VALID_KEYS = list(TOOL_CATEGORIES)


class TestToolCategories:
    def test_covers_every_category_used_in_tool_category(self):
        """The same coverage guarantee as
        test_agent_tools.py::TestToolCategoryCoverage, checked from the
        router's side: every category a tool is actually assigned to must
        have router metadata, or the classification prompt would be
        silently missing an option a tool needs."""
        assert set(TOOL_CATEGORY.values()) <= set(TOOL_CATEGORIES)

    def test_every_category_has_label_and_description(self):
        for key, meta in TOOL_CATEGORIES.items():
            assert meta.get("label"), f"{key} missing a label"
            assert meta.get("description"), f"{key} missing a description"


class TestBuildRouterPrompt:
    def test_includes_every_categorys_label(self):
        prompt = build_router_prompt("backtest AAPL", TOOL_CATEGORIES)
        for meta in TOOL_CATEGORIES.values():
            assert meta["label"] in prompt

    def test_includes_the_request_text(self):
        prompt = build_router_prompt("a very specific request marker", TOOL_CATEGORIES)
        assert "a very specific request marker" in prompt


class TestParseRouterResponse:
    def test_valid_json_array_single_category(self):
        assert parse_router_response('["backtest_execution"]', VALID_KEYS) == [
            "backtest_execution"
        ]

    def test_valid_json_array_multi_category(self):
        result = parse_router_response('["backtest_execution", "analysis"]', VALID_KEYS)
        assert result == ["backtest_execution", "analysis"]

    def test_json_array_with_surrounding_prose(self):
        result = parse_router_response(
            'Sure, here it is: ["quant_research"] — hope that helps!', VALID_KEYS
        )
        assert result == ["quant_research"]

    def test_bare_comma_separated_fallback(self):
        result = parse_router_response("screener, custom_signal", VALID_KEYS)
        assert result == ["screener", "custom_signal"]

    def test_deduplicates_while_preserving_order(self):
        result = parse_router_response(
            '["analysis", "analysis", "screener"]', VALID_KEYS
        )
        assert result == ["analysis", "screener"]

    def test_malformed_text_fails_open_to_every_category(self):
        result = parse_router_response("I'm sorry, I don't understand.", VALID_KEYS)
        assert set(result) == set(VALID_KEYS)

    def test_empty_string_fails_open_to_every_category(self):
        assert set(parse_router_response("", VALID_KEYS)) == set(VALID_KEYS)

    def test_all_unknown_keys_fails_open_to_every_category(self):
        result = parse_router_response(
            '["not_a_real_category", "also_fake"]', VALID_KEYS
        )
        assert set(result) == set(VALID_KEYS)

    def test_malformed_json_array_falls_back_to_bare_token_scan(self):
        """A truncated/invalid JSON array shouldn't take down parsing
        entirely -- the bare-token fallback should still find valid keys in
        the surrounding text."""
        result = parse_router_response(
            'category: ["backtest_execution", INVALID JSON HERE analysis',
            VALID_KEYS,
        )
        assert "backtest_execution" in result or "analysis" in result

    def test_valid_categories_mixed_with_unknown_ones_keeps_only_valid(self):
        result = parse_router_response(
            '["screener", "not_real", "analysis"]', VALID_KEYS
        )
        assert result == ["screener", "analysis"]


class TestRoutingAccuracyEval:
    """First actual measurement of routing correctness in this codebase --
    the pre-existing multi-agent coverage test only checks tool-set
    coverage/disjointness, never whether a classifier picks the *right*
    category for a real request. Gated behind @pytest.mark.integration
    since it costs real API calls; run manually with
    `pytest -m integration tests/test_router.py`."""

    # (request, expected category) -- representative of the kind of
    # requests Implementation/Anthropic/Agent_*.py scripts actually send.
    EVAL_CASES = [
        ("Run an SMA crossover backtest on AAPL for 2023.", "backtest_execution"),
        (
            "Grid-search RSI parameters and find the best combination.",
            "backtest_validation",
        ),
        (
            "Screen the S&P 500 for stocks with P/E under 20 and RSI under 30.",
            "screener",
        ),
        ("What's the alpha, beta, and VaR for TSLA?", "analysis"),
        ("Test if KO and PEP are cointegrated.", "quant_research"),
        ("Backtest this signal I already computed: {...}.", "custom_signal"),
        (
            "Build me an optimal portfolio using Markowitz mean-variance.",
            "portfolio_risk",
        ),
        ("Walk-forward validate a MACD strategy on MSFT.", "backtest_validation"),
        (
            "How much should I size this position given my account equity?",
            "portfolio_risk",
        ),
        ("Get me NVDA's fundamentals — P/E, ROE, debt/equity.", "screener"),
    ]

    @pytest.mark.integration
    def test_routing_accuracy_against_labeled_requests(self):
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        import sys
        from pathlib import Path

        sys.path.insert(0, str(REPO_ROOT / "Implementation" / "Anthropic"))
        from _agent_utils import route_request  # type: ignore[import-not-found]

        correct = 0
        results = []
        for request, expected in self.EVAL_CASES:
            routed = route_request(request, api_key=api_key)
            hit = expected in routed
            correct += hit
            results.append((request, expected, routed, hit))

        accuracy = correct / len(self.EVAL_CASES)
        report = "\n".join(
            f"  {'OK ' if hit else 'MISS'} expected={expected!r:24s} routed={routed}"
            for _, expected, routed, hit in results
        )
        print(
            f"\nRouting accuracy: {accuracy:.0%} ({correct}/{len(self.EVAL_CASES)})\n{report}"
        )
        assert accuracy >= 0.7, (
            f"Routing accuracy {accuracy:.0%} below the 70% baseline "
            f"threshold:\n{report}"
        )


class TestRoutingAndEnforcementSpeakTheSameLanguage:
    """
    The router picks CATEGORIES. `Runtime.dispatch` enforces by RUNTIME.

    Those are two vocabularies for one surface, and they were never
    reconciled. Measured: of the 55 category PAIRS this router can return,
    48 span more than one runtime -- so a caller who routed to categories
    and then named a runtime by hand was usually describing an
    intersection nobody had checked.

    The sharp end of that: `registry="research"` with the routed pair
    `["portfolio_risk", "derivatives"]` advertised ZERO tools, silently. An
    agent handed no tools does not sit quietly -- it reads as a broken
    install, and then it invents tool names, which is the exact failure
    routing exists to prevent.

    Two mechanisms close it, and both are pinned here: `registry_for`
    DERIVES the scope from the routing decision, and `_registry_tools`
    refuses to hand back an empty list when a fixed runtime and a routed
    category set disagree.
    """

    def test_a_routed_pair_spanning_runtimes_resolves_to_both(self):
        from standard_quant_tools.agent.router import registry_for

        assert registry_for(["quant_research", "portfolio_risk"]) == (
            "research+portfolio"
        )

    def test_the_disjoint_pair_that_used_to_yield_nothing_now_resolves(self):
        """The regression, named. This pair against `research` gave 0."""
        from standard_quant_tools.agent.router import registry_for
        from standard_quant_tools.agent.runtimes import combine

        spec = registry_for(["portfolio_risk", "derivatives"])
        assert spec == "portfolio+derivatives"
        runtime = combine(spec.split("+"))
        served = runtime.get_tools(categories=["portfolio_risk", "derivatives"])
        assert served, "the derived scope must not be empty"

    def test_a_single_category_does_not_widen_beyond_its_owner(self):
        """Deriving the scope must not become a way of quietly serving
        more than was routed to."""
        from standard_quant_tools.agent.router import registry_for

        assert registry_for(["screener"]) == "research"
        assert registry_for(["derivatives"]) == "derivatives"

    def test_an_unroutable_category_fails_open_rather_than_empty(self):
        """Same rule as `parse_router_response`: a classifier that returns
        something unusable should WIDEN the surface, never empty it."""
        from standard_quant_tools.agent.router import registry_for

        assert registry_for(["not_a_category"]) == "analysis"
        assert registry_for([]) == "analysis"

    def test_every_router_category_is_owned_by_exactly_one_runtime(self):
        """If a category the router can return has no runtime, routing to
        it produces an agent that cannot execute anything it was routed
        to."""
        from standard_quant_tools.agent.router import TOOL_CATEGORIES
        from standard_quant_tools.agent.runtimes import CATEGORY_RUNTIME

        orphans = sorted(set(TOOL_CATEGORIES) - set(CATEGORY_RUNTIME))
        assert not orphans, (
            f"the router can return {orphans}, which no runtime owns -- a "
            "request routed there cannot be executed by anything."
        )
