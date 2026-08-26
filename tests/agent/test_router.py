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
    #
    # EVERY ROUTABLE CATEGORY APPEARS HERE, and a test below enforces that
    # rather than trusting it. This set had decayed to 7 of 12: `derivatives`
    # and `microstructure` became categories when those runtimes split out,
    # `data` when that runtime landed, and `discovery`/`provenance` were
    # never covered at all. Nothing failed, because an eval that silently
    # stops covering the surface still passes -- it just stops meaning
    # anything, which is the same failure mode as a stale count in prose.
    EVAL_CASES = [
        # backtest_execution -- run one strategy, fixed parameters
        ("Run an SMA crossover backtest on AAPL for 2023.", "backtest_execution"),
        (
            "Buy and hold SPY from 2019 to 2024 and show me the equity curve.",
            "backtest_execution",
        ),
        # backtest_validation -- optimize, validate, diagnose
        (
            "Grid-search RSI parameters and find the best combination.",
            "backtest_validation",
        ),
        ("Walk-forward validate a MACD strategy on MSFT.", "backtest_validation"),
        (
            "How likely is it that this strategy selection was overfit?",
            "backtest_validation",
        ),
        # screener
        (
            "Screen the S&P 500 for stocks with P/E under 20 and RSI under 30.",
            "screener",
        ),
        ("Get me NVDA's fundamentals — P/E, ROE, debt/equity.", "screener"),
        # analysis
        ("What's the alpha, beta, and VaR for TSLA?", "analysis"),
        ("Give me a technical snapshot of AMD — RSI, MACD, ADX.", "analysis"),
        # quant_research
        ("Test if KO and PEP are cointegrated.", "quant_research"),
        (
            "Run a Fama-French factor regression on this fund's returns.",
            "quant_research",
        ),
        # custom_signal
        ("Backtest this signal I already computed: {...}.", "custom_signal"),
        # portfolio_risk
        (
            "Build me an optimal portfolio using Markowitz mean-variance.",
            "portfolio_risk",
        ),
        (
            "How much should I size this position given my account equity?",
            "portfolio_risk",
        ),
        # derivatives -- split out of research; never covered before
        ("What are the vanna and volga on this call?", "derivatives"),
        ("Fit a volatility smile to these strikes and implied vols.", "derivatives"),
        # microstructure -- split out of portfolio; never covered before
        (
            "Estimate the effective spread on this name from daily bars.",
            "microstructure",
        ),
        (
            "What is the Amihud illiquidity for AAPL over the last year?",
            "microstructure",
        ),
        # data -- the newest category
        ("Fetch OHLCV for these fifty tickers and save it for later.", "data"),
        ("Does this provider guarantee point-in-time data?", "data"),
        # discovery
        ("What strategies does this library actually support?", "discovery"),
        ("What arguments does run_walk_forward_backtest take?", "discovery"),
        # provenance
        ("Show me the decision record for request abc123.", "provenance"),
        ("Verify that the audit log has not been tampered with.", "provenance"),
    ]

    @pytest.mark.integration
    def test_routing_accuracy_against_labeled_requests(self):
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")

        import sys

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


class TestTheEvalStillCoversTheSurface:
    """
    An eval that stops covering the surface still passes.

    That is the whole hazard, and it is not hypothetical here: this set was
    written at 7 categories and stayed at 7 while the surface grew to 12.
    `derivatives` and `microstructure` became routable when those runtimes
    split out, `data` when that runtime landed, and `discovery` and
    `provenance` were never covered. Routing to any of the five was
    completely unmeasured, and the accuracy number stayed reassuring
    because it was computed over the categories that were covered.

    These checks cost NO API call, which is the point -- the accuracy eval
    itself is integration-gated and skipped in CI, so a guard that also
    needed a key would decay exactly the same way. They run every time.
    """

    def test_every_routable_category_appears_in_the_eval(self):
        from standard_quant_tools.agent.router import TOOL_CATEGORIES

        labelled = {
            expected for _request, expected in TestRoutingAccuracyEval.EVAL_CASES
        }
        missing = sorted(set(TOOL_CATEGORIES) - labelled)
        assert not missing, (
            f"{len(missing)} routable categor(ies) have no labeled request: "
            f"{missing}. Routing to them is unmeasured, and the accuracy "
            "number is computed over the rest -- which is how this set sat "
            "at 7 of 12 without anything failing."
        )

    def test_the_eval_labels_are_all_real_categories(self):
        """The other direction: a label the router cannot return makes a
        case unfalsifiable, because no answer could ever match it."""
        from standard_quant_tools.agent.router import TOOL_CATEGORIES

        labelled = {
            expected for _request, expected in TestRoutingAccuracyEval.EVAL_CASES
        }
        unknown = sorted(labelled - set(TOOL_CATEGORIES))
        assert not unknown, f"eval expects categories that do not exist: {unknown}"

    def test_the_categories_stay_distinguishable_to_a_classifier(self):
        """
        The router prompt is category DESCRIPTIONS, and a classifier can
        only separate what they separate. Two categories described in
        largely the same words are a coin flip no eval can fix -- and
        unlike a wrong answer, the failure looks like model error rather
        than like a prompt that never carried the distinction.

        Measured at the time of writing: the worst pair overlaps at 0.14,
        so 0.35 leaves real room while still catching a genuinely
        duplicated description.
        """
        import re

        from standard_quant_tools.agent.router import TOOL_CATEGORIES

        stop = {
            "the",
            "a",
            "an",
            "of",
            "and",
            "for",
            "to",
            "in",
            "is",
            "it",
            "that",
            "this",
            "what",
            "with",
            "on",
            "as",
            "its",
            "from",
            "by",
            "each",
            "not",
            "or",
            "one",
            "at",
            "how",
            "you",
            "your",
            "than",
            "are",
            "be",
        }

        def words(text):
            return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in stop}

        names = sorted(TOOL_CATEGORIES)
        overlapping = []
        for i, a in enumerate(names):
            wa = words(TOOL_CATEGORIES[a]["description"])
            for b in names[i + 1 :]:
                wb = words(TOOL_CATEGORIES[b]["description"])
                union = wa | wb
                if not union:
                    continue
                jaccard = len(wa & wb) / len(union)
                if jaccard >= 0.35:
                    overlapping.append(f"{a} | {b} ({jaccard:.2f})")
        assert not overlapping, (
            "category descriptions overlap enough that a classifier cannot "
            f"reliably separate them: {overlapping}. The router prompt is "
            "these descriptions -- if they do not carry the distinction, no "
            "amount of eval tuning will."
        )

    def test_every_category_narrows_the_surface_materially(self):
        """A category holding almost every tool is not routing, it is a
        rounding error with a name."""
        from collections import Counter

        from standard_quant_tools.agent.router import TOOL_CATEGORIES
        from standard_quant_tools.agent.tools import TOOL_CATEGORY

        counts = Counter(TOOL_CATEGORY.values())
        total = sum(counts.values())
        too_broad = {
            category: counts[category]
            for category in TOOL_CATEGORIES
            if counts.get(category, 0) > total * 0.35
        }
        assert not too_broad, (
            f"categories holding over a third of the facade: {too_broad}. "
            "Routing to one of these narrows almost nothing, so the "
            "classification call is being paid for without buying anything."
        )
