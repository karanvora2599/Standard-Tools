"""
A thinned tool is advertised by the first sentence of its description, whole.

The thin entry is all an agent reads before deciding whether a tool is worth
a `describe_tool` round trip, and it was cut at the first ". " or at 180
characters, whichever came first. The first rule split "vs." --
`get_volatility_estimators` was advertised as "... Yang-Zhang estimators
vs." -- and the second stopped 43 longer first sentences mid-clause,
`run_portfolio_optimization` and `estimate_trade_cost` among them, the two
tools `auto` thins in `portfolio`. Forty-six tools in all. The boundary now
skips known abbreviations and ends at `?` and `!` too, and every first
sentence was rewritten to fit, so the cut is never needed. See the CHANGELOG
entry of 2026-10-22.
"""

from __future__ import annotations

import re

import pytest

from standard_quant_tools.mcp.catalog import (
    THIN_DESCRIPTION_LIMIT,
    build_catalog,
    first_sentence,
    thin_description,
)
from standard_quant_tools.mcp.config import ServerConfig
from standard_quant_tools.mcp.server import build_server


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


class TestEveryToolIsAdvertisedByAWholeSentence:
    def test_every_thinned_description_is_a_complete_first_sentence(self, catalog):
        """
        For all 241 tools: the thin text is the description's first
        sentence exactly -- not a prefix cut short -- it fits the limit, and
        it ends at a sentence terminator rather than on an abbreviation.
        """
        wrong = {}
        for name, entry in sorted(catalog.items()):
            thin = thin_description(entry.description)
            sentence = first_sentence(entry.description)
            if thin != sentence:
                wrong[name] = f"cut: {thin!r}"
            elif len(thin) > THIN_DESCRIPTION_LIMIT:
                wrong[name] = f"{len(thin)} characters"
            elif not re.search(r"[.?!][)\"'`]?$", thin):
                wrong[name] = f"no terminator: {thin!r}"
            elif re.search(r"\b(?:vs|e\.g|i\.e|cf)\.$", thin):
                wrong[name] = f"ends on an abbreviation: {thin!r}"
        assert not wrong, (
            f"{len(wrong)} tool(s) would be advertised by a broken sentence "
            f"when thinned: {wrong}. Shorten the first sentence of the "
            f"description to {THIN_DESCRIPTION_LIMIT} characters or fewer; "
            "the rest belongs in the sentences after it."
        )

    def test_the_two_tools_portfolio_thins_read_whole(self):
        """The case that showed the cost: `auto` thins exactly these two in
        `portfolio`, and both were advertised ending mid-list."""
        _s, handlers = build_server(
            ServerConfig(categories=("portfolio_risk",), runtimes=("portfolio",))
        )
        thin = {
            t.name: t.description for t in handlers.tools if t.name in handlers.thinned
        }
        assert set(thin) >= {"run_portfolio_optimization", "estimate_trade_cost"}
        assert thin["run_portfolio_optimization"].startswith(
            "Produce portfolio weights via Markowitz mean-variance "
            "(max_sharpe/min_volatility/target_return/target_volatility), "
            "risk parity, or Black-Litterman. [args:"
        )
        assert thin["estimate_trade_cost"].startswith(
            "Itemized cost of one hypothetical trade under a composed cost "
            "model: commission, spread, square-root market impact, short "
            "borrow and margin interest. [args:"
        )
        assert all("…" not in text for text in thin.values())

    def test_vs_no_longer_ends_a_sentence(self, catalog):
        assert thin_description(catalog["get_volatility_estimators"].description) == (
            "Realized volatility via Parkinson, Garman-Klass, and Yang-Zhang "
            "estimators vs. plain close-to-close."
        )


class TestTheBoundary:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Compare A vs. B on cost. Then more.", "Compare A vs. B on cost."),
            ("Uses e.g. a ratio. Then more.", "Uses e.g. a ratio."),
            ("Is it rich? Then more.", "Is it rich?"),
            ("Weights at 0.5 each. Then more.", "Weights at 0.5 each."),
            ("No terminator at all", "No terminator at all"),
            ("Ends the text.", "Ends the text."),
        ],
    )
    def test_the_first_sentence(self, text, expected):
        assert first_sentence(text) == expected

    def test_an_over_long_sentence_is_cut_at_a_word_never_inside_one(self):
        """The safety net, which the catalog test keeps unused."""
        words = " ".join(["capacity"] * 40) + ". Tail."
        out = thin_description(words)
        assert len(out) <= THIN_DESCRIPTION_LIMIT
        assert out.endswith("capacity…")
        assert "capacit…" not in out
