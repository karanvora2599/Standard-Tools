"""
Provider-agnostic tool-category router: narrows the ~45-tool
`get_agent_tools()` registry down to the 1-2 categories a given request
actually needs, via one cheap classification call, before the real
tool-calling completion. Reusable by both `Implementation/*/Agent_*.py`
(single-agent scripts) and `Multi_Agent_Implementation/` (whose 6-7 workers
are exactly this same category taxonomy, just realized as separate agent
sessions instead of a tool-list filter).

This module makes zero network calls itself. Provider-specific glue (which
client, which cheap model) lives in each provider's own `_agent_utils.py`
(`Implementation/Anthropic/_agent_utils.py`, etc.) — this module only
builds the classification prompt and parses its response.

**Design principle: fail open, not closed.** `parse_router_response`
returns every category (i.e. no narrowing) whenever it can't confidently
extract at least one valid category from the model's response — malformed
output, an empty response, or a response that names only unknown category
keys. A router that wrongly excludes a tool the caller actually needed is
worse than today's unfiltered list; narrowing is a confidence optimization,
never a hard gate.
"""

import json
import re
from typing import Dict, Iterable, List

from .tools import TOOL_CATEGORY

# One entry per TOOL_CATEGORY value. Descriptions are distilled from
# Multi_Agent_Implementation/worker_agents.py's WORKER_AGENTS system
# prompts — that prose already exists and is already tuned to disambiguate
# confusable tools, reused here rather than rewritten from scratch.
TOOL_CATEGORIES: Dict[str, Dict[str, str]] = {
    "data": {
        "label": "Data",
        "description": (
            "Get the data and publish it for other tools to read: OHLCV for "
            "one name or a whole universe, return panels, tick tapes and "
            "top-of-book quotes, what the provider guarantees about "
            "adjustment/survivorship/point-in-time, temporal contracts for "
            "frames this library did not fetch, and bundles pairing frames "
            "with their sources' promises. Route here when the request is "
            "about GETTING or CHECKING data rather than analyzing it."
        ),
    },
    "screener": {
        "label": "Screener",
        "description": (
            "Filter a ticker universe by fundamental/technical criteria and "
            "fetch a single ticker's company fundamentals (P/E, P/B, "
            "debt/equity, ROE, market cap)."
        ),
    },
    "analysis": {
        "label": "Technical & Risk Analysis",
        "description": (
            "Single-asset risk profiling (alpha/beta/Sharpe/VaR/CVaR), "
            "technical indicator snapshots, rolling beta drift, realized "
            "and GARCH-forecast volatility, Extreme Value Theory tail "
            "risk, multi-asset portfolio metrics, dataset quality checks, "
            "and European option pricing/Greeks/implied volatility."
        ),
    },
    "quant_research": {
        "label": "Quant Research",
        "description": (
            "Factor regression, cointegration and pairs screening, "
            "Kalman-filter time-varying hedge ratios, PCA on returns, "
            "Hurst regime detection, and correlation/diversification "
            "analytics across a universe. Statistical structure analysis, "
            "not strategy backtesting."
        ),
    },
    "backtest_execution": {
        "label": "Backtest Execution",
        "description": (
            "Run one of the library's built-in strategies (SMA/RSI/MACD/"
            "Bollinger crossover, buy-and-hold) once, compare them, run a "
            "true shared-cash portfolio simulation, or backtest a "
            "synchronized two-leg pair trade. A single run/comparison, not "
            "parameter optimization or out-of-sample validation."
        ),
    },
    "backtest_validation": {
        "label": "Backtest Validation",
        "description": (
            "Optimize a built-in strategy's parameters via grid search, "
            "validate it out-of-sample via walk-forward or regime-adaptive "
            "backtesting, diagnose an existing backtest's drawdowns/trade "
            "stats/exposure, check robustness (parameter sensitivity, "
            "Deflated Sharpe Ratio), or run a Monte Carlo forward "
            "projection of a portfolio's equity paths."
        ),
    },
    "custom_signal": {
        "label": "Custom Signal Backtesting",
        "description": (
            "Backtest a trading signal the caller already computed "
            "themselves — never one of the library's built-in strategies "
            "— for one ticker or a multi-ticker signal panel."
        ),
    },
    "derivatives": {
        "label": "Derivatives",
        "description": (
            "Options: pricing under four models, the second-order greeks "
            "that explain a hedged book's P&L, multi-leg payoffs, smile and "
            "term-structure fitting with arbitrage checks, put-call parity, "
            "expected move, delta-hedge simulation and revaluation grids. "
            "Quotes are passed in rather than fetched."
        ),
    },
    "delta_one": {
        "label": "Delta One",
        "description": (
            "Futures, ETFs, baskets, forwards and total return swaps -- "
            "instruments moving one-for-one with an underlying. Cash-future "
            "basis and which carry component explains it, implied "
            "financing, the futures curve and roll economics, translating a "
            "portfolio beta into a contract count, basket-versus-index "
            "value, and ranking every way of expressing one exposure by "
            "all-in annualized cost. Route here for hedging with futures, "
            "carry, basis or choosing between instruments -- not for "
            "options, whose pricing is derivatives."
        ),
    },
    "microstructure": {
        "label": "Microstructure",
        "description": (
            "Spreads MEASURED from tick data rather than estimated from "
            "bars: quoted and effective spread, the realized/impact "
            "decomposition, signed order flow, trade-size and intraday "
            "volume profiles, and a check of the OHLCV spread proxy against "
            "the real thing. Requires a data provider with a tick feed; "
            "nothing here works from bars."
        ),
    },
    "provenance": {
        "label": "Provenance & Audit",
        "description": (
            "Read and verify the decision log: what a recorded tool call "
            "did and on what data, whether re-running it still reproduces "
            "the same answer (and whether a difference is the data's fault "
            "or the code's), how two runs differ, and whether the log's "
            "tamper-evident hash chain is intact. Read-only — nothing here "
            "alters a record."
        ),
    },
    "discovery": {
        "label": "Discovery",
        "description": (
            "What the library itself accepts and what the data provider can "
            "serve: every built-in strategy's parameter contract, the named "
            "historical stress windows, and whether the active provider has "
            "tick trades, quotes, or async OHLCV. Offline, cheap, and about "
            "the tools rather than about a market."
        ),
    },
    "portfolio_risk": {
        "label": "Portfolio Risk & Sizing",
        "description": (
            "Decompose portfolio risk (marginal contribution, PCA, factor "
            "model), produce optimal portfolio weights (Markowitz "
            "mean-variance, risk parity, Black-Litterman), size a position "
            "(ATR/Kelly), assess capacity/liquidity constraints, and "
            "stress-test a portfolio against named historical crash "
            "windows."
        ),
    },
}

_CATEGORY_KEY_RE = re.compile(r"[a-z_]+")


def build_router_prompt(request: str, categories: Dict[str, Dict[str, str]]) -> str:
    """
    Build the classification prompt: lists every category's label and
    description, asks for the 1-2 most relevant category keys for
    `request`. Pure string template — no network call.
    """
    lines = [
        "You are a routing classifier for a quantitative finance tool "
        "library. Given a user request, identify which 1-2 of the "
        "following tool categories are most relevant to it. Respond with "
        'ONLY a JSON array of category keys (e.g. ["backtest_execution"]) '
        "— no other text.",
        "",
        "Categories:",
    ]
    for key, meta in categories.items():
        lines.append(f'- "{key}" ({meta["label"]}): {meta["description"]}')
    lines.append("")
    lines.append(f"Request: {request}")
    return "\n".join(lines)


def parse_router_response(raw_text: str, valid_keys: Iterable[str]) -> List[str]:
    """
    Parse a classification response into a list of category keys. Tries a
    JSON array first, falls back to scanning for bare category-key-shaped
    tokens in the text. Returns `list(valid_keys)` — i.e. no narrowing —
    whenever it can't confidently extract at least one valid key from
    `raw_text`. See the module docstring's "fail open" principle.
    """
    valid = list(valid_keys)
    valid_set = set(valid)

    if raw_text and raw_text.strip():
        match = re.search(r"\[.*?\]", raw_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    found = [
                        item.strip()
                        for item in parsed
                        if isinstance(item, str) and item.strip() in valid_set
                    ]
                    if found:
                        return _dedupe_preserve_order(found)
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: scan for bare category-key-shaped tokens (comma/
        # whitespace-separated, or embedded in prose).
        candidates = _CATEGORY_KEY_RE.findall(raw_text.lower())
        found = [c for c in candidates if c in valid_set]
        if found:
            return _dedupe_preserve_order(found)

    # Fail open: nothing confidently parsed, don't narrow at all.
    return valid


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def registry_for(categories: Iterable[str]) -> str:
    """
    The `registry=` value that can actually EXECUTE these routed categories.

    THE MISMATCH THIS CLOSES. `route_request` returns CATEGORIES, and
    `Runtime.dispatch` enforces by RUNTIME. Those are two vocabularies for
    one surface, and 48 of the 55 category pairs this router can return
    span more than one runtime -- so a caller who routed to categories and
    then named a runtime by hand was usually describing an intersection
    nobody had checked.

    Measured before this existed: `registry="research"` with the routed
    pair `["portfolio_risk", "derivatives"]` advertises ZERO tools, and
    nothing says so. An agent handed no tools reads as a broken install
    rather than as two decisions disagreeing, which is the same failure the
    MCP server already refuses at `--runtime`/`--categories`.

    Deriving the scope from the routing decision removes the disagreement
    instead of detecting it:

        cats = route_request(request, api_key=key)
        run_agent(..., categories=cats, registry=registry_for(cats))

    The widening stays VISIBLE -- the result is a "+"-joined spec like
    `"research+portfolio"`, so a reader sees that two runtimes were opened
    and why. What it stops being is silent.

    Returns `"analysis"` when nothing maps, which is the union view. That
    is the same fail-open rule `parse_router_response` uses: a classifier
    that returned something unusable should widen the surface, never empty
    it.
    """
    from .runtimes import runtimes_for_categories

    owners = runtimes_for_categories(categories)
    return "+".join(owners) if owners else "analysis"


def _assert_categories_cover_every_tool() -> None:
    """Sanity check run at import time: every TOOL_CATEGORY value must have
    a corresponding TOOL_CATEGORIES entry, or the router prompt would be
    silently missing a category some tool actually belongs to."""
    used_categories = set(TOOL_CATEGORY.values())
    known_categories = set(TOOL_CATEGORIES)
    missing = used_categories - known_categories
    if missing:
        raise RuntimeError(
            f"TOOL_CATEGORIES is missing metadata for categories in use: "
            f"{sorted(missing)}"
        )


_assert_categories_cover_every_tool()
