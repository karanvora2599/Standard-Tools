"""
`idempotentHint` says what a repeated call does, for every tool.

MCP's hint asks whether calling a tool again with the same arguments has any
further effect. The catalog derived it from one shape only -- a result field
ending in `_uri`, a file saved per call -- and so advertised 32 tools as
idempotent that publish a reference at the caller's `run_id`. Those are the
opposite of idempotent: `handoff.publish` never overwrites by default, so the
second identical call is refused ("a value is already published at
run_id=..."). A client that trusted the hint and retried got a refusal. See
the CHANGELOG entry of 2026-10-22.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.tools import _TOOL_DISPATCH
from standard_quant_tools.error import ValidationError
from standard_quant_tools.mcp.catalog import ALL_CATEGORIES, build_catalog
from standard_quant_tools.mcp.config import ServerConfig
from standard_quant_tools.mcp.server import build_server
from standard_quant_tools.modeling.agent import MODELING_TOOL_DISPATCH
from standard_quant_tools.modeling.agent.feature_tools import FEATURE_TOOL_DISPATCH

#: The tools whose hint changed from true to false: each takes a `run_id`
#: and answers with the reference it published there.
PUBLISH_AT_RUN_ID = frozenset(
    {
        "analyze_basis_history",
        "attach_model_outcomes",
        "backtest_model_signal",
        "build_continuous_futures_series",
        "build_data_bundle",
        "build_model_ensemble",
        "calculate_series_metrics",
        "classify_trade_direction",
        "compute_indicator_panel",
        "construct_weights_from_scores",
        "convert_reference",
        "detect_regimes",
        "fetch_ohlcv",
        "fetch_ohlcv_panel",
        "fetch_order_book",
        "fetch_order_events",
        "fetch_quote_panel",
        "fetch_returns_panel",
        "fetch_tick_tape",
        "get_amihud_illiquidity",
        "get_effective_spread_series",
        "get_quoted_spread_series",
        "get_sharpe_stability",
        "register_external_dataset",
        "run_futures_backtest",
        "run_garch_volatility_forecast",
        "run_kalman_hedge_ratio",
        "run_pair_trade_backtest",
        "run_pca_analysis",
        "run_portfolio_simulation",
        "run_signal_panel_backtest",
        "run_walk_forward_backtest",
    }
)

#: Tools that take a `run_id` to NARROW what they read, not to publish.
RUN_ID_AS_A_FILTER = frozenset({"list_artifacts"})


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


def _dispatch_tables():
    """Every runtime's dispatch table, as the tools are actually registered."""
    for table in (_TOOL_DISPATCH, MODELING_TOOL_DISPATCH, FEATURE_TOOL_DISPATCH):
        for name, entry in table.items():
            fn, model = entry if isinstance(entry, tuple) else (entry, None)
            if model is None:
                model = typing.get_type_hints(fn)["input_data"]
            yield name, model


class TestTheHintFollowsWhatACallWrites:
    def test_the_32_publishers_are_no_longer_advertised_idempotent(self, catalog):
        """The pinned list: every one of them was `true` before."""
        assert len(PUBLISH_AT_RUN_ID) == 32
        wrong = sorted(n for n in PUBLISH_AT_RUN_ID if catalog[n].idempotent)
        assert not wrong, f"advertised idempotent but refuse a repeat: {wrong}"

    def test_the_server_sends_false_for_every_one_of_them(self):
        """The hint a client reads, not only the catalog's property."""
        _server, handlers = build_server(
            ServerConfig(
                categories=ALL_CATEGORIES,
                enable_long_running=True,
                tool_detail="full",
            )
        )
        sent = {t.name: t.annotations.idempotent_hint for t in handlers.tools}
        assert {n: sent[n] for n in PUBLISH_AT_RUN_ID} == {
            n: False for n in PUBLISH_AT_RUN_ID
        }

    def test_every_tool_taking_a_run_id_is_held_to_the_rule(self, catalog):
        """
        Read off the dispatch tables rather than the catalog: every
        registered tool whose input declares `run_id` publishes there --
        and so is not idempotent -- unless it is a known filter. A new tool
        taking a `run_id` whose result carries no reference fails here,
        where an author can decide which it is, instead of being
        advertised as idempotent by default.
        """
        takes_run_id = {
            name for name, model in _dispatch_tables() if "run_id" in model.model_fields
        }
        assert RUN_ID_AS_A_FILTER <= takes_run_id
        misread = sorted(
            name
            for name in takes_run_id
            if catalog[name].idempotent != (name in RUN_ID_AS_A_FILTER)
        )
        assert not misread, (
            f"{misread}: a run_id-taking tool whose hint disagrees with what "
            "it does. A publisher must advertise idempotentHint=false; a "
            "tool that only filters by run_id belongs in RUN_ID_AS_A_FILTER."
        )

    def test_a_reference_the_caller_passes_in_is_not_a_write(self, catalog):
        """The other half of the rule: describing or reading a reference
        the caller already holds writes nothing."""
        for name in (
            "describe_reference",
            "read_reference",
            "describe_data_bundle",
            "validate_external_dataset",
            "list_artifacts",
        ):
            assert catalog[name].idempotent is True, name

    def test_files_saved_per_call_are_still_not_idempotent(self, catalog):
        for name in ("run_backtest_compact", "score_model", "join_point_in_time"):
            assert catalog[name].idempotent is False, name


class TestARepeatIsActuallyRefused:
    """The hint is a claim about behaviour, so the behaviour is checked."""

    @pytest.fixture
    def runs_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        return tmp_path

    def test_the_same_publishing_call_twice_is_refused(self, runs_dir, catalog):
        returns = np.random.default_rng(3).normal(0.0004, 0.01, 300).tolist()
        args = {
            "returns": returns,
            "window": 63,
            "run_id": "repeat_check",
            "name": "rolling",
        }
        research = resolve("research")
        first = research.dispatch("get_sharpe_stability", dict(args))
        assert first["rolling_sharpe_ref"] == (
            "sqt://analytic_series/repeat_check/rolling"
        )
        with pytest.raises(ValidationError, match="already published"):
            research.dispatch("get_sharpe_stability", dict(args))
        assert catalog["get_sharpe_stability"].idempotent is False
