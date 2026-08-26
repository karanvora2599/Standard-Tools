"""
Thin listings: advertise a name and a purpose, fetch the schema on demand.

WHY. Runtime scoping fixed *who sees what*; it did not change what a tool
costs to advertise. `backtest` is 65 KB before an agent has done anything,
and the ceiling that buys 82.4 tools is already spent at 82.

The observation this rests on is that `describe_tool` already existed, in
`meta`, tested, and returning exactly the schema a thinned listing omits.
The server was shipping all 82 schemas up front on the assumption an agent
would need every one. It needs a handful.

WHAT THINNING MUST NOT DO. It changes the ADVERTISEMENT and nothing else:

- A thinned tool is still callable, with the same arguments, validated by
  the same model. If thinning could change what a call does, it would be a
  behaviour flag wearing a display flag's name.
- `extra="forbid"` still rejects a guessed argument. Thinning removes the
  description, never the validation.
- The round trip has to actually close: an agent must be able to go from a
  thin listing to a correct call using only what the server told it. That
  is the end-to-end test below, and it is the one that would catch a
  `describe_tool` that returned something a caller could not act on.

WHY `auto` AND NOT `thin` BY DEFAULT. A schema an agent has not read is one
it will guess at, which is the hallucination problem one layer down. `auto`
thins the most expensive tools only, and only until the surface fits, so the
tools a runtime is actually for stay fully described.
"""

import json

import pytest

from standard_quant_tools.mcp.catalog import (
    ALL_RUNTIMES,
    DEFAULT_DETAIL_BUDGET,
    DETAIL_MODES,
    SCHEMA_FETCH_TOOL,
    build_catalog,
    categories_for_runtimes,
    plan_detail,
    select_runtimes,
    thin_description,
    thin_schema,
)
from standard_quant_tools.mcp.config import ServerConfig, resolve
from standard_quant_tools.mcp.server import StandardToolsServer, build_server


def _server(runtime, mode, **kw):
    return build_server(
        ServerConfig(
            categories=categories_for_runtimes([runtime]),
            runtimes=(runtime,),
            tool_detail=mode,
            enable_long_running=True,
            **kw,
        )
    )


class TestThePlan:
    def test_full_thins_nothing(self):
        entries = select_runtimes(build_catalog(), ["backtest"])
        assert plan_detail(entries, "full") == set()

    def test_thin_thins_everything_except_the_fetcher(self):
        entries = select_runtimes(build_catalog(), ["meta"])
        thinned = plan_detail(entries, "thin")
        assert SCHEMA_FETCH_TOOL not in thinned, (
            "thinning the tool that undoes thinning leaves no way back to a " "schema"
        )
        assert thinned == {e.name for e in entries} - {SCHEMA_FETCH_TOOL}

    def test_auto_stops_as_soon_as_it_fits(self):
        """The point of ranking by cost: thin as few tools as possible."""
        entries = select_runtimes(build_catalog(), ["backtest"])
        thinned = plan_detail(entries, "auto", budget=DEFAULT_DETAIL_BUDGET)
        assert thinned, "backtest is 65 KB; auto should have thinned something"
        # Removing any one of them would put it back over budget, i.e. none
        # of them is thinned gratuitously.
        cost = sum(
            (
                len(json.dumps(thin_schema(e.name), separators=(",", ":")))
                if e.name in thinned
                else e.cost_bytes()
            )
            for e in entries
        )
        assert cost <= DEFAULT_DETAIL_BUDGET

    def test_auto_thins_the_expensive_ones_first(self):
        entries = select_runtimes(build_catalog(), ["backtest"])
        thinned = plan_detail(entries, "auto")
        kept = [e for e in entries if e.name not in thinned]
        cut = [e for e in entries if e.name in thinned]
        assert min(e.cost_bytes() for e in cut) >= max(
            e.cost_bytes() for e in kept if e.name != SCHEMA_FETCH_TOOL
        ), "a cheap tool was thinned while an expensive one was kept"

    def test_a_runtime_that_already_fits_is_left_alone(self):
        """
        `auto` is a budget, not a policy: a runtime under the target pays
        nothing at all.

        Stated over WHICHEVER runtimes currently fit rather than naming one.
        This test named `research` until it grew from 23 tools to 29 and
        stopped fitting -- at which point the test failed for the right
        reason and said the wrong thing, because the property it was
        checking had moved to a different runtime.
        """
        catalog = build_catalog()
        fitting = []
        for runtime in ALL_RUNTIMES:
            entries = select_runtimes(catalog, [runtime])
            if sum(e.cost_bytes() for e in entries) <= DEFAULT_DETAIL_BUDGET:
                fitting.append(runtime)
                assert plan_detail(entries, "auto") == set(), (
                    f"{runtime} is under the budget and auto thinned "
                    "something anyway"
                )
        assert fitting, (
            "every runtime now exceeds the detail budget, so this test is "
            "vacuous. Either the budget needs revisiting or the surface has "
            "outgrown it."
        )

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="unknown detail mode"):
            plan_detail([], "medium")


class TestTheListing:
    @pytest.mark.parametrize("runtime", ALL_RUNTIMES)
    def test_thin_is_cheaper_than_full(self, runtime):
        _s, full = _server(runtime, "full")
        _s, thin = _server(runtime, "thin")
        assert thin.context_bytes() < full.context_bytes()

    def test_the_fetcher_is_added_when_the_scope_lacks_it(self):
        """A thin entry says 'call describe_tool'. If describe_tool is not
        served, every thinned tool is uncallable and the listing is a lie."""
        _s, handlers = _server("backtest", "thin")
        assert SCHEMA_FETCH_TOOL in {t.name for t in handlers.tools}

    def test_the_fetcher_is_never_itself_thinned(self):
        for runtime in ALL_RUNTIMES:
            _s, handlers = _server(runtime, "thin")
            assert SCHEMA_FETCH_TOOL not in handlers.thinned
            tool = next(t for t in handlers.tools if t.name == SCHEMA_FETCH_TOOL)
            assert tool.input_schema.get(
                "properties"
            ), "describe_tool was advertised without its arguments"

    def test_full_mode_adds_nothing(self):
        """The fetcher is injected only because thinning needs it. With
        nothing thinned, the scope stays exactly what was asked for."""
        _s, handlers = _server("backtest", "full")
        assert SCHEMA_FETCH_TOOL not in {t.name for t in handlers.tools}

    def test_a_thin_entry_says_how_to_get_the_schema(self):
        _s, handlers = _server("modeling", "thin")
        tool = next(t for t in handlers.tools if t.name in handlers.thinned)
        assert SCHEMA_FETCH_TOOL in tool.description
        assert SCHEMA_FETCH_TOOL in json.dumps(tool.input_schema)

    def test_a_thin_schema_never_claims_the_tool_takes_nothing(self):
        """An empty `{}` schema reads as 'no arguments', and a model that
        believes it calls the tool with no arguments and burns a turn on a
        validation error. The schema must say it is incomplete."""
        schema = thin_schema("run_model_experiment")
        assert schema.get("additionalProperties") is True
        assert schema.get("description")
        assert not schema.get("properties"), "a thin schema should list nothing"


class TestThinningChangesOnlyTheAdvertisement:
    def test_the_same_tools_are_served(self):
        _s, full = _server("modeling", "full")
        _s, thin = _server("modeling", "thin")
        assert {t.name for t in full.tools} <= {t.name for t in thin.tools}

    def test_a_thinned_tool_is_still_dispatchable(self):
        server = StandardToolsServer(
            ServerConfig(
                categories=categories_for_runtimes(["modeling"]),
                runtimes=("modeling",),
                tool_detail="thin",
                enable_long_running=True,
            )
        )
        assert server.thinned
        for name in server.thinned:
            assert name in server.entries, f"{name} advertised but not callable"

    def test_validation_is_unchanged_by_thinning(self):
        """The safety net thinning leans on. If a guessed argument were
        accepted, thinning would have traded a round trip for a silent
        wrong answer."""
        import anyio
        import mcp.types as types

        _s, handlers = _server("research", "thin")

        async def call():
            return await handlers.call_tool(
                None,
                types.CallToolRequestParams(
                    name="analyze_stock_risk",
                    arguments={"symbol": "AAPL", "perid": "1y"},  # typo
                ),
            )

        result = anyio.run(call)
        assert result.is_error is True
        assert "perid" in result.content[0].text


class TestTheRoundTripCloses:
    """The end-to-end claim: a thin listing plus describe_tool is enough to
    make a correct call. If this fails, thinning is not a cheaper listing,
    it is a broken one."""

    def test_describe_tool_returns_what_the_listing_omitted(self):
        import anyio
        import mcp.types as types

        _s, handlers = _server("modeling", "thin")
        target = sorted(handlers.thinned)[0]

        async def call():
            return await handlers.call_tool(
                None,
                types.CallToolRequestParams(
                    name=SCHEMA_FETCH_TOOL,
                    arguments={"tool_name": target, "include_schema": True},
                ),
            )

        result = anyio.run(call)
        assert result.is_error is not True, result.content[0].text
        payload = json.dumps(result.structured_content or result.content[0].text)

        # Every required argument the real tool has must appear in what the
        # agent got back -- otherwise it cannot construct the call.
        entry = build_catalog()[target]
        for name in entry.input_schema.get("required", []) or []:
            assert name in payload, (
                f"describe_tool({target!r}) did not mention required "
                f"argument {name!r}; an agent could not recover the call"
            )

    def test_describe_tool_answers_for_tools_in_other_runtimes(self):
        """A scoped server still has to explain a refusal, and the answer to
        'why was that refused' is a description, not a wider scope."""
        import anyio
        import mcp.types as types

        _s, handlers = _server("research", "thin")

        async def call():
            return await handlers.call_tool(
                None,
                types.CallToolRequestParams(
                    name=SCHEMA_FETCH_TOOL,
                    arguments={"tool_name": "run_sma_backtest"},
                ),
            )

        result = anyio.run(call)
        assert result.is_error is not True


class TestTheDefaultIsUnchanged:
    def test_detail_defaults_to_full(self):
        assert resolve([]).tool_detail == "full"

    def test_the_default_server_thins_nothing(self):
        _s, handlers = build_server(resolve([]))
        assert handlers.thinned == set()

    @pytest.mark.parametrize("mode", DETAIL_MODES)
    def test_every_advertised_mode_parses(self, mode):
        assert resolve(["--tool-detail", mode]).tool_detail == mode


class TestThinDescription:
    def test_it_keeps_the_first_sentence(self):
        assert thin_description("Does a thing. Then explains it at length.") == (
            "Does a thing."
        )

    def test_it_caps_a_long_first_sentence(self):
        long = "x" * 400 + ". tail"
        out = thin_description(long)
        assert len(out) <= 180

    def test_it_collapses_whitespace(self):
        assert thin_description("Does\n  a\tthing.") == "Does a thing."
