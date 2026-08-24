"""
Drift-proofing for the MCP surface.

The point of every test here is that it fails when something in the library
changes underneath the server, rather than when the server itself is edited.
A tool added to either registry without a category, a result model that
stops being JSON-safe, a stray print() in a library module, a schema that
grows a `$ref` -- each of those breaks a real client in a way that looks
like a protocol bug, and each is caught here without a network call.

No API key and no market data. Everything is schema- and wiring-level.
"""

from __future__ import annotations

import io
import json
import re
from contextlib import redirect_stdout

import pytest

from standard_quant_tools.agent.tools import _TOOL_DISPATCH
from standard_quant_tools.mcp import prompts as _prompts
from standard_quant_tools.mcp import resources as _resources
from standard_quant_tools.mcp.catalog import (
    ALL_CATEGORIES,
    DEFAULT_CATEGORIES,
    build_catalog,
    category_costs,
    dispatch_for,
    select,
)
from standard_quant_tools.mcp.config import ServerConfig
from standard_quant_tools.mcp.schemas import contains_ref, dereference, schema_bytes
from standard_quant_tools.mcp.server import (
    LONG_RUNNING,
    StandardToolsServer,
    build_server,
)
from standard_quant_tools.modeling.agent import MODELING_TOOL_DISPATCH

MCP_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")

#: Ceiling for the full 54-tool surface, in bytes. Measured at ~122 KB with
#: output schemas omitted. This is a budget, not a fact: a tool that doubles
#: it should fail here and be argued for, because every byte is spent from
#: every client's context at connect.
FULL_SURFACE_CEILING = 150_000


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


class TestCoverage:
    """Every dispatchable tool is exposed exactly once, both directions."""

    def test_catalog_covers_both_registries_exactly(self, catalog):
        library = set(_TOOL_DISPATCH) | set(MODELING_TOOL_DISPATCH)
        missing = library - set(catalog)
        extra = set(catalog) - library
        assert not missing, f"tools the MCP catalog does not expose: {sorted(missing)}"
        assert not extra, f"catalog names no dispatcher has: {sorted(extra)}"

    def test_every_tool_has_a_known_category(self, catalog):
        bad = {
            n: e.category
            for n, e in catalog.items()
            if e.category not in ALL_CATEGORIES
        }
        assert not bad, f"tools with an unknown category: {bad}"

    def test_categories_partition_the_catalog(self, catalog):
        seen = set()
        for category in ALL_CATEGORIES:
            names = {e.name for e in select(catalog, [category])}
            assert not (seen & names), f"{category} overlaps an earlier category"
            seen |= names
        assert seen == set(catalog)

    def test_each_tool_routes_to_its_own_registrys_dispatcher(self, catalog):
        # The two dispatchers have identical signatures, so a mispaired tool
        # would fail only at call time with an "unknown tool" error naming
        # the model's choice rather than this wiring.
        for entry in catalog.values():
            table = (
                _TOOL_DISPATCH
                if entry.registry == "analysis"
                else MODELING_TOOL_DISPATCH
            )
            assert entry.name in table, f"{entry.name} routed to the wrong registry"
            assert callable(dispatch_for(entry))

    def test_the_two_registries_share_no_tool_name(self):
        overlap = set(_TOOL_DISPATCH) & set(MODELING_TOOL_DISPATCH)
        assert (
            not overlap
        ), f"a name in both registries is unroutable: {sorted(overlap)}"


class TestSchemas:
    def test_every_exposed_name_is_mcp_legal(self, catalog):
        bad = [n for n in catalog if not MCP_NAME_RE.match(n)]
        assert not bad, f"names MCP will reject: {bad}"

    def test_no_input_schema_contains_a_ref(self, catalog):
        # Seven tools carry $defs upstream. They are the most complex tools
        # in the library, so they are the worst ones to hand a client that
        # resolves references poorly.
        bad = [n for n, e in catalog.items() if contains_ref(e.input_schema)]
        assert not bad, f"input schemas still containing $ref: {bad}"

    def test_no_output_schema_contains_a_ref(self, catalog):
        bad = [
            n
            for n, e in catalog.items()
            if e.output_schema is not None and contains_ref(e.output_schema)
        ]
        assert not bad, f"output schemas still containing $ref: {bad}"

    def test_every_tool_has_an_output_schema(self, catalog):
        # All 54 have typed Pydantic returns today. If one loses its
        # annotation the server silently stops being able to declare
        # structured output for it.
        missing = [n for n, e in catalog.items() if e.output_schema is None]
        assert not missing, f"tools with no usable return annotation: {missing}"

    def test_schemas_are_json_serializable(self, catalog):
        for name, entry in catalog.items():
            json.dumps(entry.input_schema, allow_nan=False)
            json.dumps(entry.output_schema, allow_nan=False)

    def test_dereference_is_idempotent(self, catalog):
        for entry in catalog.values():
            once = entry.input_schema
            assert dereference(once) == once


class TestBudget:
    """The context cost is the constraint the whole design exists to manage."""

    def test_full_surface_stays_under_the_ceiling(self):
        # Measured on what actually goes over the wire, not on what the
        # catalog happens to know: output schemas are always computed and
        # only sent when asked for.
        _server, handlers = build_server(
            ServerConfig(categories=ALL_CATEGORIES, enable_long_running=True)
        )
        total = handlers.context_bytes()
        assert total < FULL_SURFACE_CEILING, (
            f"the full tool surface is {total:,} bytes (~{total // 4:,} tokens), "
            f"over the {FULL_SURFACE_CEILING:,} ceiling. Every byte is spent "
            "from every client's context at connect -- either shrink a schema "
            "or argue the ceiling up deliberately."
        )

    def test_declaring_output_schemas_is_the_expensive_option(self):
        _s1, off = build_server(ServerConfig(categories=ALL_CATEGORIES))
        _s2, on = build_server(
            ServerConfig(categories=ALL_CATEGORIES, include_output_schemas=True)
        )
        # ~77% more, measured. This is why it is a flag and not a default.
        assert on.context_bytes() > off.context_bytes() * 1.5

    def test_the_default_selection_is_the_cheap_one(self, catalog):
        default = sum(e.cost_bytes() for e in select(catalog, DEFAULT_CATEGORIES))
        full = sum(e.cost_bytes() for e in catalog.values())
        assert default < full / 3, (
            f"the default categories cost {default:,} of {full:,} bytes. The "
            "default exists to be cheap; if it has grown past a third of the "
            "whole surface it is no longer doing its job."
        )

    def test_every_category_is_reported(self, catalog):
        costs = category_costs(catalog)
        assert set(costs) == set(ALL_CATEGORIES)
        assert all(count > 0 and size > 0 for count, size in costs.values())


class TestAnnotations:
    def test_every_tool_declares_all_four_hints(self):
        _server, handlers = build_server(ServerConfig(categories=ALL_CATEGORIES))
        for tool in handlers.tools:
            a = tool.annotations
            assert a is not None, f"{tool.name} has no annotations"
            for hint in (
                "read_only_hint",
                "destructive_hint",
                "idempotent_hint",
                "open_world_hint",
            ):
                assert getattr(a, hint) is not None, f"{tool.name} missing {hint}"

    def test_nothing_is_write_capable(self):
        # This library does not place orders or move money. One tool
        # breaking that forces every client to treat the whole server as
        # write-capable.
        _server, handlers = build_server(ServerConfig(categories=ALL_CATEGORIES))
        for tool in handlers.tools:
            assert tool.annotations.read_only_hint is True, tool.name
            assert tool.annotations.destructive_hint is False, tool.name

    def test_artifact_writers_are_not_marked_idempotent(self, catalog):
        # A tool that persists a new Parquet artifact per call genuinely has
        # an additional effect each time.
        writers = {n for n, e in catalog.items() if e.persists_artifact}
        assert writers, "expected at least one artifact-persisting tool"
        for name in writers:
            assert catalog[name].idempotent is False, name

    def test_offline_tools_are_not_marked_open_world(self, catalog):
        # Black-Scholes pricing is arithmetic; the modeling registry reads
        # its own store. Neither reaches the network.
        for name in ("get_option_pricing", "get_implied_volatility", "inspect_model"):
            assert catalog[name].reads_market_data is False, name

    def test_market_data_tools_are_marked_open_world(self, catalog):
        for name in ("analyze_stock_risk", "run_sma_backtest", "build_model_dataset"):
            assert catalog[name].reads_market_data is True, name


class TestSelection:
    def test_default_categories_are_all_real(self):
        assert set(DEFAULT_CATEGORIES) <= set(ALL_CATEGORIES)

    def test_unknown_category_is_rejected(self, catalog):
        with pytest.raises(ValueError, match="unknown categor"):
            select(catalog, ["screener", "not_a_category"])

    def test_long_running_tools_are_hidden_by_default(self):
        _s, default = build_server(ServerConfig(categories=ALL_CATEGORIES))
        names = {t.name for t in default.tools}
        assert not (names & LONG_RUNNING), "long-running tools exposed by default"

        _s2, opted = build_server(
            ServerConfig(categories=ALL_CATEGORIES, enable_long_running=True)
        )
        assert LONG_RUNNING <= {t.name for t in opted.tools}

    def test_long_running_names_exist(self, catalog):
        unknown = LONG_RUNNING - set(catalog)
        assert not unknown, f"LONG_RUNNING names no tool has: {sorted(unknown)}"

    def test_output_schemas_are_opt_in(self):
        _s, off = build_server(ServerConfig(categories=ALL_CATEGORIES))
        _s2, on = build_server(
            ServerConfig(categories=ALL_CATEGORIES, include_output_schemas=True)
        )
        assert all(t.output_schema is None for t in off.tools)
        assert all(t.output_schema is not None for t in on.tools)
        # Declaring them is not free -- that is why it is a flag.
        assert on.context_bytes() > off.context_bytes() * 1.5


class TestResultTruncation:
    def test_small_results_pass_through_untouched(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        payload = {"sharpe": 1.2, "trades": 4}
        out, uri = _resources.store_result("t", payload, 4096)
        assert out == payload
        assert uri is None

    def test_large_results_are_summarized_and_say_so(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        payload = {
            "sharpe": 1.2,
            "equity_curve": [{"d": i, "v": float(i)} for i in range(2000)],
        }
        out, uri = _resources.store_result("run_sma_backtest", payload, 4096)
        assert uri is not None
        assert out["sharpe"] == 1.2
        assert "equity_curve" not in out
        # The omission must be legible, or a model reports the summary as
        # if it were the whole result.
        truncated = out["_truncated"]
        assert any("equity_curve" in f for f in truncated["omitted_fields"])
        assert truncated["result_uri"] == uri

    def test_the_stored_result_round_trips_whole(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        payload = {"equity_curve": [{"d": i} for i in range(2000)]}
        _out, uri = _resources.store_result("x", payload, 4096)
        restored = _resources.read(uri)
        assert len(restored["result"]["equity_curve"]) == 2000


class TestResourceSandbox:
    @pytest.mark.parametrize(
        "uri",
        [
            "sqt://result/../../etc/passwd",
            "sqt://artifact/../../secrets/x",
            "sqt://model/..",
            "sqt://dataset/a/b/c",
            "sqt://nope/x",
            "http://example.com",
            "sqt://",
        ],
    )
    def test_hostile_uris_are_rejected(self, uri, tmp_path, monkeypatch):
        # A URI is untrusted client input and is the only place in this
        # server where a traversal bug is reachable from outside.
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        with pytest.raises(Exception):
            _resources.read(uri)

    def test_static_catalogs_resolve(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        for uri in (
            _resources.CATALOG_CATEGORIES,
            _resources.CATALOG_FEATURES,
            _resources.CATALOG_CAPABILITIES,
        ):
            payload = _resources.read(uri)
            assert isinstance(payload, dict) and payload
            json.dumps(payload, allow_nan=False, default=str)

    def test_every_template_is_well_formed(self):
        for template, title, description in _resources.TEMPLATES:
            assert template.startswith("sqt://")
            assert "{" in template and "}" in template
            assert title and description


class TestPrompts:
    def test_every_prompt_renders_with_its_required_arguments(self):
        for prompt in _prompts.PROMPTS:
            values = {a.name: f"<{a.name}>" for a in prompt.arguments if a.required}
            text = prompt.build(values)
            assert text.strip()
            for arg in prompt.arguments:
                if arg.required:
                    assert (
                        f"<{arg.name}>" in text
                    ), f"{prompt.name} ignores its required {arg.name!r} argument"

    def test_missing_required_argument_is_refused(self):
        prompt = _prompts.BY_NAME["pair_trade_study"]
        with pytest.raises(ValueError, match="needs argument"):
            prompt.build({"symbol_a": "AAPL"})

    def test_every_prompt_declares_real_categories(self):
        for prompt in _prompts.PROMPTS:
            for category in _prompts.required_categories(prompt):
                assert (
                    category in ALL_CATEGORIES
                ), f"{prompt.name} needs unknown category {category!r}"

    def test_prompt_names_are_unique_and_legal(self):
        names = _prompts.names()
        assert len(names) == len(set(names))
        assert all(MCP_NAME_RE.match(n) for n in names)


class TestStdoutHygiene:
    def test_importing_the_library_writes_nothing_to_stdout(self):
        # stdio transport shares stdout with JSON-RPC. One stray print in
        # any library module corrupts every session, and it surfaces as an
        # unintelligible protocol error rather than as a Python problem.
        import importlib

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            for module in (
                "standard_quant_tools.agent.tools",
                "standard_quant_tools.modeling.agent",
                "standard_quant_tools.mcp.server",
                "standard_quant_tools.mcp.resources",
                "standard_quant_tools.backtest.engine",
                "standard_quant_tools.analysis.correlation",
            ):
                importlib.reload(importlib.import_module(module))
        assert (
            buffer.getvalue() == ""
        ), f"library import wrote to stdout: {buffer.getvalue()[:200]!r}"

    def test_building_the_server_writes_nothing_to_stdout(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            build_server(ServerConfig(categories=ALL_CATEGORIES))
        assert buffer.getvalue() == ""


class TestServerWiring:
    def test_server_builds_with_every_category(self):
        server, handlers = build_server(ServerConfig(categories=ALL_CATEGORIES))
        assert server is not None
        assert len(handlers.tools) == len(build_catalog()) - len(LONG_RUNNING)

    def test_tools_serialize_for_the_wire(self):
        _server, handlers = build_server(ServerConfig(categories=ALL_CATEGORIES))
        for tool in handlers.tools:
            json.dumps(
                tool.model_dump(by_alias=True, exclude_none=True), allow_nan=False
            )

    def test_the_resource_handlers_build_and_serialize(self):
        # This class of bug -- a handler that constructs an object the SDK
        # rejects -- was originally caught only by the stdio session test,
        # which costs a subprocess. Exercising every handler in-process
        # catches it in milliseconds instead.
        import anyio

        _server, handlers = build_server(ServerConfig(categories=ALL_CATEGORIES))

        async def call_all():
            return (
                await handlers.list_resources(None, None),
                await handlers.list_resource_templates(None, None),
                await handlers.list_prompts(None, None),
                await handlers.list_tools(None, None),
            )

        for result in anyio.run(call_all):
            json.dumps(
                result.model_dump(by_alias=True, mode="json", exclude_none=True),
                allow_nan=False,
            )

    def test_every_listed_resource_uri_is_readable(self, tmp_path, monkeypatch):
        import anyio

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        _server, handlers = build_server(ServerConfig(categories=ALL_CATEGORIES))

        async def go():
            listed = await handlers.list_resources(None, None)
            import mcp.types as types

            out = []
            for resource in listed.resources:
                out.append(
                    await handlers.read_resource(
                        None,
                        types.ReadResourceRequestParams(uri=str(resource.uri)),
                    )
                )
            return out

        for result in anyio.run(go):
            assert result.contents
            json.loads(result.contents[0].text)

    def test_unknown_tool_returns_an_error_naming_the_categories(self):
        import anyio

        _server, handlers = build_server(ServerConfig(categories=("screener",)))

        async def call():
            import mcp.types as types

            return await handlers.call_tool(
                None, types.CallToolRequestParams(name="not_a_tool", arguments={})
            )

        result = anyio.run(call)
        assert result.is_error is True
        assert "screener" in result.content[0].text
