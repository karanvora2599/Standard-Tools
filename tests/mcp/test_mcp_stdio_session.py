"""
End-to-end: drive the server as a real MCP client would, over stdio.

WHY THIS EXISTS SEPARATELY. Everything in test_mcp_surface.py inspects
objects in-process. That catches schema and wiring drift, and it would not
catch the failure this file exists for: the server launching as a
subprocess, on a machine whose working directory and environment it did not
choose, speaking JSON-RPC on a stdout it shares with anything the library
might print.

That is the failure mode most likely to reach a user, because it looks like
a protocol bug rather than a Python one -- and it cannot be reproduced by
importing anything.

Marked `integration` because it spawns a process. No network: every call
made here either reads a catalog or is rejected before it fetches anything.
"""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = pytest.mark.integration

anyio = pytest.importorskip("anyio")
pytest.importorskip("mcp.client.stdio")

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

from .. import REPO_ROOT  # noqa: E402


def _server_params(tmp_path, *args: str) -> StdioServerParameters:
    env = dict(os.environ)
    env.update(
        {
            "SQT_RUNS_DIR": str(tmp_path / "runs"),
            "SQT_AUDIT_DIR": str(tmp_path / "audit"),
            "SQT_CACHE_DIR": str(tmp_path / "cache"),
            "SQT_AUDIT_ENABLED": "0",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        }
    )
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "standard_quant_tools.mcp.server", *args],
        env=env,
        cwd=str(REPO_ROOT),
    )


def run_session(tmp_path, body, *args):
    """
    Open a session, run `body(session)`, and tear down in the same task.

    Deliberately NOT an async generator yielding the session. anyio refuses a
    cancel scope exited from a different task, and an async generator's
    cleanup runs whenever the caller stops iterating, which is not reliably
    the task that entered it. That surfaces as "Attempted to exit cancel
    scope in a different task" during teardown and masks whatever the test
    was actually asserting -- which is exactly how it wasted a debugging
    pass here.
    """

    async def go():
        params = _server_params(tmp_path, *args)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await body(session)

    return anyio.run(go)


class TestStdioSession:
    def test_initialize_and_list_tools(self, tmp_path):
        async def body(session):
            result = await session.list_tools()
            return [t.name for t in result.tools]

        names = run_session(tmp_path, body)
        assert names, "server exposed no tools"
        assert "analyze_stock_risk" in names
        assert "run_screener" in names
        assert "scan_pairs" not in names, "long-running tool exposed by default"

    def test_categories_flag_changes_the_surface(self, tmp_path):
        async def body(session):
            result = await session.list_tools()
            return [t.name for t in result.tools]

        names = run_session(tmp_path, body, "--categories", "screener")
        assert set(names) <= {"run_screener", "get_stock_fundamentals"}
        assert names

    def test_tools_carry_annotations_over_the_wire(self, tmp_path):
        async def body(session):
            result = await session.list_tools()
            return result.tools

        for tool in run_session(tmp_path, body):
            assert tool.annotations is not None, tool.name
            assert tool.annotations.read_only_hint is True, tool.name

    def test_resources_and_templates_are_listed(self, tmp_path):
        async def body(session):
            resources = await session.list_resources()
            templates = await session.list_resource_templates()
            return (
                [str(r.uri) for r in resources.resources],
                [t.uri_template for t in templates.resource_templates],
            )

        uris, templates = run_session(tmp_path, body)
        assert any("catalog/categories" in u for u in uris), uris
        assert any("sqt://audit/" in t for t in templates), templates

    def test_reading_a_catalog_resource(self, tmp_path):
        async def body(session):
            return await session.read_resource("sqt://catalog/categories")

        result = run_session(tmp_path, body)
        assert result.contents
        assert "categories" in result.contents[0].text

    def test_prompts_are_listed_and_render(self, tmp_path):
        async def body(session):
            listed = await session.list_prompts()
            got = await session.get_prompt(
                "pair_trade_study", {"symbol_a": "KO", "symbol_b": "PEP"}
            )
            return [p.name for p in listed.prompts], got.messages[0].content.text

        names, text = run_session(tmp_path, body)
        assert "build_and_validate_model" in names
        assert "KO" in text and "PEP" in text

    def test_a_prompt_warns_when_its_tools_are_absent(self, tmp_path):
        # build_and_validate_model needs the modeling category, which the
        # default selection does not load. A workflow whose tools were never
        # loaded is worse than no workflow -- the model improvises the steps
        # it cannot run.
        async def body(session):
            got = await session.get_prompt(
                "build_and_validate_model", {"universe": "AAPL,MSFT"}
            )
            return got.messages[0].content.text

        text = run_session(tmp_path, body)
        assert "modeling" in text
        assert "not started with" in text

    def test_a_bad_tool_call_comes_back_as_an_error_not_a_crash(self, tmp_path):
        async def body(session):
            return await session.call_tool("no_such_tool", {})

        result = run_session(tmp_path, body)
        assert result.is_error is True

    def test_an_offline_tool_executes_end_to_end(self, tmp_path):
        # Black-Scholes is arithmetic: a real tool call through real dispatch
        # returning a real structured result, with no network involved.
        async def body(session):
            return await session.call_tool(
                "get_option_pricing",
                {
                    "spot": 100.0,
                    "strike": 100.0,
                    "time_to_expiry": 1.0,
                    "risk_free_rate": 0.05,
                    "volatility": 0.2,
                    "option_type": "call",
                },
            )

        # `derivatives`, not `analysis`: option pricing moved out of
        # `research` when derivatives became its own runtime. The tool, its
        # arguments and its result are unchanged -- only the scope that
        # serves it moved, which is exactly what this call has to name.
        result = run_session(tmp_path, body, "--categories", "derivatives")
        assert result.is_error is not True, result.content[0].text
        assert result.structured_content, "no structuredContent returned"
        assert "price" in result.structured_content

    def test_stdout_stays_clean_enough_for_jsonrpc(self, tmp_path):
        # If any library module wrote to stdout at import, initialize() would
        # fail to parse the stream. Reaching a full tool listing is the
        # proof, and it is why this test spawns a real process rather than
        # importing anything.
        async def body(session):
            result = await session.list_tools()
            return len(result.tools)

        assert run_session(tmp_path, body, "--categories", "all") > 50
