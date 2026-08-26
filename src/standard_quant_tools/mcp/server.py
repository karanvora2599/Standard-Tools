"""
The MCP server: tools, resources and prompts, over stdio or HTTP.

WHAT THIS FILE IS ALLOWED TO DO. Convert protocol shapes, route to a
dispatcher, convert back. Nothing else. Every tool it serves already exists
and is already tested in the library; if the MCP layer ever needs a tool
`dispatch()` does not have, that tool belongs in the library.

The reason for the rule is history. `Implementation/` holds four copies of
an agent loop, and what kept them consistent is that none of them contains
logic. This is the fifth surface onto the same registries, and it gets the
same discipline -- otherwise it becomes a third tool registry that nobody is
testing.

TWO REGISTRIES, PAIRED AT THE POINT OF USE. `catalog.dispatch_for(entry)`
returns the dispatch function belonging to that tool's registry, so the
schemas and the executor are never chosen separately. The two dispatchers
have identical signatures, which is exactly why the pairing has to be
deliberate.

THE TRANSPORT IS NOT THIS FILE'S BUSINESS. `build_server()` returns a
configured `Server` that knows nothing about how bytes reach it; stdio is
wired below and streamable HTTP in `http.py`, which is imported lazily
because it needs starlette and uvicorn and a stdio user should not be
stopped from starting for want of a web server.

THE AUDIT TRAIL COMES FOR FREE, AND MUST NOT BE BROKEN. Both dispatchers
already route through `audit._run_and_record`, so every call made through
this server produces a hash-chained, replayable decision record. The server
sets a request-id context per call so a record ties back to the client
conversation that caused it, and serves the records at `sqt://audit/{id}`.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Set

try:
    # NOTE: absolute imports, so this is the MCP SDK on sys.path and not the
    # package this module lives in. Python 3 resolves `mcp` at top level;
    # the shadowing is a readability hazard, not a runtime one, and
    # tests/mcp pins that both resolve correctly.
    import anyio
    import mcp.types as types
    from mcp.server import (
        InitializationOptions,
        NotificationOptions,
        Server,
        ServerRequestContext,
    )
    from mcp.server.stdio import stdio_server
except ModuleNotFoundError as exc:  # pragma: no cover - install-shape error
    raise ModuleNotFoundError(
        "The MCP server needs the Model Context Protocol SDK, which is not a "
        "dependency of the core package. Install it with:\n\n"
        "    pip install 'standard_quant_tools[mcp]'\n\n"
        f"(original error: {exc})"
    ) from exc

from standard_quant_tools import __version__ as _sqt_version
from standard_quant_tools.error import QuantError
from standard_quant_tools.mcp import prompts as _prompts
from standard_quant_tools.mcp import resources as _resources
from standard_quant_tools.mcp.catalog import (
    SCHEMA_FETCH_TOOL,
    ToolEntry,
    build_catalog,
    category_costs,
    dispatch_for,
    plan_detail,
    runtime_costs,
    select,
    thin_description,
    thin_schema,
)
from standard_quant_tools.mcp.config import ServerConfig, report, resolve

SERVER_NAME = "standard-quant-tools"

#: Hidden unless --enable-long-running is passed. `scan_pairs` is measured
#: at 5.31 minutes over a 2,000-ticker universe, and `run_backtest_optimization`
#: grows with the grid, so both can outlast a default client timeout -- and a
#: timeout that fires after most of the work is done is worse than not
#: offering the tool at all.
#:
#: `run_screener` is deliberately NOT here. Its runtime is set by the
#: universe the caller passes rather than being long by nature, and hiding
#: it would leave the `screener` category holding one tool -- worse than the
#: risk it avoids. It gets the runtime note instead.
LONG_RUNNING = frozenset({"scan_pairs", "run_backtest_optimization"})

#: Not hidden, but worth warning about: runtime is set by the caller's input.
SCALES_WITH_INPUT = frozenset({"run_screener", "run_portfolio_simulation"})


def _annotations(entry: ToolEntry) -> types.ToolAnnotations:
    """
    MCP hints, derived from the code rather than hand-maintained.

    `read_only_hint` is True for all 54 because this library does not place
    orders, hold positions, or mutate anything outside its own artifact
    store. That is a property worth encoding in the protocol and worth
    keeping: the moment one tool breaks it, a client has to treat the whole
    server as write-capable.
    """
    return types.ToolAnnotations(
        title=entry.name,
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=entry.idempotent,
        open_world_hint=entry.reads_market_data,
    )


def _to_mcp_tool(
    entry: ToolEntry, include_output_schema: bool, thin: bool = False
) -> types.Tool:
    description = entry.description
    if thin:
        # One line of purpose, and an explicit instruction to go and fetch
        # the rest. An agent that skips that step and guesses gets a clean
        # rejection from `extra="forbid"` rather than a silent default, but
        # a rejected call is still a wasted turn -- so the instruction is in
        # the description AND in the schema, not only one of them.
        return types.Tool(
            name=entry.name,
            description=(
                f"{thin_description(description)} "
                f"[args: {SCHEMA_FETCH_TOOL}({entry.name!r})]"
            ),
            inputSchema=thin_schema(entry.name),
            outputSchema=None,
            annotations=_annotations(entry),
        )
    if entry.name in LONG_RUNNING or entry.name in SCALES_WITH_INPUT:
        description = (
            f"{description}\n\nRUNTIME: this tool can take minutes on a large "
            "universe or grid. Tell the user before starting a big one."
        )
    return types.Tool(
        name=entry.name,
        description=description,
        inputSchema=entry.input_schema,
        outputSchema=entry.output_schema if include_output_schema else None,
        annotations=_annotations(entry),
    )


class StandardToolsServer:
    """Holds the resolved config and the selected tools for one process."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        catalog = build_catalog()
        # Runtime first, then category. The order does not change the result
        # -- a category is owned by exactly one runtime -- but it makes the
        # scoping read the way it is documented: the runtime is the boundary,
        # the category narrows within it.
        selected = [
            e
            for e in select(catalog, config.categories)
            if e.runtime in set(config.runtimes)
        ]
        if not config.enable_long_running:
            selected = [e for e in selected if e.name not in LONG_RUNNING]

        self.thinned: Set[str] = plan_detail(
            selected,
            mode=config.tool_detail,
            budget=config.detail_budget,
            include_output_schemas=config.include_output_schemas,
        )
        if self.thinned and SCHEMA_FETCH_TOOL not in {e.name for e in selected}:
            # A thin entry says "call describe_tool for the schema". If the
            # chosen scope does not contain describe_tool, that instruction
            # is unfollowable and every thinned tool becomes uncallable. It
            # is 700 bytes and it belongs to `meta`, so rather than force
            # every scoped deployment to remember `+meta`, the server adds
            # it and says so at startup. Scope is still real: this is the
            # one tool whose absence would make the listing itself a lie.
            fetch = catalog.get(SCHEMA_FETCH_TOOL)
            if fetch is not None:
                selected = sorted(selected + [fetch], key=lambda e: e.name)

        self.entries: Dict[str, ToolEntry] = {e.name: e for e in selected}
        self.tools: List[types.Tool] = [
            _to_mcp_tool(e, config.include_output_schemas, thin=e.name in self.thinned)
            for e in selected
        ]
        self.catalog = catalog

    def context_bytes(self) -> int:
        return sum(
            len(json.dumps(t.model_dump(by_alias=True, exclude_none=True)))
            for t in self.tools
        )

    # ── tools ────────────────────────────────────────────────────────

    async def list_tools(
        self,
        _ctx: ServerRequestContext[Any],
        _params: Optional[types.PaginatedRequestParams] = None,
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=self.tools)

    def _refusal(self, name: str) -> str:
        """
        Why a tool this server does not serve was refused.

        Three different situations, and an agent can only correct the one it
        is actually in. A tool that exists in another runtime is a SCOPE
        problem the operator can fix; a tool hidden by --enable-long-running
        is a policy decision; a name that exists nowhere is a hallucination,
        and saying so plainly is more useful than implying a flag would
        help. This mirrors `Runtime.dispatch`'s refusal, which names the
        owner for the same reason.
        """
        known = self.catalog.get(name)
        if known is None:
            return (
                f"unknown tool {name!r}. No tool by that name exists in this "
                f"library. This server serves {len(self.entries)} tools from "
                f"the {'+'.join(self.config.runtimes)} runtime"
                f"{'s' if len(self.config.runtimes) > 1 else ''}. "
                "Read sqt://catalog/categories for what exists, or call "
                "tools/list for what this server serves -- do not guess "
                "another name."
            )
        if known.runtime not in set(self.config.runtimes):
            from standard_quant_tools.agent.runtimes import MOVED_FROM

            moved = MOVED_FROM.get(name)
            history = (
                f" It used to be served by {moved!r}, which this server IS "
                "scoped to -- it moved, and its arguments are unchanged."
                if moved and moved in set(self.config.runtimes)
                else ""
            )
            return (
                f"{name!r} exists, but belongs to the {known.runtime!r} "
                f"runtime and this server is scoped to "
                f"{'+'.join(self.config.runtimes)}.{history} Restart with "
                f"`--runtime {known.runtime}` to serve it -- widening scope "
                "is a decision, not a fallback."
            )
        if name in LONG_RUNNING:
            return (
                f"{name!r} is served by this runtime but is hidden because it "
                "can run for minutes. Restart with --enable-long-running to "
                "expose it, and expect a client timeout to be the risk you "
                "are taking on."
            )
        return (
            f"{name!r} exists in the {known.runtime!r} runtime but is not in "
            f"the selected categories ({', '.join(self.config.categories)}). "
            f"It belongs to {known.category!r}."
        )

    async def call_tool(
        self,
        _ctx: ServerRequestContext[Any],
        params: types.CallToolRequestParams,
    ) -> types.CallToolResult:
        entry = self.entries.get(params.name)
        if entry is None:
            return _error(self._refusal(params.name))

        dispatch = dispatch_for(entry)
        arguments = dict(params.arguments or {})

        try:
            # The library's tools are synchronous and some are slow, so they
            # run on a worker thread -- otherwise one backtest blocks the
            # event loop and the server stops answering pings.
            result = await anyio.to_thread.run_sync(
                lambda: dispatch(entry.name, arguments)
            )
        except QuantError as exc:
            # The library's own errors are written to be self-correcting, so
            # they go back verbatim rather than being flattened to
            # "tool failed".
            return _error(f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # pragma: no cover - unexpected
            return _error(f"{type(exc).__name__}: {exc}")

        payload, _uri = _resources.store_result(
            entry.name, result, self.config.inline_limit_bytes
        )
        text = json.dumps(payload, indent=2, default=str, allow_nan=False)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            structuredContent=payload,
        )

    # ── resources ────────────────────────────────────────────────────

    async def list_resources(
        self,
        _ctx: ServerRequestContext[Any],
        _params: Optional[types.PaginatedRequestParams] = None,
    ) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[
                types.Resource(
                    uri=(_resources.CATALOG_CATEGORIES),
                    name="tool_categories",
                    title="Tool categories and their context cost",
                    description=(
                        "Every category, how many tools it holds and what it "
                        "costs a client's context at connect."
                    ),
                    mimeType="application/json",
                ),
                types.Resource(
                    uri=(_resources.CATALOG_FEATURES),
                    name="feature_catalog",
                    title="Modeling feature catalog",
                    description="Every built-in feature the modeling runtime offers.",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri=(_resources.CATALOG_CAPABILITIES),
                    name="modeling_capabilities",
                    title="Modeling capabilities of this install",
                    description=(
                        "Tasks, estimators, targets, validation schemes and "
                        "which optional libraries are importable here."
                    ),
                    mimeType="application/json",
                ),
            ]
        )

    async def list_resource_templates(
        self,
        _ctx: ServerRequestContext[Any],
        _params: Optional[types.PaginatedRequestParams] = None,
    ) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(
            resourceTemplates=[
                types.ResourceTemplate(
                    uriTemplate=template,
                    name=title.lower().replace(" ", "_"),
                    title=title,
                    description=description,
                    mimeType="application/json",
                )
                for template, title, description in _resources.TEMPLATES
            ]
        )

    async def read_resource(
        self,
        _ctx: ServerRequestContext[Any],
        params: types.ReadResourceRequestParams,
    ) -> types.ReadResourceResult:
        uri = str(params.uri)
        try:
            payload = _resources.read(uri)
        except QuantError as exc:
            raise ValueError(str(exc)) from exc
        return types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=params.uri,
                    mimeType="application/json",
                    text=json.dumps(payload, indent=2, default=str, allow_nan=False),
                )
            ]
        )

    # ── prompts ──────────────────────────────────────────────────────

    async def list_prompts(
        self,
        _ctx: ServerRequestContext[Any],
        _params: Optional[types.PaginatedRequestParams] = None,
    ) -> types.ListPromptsResult:
        return types.ListPromptsResult(
            prompts=[
                types.Prompt(
                    name=p.name,
                    title=p.title,
                    description=p.description,
                    arguments=[
                        types.PromptArgument(
                            name=a.name,
                            description=a.description,
                            required=a.required,
                        )
                        for a in p.arguments
                    ],
                )
                for p in _prompts.PROMPTS
            ]
        )

    async def get_prompt(
        self,
        _ctx: ServerRequestContext[Any],
        params: types.GetPromptRequestParams,
    ) -> types.GetPromptResult:
        prompt = _prompts.BY_NAME.get(params.name)
        if prompt is None:
            raise ValueError(f"unknown prompt {params.name!r}")
        try:
            text = prompt.build(params.arguments or {})
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        # A workflow whose tools were never loaded is worse than no workflow:
        # the model improvises the steps it cannot run. Say so up front.
        needed = set(_prompts.required_categories(prompt))
        absent = sorted(needed - set(self.config.categories))
        if absent:
            text = (
                f"NOTE: this workflow needs the {', '.join(absent)} tool "
                f"categor{'y' if len(absent) == 1 else 'ies'}, which this "
                "server was not started with. Say so rather than "
                "improvising the steps you cannot run.\n\n" + text
            )

        return types.GetPromptResult(
            description=prompt.description,
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(type="text", text=text),
                )
            ],
        )


def _error(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)],
        isError=True,
    )


def build_server(config: ServerConfig) -> tuple[Server[Any], StandardToolsServer]:
    handlers = StandardToolsServer(config)
    server: Server[Any] = Server(
        SERVER_NAME,
        version=_sqt_version,
        title="Standard Quant Tools",
        instructions=(
            "Quantitative research tools: market data, indicators, risk "
            "metrics, backtesting, portfolio construction, and a separate "
            "runtime for building walk-forward-validated statistical models. "
            "Every tool is read-only analysis -- nothing here places orders "
            "or moves money. Results larger than the inline limit come back "
            "summarized with a sqt:// resource link to the full payload; "
            "read that link rather than reporting the summary as complete."
        ),
        on_list_tools=handlers.list_tools,
        on_call_tool=handlers.call_tool,
        on_list_resources=handlers.list_resources,
        on_list_resource_templates=handlers.list_resource_templates,
        on_read_resource=handlers.read_resource,
        on_list_prompts=handlers.list_prompts,
        on_get_prompt=handlers.get_prompt,
    )
    return server, handlers


async def _serve_stdio(server: Server[Any]) -> None:
    options = InitializationOptions(
        server_name=SERVER_NAME,
        server_version=_sqt_version,
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
        instructions=server.instructions,
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)


def _budget_table(title: str, rows, width: int) -> None:
    print(f"{title.ljust(width)}  tools    bytes   ~tokens", file=sys.stderr)
    for name, (count, size) in rows:
        print(
            f"{name.ljust(width)}  {count:>5}  {size:>7,}  {size // 4:>8,}",
            file=sys.stderr,
        )


def print_budget() -> None:
    """
    The context cost, per runtime and per category.

    Runtimes come first because they are the scope a client is served at:
    `--runtime backtest` is one number a reader can act on, where the three
    backtest categories are a sum they have to do themselves. Categories
    stay because they are still the filter WITHIN a runtime.
    """
    catalog = build_catalog()
    rt = sorted(runtime_costs(catalog).items(), key=lambda kv: -kv[1][1])
    cat = sorted(category_costs(catalog).items(), key=lambda kv: -kv[1][1])
    width = max(len(n) for n, _ in rt + cat)

    _budget_table("runtime", rt, width)
    total = sum(size for _, (_, size) in rt)
    count = sum(c for _, (c, _) in rt)
    print(
        f"{'all'.ljust(width)}  {count:>5}  {total:>7,}  {total // 4:>8,}",
        file=sys.stderr,
    )
    # What a client pays is one row above, not the total. Say so, because
    # the total is the number that looks alarming and is never paid.
    heaviest, (_, worst) = rt[0]
    print(
        "\n"
        f"  a client is served ONE runtime: {heaviest} is the most "
        f"expensive at {worst:,} bytes ({worst / total:.0%} of the total).",
        file=sys.stderr,
    )

    print(file=sys.stderr)
    _budget_table("category", cat, width)


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--print-budget" in argv:
        print_budget()
        return
    config = resolve(argv)

    # Built and reported once, whichever transport carries it: the tool
    # surface and its context cost are properties of the configuration, not
    # of the socket, and a reader comparing two deployments should be able
    # to compare the same lines.
    server, handlers = build_server(config)
    report(config, len(handlers.tools), handlers.context_bytes())

    if config.transport == "http":
        from standard_quant_tools.mcp.http import serve_http

        serve_http(config, server, len(handlers.tools))
        return

    anyio.run(_serve_stdio, server)


if __name__ == "__main__":  # pragma: no cover
    main()
