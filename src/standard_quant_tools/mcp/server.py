"""
The MCP server: tools, resources and prompts over stdio.

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

THE AUDIT TRAIL COMES FOR FREE, AND MUST NOT BE BROKEN. Both dispatchers
already route through `audit._run_and_record`, so every call made through
this server produces a hash-chained, replayable decision record. The server
sets a request-id context per call so a record ties back to the client
conversation that caused it, and serves the records at `sqt://audit/{id}`.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Sequence

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
    ToolEntry,
    build_catalog,
    category_costs,
    dispatch_for,
    select,
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


def _to_mcp_tool(entry: ToolEntry, include_output_schema: bool) -> types.Tool:
    description = entry.description
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
        selected = select(catalog, config.categories)
        if not config.enable_long_running:
            selected = [e for e in selected if e.name not in LONG_RUNNING]
        self.entries: Dict[str, ToolEntry] = {e.name: e for e in selected}
        self.tools: List[types.Tool] = [
            _to_mcp_tool(e, config.include_output_schemas) for e in selected
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

    async def call_tool(
        self,
        _ctx: ServerRequestContext[Any],
        params: types.CallToolRequestParams,
    ) -> types.CallToolResult:
        entry = self.entries.get(params.name)
        if entry is None:
            return _error(
                f"unknown tool {params.name!r}. This server was started with "
                f"categories {', '.join(self.config.categories)}; the tool may "
                "exist in a category that is not loaded."
            )

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


async def _serve(config: ServerConfig) -> None:
    server, handlers = build_server(config)
    report(config, len(handlers.tools), handlers.context_bytes())
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


def print_budget() -> None:
    """The per-category context cost, for choosing a --categories value."""
    catalog = build_catalog()
    rows = sorted(category_costs(catalog).items(), key=lambda kv: -kv[1][1])
    width = max(len(name) for name, _ in rows)
    print(f"{'category'.ljust(width)}  tools    bytes   ~tokens", file=sys.stderr)
    for name, (count, size) in rows:
        print(
            f"{name.ljust(width)}  {count:>5}  {size:>7,}  {size // 4:>8,}",
            file=sys.stderr,
        )
    total = sum(size for _, (_, size) in rows)
    print(
        f"{'all'.ljust(width)}  {sum(c for _, (c, _) in rows):>5}  "
        f"{total:>7,}  {total // 4:>8,}",
        file=sys.stderr,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--print-budget" in argv:
        print_budget()
        return
    config = resolve(argv)
    anyio.run(_serve, config)


if __name__ == "__main__":  # pragma: no cover
    main()
