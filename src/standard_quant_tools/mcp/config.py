"""
Server configuration, resolved once at startup.

WHY THIS IS FAIL-FAST. An MCP server is launched by a client, with a working
directory nobody chose and an environment stripped down to whatever the
client passes through. Every path this library writes to comes from an
environment variable, so an unset `SQT_RUNS_DIR` does not fail at startup --
it fails three turns into a conversation, when a backtest tries to persist
an artifact somewhere unintended, and the user sees a tool error rather than
a configuration problem.

So everything is resolved and checked here, before the first request is
served, and reported once to stderr.

STDERR, NEVER STDOUT. stdio transport puts JSON-RPC on stdout. A single
stray write corrupts the stream in a way that surfaces as an unintelligible
protocol error, so this module logs to stderr and `tests/mcp/` asserts that
importing the library writes nothing to stdout.

THE HTTP TRANSPORT INHERITS ALL OF THAT AND ADDS A REFUSAL. Over stdio the
only caller is the process that launched the server; over HTTP it is
anything that can route to the port. So `--transport http` will not start
on a non-loopback address without either SQT_MCP_TOKEN or an explicit
`--no-auth`, and it will not start on one without being told which Host
headers to accept. Both are failures at startup rather than at the first
request, for the same reason as everything else here: the alternative is a
tool error three turns into somebody else's conversation.

NO SDK IMPORTS. This module is parsed by tests that do not install the MCP
SDK, and `http.py` imports the loopback helper from here rather than the
other way round.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from standard_quant_tools.mcp.catalog import (
    ALL_CATEGORIES,
    ALL_RUNTIMES,
    DEFAULT_CATEGORIES,
    DEFAULT_DETAIL_BUDGET,
    DETAIL_MODES,
    RUNTIME_CATEGORY_MAP,
    categories_for_runtimes,
)

#: Results larger than this are persisted and returned as a resource link
#: instead of inlined. A five-year daily backtest carries ~1,250 equity
#: points plus a trade log; inlined, that sits in the client's context for
#: the rest of the session. 4 KB is the default rather than a law -- it is
#: reported in every truncated result so nothing is ever silently dropped.
DEFAULT_INLINE_LIMIT = 4096

#: Seconds between liveness notifications while a tool runs. Only sent
#: when the client supplies a progressToken; see mcp/progress.py for why
#: they carry elapsed time and no total.
DEFAULT_HEARTBEAT_SECONDS = 5.0

#: The context a client may be told it is spending, in bytes, or None for
#: no ceiling at all. **Uncapped by default.**
#:
#: This was 180,000 and it was a real constraint: a test failed when the
#: whole surface crossed it, so the tool count was bounded by a number
#: chosen when the library had 54 tools. That is the wrong control. What
#: actually governs cost is `--tool-detail auto`, which thins the most
#: expensive schemas until the listing fits `detail_budget` and leaves
#: every tool advertised -- a mechanism that scales with the surface
#: instead of capping it. A fixed ceiling on top of that mechanism only
#: decided how many tools the library was allowed to have.
#:
#: Set an integer to get the old startup warning back for a deployment
#: that genuinely has a context limit. None means never warn, and nothing
#: fails on size.
CONTEXT_CEILING_BYTES = None

_ENV_DIRS = (
    ("runs_dir", "SQT_RUNS_DIR", "artifact store"),
    ("audit_dir", "SQT_AUDIT_DIR", "audit trail"),
    ("cache_dir", "SQT_CACHE_DIR", "Parquet cache"),
)


#: Addresses reachable only from the machine itself. Binding to one of
#: these is what makes a missing token acceptable rather than negligent.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

#: The bearer token is read from the environment and never from a flag. A
#: command line is visible in the process table to every user on the box,
#: and a shared secret that leaks to `ps` is not one.
TOKEN_ENV_VAR = "SQT_MCP_TOKEN"

DEFAULT_HTTP_PORT = 8765
DEFAULT_HTTP_PATH = "/mcp"


def is_loopback(host: str) -> bool:
    return host.strip().strip("[]").lower() in LOOPBACK_HOSTS


@dataclass(frozen=True)
class ServerConfig:
    categories: Tuple[str, ...] = DEFAULT_CATEGORIES
    #: The runtimes this server is scoped to. Defaults to all of them,
    #: which is what `--categories` alone has always meant. Narrowing it
    #: narrows the DISPATCH as well as the listing: a server scoped to
    #: research cannot execute a backtest tool even if a client names one.
    runtimes: Tuple[str, ...] = ALL_RUNTIMES
    runs_dir: Optional[Path] = None
    audit_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    inline_limit_bytes: int = DEFAULT_INLINE_LIMIT
    include_output_schemas: bool = False
    #: How tools are advertised: 'full' sends every schema up
    #: front, 'auto' thins the most expensive until the surface
    #: fits `detail_budget`, 'thin' thins everything. A thinned
    #: tool is listed and callable; its ARGUMENTS come from
    #: describe_tool instead of from the listing.
    # DEFAULTS TO AUTO, not full. The backtest runtime is the most
    # expensive by a wide margin at full detail, and paying that on
    # every connection to reach a handful of its tools is waste --
    # there is no fixed ceiling being breached, because what a client
    # can afford is a property of the client rather than of this
    # library. `auto` thins only what exceeds `detail_budget`, so the
    # runtimes already under it are returned unchanged and only the
    # expensive ones differ. `describe_tool` is injected whenever anything is
    # thinned, so no schema becomes unreachable -- it becomes one
    # call away, paid for by the callers who need it rather than by
    # every session at startup.
    tool_detail: str = "auto"
    detail_budget: int = DEFAULT_DETAIL_BUDGET
    enable_long_running: bool = False
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS
    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = DEFAULT_HTTP_PORT
    path: str = DEFAULT_HTTP_PATH
    stateless: bool = False
    json_response: bool = False
    auth_token: Optional[str] = None
    allowed_hosts: Tuple[str, ...] = field(default=())
    allowed_origins: Tuple[str, ...] = field(default=())
    warnings: Tuple[str, ...] = field(default=())


def _split_runtimes(raw: str) -> List[str]:
    """Parse --runtime. Accepts `research`, `research+meta`, `a,b`, `all`."""
    if raw.strip().lower() == "all":
        return list(ALL_RUNTIMES)
    return [p.strip() for p in raw.replace("+", ",").split(",") if p.strip()]


def _split_categories(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(ALL_CATEGORIES)
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sqt-mcp",
        description=(
            "Serve Standard Tools over the Model Context Protocol, on stdio "
            "or over streamable HTTP. Tools are selected by category because "
            "the full set of 198 costs roughly 80,000 tokens of a client's "
            "context at connect."
        ),
    )
    parser.add_argument(
        "--categories",
        default=None,
        help=(
            "Comma-separated tool categories, or 'all'. "
            f"Available: {', '.join(ALL_CATEGORIES)}. "
            f"Default: {','.join(DEFAULT_CATEGORIES)}."
        ),
    )
    parser.add_argument(
        "--runtime",
        default=None,
        metavar="NAME",
        help=(
            "Serve one runtime, or several joined with '+'. "
            f"Available: {', '.join(ALL_RUNTIMES)}, or 'all'. "
            "This is the coarse scope: a runtime owns its categories, so "
            "--runtime research serves ALL of research rather than research "
            "narrowed by the --categories default. Use --categories to "
            "narrow further WITHIN the chosen runtime. Scoping here also "
            "scopes execution -- the server refuses to dispatch a tool it "
            "does not serve. Default: every runtime."
        ),
    )
    parser.add_argument(
        "--tool-detail",
        choices=DETAIL_MODES,
        default="auto",
        help=(
            "How much of each tool to advertise. 'full' (default) sends "
            "every input schema at connect, which is what this server has "
            "always done. 'auto' sends the whole schema for most tools and "
            "thins the most expensive ones until the surface fits "
            "--detail-budget; a thinned tool is still listed and still "
            "callable, but an agent must call describe_tool for its "
            "arguments first. 'thin' thins everything, which is the "
            "cheapest and the least safe -- a schema an agent has not read "
            "is one it will guess at."
        ),
    )
    parser.add_argument(
        "--detail-budget",
        type=int,
        default=DEFAULT_DETAIL_BUDGET,
        help=(
            "Byte target for --tool-detail auto. Default: "
            f"{DEFAULT_DETAIL_BUDGET:,}."
        ),
    )
    parser.add_argument(
        "--inline-limit",
        type=int,
        default=DEFAULT_INLINE_LIMIT,
        help=(
            "Results larger than this many bytes are persisted and returned "
            f"as a resource link. Default: {DEFAULT_INLINE_LIMIT}."
        ),
    )
    parser.add_argument(
        "--output-schemas",
        action="store_true",
        help=(
            "Declare outputSchema for every tool. Costs about 102%% more "
            "context (321 KB across all 198 tools) and is off by default: "
            "structuredContent is returned either way, and the declaration "
            "only helps clients that validate against it."
        ),
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=DEFAULT_HEARTBEAT_SECONDS,
        metavar="SECONDS",
        help=(
            "Seconds between liveness notifications while a tool runs. Sent "
            "only to clients that supplied a progressToken, and they carry "
            "elapsed time with NO total -- the server dispatches an opaque "
            "call and cannot know how far through it is. 0 disables them. "
            f"Default: {DEFAULT_HEARTBEAT_SECONDS}."
        ),
    )
    parser.add_argument(
        "--enable-long-running",
        action="store_true",
        help=(
            "Expose the tools that can run for minutes "
            "(scan_pairs is measured at 5.31 min over a "
            "2,000-ticker universe). Off by default, because a client "
            "timeout that fires after most of the work is done is worse "
            "than not offering the tool."
        ),
    )
    parser.add_argument(
        "--print-budget",
        action="store_true",
        help="Print the per-category context cost and exit.",
    )

    http_group = parser.add_argument_group(
        "http transport",
        "Options for --transport http. Ignored on stdio.",
    )
    http_group.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help=(
            "stdio: the client launches this process and owns it. "
            "http: run as a service that many clients, on this machine or "
            "elsewhere, connect to. Default: stdio."
        ),
    )
    http_group.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address. Default 127.0.0.1, which is reachable only from "
            "this machine. Any other value needs a token and --allow-host."
        ),
    )
    http_group.add_argument(
        "--port",
        type=int,
        default=DEFAULT_HTTP_PORT,
        help=f"Bind port. Default: {DEFAULT_HTTP_PORT}.",
    )
    http_group.add_argument(
        "--path",
        default=DEFAULT_HTTP_PATH,
        help=f"Path the MCP endpoint is mounted at. Default: {DEFAULT_HTTP_PATH}.",
    )
    http_group.add_argument(
        "--stateless",
        action="store_true",
        help=(
            "Handle every request without server-side session state, so any "
            "replica can serve any request and no load balancer needs "
            "affinity. Costs server-initiated messages: no progress "
            "notifications and no resumable streams."
        ),
    )
    http_group.add_argument(
        "--json-response",
        action="store_true",
        help=(
            "Answer with a single JSON body instead of an SSE stream. Easier "
            "for proxies and simple clients; gives up incremental delivery "
            "on the long-running tools."
        ),
    )
    http_group.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="HOST[:PORT]",
        help=(
            "Host header value to accept, repeatable. 'name:*' matches any "
            "port. Required on a non-loopback bind: the check is what stops "
            "a browser being used to reach this port from a page the user "
            "did not open."
        ),
    )
    http_group.add_argument(
        "--allow-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help=(
            "Origin header value to accept, repeatable. Only needed for "
            "browser clients; a request with no Origin is unaffected."
        ),
    )
    http_group.add_argument(
        "--no-auth",
        action="store_true",
        help=(
            "Serve without a bearer token. Only meaningful when something "
            "in front already authenticates callers. Required to be "
            "explicit, so an unauthenticated server is never the result of "
            "forgetting to set " + TOKEN_ENV_VAR + "."
        ),
    )
    return parser


def _resolve_dir(
    env_var: str,
    purpose: str,
    warnings: List[str],
    transport: str = "stdio",
) -> Optional[Path]:
    raw = os.environ.get(env_var)
    if not raw:
        chose = (
            "which the MCP client chose, not you. Set it in the client's "
            "server config."
            if transport == "stdio"
            else "which is wherever this service happened to be started from. "
            "Set it in the unit file or the container spec."
        )
        warnings.append(
            f"{env_var} is not set, so the {purpose} will use its default "
            f"location relative to this server's working directory -- {chose} "
            "Artifacts and resource URIs may not survive a restart."
        )
        return None
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".sqt-mcp-write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(
            f"sqt-mcp: {env_var} points at {path}, which is not writable "
            f"({exc}). The {purpose} needs it. Fix the path or the "
            "permissions in the client's server configuration."
        )
    return path


def _resolve_transport(args: argparse.Namespace, warnings: List[str]) -> dict:
    """Everything the transport choice implies, checked before the port opens.

    The refusals here are all the same refusal: a server reachable by more
    callers than the operator meant is not a condition that shows up in a log
    line. It shows up as somebody else's tool call in your audit trail. Each
    check is cheap to satisfy now and expensive to discover later.
    """
    token = (os.environ.get(TOKEN_ENV_VAR) or "").strip() or None
    http_options_given = (
        args.host != "127.0.0.1"
        or args.port != DEFAULT_HTTP_PORT
        or args.path != DEFAULT_HTTP_PATH
        or args.stateless
        or args.json_response
        or bool(args.allow_host)
        or bool(args.allow_origin)
        or args.no_auth
    )

    if args.transport == "stdio":
        if http_options_given:
            warnings.append(
                "HTTP options were passed but the transport is stdio, so they "
                "do nothing. Add --transport http if a service was intended."
            )
        if token:
            warnings.append(
                f"{TOKEN_ENV_VAR} is set, but stdio has no request to "
                "authenticate: the client already owns this process. Ignored."
            )
        return {"transport": "stdio", "auth_token": None}

    if not 1 <= args.port <= 65535:
        raise SystemExit("sqt-mcp: --port must be between 1 and 65535")

    path = args.path if args.path.startswith("/") else "/" + args.path
    path = path.rstrip("/") or "/"

    if token and args.no_auth:
        raise SystemExit(
            f"sqt-mcp: {TOKEN_ENV_VAR} is set and --no-auth was passed. Those "
            "ask for opposite things, and guessing which was meant is the one "
            "mistake here that fails open. Unset the variable or drop the flag."
        )
    if not token and not args.no_auth:
        raise SystemExit(
            f"sqt-mcp: --transport http needs a bearer token. Set {TOKEN_ENV_VAR} "
            "to a secret that clients will send as 'Authorization: Bearer ...':\n\n"
            "    export SQT_MCP_TOKEN=$(openssl rand -base64 32)\n\n"
            "If something in front of this server already authenticates callers "
            "(an OAuth proxy, mTLS, a service mesh), pass --no-auth to say so "
            "deliberately."
        )

    loopback = is_loopback(args.host)
    if not loopback and not args.allow_host:
        raise SystemExit(
            f"sqt-mcp: binding to {args.host} needs at least one --allow-host. "
            "The Host header check is what stops a page in somebody's browser "
            "from reaching this port, and it cannot be derived from a wildcard "
            "bind address. Pass the names clients will actually use, for "
            f"example: --allow-host sqt.internal:{args.port}"
        )
    if not loopback and args.no_auth:
        warnings.append(
            f"serving UNAUTHENTICATED on {args.host}:{args.port}. Every caller "
            "that can route to this port can run every exposed tool and read "
            "every stored result. Only correct if something in front of this "
            "server authenticates."
        )

    return {
        "transport": "http",
        "host": args.host,
        "port": int(args.port),
        "path": path,
        "stateless": bool(args.stateless),
        "json_response": bool(args.json_response),
        "auth_token": token,
        "allowed_hosts": tuple(args.allow_host),
        "allowed_origins": tuple(args.allow_origin),
    }


def _resolve_scope(
    runtime_arg: Optional[str], category_arg: Optional[str]
) -> Tuple[List[str], List[str]]:
    """
    Turn --runtime and --categories into the (runtimes, categories) pair the
    server is scoped by.

    The two flags are not alternatives, they are nested: a runtime owns
    categories, so --runtime is the outer scope and --categories narrows
    within it. That makes four cases, and the only interesting ones are the
    two where a default would otherwise say something the caller did not
    mean:

    - **Neither given.** DEFAULT_CATEGORIES, every runtime. Exactly what
      this server did before --runtime existed; adding a flag must not
      change what happens when nobody passes it.
    - **--runtime only.** ALL of that runtime's categories. If --categories
      kept its old default here, `--runtime backtest` would serve nothing at
      all -- the default categories belong to research and meta -- and the
      failure would look like an empty server rather than like a flag
      disagreeing with itself.
    - **--categories only.** As given, across every runtime. The runtimes
      are then derived from the categories, so the dispatch scope still
      matches what is listed rather than silently staying wider.
    - **Both.** Intersected, and REFUSED if a named category is not owned by
      a named runtime. Serving the intersection silently would hand back a
      surface neither flag describes.
    """
    runtimes = _split_runtimes(runtime_arg) if runtime_arg is not None else None
    categories = _split_categories(category_arg) if category_arg is not None else None

    if runtimes is not None:
        unknown = [r for r in runtimes if r not in RUNTIME_CATEGORY_MAP]
        if unknown:
            raise ValueError(
                f"unknown runtime{'s' if len(unknown) > 1 else ''} {unknown}. "
                f"Available: {', '.join(ALL_RUNTIMES)}."
            )
        if not runtimes:
            raise ValueError("--runtime selected nothing")

    if categories is not None:
        unknown = [c for c in categories if c not in ALL_CATEGORIES]
        if unknown:
            raise ValueError(
                f"unknown categor{'y' if len(unknown) == 1 else 'ies'} "
                f"{unknown}. Available: {', '.join(ALL_CATEGORIES)}."
            )
        if not categories:
            raise ValueError("--categories selected nothing")

    if runtimes is None and categories is None:
        return list(ALL_RUNTIMES), list(DEFAULT_CATEGORIES)

    if runtimes is not None and categories is None:
        return runtimes, list(categories_for_runtimes(runtimes))

    if runtimes is None and categories is not None:
        owners = _owners_of(categories)
        return owners, categories

    # Both given: the categories must live inside the runtimes.
    allowed = set(categories_for_runtimes(runtimes))
    stray = [c for c in categories if c not in allowed]
    if stray:
        detail = "; ".join(f"{c!r} belongs to {_owner_of_category(c)!r}" for c in stray)
        raise ValueError(
            f"--categories {stray} not served by --runtime "
            f"{'+'.join(runtimes)} ({detail}). Either drop "
            f"{'them' if len(stray) > 1 else 'it'}, or widen --runtime "
            "deliberately -- serving the intersection would give you a "
            "surface neither flag describes."
        )
    return runtimes, categories


def _owner_of_category(category: str) -> str:
    for runtime, cats in RUNTIME_CATEGORY_MAP.items():
        if category in cats:
            return runtime
    return "unknown"


def _owners_of(categories: Sequence[str]) -> List[str]:
    """The runtimes that own the given categories, in ALL_RUNTIMES order."""
    owners = {_owner_of_category(c) for c in categories}
    return [r for r in ALL_RUNTIMES if r in owners]


def resolve(argv: Optional[Sequence[str]] = None) -> ServerConfig:
    """Parse arguments, resolve the environment, and fail fast if it is wrong."""
    args = build_parser().parse_args(argv)

    try:
        runtimes, categories = _resolve_scope(args.runtime, args.categories)
    except ValueError as exc:
        raise SystemExit(f"sqt-mcp: {exc}") from None
    if args.inline_limit < 0:
        raise SystemExit("sqt-mcp: --inline-limit must be >= 0")
    if args.heartbeat < 0:
        raise SystemExit("sqt-mcp: --heartbeat must be >= 0")

    warnings: List[str] = []
    transport = _resolve_transport(args, warnings)
    dirs = {
        attr: _resolve_dir(env_var, purpose, warnings, transport["transport"])
        for attr, env_var, purpose in _ENV_DIRS
    }

    return ServerConfig(
        categories=tuple(categories),
        runtimes=tuple(runtimes),
        inline_limit_bytes=args.inline_limit,
        include_output_schemas=bool(args.output_schemas),
        tool_detail=args.tool_detail,
        detail_budget=args.detail_budget,
        enable_long_running=bool(args.enable_long_running),
        heartbeat_seconds=float(args.heartbeat),
        warnings=tuple(warnings),
        **transport,
        **dirs,
    )


def check_context_budget(
    config: "ServerConfig", schema_bytes_total: int
) -> Optional[str]:
    """
    A warning when this configuration costs more context than it should.

    Returns None when it fits, and ALWAYS when `CONTEXT_CEILING_BYTES` is
    None -- which is the default. The suggestion is specific on purpose:
    the fix is almost always one flag, and a warning that says "too big"
    without saying what to do about it just moves the problem to the
    reader.
    """
    if CONTEXT_CEILING_BYTES is None:
        return None
    if schema_bytes_total < CONTEXT_CEILING_BYTES:
        return None
    over = schema_bytes_total - CONTEXT_CEILING_BYTES
    remedy = (
        "narrow with --runtime, or lower --detail-budget"
        if config.tool_detail == "auto"
        else "serve one runtime with --runtime, or add --tool-detail auto"
    )
    return (
        f"this configuration costs {schema_bytes_total:,} bytes of context "
        f"(~{schema_bytes_total // 4:,} tokens), {over:,} over the "
        f"{CONTEXT_CEILING_BYTES:,} budget. Every byte is held for the whole "
        f"session, before the agent has done anything. To fix: {remedy}."
    )


def report(config: ServerConfig, tool_count: int, schema_bytes_total: int) -> None:
    """Say what was configured, once, on stderr."""
    lines = [
        "sqt-mcp starting",
        f"  transport         : {'streamable-http' if config.transport == 'http' else 'stdio'}",
    ]
    if config.transport == "http":
        # A wildcard bind is not a name a client can dial, so show one that is.
        display_host = "127.0.0.1" if config.host in ("0.0.0.0", "::") else config.host
        auth = (
            f"Bearer token from {TOKEN_ENV_VAR}"
            if config.auth_token
            else "NONE (--no-auth)"
        )
        hosts = (
            ", ".join(config.allowed_hosts)
            if config.allowed_hosts
            else "loopback defaults"
        )
        origins = (
            ", ".join(config.allowed_origins)
            if config.allowed_origins
            else "(none; browser clients rejected)"
        )
        lines += [
            f"  bind              : {config.host}:{config.port}",
            f"  endpoint          : http://{display_host}:{config.port}{config.path}",
            f"  authorization     : {auth}",
            f"  sessions          : {'stateless' if config.stateless else 'stateful'}",
            f"  responses         : {'json' if config.json_response else 'sse stream'}",
            f"  allowed hosts     : {hosts}",
            f"  allowed origins   : {origins}",
        ]
    lines += [
        f"  runtimes          : {'+'.join(config.runtimes)}",
        f"  categories        : {', '.join(config.categories)}",
        f"  tools exposed     : {tool_count}",
        f"  context cost      : {schema_bytes_total / 1024:.1f} KB "
        f"(~{schema_bytes_total // 4:,} tokens at connect)",
        f"  tool detail       : {config.tool_detail}"
        + (
            f" (budget {config.detail_budget:,} B)"
            if config.tool_detail == "auto"
            else ""
        ),
        f"  output schemas    : {'declared' if config.include_output_schemas else 'omitted (structuredContent still sent)'}",
        f"  inline limit      : {config.inline_limit_bytes} bytes",
        f"  long-running tools: {'enabled' if config.enable_long_running else 'hidden'}",
        f"  heartbeat: "
        + (
            f"every {config.heartbeat_seconds:g}s when a client asks"
            if config.heartbeat_seconds > 0
            else "off"
        ),
    ]
    for attr, env_var, _purpose in _ENV_DIRS:
        value = getattr(config, attr)
        lines.append(f"  {env_var:<18}: {value if value else '(unset)'}")
    if config.transport == "http":
        # Worth saying to anyone moving over from stdio, where each client got
        # its own process and therefore its own store.
        lines.append(
            "  NOTE: one instance, one artifact store, one audit trail. Every "
            "connected client can read every sqt:// result any of them produced."
        )
    budget_warning = check_context_budget(config, schema_bytes_total)
    if budget_warning:
        lines.append(f"  WARNING: {budget_warning}")
    for warning in config.warnings:
        lines.append(f"  WARNING: {warning}")
    print("\n".join(lines), file=sys.stderr, flush=True)
