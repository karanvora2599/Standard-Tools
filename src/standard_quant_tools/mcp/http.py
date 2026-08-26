"""
Streamable HTTP transport: the same server, reachable from another machine.

WHY THIS EXISTS. Over stdio the client owns the server. It picks the
working directory, the environment and the lifetime, and it has to be on
the same machine because it launches the process itself. That is the right
shape for a developer running an editor, and the wrong shape for a desk:
you cannot point four analysts, a scheduler and an internal agent at one
configured instance, because there is no instance, only a command line each
of them runs separately.

Over HTTP the server is a service. It is configured once by whoever runs
it, it holds one artifact store and one audit trail rather than one per
client, and anything that can speak the protocol can connect, including
clients on other hosts and agents that are not editors at all.

WHAT THIS FILE IS ALLOWED TO DO. Transport only. It does not touch tools,
schemas, resources or prompts; it takes an already-built `Server` and puts
it behind a socket. The rule `server.py` sets for itself, that a surface
onto the registries must contain no logic of its own, applies here for the
same reason: this is the sixth way to reach the same two registries and it
must not become a place where behaviour can diverge.

THE THREE THINGS A NETWORK ADDS, AND WHAT IS DONE ABOUT EACH.

1. Anyone who can route to the port can call the tools. A shared bearer
   token is required by default, and refused as a default: the server will
   not start on a non-loopback address without either a token or an
   explicit `--no-auth`. A static token is not an authorization server, and
   this file does not pretend otherwise. It is the right primitive when
   something else already terminates identity (an OAuth proxy, mTLS, a
   service mesh) and the wrong one when nothing does; the SDK ships
   `mcp.server.auth` for the latter, and wiring it needs an issuer and a
   client registry, which is a decision rather than a default.

2. A browser on the user's machine can reach a loopback port that a remote
   attacker cannot, which is what DNS rebinding exploits. The SDK's Host
   and Origin checks are switched on here, with the allowed Host values
   derived from the address actually bound. A non-browser client sends no
   Origin header and is unaffected.

3. The artifact store is now shared. `sqt://` URIs resolve against
   SQT_RUNS_DIR for every connected client, so a result link handed to one
   client is readable by all of them. That is a property of running one
   instance for a team and is called out in `report()`, because a stdio
   user who moves to HTTP has not been told otherwise.
"""

from __future__ import annotations

import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, List

try:
    import uvicorn
    from mcp.server import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from starlette.types import Receive, Scope, Send
except ModuleNotFoundError as exc:  # pragma: no cover - install-shape error
    raise ModuleNotFoundError(
        "The HTTP transport needs the Model Context Protocol SDK and its "
        "server extras, which are not dependencies of the core package. "
        "Install them with:\n\n"
        "    pip install 'standard_quant_tools[mcp-http]'\n\n"
        f"(original error: {exc})"
    ) from exc

from standard_quant_tools import __version__ as _sqt_version
from standard_quant_tools.mcp.config import ServerConfig, is_loopback

log = logging.getLogger("sqt.mcp.http")


def default_allowed_hosts(host: str, port: int) -> List[str]:
    """Host header values to accept when the operator named none.

    Only meaningful for a loopback bind, where the set of names that can
    reach the port is known. On any other address the operator is asked for
    the hostnames instead, because guessing produces a server that returns
    421 to every request for a reason nothing explains.
    """
    if not is_loopback(host):
        return []
    return [
        f"127.0.0.1:{port}",
        f"localhost:{port}",
        f"[::1]:{port}",
        "127.0.0.1",
        "localhost",
    ]


class MCPEndpoint:
    """The MCP transport as an ASGI app, mounted at an exact path.

    A class and not a function on purpose. Starlette wraps a function
    endpoint in its request/response machinery, and a `Mount` would answer
    the advertised URL with a 307 to the same path plus a slash. Redirecting
    a POST that carries a JSON-RPC body is the kind of thing that works in
    curl, works in most clients, and fails in one of them on the day it
    matters. Routing the exact path to a raw ASGI app avoids the question.
    """

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        self.session_manager = session_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.session_manager.handle_request(scope, receive, send)


class BearerAuth:
    """Shared-secret gate in front of the MCP endpoint.

    Compared with `hmac.compare_digest` rather than `==`: the difference is
    only theoretically exploitable over a network, and writing the
    vulnerable version in a file about exposing a server would be a poor
    advertisement for the rest of it.

    Wraps only the MCP mount. The health route stays open so a load
    balancer does not need the token to decide whether the process is
    alive, and it reports nothing an unauthenticated caller could not learn
    by watching the port answer.
    """

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        provided = ""
        for key, value in scope.get("headers") or ():
            if key == b"authorization":
                provided = value.decode("latin-1")
                break

        scheme, _, presented = provided.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            presented.strip(), self.token
        ):
            log.warning(
                "rejected unauthenticated %s %s from %s",
                scope.get("method"),
                scope.get("path"),
                (scope.get("client") or ("unknown", 0))[0],
            )
            body = json.dumps(
                {
                    "error": "unauthorized",
                    "detail": (
                        "This server requires a bearer token. Send "
                        "'Authorization: Bearer <token>' with the value of "
                        "SQT_MCP_TOKEN configured on the server."
                    ),
                }
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                        (b"www-authenticate", b'Bearer realm="standard-quant-tools"'),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)


def build_app(
    config: ServerConfig,
    mcp_server: "Server[Any]",
    tool_count: int,
) -> Starlette:
    """The ASGI application: one MCP endpoint and one health route.

    Exposed separately from `serve_http` so it can be mounted inside a
    larger application, put behind a real ASGI server, or exercised by a
    test client without binding a port.
    """
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(config.allowed_hosts)
        or default_allowed_hosts(config.host, config.port),
        allowed_origins=list(config.allowed_origins),
    )

    session_manager = StreamableHTTPSessionManager(
        app=mcp_server,
        # No event store: resumability means replaying a stream a client
        # dropped, which needs somewhere durable to keep every event. That
        # is a real feature and a real storage decision, and shipping an
        # in-memory one would make it look handled while losing everything
        # on the restart that most often causes the disconnect.
        event_store=None,
        json_response=config.json_response,
        stateless=config.stateless,
        security_settings=security,
    )

    endpoint: Any = MCPEndpoint(session_manager)
    if config.auth_token:
        endpoint = BearerAuth(endpoint, config.auth_token)

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "server": "standard-quant-tools",
                "version": _sqt_version,
                "transport": "streamable-http",
                "stateless": config.stateless,
                "categories": list(config.categories),
                "tools": tool_count,
            }
        )

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        # The session manager's task group is what actually runs connected
        # sessions. Without this the endpoint accepts requests and then
        # fails on the first one, which reads as a protocol bug.
        async with session_manager.run():
            log.info("MCP session manager running")
            yield
            log.info("MCP session manager shutting down")

    return Starlette(
        debug=False,
        routes=[
            Route("/healthz", health, methods=["GET"]),
            # GET opens the server-to-client SSE stream, POST carries
            # JSON-RPC, DELETE ends a session. OPTIONS is deliberately absent:
            # browser clients are rejected by the Origin check by default, and
            # a 405 on a preflight is a clearer answer than a CORS policy
            # nobody configured. Mount this app inside your own if you need one.
            Route(config.path, endpoint=endpoint, methods=["GET", "POST", "DELETE"]),
        ],
        lifespan=lifespan,
    )


def serve_http(
    config: ServerConfig, mcp_server: "Server[Any]", tool_count: int
) -> None:
    """Bind the port and serve until interrupted.

    Blocking, and not run inside `anyio.run`: uvicorn owns its own event
    loop, and nesting it inside one that is already running is the usual
    way this fails.
    """
    app = build_app(config, mcp_server, tool_count)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        # Off deliberately. Uvicorn's access logger writes to stdout, and a
        # line per POST on a long-lived SSE stream is noise that buries the
        # startup report a reader actually needs.
        access_log=False,
    )
