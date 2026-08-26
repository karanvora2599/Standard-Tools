"""
The HTTP transport: what it refuses, and what it serves.

TWO KINDS OF TEST, AND WHY THE SPLIT. Everything about configuration is
checked in-process, because every one of those refusals is a decision made
before a socket exists and asserting it through a running server would only
make the failures slower and less specific. Everything about the wire is
checked by driving the ASGI app directly with hand-built scopes: a 401 is a
status code and three headers, and that is exactly the level a client sees
it at.

The one thing neither of those can catch is the server failing to come up
at all as a subprocess, so `TestHttpIntegration` starts one and connects a
real client, mirroring `test_mcp_stdio_session.py`.

WHAT IS BEING PROTECTED. Over stdio the only caller is the process that
launched the server. Over HTTP it is anything that can route to the port,
and the gap between those two sentences is the entire subject of this file.
No network beyond loopback, no API key, no market data.
"""

from __future__ import annotations

import json
import os
import socket
import sys

import pytest

from standard_quant_tools.mcp.config import (
    DEFAULT_HTTP_PATH,
    DEFAULT_HTTP_PORT,
    TOKEN_ENV_VAR,
    is_loopback,
    report,
    resolve,
)

TOKEN = "test-token-not-a-real-secret"


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch):
    """A token in the developer's shell must not decide what these assert."""
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)


# --------------------------------------------------------------------------
# Configuration: every refusal happens before the port opens
# --------------------------------------------------------------------------


class TestHttpConfig:
    def test_stdio_is_still_the_default(self):
        config = resolve([])
        assert config.transport == "stdio"
        assert config.auth_token is None

    def test_http_without_a_token_refuses_to_start(self):
        with pytest.raises(SystemExit) as exc:
            resolve(["--transport", "http"])
        message = str(exc.value)
        assert TOKEN_ENV_VAR in message
        assert "--no-auth" in message, "the refusal must name the deliberate way out"

    def test_http_with_a_token_resolves(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)
        config = resolve(["--transport", "http"])
        assert config.transport == "http"
        assert config.auth_token == TOKEN
        assert config.host == "127.0.0.1"
        assert config.port == DEFAULT_HTTP_PORT
        assert config.path == DEFAULT_HTTP_PATH

    def test_no_auth_is_allowed_on_loopback_without_alarm(self):
        """Unauthenticated on loopback is a choice. Unauthenticated on a network is not."""
        config = resolve(["--transport", "http", "--no-auth"])
        assert config.auth_token is None
        assert not any("UNAUTHENTICATED" in w for w in config.warnings)

    def test_token_and_no_auth_together_is_refused(self, monkeypatch):
        """Ambiguity here is the one mistake that fails open, so it fails loudly."""
        monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)
        with pytest.raises(SystemExit) as exc:
            resolve(["--transport", "http", "--no-auth"])
        assert "opposite things" in str(exc.value)

    def test_public_bind_requires_an_allowed_host(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)
        with pytest.raises(SystemExit) as exc:
            resolve(["--transport", "http", "--host", "0.0.0.0"])
        assert "--allow-host" in str(exc.value)

    def test_public_bind_with_a_host_and_a_token_resolves(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)
        config = resolve(
            [
                "--transport",
                "http",
                "--host",
                "0.0.0.0",
                "--allow-host",
                "sqt.internal:8765",
                "--allow-origin",
                "https://desk.internal",
            ]
        )
        assert config.allowed_hosts == ("sqt.internal:8765",)
        assert config.allowed_origins == ("https://desk.internal",)

    def test_unauthenticated_public_bind_is_allowed_but_shouted_about(self):
        config = resolve(
            [
                "--transport",
                "http",
                "--host",
                "0.0.0.0",
                "--allow-host",
                "sqt.internal:8765",
                "--no-auth",
            ]
        )
        assert config.auth_token is None
        assert any("UNAUTHENTICATED" in w for w in config.warnings)

    def test_stdio_says_so_when_handed_http_options(self):
        config = resolve(["--port", "9000"])
        assert config.transport == "stdio"
        assert any("transport is stdio" in w for w in config.warnings)

    def test_stdio_says_so_when_a_token_is_set(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)
        config = resolve([])
        assert config.auth_token is None
        assert any(TOKEN_ENV_VAR in w for w in config.warnings)

    @pytest.mark.parametrize("port", ["0", "70000", "-1"])
    def test_port_must_be_a_real_port(self, monkeypatch, port):
        monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)
        with pytest.raises(SystemExit):
            resolve(["--transport", "http", "--port", port])

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("/mcp", "/mcp"),
            ("mcp", "/mcp"),
            ("/mcp/", "/mcp"),
            ("/", "/"),
            ("/a/b/", "/a/b"),
        ],
    )
    def test_path_is_normalised(self, monkeypatch, given, expected):
        monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)
        assert resolve(["--transport", "http", "--path", given]).path == expected

    def test_report_never_prints_the_token(self, monkeypatch, capsys):
        monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)
        config = resolve(["--transport", "http"])
        report(config, 6, 12000)
        printed = capsys.readouterr().err
        assert (
            TOKEN not in printed
        ), "the startup report is the likeliest place to leak it"
        assert TOKEN_ENV_VAR in printed
        assert "streamable-http" in printed

    def test_report_says_the_store_is_now_shared(self, monkeypatch, capsys):
        """A stdio user got a store per client. Moving to HTTP silently changes that."""
        monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)
        report(resolve(["--transport", "http"]), 6, 12000)
        assert "one artifact store" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "host,loopback",
        [
            ("127.0.0.1", True),
            ("localhost", True),
            ("::1", True),
            ("[::1]", True),
            ("0.0.0.0", False),
            ("10.0.0.4", False),
            ("sqt.internal", False),
        ],
    )
    def test_is_loopback(self, host, loopback):
        assert is_loopback(host) is loopback


# --------------------------------------------------------------------------
# The wire
# --------------------------------------------------------------------------

anyio = pytest.importorskip("anyio")
pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from standard_quant_tools.mcp.http import (  # noqa: E402
    BearerAuth,
    build_app,
    default_allowed_hosts,
)
from standard_quant_tools.mcp.server import build_server  # noqa: E402


async def _request(app, method, path, headers, body):
    """One HTTP round trip against an ASGI app, collapsed to (status, headers, body)."""
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": [
            (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
        ],
        "client": ("127.0.0.1", 54321),
        "server": ("127.0.0.1", DEFAULT_HTTP_PORT),
    }

    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    out = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return (
        start["status"],
        {k.decode(): v.decode() for k, v in start.get("headers", [])},
        out,
    )


def call_asgi(app, method="POST", path="/mcp", headers=None, body=b"{}"):
    """Drive an ASGI app once, with no lifespan.

    A hand-built scope rather than a test client: the assertions here are
    about status codes and headers, which is the level a client sees it at,
    and it keeps the tests free of an HTTP client dependency the package
    does not otherwise need.
    """

    async def go():
        return await _request(app, method, path, headers, body)

    return anyio.run(go)


def call_asgi_live(app, method="POST", path="/mcp", headers=None, body=b"{}"):
    """The same, with the app's lifespan running around it.

    Needed for anything that reaches the transport itself: the session
    manager's task group is started by the lifespan, and without it the
    endpoint raises "Task group is not initialized" instead of answering.
    Which is also worth pinning, since that is what a forgotten
    `session_manager.run()` would look like in production.
    """
    result: dict = {}

    async def go():
        send_events, receive_events = anyio.create_memory_object_stream(8)
        seen: list[dict] = []

        async def lifespan_receive():
            return await receive_events.receive()

        async def lifespan_send(message):
            seen.append(message)

        async def wait_for(event_type: str) -> None:
            with anyio.fail_after(30):
                while not any(m["type"] == event_type for m in seen):
                    await anyio.sleep(0.01)

        scope = {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}}
        async with anyio.create_task_group() as tg:
            tg.start_soon(app, scope, lifespan_receive, lifespan_send)
            await send_events.send({"type": "lifespan.startup"})
            await wait_for("lifespan.startup.complete")
            result["reply"] = await _request(app, method, path, headers, body)
            await send_events.send({"type": "lifespan.shutdown"})
            await wait_for("lifespan.shutdown.complete")

    anyio.run(go)
    return result["reply"]


class Inner:
    """A stand-in for the MCP endpoint, so auth is tested without the protocol."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


class TestBearerAuth:
    def test_a_request_with_no_header_is_rejected(self):
        inner = Inner()
        status, headers, body = call_asgi(BearerAuth(inner, TOKEN))
        assert status == 401
        assert headers["www-authenticate"].startswith("Bearer ")
        assert json.loads(body)["error"] == "unauthorized"
        assert inner.calls == 0, "an unauthenticated request must not reach the tools"

    @pytest.mark.parametrize(
        "header",
        [
            "Bearer wrong",
            "Bearer ",
            "bearer",
            TOKEN,
            f"Basic {TOKEN}",
            f"Bearer {TOKEN}x",
        ],
    )
    def test_a_wrong_credential_is_rejected(self, header):
        inner = Inner()
        status, _h, _b = call_asgi(
            BearerAuth(inner, TOKEN), headers={"Authorization": header}
        )
        assert status == 401
        assert inner.calls == 0

    @pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
    def test_the_right_token_passes_whatever_the_scheme_case(self, scheme):
        inner = Inner()
        status, _h, _b = call_asgi(
            BearerAuth(inner, TOKEN), headers={"Authorization": f"{scheme} {TOKEN}"}
        )
        assert status == 204
        assert inner.calls == 1

    def test_the_401_says_how_to_authenticate(self):
        _s, _h, body = call_asgi(BearerAuth(Inner(), TOKEN))
        detail = json.loads(body)["detail"]
        assert TOKEN_ENV_VAR in detail
        assert TOKEN not in detail


class TestAllowedHosts:
    def test_loopback_gets_sensible_defaults(self):
        hosts = default_allowed_hosts("127.0.0.1", 8765)
        assert "127.0.0.1:8765" in hosts
        assert "localhost:8765" in hosts

    def test_a_public_bind_gets_none(self):
        """Guessing here produces a server that 421s every request for no stated reason."""
        assert default_allowed_hosts("0.0.0.0", 8765) == []


class TestApp:
    def _app(self, monkeypatch, *args):
        monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)
        config = resolve(["--transport", "http", "--categories", "screener", *args])
        server, handlers = build_server(config)
        return build_app(config, server, len(handlers.tools)), config

    def test_health_needs_no_token(self, monkeypatch):
        app, _config = self._app(monkeypatch)
        status, _h, body = call_asgi(app, method="GET", path="/healthz")
        assert status == 200
        payload = json.loads(body)
        assert payload["ok"] is True
        assert payload["transport"] == "streamable-http"
        assert payload["tools"] > 0
        assert TOKEN not in body.decode()

    def test_the_endpoint_is_an_exact_path_not_a_redirect(self, monkeypatch):
        """A Mount would answer the advertised URL with a 307 on a POST with a body."""
        app, _config = self._app(monkeypatch)
        status, _h, _b = call_asgi(app, method="POST", path="/mcp")
        assert status == 401, f"expected the auth gate, got {status}"

    def test_a_custom_path_is_where_the_endpoint_lands(self, monkeypatch):
        app, config = self._app(monkeypatch, "--path", "/desk/mcp")
        assert config.path == "/desk/mcp"
        assert call_asgi(app, path="/desk/mcp")[0] == 401
        assert call_asgi(app, path="/mcp")[0] == 404

    def test_an_unsupported_method_is_refused(self, monkeypatch):
        app, _config = self._app(monkeypatch)
        assert call_asgi(app, method="PUT", path="/mcp")[0] == 405

    def test_no_auth_leaves_the_endpoint_open(self, monkeypatch):
        """The flag has to actually remove the gate, or --no-auth is theatre."""
        monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
        config = resolve(
            ["--transport", "http", "--no-auth", "--categories", "screener"]
        )
        server, handlers = build_server(config)
        app = build_app(config, server, len(handlers.tools))
        # Reaches the transport rather than the auth gate. The transport then
        # rejects this particular POST for its own reasons (no session, not an
        # initialize), which is the point: the rejection is no longer a 401.
        status, _headers, _body = call_asgi_live(app, method="POST", path="/mcp")
        assert status != 401

    def test_the_lifespan_starts_the_session_manager(self, monkeypatch):
        """Without it every request raises instead of answering."""
        app, _config = self._app(monkeypatch)
        status, _headers, _body = call_asgi_live(
            app,
            method="POST",
            path="/mcp",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert 400 <= status < 500, "an unsessioned POST is refused, not crashed"


# --------------------------------------------------------------------------
# It actually serves
# --------------------------------------------------------------------------


@pytest.mark.integration
class TestHttpIntegration:
    """Start the server as a subprocess and connect a real client to it.

    The one failure the in-process tests cannot see: the process coming up,
    binding, and speaking the protocol on a socket.
    """

    def test_a_real_client_can_connect_and_list_tools(self, tmp_path):
        pytest.importorskip("mcp.client.streamable_http")
        import subprocess

        from mcp import ClientSession
        from mcp.client.streamable_http import (
            create_mcp_http_client,
            streamable_http_client,
        )

        from .. import REPO_ROOT

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        env = dict(os.environ)
        env.update(
            {
                "SQT_RUNS_DIR": str(tmp_path / "runs"),
                "SQT_AUDIT_DIR": str(tmp_path / "audit"),
                "SQT_CACHE_DIR": str(tmp_path / "cache"),
                "SQT_AUDIT_ENABLED": "0",
                "PYTHONPATH": str(REPO_ROOT / "src"),
                TOKEN_ENV_VAR: TOKEN,
            }
        )
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "standard_quant_tools.mcp.server",
                "--transport",
                "http",
                "--port",
                str(port),
                "--categories",
                "screener",
            ],
            env=env,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            self._wait_for_port(proc, port)

            async def go():
                client = create_mcp_http_client(
                    headers={"Authorization": f"Bearer {TOKEN}"}
                )
                url = f"http://127.0.0.1:{port}/mcp"
                async with streamable_http_client(url, http_client=client) as (
                    read,
                    write,
                ):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        return await session.list_tools()

            tools = anyio.run(go)
            assert tools.tools, "a connected client must see the screener category"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:  # pragma: no cover - teardown only
                proc.kill()

    @staticmethod
    def _wait_for_port(proc, port, timeout=120.0):
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout else ""
                raise AssertionError(f"server exited before binding:\n{output}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                    return
            except OSError:
                time.sleep(0.25)
        raise AssertionError("server never bound its port")
