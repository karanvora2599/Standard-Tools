"""
The decision record's id reaches whoever made the call.

`_run_and_record` minted a request id for every call and returned only
the result, so the three provenance tools that take an id could never be
given one from inside a session. The id is now readable after a dispatch
on the same thread, and the MCP server returns it as `_meta.request_id`.
"""

import json
import os
from pathlib import Path

import pytest

from standard_quant_tools import audit
from standard_quant_tools.agent.tools import dispatch


@pytest.fixture(autouse=True)
def _own_audit_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")


def _records():
    out = []
    for path in sorted(Path(os.environ["SQT_AUDIT_DIR"]).glob("*.jsonl")):
        if not audit._DAY_FILE_RE.match(path.name):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


class TestPythonDispatch:
    def test_the_last_request_id_is_the_record_just_written(self):
        assert audit.last_request_id() is None or isinstance(
            audit.last_request_id(), str
        )
        dispatch("list_strategies", {})
        request_id = audit.last_request_id()
        assert request_id
        records = _records()
        assert records[-1]["request_id"] == request_id
        assert records[-1]["tool_name"] == "list_strategies"
        dispatch("list_strategies", {})
        assert audit.last_request_id() != request_id


class TestMcpResult:
    def test_the_result_carries_the_id_in_meta(self):
        import anyio
        import mcp.types as types

        from standard_quant_tools.mcp.config import ServerConfig
        from standard_quant_tools.mcp.server import build_server

        _server, handlers = build_server(
            ServerConfig(categories=("discovery",), runtimes=("meta",))
        )

        async def call():
            return await handlers.call_tool(
                None, types.CallToolRequestParams(name="list_strategies", arguments={})
            )

        result = anyio.run(call)
        assert not result.is_error
        request_id = result.meta["request_id"]
        assert _records()[-1]["request_id"] == request_id
        # The structured payload is the tool's own result, untouched.
        assert "request_id" not in result.structured_content
