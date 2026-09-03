"""
Liveness notifications, and the four ways a progress report lies.

WHAT THESE PIN.

  1. `total` is NEVER sent. The server dispatches an opaque synchronous
     call and cannot know how far through it is. A fabricated total renders
     client-side as a progress bar, and a bar that stops at 90% is worse
     than no bar -- it converts "I do not know" into a specific false claim.
  2. `progress` strictly increases. The protocol requires it per token, and
     any fraction-of-work estimate can revise downward; elapsed time cannot.
  3. Nothing is sent unless the client supplied a progressToken.
  4. A heartbeat is a courtesy: a client that has gone away, a closed
     stream, or a malformed `_meta` must not cost the caller its result.
"""

from __future__ import annotations

from types import SimpleNamespace

import anyio
import pytest

from standard_quant_tools.mcp.progress import (
    DEFAULT_HEARTBEAT_SECONDS,
    progress_token,
    report_liveness,
)

FAST = 0.05


class _Session:
    """Records what the server tried to tell the client."""

    def __init__(self, fail: bool = False) -> None:
        self.sent = []
        self.fail = fail

    async def send_progress_notification(self, **kwargs):
        if self.fail:
            raise RuntimeError("client went away")
        self.sent.append(kwargs)


def _ctx(token="tok", session=None, request_id="req-1", meta=True):
    if meta:
        meta_value = {"progress_token": token} if token is not None else {}
    else:
        meta_value = None
    return SimpleNamespace(
        meta=meta_value,
        session=session if session is not None else _Session(),
        request_id=request_id,
    )


class TestOnlyWhenTheClientAsks:
    def test_no_token_sends_nothing(self) -> None:
        ctx = _ctx(token=None)

        async def go():
            async with report_liveness(ctx, "scan_pairs", interval=FAST):
                await anyio.sleep(FAST * 5)

        anyio.run(go)
        assert ctx.session.sent == []

    def test_absent_meta_sends_nothing(self) -> None:
        ctx = _ctx(meta=False)

        async def go():
            async with report_liveness(ctx, "scan_pairs", interval=FAST):
                await anyio.sleep(FAST * 5)

        anyio.run(go)
        assert ctx.session.sent == []

    def test_a_none_context_is_tolerated(self) -> None:
        """Every existing test calls the handler with ctx=None."""

        async def go():
            async with report_liveness(None, "scan_pairs", interval=FAST):
                await anyio.sleep(FAST)

        anyio.run(go)  # must not raise

    def test_progress_token_reads_defensively(self) -> None:
        assert progress_token(None) is None
        assert progress_token(SimpleNamespace(meta=None)) is None
        assert progress_token(SimpleNamespace(meta={})) is None
        assert progress_token(SimpleNamespace(meta={"progress_token": 7})) == 7


class TestWhatTheNotificationsSay:
    @staticmethod
    def _run(interval=FAST, hold=None):
        ctx = _ctx()

        async def go():
            async with report_liveness(ctx, "scan_pairs", interval=interval):
                await anyio.sleep(hold if hold is not None else interval * 6)

        anyio.run(go)
        return ctx

    def test_something_is_sent_while_the_work_runs(self) -> None:
        ctx = self._run()
        assert len(ctx.session.sent) >= 2

    def test_total_is_never_sent(self) -> None:
        """
        THE HONESTY INVARIANT. The server has no progress hook into the
        library, so any total would be invented.
        """
        ctx = self._run()
        assert ctx.session.sent
        assert all(note["total"] is None for note in ctx.session.sent)

    def test_progress_strictly_increases(self) -> None:
        """Required per token by the protocol; elapsed time gives it free."""
        ctx = self._run()
        values = [note["progress"] for note in ctx.session.sent]
        assert values == sorted(values)
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_the_message_names_the_tool_and_the_elapsed_time(self) -> None:
        ctx = self._run()
        message = ctx.session.sent[0]["message"]
        assert "scan_pairs" in message
        assert "elapsed" in message

    def test_the_token_and_request_id_are_carried_through(self) -> None:
        ctx = self._run()
        assert all(note["progress_token"] == "tok" for note in ctx.session.sent)
        assert all(note["related_request_id"] == "req-1" for note in ctx.session.sent)

    def test_nothing_is_sent_after_the_block_exits(self) -> None:
        """A heartbeat outliving its call reports work that already answered."""
        ctx = self._run()
        count = len(ctx.session.sent)

        async def idle():
            await anyio.sleep(FAST * 6)

        anyio.run(idle)
        assert len(ctx.session.sent) == count

    def test_a_fast_tool_reports_nothing(self) -> None:
        """The first pulse comes AFTER one interval, so a quick call is silent."""
        ctx = self._run(interval=1.0, hold=0.01)
        assert ctx.session.sent == []


class TestItNeverCostsTheCallerTheResult:
    def test_a_failing_session_does_not_break_the_block(self) -> None:
        ctx = _ctx(session=_Session(fail=True))
        reached = []

        async def go():
            async with report_liveness(ctx, "scan_pairs", interval=FAST):
                await anyio.sleep(FAST * 4)
                reached.append(True)

        anyio.run(go)
        assert reached == [True]

    def test_the_block_raises_what_it_raised_not_an_exception_group(self) -> None:
        """
        THE DEFECT THIS EXISTS TO CATCH. A task group re-raises whatever
        escaped it WRAPPED in an ExceptionGroup. `call_tool` catches
        QuantError to return the library's own self-correcting message
        verbatim -- wrapped, that except clause stops matching and every
        such message collapses to "ExceptionGroup: unhandled errors in a
        TaskGroup", for exactly the clients that asked for progress.
        """
        ctx = _ctx()

        async def go():
            async with report_liveness(ctx, "scan_pairs", interval=FAST):
                await anyio.sleep(FAST * 2)
                raise ValueError("tool failed")

        with pytest.raises(ValueError, match="tool failed"):
            anyio.run(go)
        count = len(ctx.session.sent)

        async def idle():
            await anyio.sleep(FAST * 4)

        anyio.run(idle)
        assert len(ctx.session.sent) == count


class TestTheOffSwitch:
    def test_interval_zero_disables_it(self) -> None:
        """
        `--heartbeat 0` documents itself as off. Clamping it up to the
        floor instead would make the flag asking for none produce the
        fastest possible heartbeat.
        """
        ctx = _ctx()

        async def go():
            async with report_liveness(ctx, "scan_pairs", interval=0):
                await anyio.sleep(FAST * 6)

        anyio.run(go)
        assert ctx.session.sent == []

    def test_a_negative_interval_is_off_too(self) -> None:
        ctx = _ctx()

        async def go():
            async with report_liveness(ctx, "scan_pairs", interval=-5):
                await anyio.sleep(FAST * 4)

        anyio.run(go)
        assert ctx.session.sent == []


class TestThroughTheRealHandler:
    def test_a_tool_call_with_a_token_stays_correct(self) -> None:
        """
        The heartbeat must not change what the tool returns. A real call
        with a progressToken produces the same result as one without.
        """
        import mcp.types as types

        from standard_quant_tools.mcp.config import resolve
        from standard_quant_tools.mcp.server import build_server

        config = resolve(["--runtime", "meta", "--heartbeat", "0.05"])
        _server, handlers = build_server(config)
        params = types.CallToolRequestParams(name="list_strategies", arguments={})

        async def call(ctx):
            return await handlers.call_tool(ctx, params)

        ctx = _ctx()
        with_token = anyio.run(call, ctx)
        without = anyio.run(call, None)

        assert not with_token.is_error
        assert with_token.structured_content == without.structured_content

    def test_the_config_carries_the_interval(self) -> None:
        from standard_quant_tools.mcp.config import resolve

        assert resolve(["--runtime", "meta"]).heartbeat_seconds == (
            DEFAULT_HEARTBEAT_SECONDS
        )
        assert resolve(["--runtime", "meta", "--heartbeat", "0"]).heartbeat_seconds == 0


class TestTheLibrarysErrorsSurviveTheHeartbeat:
    """
    The property the server is built around: a QuantError goes back to the
    caller verbatim, because those messages are written to be
    self-correcting. A progress token must not change that.
    """

    @staticmethod
    def _call(ctx):
        import mcp.types as types

        from standard_quant_tools.mcp.config import resolve
        from standard_quant_tools.mcp.server import build_server

        _server, handlers = build_server(
            resolve(["--runtime", "meta", "--heartbeat", "0.05"])
        )
        params = types.CallToolRequestParams(
            name="describe_tool", arguments={"name": "no_such_tool"}
        )
        return anyio.run(lambda: handlers.call_tool(ctx, params))

    def test_a_failing_tool_reports_the_same_thing_either_way(self) -> None:
        without = self._call(None).content[0].text
        with_token = self._call(_ctx()).content[0].text
        assert with_token == without
        assert "ExceptionGroup" not in with_token
        assert "TaskGroup" not in with_token

    def test_the_message_is_the_librarys_own(self) -> None:
        text = self._call(_ctx()).content[0].text
        assert "ValidationError" in text
        assert "DescribeToolInput" in text
