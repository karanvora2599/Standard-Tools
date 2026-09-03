"""
Liveness for tools that take minutes, without inventing a percentage.

THE PROBLEM THIS SOLVES. `--enable-long-running` exposes `scan_pairs` and
`run_backtest_optimization`, and a walk-forward run or a pair scan is
minutes of silence. To the client that is indistinguishable from a hung
server: nothing arrives, no reason is given, and the usual response is a
timeout that abandons work which was going to finish. The tools were
reachable and not really usable.

WHY `total` IS ALWAYS None. The server dispatches an opaque synchronous
call into the library and genuinely does not know how far through it is --
there is no progress hook, and inventing one would mean every tool
reporting its own. A notification carrying `progress` and no `total` is
exactly how the protocol says "still working, completion unknown", and a
client renders it as an indeterminate spinner. Supplying a fabricated
percentage would render as a bar that lies, and a bar that reaches 90% and
stops is worse than no bar at all -- it converts "I do not know" into a
specific false claim.

WHAT `progress` CARRIES. Elapsed seconds. The protocol requires the value
to increase on every notification for a token, and elapsed time is
monotonic by construction, which no fraction-of-work estimate would be:
a revised estimate can go backwards and would violate the spec.

ONLY WHEN ASKED. Notifications go out only if the client supplied a
`progressToken` on the request. Sending them unrequested is protocol noise
aimed at a client that has not said it can display them.

WHY THIS WORKS AT ALL. `call_tool` runs the library on a worker thread
precisely so one backtest cannot block the event loop. That same decision
is what leaves the loop free to emit these while the work proceeds -- the
heartbeat is a task in the loop, not a callback from the tool.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

import anyio

#: Seconds between heartbeats. Short enough that a client's idle timeout
#: never fires between two of them, long enough that a tool taking minutes
#: does not fill the transcript. Not configurable per tool: the server has
#: no basis for treating one opaque call differently from another.
DEFAULT_HEARTBEAT_SECONDS = 5.0

#: Below this a heartbeat would fire more often than most transports flush,
#: which is noise rather than liveness.
MIN_HEARTBEAT_SECONDS = 0.05


def progress_token(ctx: Any) -> Optional[Any]:
    """
    The token the client attached, or None if it asked for no updates.

    `meta` is an open mapping and may be absent entirely, so this reads
    defensively rather than assuming the shape -- a malformed `_meta` from
    some other client must not take down a tool call.
    """
    meta = getattr(ctx, "meta", None)
    if not meta:
        return None
    try:
        return meta.get("progress_token")
    except AttributeError:  # pragma: no cover - non-mapping _meta
        return None


async def _pulse(
    ctx: Any,
    token: Any,
    label: str,
    interval: float,
    started: float,
) -> None:
    """Emit elapsed-time notifications until cancelled."""
    while True:
        await anyio.sleep(interval)
        elapsed = time.monotonic() - started
        try:
            await ctx.session.send_progress_notification(
                progress_token=token,
                progress=elapsed,
                total=None,
                message=f"{label}: running, {elapsed:.0f}s elapsed",
                related_request_id=getattr(ctx, "request_id", None),
            )
        except Exception:
            # A heartbeat is a courtesy. If the client has gone away or the
            # stream is closed, the TOOL must still be allowed to finish and
            # return -- failing the call because its progress note could not
            # be delivered would turn a cosmetic problem into a lost result.
            return


@asynccontextmanager
async def report_liveness(
    ctx: Any,
    label: str,
    *,
    interval: float = DEFAULT_HEARTBEAT_SECONDS,
) -> AsyncIterator[None]:
    """
    Heartbeat for the duration of the block, if the client asked for one.

    Yields immediately either way: a tool call must never wait on, or fail
    because of, its own progress reporting.
    """
    token = progress_token(ctx)
    session = getattr(ctx, "session", None)
    # interval <= 0 is the operator turning heartbeats OFF (--heartbeat 0).
    # Clamping it up to the floor instead would produce the fastest possible
    # heartbeat from the flag that asks for none.
    if token is None or session is None or float(interval) <= 0:
        yield
        return

    interval = max(float(interval), MIN_HEARTBEAT_SECONDS)
    started = time.monotonic()
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(_pulse, ctx, token, label, interval, started)
            try:
                yield
            finally:
                # Stops the heartbeat whether the tool returned or raised.
                # Left running it would report a call that has already
                # answered.
                group.cancel_scope.cancel()
    except BaseExceptionGroup as bundled:
        # A task group re-raises anything that escaped the block WRAPPED in
        # an ExceptionGroup, and that is not a detail callers can be asked
        # to know. `call_tool` catches QuantError to return the library's
        # own self-correcting message verbatim; wrapped, that except clause
        # stops matching and every such message collapses into
        # "ExceptionGroup: unhandled errors in a TaskGroup" -- for exactly
        # the clients that asked for progress, and no others. Adding a
        # heartbeat must not change what a block raises.
        if len(bundled.exceptions) == 1:
            raise bundled.exceptions[0] from None
        raise


__all__ = [
    "DEFAULT_HEARTBEAT_SECONDS",
    "MIN_HEARTBEAT_SECONDS",
    "progress_token",
    "report_liveness",
]
