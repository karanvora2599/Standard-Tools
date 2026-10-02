"""Per-call request context: the `request_id`/in-flight `data_sources` list
threaded through a `dispatch()` call via `contextvars` (so it survives the
thread-pool hop in async data fetches), the clock that times each data
access, plus the opt-in correlated-logging helper that reads `request_id`
back out of that same context."""

import contextvars
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_request_id_var: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "sqt_request_id", default=None
)
_data_sources_var: "contextvars.ContextVar[Optional[List[Dict[str, Any]]]]" = (
    contextvars.ContextVar("sqt_data_sources", default=None)
)


class _FetchClock:
    """
    Where one call's timeline stands: the moment the call started, then the
    moment each of its data accesses completed.

    A provider reports a data access once it has the frame, and says
    nothing when it starts fetching, so the time a fetch took is read here
    as the time since the previous mark -- the start of the call, or the
    data access before it. Laid end to end those laps cover the call from
    its start to its last data access, which is the share of `duration_ms`
    spent getting data; what follows the last access is computation. Any
    computation a tool does BETWEEN two fetches lands in the second lap, so
    a lap is an upper bound on that fetch, exact for the usual shape of a
    tool (fetch, then compute).

    One object per call, shared by reference: a context copied into a
    worker thread copies the reference, so concurrent fetches advance the
    same clock, and the first to finish carries the wait they shared.
    """

    __slots__ = ("_mark", "_lock")

    def __init__(self, start: Optional[float] = None) -> None:
        self._mark = time.perf_counter() if start is None else start
        self._lock = threading.Lock()

    def lap_ms(self) -> float:
        """Milliseconds since the previous mark, and move the mark to now."""
        with self._lock:
            now = time.perf_counter()
            elapsed = (now - self._mark) * 1000.0
            self._mark = now
        return elapsed


_fetch_clock_var: "contextvars.ContextVar[Optional[_FetchClock]]" = (
    contextvars.ContextVar("sqt_fetch_clock", default=None)
)


def new_request_id() -> str:
    return uuid.uuid4().hex


class RequestIdFilter(logging.Filter):
    """
    Stamps `record.request_id` from the active decision-record context.

    Must be attached to a *handler*, not a logger: `Logger.filter()` only
    runs for records originating at that exact logger — records from child
    module loggers (e.g. `indicators.momentum`) reach ancestor loggers via
    `callHandlers()`, which invokes handlers directly without re-running
    ancestor `Logger.filter()`. A handler-level filter sees every record
    that reaches that handler regardless of which logger emitted it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get() or "-"
        return True


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Handler:
    """
    Opt-in helper: attach a formatted, request-id-correlated handler to the
    package logger. Never called automatically by this library.
    """
    pkg_logger = logging.getLogger("standard_quant_tools")
    pkg_logger.setLevel(level)

    handler: logging.Handler
    if log_file:
        handler = logging.FileHandler(str(log_file), encoding="utf-8")
    else:
        handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s"
        )
    )
    handler.addFilter(RequestIdFilter())
    pkg_logger.addHandler(handler)
    return handler


def recording_data_access() -> bool:
    """
    Whether a data access reported now would be kept: True inside a
    `dispatch()` call (or a replay of one), False otherwise.

    A provider asks this before it digests a frame for `record_data_access`,
    because outside a decision record the digest is computed only to be
    thrown away -- and it is the expensive half of the report.
    """
    return _data_sources_var.get() is not None


def record_data_access(
    symbol: str,
    start: str,
    end: str,
    interval: str,
    source: str,
    content_hash: str,
    fetch_ms: Optional[float] = None,
) -> None:
    """
    Report an OHLCV pull into the currently-open decision record, if any.
    No-op when no decision record is in progress (e.g. calling a data
    provider directly outside of `dispatch()`).

    The entry carries `fetch_ms`, how long the call spent getting this
    frame. A provider that timed its own fetch passes that figure; without
    one it is the lap of the call's fetch clock (see `_FetchClock`): the
    time since the call started or since its previous data access
    completed. Either way the clock moves to now, so the next lap starts
    here.
    """
    sources = _data_sources_var.get()
    if sources is None:
        return
    clock = _fetch_clock_var.get()
    lap = clock.lap_ms() if clock is not None else None
    measured = fetch_ms if fetch_ms is not None else lap
    entry: Dict[str, Any] = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "source": source,
        "content_hash": content_hash,
    }
    if measured is not None:
        entry["fetch_ms"] = round(float(measured), 3)
    sources.append(entry)
