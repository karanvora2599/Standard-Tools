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
from typing import Any, Dict, List, Optional, Tuple, Union

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


#: The form of `content_hash` a data source entry records, written beside it
#: as `content_hash_version`. 1 -- an entry without the key, which is every
#: entry written before the key existed -- is `hashing.hash_dataframe`,
#: which covers how the installed pandas spells each column's dtype and
#: stores its datetimes, so the same frame hashes differently under pandas 2
#: and pandas 3. 2 is `hashing.canonical_frame_hash`, which does not depend
#: on the pandas version.
DATA_SOURCE_HASH_VERSION = 2


class _ReplaySources(list):
    """
    The data-source list a replay opens: a list like the one `dispatch()`
    opens, carrying the version-1 hash each data source of the record being
    replayed was recorded with, by (symbol, start, end, interval).

    For those sources `record_frame_access` also hashes the replayed frame
    the version-1 way, and on a miss under the other pandas's dtype names
    and datetime resolutions, so the replay compares like with like. The
    extra keys live only on this list, which no record is written from: a
    call `dispatch()`es inside the replay opens a plain list of its own.
    """

    def __init__(self, legacy: Optional[Dict[Tuple[str, str, str, str], str]] = None):
        super().__init__()
        self.legacy: Dict[Tuple[str, str, str, str], str] = dict(legacy or {})


def recording_data_access() -> bool:
    """
    Whether a data access reported now would be kept: True inside a
    `dispatch()` call (or a replay of one), False otherwise.

    A provider asks this before it digests a frame for `record_data_access`,
    because outside a decision record the digest is computed only to be
    thrown away -- and it is the expensive half of the report.
    """
    return _data_sources_var.get() is not None


def _append_data_source(
    sources: List[Dict[str, Any]],
    fields: Dict[str, Any],
    fetch_ms: Optional[float],
) -> None:
    """Append one entry with its `fetch_ms` (see `record_data_access`)."""
    clock = _fetch_clock_var.get()
    lap = clock.lap_ms() if clock is not None else None
    measured = fetch_ms if fetch_ms is not None else lap
    entry: Dict[str, Any] = dict(fields)
    if measured is not None:
        entry["fetch_ms"] = round(float(measured), 3)
    sources.append(entry)


def record_data_access(
    symbol: str,
    start: str,
    end: str,
    interval: str,
    source: str,
    content_hash: str,
    fetch_ms: Optional[float] = None,
    content_hash_version: Optional[int] = None,
) -> None:
    """
    Report an OHLCV pull into the currently-open decision record, if any.
    No-op when no decision record is in progress (e.g. calling a data
    provider directly outside of `dispatch()`).

    `content_hash` is the caller's digest of the frame, recorded as given.
    `content_hash_version` says which form it is (see
    `DATA_SOURCE_HASH_VERSION`) and is recorded beside it when given; an
    entry without it is read as version 1. A provider holding the frame
    calls `record_frame_access` instead, which hashes it the current way.

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
    fields: Dict[str, Any] = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "source": source,
        "content_hash": content_hash,
    }
    if content_hash_version is not None:
        fields["content_hash_version"] = int(content_hash_version)
    _append_data_source(sources, fields, fetch_ms)


def record_frame_access(
    symbol: str,
    start: str,
    end: str,
    interval: str,
    source: str,
    frame: Any,
    fetch_ms: Optional[float] = None,
) -> None:
    """
    Report a fetched frame into the currently-open decision record, if any,
    hashed with `hashing.canonical_frame_hash` and recorded as
    `content_hash_version` 2. No-op, and no hashing, when no decision
    record is in progress.

    Inside a replay of a record whose entry for the same (symbol, start,
    end, interval) carries a version-1 hash, the entry also carries
    `legacy_content_hash` (`hashing.hash_dataframe` of this frame, as read)
    and, when that misses the recorded value, `legacy_variant`: how
    `legacy_hash.legacy_hash_variant` reproduced it under the other
    pandas's representation, or None. Those two keys exist only on the
    replay's own list and never reach a written record.

    The digest is taken before the fetch clock's lap, as the callers of
    `record_data_access` took theirs, so `fetch_ms` covers it as before.
    """
    sources = _data_sources_var.get()
    if sources is None:
        return
    from .hashing import canonical_frame_hash

    fields: Dict[str, Any] = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "source": source,
        "content_hash": canonical_frame_hash(frame),
        "content_hash_version": DATA_SOURCE_HASH_VERSION,
    }
    expected = (
        sources.legacy.get((symbol, start, end, interval))
        if isinstance(sources, _ReplaySources)
        else None
    )
    if expected is not None:
        fields.update(_legacy_comparison(frame, expected))
    _append_data_source(sources, fields, fetch_ms)


def _legacy_comparison(frame: Any, expected: str) -> Dict[str, Any]:
    """`legacy_content_hash` and `legacy_variant` for a replayed frame
    whose record carries the version-1 hash `expected`."""
    from .hashing import hash_dataframe
    from .legacy_hash import legacy_hash_variant

    as_read = hash_dataframe(frame)
    variant: Optional[str] = None
    if as_read != expected:
        try:
            variant = legacy_hash_variant(frame, expected, as_read=as_read)
        except Exception:  # an unhashable variant is a miss, not a failed replay
            logging.getLogger(__name__).debug(
                "version-1 hash variants could not be computed", exc_info=True
            )
    return {"legacy_content_hash": as_read, "legacy_variant": variant}
