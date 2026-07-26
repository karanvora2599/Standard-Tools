"""Per-call request context: the `request_id`/in-flight `data_sources` list
threaded through a `dispatch()` call via `contextvars` (so it survives the
thread-pool hop in async data fetches), plus the opt-in correlated-logging
helper that reads `request_id` back out of that same context."""

import contextvars
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_request_id_var: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "sqt_request_id", default=None
)
_data_sources_var: "contextvars.ContextVar[Optional[List[Dict[str, Any]]]]" = (
    contextvars.ContextVar("sqt_data_sources", default=None)
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


def record_data_access(
    symbol: str,
    start: str,
    end: str,
    interval: str,
    source: str,
    content_hash: str,
) -> None:
    """
    Report an OHLCV pull into the currently-open decision record, if any.
    No-op when no decision record is in progress (e.g. calling a data
    provider directly outside of `dispatch()`).
    """
    sources = _data_sources_var.get()
    if sources is None:
        return
    sources.append(
        {
            "symbol": symbol,
            "start": start,
            "end": end,
            "interval": interval,
            "source": source,
            "content_hash": content_hash,
        }
    )
