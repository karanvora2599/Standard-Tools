"""
Decision-record audit trail for agent-tool calls.

Every call routed through `agent.tools.dispatch()` can produce an immutable
JSONL record capturing its inputs, the market data it pulled (with content
hashes), which execution path ran, and a hash of its output. `verify_replay()`
re-runs a recorded call and reports whether the data and output still match —
enough to tell a stale/tampered cache apart from a genuine code change.

Nothing here runs automatically. The package only ever attaches a
`NullHandler` by default (see `standard_quant_tools/__init__.py`); call
`configure_logging()` explicitly to see log output, and set
`SQT_AUDIT_ENABLED=0` to disable decision-record writes entirely.
"""

import contextvars
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────
# Per-call context
# ──────────────────────────────────────────────────────────────────

_request_id_var: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "sqt_request_id", default=None
)
_data_sources_var: "contextvars.ContextVar[Optional[List[Dict[str, Any]]]]" = contextvars.ContextVar(
    "sqt_data_sources", default=None
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
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s"
    ))
    handler.addFilter(RequestIdFilter())
    pkg_logger.addHandler(handler)
    return handler


# ──────────────────────────────────────────────────────────────────
# Hashing
# ──────────────────────────────────────────────────────────────────

def hash_dataframe(df: Any) -> str:
    """Content fingerprint of a DataFrame (values + index), stable across runs."""
    import numpy as np
    import pandas as pd
    hashed = pd.util.hash_pandas_object(df, index=True)
    return hashlib.sha256(np.asarray(hashed.values).tobytes()).hexdigest()[:16]


def hash_payload(obj: Any) -> str:
    """Content fingerprint of a JSON-serializable object (dict/list/scalar)."""
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────
# Data provenance
# ──────────────────────────────────────────────────────────────────

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
    sources.append({
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "source": source,
        "content_hash": content_hash,
    })


def _cpp_available() -> bool:
    try:
        import standard_quant_tools._sqt_core  # type: ignore[attr-defined]  # noqa: F401
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────
# Decision record + writer
# ──────────────────────────────────────────────────────────────────

class DecisionRecord(BaseModel):
    request_id: str
    timestamp_utc: str
    tool_name: str
    input: Dict[str, Any]
    data_sources: List[Dict[str, Any]] = Field(default_factory=list)
    cpp_available: bool
    n_workers: Optional[int] = None
    duration_ms: float
    output_hash: Optional[str] = None
    status: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None


def _audit_enabled() -> bool:
    return os.environ.get("SQT_AUDIT_ENABLED", "1").lower() not in ("0", "false", "")


def _audit_dir() -> Path:
    return Path(os.environ.get(
        "SQT_AUDIT_DIR",
        str(Path.home() / ".cache" / "standard_quant_tools" / "audit"),
    ))


class AuditWriter:
    """Append-only JSONL writer, one file per UTC day."""

    def __init__(self, audit_dir: Optional[Union[str, Path]] = None):
        self._dir = Path(audit_dir) if audit_dir else _audit_dir()

    def _path_for(self, when: datetime) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir / f"{when.strftime('%Y-%m-%d')}.jsonl"

    def write(self, record: DecisionRecord) -> Path:
        when = datetime.now(timezone.utc)
        path = self._path_for(when)
        with open(path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
        return path


def _run_and_record(tool_name: str, fn: Callable[[Any], Any], model_instance: Any) -> Dict[str, Any]:
    """
    Shared core used by `agent.tools.dispatch()`: runs `fn(model_instance)`,
    and — unless disabled via `SQT_AUDIT_ENABLED=0` — writes a DecisionRecord
    capturing inputs, data provenance, execution context, and an output hash.
    """
    request_id = new_request_id()
    token_req = _request_id_var.set(request_id)
    token_data = _data_sources_var.set([])

    t0 = time.perf_counter()
    status = "ok"
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    output: Optional[Dict[str, Any]] = None

    try:
        result_obj = fn(model_instance)
        result_dict: Dict[str, Any] = result_obj.model_dump()
        output = result_dict
        return result_dict
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        error_message = str(exc)
        raise
    finally:
        duration_ms = (time.perf_counter() - t0) * 1000
        if _audit_enabled():
            try:
                record = DecisionRecord(
                    request_id=request_id,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    tool_name=tool_name,
                    input=model_instance.model_dump(),
                    data_sources=list(_data_sources_var.get() or []),
                    cpp_available=_cpp_available(),
                    n_workers=getattr(model_instance, "n_workers", None),
                    duration_ms=round(duration_ms, 3),
                    output_hash=hash_payload(output) if output is not None else None,
                    status=status,
                    error_type=error_type,
                    error_message=error_message,
                )
                AuditWriter().write(record)
            except Exception:
                logger.warning("[audit] failed to write decision record for %s", tool_name, exc_info=True)
        _request_id_var.reset(token_req)
        _data_sources_var.reset(token_data)


# ──────────────────────────────────────────────────────────────────
# Replay verification
# ──────────────────────────────────────────────────────────────────

@dataclass
class ReplayResult:
    request_id: str
    tool_name: str
    output_match: Optional[bool]
    data_source_matches: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def verify_replay(record: Dict[str, Any]) -> ReplayResult:
    """
    Re-run a recorded tool call and compare data + output hashes against
    what was stored. A data-source mismatch with a matching output usually
    means the provider revised historical data; an output mismatch with
    matching data sources means the code/logic changed since the record
    was written.
    """
    # Local import: agent.tools imports this module, so importing it back
    # at module load time would create a circular import.
    from standard_quant_tools.agent.tools import _TOOL_DISPATCH

    tool_name = record["tool_name"]
    if tool_name not in _TOOL_DISPATCH:
        raise ValueError(f"Unknown tool '{tool_name}' in decision record.")
    fn, model_cls = _TOOL_DISPATCH[tool_name]

    token_data = _data_sources_var.set([])
    try:
        result_obj = fn(model_cls(**record["input"]))
        new_output = result_obj.model_dump()
        new_sources = list(_data_sources_var.get() or [])
    finally:
        _data_sources_var.reset(token_data)

    new_output_hash = hash_payload(new_output)
    stored_output_hash = record.get("output_hash")
    output_match: Optional[bool] = (
        new_output_hash == stored_output_hash if stored_output_hash is not None else None
    )

    old_by_key = {
        (s["symbol"], s["start"], s["end"], s["interval"]): s["content_hash"]
        for s in record.get("data_sources", [])
    }
    data_matches: List[Dict[str, Any]] = []
    for s in new_sources:
        key = (s["symbol"], s["start"], s["end"], s["interval"])
        old_hash = old_by_key.get(key)
        data_matches.append({
            "symbol": s["symbol"], "start": s["start"], "end": s["end"], "interval": s["interval"],
            "old_hash": old_hash, "new_hash": s["content_hash"], "match": old_hash == s["content_hash"],
        })

    notes: List[str] = []
    data_all_match = all(m["match"] for m in data_matches) if data_matches else True
    if not data_all_match:
        if output_match is False:
            notes.append(
                "Underlying data changed and the output changed accordingly — "
                "the provider likely revised historical values."
            )
        else:
            notes.append(
                "Underlying data changed but the output is unaffected "
                "(e.g. a scale- or shift-invariant metric) — worth a closer look."
            )
    elif output_match is False:
        notes.append(
            "Output changed even though input data is identical — "
            "code/logic likely changed since the record was written."
        )

    return ReplayResult(
        request_id=record["request_id"],
        tool_name=tool_name,
        output_match=output_match,
        data_source_matches=data_matches,
        notes=notes,
    )
