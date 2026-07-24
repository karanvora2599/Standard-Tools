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
import sys
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


def _cpp_available() -> bool:
    try:
        import standard_quant_tools._sqt_core  # type: ignore[attr-defined]  # noqa: F401

        return True
    except ImportError:
        return False


_git_sha_cache: Optional[str] = None
_git_sha_resolved = False


def _git_sha() -> Optional[str]:
    """
    Best-effort `git rev-parse HEAD` in the repo containing this file.
    Returns None (never raises) outside a git checkout, without git
    installed, or in any other failure mode — provenance is a nice-to-have,
    not something that should ever break a tool call. Resolved once per
    process and cached.
    """
    global _git_sha_cache, _git_sha_resolved
    if _git_sha_resolved:
        return _git_sha_cache
    _git_sha_resolved = True
    try:
        import subprocess

        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            _git_sha_cache = result.stdout.strip() or None
    except Exception:
        _git_sha_cache = None
    return _git_sha_cache


def _package_version() -> Optional[str]:
    try:
        from standard_quant_tools import __version__

        return __version__
    except Exception:
        return None


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
    # Reproducibility provenance — None when unavailable (e.g. no git
    # checkout), never a reason to fail the call itself.
    git_commit_sha: Optional[str] = None
    package_version: Optional[str] = None
    random_seed: Optional[int] = None
    strategy_source_hash: Optional[str] = None
    # Hash-chain tamper-evidence: each record's hash covers its own content
    # plus the previous record's hash, so editing a past line changes that
    # line's hash and breaks the chain for every record after it (unless an
    # attacker also rewrites every subsequent line to match — this detects
    # accidental/partial tampering, not a fully-rewritten log; there is no
    # external anchor/signature to detect a wholesale rewrite). "0" * 16 for
    # the first record of a day's file.
    prev_record_hash: Optional[str] = None
    record_hash: Optional[str] = None


def _audit_enabled() -> bool:
    return os.environ.get("SQT_AUDIT_ENABLED", "1").lower() not in ("0", "false", "")


def _audit_dir() -> Path:
    return Path(
        os.environ.get(
            "SQT_AUDIT_DIR",
            str(Path.home() / ".cache" / "standard_quant_tools" / "audit"),
        )
    )


_GENESIS_HASH = "0" * 16


def _acquire_lock(lock_path: Path) -> Optional[Any]:
    """
    Best-effort cross-process exclusive lock via a small sidecar file (not
    the growing JSONL file itself — locking a fixed, tiny file avoids the
    platform-specific complexity of byte-range-locking a file whose EOF
    offset keeps moving). Returns an open file handle the caller must pass
    to `_release_lock`, or None if locking isn't available on this platform
    — in which case writes proceed unlocked rather than blocking a tool
    call on a missing OS primitive.
    """
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lf = open(lock_path, "a+b")
        if sys.platform == "win32":
            import msvcrt

            lf.seek(0)
            msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        return lf
    except Exception:
        logger.debug("[audit] advisory file lock unavailable", exc_info=True)
        return None


def _release_lock(lf: Optional[Any]) -> None:
    if lf is None:
        return
    try:
        if sys.platform == "win32":
            import msvcrt

            lf.seek(0)
            msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    finally:
        lf.close()


class AuditWriter:
    """Append-only JSONL writer, one file per UTC day."""

    def __init__(self, audit_dir: Optional[Union[str, Path]] = None):
        self._dir = Path(audit_dir) if audit_dir else _audit_dir()

    def _path_for(self, when: datetime) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir / f"{when.strftime('%Y-%m-%d')}.jsonl"

    def _last_record_hash(self, path: Path) -> str:
        """Hash of the last line in `path`, or the genesis hash if the file
        doesn't exist/is empty. Must be called while the write lock is held,
        since it establishes the chain link the new record commits to."""
        if not path.exists():
            return _GENESIS_HASH
        last_line: Optional[str] = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if last_line is None:
            return _GENESIS_HASH
        try:
            return json.loads(last_line).get("record_hash") or _GENESIS_HASH
        except Exception:
            return _GENESIS_HASH

    def write(self, record: DecisionRecord) -> Path:
        when = datetime.now(timezone.utc)
        path = self._path_for(when)
        lock_path = path.with_name(path.name + ".lock")

        lf = _acquire_lock(lock_path)
        try:
            record.prev_record_hash = self._last_record_hash(path)
            # Hash over the record with record_hash itself left unset, so
            # the chain link (prev_record_hash) and the record's own content
            # are both covered without the field hashing itself.
            record.record_hash = hash_payload(
                {**record.model_dump(exclude={"record_hash"}), "record_hash": None}
            )
            with open(path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        finally:
            _release_lock(lf)
        return path


def verify_audit_log_integrity(path: Union[str, Path]) -> List[str]:
    """
    Walk a single day's JSONL audit file and confirm its hash chain is
    intact. Returns a list of human-readable problems (empty if the file is
    clean or doesn't exist). Detects a record whose content was edited after
    the fact, or a record removed/reordered/inserted — as long as every
    later record in the file wasn't *also* rewritten to match, which is a
    fundamentally unreachable guarantee without an external, independently
    stored anchor (e.g. signing the last hash of each day into a separate
    system) — this function does not attempt that.
    """
    path = Path(path)
    if not path.exists():
        return []
    problems: List[str] = []
    prev_hash = _GENESIS_HASH
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            claimed_prev = record.get("prev_record_hash")
            if claimed_prev != prev_hash:
                problems.append(
                    f"line {lineno} (request_id={record.get('request_id')}): "
                    f"prev_record_hash={claimed_prev!r} does not match the "
                    f"preceding record's hash {prev_hash!r} — chain broken "
                    "(a record was edited, removed, reordered, or inserted)."
                )
            recomputed = hash_payload({**record, "record_hash": None})
            claimed_hash = record.get("record_hash")
            if recomputed != claimed_hash:
                problems.append(
                    f"line {lineno} (request_id={record.get('request_id')}): "
                    f"record_hash={claimed_hash!r} does not match its own "
                    f"recomputed content hash {recomputed!r} — this line's "
                    "content was altered after it was written."
                )
            prev_hash = claimed_hash or prev_hash
    return problems


def _strategy_source_hash(model_instance: Any) -> Optional[str]:
    """
    Content hash of a registered strategy's source code, when
    `model_instance` names one via a `strategy` or `strategy_type` field
    (e.g. WalkForwardInput.strategy, BacktestDiagnosticsInput.strategy_type).
    None when neither field is present, the name isn't a registered
    strategy (e.g. a custom-signal tool), or on any lookup failure —
    provenance is a nice-to-have, not something that should ever break a
    tool call.
    """
    strategy_name = getattr(model_instance, "strategy", None) or getattr(
        model_instance, "strategy_type", None
    )
    if not strategy_name:
        return None
    try:
        import inspect

        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        fn = STRATEGY_REGISTRY.get(strategy_name)
        if fn is None:
            return None
        return hash_payload(inspect.getsource(fn))
    except Exception:
        return None


def _run_and_record(
    tool_name: str, fn: Callable[[Any], Any], model_instance: Any
) -> Dict[str, Any]:
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
                    git_commit_sha=_git_sha(),
                    package_version=_package_version(),
                    random_seed=getattr(model_instance, "random_seed", None),
                    strategy_source_hash=_strategy_source_hash(model_instance),
                )
                AuditWriter().write(record)
            except Exception:
                logger.warning(
                    "[audit] failed to write decision record for %s",
                    tool_name,
                    exc_info=True,
                )
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
        new_output_hash == stored_output_hash
        if stored_output_hash is not None
        else None
    )

    old_by_key = {
        (s["symbol"], s["start"], s["end"], s["interval"]): s["content_hash"]
        for s in record.get("data_sources", [])
    }
    new_by_key = {
        (s["symbol"], s["start"], s["end"], s["interval"]): s["content_hash"]
        for s in new_sources
    }
    # Iterate the union of old and new keys, not just new_sources — a key
    # present in the original record but absent from the replay (e.g. the
    # tool no longer fetches a symbol/range it used to) must still be
    # reported, not silently dropped just because the loop only walked
    # what the replay happened to touch.
    data_matches: List[Dict[str, Any]] = []
    for key in sorted(set(old_by_key) | set(new_by_key)):
        symbol, start, end, interval = key
        old_hash = old_by_key.get(key)
        new_hash = new_by_key.get(key)
        data_matches.append(
            {
                "symbol": symbol,
                "start": start,
                "end": end,
                "interval": interval,
                "old_hash": old_hash,
                "new_hash": new_hash,
                "match": old_hash == new_hash,
            }
        )

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
