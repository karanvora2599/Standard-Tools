"""`_run_and_record`: the shared core used by `agent.tools.dispatch()` to run
a tool call and -- unless disabled -- write its DecisionRecord."""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from .context import _data_sources_var, _request_id_var, new_request_id
from .hashing import hash_payload
from .models import DecisionRecord
from .paths import _audit_enabled
from .provenance import (
    _cpp_available,
    _git_sha,
    _package_version,
    _strategy_source_hash,
)
from .redaction import _redact, _redact_fields
from .writer import AuditWriter

logger = logging.getLogger(__name__)


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
                    input=_redact(model_instance.model_dump(), _redact_fields()),
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
