"""`_run_and_record`: the shared core used by `agent.tools.dispatch()` to run
a tool call and -- unless disabled -- write its DecisionRecord."""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from standard_quant_tools.config import load_env
from standard_quant_tools.error import AuditIntegrityError

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
from .redaction import _redact, _redact_fields, redact_text
from .replay import normalize_identifiers
from .writer import AuditWriter

logger = logging.getLogger(__name__)


def _audit_fail_closed() -> bool:
    """
    Whether a failure to WRITE an audit record should fail the tool call.

    Defaults to False — fail-open — because for an open-source analytics
    library a full disk should not destroy a legitimate result the caller
    already paid to compute. That default is a judgement about the common
    case, not a claim that it is always right: under a governance or
    compliance regime, an action taken without a record of it is precisely
    the thing the audit trail exists to prevent, and the result should not be
    returned at all. `SQT_AUDIT_FAIL_CLOSED=1` selects that behaviour.

    Note this governs only WRITE failures. A corrupted existing chain
    (AuditIntegrityError) always propagates — see _run_and_record — because
    it is a statement about the whole log rather than about one record.
    """
    load_env()
    return os.environ.get("SQT_AUDIT_FAIL_CLOSED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


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
                fields = _redact_fields()
                raw_input = model_instance.model_dump()
                # Redacting `input` alone isn't enough -- a tool exception's
                # own message can echo a redacted value back (e.g.
                # ValueError(f"Unknown account: {account_id}")), leaking it
                # unredacted in the same record where `input` is masked.
                safe_error_message = (
                    redact_text(error_message, raw_input, fields)
                    if error_message is not None
                    else None
                )
                record = DecisionRecord(
                    request_id=request_id,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    tool_name=tool_name,
                    input=_redact(raw_input, fields),
                    data_sources=list(_data_sources_var.get() or []),
                    cpp_available=_cpp_available(),
                    n_workers=getattr(model_instance, "n_workers", None),
                    duration_ms=round(duration_ms, 3),
                    output_hash=hash_payload(output) if output is not None else None,
                    # A second hash with run-specific dataset/model ids
                    # normalized away. Modeling mints a fresh id per run and
                    # embeds it in artifact paths, so a byte-identical
                    # re-run never matches the literal hash -- without this,
                    # every modeling replay reports a false mismatch, which
                    # is worse than no replay support because it looks like
                    # evidence of drift. Both hashes are stored: the literal
                    # one still detects any change for deterministic tools.
                    output_hash_normalized=(
                        hash_payload(normalize_identifiers(output))
                        if output is not None
                        else None
                    ),
                    status=status,
                    error_type=error_type,
                    error_message=safe_error_message,
                    git_commit_sha=_git_sha(),
                    package_version=_package_version(),
                    random_seed=getattr(model_instance, "random_seed", None),
                    strategy_source_hash=_strategy_source_hash(model_instance),
                )
                AuditWriter().write(record)
            except AuditIntegrityError:
                # NEVER swallowed, regardless of the fail-open policy below.
                #
                # This is the interaction that matters: the writer now refuses
                # to extend a chain whose tail it cannot read, and a bare
                # `except Exception` here would have caught that refusal and
                # logged it as an ordinary write failure — leaving the tool
                # result returned and the corruption invisible, which is
                # exactly the state the writer's check exists to prevent.
                #
                # A transient write failure (disk full, permissions) and a
                # CORRUPTED CHAIN are different events. The first is about
                # this one record; the second says the log as a whole is no
                # longer trustworthy.
                raise
            except Exception:
                if _audit_fail_closed():
                    raise
                logger.warning(
                    "[audit] failed to write decision record for %s "
                    "(fail-open: the tool result is still returned; set "
                    "SQT_AUDIT_FAIL_CLOSED=1 to make this fatal)",
                    tool_name,
                    exc_info=True,
                )
        _request_id_var.reset(token_req)
        _data_sources_var.reset(token_data)
