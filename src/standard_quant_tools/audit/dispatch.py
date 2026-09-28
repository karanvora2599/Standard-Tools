"""`_run_and_record`: the shared core used by `agent.tools.dispatch()` to run
a tool call and -- unless disabled -- write its DecisionRecord."""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from standard_quant_tools._env import env_flag
from standard_quant_tools.error import AuditIntegrityError
from standard_quant_tools.numeric_contract import require_finite_scalar_fields

from .context import _data_sources_var, _request_id_var, new_request_id
from .hashing import hash_payload
from .json_native import to_json_native
from .models import DecisionRecord
from .paths import _audit_enabled
from .provenance import (
    _cpp_available,
    _git_sha,
    _native_build_label,
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

    Read through the library's one flag reader, so it agrees with
    SQT_AUDIT_ENABLED about what a word means: 1/true/yes/on and
    0/false/no/off in any case and padding, blank for the default, and any
    other word refused by name. "on" used to read as off here.

    Note this governs only WRITE failures. A corrupted existing chain
    (AuditIntegrityError) always propagates — see _run_and_record — because
    it is a statement about the whole log rather than about one record.
    """
    return env_flag("SQT_AUDIT_FAIL_CLOSED", False)


_last_record = threading.local()


def last_request_id() -> Optional[str]:
    """
    The request id of the most recent decision record this THREAD wrote,
    or None when the most recent dispatch on this thread wrote none.

    `_run_and_record` minted an id for every call and returned only the
    result, so nothing that dispatched a tool could ever learn which
    record it produced -- and `explain_decision`, `replay_decision` and
    `compare_decisions` all need exactly that id. It is thread-local
    because the MCP server runs each call on a worker thread and reads
    the id back on the same thread, right after the dispatch returns.

    It names a record that EXISTS. It used to be set when the id was
    minted, before anything was written, and was never cleared -- so after
    a call whose arguments failed validation it still named the previous
    call's record, and with recording off or a write that failed it named
    a record that was never written. Either way an explain or a replay
    described a different call. See the CHANGELOG entry of 2026-09-28.
    """
    return getattr(_last_record, "request_id", None)


def _forget_last_request_id() -> None:
    """Clear `last_request_id()` for this thread. Every dispatch entry point
    calls this first, so a call that fails before a record could be
    written -- an unknown tool, arguments its input model refuses -- leaves
    None behind rather than the previous call's id."""
    _last_record.request_id = None


def _run_and_record(
    tool_name: str, fn: Callable[[Any], Any], model_instance: Any
) -> Dict[str, Any]:
    """
    Shared core used by `agent.tools.dispatch()`: runs `fn(model_instance)`,
    and — unless disabled via `SQT_AUDIT_ENABLED=0` — writes a DecisionRecord
    capturing inputs, data provenance, execution context, and an output hash.

    Both audit settings are read BEFORE the tool runs. A setting that is
    refused (a word that is neither on nor off) then refuses the call
    itself, instead of surfacing after the tool has acted with no record
    of it.

    A NaN or infinite scalar parameter is refused FIRST, before a request
    id is minted, so -- like an argument the input schema refuses -- it
    writes no record. It is part of the input contract rather than
    something the tool did, and every dispatch path runs through here.
    """
    _forget_last_request_id()
    require_finite_scalar_fields(model_instance, tool_name)
    recording = _audit_enabled()
    fail_closed = _audit_fail_closed() if recording else False

    request_id = new_request_id()
    token_req = _request_id_var.set(request_id)
    token_data = _data_sources_var.set([])
    # The context variables are reset on EVERY way out, including the two
    # audit failures below that re-raise. They used to be reset only after
    # the write, so a refused write left this call's request id in the
    # context and stamped every later log line on the thread with it.
    try:
        t0 = time.perf_counter()
        status = "ok"
        error_type: Optional[str] = None
        error_message: Optional[str] = None
        output: Optional[Dict[str, Any]] = None

        try:
            result_obj = fn(model_instance)
            # A TOOL MAY RETURN A PLAIN DICT. Most return a Pydantic result
            # model, and calling `.model_dump()` unconditionally assumed all
            # of them did -- a tool whose library function already produces
            # a documented dict (option greeks, a volatility cone, a set of
            # liquidity estimates) died here with AttributeError instead of
            # returning. The audit record wants a dict either way, so
            # accepting one directly costs a branch and removes the
            # requirement to restate a well-shaped dict as a model purely to
            # satisfy this line.
            if isinstance(result_obj, dict):
                result_dict: Dict[str, Any] = result_obj
            else:
                result_dict = result_obj.model_dump()
            output = result_dict
            return result_dict
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            error_message = str(exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - t0) * 1000
            if recording:
                try:
                    fields = _redact_fields()
                    # JSON-native before anything reads it, and the SAME
                    # converted dict feeds both redactions below, so the
                    # value a placeholder stands for in `input` is the value
                    # scrubbed from `error_message`. A NaN is kept as the
                    # token "NaN" rather than null so a replay rebuilds the
                    # call that was made; a numpy value no longer makes the
                    # record unwritable. See the CHANGELOG entry of
                    # 2026-09-27.
                    raw_input = to_json_native(model_instance.model_dump())
                    # Redacting `input` alone isn't enough -- a tool
                    # exception's own message can echo a redacted value back
                    # (e.g. ValueError(f"Unknown account: {account_id}")),
                    # leaking it unredacted in the same record where `input`
                    # is masked.
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
                        native_build=_native_build_label(),
                        n_workers=getattr(model_instance, "n_workers", None),
                        duration_ms=round(duration_ms, 3),
                        output_hash=(
                            hash_payload(output) if output is not None else None
                        ),
                        # A second hash with run-specific dataset/model ids
                        # normalized away. Modeling mints a fresh id per run
                        # and embeds it in artifact paths, so a
                        # byte-identical re-run never matches the literal
                        # hash -- without this, every modeling replay
                        # reports a false mismatch, which is worse than no
                        # replay support because it looks like evidence of
                        # drift. Both hashes are stored: the literal one
                        # still detects any change for deterministic tools.
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
                    # Only now does a record with this id exist.
                    _last_record.request_id = request_id
                except AuditIntegrityError:
                    # NEVER swallowed, regardless of the fail-open policy
                    # below.
                    #
                    # This is the interaction that matters: the writer
                    # refuses to extend a chain whose tail it cannot read,
                    # and a bare `except Exception` here would have caught
                    # that refusal and logged it as an ordinary write
                    # failure — leaving the tool result returned and the
                    # corruption invisible, which is exactly the state the
                    # writer's check exists to prevent.
                    #
                    # A transient write failure (disk full, permissions) and
                    # a CORRUPTED CHAIN are different events. The first is
                    # about this one record; the second says the log as a
                    # whole is no longer trustworthy.
                    raise
                except Exception:
                    if fail_closed:
                        raise
                    logger.warning(
                        "[audit] failed to write decision record for %s "
                        "(fail-open: the tool result is still returned; set "
                        "SQT_AUDIT_FAIL_CLOSED=1 to make this fatal)",
                        tool_name,
                        exc_info=True,
                    )
    finally:
        _request_id_var.reset(token_req)
        _data_sources_var.reset(token_data)
