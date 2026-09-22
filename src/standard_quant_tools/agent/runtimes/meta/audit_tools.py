"""
What the decision log HOLDS, before anything is verified or explained.

The provenance tools all take something a caller already has to possess.
`explain_decision`, `replay_decision` and `compare_decisions` each need a
request id; `export_audit_bundle` needs a date range; `verify_audit_integrity`
needs a date or nothing. An agent holding a two-hundred-record log could
verify wholesale, export wholesale, and explain exactly one record -- if it
happened to know an id. It usually does not: `dispatch()` returns the
payload and nothing else, deliberately, so an in-process caller never sees
the id its own call was recorded under.

That made three of the five tools unreachable outside an MCP client. These
two are the doors:

  `describe_audit_log` answers what is there -- which dates, how many
  records, how large, which days are held, sealed or signed -- and what the
  trail is CONFIGURED to do, which zero tools reported.

  `find_decisions` answers which calls were made, and hands back the
  request ids the other three tools take. It is also the only way to see a
  FAILED call: an error record is written like any other and nothing could
  read one back.

BOTH ARE READ-ONLY, AND THAT IS STRUCTURAL. The mutating side of the audit
package -- collect, seal, hold, release, keypair, checkpoint -- stays
CLI-only and is absent from every dispatch table. An agent that can prune,
seal, unhold or re-sign its own history is not audited by it, and a
policy-driven deletion leaves exactly the evidence a tampered log leaves,
so the chain cannot tell them apart. What appears here is the read-only
counterpart: a hold is REPORTED, never placed or lifted, and the retention
window is reported as a PREVIEW of what a policy would make eligible,
never acted on.

Reads go through the audit package's own day-file discovery, so this tool
and the verifier always agree about which files are part of the trail.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools.agent.runtimes._json_safe import (
    finite_or_none as _finite_or_none,
)
from standard_quant_tools.error import ValidationError

from .config_tools import AUDIT_SETTINGS, resolve_setting

logger = logging.getLogger(__name__)
Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(default_factory=list)


class AuditLogInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_days: bool = Field(
        False,
        description=(
            "Add a per-day breakdown: record count, bytes, the first and "
            "last timestamp, and whether that day is held, sealed or has a "
            "signed checkpoint. Off by default because the summary answers "
            "'what is there' and costs one stat and one line count per file."
        ),
    )
    max_days: int = Field(
        90,
        ge=1,
        le=3660,
        description=(
            "Cap on the per-day breakdown, newest first. The summary counts "
            "every day regardless, so a capped listing never changes the "
            "totals above it."
        ),
    )


class AuditDaySummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: str = ""
    records: int = 0
    bytes: int = 0
    first_utc: Optional[str] = None
    last_utc: Optional[str] = None
    held: bool = Field(
        False,
        description=(
            "Under a legal/retention hold, so a retention policy will not "
            "make it a deletion candidate. Reported only -- placing and "
            "lifting a hold are operator actions with a CLI."
        ),
    )
    sealed: bool = Field(
        False,
        description=(
            "The day file is not writable by this process. A deployer's "
            "safeguard against accidental modification, not write-once "
            "storage: sufficient privilege can make it writable again."
        ),
    )
    checkpoint_signed: bool = Field(
        False,
        description=(
            "A signature sidecar sits beside this day. That is the only "
            "check that catches a wholesale rewrite, which the hash chain "
            "can recompute from its own published head."
        ),
    )


class AuditLogResult(_Result):
    audit_dir: str = ""
    recording_enabled: bool = True
    days: int = 0
    oldest_date: Optional[str] = None
    newest_date: Optional[str] = None
    total_records: int = 0
    total_bytes: int = 0
    retention_days: Optional[int] = Field(
        None,
        description=(
            "The configured retention window in days, or null when none is "
            "set. Unset means nothing is ever a deletion candidate."
        ),
    )
    gc_candidate_dates: List[str] = Field(
        default_factory=list,
        description=(
            "A PREVIEW: the dates a retention policy would make eligible "
            "for deletion, excluding any date under hold. Nothing here is "
            "deleted by this call or by any tool -- deletion is an operator "
            "action with a CLI, because the hash chain cannot tell a "
            "policy-driven deletion from the tampering it exists to detect."
        ),
    )
    redacted_fields: List[str] = Field(default_factory=list)
    redaction_salt_set: bool = Field(
        False,
        description=(
            "Whether a redaction salt is configured. The salt itself is "
            "never reported: it exists to make a redaction placeholder "
            "unrecoverable, so disclosing it would undo the redaction."
        ),
    )
    signing_configured: bool = False
    signing_available: bool = Field(
        False,
        description=(
            "Whether the optional cryptography dependency is importable. "
            "Configured-but-unavailable is a real state and a silent one."
        ),
    )
    fail_closed: bool = False
    day_summaries: List[AuditDaySummary] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class FindDecisionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: Optional[str] = Field(
        None,
        description=("Restrict to one tool, matched exactly. Omit for every tool."),
    )
    status: Optional[Literal["ok", "error"]] = Field(
        None,
        description=(
            "Restrict to calls that succeeded or to calls that raised. A "
            "failed call is recorded like any other and is otherwise "
            "unreadable -- this is the only way to reach one."
        ),
    )
    start_date: Optional[str] = Field(
        None,
        description=(
            "Earliest calendar day to scan, YYYY-MM-DD, inclusive. Whole "
            "day files outside the range are skipped without being parsed."
        ),
    )
    end_date: Optional[str] = Field(
        None, description="Latest calendar day to scan, YYYY-MM-DD, inclusive."
    )
    limit: int = Field(
        50,
        ge=1,
        le=500,
        description=(
            "Cap on returned matches. `total_scanned` and `truncated` say "
            "whether narrowing the filters would show you more."
        ),
    )
    newest_first: bool = Field(
        True,
        description=(
            "Most recent first. Turn it off to read the trail forwards, "
            "which is what a reconstruction of a session wants."
        ),
    )


class DecisionMatch(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str = Field(
        "",
        description=(
            "What explain_decision, replay_decision and compare_decisions "
            "take. This is the field those three tools had no door to."
        ),
    )
    timestamp_utc: str = ""
    tool_name: str = ""
    status: str = ""
    duration_ms: Stat = None
    error_type: Optional[str] = None
    output_hash: Optional[str] = None


class FindDecisionsResult(_Result):
    n_matches: int = 0
    matches: List[DecisionMatch] = Field(default_factory=list)
    total_scanned: int = Field(
        0,
        description=(
            "Records read from the day files the date filters left in "
            "scope, before the tool and status filters were applied. A "
            "FLOOR rather than a total when scan_complete is false."
        ),
    )
    truncated: bool = False
    scan_complete: bool = Field(
        True,
        description=(
            "False when the scan stopped as soon as it had a full page. "
            "The page is still the right one -- records are read in the "
            "order asked for -- but total_scanned then counts what was "
            "read rather than what exists."
        ),
    )
    days_scanned: int = 0
    days_skipped: int = Field(
        0,
        description=(
            "Whole day files the date filters excluded by their name, so "
            "not a line of them was parsed."
        ),
    )
    notes: List[str] = Field(default_factory=list)


def _require_date(value: Optional[str], field: str) -> Optional[str]:
    if value is None:
        return None
    if not _DATE_RE.match(value.strip()):
        raise ValidationError(
            f"{field}={value!r} is not a calendar day. Give it as "
            "YYYY-MM-DD, the same spelling the day files use -- "
            "describe_audit_log reports oldest_date and newest_date, which "
            "are the range there is anything to find in."
        )
    return value.strip()


def _count_lines(path: Path) -> int:
    """Records in a day file, without parsing one.

    Every record is one JSON line, so counting newlines is the record
    count -- and it is what keeps the summary cheap enough to call before
    deciding whether the per-day breakdown is worth asking for.
    """
    total = 0
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                total += block.count(b"\n")
    except OSError:
        logger.debug("[describe_audit_log] unreadable day file %s", path)
        return 0
    return total


def _edges(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """The first and last timestamp in a day file, or (None, None)."""
    first: Optional[str] = None
    last: Optional[str] = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            stamp = record.get("timestamp_utc")
            if stamp is None:
                continue
            if first is None:
                first = str(stamp)
            last = str(stamp)
    except OSError:
        logger.debug("[describe_audit_log] unreadable day file %s", path)
    return first, last


def _audit_configuration() -> Dict[str, Any]:
    """The seven audit settings, read from the one table that declares
    them, so this tool and `describe_effective_config` cannot disagree."""
    out: Dict[str, Any] = {}
    for setting in AUDIT_SETTINGS:
        value, present, _problem = resolve_setting(setting)
        out[setting.name] = (value, present)
    return out


def describe_audit_log(input_data: AuditLogInput) -> AuditLogResult:
    """
    What the decision log contains, and what it is configured to do.

    The trail's other tools each need something a caller already has -- a
    request id, a date range -- and nothing said what was there. An export
    over a range covering no day file used to come back as a bundle of
    nothing, indistinguishable from a real export; the dates reported here
    are the range there is anything to export.

    Recording, redaction, retention and signing are reported alongside the
    contents, because a count of zero means something different under each.
    The redaction salt is reported as set or unset and never as a value:
    it exists to make redaction placeholders unrecoverable.

    Nothing here deletes, seals, holds or releases anything. A hold is
    shown, and the retention window is shown as a preview of what a policy
    would make eligible -- the chain cannot distinguish a policy-driven
    deletion from the tampering it exists to detect, so deletion stays an
    operator action with a CLI.
    """
    from standard_quant_tools.audit.paths import _audit_dir, _iter_day_files
    from standard_quant_tools.audit.retention import gc_candidates, is_held
    from standard_quant_tools.audit.signing import HAS_CRYPTOGRAPHY

    directory = _audit_dir()
    day_files = _iter_day_files(directory)
    config = _audit_configuration()

    recording_enabled = config["SQT_AUDIT_ENABLED"][0] == "true"
    fail_closed = config["SQT_AUDIT_FAIL_CLOSED"][0] == "true"
    redacted_raw = config["SQT_AUDIT_REDACT_FIELDS"][0]
    redacted_fields = redacted_raw.split(",") if redacted_raw else []
    salt_set = bool(config["SQT_AUDIT_REDACT_SALT"][1])
    signing_configured = bool(config["SQT_AUDIT_SIGNING_KEY_PATH"][1])
    retention_raw = config["SQT_AUDIT_RETENTION_DAYS"][0]
    retention_days = int(retention_raw) if retention_raw is not None else None

    total_records = 0
    total_bytes = 0
    sizes: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    for path in day_files:
        try:
            size = int(path.stat().st_size)
        except OSError:
            size = 0
        records = _count_lines(path)
        sizes[path.stem] = size
        counts[path.stem] = records
        total_bytes += size
        total_records += records

    candidates = list(gc_candidates(directory))

    summaries: List[AuditDaySummary] = []
    if input_data.include_days:
        for path in list(reversed(day_files))[: input_data.max_days]:
            first, last = _edges(path)
            summaries.append(
                AuditDaySummary(
                    date=path.stem,
                    records=counts.get(path.stem, 0),
                    bytes=sizes.get(path.stem, 0),
                    first_utc=first,
                    last_utc=last,
                    held=bool(is_held(path.stem, directory)),
                    # Writability, not a permission bit: on Windows chmod
                    # can only toggle the read-only attribute, so asking
                    # the OS whether this process may write is the one
                    # question with the same answer on both platforms.
                    sealed=not os.access(path, os.W_OK),
                    checkpoint_signed=(
                        directory / f"{path.stem}.checkpoint.sig"
                    ).exists(),
                )
            )

    notes: List[str] = [
        "`gc_candidate_dates` is a PREVIEW of what a retention policy would "
        "make eligible. No tool deletes an audit day, and this call did "
        "not: a deleted day leaves the same evidence as a tampered one, so "
        "the deletion stays with an operator and a CLI.",
        "A hold, a seal and a checkpoint are REPORTED here and placed "
        "elsewhere. Lifting a hold is the precondition for a deletion, and "
        "re-signing a checkpoint could re-anchor a rewritten day, so "
        "neither is reachable from a tool.",
    ]
    if input_data.include_days and len(day_files) > input_data.max_days:
        notes.append(
            f"{len(day_files)} days exist and the {input_data.max_days} "
            "newest are listed. The totals above cover all of them."
        )
    elif not input_data.include_days and day_files:
        notes.append(
            "Per-day detail was not asked for. Set include_days to see "
            "which days are held, sealed or signed."
        )

    warnings: List[str] = []
    if not recording_enabled:
        warnings.append(
            "Recording is OFF, so calls made from now on leave no record "
            "and an empty trail here is not evidence that nothing happened. "
            "Set SQT_AUDIT_ENABLED=1 to have calls recorded."
        )
    if not day_files:
        warnings.append(
            f"There are no day files under {directory}. An empty trail is "
            "not an intact one -- verify_audit_integrity separates the two."
        )
    if retention_days is None and day_files:
        warnings.append(
            "No retention window is configured, so `gc_candidate_dates` is "
            "empty. That means nothing is ever a deletion candidate, not "
            "that everything is."
        )
    if redacted_fields and not salt_set:
        warnings.append(
            f"{len(redacted_fields)} field(s) are redacted with no salt "
            "configured, so the placeholders are unsalted and "
            "brute-forceable offline for any field with a small value "
            "space. Set SQT_AUDIT_REDACT_SALT and keep it stable."
        )
    if signing_configured and not HAS_CRYPTOGRAPHY:
        warnings.append(
            "A signing key is configured but the cryptography package is "
            "not importable, so no checkpoint can be signed. Install the "
            "signing extra, or the trail has no external anchor."
        )
    signed_days = sum(1 for s in summaries if s.checkpoint_signed)
    if input_data.include_days and summaries and not signed_days:
        warnings.append(
            "None of the listed days carries a signed checkpoint. The hash "
            "chain detects partial tampering; a wholesale rewrite can "
            "recompute it, and only a signature catches that."
        )

    logger.debug(
        "[describe_audit_log] dir=%s days=%d records=%d",
        directory,
        len(day_files),
        total_records,
    )
    return AuditLogResult(
        audit_dir=str(directory),
        recording_enabled=recording_enabled,
        days=len(day_files),
        oldest_date=day_files[0].stem if day_files else None,
        newest_date=day_files[-1].stem if day_files else None,
        total_records=total_records,
        total_bytes=total_bytes,
        retention_days=retention_days,
        gc_candidate_dates=candidates,
        redacted_fields=redacted_fields,
        redaction_salt_set=salt_set,
        signing_configured=signing_configured,
        signing_available=bool(HAS_CRYPTOGRAPHY),
        fail_closed=fail_closed,
        day_summaries=summaries,
        notes=notes,
        warnings=warnings,
    )


def find_decisions(input_data: FindDecisionsInput) -> FindDecisionsResult:
    """
    Which calls were recorded, and the request ids the rest of the
    provenance family takes.

    `explain_decision`, `replay_decision` and `compare_decisions` all
    require a request id, and `dispatch()` returns the payload alone --
    deliberately, since the id belongs to the record rather than to the
    answer. So an in-process caller had no way to obtain one and three
    tools were unreachable. This is that door.

    It is also the only way to reach a FAILED call. An error record carries
    the tool, the error type and the timing like any other, and nothing
    could read one back: filter on the error status to see them.

    Date filters are applied to the day file NAMES, so a range outside the
    trail parses nothing at all. Reads only; nothing is re-run here --
    `replay_decision` is the tool that re-runs a recorded call.
    """
    from standard_quant_tools.audit.paths import _audit_dir, _iter_day_files

    start = _require_date(input_data.start_date, "start_date")
    end = _require_date(input_data.end_date, "end_date")
    if start and end and start > end:
        raise ValidationError(
            f"start_date {start} is after end_date {end}. Swap them, or "
            "omit one to leave that end of the range open."
        )

    directory = _audit_dir()
    day_files = _iter_day_files(directory)
    in_range = [
        path
        for path in day_files
        if (start is None or path.stem >= start) and (end is None or path.stem <= end)
    ]
    if input_data.newest_first:
        in_range = list(reversed(in_range))

    total_scanned = 0
    scan_complete = True
    matches: List[DecisionMatch] = []
    for path in in_range:
        if len(matches) > input_data.limit:
            # A full page plus the one record that proves there are more.
            # Records are read in the order asked for, so the page is
            # already the right one and reading the rest of a long trail
            # would only make the same answer slower.
            scan_complete = False
            break
        try:
            lines = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError:
            logger.debug("[find_decisions] unreadable day file %s", path)
            continue
        if input_data.newest_first:
            lines = list(reversed(lines))
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                # A line that will not parse is a finding for the verifier,
                # not for a search: skipping it keeps the readable records
                # readable and leaves the integrity verdict where it
                # belongs.
                continue
            if len(matches) > input_data.limit:
                scan_complete = False
                break
            total_scanned += 1
            if input_data.tool_name and record.get("tool_name") != (
                input_data.tool_name
            ):
                continue
            if input_data.status and record.get("status") != input_data.status:
                continue
            matches.append(
                DecisionMatch(
                    request_id=str(record.get("request_id") or ""),
                    timestamp_utc=str(record.get("timestamp_utc") or ""),
                    tool_name=str(record.get("tool_name") or ""),
                    status=str(record.get("status") or ""),
                    duration_ms=record.get("duration_ms"),
                    error_type=record.get("error_type"),
                    output_hash=record.get("output_hash"),
                )
            )

    truncated = len(matches) > input_data.limit
    kept = matches[: input_data.limit]

    notes: List[str] = [
        "`request_id` is what explain_decision, replay_decision and "
        "compare_decisions take. Nothing else hands one back in process: "
        "dispatch() returns the payload and the id stays with the record.",
    ]
    if input_data.status is None:
        notes.append(
            "Both successful and failed calls are included. Filter on the "
            "error status to see only the calls that raised."
        )
    if truncated:
        notes.append(
            f"More than {input_data.limit} records matched and the first "
            f"{input_data.limit} are returned, in the order asked for. "
            "Narrow by tool, by status or by date rather than raising the "
            "limit -- the ids you need are usually the newest ones."
        )
    if not scan_complete:
        notes.append(
            "The scan stopped once it had a full page, so total_scanned is "
            "how many records were read rather than how many exist. "
            "describe_audit_log reports the trail's real totals."
        )

    warnings: List[str] = []
    if not day_files:
        warnings.append(
            f"There are no day files under {directory}, so there was "
            "nothing to search. describe_audit_log says whether recording "
            "is even on."
        )
    elif not in_range:
        warnings.append(
            "No day file falls in the requested range. describe_audit_log "
            "reports oldest_date and newest_date, which is the range there "
            "is anything to find in."
        )
    elif not matches:
        warnings.append(
            f"{total_scanned} record(s) were read and none matched the "
            "filters. A tool name is matched exactly; describe_runtime "
            "lists the names as they are recorded."
        )
    if input_data.tool_name and input_data.tool_name not in _known_tools():
        warnings.append(
            f"{input_data.tool_name!r} is not a tool this library "
            "dispatches, so nothing can ever have been recorded under it. "
            "describe_runtime lists the names."
        )

    logger.debug(
        "[find_decisions] scanned=%d matched=%d files=%d",
        total_scanned,
        len(matches),
        len(in_range),
    )
    return FindDecisionsResult(
        n_matches=len(kept),
        matches=kept,
        total_scanned=total_scanned,
        truncated=truncated,
        scan_complete=scan_complete,
        days_scanned=len(in_range),
        days_skipped=len(day_files) - len(in_range),
        notes=notes,
        warnings=warnings,
    )


def _known_tools() -> set:
    """Every dispatchable name, for telling a typo from an empty result."""
    try:
        from standard_quant_tools.agent.runtimes import all_runtimes

        return {t for rt in all_runtimes().values() for t in rt.dispatch_table}
    except Exception:  # noqa: BLE001 - a hint, never a failure
        logger.debug("[find_decisions] registry unavailable", exc_info=True)
        return set()


__all__ = [
    "AuditDaySummary",
    "AuditLogInput",
    "AuditLogResult",
    "DecisionMatch",
    "FindDecisionsInput",
    "FindDecisionsResult",
    "describe_audit_log",
    "find_decisions",
]
