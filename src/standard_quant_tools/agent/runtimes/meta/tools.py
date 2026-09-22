"""
The `meta` runtime: questions about the library and the session.

What this library accepts (strategy parameter contracts, stress-scenario
windows, what the data provider can actually serve) and what it already did
(a recorded call's inputs and execution path, whether it still reproduces,
whether the decision log is intact).

Nothing here reads a market. Retention operations that could destroy the
audit record are deliberately absent -- see the provenance tools' own notes.
"""

import datetime
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from standard_quant_tools._containment import require_within
from standard_quant_tools.agent.models import (
    ArgumentProblem,
    CompareDataSourcesInput,
    CompareDataSourcesResult,
    CompareDecisionsInput,
    CompareDecisionsResult,
    ConvertReferenceInput,
    ConvertReferenceResult,
    DataCapabilitiesInput,
    DataCapabilitiesResult,
    DataSourceMatch,
    DataSourceRef,
    DeclaredNote,
    DescribeArtifactInput,
    DescribeArtifactResult,
    DescribeReferenceInput,
    DescribeReferenceResult,
    DescribeToolInput,
    DescribeToolResult,
    ExplainDecisionInput,
    ExplainDecisionResult,
    ExportAuditBundleInput,
    ExportAuditBundleResult,
    FieldDivergence,
    ListReferenceKindsInput,
    ListReferenceKindsResult,
    ListStrategiesInput,
    ListStrategiesResult,
    ListStressScenariosInput,
    ListStressScenariosResult,
    ReadReferenceInput,
    ReadReferenceResult,
    ReferenceKind,
    ReplayDecisionInput,
    ReplayDecisionResult,
    StrategyDescriptor,
    StrategyParameter,
    StrategyRelation,
    StressScenario,
    TemporalContractInput,
    TemporalContractResult,
    ValidateToolCallInput,
    ValidateToolCallResult,
    VerifyAuditIntegrityInput,
    VerifyAuditIntegrityResult,
)
from standard_quant_tools.audit.export import export_bundle as _export_bundle
from standard_quant_tools.audit.paths import (
    _INDEX_FILENAME,
    _audit_dir,
    _audit_enabled,
    _iter_day_files,
)
from standard_quant_tools.audit.replay import verify_replay as _verify_replay
from standard_quant_tools.audit.verify import verify_audit_log_integrity as _verify_day
from standard_quant_tools.audit.verify import (
    verify_audit_trail_integrity as _verify_trail,
)
from standard_quant_tools.backtest.artifacts import load_artifact
from standard_quant_tools.backtest.strategy_params import (
    _MAX_WINDOW_BARS,
    _RELATIONS,
    STRATEGY_PARAM_SCHEMA,
    resolve_strategy_params,
)
from standard_quant_tools.backtest.stress_test import (
    list_stress_scenarios as _library_stress_scenarios,
)
from standard_quant_tools.data import _cache as _cache_module
from standard_quant_tools.data._cache import dead_generations
from standard_quant_tools.data.base import DataProvider
from standard_quant_tools.data.bloomberg_provider import BloombergProvider
from standard_quant_tools.data.databento_provider import DatabentoProvider
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.data.polygon_provider import PolygonProvider
from standard_quant_tools.data.yfinance_provider import YFinanceProvider
from standard_quant_tools.error import ValidationError

#: Provider classes by source name, for describing one that cannot be
#: constructed here (no API key, SDK absent). DataFactory raises in that
#: case rather than returning an instance, and "you would need a key" is a
#: more useful answer than the raise.
#:
#: DATABENTO BELONGS HERE for that reason above all: it is the one provider
#: that serves depth and order events, it reads its credential lazily, and
#: without a key it is exactly the provider a caller needs described rather
#: than raised at. Its absence was a KeyError on that path.
_PROVIDER_CLASSES: Dict[str, type] = {
    "yfinance": YFinanceProvider,
    "polygon": PolygonProvider,
    "bloomberg": BloombergProvider,
    "databento": DatabentoProvider,
}


def _jsonable(value: Any) -> Any:
    """One artifact cell as a JSON-safe scalar.

    Parquet round trips Timestamps and numpy scalars, neither of which
    survives json.dumps. Stringifying timestamps rather than converting to
    epoch keeps the preview readable, which is the only thing a preview is
    for.
    """
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if value is not None and isinstance(value, float) and not math.isfinite(value):
        return None
    return value


#: Day files are named YYYY-MM-DD.jsonl. The date argument is LLM-reachable
#: and is joined into a filesystem path, so it is matched against this
#: before it becomes one -- the same reason artifacts validate identifiers.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _find_audit_record(request_id: str) -> Dict[str, Any]:
    """One record by id, as a plain dict.

    Wraps cli.find_record so its ValueError becomes the ValidationError
    every other tool raises for a bad argument -- an unknown request id is
    a caller mistake, not an internal failure, and it should read like one.
    """
    from standard_quant_tools import cli as _cli

    try:
        return _cli.find_record(request_id)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def explain_decision(input_data: ExplainDecisionInput) -> ExplainDecisionResult:
    """
    What one recorded tool call actually did: its inputs, the market data it
    read (with the content hashes those inputs had AT THE TIME), which
    execution path ran, how long it took, and the code state it ran under.

    The execution path is the field that cannot be reconstructed later by
    any other means. C++, Numba and pure Python are chosen at call time and
    fall back transparently, so "which one ran" is knowable only because
    the record says so.

    Every field the record carries crosses. Four of them used not to, and
    the notable one is `strategy_source_hash`: a registered strategy's
    source is hashed at call time precisely so a run can be tied to the
    code that produced it, which `git_commit_sha` does only for a checkout
    that still exists and only at repository granularity.
    """
    record = _find_audit_record(input_data.request_id)
    sources = [
        DataSourceRef(
            symbol=source.get("symbol"),
            start_date=source.get("start") or source.get("start_date"),
            end_date=source.get("end") or source.get("end_date"),
            rows=source.get("rows"),
            content_hash=source.get("content_hash") or source.get("hash"),
            # The record has written both of these since provider fetches
            # started reporting themselves, and neither crossed. `source`
            # is the one that says WHICH vendor dataset answered, which is
            # date-dependent for at least one provider here and is
            # otherwise unrecoverable after the fact.
            source=source.get("source"),
            interval=source.get("interval"),
        )
        for source in record.get("data_sources", [])
    ]
    return ExplainDecisionResult(
        request_id=record["request_id"],
        timestamp_utc=record["timestamp_utc"],
        tool_name=record["tool_name"],
        status=record.get("status", "unknown"),
        input=record.get("input", {}),
        data_sources=sources,
        duration_ms=float(record.get("duration_ms", 0.0)),
        execution_path="C++" if record.get("cpp_available") else "Python/Numba",
        n_workers=record.get("n_workers"),
        output_hash=record.get("output_hash"),
        output_hash_normalized=record.get("output_hash_normalized"),
        strategy_source_hash=record.get("strategy_source_hash"),
        git_commit_sha=record.get("git_commit_sha"),
        package_version=record.get("package_version"),
        random_seed=record.get("random_seed"),
        error_type=record.get("error_type"),
        error_message=record.get("error_message"),
        prev_record_hash=record.get("prev_record_hash"),
        record_hash=record.get("record_hash"),
    )


def replay_decision(input_data: ReplayDecisionInput) -> ReplayDecisionResult:
    """
    Re-run a recorded call and say whether it still produces the same answer.

    The useful part is the four-way verdict, not the boolean. A different
    output on its own means nothing: the market data behind the call may
    have been revised, and yfinance guarantees neither point-in-time values
    nor that adjusted prices stay put. So the data hashes are checked
    FIRST, and only "the inputs still hash the same but the output does
    not" implicates the library -- that is `code_changed`. When the inputs
    moved, the verdict is `data_changed` and the output difference is
    expected rather than suspicious.

    The hashes behind the verdict come with it: the stored and new output
    hash, and both hashes of every data source, so a caller told the code
    changed can see WHICH input still matched and which output did not.
    """
    record = _find_audit_record(input_data.request_id)
    try:
        result = _verify_replay(record)
    except Exception as exc:  # a replay that cannot run is a real answer
        logger.warning("[replay_decision] %s failed: %s", input_data.request_id, exc)
        return ReplayDecisionResult(
            request_id=input_data.request_id,
            tool_name=record.get("tool_name", "unknown"),
            output_match=None,
            verdict="failed",
            notes=[f"replay could not run: {exc}"],
        )

    matches = [
        DataSourceMatch(
            symbol=match.get("symbol"),
            matches=match.get("match"),
            detail=(
                f"{match.get('start')} -> {match.get('end')} "
                f"({match.get('interval')})"
            ),
            old_hash=match.get("old_hash"),
            new_hash=match.get("new_hash"),
        )
        for match in result.data_source_matches
    ]
    checked = [m.matches for m in matches if m.matches is not None]
    data_moved = any(m is False for m in checked)

    if result.output_match is None:
        verdict = "not_comparable"
    elif result.output_match:
        verdict = "reproduced"
    elif data_moved:
        verdict = "data_changed"
    else:
        verdict = "code_changed"

    notes = list(result.notes)
    if verdict == "data_changed":
        notes.append(
            "The recorded inputs no longer hash the same, so a different "
            "output is expected and says nothing about the library. This is "
            "the normal consequence of a provider that does not guarantee "
            "point-in-time values."
        )
    elif verdict == "code_changed":
        notes.append(
            "Every checked input still hashes identically and the output "
            "does not match. That combination points at the code, not the "
            "data -- compare git_commit_sha via explain_decision."
        )
    elif verdict == "not_comparable" and not notes:
        notes.append(
            "The record carries no comparable output hash, so replay can "
            "neither confirm nor deny reproduction."
        )

    logger.debug("[replay_decision] %s verdict=%s", input_data.request_id, verdict)
    return ReplayDecisionResult(
        request_id=result.request_id,
        tool_name=result.tool_name,
        output_match=result.output_match,
        data_source_matches=matches,
        verdict=verdict,
        # The two hashes the verdict was decided from. "The code changed"
        # without them is an assertion; with them it is an answer the
        # caller can carry to a diff.
        stored_output_hash=result.stored_output_hash,
        new_output_hash=result.new_output_hash,
        notes=notes,
    )


def compare_decisions(input_data: CompareDecisionsInput) -> CompareDecisionsResult:
    """
    Diff two recorded calls: tool, inputs, output hash and code provenance.

    The question this answers is "why did these two runs disagree", and the
    summary states which of the three candidate causes the evidence
    supports -- different inputs, different code, or the same of both with a
    different answer, which means the data moved.
    """
    from standard_quant_tools import cli as _cli

    a = _find_audit_record(input_data.request_id_a)
    b = _find_audit_record(input_data.request_id_b)
    diff = _cli.cmd_compare(input_data.request_id_a, input_data.request_id_b)

    same_tool = a.get("tool_name") == b.get("tool_name")
    same_input = a.get("input") == b.get("input")
    same_output = a.get("output_hash") == b.get("output_hash")

    summary: List[str] = []
    if not same_tool:
        summary.append(
            f"Different tools ({a.get('tool_name')!r} vs "
            f"{b.get('tool_name')!r}); nothing below is comparable."
        )
    elif not same_input:
        summary.append(
            "Same tool, different inputs — the outputs are expected to "
            "differ and this diff explains why."
        )
    elif same_output:
        summary.append(
            "Same tool, same inputs, same output hash: these two runs are "
            "reproductions of each other."
        )
    else:
        summary.append(
            "Same tool and identical inputs but a different output hash. "
            "Either the code changed between them (compare git_commit_sha "
            "and package_version above) or the underlying market data was "
            "revised — replay_decision on each id distinguishes the two."
        )
    if a.get("git_commit_sha") != b.get("git_commit_sha"):
        summary.append(
            f"They ran at different commits ({a.get('git_commit_sha')} vs "
            f"{b.get('git_commit_sha')})."
        )

    return CompareDecisionsResult(
        request_id_a=input_data.request_id_a,
        request_id_b=input_data.request_id_b,
        same_tool=same_tool,
        same_input=same_input,
        same_output=same_output,
        diff=diff,
        summary=summary,
    )


def _indexed_chain_head(date: str, directory: Path) -> Optional[str]:
    """
    The chain head the index says a day's first record must claim, or None
    when the index has no entry for that date.

    A day file verified with no head given is checked against the genesis
    hash, which is right only for the very first day the trail ever wrote.
    Every later day's first record chains onto the previous day's last one,
    and the chain index is where that head is written down -- so verifying
    a single day without consulting it reported every day but the first as
    a broken chain, permanently, for a log nobody had touched.

    The last matching entry wins: the index is append-only and a date
    appears once, but a duplicated entry means the most recent claim is
    the one the writer committed to.
    """
    index_path = directory / _INDEX_FILENAME
    if not index_path.exists():
        return None
    head: Optional[str] = None
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except Exception:
                # A malformed index line is the trail check's finding to
                # report, not this helper's to raise on.
                continue
            if entry.get("date") == date and entry.get("chain_head"):
                head = entry["chain_head"]
    return head


#: The checkpoint states where a check RAN and did not pass. The others --
#: a day nobody ever anchored, a checkpoint with no signature beside it, a
#: check that could not be made at all -- are the ABSENCE of evidence, and
#: reporting them as False made "nobody signed this day" read exactly like
#: "this day was forged".
_SIGNATURE_CHECK_FAILED = {"key_mismatch", "corrupt_signature", "content_drift"}

#: What each non-valid checkpoint state means and what to do about it. A
#: single boolean collapsed all six into "no", which is the same shape of
#: answer as an integrity check that cannot tell an empty directory from
#: an intact trail.
_SIGNATURE_STATE_NOTES: Dict[str, str] = {
    "no_checkpoint": (
        "No checkpoint file exists for this day, so there is nothing to "
        "verify -- the day was never anchored. This is not evidence of "
        "tampering; it is the absence of the evidence that would detect it."
    ),
    "no_signature": (
        "A checkpoint exists for this day but its signature file does not, "
        "so the checkpoint is unsigned and proves nothing on its own."
    ),
    "key_mismatch": (
        "The signature is well-formed but was not made by the key that "
        "matches the public key supplied. Either the wrong public key was "
        "given, or the checkpoint was signed by someone else."
    ),
    "corrupt_signature": (
        "The signature file could not be read as a signature over this "
        "checkpoint -- truncated, re-encoded or altered bytes."
    ),
    "content_drift": (
        "The signature is valid, but the day's content has moved since it "
        "was signed. Verifying a signed checkpoint is itself a recorded "
        "call -- THIS call appends a record to today's file -- so today's "
        "day drifts by design. Sign a day after it closes, or verify "
        "yesterday's date, before reading this as tampering."
    ),
    "unavailable": (
        "The signature could not be checked at all -- no public key file at "
        "that path, an unreadable key or checkpoint, a day whose tail cannot "
        "be read, or no Ed25519 implementation installed. This is a MISSING "
        "check, not a failed one, so it is reported as unknown rather than "
        "as a broken signature."
    ),
}


def verify_audit_integrity(
    input_data: VerifyAuditIntegrityInput,
) -> VerifyAuditIntegrityResult:
    """
    Check the audit log's hash chain, and optionally a day's signature.

    Each record's hash covers its own content plus the previous record's,
    so editing a past line breaks every line after it. That detects
    accidental or partial tampering. It does NOT detect a wholesale rewrite
    of the file, because a rewriter can recompute the whole chain — only
    the Ed25519 checkpoint signature catches that, which is why supplying
    a public key is a materially stronger check and not merely a longer one.

    With no date, the full cross-day trail is verified, which additionally
    catches a missing day that a per-file check cannot see.

    READ `verdict`, NOT `intact`. "Nothing was found broken" is true of a
    directory with nothing in it, and of a directory nothing is being
    written to — three different states that all produced the same
    reassuring boolean. `verdict` separates them, `recording_enabled` says
    whether calls made now are recorded at all, and `signature_state`
    names which of six things a failed checkpoint check means.
    """
    notes: List[str] = []
    signature_valid: Optional[bool] = None
    signature_state: Optional[str] = None
    directory = _audit_dir()
    recording_enabled = _audit_enabled()
    day_files = _iter_day_files(directory)

    if input_data.date is None:
        problems = list(_verify_trail())
        scope = "trail"
    else:
        # Day files are named YYYY-MM-DD.jsonl under the audit dir. Built
        # from the validated date rather than joined from raw input: this
        # argument is LLM-reachable and goes into a filesystem path.
        if not _DATE_RE.match(input_data.date):
            raise ValidationError(f"date={input_data.date!r} must be YYYY-MM-DD.")
        path = directory / f"{input_data.date}.jsonl"
        if not path.exists():
            raise ValidationError(
                f"no audit file for {input_data.date}. Verify the whole "
                "trail (omit `date`) to see which days exist."
            )
        # Seeded with the head the chain index recorded for this day, which
        # is what makes a single-day check agree with the trail check.
        # Without it the file is measured against genesis and every day but
        # the first is called broken.
        expected_head = _indexed_chain_head(input_data.date, directory)
        if expected_head is None:
            problems = list(_verify_day(path))
            notes.append(
                f"The chain index has no entry for {input_data.date}, so "
                "this file's first record was checked against the genesis "
                "hash. That is correct for the first day the trail ever "
                "wrote and for days that predate the index; a later day "
                "missing from the index is itself a finding, which the "
                "trail check reports (omit `date`)."
            )
        else:
            problems = list(_verify_day(path, expected_prev_hash=expected_head))
        scope = input_data.date
        notes.append(
            "A single day verified in isolation cannot detect a MISSING "
            "day. Omit `date` to verify the cross-day trail as well."
        )

    if input_data.public_key_path is not None:
        try:
            from standard_quant_tools.audit.signing import verify_checkpoint_state

            signature_state = str(
                verify_checkpoint_state(
                    input_data.date,  # type: ignore[arg-type]
                    input_data.public_key_path,
                )
            )
        except Exception as exc:
            signature_state = "unavailable"
            notes.append(f"checkpoint signature could not be verified: {exc}")
        if signature_state == "valid":
            signature_valid = True
        elif signature_state in _SIGNATURE_CHECK_FAILED:
            signature_valid = False
        else:
            # Nothing to check, or nothing able to check it. Left unknown
            # so the verdict stays what the chain says rather than calling
            # an unsigned day tampered.
            signature_valid = None
        if signature_state in _SIGNATURE_STATE_NOTES:
            notes.append(
                f"signature_state={signature_state}: "
                f"{_SIGNATURE_STATE_NOTES[signature_state]}"
            )
    elif input_data.date is not None:
        notes.append(
            "No public key supplied, so this is a chain check only. The "
            "chain detects partial tampering; a wholesale rewrite can "
            "recompute it, and only a signed checkpoint catches that."
        )

    intact = not problems and signature_valid is not False
    if not intact:
        verdict = "tampered"
    elif day_files:
        verdict = "intact"
    elif recording_enabled:
        verdict = "no_trail"
    else:
        verdict = "recording_disabled"

    if verdict == "no_trail":
        notes.append(
            f"There are no day files under {directory}, so nothing was "
            "verified. An empty trail is not an intact one: recording is "
            "on, and this directory holds no record of any call."
        )
    elif verdict == "recording_disabled":
        notes.append(
            "SQT_AUDIT_ENABLED is off, so dispatch() writes no record and "
            "this directory is empty because nothing was ever recorded, not "
            "because nothing was tampered with. Set SQT_AUDIT_ENABLED=1 to "
            "have calls recorded."
        )
    elif not recording_enabled:
        notes.append(
            "SQT_AUDIT_ENABLED is off. The days already on disk were "
            "verified as reported, but calls made from now on leave no "
            "record."
        )

    logger.debug(
        "[verify_audit_integrity] scope=%s verdict=%s problems=%d",
        scope,
        verdict,
        len(problems),
    )
    return VerifyAuditIntegrityResult(
        scope=scope,
        intact=intact,
        verdict=verdict,
        recording_enabled=recording_enabled,
        problems=problems,
        checkpoint_signature_valid=signature_valid,
        signature_state=signature_state,
        notes=notes,
    )


def _contained_bundle_path(requested: str) -> Path:
    """
    Where an audit bundle may be written.

    TWO RULES, and neither is "must live in the sandbox". Exporting a bundle
    is for handing to someone outside this process, so an absolute
    destination is the point of the tool and confining it there would be
    wrong -- my first attempt did exactly that and broke the tests that use
    a tmp_path.

    What was actually wrong is narrower:

      1. IT OVERWROTE. `out_path` is a free string chosen by a model and the
         old code resolved it directly, noting "Overwrote an existing file
         at {out_path}" when it clobbered something. This is the only tool
         in the provenance set that writes; refusing an existing
         destination bounds the damage to creating new files.

      2. A BARE NAME LANDED IN THE WORKING DIRECTORY. Once the adversarial
         sweep began actually executing this tool -- which it only started
         doing when `strategy_type` stopped being a bare `str` -- it wrote
         two zips into the REPOSITORY ROOT named from its fuzz values,
         `zzz_not_a_valid_choice` and a Japanese/emoji filename, and they
         were committed. A relative name now resolves under the runs
         directory instead of wherever the process happens to be standing.
    """
    candidate = Path(requested)
    if candidate.is_absolute():
        resolved = candidate
        if not resolved.parent.exists():
            raise ValidationError(
                f"export_audit_bundle: the directory for out_path "
                f"{requested!r} does not exist. This tool writes a bundle; "
                "it does not create the tree around it."
            )
    else:
        root = Path(
            os.environ.get(
                "SQT_RUNS_DIR",
                str(Path.home() / ".cache" / "standard_quant_tools" / "runs"),
            )
        ).resolve()
        bundles = root / "bundles"
        resolved = (bundles / candidate).resolve()
        require_within(
            resolved,
            bundles,
            f"export_audit_bundle: out_path {requested!r} resolves to "
            f"{resolved}, which escapes {bundles}. Give a name, or an "
            "absolute path if the bundle belongs somewhere specific.",
        )
        resolved.parent.mkdir(parents=True, exist_ok=True)

    if resolved.exists():
        raise ValidationError(
            f"export_audit_bundle: {resolved} already exists and this will "
            "not overwrite it. An audit bundle is evidence; silently "
            "replacing one is exactly what the audit log exists to make "
            "impossible. Choose another name or remove that file yourself."
        )
    return resolved


def export_audit_bundle(
    input_data: ExportAuditBundleInput,
) -> ExportAuditBundleResult:
    """
    Package a date range of the audit log, plus the chain index, any signed
    checkpoints and a manifest, into one zip for handing to someone outside
    this process.

    This is the only tool in the provenance set that writes anything, and
    what it writes is a NEW file — no existing record is modified, moved or
    removed. Retention operations that could destroy evidence (gc, seal,
    hold) stay CLI-only on purpose.

    A range covering no day file is REFUSED. The manifest, the README and
    the standalone verifier weigh several kilobytes on their own, so a
    bundle of nothing came back with a plausible size and an ok status and
    was indistinguishable from a real export.
    """
    # Refused BEFORE anything is written, so a range that names no day
    # leaves no file behind to be mistaken for evidence.
    directory = _audit_dir()
    in_range = [
        p
        for p in _iter_day_files(directory)
        if input_data.start_date <= p.stem <= input_data.end_date
    ]
    if not in_range:
        raise ValidationError(
            f"export_audit_bundle: no audit day file falls in "
            f"{input_data.start_date}..{input_data.end_date}, so the bundle "
            "would hold a manifest, a README and a verifier and not one "
            "record. Call describe_audit_log to see which dates the log "
            "actually holds, then export a range that covers them."
        )
    # CONTAINED, like every other write in this library.
    #
    # `out_path` is a free string chosen by a model, and this is the only
    # tool in the provenance set that writes. It resolved that string
    # directly and wrote there -- so a bundle could land anywhere the
    # process can reach, and the branch below shows it OVERWRITES what it
    # finds. `backtest/artifacts.py` and `modeling/artifacts.py` both guard
    # exactly this shape with `_resolved_within_runs_dir`; this did not.
    #
    # Found because the adversarial sweep began actually executing this
    # tool once `strategy_type` stopped being a bare `str`, and it wrote
    # two zips into the repository root named from its fuzz values --
    # `zzz_not_a_valid_choice` and a Japanese/emoji filename -- which then
    # got committed.
    out_path = _contained_bundle_path(input_data.out_path)
    notes: List[str] = []
    exported = _export_bundle(input_data.start_date, input_data.end_date, out_path)
    written = Path(exported)
    size = int(written.stat().st_size)
    logger.debug(
        "[export_audit_bundle] wrote %s (%d bytes, %d days, %d records)",
        written,
        size,
        exported.day_files,
        exported.record_count,
    )
    notes.append(
        "The bundle is a copy. Verifying it proves the copy is internally "
        "consistent, not that the live log was untouched — run "
        "verify_audit_integrity against the log itself for that."
    )
    return ExportAuditBundleResult(
        out_path=str(written),
        start_date=input_data.start_date,
        end_date=input_data.end_date,
        size_bytes=size,
        day_files=exported.day_files,
        record_count=exported.record_count,
        notes=notes,
    )


def _artifact_location(uri: str) -> str:
    """
    A store KEY or a path, resolved to something `load_artifact` can open.

    `list_artifacts` hands back `<run_id>/<filename>`, which is how the
    artifact store addresses its own contents, and this tool resolved it
    with `Path(uri).resolve()` -- against the PROCESS WORKING DIRECTORY.
    The key then pointed outside the runs root, containment refused it, and
    the two tools that describe the same file could not exchange a name for
    it. A relative key is resolved against the store root first, which is
    where the file actually is.

    A key that tries to traverse is still refused: the store's own key rule
    rejects it before any path is built, and a relative path that is not a
    key falls through to the containment check unchanged.
    """
    if Path(uri).is_absolute():
        return uri
    from standard_quant_tools.artifact_store import LocalArtifactStore, validate_key

    try:
        key = validate_key(uri)
    except ValidationError:
        return uri
    if "/" not in key:
        # A bare run id is a PREFIX, not an artifact. Left alone so the
        # refusal names the missing file rather than a directory.
        return uri
    return LocalArtifactStore().uri(key)


def describe_artifact(input_data: DescribeArtifactInput) -> DescribeArtifactResult:
    """
    What is in a persisted artifact, without moving it into the conversation.

    Tools that write Parquet hand back a URI, and until now nothing could
    read one: the only way to learn what a run produced was to re-run it.
    This reports the shape, the date span, per-column summary statistics and
    the two ends of the frame — enough to decide what to do next.

    Takes either spelling of the same file: an absolute URI as a tool
    returned it, or the `<run_id>/<filename>` key `list_artifacts` reports.
    A relative key used to be resolved against the process working
    directory, so the store's own key format failed containment and the
    listing and the description had no name in common.

    The middle is never returned. `preview_rows` caps each end because the
    failure mode this tool exists to avoid is a five-year equity curve
    entering a client's context and taxing every turn after it.

    `content_hash` is the store's digest over the file's bytes, so two
    tools reading the same artifact can confirm they saw the same one and a
    re-run that changed it is visible without diffing anything.
    """
    from standard_quant_tools.artifact_store import hash_bytes

    location = _artifact_location(input_data.uri)
    frame = load_artifact(location)
    path = Path(location)
    # The store's digest, not a bare SHA-256: `list_artifacts` reports the
    # same 16 characters over the same bytes, and two hashes of one file
    # that never compare equal are worse than one.
    digest = hash_bytes(path.read_bytes())

    def _edge(rows: pd.DataFrame) -> List[Dict[str, Any]]:
        records = rows.reset_index().to_dict(orient="records")
        return [{str(k): _jsonable(v) for k, v in row.items()} for row in records]

    n = input_data.preview_rows
    head = _edge(frame.head(n)) if n else []
    tail = _edge(frame.tail(n)) if n and len(frame) > n else []

    summary: Dict[str, Dict[str, float]] = {}
    for column in frame.columns:
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        valid = series.dropna()
        summary[str(column)] = {
            "min": round(float(valid.min()), 6) if not valid.empty else float("nan"),
            "max": round(float(valid.max()), 6) if not valid.empty else float("nan"),
            "mean": round(float(valid.mean()), 6) if not valid.empty else float("nan"),
            "nan_count": float(int(series.isna().sum())),
        }

    index_start = str(frame.index[0]) if len(frame) else None
    index_end = str(frame.index[-1]) if len(frame) else None
    logger.debug(
        "[describe_artifact] %s rows=%d cols=%d",
        input_data.uri,
        len(frame),
        len(frame.columns),
    )
    return DescribeArtifactResult(
        uri=input_data.uri,
        rows=int(len(frame)),
        columns=[str(c) for c in frame.columns],
        index_name=str(frame.index.name) if frame.index.name is not None else None,
        index_start=index_start,
        index_end=index_end,
        content_hash=digest,
        head=head,
        tail=tail,
        column_summary=summary,
    )


#: Accepted `strategy_type` values that are not in STRATEGY_REGISTRY and
#: take no parameters. Kept beside the tool that reports them rather than
#: derived from BacktestInput's Literal, because that Literal mixes the two
#: kinds together and the difference is exactly what a caller needs to know.
_SYNTHETIC_STRATEGY_LABELS = ("buy_and_hold", "custom_signal")


def list_strategies(input_data: ListStrategiesInput) -> ListStrategiesResult:
    """
    Every built-in strategy and its parameter contract: names, kinds,
    defaults, bounds, and the cross-parameter relations that must hold.

    This reports STRATEGY_PARAM_SCHEMA itself, so it cannot drift from what
    the backtest engine will actually accept. Before this tool the same
    contract was available only as prose inside BacktestInput's field
    description — which meant a caller guessing `lookback=-20` learned the
    rule from a ValidationError after a round trip, if at all. The bounds
    are not stylistic: a negative window makes pandas look FORWARD, so it
    is look-ahead rather than a rejected input.
    """
    wanted = input_data.strategy_type
    if wanted is not None and wanted not in STRATEGY_PARAM_SCHEMA:
        raise ValidationError(
            f"Unknown strategy_type {wanted!r}. Available: "
            f"{sorted(STRATEGY_PARAM_SCHEMA)} (plus the parameterless labels "
            f"{list(_SYNTHETIC_STRATEGY_LABELS)})."
        )

    descriptors: List[StrategyDescriptor] = []
    for name, schema in STRATEGY_PARAM_SCHEMA.items():
        if wanted is not None and name != wanted:
            continue
        descriptors.append(
            StrategyDescriptor(
                name=name,
                parameters=[
                    StrategyParameter(
                        name=param,
                        kind=spec.kind,
                        default=spec.default,
                        minimum=1.0 if spec.kind == "window" else spec.minimum,
                        maximum=(
                            float(_MAX_WINDOW_BARS)
                            if spec.kind == "window"
                            else spec.maximum
                        ),
                    )
                    for param, spec in schema.items()
                ],
                relations=[
                    StrategyRelation(
                        left=left,
                        right=right,
                        requirement=f"{left} < {right}",
                        why=why,
                    )
                    for left, right, why in _RELATIONS.get(name, ())
                ],
            )
        )

    logger.debug("[list_strategies] returned %d strategies", len(descriptors))
    return ListStrategiesResult(
        strategies=descriptors,
        max_window_bars=_MAX_WINDOW_BARS,
        synthetic_labels=list(_SYNTHETIC_STRATEGY_LABELS),
    )


def list_stress_scenarios(
    input_data: ListStressScenariosInput,
) -> ListStressScenariosResult:
    """
    The named historical crash windows `run_stress_test` accepts.

    Offline and free — the table is a module constant. The windows are
    informal, widely-cited market-history dates rather than research-grade
    event-study boundaries, which is a reason to report them explicitly
    rather than have a caller infer them from a scenario's name.
    """
    scenarios = []
    for name, window in sorted(_library_stress_scenarios().items()):
        start = datetime.date.fromisoformat(window["start"])
        end = datetime.date.fromisoformat(window["end"])
        scenarios.append(
            StressScenario(
                name=name,
                start=window["start"],
                end=window["end"],
                calendar_days=(end - start).days,
            )
        )
    return ListStressScenariosResult(scenarios=scenarios)


#: A cache file's format-version prefix, `v3_...`. Matched here rather than
#: imported so this reports what is ON DISK, including generations the
#: reader no longer knows about.
_CACHE_GENERATION_RE = re.compile(r"^(v\d+)_")


def _cache_census() -> Dict[str, Any]:
    """
    What is in the persistent OHLCV cache: how much, and how much is dead.

    ONE directory listing, no reads and no deletions. Nothing on the tool
    surface could say anything about the cache beyond its path, so "is this
    the cache answering, and how much of it is stale" was a question an
    agent could only answer by leaving the tool surface -- while the live
    cache held 1,574 files, 501 of them written under a format version the
    reader no longer looks up.

    `dead_generations(dry_run=True)` is the COUNT. Removing them belongs to
    `sqt cache gc`: a describe tool that deleted files as a side effect of
    being asked a question would be the worst possible place to put it.
    An absent directory is zeros, not a refusal -- a cold cache is a normal
    state and not an error to report.
    """
    root = _cache_module._CACHE_ROOT
    if not root.exists():
        return {
            "cache_files": 0,
            "cache_bytes": 0,
            "cache_generations": [],
            "cache_dead_files": 0,
        }
    files = [p for p in root.glob("*.parquet") if p.is_file()]
    generations = sorted(
        {p.name.split("_", 1)[0] for p in files if _CACHE_GENERATION_RE.match(p.name)}
    )
    total = 0
    for path in files:
        try:
            total += int(path.stat().st_size)
        except OSError:  # a file evicted between the glob and the stat
            continue
    return {
        "cache_files": len(files),
        "cache_bytes": total,
        "cache_generations": generations,
        "cache_dead_files": len(dead_generations(dry_run=True)),
    }


def describe_data_capabilities(
    input_data: DataCapabilitiesInput,
) -> DataCapabilitiesResult:
    """
    What one data provider can actually serve — before a tool that needs it
    fails partway through an analysis.

    Capability is probed by asking whether the provider's class OVERRIDES
    the base method, not by calling it: `DataProvider.get_trades` raises
    NotImplementedError by design, so "does this provider have ticks" was
    otherwise only answerable by triggering that error. No market data is
    fetched. A provider that cannot even be constructed (no API key, SDK
    not installed) reports `available=False` with the reason, and the
    capability flags below it then describe the class rather than a live
    connection — which is still the right answer to "could I use ticks if I
    configured this?"
    """
    source = input_data.source.lower()
    notes: List[str] = []

    provider: Optional[DataProvider] = None
    available = True
    unavailable_reason: Optional[str] = None
    try:
        provider = DataFactory.get_provider(source)
    except (NotImplementedError, ValueError) as exc:
        # An unknown or unimplemented source is a caller error, not a
        # configuration state to report — there is no class to describe.
        raise ValidationError(str(exc)) from exc
    except Exception as exc:  # missing API key, uninstalled SDK
        available = False
        unavailable_reason = str(exc)

    # A provider that constructs without credentials and fails on its
    # first fetch (Databento reads DATABENTO_API_KEY lazily) reported
    # available=True; it says so itself now.
    if provider is not None and getattr(provider, "is_configured", True) is False:
        available = False
        unavailable_reason = (
            getattr(provider, "unconfigured_reason", None)
            or "the provider has no credentials configured"
        )
    provider_cls = type(provider) if provider is not None else _PROVIDER_CLASSES[source]

    def _overrides(method: str) -> bool:
        return getattr(provider_cls, method, None) is not getattr(
            DataProvider, method, None
        )

    trades = _overrides("get_trades")
    quotes = _overrides("get_quotes")
    # THE FOUR THAT ACTUALLY SEPARATE THE PROVIDERS, and the four this tool
    # did not report. Depth and order events are served by exactly one
    # provider here, point-in-time records by exactly one other -- so an
    # agent asked to consult this tool before choosing a source could not
    # learn the only facts that would have decided the choice.
    order_book = _overrides("get_order_book")
    order_events = _overrides("get_order_events")
    point_in_time_records = _overrides("get_point_in_time_records")
    temporal_contract = _overrides("get_temporal_contract")
    if not trades:
        notes.append(
            "No tick feed: the microstructure tools cannot run on this "
            "provider. Bar data is not a substitute — spreads and signed "
            "order flow are not recoverable from an OHLCV row, and nothing "
            "here synthesizes them."
        )
    if quotes:
        notes.append(
            "Quotes are TOP OF BOOK only. Only provider='databento' offers "
            "depth, through a separate get_order_book call, "
            "so queue position and resting size at a level are out of reach."
        )

    intervals = getattr(provider_cls, "SUPPORTED_INTERVALS", None)

    if provider is not None:
        metadata = provider.get_metadata("AAPL")
        guarantees = {
            "adjusted": metadata.adjusted,
            "survivorship_free": metadata.survivorship_free,
            "point_in_time": metadata.point_in_time,
        }
        if not metadata.point_in_time:
            notes.append(
                "point_in_time=False: historical values may be silently "
                "revised after the fact, so a backtest re-run on a later "
                "date can legitimately differ. verify_replay distinguishes "
                "that from a code change."
            )
    else:
        guarantees = {}
        notes.append(
            "Guarantees are unknown because the provider could not be "
            "constructed; they are reported by an instance, not the class."
        )

    cache = _cache_census()
    if cache["cache_dead_files"]:
        notes.append(
            f"{cache['cache_dead_files']} of {cache['cache_files']} cached "
            "file(s) were written under an earlier cache format and will "
            "never be read again. They are COUNTED here and not touched -- "
            "`sqt cache gc` is what removes them."
        )

    return DataCapabilitiesResult(
        provider=provider_cls.__name__,
        available=available,
        unavailable_reason=unavailable_reason,
        ohlcv=True,
        ohlcv_async=_overrides("get_ohlcv_async"),
        ticker_info=_overrides("get_ticker_info") or provider is not None,
        financial_ratios=_overrides("get_financial_ratios") or provider is not None,
        trades=trades,
        quotes=quotes,
        order_book=order_book,
        order_events=order_events,
        point_in_time_records=point_in_time_records,
        temporal_contract=temporal_contract,
        supported_intervals=sorted(intervals) if intervals else None,
        guarantees=guarantees,
        cache_dir=str(_cache_module._CACHE_ROOT),
        **cache,
        notes=notes,
    )


# ──────────────────────────────────────────────────────────────────
# Handoff references — inspecting and converting the interconnect
# ──────────────────────────────────────────────────────────────────


def describe_reference(
    input_data: DescribeReferenceInput,
) -> DescribeReferenceResult:
    """
    What a handoff reference points at, from any runtime.

    A reference is the unit of exchange between runtimes, so being able to
    ask what one holds without loading it into the conversation is what
    makes passing them around safe. The KIND is the useful part: it says
    which tools will accept this value, and it is checked on resolve so a
    mismatch fails by name rather than as a missing column three frames
    down.
    """
    from standard_quant_tools.agent.runtimes import handoff

    described = handoff.describe(input_data.ref)
    return DescribeReferenceResult(
        ref=described["ref"],
        kind=described["kind"],
        kind_description=described["description"],
        producer=described["producer"],
        rows=described["rows"],
        columns=described["columns"],
        index_start=described["index_start"],
        index_end=described["index_end"],
        # WHICH VENDOR DATASET IS IN THERE. The provider writes it onto the
        # frame, Parquet round-trips it, and the answer survives the
        # process boundary this tool exists to be called across -- so the
        # agent that RESOLVES a reference can learn what the agent that
        # fetched it was answered by. Null for an externally registered
        # dataset, which carries no such attributes.
        dataset=described.get("dataset"),
        provider=described.get("provider"),
        adjusted=described.get("adjusted"),
        source=described.get("source"),
    )


#: The cap on `read_reference`. References exist to keep bulk values OUT of
#: the conversation; a reader that could return ten thousand rows would undo
#: the thing references are for. Sized to hold a quarter of dailies.
_READ_REFERENCE_MAX_ROWS = 64


def read_reference(input_data: ReadReferenceInput) -> ReadReferenceResult:
    """
    The actual values at chosen rows of a handoff reference.

    `describe_reference` says what a reference holds; this says what is IN
    it. Both exist because the honest answer to "what was NVDA's close on
    2026-06-04" is a number, and until this tool there was no way to get one
    out of a published frame -- the data crossed runtimes perfectly and was
    unreadable by the agent that fetched it.

    Bounded on purpose: ask for the rows you intend to cite, by date where
    you know them. Over the cap the window is truncated and says so.
    """
    from standard_quant_tools.agent.runtimes import handoff

    reference = handoff.parse(input_data.ref)
    frame = handoff.resolve(input_data.ref)

    # `resolve` returns whatever the kind carries. Only a table has rows to
    # read; say so by name rather than failing on a missing attribute three
    # frames down.
    if not hasattr(frame, "index") or not hasattr(frame, "columns"):
        raise ValueError(
            f"{input_data.ref} carries kind '{reference.kind}', which is not "
            "tabular — there are no rows to read. Use describe_reference."
        )

    total = int(len(frame))
    columns = [str(c) for c in frame.columns]
    missing_columns: list[str] = []
    if input_data.columns:
        wanted = [c for c in input_data.columns if c in columns]
        missing_columns = [c for c in input_data.columns if c not in columns]
        frame = frame[wanted]
        columns = wanted

    labels = [str(label) for label in frame.index]
    missing: list[str] = []
    if input_data.dates:
        # Match on the rendered label so a caller can pass "2026-06-04"
        # against a timestamp index without knowing its dtype.
        by_label = {label: position for position, label in enumerate(labels)}
        positions = []
        for wanted_date in input_data.dates:
            position = by_label.get(wanted_date)
            if position is None:
                # A date index renders as "2026-06-04 00:00:00"; accept the
                # date alone, which is how anyone would ask for it.
                matches = [
                    i for i, label in enumerate(labels) if label.startswith(wanted_date)
                ]
                position = matches[0] if matches else None
            if position is None:
                missing.append(wanted_date)
            else:
                positions.append(position)
    else:
        positions = list(range(min(input_data.head, total)))
        if input_data.tail:
            positions += list(
                range(max(total - input_data.tail, len(positions)), total)
            )

    positions = sorted(set(positions))
    truncated = len(positions) > _READ_REFERENCE_MAX_ROWS
    positions = positions[:_READ_REFERENCE_MAX_ROWS]

    rows: list[dict] = []
    for position in positions:
        row: dict = {"index": labels[position]}
        for column in columns:
            value = frame.iloc[position][column]
            # A JSON result must not carry numpy scalars or NaN.
            try:
                value = value.item()
            except AttributeError:
                pass
            if isinstance(value, float) and value != value:
                value = None
            row[column] = value
        rows.append(row)

    return ReadReferenceResult(
        ref=reference.ref,
        kind=reference.kind,
        columns=columns,
        rows=rows,
        total_rows=total,
        returned=len(rows),
        truncated=truncated,
        missing=missing,
        missing_columns=missing_columns,
    )


def list_reference_kinds(
    input_data: ListReferenceKindsInput,
) -> ListReferenceKindsResult:
    """
    Every content kind a reference can carry, and what converts to what.

    This is the map of the interconnect: it says which producer outputs can
    reach which consumer inputs, and by what route. Offline.
    """
    from standard_quant_tools.agent.runtimes import handoff
    from standard_quant_tools.agent.runtimes.meta.convert import CONVERSIONS

    targets: Dict[str, List[str]] = {}
    for source, destination in CONVERSIONS:
        targets.setdefault(source, []).append(destination)
    return ListReferenceKindsResult(
        kinds=[
            ReferenceKind(
                kind=kind,
                description=description,
                convertible_to=sorted(targets.get(kind, [])),
            )
            for kind, description in sorted(handoff.kinds().items())
        ]
    )


def convert_reference(input_data: ConvertReferenceInput) -> ConvertReferenceResult:
    """
    Turn one kind of published value into another, and publish the result.

    This is the general form of what would otherwise be a bridge tool per
    producer/consumer pair. With N producers and M consumers, bridges cost
    N x M and every one of them has to be kept in step with both ends;
    conversion between KINDS costs N + M, and a producer never has to know
    which consumer will eventually read it.

    The conversions are the ones that are genuinely well defined. Turning
    raw predictions into a signal panel discards magnitude on purpose,
    because the engine that consumes a signal panel reads a value as a
    leverage multiplier — a 0.02 forward-return prediction passed through
    unchanged would size a 2%-leveraged position. Turning a score panel
    into weights goes through backtest.sizing rather than reimplementing
    it, so a converted panel is the same object that tool would have built.
    """
    from standard_quant_tools.agent.runtimes import handoff
    from standard_quant_tools.agent.runtimes.meta.convert import convert

    source = handoff.parse(input_data.ref)
    converted, notes = convert(input_data, source)

    ref = handoff.publish(
        converted,
        input_data.to_kind,
        input_data.run_id,
        input_data.name,
        producer="meta.convert_reference",
    )
    entities = len(converted) if isinstance(converted, dict) else len(converted.columns)
    rows = (
        len({date for per_entity in converted.values() for date in per_entity})
        if isinstance(converted, dict)
        else len(converted)
    )
    return ConvertReferenceResult(
        source_ref=input_data.ref,
        source_kind=source.kind,
        ref=ref,
        kind=input_data.to_kind,
        rows=rows,
        entities=entities,
        notes=notes,
    )


# ──────────────────────────────────────────────────────────────────
# Pre-flight — describe one tool, and check a call before making it
# ──────────────────────────────────────────────────────────────────


def describe_tool(input_data: DescribeToolInput) -> DescribeToolResult:
    """
    One tool's contract: what it takes, what it returns, which runtime can
    run it, and whether calling it will go and fetch data.

    The alternative was loading all 73 schemas, which is exactly what the
    MCP category budget exists to avoid — so an agent given a narrow tool
    list had no way to find out about a tool it had heard of without
    paying for every tool it had not.

    Describing a tool is not calling it, so this answers for tools in any
    runtime, including ones the caller is not scoped to. That is the point:
    the answer to "why was that refused" is a description, not a wider
    scope.
    """
    from standard_quant_tools.mcp.catalog import build_catalog

    catalog = build_catalog()
    entry = catalog.get(input_data.tool_name)
    if entry is None:
        from difflib import get_close_matches

        near = get_close_matches(input_data.tool_name, sorted(catalog), n=3)
        suggestion = f" Did you mean: {near}?" if near else ""
        raise ValidationError(
            f"no tool named {input_data.tool_name!r} in any runtime.{suggestion}"
        )

    schema = entry.input_schema
    properties = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    result_fields: List[str] = []
    if entry.output_schema:
        result_fields = sorted((entry.output_schema.get("properties", {}) or {}))

    return DescribeToolResult(
        tool_name=entry.name,
        runtime=entry.runtime,
        category=entry.category,
        description=entry.description,
        required_arguments=sorted(required),
        optional_arguments=sorted(set(properties) - required),
        reads_market_data=entry.reads_market_data,
        persists_artifact=entry.persists_artifact,
        input_schema=schema if input_data.include_schema else None,
        result_fields=result_fields,
    )


#: Tools whose `parameters` dict is validated by the strategy contract
#: rather than by the JSON schema. The schema types it as an open dict, so
#: a bad window passes schema validation and fails only once the data has
#: been fetched -- which is the round trip this tool exists to save.
_STRATEGY_PARAM_TOOLS = ("strategy_type", "parameters")

#: The three keys a polymorphic data source dumps to. A dict carrying all
#: of them is one wherever it appears, which is what lets the numeric
#: contract be applied without the tool's own field types being consulted.
_DATA_SOURCE_KEYS = frozenset({"symbol", "ref", "values"})

#: Field-name fragments that mean the inline numbers are PRICES or LEVELS
#: rather than returns, and so get the stronger rule: strictly positive
#: rather than merely finite. Matched on the name because the contract is a
#: property of what the numbers MEAN, and the annotation says only that a
#: source was passed.
_PRICE_LIKE_FIELDS = ("price", "close", "equity", "level")


def _inline_numeric_payloads(node: Any, field: str = "(tool)") -> List[tuple]:
    """
    Every number already present in a proposed call that the numeric
    contract has something to say about.

    Two shapes only, and deliberately so: a data source carrying literal
    `values`, and an annualization factor. A source naming a symbol or a
    reference is NOT checked here, because checking it would mean fetching
    or resolving -- and a validator that fetched would defeat its own
    purpose. What can be checked without touching anything is checked.
    """
    found: List[tuple] = []
    if isinstance(node, dict):
        if _DATA_SOURCE_KEYS <= set(node) and node.get("values") is not None:
            found.append(("series", field, node["values"]))
        for key, value in node.items():
            if key == "periods_per_year" and value is not None:
                found.append(("periods_per_year", key, value))
            else:
                found.extend(_inline_numeric_payloads(value, key))
    elif isinstance(node, list):
        for item in node:
            found.extend(_inline_numeric_payloads(item, field))
    return found


def validate_tool_call(input_data: ValidateToolCallInput) -> ValidateToolCallResult:
    """
    Check arguments against a tool's contract WITHOUT calling it.

    A wrong argument is otherwise discovered by making the call: at best a
    round trip, and for anything that fetches, a network fetch and possibly
    a full backtest before the error appears. Worse, an unknown argument
    name — the usual shape of a hallucinated one — is the cheapest mistake
    to make and among the more expensive to diagnose from a stack trace.

    Three layers are checked, because the library has three. The Pydantic
    schema catches missing, unknown and out-of-range arguments. Then, for
    tools that carry a strategy `parameters` dict, the strategy's own
    contract is checked as well — that layer is invisible to the JSON
    schema, which types `parameters` as an open dict, so `lookback=-20`
    would pass a schema check and still be look-ahead by construction.

    The third is the numerical contract every boundary enforces, run
    against numbers ALREADY PRESENT in the call: a data source carrying
    literal values, an annualization factor. An all-NaN series passed a
    schema check cleanly and then failed at execution with "contains no
    observations", after the rest of the call had been paid for.
    `describe_numeric_contract` states the rules this layer applies.

    Nothing here fetches, runs or writes — which is why a source naming a
    symbol or a reference is left unchecked rather than resolved.
    """
    from pydantic import ValidationError as PydanticValidationError

    from standard_quant_tools.agent.tools import _TOOL_DISPATCH
    from standard_quant_tools.modeling.agent import MODELING_TOOL_DISPATCH
    from standard_quant_tools.modeling.agent.feature_tools import (
        FEATURE_TOOL_DISPATCH,
    )

    every = {**_TOOL_DISPATCH, **MODELING_TOOL_DISPATCH, **FEATURE_TOOL_DISPATCH}
    entry = every.get(input_data.tool_name)
    if entry is None:
        from difflib import get_close_matches

        near = get_close_matches(input_data.tool_name, sorted(every), n=3)
        suggestion = f" Did you mean: {near}?" if near else ""
        raise ValidationError(
            f"no tool named {input_data.tool_name!r} in any runtime.{suggestion}"
        )

    _fn, model_cls = entry
    problems: List[ArgumentProblem] = []
    notes: List[str] = []
    normalized: Dict[str, Any] = {}

    try:
        instance = model_cls(**input_data.arguments)
        normalized = instance.model_dump()
    except PydanticValidationError as exc:
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "(tool)"
            kind = {
                "missing": "missing",
                "extra_forbidden": "unknown",
            }.get(error["type"], "invalid")
            if error["type"] == "value_error" and not error["loc"]:
                kind = "relation"
            problems.append(
                ArgumentProblem(
                    field=location, problem=error["msg"], kind=kind  # type: ignore[arg-type]
                )
            )
    except Exception as exc:  # a validator that raises something else
        problems.append(
            ArgumentProblem(field="(tool)", problem=str(exc), kind="invalid")
        )

    # Second layer: the strategy parameter contract.
    checked_strategy = False
    fields = set(model_cls.model_fields)
    if all(name in fields for name in _STRATEGY_PARAM_TOOLS) and not problems:
        strategy = normalized.get("strategy_type")
        parameters = normalized.get("parameters") or {}
        if strategy in STRATEGY_PARAM_SCHEMA:
            checked_strategy = True
            try:
                resolve_strategy_params(strategy, parameters)
            except ValidationError as exc:
                # Only a ValidationError is the CALLER's problem. A broader
                # catch here would report an internal failure as a bad
                # argument, sending the caller to fix something that is not
                # wrong -- the worst possible advice from a validator.
                problems.append(
                    ArgumentProblem(
                        field="parameters", problem=str(exc), kind="invalid"
                    )
                )
        elif strategy is not None:
            notes.append(
                f"strategy_type={strategy!r} takes no parameters, so the "
                "`parameters` dict was not checked against a contract."
            )

    # Third layer: the numerical contract, on the numbers that are already
    # here. Nothing is fetched and nothing is resolved to get them.
    checked_numeric = False
    if normalized:
        from standard_quant_tools.numeric_contract import (
            require_finite_series,
            require_periods_per_year,
            require_positive_price_series,
        )

        for kind, field, value in _inline_numeric_payloads(normalized):
            checked_numeric = True
            try:
                if kind == "periods_per_year":
                    require_periods_per_year(value, input_data.tool_name)
                    continue
                try:
                    series = pd.Series(value, dtype="float64")
                except (TypeError, ValueError):
                    # Not numbers at all. The schema layer above owns that
                    # verdict; the contract has nothing to say about it and
                    # must not report its own confusion as a second fault.
                    checked_numeric = False
                    continue
                if any(token in field.lower() for token in _PRICE_LIKE_FIELDS):
                    require_positive_price_series(series, field, input_data.tool_name)
                else:
                    require_finite_series(series, field, input_data.tool_name)
            except ValidationError as exc:
                # Only the contract's own refusal is the caller's problem.
                # Anything else here would be this validator failing, and
                # reporting that as a bad argument is the worst advice a
                # validator can give.
                problems.append(
                    ArgumentProblem(field=field, problem=str(exc), kind="invalid")
                )
        if checked_numeric:
            notes.append(
                "The numerical contract was run on the values written into "
                "this call. A source naming a symbol or a reference was not "
                "checked, because checking it would mean fetching or "
                "resolving it — see describe_numeric_contract for the rules."
            )

    if not problems:
        notes.append(
            "Valid. normalized_arguments shows what the tool would actually "
            "receive, defaults included — worth reading, since it is often "
            "not quite what was written."
        )

    logger.debug(
        "[validate_tool_call] %s valid=%s problems=%d",
        input_data.tool_name,
        not problems,
        len(problems),
    )
    return ValidateToolCallResult(
        tool_name=input_data.tool_name,
        valid=not problems,
        problems=problems,
        normalized_arguments=normalized if not problems else {},
        checked_strategy_parameters=checked_strategy,
        checked_numeric_contract=checked_numeric,
        notes=notes,
    )


def describe_temporal_contract(
    input_data: TemporalContractInput,
) -> TemporalContractResult:
    """
    What a data source can say about WHEN its facts became knowable — asked
    BEFORE fetching anything.

    Every non-price dataset carries a leak waiting to happen. A quarterly
    filing describes 30 September and is published on 25 October, so a model
    that joins it on 30 September has three weeks of hindsight in every row,
    and the backtest that results looks like skill rather than like a bug.

    The point-in-time join already refuses a frame with no `available_time`.
    That refusal arrives late — after a universe has been chosen, a history
    fetched and a cache written. This answers the same question first, in
    one call, and fetches nothing.

    Read `pit_safe` first: False means do not build this dataset from this
    source. Then `reproduces_history`, which is stricter and comes apart
    from it — a snapshot source joins without leaking the future and still
    shows a backtest numbers that were later restated.
    """
    from standard_quant_tools.data.factory import DataFactory

    logger.debug(
        "[describe_temporal_contract] source=%s kind=%s",
        input_data.source,
        input_data.frame_kind,
    )
    provider = DataFactory.get_provider(input_data.source)
    contract = provider.get_temporal_contract(input_data.frame_kind)
    return TemporalContractResult(
        source=contract.source,
        frame_kind=contract.frame_kind,
        has_event_time=contract.has_event_time,
        has_available_time=contract.has_available_time,
        entity_scoped=contract.entity_scoped,
        revisions=contract.revisions,
        pit_safe=contract.pit_safe,
        reproduces_history=contract.reproduces_history,
        caveats=contract.caveats(),
    )


def compare_data_sources(
    input_data: CompareDataSourcesInput,
) -> CompareDataSourcesResult:
    """
    Fetch the same fundamentals from two providers and report where they
    disagree — separating three cases that look identical in a diff.

    `FinancialRatios` already documents that `debt_to_equity` means
    different things depending on where it came from: Polygon derives it
    from total LIABILITIES, which include payables and deferred revenue, so
    it is systematically higher for reasons unrelated to leverage. That is
    written down in a docstring somebody has to read, and nothing checks it.
    A screen that ranks a universe on `debt_to_equity` fetched from two
    providers is ordering it partly by which provider answered, and no error
    appears anywhere.

    The three verdicts need different responses:

    - **scale** — a constant ratio, so a unit conversion was missed. The fix
      is arithmetic.
    - **definition** — systematic with NO constant ratio, so the two are
      computing different quantities. No conversion exists; one has to be
      chosen deliberately and recorded.
    - **agree** — within rounding. Vendors differ at the margin about
      everything and that is not a finding.

    Also surfaces `declared_definition_notes`: differences the providers
    declared about themselves. A declared difference is not a bug, and is
    not convertible either.
    """
    from standard_quant_tools.data.comparison import compare_ratio_sources
    from standard_quant_tools.data.factory import DataFactory

    logger.debug(
        "[compare_data_sources] %s vs %s on %d symbol(s)",
        input_data.left,
        input_data.right,
        len(input_data.symbols),
    )

    unavailable: List[str] = []
    fetched: Dict[str, Dict[str, Any]] = {}
    for name in (input_data.left, input_data.right):
        try:
            provider = DataFactory.get_provider(name)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            unavailable.append(f"{name}: {exc}")
            continue
        got: Dict[str, Any] = {}
        for symbol in input_data.symbols:
            try:
                got[symbol] = provider.get_financial_ratios(symbol)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[compare_data_sources] %s/%s: %s", name, symbol, exc)
        fetched[name] = got

    if unavailable:
        # A comparison against a provider that never answered is not a
        # comparison, and returning an empty "they agree" would be worse
        # than saying nothing.
        return CompareDataSourcesResult(
            left=input_data.left,
            right=input_data.right,
            n_entities_compared=0,
            fields=[],
            warnings=[
                "no comparison was made: "
                + "; ".join(unavailable)
                + ". Configure the provider or pick two that are available."
            ],
            unavailable=unavailable,
        )

    report = compare_ratio_sources(
        fetched[input_data.left],
        fetched[input_data.right],
        left_name=input_data.left,
        right_name=input_data.right,
        fields=input_data.fields,
    )
    return CompareDataSourcesResult(
        left=report["left"],
        right=report["right"],
        n_entities_compared=report["n_entities_compared"],
        fields=[FieldDivergence(**f) for f in report["fields"]],
        declared_definition_notes=[
            DeclaredNote(**n) for n in report["declared_definition_notes"]
        ],
        warnings=report["warnings"],
    )
