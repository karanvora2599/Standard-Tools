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
import hashlib
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

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
from standard_quant_tools.audit.paths import _audit_dir
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
from standard_quant_tools.data._cache import _CACHE_ROOT
from standard_quant_tools.data.base import DataProvider
from standard_quant_tools.data.bloomberg_provider import BloombergProvider
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.data.polygon_provider import PolygonProvider
from standard_quant_tools.data.yfinance_provider import YFinanceProvider
from standard_quant_tools.error import ValidationError

#: Provider classes by source name, for describing one that cannot be
#: constructed here (no API key, SDK absent). DataFactory raises in that
#: case rather than returning an instance, and "you would need a key" is a
#: more useful answer than the raise.
_PROVIDER_CLASSES: Dict[str, type] = {
    "yfinance": YFinanceProvider,
    "polygon": PolygonProvider,
    "bloomberg": BloombergProvider,
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
    """
    record = _find_audit_record(input_data.request_id)
    sources = [
        DataSourceRef(
            symbol=source.get("symbol"),
            start_date=source.get("start") or source.get("start_date"),
            end_date=source.get("end") or source.get("end_date"),
            rows=source.get("rows"),
            content_hash=source.get("content_hash") or source.get("hash"),
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
        output_hash=record.get("output_hash"),
        git_commit_sha=record.get("git_commit_sha"),
        package_version=record.get("package_version"),
        random_seed=record.get("random_seed"),
        error_type=record.get("error_type"),
        error_message=record.get("error_message"),
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
    """
    notes: List[str] = []
    signature_valid: Optional[bool] = None

    if input_data.date is None:
        problems = list(_verify_trail())
        scope = "trail"
    else:
        # Day files are named YYYY-MM-DD.jsonl under the audit dir. Built
        # from the validated date rather than joined from raw input: this
        # argument is LLM-reachable and goes into a filesystem path.
        if not _DATE_RE.match(input_data.date):
            raise ValidationError(f"date={input_data.date!r} must be YYYY-MM-DD.")
        path = _audit_dir() / f"{input_data.date}.jsonl"
        if not path.exists():
            raise ValidationError(
                f"no audit file for {input_data.date}. Verify the whole "
                "trail (omit `date`) to see which days exist."
            )
        problems = list(_verify_day(path))
        scope = input_data.date
        notes.append(
            "A single day verified in isolation cannot detect a MISSING "
            "day. Omit `date` to verify the cross-day trail as well."
        )

    if input_data.public_key_path is not None:
        from standard_quant_tools.audit.signing import verify_checkpoint_signature

        try:
            signature_valid = bool(
                verify_checkpoint_signature(
                    input_data.date, input_data.public_key_path  # type: ignore[arg-type]
                )
            )
        except Exception as exc:
            signature_valid = False
            notes.append(f"checkpoint signature could not be verified: {exc}")
    elif input_data.date is not None:
        notes.append(
            "No public key supplied, so this is a chain check only. The "
            "chain detects partial tampering; a wholesale rewrite can "
            "recompute it, and only a signed checkpoint catches that."
        )

    intact = not problems and signature_valid is not False
    logger.debug(
        "[verify_audit_integrity] scope=%s intact=%s problems=%d",
        scope,
        intact,
        len(problems),
    )
    return VerifyAuditIntegrityResult(
        scope=scope,
        intact=intact,
        problems=problems,
        checkpoint_signature_valid=signature_valid,
        notes=notes,
    )


def export_audit_bundle(
    input_data: ExportAuditBundleInput,
) -> ExportAuditBundleResult:
    """
    Package a date range of the audit log, plus the chain index and a
    manifest, into one zip for handing to someone outside this process.

    This is the only tool in the provenance set that writes anything, and
    what it writes is a NEW file — no existing record is modified, moved or
    removed. Retention operations that could destroy evidence (gc, seal,
    hold) stay CLI-only on purpose.
    """
    out_path = Path(input_data.out_path)
    notes: List[str] = []
    if out_path.exists():
        notes.append(f"Overwrote an existing file at {out_path}.")
    written = _export_bundle(input_data.start_date, input_data.end_date, out_path)
    size = int(Path(written).stat().st_size)
    logger.debug("[export_audit_bundle] wrote %s (%d bytes)", written, size)
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
        notes=notes,
    )


def describe_artifact(input_data: DescribeArtifactInput) -> DescribeArtifactResult:
    """
    What is in a persisted artifact, without moving it into the conversation.

    Tools that write Parquet hand back a URI, and until now nothing could
    read one: the only way to learn what a run produced was to re-run it.
    This reports the shape, the date span, per-column summary statistics and
    the two ends of the frame — enough to decide what to do next.

    The middle is never returned. `preview_rows` caps each end because the
    failure mode this tool exists to avoid is a five-year equity curve
    entering a client's context and taxing every turn after it.

    `content_hash` is over the file's bytes, so two tools reading the same
    URI can confirm they saw the same artifact and a re-run that changed it
    is visible without diffing anything.
    """
    frame = load_artifact(input_data.uri)
    path = Path(input_data.uri)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

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

    provider_cls = type(provider) if provider is not None else _PROVIDER_CLASSES[source]

    def _overrides(method: str) -> bool:
        return getattr(provider_cls, method, None) is not getattr(
            DataProvider, method, None
        )

    trades = _overrides("get_trades")
    quotes = _overrides("get_quotes")
    if not trades:
        notes.append(
            "No tick feed: the microstructure tools cannot run on this "
            "provider. Bar data is not a substitute — spreads and signed "
            "order flow are not recoverable from an OHLCV row, and nothing "
            "here synthesizes them."
        )
    if quotes:
        notes.append(
            "Quotes are TOP OF BOOK only. No shipped provider offers depth, "
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
        supported_intervals=sorted(intervals) if intervals else None,
        guarantees=guarantees,
        cache_dir=str(_CACHE_ROOT),
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


def validate_tool_call(input_data: ValidateToolCallInput) -> ValidateToolCallResult:
    """
    Check arguments against a tool's contract WITHOUT calling it.

    A wrong argument is otherwise discovered by making the call: at best a
    round trip, and for anything that fetches, a network fetch and possibly
    a full backtest before the error appears. Worse, an unknown argument
    name — the usual shape of a hallucinated one — is the cheapest mistake
    to make and among the more expensive to diagnose from a stack trace.

    Two layers are checked, because the library has two. The Pydantic
    schema catches missing, unknown and out-of-range arguments. Then, for
    tools that carry a strategy `parameters` dict, the strategy's own
    contract is checked as well — that layer is invisible to the JSON
    schema, which types `parameters` as an open dict, so `lookback=-20`
    would pass a schema check and still be look-ahead by construction.

    Nothing here fetches, runs or writes.
    """
    from pydantic import ValidationError as PydanticValidationError

    from standard_quant_tools.agent.tools import _TOOL_DISPATCH
    from standard_quant_tools.modeling.agent import MODELING_TOOL_DISPATCH

    every = {**_TOOL_DISPATCH, **MODELING_TOOL_DISPATCH}
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
