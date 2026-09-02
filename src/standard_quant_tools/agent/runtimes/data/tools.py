"""
The `data` runtime: fetch once, publish a reference, let every other runtime
read it.

WHAT THIS RUNTIME IS FOR. Before it, the raw data layer was reachable only
by going through an analysis tool that wanted to do something else with the
bars. Two agents asking about the same universe fetched it twice, and the
frame each one built died inside the call that built it -- so a return
panel, a tick tape or an OHLCV panel could never be handed to the next
runtime without being recomputed from scratch.

These tools do the fetch and return an `sqt://` reference rather than the
data. That is the whole point: a reference crosses runtimes and processes,
survives the boundary between two agents that cannot see each other's
context, and shows up in the audit log as an input to whatever consumed it.
Returning the frame inline would put megabytes into a conversation that
then carries them on every subsequent turn.

WHAT IT IS NOT. It is not a second home for data QUALITY checks --
`get_data_quality_report` in `research` already reports missing bars, stale
prices and price jumps, and a second name for those would be exactly the
confusable duplication the runtime split exists to avoid. It also does not
fetch order books: `DataProvider.get_order_book` is a declared contract that
no shipped provider implements, and a tool that always refuses is worse than
no tool.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from standard_quant_tools.data.bundle import DataBundle, validate_bundle
from standard_quant_tools.data.comparison import compare_ratio_sources
from standard_quant_tools.data.continuous import build_continuous_futures
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.data.ratios import implausible_value_warnings
from standard_quant_tools.data.temporal import contract_for_frame
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.portfolio import (
    fetch_ohlcv_panel_sync,
    fetch_returns_sync,
)

from ..handoff import publish, resolve
from .models import (
    BuildDataBundleInput,
    CompareRatioFramesInput,
    ContinuousFuturesInput,
    DataBundleRefInput,
    DatasetMetadataInput,
    FetchFinancialRatiosInput,
    FetchOhlcvInput,
    FetchOhlcvPanelInput,
    FetchQuotePanelInput,
    FetchReturnsPanelInput,
    FetchTickTapeInput,
    InferTemporalContractInput,
    ValidateDataBundleInput,
    ValidateFinancialRatiosInput,
)
from .results import (
    BundleFrameSummary,
    BundleVerdictResult,
    ContinuousFuturesResult,
    DataBundleResult,
    DatasetMetadataResult,
    FetchResult,
    FinancialRatiosResult,
    RatioComparisonResult,
    RatioFieldComparison,
    TemporalContractResult,
)

logger = logging.getLogger(__name__)

#: The manifest columns a `data_bundle` reference stores. Fixed here rather
#: than inferred, because describe/validate read this frame back and a
#: column that quietly changed name would surface as a missing frame.
_BUNDLE_COLUMNS = ("frame_kind", "ref", "source")


def _resolved(ref: str, expect: Optional[str] = None) -> Any:
    """
    Resolve a reference, turning every way it can go wrong into a refusal
    that names the reference.

    `handoff.resolve` is written for a well-formed `sqt://` string and
    raises whatever the parse or the artifact store raises for anything
    else -- an AttributeError from inside the loader tells a caller nothing
    about which argument was wrong. The surface contract is that a tool
    either returns a result or refuses BY NAME, so the translation happens
    here rather than three frames down.
    """
    try:
        return resolve(ref, expect=expect) if expect else resolve(ref)
    except (ValidationError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001 -- any failure is one refusal
        raise ValidationError(
            f"{ref!r} could not be resolved as an artifact reference"
            + (f" of kind {expect!r}" if expect else "")
            + f": {exc}. References look like `sqt://<kind>/<run_id>/<name>` "
            "and come from the tool that published them -- list_reference_"
            "kinds says what kinds exist."
        ) from exc


def _fetched(what: str, call, tool: str) -> pd.DataFrame:
    """
    Run a provider fetch, translating a missing capability into a refusal.

    `DataProvider.get_trades` and `get_quotes` raise NotImplementedError
    with a good message when the active provider has no such feed, and most
    environments have none. That is a precondition this tool failed to
    meet, not a crash -- and the surface contract does not accept a bare
    NotImplementedError, because a caller cannot tell it from a bug.
    """
    try:
        return call()
    except NotImplementedError as exc:
        raise ValidationError(
            f"{tool} needs {what}, and the active provider does not serve "
            f"it: {exc} Call describe_data_capabilities to see what this "
            "environment can actually reach before trying again."
        ) from exc


def _span(frame: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """First and last index label as dates, when the index carries them."""
    if frame is None or len(frame) == 0:
        return None, None
    try:
        index = pd.to_datetime(pd.Index(frame.index))
        return str(index.min().date()), str(index.max().date())
    except Exception:  # noqa: BLE001 -- a non-datetime index is not an error
        return None, None


def _published(
    frame: pd.DataFrame,
    kind: str,
    run_id: str,
    name: str,
    producer: str,
    entities: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> FetchResult:
    """Publish a frame and describe what was published."""
    if frame is None or len(frame) == 0:
        raise ValidationError(
            f"{producer} fetched no rows. An empty panel published as a "
            "reference would be indistinguishable downstream from one whose "
            "data simply had not arrived yet."
        )
    ref = publish(frame, kind=kind, run_id=run_id, name=name, producer=producer)
    start, end = _span(frame)
    return FetchResult(
        ref=ref,
        kind=kind,
        rows=int(len(frame)),
        columns=[str(c) for c in frame.columns],
        entities=sorted(entities or []),
        start=start,
        end=end,
        warnings=list(warnings or []),
    )


def fetch_ohlcv(input_data: FetchOhlcvInput) -> FetchResult:
    """One symbol's OHLCV bars, published as a `price_panel` reference."""
    provider = DataFactory.get_provider()
    frame = provider.get_ohlcv(
        input_data.symbol,
        input_data.start_date,
        input_data.end_date,
        input_data.interval,
    )
    return _published(
        frame,
        "price_panel",
        input_data.run_id,
        input_data.name,
        "fetch_ohlcv",
        entities=[input_data.symbol],
    )


def fetch_ohlcv_panel(input_data: FetchOhlcvPanelInput) -> FetchResult:
    """A whole universe's OHLCV, stacked long and published once."""
    # ONE BAD TICKER FAILS THE BATCH, and that is the helper's behaviour
    # rather than a choice made here: `fetch_ohlcv_panel_async` gathers
    # without `return_exceptions`, so the first failure propagates and no
    # partial panel is produced. Translating it names the universe, because
    # the raw error names only the symbol that happened to raise first and
    # a caller cannot tell from it whether the other forty are fine.
    try:
        by_symbol: Dict[str, pd.DataFrame] = fetch_ohlcv_panel_sync(
            list(input_data.tickers),
            input_data.start_date,
            input_data.end_date,
            input_data.interval,
        )
    except (ValidationError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001 -- one refusal, not a traceback
        raise ValidationError(
            f"fetching {len(input_data.tickers)} ticker(s) failed on one of "
            f"them: {exc}. The whole batch fails together -- there is no "
            "partial panel -- so drop the symbol that cannot be fetched and "
            "run it again rather than expecting the rest to arrive."
        ) from exc

    warnings: List[str] = []
    # A ticker whose frame comes back EMPTY is dropped here rather than
    # stacked. That is the reachable case: the fetch succeeded and returned
    # nothing, which is different from the fetch raising.
    empty = sorted(t for t, f in by_symbol.items() if f is None or len(f) == 0)
    if empty:
        warnings.append(
            f"{len(empty)} ticker(s) returned an empty frame and are ABSENT "
            f"from the panel rather than present as NaN: {empty}. A "
            "downstream complete-case join will not see they were ever "
            "requested -- the panel just looks like a smaller universe."
        )
    stacked = []
    for symbol, frame in by_symbol.items():
        if frame is None or len(frame) == 0:
            continue
        part = frame.copy()
        part["entity"] = symbol
        stacked.append(part)
    if not stacked:
        raise ValidationError(
            "no ticker in the universe returned data; nothing to publish."
        )
    panel = pd.concat(stacked).sort_index()
    return _published(
        panel,
        "price_panel",
        input_data.run_id,
        input_data.name,
        "fetch_ohlcv_panel",
        entities=[t for t, f in by_symbol.items() if f is not None and len(f)],
        warnings=warnings,
    )


def fetch_returns_panel(input_data: FetchReturnsPanelInput) -> FetchResult:
    """A wide date x ticker frame of returns, ready for any panel analysis."""
    panel = fetch_returns_sync(
        list(input_data.tickers),
        input_data.start_date,
        input_data.end_date,
        input_data.interval,
    )
    warnings: List[str] = []
    missing = [t for t in input_data.tickers if t not in list(panel.columns)]
    if missing:
        warnings.append(
            f"{len(missing)} ticker(s) are absent from the panel: "
            f"{sorted(missing)}."
        )
    return _published(
        panel,
        "returns_panel",
        input_data.run_id,
        input_data.name,
        "fetch_returns_panel",
        entities=[str(c) for c in panel.columns],
        warnings=warnings,
    )


def fetch_tick_tape(input_data: FetchTickTapeInput) -> FetchResult:
    """Individual trades, published as a `tick_tape` reference."""
    provider = DataFactory.get_provider()
    frame = _fetched(
        "a tick feed",
        lambda: provider.get_trades(
            input_data.symbol,
            input_data.start_date,
            input_data.end_date,
            input_data.limit,
        ),
        "fetch_tick_tape",
    )
    warnings = []
    if input_data.limit is not None and len(frame) >= input_data.limit:
        warnings.append(
            f"the tape hit the {input_data.limit:,} row limit, so it is "
            "TRUNCATED rather than complete for the window. Any rate or "
            "total computed from it understates the real one."
        )
    return _published(
        frame,
        "tick_tape",
        input_data.run_id,
        input_data.name,
        "fetch_tick_tape",
        entities=[input_data.symbol],
        warnings=warnings,
    )


def fetch_quote_panel(input_data: FetchQuotePanelInput) -> FetchResult:
    """Top-of-book quotes, published as a `quote_panel` reference."""
    provider = DataFactory.get_provider()
    frame = _fetched(
        "a top-of-book quote feed",
        lambda: provider.get_quotes(
            input_data.symbol,
            input_data.start_date,
            input_data.end_date,
            input_data.limit,
        ),
        "fetch_quote_panel",
    )
    warnings = [
        "Top of book only. No shipped provider exposes depth, so queue "
        "position and resting size at a level are not recoverable from this."
    ]
    if input_data.limit is not None and len(frame) >= input_data.limit:
        warnings.append(
            f"the panel hit the {input_data.limit:,} row limit and is "
            "TRUNCATED rather than complete for the window."
        )
    return _published(
        frame,
        "quote_panel",
        input_data.run_id,
        input_data.name,
        "fetch_quote_panel",
        entities=[input_data.symbol],
        warnings=warnings,
    )


def fetch_financial_ratios(
    input_data: FetchFinancialRatiosInput,
) -> FinancialRatiosResult:
    """A company's ratios, with the ones that look wrong flagged."""
    provider = DataFactory.get_provider()
    ratios = provider.get_financial_ratios(input_data.symbol)
    payload = (
        ratios.model_dump()
        if hasattr(ratios, "model_dump")
        else dict(getattr(ratios, "__dict__", {}))
    )
    return FinancialRatiosResult(
        symbol=input_data.symbol,
        ratios=payload,
        implausible=list(implausible_value_warnings(ratios)),
    )


def get_dataset_metadata(
    input_data: DatasetMetadataInput,
) -> DatasetMetadataResult:
    """What the active provider guarantees about the data it serves."""
    provider = DataFactory.get_provider()
    meta = provider.get_metadata(input_data.symbol, input_data.interval)
    warnings = []
    if getattr(meta, "point_in_time", False) is False:
        warnings.append(
            "This provider does NOT guarantee point-in-time data: a value "
            "you read today may not be the value that was visible on the "
            "date it is stamped with. Any backtest joining on that date is "
            "using information it could not have had."
        )
    if getattr(meta, "survivorship_free", False) is False:
        warnings.append(
            "This provider does NOT guarantee a survivorship-free universe, "
            "so a screen run over history sees only names that still exist."
        )
    return DatasetMetadataResult(
        symbol=input_data.symbol,
        interval=input_data.interval,
        provider=getattr(meta, "provider", None),
        adjusted=getattr(meta, "adjusted", None),
        survivorship_free=getattr(meta, "survivorship_free", None),
        point_in_time=getattr(meta, "point_in_time", None),
        timezone=getattr(meta, "timezone", None),
        warnings=warnings,
    )


def infer_temporal_contract(
    input_data: InferTemporalContractInput,
) -> TemporalContractResult:
    """What a frame's own columns say about when its rows became knowable."""
    frame = _resolved(input_data.ref)
    contract = contract_for_frame(
        frame,
        source=input_data.source,
        frame_kind=input_data.frame_kind,
        entity_scoped=input_data.entity_scoped,
    )
    return TemporalContractResult(
        ref=input_data.ref,
        frame_kind=input_data.frame_kind,
        source=contract.source,
        pit_safe=bool(contract.pit_safe),
        reproduces_history=bool(contract.reproduces_history),
        revisions=getattr(contract, "revisions", None),
        available_time_column=getattr(contract, "available_time_column", None),
        why_not_pit_safe=contract.why_not_pit_safe(),
        caveats=list(contract.caveats()),
        warnings=[
            "INFERRED FROM COLUMNS, which can only say what is present -- "
            "never what the source guarantees. A frame that happens to "
            "carry no restatement is indistinguishable here from one whose "
            "provider discards them. Prefer the provider's own contract "
            "(get_dataset_metadata) whenever there is one."
        ],
    )


def _bundle_from_manifest(ref: str) -> DataBundle:
    """Rebuild a bundle by resolving every frame its manifest names."""
    manifest = _resolved(ref, expect="data_bundle")
    missing = [c for c in _BUNDLE_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValidationError(
            f"{ref} is not a bundle manifest -- it has no {missing} "
            f"column(s). Expected {list(_BUNDLE_COLUMNS)}."
        )
    bundle = DataBundle(str(ref))
    for _, row in manifest.iterrows():
        frame = _resolved(str(row["ref"]))
        bundle.add(
            str(row["frame_kind"]),
            frame,
            source=str(row["source"]),
        )
    return bundle


def build_data_bundle(input_data: BuildDataBundleInput) -> DataBundleResult:
    """Name several published frames as one unit and publish the manifest."""
    bundle = DataBundle(input_data.name)
    rows = []
    for entry in input_data.frames:
        frame = _resolved(entry.ref)
        bundle.add(entry.frame_kind, frame, source=entry.source)
        rows.append(
            {
                "frame_kind": entry.frame_kind,
                "ref": entry.ref,
                "source": entry.source,
            }
        )
    manifest = pd.DataFrame(rows, columns=list(_BUNDLE_COLUMNS))
    ref = publish(
        manifest,
        kind="data_bundle",
        run_id=input_data.run_id,
        name=input_data.name,
        producer="build_data_bundle",
    )
    described = bundle.describe()
    return DataBundleResult(
        ref=ref,
        name=described["name"],
        n_frames=described["n_frames"],
        kinds=list(described["kinds"]),
        frames=[
            BundleFrameSummary(**{**f, "ref": r["ref"]})
            for f, r in zip(described["frames"], rows)
        ],
        pit_safe=bool(described["pit_safe"]),
        reproduces_history=bool(described["reproduces_history"]),
        warnings=list(described["warnings"]),
    )


def describe_data_bundle(input_data: DataBundleRefInput) -> DataBundleResult:
    """What frames a bundle names, and what their sources can promise."""
    bundle = _bundle_from_manifest(input_data.ref)
    described = bundle.describe()
    return DataBundleResult(
        ref=input_data.ref,
        name=described["name"],
        n_frames=described["n_frames"],
        kinds=list(described["kinds"]),
        frames=[BundleFrameSummary(**f) for f in described["frames"]],
        pit_safe=bool(described["pit_safe"]),
        reproduces_history=bool(described["reproduces_history"]),
        warnings=list(described["warnings"]),
    )


def validate_data_bundle(
    input_data: ValidateDataBundleInput,
) -> BundleVerdictResult:
    """Is this bundle safe to model on, and what is wrong with it if not."""
    bundle = _bundle_from_manifest(input_data.ref)
    verdict = validate_bundle(bundle, require_pit=input_data.require_pit)
    warnings = list(verdict["warnings"])
    if not input_data.require_pit:
        warnings.append(
            "Checked WITHOUT requiring point-in-time safety, so `usable` "
            "here does not mean a leakage-free join is possible. Pass "
            "require_pit=true when that is the question."
        )
    return BundleVerdictResult(
        name=verdict["name"],
        n_frames=verdict["n_frames"],
        kinds=list(verdict["kinds"]),
        pit_safe=bool(verdict["pit_safe"]),
        reproduces_history=bool(verdict["reproduces_history"]),
        usable=bool(verdict["usable"]),
        blocking=list(verdict["blocking"]),
        warnings=warnings,
    )


def validate_financial_ratios(
    input_data: ValidateFinancialRatiosInput,
) -> FinancialRatiosResult:
    """Flag vendor ratios that are implausible on their face."""
    implausible = list(implausible_value_warnings(input_data.ratios))
    return FinancialRatiosResult(
        symbol=str(input_data.ratios.get("symbol", "")),
        ratios=dict(input_data.ratios),
        implausible=implausible,
        warnings=(
            []
            if implausible
            else [
                "Nothing failed the plausibility check, which is a weak "
                "statement: the check catches values that are wrong on "
                "their face, not values that are merely incorrect."
            ]
        ),
    )


def compare_ratio_frames(
    input_data: CompareRatioFramesInput,
) -> RatioComparisonResult:
    """Two providers' ratios side by side, with each gap classified."""
    report: Dict[str, Any] = compare_ratio_sources(
        dict(input_data.left),
        dict(input_data.right),
        left_name=input_data.left_name,
        right_name=input_data.right_name,
        fields=list(input_data.fields) if input_data.fields else None,
    )
    rows = report.get("fields") or report.get("comparisons") or []
    fields = [
        RatioFieldComparison(
            field_name=str(row.get("field", row.get("field_name", ""))),
            left=row.get("left"),
            right=row.get("right"),
            relative_difference=row.get(
                "relative_difference", row.get("relative_diff")
            ),
            classification=row.get("classification"),
        )
        for row in rows
    ]
    disagreeing = [f for f in fields if f.classification not in (None, "agree")]
    return RatioComparisonResult(
        left_name=input_data.left_name,
        right_name=input_data.right_name,
        n_compared=len(fields),
        n_disagreeing=len(disagreeing),
        fields=fields,
        warnings=list(report.get("warnings", []))
        or [
            "A classification of 'unit' is fixable by rescaling; a "
            "'definition' difference is not, and averaging the two sources "
            "would produce a number neither provider would stand behind."
        ],
    )


__all__ = [
    "build_data_bundle",
    "compare_ratio_frames",
    "describe_data_bundle",
    "fetch_financial_ratios",
    "fetch_ohlcv",
    "fetch_ohlcv_panel",
    "fetch_quote_panel",
    "fetch_returns_panel",
    "fetch_tick_tape",
    "get_dataset_metadata",
    "infer_temporal_contract",
    "validate_data_bundle",
    "validate_financial_ratios",
]


def build_continuous_futures_series(
    input_data: ContinuousFuturesInput,
) -> ContinuousFuturesResult:
    chain = [c.model_dump(exclude_none=True) for c in input_data.contracts]
    built = build_continuous_futures(
        chain,
        roll_rule=input_data.roll_rule,
        adjustment=input_data.adjustment,
        days_before_expiry=input_data.days_before_expiry,
    )

    research = pd.DataFrame(
        {"price": pd.Series(built["research_series"], dtype="float64")}
    )
    research.index = pd.to_datetime(research.index)
    tradeable = pd.DataFrame.from_dict(built["tradeable_contract_map"], orient="index")
    tradeable.index = pd.to_datetime(tradeable.index)

    # Published SEPARATELY on purpose. One reference carrying both would let
    # a caller reach for whichever column was nearer, and the whole reason
    # this tool returns two things is that using the adjusted series to size
    # a position is the error it exists to prevent.
    research_ref = publish(
        research.sort_index(),
        kind="price_panel",
        run_id=input_data.run_id,
        name=f"{input_data.name}_research",
        producer="build_continuous_futures_series",
    )
    tradeable_ref = publish(
        tradeable.sort_index(),
        kind="price_panel",
        run_id=input_data.run_id,
        name=f"{input_data.name}_tradeable",
        producer="build_continuous_futures_series",
    )
    return ContinuousFuturesResult(
        research_ref=research_ref,
        tradeable_ref=tradeable_ref,
        roll_rule=built["roll_rule"],
        adjustment=built["adjustment"],
        n_contracts=built["n_contracts"],
        n_observations=built["n_observations"],
        n_rolls=built["n_rolls"],
        roll_dates=built["roll_dates"],
        contracts_used=built["contracts_used"],
        start=built["start"],
        end=built["end"],
        warnings=built["warnings"],
    )
