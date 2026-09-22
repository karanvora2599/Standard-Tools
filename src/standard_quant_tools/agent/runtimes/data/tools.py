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
confusable duplication the runtime split exists to avoid.

IT DOES FETCH DEPTH NOW, and the argument that it should not was answerable
all along. The objection was that a fetch tool "would have to answer for
every provider" and nine of ten do not serve a book -- but `fetch_tick_tape`
and `fetch_quote_panel` have always answered for every provider through the
same mechanism, refusing BY NAME on one that does not serve and pointing at
`describe_data_capabilities`. What was really missing was the second half:
the analysis side had 872 lines that could only be fed by a book the caller
had already captured somewhere else, so the only door in was
`register_external_dataset` and there was no way to obtain what it wanted.

So `fetch_order_book` and `fetch_order_events` fetch, write ONE Parquet
under the run, and register that file rather than publishing the frame.
That is not a detour: both kinds are external-only by design -- a session of
depth is not something to hold in memory, and both consumers stream it in
batches -- so the fetch ends where a registration ends, and the reference it
returns is the one those tools already read. `preflight_vendor_request`
prices the window first, because the feed is metered and the window, not the
row cap, is what the vendor bills.
"""

from __future__ import annotations

import datetime
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from standard_quant_tools.backtest.artifacts import save_artifact
from standard_quant_tools.data.bundle import DataBundle, validate_bundle
from standard_quant_tools.data.comparison import compare_ratio_sources
from standard_quant_tools.data.continuous import build_continuous_futures
from standard_quant_tools.data.databento import SCHEMA_KINDS
from standard_quant_tools.data.external import book_levels
from standard_quant_tools.data.external_validation import validate_external
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.data.ratios import implausible_value_warnings
from standard_quant_tools.data.temporal import contract_for_frame
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.portfolio import (
    fetch_ohlcv_panel_sync,
    fetch_returns_sync,
)

from ..handoff import _vendor_provenance
from ..handoff import describe as _handoff_describe
from ..handoff import parse as _parse_ref
from ..handoff import publish, publish_external, resolve
from .models import (
    BuildDataBundleInput,
    CompareRatioFramesInput,
    ContinuousFuturesInput,
    DataBundleRefInput,
    DatasetMetadataInput,
    ExternalDatasetRefInput,
    FetchFinancialRatiosInput,
    FetchOhlcvInput,
    FetchOhlcvPanelInput,
    FetchOrderBookInput,
    FetchOrderEventsInput,
    FetchQuotePanelInput,
    FetchReturnsPanelInput,
    FetchTickTapeInput,
    InferTemporalContractInput,
    PreflightVendorRequestInput,
    PrepareVendorExtractInput,
    RegisterExternalDatasetInput,
    ValidateDataBundleInput,
    ValidateExternalDatasetInput,
    ValidateFinancialRatiosInput,
)
from .results import (
    BundleFrameSummary,
    BundleVerdictResult,
    ContinuousFuturesResult,
    DataBundleResult,
    DatasetMetadataResult,
    DepthFetchResult,
    ExternalDatasetResult,
    ExternalValidationResult,
    FetchResult,
    FinancialRatiosResult,
    RatioComparisonResult,
    RatioFieldComparison,
    TemporalContractResult,
    VendorExtractResult,
    VendorPreflightResult,
)

logger = logging.getLogger(__name__)

#: The manifest columns a `data_bundle` reference stores. Fixed here rather
#: than inferred, because describe/validate read this frame back and a
#: column that quietly changed name would surface as a missing frame.
_BUNDLE_COLUMNS = ("frame_kind", "ref", "source")

#: Reference kind -> the bundle frame kinds it may legitimately be labelled
#: as.
#:
#: WHY THIS IS CHECKED AT ALL. A bundle pairs each frame with a temporal
#: contract, and the contract is chosen FROM THE LABEL -- so a label is not
#: a comment, it is the input to the point-in-time verdict. A returns panel
#: labelled `fundamentals` was accepted and then validated as fundamentals,
#: producing a confident and entirely wrong answer about whether a
#: leakage-free join was possible.
#:
#: An empty set means the kind is a DERIVED artifact -- a model's output, an
#: account curve, a state machine -- which is not a data source and has no
#: honest contract to be validated under. Refusing those by name is better
#: than choosing a label for them.
_BUNDLE_KINDS: Dict[str, tuple] = {
    "price_panel": ("bars",),
    "returns_panel": ("bars",),
    "tick_tape": ("bars",),
    "quote_panel": ("bars",),
    "order_book_panel": ("bars",),
    "order_event_panel": ("bars",),
    "indicator_panel": ("bars",),
    # Corporate actions are events carrying an availability stamp, and the
    # event panel is the only kind shaped to hold either.
    "event_panel": ("events", "corporate_actions"),
    "equity_curve": (),
    "trade_log": (),
    "signal_panel": (),
    "weight_panel": (),
    "score_panel": (),
    "predictions": (),
    "analytic_series": (),
    "analytic_frame": (),
    "data_bundle": (),
}


def _checked_bundle_label(ref: str, frame_kind: str) -> Optional[str]:
    """
    Refuse a frame labelled as something it is not; warn when nothing says.

    Returns a warning string, or None. A raw artifact path carries no kind
    -- that is the documented trade-off of accepting one -- so there is
    nothing to check it against and the caller is told so rather than given
    a check that did not happen.
    """
    text = str(ref).strip()
    if not text.startswith("sqt://"):
        return (
            f"{ref} is a raw artifact path, which carries no kind, so "
            f"frame_kind={frame_kind!r} was TAKEN ON TRUST rather than "
            "checked. The contract this bundle is validated under is chosen "
            "from that label, so a wrong one produces a confident verdict "
            "about the wrong thing -- publish the frame with a kind if the "
            "verdict matters."
        )
    kind = _parse_ref(text).kind
    admissible = _BUNDLE_KINDS.get(kind)
    if admissible is None or frame_kind in admissible:
        return None
    if not admissible:
        raise ValidationError(
            f"{ref} is a {kind!r}, which is a DERIVED result rather than a "
            f"data source, so it cannot be labelled {frame_kind!r} -- or "
            "anything else -- in a bundle. A bundle pairs each frame with "
            "what its SOURCE promises about timing, and a "
            f"{kind!r} has no source to promise anything. Bundle the data "
            "it was computed from instead."
        )
    raise ValidationError(
        f"{ref} is a {kind!r} and was labelled frame_kind={frame_kind!r}. "
        f"A {kind!r} belongs under {list(admissible)}. The label is not a "
        "comment: validate_data_bundle picks the temporal contract from it, "
        f"so this bundle would be judged as {frame_kind!r} and answer "
        "confidently about data it does not hold. Pass "
        f"frame_kind={admissible[0]!r}, or pass the reference that really "
        f"is {frame_kind!r}."
    )


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


def _provider(input_data: Any):
    """The provider a tool input names, or the default. The runtime could
    only ever reach the default before (findings, the plumbing)."""
    source = getattr(input_data, "source", None)
    return DataFactory.get_provider(source) if source else DataFactory.get_provider()


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
            f"it: {exc} Providers that do: source='polygon' (on a plan tier "
            "with ticks) and source='databento' (from the venue tape); pass "
            "`source` to this tool, or call describe_data_capabilities to see "
            "what this environment can actually reach."
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
        # WHICH VENDOR DATASET ANSWERED. The provider writes it onto the
        # frame and it went no further than this function: a caller could
        # not tell a consolidated tape from a single-venue sample carrying
        # a few percent of volume, and which one answers is date-dependent.
        **_vendor_provenance(frame),
        warnings=list(warnings or []),
    )


def fetch_ohlcv(input_data: FetchOhlcvInput) -> FetchResult:
    """One symbol's OHLCV bars, published as a `price_panel` reference."""
    provider = _provider(input_data)
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
    provider = _provider(input_data)
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
    provider = _provider(input_data)
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
        "Top of book only. Depth is a different call -- "
        "provider='databento' serves it through get_order_book -- and queue "
        "position is in neither, because it needs an order-level feed."
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


# ── depth and order-by-order: written once, registered where they land ──
#
# WHY THESE TWO DO NOT GO THROUGH `_published`. A depth panel and an order
# tape are EXTERNAL reference kinds: they are registered by path and read
# in batches, never loaded whole, because the tools that consume them are
# built for a session that does not fit in memory. `publish()` refuses
# them for that reason, and both consumers refuse an in-memory frame. So
# the fetch writes one Parquet under the runs directory and registers that
# file -- the same door `register_external_dataset` opens for a book the
# caller already holds, reached without having to hold one first.


def _column_span(frame: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """First and last `timestamp` VALUE, for a frame that stamps in a column.

    `_span` reads the INDEX, which is right for every bar panel here and
    wrong for these two: a book update and an order event are not unique
    in time, so both feeds carry `timestamp` as a column and index by
    position. Reading the index would report `0` and `n-1` as the window.

    Instants rather than dates, because the whole window is usually inside
    one session and a pair of identical dates says nothing about it.
    """
    if frame is None or len(frame) == 0 or "timestamp" not in frame.columns:
        return None, None
    try:
        stamps = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
    except Exception:  # noqa: BLE001 -- an unreadable stamp is not an error
        return None, None
    if len(stamps) == 0:
        return None, None
    return str(stamps.min().isoformat()), str(stamps.max().isoformat())


def _registered(
    frame: pd.DataFrame,
    kind: str,
    run_id: str,
    name: str,
    producer: str,
):
    """Write one Parquet under the run, and register it as an external ref.

    The write goes through the same containment-checked, atomic path every
    artifact in this library takes, so a run_id or a name that tries to
    escape the runs directory is refused there rather than here. What is
    different is the second step: the file is REGISTERED rather than
    published, which records a pointer and a schema and leaves the bytes
    in exactly one place.
    """
    if frame is None or len(frame) == 0:
        raise ValidationError(
            f"{producer} fetched no rows for that window. An empty panel "
            "registered as a reference would be indistinguishable "
            "downstream from one whose data had simply not arrived -- widen "
            "the window, or check the coverage with "
            "preflight_vendor_request before spending another request."
        )
    path = save_artifact(frame, run_id, name)
    try:
        return publish_external(
            path,
            kind=kind,
            run_id=run_id,
            name=name,
            producer=producer,
            fmt="parquet",
        )
    except Exception:
        # A file with no sidecar is unreachable and unexplained: nothing
        # can resolve it and the next run under the same name would be
        # refused by a collision the caller never caused.
        try:
            Path(path).unlink()
        except OSError:  # pragma: no cover - a locked file is not the story
            pass
        raise


def fetch_order_book(input_data: FetchOrderBookInput) -> DepthFetchResult:
    """L2 depth snapshots, registered as an `order_book_panel` reference."""
    provider = _provider(input_data)
    frame = _fetched(
        "an L2 depth feed",
        lambda: provider.get_order_book(
            input_data.symbol,
            input_data.start_date,
            input_data.end_date,
            input_data.levels,
            input_data.limit,
        ),
        "fetch_order_book",
    )
    provenance = _vendor_provenance(frame)
    truncated = len(frame) >= input_data.limit
    warnings = [
        "ONE VENUE'S BOOK, not a national one. Depth is published per "
        "venue and consolidating it is not a thing a vendor does, so the "
        "sizes here are what rested on this venue -- an imbalance computed "
        "from them is that venue's imbalance."
    ]
    if truncated:
        warnings.append(
            f"the book hit the {input_data.limit:,} snapshot cap, so what "
            "was registered is a PREFIX of the window rather than all of "
            "it. Every mean and rate computed from it describes the "
            "opening, which is the least typical part of a session."
        )
    start, end = _column_span(frame)
    ref, handle = _registered(
        frame,
        "order_book_panel",
        input_data.run_id,
        input_data.name,
        "fetch_order_book",
    )
    levels = book_levels(handle.columns)
    if levels and levels < input_data.levels:
        warnings.append(
            f"{input_data.levels} levels were asked for and {levels} came "
            "back complete. A level counts only with all four of its "
            "columns, and a trailing level empty in every snapshot is "
            "dropped rather than left to look like depth holding nothing."
        )
    return DepthFetchResult(
        ref=ref,
        kind="order_book_panel",
        rows=int(handle.rows if handle.rows is not None else len(frame)),
        columns=[str(c) for c in handle.columns],
        entities=[input_data.symbol],
        start=start,
        end=end,
        truncated=truncated,
        levels=levels or None,
        path=str(handle.path),
        size_bytes=int(handle.size_bytes or 0),
        dataset=provenance["dataset"],
        provider=provenance["provider"],
        warnings=warnings,
    )


def fetch_order_events(input_data: FetchOrderEventsInput) -> DepthFetchResult:
    """Order-by-order events, registered as an `order_event_panel` reference."""
    provider = _provider(input_data)
    frame = _fetched(
        "an order-by-order feed",
        lambda: provider.get_order_events(
            input_data.symbol,
            input_data.start_date,
            input_data.end_date,
            input_data.limit,
        ),
        "fetch_order_events",
    )
    provenance = _vendor_provenance(frame)
    truncated = len(frame) >= input_data.limit
    warnings = [
        "A WINDOW THAT OPENS MID-SESSION SEES ORDERS IT NEVER SAW ADDED. "
        "Their true lifetime is longer than anything this window can "
        "measure, and get_order_event_metrics counts them separately "
        "rather than averaging them in."
    ]
    if truncated:
        warnings.append(
            f"the feed hit the {input_data.limit:,} event cap, so what was "
            "registered is a PREFIX of the window. A cancellation rate or "
            "an order lifetime computed from it is the opening's, and the "
            "orders still resting at the cut are right-censored."
        )
    start, end = _column_span(frame)
    ref, handle = _registered(
        frame,
        "order_event_panel",
        input_data.run_id,
        input_data.name,
        "fetch_order_events",
    )
    return DepthFetchResult(
        ref=ref,
        kind="order_event_panel",
        rows=int(handle.rows if handle.rows is not None else len(frame)),
        columns=[str(c) for c in handle.columns],
        entities=[input_data.symbol],
        start=start,
        end=end,
        truncated=truncated,
        levels=None,
        path=str(handle.path),
        size_bytes=int(handle.size_bytes or 0),
        dataset=provenance["dataset"],
        provider=provenance["provider"],
        warnings=warnings,
    )


def _utc_bound(text: str, what: str) -> pd.Timestamp:
    """A request bound as UTC, or a refusal naming the argument."""
    try:
        stamp = pd.Timestamp(str(text))
    except Exception as exc:  # noqa: BLE001 -- one refusal, not a traceback
        raise ValidationError(
            f"{what}={text!r} is not a date this library can read. Dates "
            "here are YYYY-MM-DD, and the end date is INCLUSIVE."
        ) from exc
    if stamp is pd.NaT or pd.isna(stamp):
        raise ValidationError(
            f"{what}={text!r} is not a date this library can read. Dates "
            "here are YYYY-MM-DD, and the end date is INCLUSIVE."
        )
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _reported_window(window: Any) -> tuple:
    """A provider's (first, last) coverage pair as UTC stamps, or (None, None).

    A window that cannot be read is UNKNOWN, never a partial one: half a
    pair would be a coverage claim the provider did not make.
    """
    try:
        first, last = window
        first_stamp = pd.Timestamp(str(first))
        last_stamp = pd.Timestamp(str(last))
    except Exception:  # noqa: BLE001 -- an unreadable window is simply unknown
        return None, None
    if pd.isna(first_stamp) or pd.isna(last_stamp):
        return None, None
    if first_stamp.tzinfo is None:
        first_stamp = first_stamp.tz_localize("UTC")
    if last_stamp.tzinfo is None:
        last_stamp = last_stamp.tz_localize("UTC")
    return first_stamp.tz_convert("UTC"), last_stamp.tz_convert("UTC")


def preflight_vendor_request(
    input_data: PreflightVendorRequestInput,
) -> VendorPreflightResult:
    """What a vendor request would cost, and whether the data is there."""
    provider = _provider(input_data)
    router = getattr(provider, "datasets_for_schema", None)
    if not callable(router):
        raise ValidationError(
            f"{type(provider).__name__} cannot preflight a request: it "
            "routes no named vendor datasets, publishes no coverage "
            "windows and prices nothing before it is asked, so every field "
            "here would be invented rather than reported. Pass "
            "source='databento', which answers all three from free "
            "metadata endpoints, or call describe_data_capabilities to see "
            "what this environment can reach."
        )

    schema = input_data.vendor_schema
    start = _utc_bound(input_data.start_date, "start_date")
    end = _utc_bound(input_data.end_date, "end_date")
    if end < start:
        raise ValidationError(
            f"empty window: start_date={input_data.start_date!r} is after "
            f"end_date={input_data.end_date!r}. The end date is INCLUSIVE, "
            "so a same-day request is a valid one and this is not."
        )
    # The end date is inclusive here and half-open at the vendor, so the
    # window a dataset has to cover runs to the following midnight.
    end_exclusive = end + pd.Timedelta(days=1)

    warnings = [
        "BYTES, NOT MONEY. The vendor's own cost endpoint reports $0.00 "
        "for any request an account's subscription already includes, which "
        "is silent about exactly the thing it is consulted for. The byte "
        "count is the quantity that is true for every account, and it is "
        "what a limit should be chosen against."
    ]

    try:
        candidates = [
            str(name)
            for name in router(input_data.symbol, schema, input_data.start_date)
            if name
        ]
    except (ValidationError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001 -- one refusal, not a traceback
        raise ValidationError(
            f"{type(provider).__name__} could not route {schema!r} for "
            f"{input_data.symbol!r}: {exc}. Check the symbol spelling -- a "
            "futures root that is also an equity ticker is refused as "
            "ambiguous rather than resolved to one of them."
        ) from exc

    coverage: Dict[str, Any] = {}
    try:
        coverage = dict(provider.get_dataset_coverage(candidates) or {})
    except NotImplementedError as exc:
        raise ValidationError(
            f"{type(provider).__name__} publishes no coverage windows: "
            f"{exc} Pass source='databento'."
        ) from exc
    except (ValidationError, ValueError):
        raise
    except Exception as exc:  # noqa: BLE001 -- unreachable metadata is a fact
        warnings.append(
            f"coverage windows could not be read ({exc}), so "
            "coverage_start and coverage_end are null rather than guessed. "
            "A window invented here is the one a caller would plan the "
            "request around."
        )

    # THE SAME RULE THE FETCH APPLIES. A dataset answers when it reaches
    # back to the start of the window and has not ended before it; the
    # first one that does is the one that serves, which is why the chosen
    # dataset is not always the first candidate.
    chosen: Optional[str] = None
    for name in candidates:
        window = coverage.get(name)
        if not window:
            continue
        first, last = _reported_window(window)
        if first is None or last is None:
            continue
        if first <= start < last:
            chosen = name
            break
    if chosen is None:
        chosen = next(
            (name for name in candidates if coverage.get(name)),
            candidates[0] if candidates else None,
        )
    if not candidates:
        warnings.append(
            f"{type(provider).__name__} routes no dataset for schema "
            f"{schema!r} and {input_data.symbol!r}, so there is nothing to "
            "price. The request would be refused rather than served."
        )

    coverage_start: Optional[str] = None
    coverage_end: Optional[str] = None
    covers: Optional[bool] = None
    window = coverage.get(chosen) if chosen else None
    if window:
        coverage_start, coverage_end = str(window[0]), str(window[1])
        first, last = _reported_window(window)
        if first is not None and last is not None:
            covers = bool(first <= start and last >= end_exclusive)
            if not covers:
                warnings.append(
                    f"{chosen} published {coverage_start} to "
                    f"{coverage_end}, which does not contain the window "
                    "asked for. The fetch would return the overlap, or "
                    "fall through to the next dataset -- a shorter answer "
                    "than the dates suggest."
                )
    elif chosen is not None:
        warnings.append(
            f"{chosen} reports no coverage window -- unentitled, unknown, "
            "or declined -- so whether it holds this period is UNKNOWN "
            "rather than false."
        )

    billable: Optional[int] = None
    if chosen is not None:
        try:
            billable = int(
                provider.get_billable_size(
                    input_data.symbol,
                    input_data.start_date,
                    input_data.end_date,
                    schema,
                    dataset=chosen,
                )
            )
        except NotImplementedError as exc:
            raise ValidationError(
                f"{type(provider).__name__} does not price a request "
                f"before it is made: {exc} Pass source='databento'."
            ) from exc
        except (ValidationError, ValueError):
            raise
        except Exception as exc:  # noqa: BLE001 -- a declined quote is a fact
            warnings.append(
                f"the vendor would not price this request ({exc}), so "
                "billable_bytes is null rather than zero -- a question "
                "nobody answered is not an answer of 'free'."
            )

    return VendorPreflightResult(
        symbol=input_data.symbol,
        vendor_schema=schema,
        start_date=input_data.start_date,
        end_date=input_data.end_date,
        dataset=chosen,
        datasets_considered=candidates,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        covers_request=covers,
        billable_bytes=billable,
        kind=SCHEMA_KINDS.get(schema),
        provider=type(provider).__name__,
        warnings=warnings,
    )


def fetch_financial_ratios(
    input_data: FetchFinancialRatiosInput,
) -> FinancialRatiosResult:
    """A company's ratios, with the ones that look wrong flagged."""
    provider = _provider(input_data)
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
    provider = _provider(input_data)
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
        # The provider's own prose, which the four booleans above cannot
        # carry: this is where a provider names the feed that answers an
        # early window and what is wrong with it.
        notes=[str(n) for n in (getattr(meta, "notes", None) or [])],
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
    unchecked: List[str] = []
    for entry in input_data.frames:
        # BEFORE resolving: the label decides which contract the bundle is
        # validated under, so a mismatch is a refusal rather than a frame
        # loaded and then judged as something else.
        note = _checked_bundle_label(entry.ref, entry.frame_kind)
        if note:
            unchecked.append(note)
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
        warnings=list(described["warnings"]) + unchecked,
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


#: Keys a `fetch_financial_ratios` payload carries that are not ratios.
#: Present so a flat payload can be told from an entity map by its shape
#: rather than by asking the caller which one they passed.
_RATIO_PAYLOAD_KEYS = ("symbol", "definition_notes")

#: What a single company's ratios are keyed under once wrapped. The SAME
#: key on both sides, deliberately: a flat payload holds one company by
#: construction, so the two sides are the two answers about it and keying
#: them by their own `symbol` would make a comparison impossible whenever
#: one provider spelled the ticker differently or omitted it.
_SINGLE_ENTITY = "(one company)"


def _ratio_values(payload: Dict[str, Any]) -> List[Any]:
    """The entries of a ratios payload that could be either a ratio or an
    entity -- everything except the keys that are neither."""
    return [v for k, v in payload.items() if k not in _RATIO_PAYLOAD_KEYS]


def _unwrapped(payload: Dict[str, Any]) -> Dict[str, Any]:
    """A `fetch_financial_ratios` result reduced to its ratios map."""
    inner = payload.get("ratios")
    return dict(inner) if isinstance(inner, dict) and inner else dict(payload)


def _as_entity_map(payload: Dict[str, Any], side: str) -> Dict[str, Any]:
    """
    One provider's ratios in the shape the comparison reads: entity -> ratios.

    `fetch_financial_ratios` returns ONE company's ratios as a flat
    `{field: value}` map, and the comparison reads `{entity: ratios}`. The
    obvious composition -- fetch from two providers, hand both here -- was
    therefore read as two universes of entities named `forward_pe`,
    `trailing_pe` and so on, every one of whose ratios was a bare float
    with no fields on it. Two IDENTICAL payloads came back with all eight
    fields reported as disagreeing.

    A flat payload is wrapped as a single entity rather than refused,
    because that composition is the one an agent will reach for. What is
    refused is a payload that is neither shape, and a comparison where one
    side is flat and the other nested -- those cannot be reconciled without
    guessing which entity the flat one belongs to.
    """
    if not payload:
        return {}
    payload = _unwrapped(payload)
    values = _ratio_values(payload)
    nested = [v for v in values if isinstance(v, dict)]
    if nested and len(nested) == len(values):
        return dict(payload)
    if nested:
        raise ValidationError(
            f"the {side} payload mixes shapes: {len(nested)} of "
            f"{len(values)} entries are themselves mappings. It has to be "
            "either one company's ratios ({'forward_pe': 21.4, ...}) or a "
            "map of ticker -> that, not both."
        )
    return {_SINGLE_ENTITY: {k: v for k, v in payload.items() if k != "symbol"}}


def _is_flat(payload: Dict[str, Any]) -> bool:
    """True when this is ONE company's ratios rather than a ticker map."""
    values = _ratio_values(_unwrapped(payload))
    return bool(values) and not any(isinstance(v, dict) for v in values)


def _comparable(left: Dict[str, Any], right: Dict[str, Any]) -> None:
    """Refuse the two ways this composition silently compares the wrong
    things: mismatched nesting, and two different companies."""
    if not left or not right:
        return
    if _is_flat(left) != _is_flat(right):
        flat, nested = ("left", "right") if _is_flat(left) else ("right", "left")
        raise ValidationError(
            f"the {flat} payload is ONE company's ratios and the {nested} "
            "payload is a map of ticker -> ratios. Comparing them would "
            "mean guessing which ticker the single company is, and a wrong "
            "guess reports every field as a disagreement. Pass both as one "
            "company, or both as a ticker map."
        )
    if not _is_flat(left):
        return
    symbols = {
        side: str(_unwrapped(payload).get("symbol") or payload.get("symbol") or "")
        .strip()
        .upper()
        for side, payload in (("left", left), ("right", right))
    }
    if all(symbols.values()) and symbols["left"] != symbols["right"]:
        raise ValidationError(
            f"the left payload is for {symbols['left']} and the right one "
            f"for {symbols['right']}. This tool asks whether two SOURCES "
            "disagree about one company; two different companies disagree "
            "for reasons that have nothing to do with the providers, and "
            "every field would be reported as a divergence."
        )


def compare_ratio_frames(
    input_data: CompareRatioFramesInput,
) -> RatioComparisonResult:
    """Two providers' ratios side by side, with each gap classified."""
    left, right = dict(input_data.left), dict(input_data.right)
    _comparable(left, right)
    report: Dict[str, Any] = compare_ratio_sources(
        _as_entity_map(left, "left"),
        _as_entity_map(right, "right"),
        left_name=input_data.left_name,
        right_name=input_data.right_name,
        fields=list(input_data.fields) if input_data.fields else None,
    )
    rows = report.get("fields") or report.get("comparisons") or []
    fields = [
        RatioFieldComparison(
            field_name=str(row.get("field", row.get("field_name", ""))),
            # The library reports a SPAN, not two numbers, because over a
            # universe two numbers would be one entity's pair passed off as
            # the comparison. It names the pair only when there is one.
            left=row.get("left"),
            right=row.get("right"),
            relative_difference=row.get("max_relative_difference"),
            ratio=row.get("ratio"),
            ratio_spread=row.get("ratio_spread"),
            n_compared=int(row.get("n_compared", 0) or 0),
            # `comparison.py` writes this key as "verdict"; reading only
            # "classification" made every field null and n_disagreeing
            # always 0, so a 100x unit mismatch reported as agreement.
            classification=row.get("classification", row.get("verdict")),
        )
        for row in rows
    ]
    # `no_overlap` IS NOT A DISAGREEMENT. It means the field was never
    # checked, and counting it as one made two identical inputs report
    # every field in conflict.
    no_overlap = [f for f in fields if f.classification == "no_overlap"]
    disagreeing = [
        f for f in fields if f.classification not in (None, "agree", "no_overlap")
    ]
    return RatioComparisonResult(
        left_name=input_data.left_name,
        right_name=input_data.right_name,
        n_compared=len(fields),
        n_disagreeing=len(disagreeing),
        n_no_overlap=len(no_overlap),
        fields=fields,
        warnings=list(report.get("warnings", []))
        or [
            # The classifier emits no_overlap / agree / scale / definition.
            # "unit" was never one of them.
            "A classification of 'scale' is fixable by rescaling; a "
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


# ---------------------------------------------------------------------------
# External datasets: data the caller already has, addressed where it lies.
#
# These three are the only tools in this runtime that do not fetch. That is
# the point of them -- the fetch path materializes a whole frame and then
# copies it a second time into the runs directory, which is exactly what a
# day of L2 depth cannot survive. Registration stores a pointer and a schema
# and reads nothing else.
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """One preview cell, in a form `json.dumps(allow_nan=False)` accepts."""
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):  # pragma: no cover - exotic dtypes
            pass
    return str(value)


def _preview(handle: Any, rows: int) -> List[Dict[str, Any]]:
    if rows <= 0:
        return []
    frame = handle.head(rows)
    return [
        {str(k): _json_safe(v) for k, v in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _external_result(
    ref: str,
    handle: Any,
    *,
    preview_rows: int = 0,
    changed: Optional[bool] = None,
    warnings: Optional[List[str]] = None,
) -> ExternalDatasetResult:
    notes = list(warnings or [])
    levels: Optional[int] = None
    if handle.kind == "order_book_panel":
        levels = book_levels(handle.columns)
        if levels == 1:
            notes.append(
                "WARNING: one complete depth level, so this is top of book "
                "however it was labelled. depth_slope is null on a one-level "
                "book and cumulative imbalance collapses to touch imbalance."
            )
    if handle.rows is not None and handle.rows == 0:
        notes.append("WARNING: the dataset has a valid schema and no rows.")
    if changed:
        notes.append(
            "WARNING: this file has moved or changed since it was "
            "registered. Nothing was copied, so the reference resolves to "
            "whatever is at the path NOW -- which is not necessarily what "
            "was validated under it."
        )
    return ExternalDatasetResult(
        ref=ref,
        kind=handle.kind,
        path=str(handle.path),
        file_format=handle.fmt,
        rows=handle.rows,
        columns=[str(c) for c in handle.columns],
        dtypes=dict(handle.dtypes),
        n_files=int(handle.n_files),
        size_bytes=int(handle.size_bytes),
        levels=levels,
        fingerprint=handle.fingerprint,
        changed_since_registration=changed,
        preview=_preview(handle, preview_rows),
        warnings=notes,
    )


# ──────────────────────────────────────────────────────────────────
# Vendor extracts — the conversion that had no door
# ──────────────────────────────────────────────────────────────────

#: kind -> the normalizer in `data.databento` that produces it. The module
#: is 643 lines with 39 dedicated tests and had ZERO references anywhere
#: under `agent/`: a finished, tested conversion layer with no entry point
#: on the tool surface. `external.check_schema` already told callers to run
#: these functions by name when it recognized a raw export, which made the
#: refusal a dead end -- it named a remedy nothing could execute.
_NORMALIZERS = {
    "order_book_panel": "normalize_book",
    "order_event_panel": "normalize_mbo",
    "quote_panel": "normalize_quotes",
    "tick_tape": "normalize_trades",
}


def _normalize_batch(kind: str, frame, *, price_scale, timestamp, levels, keep_empty):
    """One batch through the normalizer this kind names."""
    from standard_quant_tools.data import databento

    fn = getattr(databento, _NORMALIZERS[kind])
    if kind == "order_book_panel":
        return fn(
            frame,
            price_scale=price_scale,
            timestamp=timestamp,
            levels=levels,
            keep_empty_levels=keep_empty,
        )
    return fn(frame, price_scale=price_scale, timestamp=timestamp)


def prepare_vendor_extract(
    input_data: PrepareVendorExtractInput,
) -> VendorExtractResult:
    """
    A raw vendor extract -> this library's contract, with its judgements
    reported rather than made silently.

    THE STEP BEFORE register_external_dataset, and the one that had no
    tool. A Databento export spells the same quantity `bid_px_00` where
    this library spells it `bid_price_0`, stamps rows `ts_recv` rather than
    `timestamp`, sends prices as int64 nanodollars, and fills absent levels
    with int64-max rather than null. Registering it unconverted fails on
    the column names if you are lucky and puts $9.2 billion quotes in your
    book if you are not.

    Two of the decisions here CHANGE THE NUMBERS and neither is recoverable
    from the output: which timestamp became `timestamp`, and whether prices
    were divided by a billion. They come back in `notes` for that reason.
    """
    import pandas as pd

    from standard_quant_tools.data import external
    from standard_quant_tools.data.databento import (
        book_depth,
        level_is_empty,
        looks_like_databento,
        split_empty_levels,
    )

    out_path = Path(input_data.out_path)
    if not input_data.dry_run and out_path.exists():
        raise ValidationError(
            f"{out_path} already exists. A conversion writes a new file "
            "rather than replacing one, because the old file may already be "
            "registered and something may already hold a reference to it."
        )

    handle = external.inspect(
        input_data.path, fmt=input_data.file_format, count_rows=False
    )
    source_columns = list(handle.columns)
    available = (
        book_depth(source_columns) if input_data.kind == "order_book_panel" else None
    )

    notes: List[str] = []
    warnings: List[str] = []
    seen: set = set()

    def remember(new: List[str]) -> None:
        # One message per distinct judgement. The same sentence about the
        # price scale arrives with every batch and would otherwise produce
        # a report that is one fact repeated a hundred times.
        for line in new:
            if line not in seen:
                seen.add(line)
                notes.append(line)

    def batches():
        return handle.batches(batch_rows=input_data.batch_rows)

    # ── the book's trailing-empty-level rule needs the whole file ──────
    #
    # `normalize_book` drops a level that is empty in EVERY snapshot, which
    # a single batch cannot know. So the depth is decided in its own pass
    # and then FORCED for the writing pass, which makes every batch produce
    # the same columns as well. Skipped entirely when the caller keeps the
    # empty levels, which is the single-pass path.
    levels = input_data.levels
    keep_empty = input_data.keep_empty_levels
    levels_kept: Optional[int] = None
    if (
        input_data.kind == "order_book_panel"
        and not keep_empty
        and not input_data.dry_run
    ):
        # EMPTINESS IS A FACT ABOUT THE WHOLE EXTRACT, not about a batch:
        # a level absent from the first million rows may be quoted in the
        # next. So a level counts as occupied if ANY batch had something in
        # it, and what "empty" MEANS is imported rather than rewritten --
        # `level_is_empty` requires both sides absent, and an earlier
        # version of this loop tested only the bid, which would have dropped
        # a one-sided level that `normalize_book` keeps.
        occupied: set = set()
        depth_seen = 0
        rows_seen = 0
        for batch in batches():
            wide, batch_notes = _normalize_batch(
                input_data.kind,
                batch,
                price_scale=input_data.price_scale,
                timestamp=input_data.timestamp,
                levels=levels,
                keep_empty=True,
            )
            remember(batch_notes)
            rows_seen += len(wide)
            depth_seen = max(depth_seen, book_levels(wide.columns))
            for level in range(depth_seen):
                if not level_is_empty(wide, level):
                    occupied.add(level)
        empty = [level for level in range(depth_seen) if level not in occupied]
        trailing, inner = split_empty_levels(empty, depth_seen)
        kept = depth_seen - len(trailing)
        if not kept:
            raise ValidationError(
                f"every one of the {depth_seen} level(s) is empty in all "
                f"{rows_seen:,} snapshots, so there is no book here to "
                "convert. Check that the source really carries depth, or "
                "pass keep_empty_levels=true to write the vendor's columns "
                "through unexamined."
            )
        if trailing:
            remember(
                [
                    f"NOTE: levels {min(trailing)}-{max(trailing)} are empty "
                    f"in every one of the {rows_seen:,} snapshots, so this "
                    f"extract is {kept} levels deep, not {depth_seen}. Their "
                    "columns were dropped -- left in, the dataset would "
                    f"DECLARE {depth_seen} levels and depth_slope would "
                    "regress against levels holding nothing. Pass "
                    "keep_empty_levels=true to preserve the vendor's fixed "
                    "width."
                ]
            )
        if inner:
            # `normalize_book` raises this on a single frame, and the
            # streamed path lost it: both passes run with
            # keep_empty_levels=True, and the library only speaks up when it
            # is the one deciding.
            warnings.append(
                f"WARNING: level(s) {inner} are empty in every snapshot but "
                "sit BELOW a level that has size. That is a malformed book "
                "rather than a thin one, and the columns were kept so it "
                "stays visible instead of being renumbered away."
            )
        levels, keep_empty, levels_kept = kept, True, kept

    # ── convert, writing as we go ─────────────────────────────────────
    writer = None
    rows_written = 0
    columns: List[str] = []
    try:
        for batch in batches():
            converted, batch_notes = _normalize_batch(
                input_data.kind,
                batch,
                price_scale=input_data.price_scale,
                timestamp=input_data.timestamp,
                levels=levels,
                keep_empty=keep_empty,
            )
            remember(batch_notes)
            if not columns:
                columns = [str(c) for c in converted.columns]
                if input_data.kind == "order_book_panel" and levels_kept is None:
                    levels_kept = book_levels(converted.columns)
            if input_data.dry_run:
                rows_written = len(converted)
                break
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(converted, preserve_index=False)
            if writer is None:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(str(out_path), table.schema)
            writer.write_table(table)
            rows_written += len(converted)
    finally:
        if writer is not None:
            writer.close()

    if input_data.dry_run:
        warnings.append(
            f"DRY RUN: nothing was written. The notes describe what the "
            f"first {rows_written} row(s) decided; a trailing empty level "
            "cannot be detected from one batch, so `levels_kept` here is "
            "what the vendor sent, not what a real conversion would keep."
        )
    missing = external.check_schema(input_data.kind, columns)
    if missing:
        warnings.append(
            f"The converted columns still do not satisfy {input_data.kind!r}: "
            f"{missing[0]}"
        )

    next_step = (
        ""
        if input_data.dry_run
        else (
            f"register_external_dataset(path={str(out_path)!r}, "
            f"kind={input_data.kind!r}, run_id=..., name=..., "
            "source='databento')"
        )
    )
    logger.debug(
        "[prepare_vendor_extract] %s -> %s  rows=%d  notes=%d",
        input_data.path,
        input_data.kind,
        rows_written,
        len(notes),
    )
    return VendorExtractResult(
        out_path="" if input_data.dry_run else str(out_path),
        kind=input_data.kind,
        rows_written=rows_written,
        columns=columns,
        source_columns=source_columns[:24],
        looked_like_databento=bool(looks_like_databento(source_columns)),
        levels_available=available,
        levels_kept=levels_kept,
        notes=notes,
        next_step=next_step,
        warnings=warnings,
    )


def register_external_dataset(
    input_data: RegisterExternalDatasetInput,
) -> ExternalDatasetResult:
    """Make data on the caller's own disk resolvable, without copying it."""
    ref, handle = publish_external(
        input_data.path,
        kind=input_data.kind,
        run_id=input_data.run_id,
        name=input_data.name,
        producer=f"register_external_dataset:{input_data.source}",
        fmt=input_data.file_format,
    )
    warnings = [
        "REGISTERED, NOT COPIED. The schema was checked; the rows were not. "
        "Run validate_external_dataset before modeling on this -- a book "
        "with its bid and ask columns the wrong way round has a perfectly "
        "valid schema."
    ]
    if handle.fmt == "csv" and handle.rows is not None:
        warnings.append(
            "NOTE: CSV carries no row count, so counting them was a full "
            "scan. Parquet answers the same question from its footer, and "
            "is worth converting to for anything read more than once."
        )
    return _external_result(ref, handle, warnings=warnings)


def describe_external_dataset(
    input_data: ExternalDatasetRefInput,
) -> ExternalDatasetResult:
    """What a registered dataset holds, and whether it changed underneath."""
    described = _handoff_describe(input_data.ref)
    if described.get("storage") != "external":
        raise ValidationError(
            f"{input_data.ref!r} is a published artifact, not an external "
            "registration, so there is no path or fingerprint to report. "
            "Use describe_data_bundle for a bundle, or resolve it directly."
        )
    handle = resolve(input_data.ref)
    return _external_result(
        input_data.ref,
        handle,
        preview_rows=input_data.preview_rows,
        changed=bool(described.get("changed_since_registration")),
    )


def validate_external_dataset(
    input_data: ValidateExternalDatasetInput,
) -> ExternalValidationResult:
    """Scan a registered dataset in batches and report what would break."""
    described = _handoff_describe(input_data.ref)
    if described.get("storage") != "external":
        raise ValidationError(
            f"{input_data.ref!r} is a published artifact rather than an "
            "external registration. Anything this library published is "
            "already in memory-sized form; validate_data_bundle and "
            "validate_pit_records cover those."
        )
    handle = resolve(input_data.ref)
    report = validate_external(
        handle,
        kind=handle.kind,
        scan_limit=input_data.scan_limit,
    )
    warnings = list(report.warnings)
    if described.get("changed_since_registration"):
        warnings.insert(
            0,
            "WARNING: this file changed since it was registered, so this "
            "verdict is about the bytes on disk now, not about whatever was "
            "there when the reference was minted.",
        )
    return ExternalValidationResult(
        ref=input_data.ref,
        kind=report.kind,
        usable=bool(report.usable),
        blocking=list(report.blocking),
        rows_scanned=int(report.rows_scanned),
        rows_total=report.rows_total,
        coverage=report.coverage(),
        batches=int(report.batches),
        truncated=bool(report.truncated),
        stats=dict(report.stats),
        warnings=warnings,
    )
