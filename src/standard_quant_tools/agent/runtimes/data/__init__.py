"""
The `data` runtime's registry: what it advertises and what it can execute,
built from one list so a tool cannot be advertised without being
dispatchable or the reverse.

WHY THIS IS ITS OWN RUNTIME. Every other runtime answers a question ABOUT
markets. This one answers where the data is, and it is the only one whose
output is meant to be consumed by all the others. Folding it into `research`
would have made the fetch tools compete for attention with forty analysis
tools that have nothing to do with them, and folding it into `meta` would
have confused two different questions -- `meta` says what a provider CAN
promise, this runtime goes and gets it.

THE TOOLS RETURN REFERENCES, NOT DATA. That is the reason the runtime is
worth adding at all: an `sqt://` reference crosses runtimes and processes,
survives two agents that cannot see each other's context, and appears in the
audit log as an input to whatever consumed it. Before this, a panel built
inside one tool died there and the next runtime refetched it.

WHAT IS DELIBERATELY ABSENT. Data QUALITY checks -- `get_data_quality_report`
in `research` already reports missing bars, stale prices and price jumps, and
a second name for those is the confusable duplication runtimes exist to
prevent. Order books, because `DataProvider.get_order_book` is a contract no
shipped provider implements and a tool that always refuses is worse than no
tool.
"""

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
from .tools import (
    build_continuous_futures_series,
    build_data_bundle,
    compare_ratio_frames,
    describe_data_bundle,
    fetch_financial_ratios,
    fetch_ohlcv,
    fetch_ohlcv_panel,
    fetch_quote_panel,
    fetch_returns_panel,
    fetch_tick_tape,
    get_dataset_metadata,
    infer_temporal_contract,
    validate_data_bundle,
    validate_financial_ratios,
)

#: (name, description, input model) -- the single source for both the
#: advertised schema and the dispatch table below.
TOOL_DEFS = [
    (
        "fetch_ohlcv",
        "Fetch one symbol's OHLCV bars and publish them as an `sqt://` "
        "price_panel reference rather than returning the rows inline. Reach "
        "for this when the bars themselves are the thing another tool needs "
        "-- an indicator series, a custom signal, a panel join -- instead of "
        "going through an analysis tool that wants to do something else with "
        "them. The reference is what crosses runtimes; the frame never has "
        "to enter the conversation.",
        FetchOhlcvInput,
    ),
    (
        "fetch_ohlcv_panel",
        "Fetch a whole universe's OHLCV in one call and publish it stacked "
        "long, with an `entity` column, as a price_panel reference. Tickers "
        "that returned nothing are named in `warnings` and are ABSENT from "
        "the panel rather than present as NaN, which matters because a "
        "complete-case join downstream will not see them at all.",
        FetchOhlcvPanelInput,
    ),
    (
        "fetch_returns_panel",
        "Fetch a universe and publish a wide date-by-ticker frame of returns "
        "as a returns_panel reference. This is the shape most panel analysis "
        "wants -- PCA, correlation, factor regressions and portfolio "
        "construction all consume it directly -- so computing it once and "
        "handing over the reference avoids every consumer rebuilding it from "
        "prices.",
        FetchReturnsPanelInput,
    ),
    (
        "fetch_tick_tape",
        "Fetch individual trades and publish them as a tick_tape reference, "
        "for the microstructure tools that measure rather than estimate. "
        "Needs a provider with a tick feed. A tape is large, so `limit` caps "
        "it -- and when the cap is hit the result says so, because a "
        "truncated tape makes every rate and total computed from it "
        "understate the real one.",
        FetchTickTapeInput,
    ),
    (
        "fetch_quote_panel",
        "Fetch top-of-book quotes and publish them as a quote_panel "
        "reference, which is what signing trades by the Lee-Ready rule needs "
        "alongside a tape. Top of book ONLY: no shipped provider exposes "
        "depth, so queue position and resting size at a level are not "
        "recoverable from this and should not be inferred from it.",
        FetchQuotePanelInput,
    ),
    (
        "fetch_financial_ratios",
        "Fetch a company's financial ratios and flag the ones that are "
        "implausible on their face -- a negative price-to-book, a dividend "
        "yield above a plausible ceiling. The flag is a weak signal in one "
        "direction only: it catches values that are obviously wrong, never "
        "values that are merely incorrect.",
        FetchFinancialRatiosInput,
    ),
    (
        "get_dataset_metadata",
        "What the active provider GUARANTEES about the data it serves: "
        "whether prices are adjusted, whether the universe is "
        "survivorship-free, whether values are point-in-time, and which "
        "timezone stamps them. Read this before trusting a backtest over "
        "history, because a provider that is not point-in-time will hand you "
        "restated values under their original dates.",
        DatasetMetadataInput,
    ),
    (
        "infer_temporal_contract",
        "Read a published frame's own columns and report what they imply "
        "about when each row became knowable. For data this library did not "
        "fetch -- a vendor extract, another system's output -- where no "
        "provider contract exists. Inference reads COLUMNS, so it can only "
        "say what is present and never what a source guarantees; prefer "
        "get_dataset_metadata whenever the data came from a known provider.",
        InferTemporalContractInput,
    ),
    (
        "build_data_bundle",
        "Name several already-published frames as one unit and publish the "
        "manifest as a data_bundle reference. A bundle holds references "
        "rather than copies, so it cannot diverge from the frames it names, "
        "and it pairs each frame with what its source can say about timing "
        "-- which is the pairing a point-in-time join depends on and which a "
        "bare frame throws away.",
        BuildDataBundleInput,
    ),
    (
        "describe_data_bundle",
        "What frames a bundle names, how many rows and columns each has, and "
        "what each source can promise about revisions and point-in-time "
        "availability. Use it to see what a bundle actually contains before "
        "building a dataset on it, rather than after a model has already "
        "been fitted on whatever was in there.",
        DataBundleRefInput,
    ),
    (
        "validate_data_bundle",
        "Whether a bundle is safe to model on, returned as a verdict with "
        "the blocking reasons rather than raised as an error, because the "
        "answer is usually yes-with-caveats and a caller needs the caveats "
        "to decide. `require_pit` defaults to false: no shipped provider "
        "reports point-in-time for every frame kind, so requiring it refuses "
        "almost everything -- set it when a leakage-free join is the point.",
        ValidateDataBundleInput,
    ),
    (
        "validate_financial_ratios",
        "Check ratios you already hold -- from a vendor, a spreadsheet, "
        "another system -- for values that are implausible on their face, "
        "without fetching anything. The same check fetch_financial_ratios "
        "applies, available for data this library has no provider for.",
        ValidateFinancialRatiosInput,
    ),
    (
        "compare_ratio_frames",
        "Two providers' ratios side by side, with each disagreement "
        "CLASSIFIED rather than merely measured: a unit mismatch is fixable "
        "by rescaling, a definition difference is not, and averaging across "
        "the second kind produces a number neither provider would stand "
        "behind. Takes the values as arguments, so it works for sources this "
        "library cannot fetch.",
        CompareRatioFramesInput,
    ),
    (
        "build_continuous_futures_series",
        "Stitch a chain of futures contracts into one continuous series, and "
        "publish TWO references rather than one. The adjusted series is a "
        "research instrument and is not a price -- back-adjustment changes "
        "every historical level, and a difference-adjusted series can go "
        "negative on a contract that never traded below zero -- so sizing a "
        "position from it means sizing against a number nobody could have "
        "transacted at. The second reference carries which contract was "
        "actually active each date and what it actually traded at.",
        ContinuousFuturesInput,
    ),
]

TOOL_DISPATCH = {
    "fetch_ohlcv": (fetch_ohlcv, FetchOhlcvInput),
    "fetch_ohlcv_panel": (fetch_ohlcv_panel, FetchOhlcvPanelInput),
    "fetch_returns_panel": (fetch_returns_panel, FetchReturnsPanelInput),
    "fetch_tick_tape": (fetch_tick_tape, FetchTickTapeInput),
    "fetch_quote_panel": (fetch_quote_panel, FetchQuotePanelInput),
    "fetch_financial_ratios": (
        fetch_financial_ratios,
        FetchFinancialRatiosInput,
    ),
    "get_dataset_metadata": (get_dataset_metadata, DatasetMetadataInput),
    "infer_temporal_contract": (
        infer_temporal_contract,
        InferTemporalContractInput,
    ),
    "build_data_bundle": (build_data_bundle, BuildDataBundleInput),
    "describe_data_bundle": (describe_data_bundle, DataBundleRefInput),
    "validate_data_bundle": (validate_data_bundle, ValidateDataBundleInput),
    "validate_financial_ratios": (
        validate_financial_ratios,
        ValidateFinancialRatiosInput,
    ),
    "compare_ratio_frames": (compare_ratio_frames, CompareRatioFramesInput),
    "build_continuous_futures_series": (
        build_continuous_futures_series,
        ContinuousFuturesInput,
    ),
}

#: Every tool here belongs to the one category this runtime owns.
TOOL_CATEGORY = {name: "data" for name in TOOL_DISPATCH}

__all__ = [
    "TOOL_CATEGORY",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "build_continuous_futures_series",
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
