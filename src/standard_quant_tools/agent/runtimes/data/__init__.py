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
prevent.

DEPTH IS NOT ABSENT ANY MORE, and the argument that kept it out did not
survive being written down. It ran: `DataProvider.get_order_book` has one
implementation, a fetch tool would have to answer for every provider, and
nine of ten refuse. But refusing for nine of ten is what `fetch_tick_tape`
and `fetch_quote_panel` have always done -- by name, pointing at
`describe_data_capabilities` -- and meanwhile the two tools that consume a
book could be fed only by `register_external_dataset`, which requires
already holding one. Fetching a book and HAVING a book are different
problems and only the first was ever blocked.

WHAT THE FETCH TOOLS DO INSTEAD OF PUBLISHING. The reason the depth kinds
are external is real and unchanged: every provider call materializes a whole
frame and `publish` then writes a SECOND complete copy under SQT_RUNS_DIR,
which is survivable for a decade of daily bars and not for an afternoon of
depth. So `fetch_order_book` and `fetch_order_events` write one Parquet and
REGISTER it -- one copy, a pointer, a schema, and batched reads -- which is
the same shape the three registration tools give a file the caller already
had, reached without having to have it first.
"""

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
from .tools import (
    build_continuous_futures_series,
    build_data_bundle,
    compare_ratio_frames,
    describe_data_bundle,
    describe_external_dataset,
    fetch_financial_ratios,
    fetch_ohlcv,
    fetch_ohlcv_panel,
    fetch_order_book,
    fetch_order_events,
    fetch_quote_panel,
    fetch_returns_panel,
    fetch_tick_tape,
    get_dataset_metadata,
    infer_temporal_contract,
    preflight_vendor_request,
    prepare_vendor_extract,
    register_external_dataset,
    validate_data_bundle,
    validate_external_dataset,
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
        "alongside a tape. Top of book ONLY -- depth is a different call, "
        "and provider='databento' serves it through get_order_book. Queue "
        "position is in neither: it needs an order-level feed and cannot be "
        "inferred from aggregated size at a level.",
        FetchQuotePanelInput,
    ),
    (
        "fetch_order_book",
        "Fetch L2 depth snapshots -- price and resting size at each level, "
        "level 1 the touch -- write them once under the run and return an "
        "order_book_panel reference. THIS IS THE ONLY WAY TO OBTAIN A BOOK "
        "in this library other than already having one: the analysis that "
        "reads depth could previously be fed only by register_external_"
        "dataset, which wants a file you captured elsewhere. The reference "
        "is external, so resolving it streams the file in batches rather "
        "than loading a session into memory; hand it to "
        "get_order_book_metrics as `ref`, and to detect_liquidity_events' "
        "depth channels once that detector grows the per-snapshot series "
        "step they name. THE FEED IS METERED AND DEPTH IS THE EXPENSIVE "
        "SCHEMA: five minutes of one active name at ten levels measured "
        "about 42 MB. The WINDOW is what the vendor bills -- `limit` "
        "(20,000 snapshots by default, roughly half an hour of an active "
        "name) caps only what is written, and the result says when it "
        "bound. Price the window with preflight_vendor_request before "
        "widening it. Needs a provider that serves depth; one that does not "
        "refuses by name and points at describe_data_capabilities.",
        FetchOrderBookInput,
    ),
    (
        "fetch_order_events",
        "Fetch order-by-order events -- every add, cancel, modify and fill "
        "with its own id -- write them once under the run and return an "
        "order_event_panel reference for get_order_event_metrics to read as "
        "`ref`. A STRICTLY DEEPER FEED THAN DEPTH, and the difference is "
        "identity rather than levels: a book snapshot aggregates size per "
        "price, and that aggregation is what makes queue position, order "
        "lifetime and a true cancellation rate impossible to recover. It is "
        "also far denser -- the window that yields thousands of book "
        "snapshots yields millions of events -- though cheaper per unit "
        "time than depth: the same five minutes of one name measured about "
        "14 MB. `limit` defaults to 100,000 events, a few minutes of an "
        "active name, and caps what is WRITTEN rather than what the vendor "
        "bills; the window does that. Needs a provider with an order-level "
        "feed; one without refuses by name pointing at "
        "describe_data_capabilities.",
        FetchOrderEventsInput,
    ),
    (
        "preflight_vendor_request",
        "What a vendor request would cost and whether the data is even "
        "there -- asked for free, before the request is made. Returns the "
        "DATASET that would answer (the routing is date-dependent and "
        "feed-dependent: daily bars prefer a consolidated summary where it "
        "reaches and fall back to a sample feed carrying a few percent of "
        "volume, while depth comes from one venue), that dataset's coverage "
        "window, the BYTES the vendor would bill, and the reference kind "
        "the schema produces. Bytes rather than money on purpose: an "
        "account whose subscription already includes the feed is quoted "
        "$0.00 for a request of any size, so the price is silent about "
        "exactly the thing it is consulted for. A window the dataset does "
        "not cover is said so rather than fabricated, and a dataset that "
        "reports nothing comes back null rather than as a guess. Call it "
        "before fetch_order_book or fetch_order_events, which are the "
        "metered ones. Needs a provider that routes named vendor datasets; "
        "one without refuses naming source='databento'.",
        PreflightVendorRequestInput,
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
        "restated values under their original dates. `notes` carries what "
        "the booleans cannot -- which feed answers which window, and what is "
        "wrong with it.",
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
        "prepare_vendor_extract",
        "Convert a RAW vendor extract into this library's contract, the step "
        "BEFORE register_external_dataset. A Databento export spells the same "
        "quantity `bid_px_00` where this library spells it `bid_price_0`, "
        "stamps rows `ts_recv` rather than `timestamp`, sends prices as int64 "
        "nanodollars and fills absent levels with int64-max rather than null "
        "-- so registering it unconverted fails on the column names if you are "
        "lucky and puts $9.2 billion quotes in your book if you are not. "
        "Writes a new Parquet and REPORTS the two judgements that change the "
        "numbers and cannot be recovered from the output: which timestamp "
        "became `timestamp`, and whether prices were divided by a billion. "
        "Use dry_run to see both before committing a large file.",
        PrepareVendorExtractInput,
    ),
    (
        "register_external_dataset",
        "Make a Parquet or CSV dataset already on your disk resolvable as an "
        "`sqt://` reference WITHOUT copying it, for data too large to fetch "
        "and republish -- L2 depth, a full tick tape, an event history. Every "
        "other tool here fetches a frame and then writes a second complete "
        "copy under the runs directory; a book cannot survive that twice. "
        "Registration reads the schema, checks the columns the declared kind "
        "requires, and stores a pointer. It does NOT read the rows, so a book "
        "with its bid and ask columns transposed registers cleanly -- run "
        "validate_external_dataset next.",
        RegisterExternalDatasetInput,
    ),
    (
        "describe_external_dataset",
        "What a registered dataset holds -- columns, dtypes, row count, depth "
        "levels, file count and size -- plus whether the file has changed "
        "since it was registered. That last field has no equivalent for a "
        "published artifact and is the price of not copying: this library "
        "wrote and froze its own artifacts, but an external file belongs to "
        "you and can be re-extracted underneath a live reference. Returns a "
        "bounded preview of leading rows for looking at, never the dataset.",
        ExternalDatasetRefInput,
    ),
    (
        "validate_external_dataset",
        "Scan a registered dataset in batches and report what would produce "
        "wrong numbers downstream, as a verdict with blocking reasons rather "
        "than an exception -- because three crossed books in nine million "
        "rows is fine and a third of them crossed is transposed columns, and "
        "only a count separates the two. Checks are per kind: crossed and "
        "empty books, non-positive trade prices and sizes, unparseable or "
        "out-of-order timestamps, and for an event panel the available_time "
        "versus event_time rule that makes a model look prescient. Bounded by "
        "scan_limit, and says so when it stops early.",
        ValidateExternalDatasetInput,
    ),
    (
        "build_data_bundle",
        "Name several already-published frames as one unit and publish the "
        "manifest as a data_bundle reference. A bundle holds references "
        "rather than copies, so it cannot diverge from the frames it names, "
        "and it pairs each frame with what its source can say about timing "
        "-- which is the pairing a point-in-time join depends on and which a "
        "bare frame throws away. Each `frame_kind` label is CHECKED against "
        "what its reference actually is, because the label chooses the "
        "contract the bundle is later validated under: mislabelling a "
        "returns panel as fundamentals used to buy a confident "
        "point-in-time verdict about the wrong thing.",
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
        "library cannot fetch, and accepts fetch_financial_ratios' own "
        "output on either side -- one company's flat field map or a map of "
        "ticker -> ratios, the same shape on both sides. A field neither "
        "source reported is counted as `n_no_overlap`, never as a "
        "disagreement.",
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
    "fetch_order_book": (fetch_order_book, FetchOrderBookInput),
    "fetch_order_events": (fetch_order_events, FetchOrderEventsInput),
    "preflight_vendor_request": (
        preflight_vendor_request,
        PreflightVendorRequestInput,
    ),
    "fetch_financial_ratios": (
        fetch_financial_ratios,
        FetchFinancialRatiosInput,
    ),
    "get_dataset_metadata": (get_dataset_metadata, DatasetMetadataInput),
    "infer_temporal_contract": (
        infer_temporal_contract,
        InferTemporalContractInput,
    ),
    "prepare_vendor_extract": (
        prepare_vendor_extract,
        PrepareVendorExtractInput,
    ),
    "register_external_dataset": (
        register_external_dataset,
        RegisterExternalDatasetInput,
    ),
    "describe_external_dataset": (
        describe_external_dataset,
        ExternalDatasetRefInput,
    ),
    "validate_external_dataset": (
        validate_external_dataset,
        ValidateExternalDatasetInput,
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
    "describe_external_dataset",
    "fetch_financial_ratios",
    "fetch_ohlcv",
    "fetch_ohlcv_panel",
    "fetch_order_book",
    "fetch_order_events",
    "fetch_quote_panel",
    "fetch_returns_panel",
    "fetch_tick_tape",
    "get_dataset_metadata",
    "infer_temporal_contract",
    "preflight_vendor_request",
    "prepare_vendor_extract",
    "register_external_dataset",
    "validate_data_bundle",
    "validate_external_dataset",
    "validate_financial_ratios",
]
