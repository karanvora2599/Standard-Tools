"""
Typed results for the `data` runtime.

WHY TYPED. The MCP server builds each tool's structured-output schema from
its return annotation, and an untyped return silently drops it -- a client
then receives JSON it has no schema for and an agent guesses at key names.
`test_every_tool_has_an_output_schema` pins it.

WHAT THESE RESULTS MOSTLY CARRY IS A REFERENCE, not data. A fetch tool that
returned its frame inline would put the whole panel into the conversation,
where every subsequent turn pays for it again -- a 2,000-ticker daily panel
is megabytes. Returning `sqt://...` plus enough shape to decide what to do
next (rows, date span, entities) is what makes the fetch reusable across
runtimes instead of repeated inside each one.

NUMERIC FIELDS ARE `Stat`, mapping non-finite to null, for the reason the
rest of the surface does it: JSON has no NaN literal and a strict client
rejects the whole document over one.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools.agent.runtimes._json_safe import (
    finite_or_none as _finite_or_none,
)

Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


class _Result(BaseModel):
    model_config = ConfigDict(extra="forbid")
    warnings: List[str] = Field(default_factory=list)


class FetchResult(_Result):
    """A published frame: where it is, and enough shape to use it."""

    ref: str = Field(..., description="Resolve this from any runtime.")
    kind: str = Field(..., description="The reference kind that was published.")
    rows: int = 0
    columns: List[str] = Field(default_factory=list)
    entities: List[str] = Field(
        default_factory=list, description="Tickers present, where applicable."
    )
    start: Optional[str] = None
    end: Optional[str] = None
    dataset: Optional[str] = Field(
        None,
        description=(
            "The vendor dataset that actually answered, when the provider "
            "says. This is not decoration: a provider that picks between "
            "feeds by date can answer one window from the consolidated "
            "tape and an earlier one from a single-venue sample carrying a "
            "few percent of volume, and the rows look the same either way. "
            "None when the provider records no dataset."
        ),
    )
    provider: Optional[str] = Field(
        None, description="Which provider served it, when the frame records it."
    )
    adjusted: Optional[bool] = Field(
        None,
        description=(
            "Whether prices carry split and dividend adjustment. False "
            "means a split is a real -50% bar. None when unrecorded, which "
            "is not the same as False."
        ),
    )
    source: Optional[str] = Field(
        None,
        description=(
            "Provider and dataset as one string, the spelling the audit "
            "record uses for the same fetch, so a result and a decision "
            "record can be matched up without re-deriving it."
        ),
    )


class FinancialRatiosResult(_Result):
    symbol: str
    ratios: Dict[str, Any] = Field(default_factory=dict)
    implausible: List[str] = Field(
        default_factory=list,
        description="Values that fail a plausibility check on their face.",
    )


class DatasetMetadataResult(_Result):
    symbol: str
    interval: str
    provider: Optional[str] = None
    adjusted: Optional[bool] = None
    survivorship_free: Optional[bool] = None
    point_in_time: Optional[bool] = None
    timezone: Optional[str] = None
    notes: List[str] = Field(
        default_factory=list,
        description=(
            "The provider's own prose about what it serves -- which feed "
            "answers which window, how the index is stamped, which symbols "
            "it refuses. The four booleans above are the guarantees; this "
            "is where a provider names a sampling problem that no boolean "
            "has a slot for, so it was the field worth not dropping."
        ),
    )


class TemporalContractResult(_Result):
    ref: str
    frame_kind: str
    source: str
    pit_safe: bool = False
    reproduces_history: bool = False
    revisions: Optional[str] = None
    available_time_column: Optional[str] = None
    why_not_pit_safe: Optional[str] = None
    caveats: List[str] = Field(default_factory=list)


class BundleFrameSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_kind: str
    ref: Optional[str] = None
    source: Optional[str] = None
    rows: Optional[int] = None
    columns: List[str] = Field(default_factory=list)
    pit_safe: Optional[bool] = None
    reproduces_history: Optional[bool] = None
    revisions: Optional[str] = None
    caveats: List[str] = Field(default_factory=list)


class DataBundleResult(_Result):
    """A bundle: what it names, and what those sources can promise."""

    ref: Optional[str] = Field(
        None, description="Set when the bundle was just published."
    )
    name: str = ""
    n_frames: int = 0
    kinds: List[str] = Field(default_factory=list)
    frames: List[BundleFrameSummary] = Field(default_factory=list)
    pit_safe: bool = False
    reproduces_history: bool = False


class BundleVerdictResult(_Result):
    """Whether a bundle is safe to model on, and what blocks it."""

    name: str = ""
    n_frames: int = 0
    kinds: List[str] = Field(default_factory=list)
    pit_safe: bool = False
    reproduces_history: bool = False
    usable: bool = False
    blocking: List[str] = Field(
        default_factory=list,
        description="Reasons the bundle fails the requirement it was given.",
    )


class ExternalDatasetResult(_Result):
    """A dataset registered where it lies: what it is, and how big."""

    ref: str = Field(..., description="Resolve this from any runtime.")
    kind: str = ""
    path: str = Field("", description="Where the data actually is. Not copied.")
    file_format: str = ""
    rows: Optional[int] = Field(
        None,
        description=(
            "Exact. Free for Parquet, which carries it in the footer; a "
            "full scan for CSV, which does not."
        ),
    )
    columns: List[str] = Field(default_factory=list)
    dtypes: Dict[str, str] = Field(default_factory=dict)
    n_files: int = 0
    size_bytes: int = 0
    levels: Optional[int] = Field(
        None,
        description=(
            "Complete depth levels, for an order_book_panel. A level needs "
            "all four of its columns; counting stops at the first gap, "
            "because a price with no size is not a level."
        ),
    )
    fingerprint: str = Field(
        "",
        description=(
            "Digest of every file's name, size and mtime -- NOT a content "
            "hash. It catches a re-extract or a truncated copy; it does not "
            "catch an edit that preserves size and mtime. Hashing the bytes "
            "would cost the full read this whole path exists to avoid."
        ),
    )
    changed_since_registration: Optional[bool] = Field(
        None,
        description=(
            "Whether the file moved or changed since it was registered. An "
            "external file belongs to the caller and can be re-extracted "
            "under a live reference, which a copied artifact cannot."
        ),
    )
    preview: List[Dict[str, Any]] = Field(
        default_factory=list, description="Leading rows, for looking at."
    )


class VendorExtractResult(_Result):
    """A converted extract: where it went, and every judgement it made."""

    out_path: str = Field(
        "",
        description=(
            "The converted Parquet. Empty on a dry run, which writes "
            "nothing. Hand this to register_external_dataset as `path`."
        ),
    )
    kind: str = Field("", description="The kind it is now ready to register as.")
    rows_written: int = 0
    columns: List[str] = Field(default_factory=list)
    source_columns: List[str] = Field(
        default_factory=list,
        description="The vendor's own spelling, for comparison. First 24.",
    )
    looked_like_databento: bool = Field(
        False,
        description=(
            "Whether the source columns carry Databento's spelling. False "
            "does not mean the conversion is wrong -- another vendor may "
            "use the same shapes -- only that this was not recognized."
        ),
    )
    levels_available: Optional[int] = Field(
        None, description="Book only: complete depth levels in the source."
    )
    levels_kept: Optional[int] = Field(
        None,
        description=(
            "Book only: how many survived. Below `levels_available` when a "
            "trailing level was empty in every snapshot."
        ),
    )
    notes: List[str] = Field(
        default_factory=list,
        description=(
            "Every judgement the conversion made, in its own words -- which "
            "timestamp was taken, what price scale was applied, how many "
            "sentinels became null, which levels were dropped. Two of these "
            "change the numbers and neither is inferable from the result, "
            "which is why they are returned rather than logged. Deduplicated "
            "across batches: the same sentence about each batch is one fact, "
            "not a hundred."
        ),
    )
    next_step: str = Field(
        "",
        description="The register_external_dataset call this file is ready for.",
    )


class ExternalValidationResult(_Result):
    """Whether a registered dataset is safe to model on, and what blocks it."""

    ref: str = ""
    kind: str = ""
    usable: bool = False
    blocking: List[str] = Field(
        default_factory=list,
        description="Reasons this dataset would produce wrong numbers.",
    )
    rows_scanned: int = 0
    rows_total: Optional[int] = None
    coverage: Stat = Field(
        None, description="Fraction of the dataset the scan actually read."
    )
    batches: int = 0
    truncated: bool = Field(
        False,
        description=(
            "The scan hit scan_limit. Every count is over what was scanned, "
            "not over the whole dataset."
        ),
    )
    stats: Dict[str, Any] = Field(default_factory=dict)


class RatioFieldComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_name: str
    left: Stat = Field(
        None,
        description=(
            "The left source's value, when exactly one entity carried this "
            "field on both sides. Null across a universe, where a single "
            "pair of numbers would be a sample rather than the comparison."
        ),
    )
    right: Stat = Field(
        None, description="The right source's value, on the same terms."
    )
    relative_difference: Stat = Field(
        None,
        description=(
            "The LARGEST relative gap over the entities compared, scaled by "
            "the bigger of the two values. The worst case rather than the "
            "average, because one entity off by 100x is the finding."
        ),
    )
    ratio: Stat = Field(
        None,
        description=(
            "right / left, averaged over the entities where both sides are "
            "non-zero. Set only for a `scale` verdict, where it IS the "
            "conversion -- 100 means one source reports a percentage and "
            "the other a fraction."
        ),
    )
    ratio_spread: Stat = Field(
        None,
        description=(
            "How far that ratio wanders, as its coefficient of variation. "
            "Near zero is a unit error; a wandering ratio is why a "
            "`definition` difference cannot be converted away."
        ),
    )
    n_compared: int = Field(
        0, description="Entities where BOTH sources reported this field."
    )
    classification: Optional[str] = Field(
        None,
        description=(
            "What the gap most likely IS -- a unit mismatch, a definition "
            "difference, or a genuine data disagreement. The distinction is "
            "the point: only one of them is fixable by rescaling. "
            "'no_overlap' means neither source could be checked against the "
            "other for this field, which is silence rather than agreement."
        ),
    )


class RatioComparisonResult(_Result):
    left_name: str = "left"
    right_name: str = "right"
    n_compared: int = 0
    n_disagreeing: int = 0
    n_no_overlap: int = Field(
        0,
        description=(
            "Fields no entity reported on both sides. These are counted "
            "here and NOT in `n_disagreeing`: an unanswered question is not "
            "a disagreement, and counting it as one made two identical "
            "inputs read as a total mismatch."
        ),
    )
    fields: List[RatioFieldComparison] = Field(default_factory=list)


__all__ = [
    "BundleFrameSummary",
    "BundleVerdictResult",
    "DataBundleResult",
    "DatasetMetadataResult",
    "ExternalDatasetResult",
    "ExternalValidationResult",
    "FetchResult",
    "FinancialRatiosResult",
    "RatioComparisonResult",
    "RatioFieldComparison",
    "Stat",
    "TemporalContractResult",
]


class ContinuousFuturesResult(BaseModel):
    """Two references, deliberately, and they are not interchangeable."""

    model_config = ConfigDict(extra="forbid")

    research_ref: str = Field(
        ...,
        description="The ADJUSTED continuous series, as a `price_panel`. For "
        "indicators and signals. NOT a price -- do not size from it.",
    )
    tradeable_ref: str = Field(
        ...,
        description="Which contract was active on each date and what it "
        "actually traded at, as a `price_panel`. Size positions, cost "
        "trades and place stops from THIS one.",
    )
    roll_rule: str = ""
    adjustment: str = ""
    n_contracts: int = 0
    n_observations: int = 0
    n_rolls: int = 0
    roll_dates: List[str] = Field(default_factory=list)
    contracts_used: List[str] = Field(default_factory=list)
    start: Optional[str] = None
    end: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
