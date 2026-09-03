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

import math
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if math.isfinite(float(value)) else None
    return value


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
    left: Stat = None
    right: Stat = None
    relative_difference: Stat = None
    classification: Optional[str] = Field(
        None,
        description=(
            "What the gap most likely IS -- a unit mismatch, a definition "
            "difference, or a genuine data disagreement. The distinction is "
            "the point: only one of them is fixable by rescaling."
        ),
    )


class RatioComparisonResult(_Result):
    left_name: str = "left"
    right_name: str = "right"
    n_compared: int = 0
    n_disagreeing: int = 0
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
