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
    "FetchResult",
    "FinancialRatiosResult",
    "RatioComparisonResult",
    "RatioFieldComparison",
    "Stat",
    "TemporalContractResult",
]
