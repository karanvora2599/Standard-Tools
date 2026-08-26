"""
Inputs for the `data` runtime, and the polymorphic source every new tool
takes.

WHY A SOURCE OBJECT RATHER THAN A SYMBOL. The rest of this surface grew up
assuming a tool fetches its own data, so a question like "what is this
series' Sharpe" is reachable only for a ticker this library can fetch. The
same arithmetic applied to a model's out-of-sample returns, an external
fund's monthly series, or a panel another agent already published needs a
different tool name under that assumption -- and that is how a surface ends
up with `calculate_beta`, `calculate_beta_from_returns` and
`calculate_beta_from_artifact` answering one question three times.

`DataSource` says the tool is the QUESTION and the input says where the
bytes are. Three origins, one schema:

    {"symbol": "NVDA"}                  fetch it
    {"ref": "sqt://returns_panel/..."}  something already computed it
    {"values": [0.01, -0.004, ...]}     the caller has it in hand

EXACTLY ONE, enforced. Accepting two and picking a winner would make the
precedence rule part of the contract, and a caller who passed both almost
certainly meant something the tool cannot know.

THIS CONVENTION IS FOR NEW TOOLS ONLY. Every tool that existed before this
runtime keeps its signature: retrofitting ~130 input models is a separate
decision with its own blast radius, and doing it halfway would leave
callers unable to predict which convention a given tool follows.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataSource(BaseModel):
    """Where a tool's data comes from: a symbol, a reference, or values."""

    model_config = ConfigDict(extra="forbid")

    symbol: Optional[str] = Field(
        None, description="A ticker this library's provider can fetch."
    )
    ref: Optional[str] = Field(
        None,
        description=(
            "An `sqt://` reference published by another tool. Resolving it "
            "does not refetch anything."
        ),
    )
    values: Optional[List[float]] = Field(
        None,
        description=(
            "The series inline, oldest first. For data this library has no "
            "provider for -- an external fund's returns, a vendor extract."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "DataSource":
        given = [
            name
            for name, value in (
                ("symbol", self.symbol),
                ("ref", self.ref),
                ("values", self.values),
            )
            if value is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "a data source needs exactly one of symbol / ref / values; "
                f"got {given or 'none'}. Two sources would make the "
                "precedence rule part of the contract, and a caller who "
                "passed both meant something this tool cannot infer."
            )
        return self


class _Fetch(BaseModel):
    """Shared shape for the fetch tools: what, when, and where to put it."""

    model_config = ConfigDict(extra="forbid")

    start_date: str = Field(..., description="Inclusive, YYYY-MM-DD.")
    end_date: str = Field(..., description="Inclusive, YYYY-MM-DD.")
    run_id: str = Field(
        ...,
        description=(
            "Groups the artifacts of one workflow. Reused ids collide and "
            "are refused rather than overwritten, because a reference is a "
            "promise that resolving it twice gives the same value."
        ),
    )
    name: str = Field(..., description="Names this artifact within the run.")


class FetchOhlcvInput(_Fetch):
    symbol: str = Field(..., description="One ticker.")
    interval: str = Field("1d", description="Bar interval, e.g. 1d, 1h, 5m.")


class FetchOhlcvPanelInput(_Fetch):
    tickers: List[str] = Field(..., min_length=1, description="The universe.")
    interval: str = Field("1d", description="Bar interval, e.g. 1d, 1h, 5m.")


class FetchReturnsPanelInput(_Fetch):
    tickers: List[str] = Field(..., min_length=1, description="The universe.")
    interval: str = Field("1d", description="Bar interval, e.g. 1d, 1h, 5m.")


class FetchTickTapeInput(_Fetch):
    symbol: str = Field(..., description="One ticker.")
    limit: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Cap on trades returned. A tape is large; without a cap a wide "
            "window can be millions of rows."
        ),
    )


class FetchQuotePanelInput(_Fetch):
    symbol: str = Field(..., description="One ticker.")
    limit: Optional[int] = Field(None, gt=0, description="Cap on quotes returned.")


class FetchFinancialRatiosInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(..., description="One ticker.")


class DatasetMetadataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(..., description="A representative ticker.")
    interval: str = Field("1d", description="Bar interval the claim is about.")


class InferTemporalContractInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ref: str = Field(..., description="The `sqt://` reference to inspect.")
    source: str = Field(
        "unknown", description="Who supplied it, recorded on the contract."
    )
    frame_kind: str = Field(
        "bars", description="What the frame holds: bars, fundamentals, ..."
    )
    entity_scoped: bool = Field(
        True, description="Whether rows are per entity rather than global."
    )


class BundleFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_kind: str = Field(
        ..., description="bars, fundamentals, estimates, macro, events, ..."
    )
    ref: str = Field(..., description="An `sqt://` reference to that frame.")
    source: str = Field("unknown", description="Who supplied it.")


class BuildDataBundleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames: List[BundleFrame] = Field(..., min_length=1)
    run_id: str = Field(..., description="Groups this workflow's artifacts.")
    name: str = Field(..., description="Names this bundle within the run.")


class DataBundleRefInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ref: str = Field(..., description="An `sqt://data_bundle/...` reference.")


class ValidateDataBundleInput(DataBundleRefInput):
    require_pit: bool = Field(
        False,
        description=(
            "Refuse the bundle unless every frame can be joined "
            "point-in-time. Defaults FALSE because no shipped provider "
            "reports point_in_time=True for every frame kind, so requiring "
            "it refuses almost everything; set it when a leakage-sensitive "
            "join is the point."
        ),
    )


class ValidateFinancialRatiosInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ratios: dict = Field(
        ...,
        description=(
            "Field name to value, as a provider returned them. Checked for "
            "values that are implausible on their face."
        ),
    )


class CompareRatioFramesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: dict = Field(..., description="One provider's ratios.")
    right: dict = Field(..., description="The other provider's ratios.")
    left_name: str = Field("left", description="Label for the first source.")
    right_name: str = Field("right", description="Label for the second.")
    fields: Optional[List[str]] = Field(
        None, description="Restrict the comparison to these fields."
    )


__all__ = [
    "BuildDataBundleInput",
    "BundleFrame",
    "CompareRatioFramesInput",
    "DataBundleRefInput",
    "DataSource",
    "DatasetMetadataInput",
    "FetchFinancialRatiosInput",
    "FetchOhlcvInput",
    "FetchOhlcvPanelInput",
    "FetchQuotePanelInput",
    "FetchReturnsPanelInput",
    "FetchTickTapeInput",
    "InferTemporalContractInput",
    "ValidateDataBundleInput",
    "ValidateFinancialRatiosInput",
]
