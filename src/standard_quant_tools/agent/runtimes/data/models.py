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

from typing import Dict, List, Literal, Optional

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


class RegisterExternalDatasetInput(BaseModel):
    """Where the data is, and what it claims to be."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description=(
            "A Parquet or CSV file, or a directory read as one partitioned "
            "dataset. Nothing is copied, so this has to be readable from "
            "wherever this library runs."
        ),
    )
    kind: Literal["order_book_panel", "event_panel", "tick_tape", "quote_panel"] = (
        Field(
            ...,
            description=(
                "What the dataset holds. Checked against its columns at "
                "registration, so a mismatch fails here rather than inside "
                "whatever first read a column that is not there."
            ),
        )
    )
    run_id: str = Field(..., description="Groups this workflow's artifacts.")
    name: str = Field(..., description="Names this dataset within the run.")
    file_format: Optional[Literal["parquet", "csv"]] = Field(
        None,
        description=(
            "Override the format inferred from the suffix. Needed for an "
            "extract whose name does not end in .parquet or .csv."
        ),
    )
    source: str = Field(
        "unknown",
        description="Who supplied it, recorded on the registration.",
    )


class ExternalDatasetRefInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ref: str = Field(..., description="A reference from register_external_dataset.")
    preview_rows: int = Field(
        5,
        ge=0,
        le=50,
        description=(
            "How many leading rows to return for looking at. Bounded hard: "
            "a preview is for seeing the shape, and anything that has to be "
            "right about the whole dataset reads it in batches instead."
        ),
    )


class ValidateExternalDatasetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ref: str = Field(..., description="A reference from register_external_dataset.")
    scan_limit: int = Field(
        2_000_000,
        gt=0,
        le=200_000_000,
        description=(
            "Stop after this many rows. A verdict over a bounded prefix "
            "reports what it covered; a full scan of a billion-row tape is "
            "not something to do inside one tool call."
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
    "ExternalDatasetRefInput",
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
    "RegisterExternalDatasetInput",
    "ValidateDataBundleInput",
    "ValidateExternalDatasetInput",
    "ValidateFinancialRatiosInput",
]


#: Typical |value| above this is not a fractional return. An equity curve
#: at any base -- 10 000, 100, or normalized to 1.0 -- clears it; a daily
#: return series sits five or six orders of magnitude below it. Median
#: rather than max, so one genuine +200% day does not trip it.
_MAX_PLAUSIBLE_TYPICAL_RETURN = 1.0


def _refuse_if_not_returns(series, what: str):
    """Refuse a series that cannot be returns, naming the likely cause.

    Deliberately ONE rule, and a blunt one. The tempting second rule --
    refuse a series with no negative observation -- would refuse a
    money-market or T-bill return series, which really is all-positive and
    really does have an enormous Sharpe against a 0% rate. Being unable to
    measure a real series is worse than the narrower guard, so the softer
    signal lives where there is a warnings channel to carry it.
    """
    from standard_quant_tools.error import ValidationError

    finite = series[series.notna()]
    if finite.empty:
        return
    typical = float(finite.abs().median())
    if typical > _MAX_PLAUSIBLE_TYPICAL_RETURN:
        raise ValidationError(
            f"{what}: this is a RETURN series, and the typical |value| here "
            f"is {typical:.4g} -- a {typical * 100:.0f}% move per period. "
            "That is a level series (an equity curve, or prices), not "
            "returns. A Sharpe taken on levels is scale-invariant and comes "
            "back two to three orders of magnitude too high with nothing "
            "looking wrong. Convert first -- `.pct_change().dropna()` -- or "
            "name a reference that already holds returns."
        )


def resolve_source(source: "DataSource", *, what: str = "series"):
    """
    Turn a `DataSource` into a pandas Series, whichever origin it named.

    THIS IS THE HALF THAT MAKES THE POLYMORPHISM WORTH HAVING. A tool that
    accepted three shapes and then branched on them in its own body would
    have three code paths to keep correct and three places for the
    date-alignment rules to drift. Resolving here means the tool sees one
    Series and never learns where it came from.

    A symbol is fetched as CLOSE-TO-CLOSE RETURNS rather than prices,
    because every consumer of this helper so far wants returns and a tool
    that silently handed one prices would produce a Sharpe off by orders of
    magnitude with nothing looking wrong.

    THAT REASONING GUARDED ONE OF THE THREE PATHS. `values` took whatever it
    was given and `ref` resolved ANY kind -- no `expect=` -- so an
    `sqt://equity_curve` reference, which is levels, arrived here and was
    treated as returns. Measured: a series whose true annualized Sharpe is
    0.2953 reported 302.5466, at every base, with no warning, because
    mean/std is scale-invariant. Whether it was caught depended on which
    METRICS were asked for: a raw curve overflows `(1+x).cumprod()` and
    `max_drawdown`'s own validator refuses the infinities, so asking for a
    drawdown protected you and asking for a Sharpe alone did not.

    `_refuse_if_not_returns` closes that, on all three paths.
    """
    import pandas as pd

    from standard_quant_tools.error import ValidationError

    if source.values is not None:
        series = pd.Series(source.values, dtype="float64")
        if series.empty:
            raise ValidationError(f"{what}: `values` was empty.")
        _refuse_if_not_returns(series, what)
        return series

    if source.ref is not None:
        from ..handoff import resolve as _resolve

        try:
            data = _resolve(source.ref)
        except Exception as exc:  # noqa: BLE001 -- one refusal, not a trace
            raise ValidationError(
                f"{what}: {source.ref!r} could not be resolved -- {exc}"
            ) from exc
        if isinstance(data, pd.DataFrame):
            if data.shape[1] != 1:
                raise ValidationError(
                    f"{what}: {source.ref!r} resolves to a frame with "
                    f"{data.shape[1]} columns, and this tool needs ONE "
                    "series. Name a single-column artifact, or pass the "
                    "column you mean inline."
                )
            data = data.iloc[:, 0]
        series = pd.Series(data).astype("float64")
        if series.empty:
            raise ValidationError(f"{what}: {source.ref!r} resolved to nothing.")
        _refuse_if_not_returns(series, f"{what}: {source.ref!r}")
        return series

    from standard_quant_tools.data.factory import DataFactory

    provider = DataFactory.get_provider()
    frame = provider.get_ohlcv(source.symbol, "1900-01-01", "2100-01-01")
    if frame is None or frame.empty:
        raise ValidationError(f"{what}: {source.symbol!r} returned no bars.")
    return frame["Close"].pct_change(fill_method=None).dropna()


class FuturesContractSeries(BaseModel):
    """One contract's history in a chain."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="Contract code, e.g. 'ESH6'.")
    expiry: str = Field(..., description="ISO expiry date.")
    prices: Dict[str, float] = Field(
        ..., min_length=1, description="ISO date to close, for THIS contract."
    )
    volume: Optional[Dict[str, float]] = Field(
        None, description="Required when roll_rule is 'volume'."
    )
    open_interest: Optional[Dict[str, float]] = Field(
        None, description="Required when roll_rule is 'open_interest'."
    )


class ContinuousFuturesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contracts: List[FuturesContractSeries] = Field(
        ..., min_length=2, description="The chain, in any order."
    )
    roll_rule: Literal["volume", "open_interest", "days_before_expiry"] = Field(
        "volume",
        description="The three produce different series from the same "
        "contracts and disagree most where liquidity moved unusually.",
    )
    adjustment: Literal["none", "difference", "ratio"] = Field(
        "ratio",
        description="difference preserves point moves and can drive history "
        "negative; ratio preserves compounding and cannot.",
    )
    days_before_expiry: int = Field(
        5, ge=0, le=365, description="Only read by the days_before_expiry rule."
    )
    run_id: str = Field(..., description="Groups this workflow's artifacts.")
    name: str = Field(..., description="Names the series within the run.")
