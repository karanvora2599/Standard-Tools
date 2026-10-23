"""
How hard to lever a return stream, what an overlay that changes its risk
over time does to it, and how much capital it can take.

- `get_kelly_report` -- the Kelly leverage of one stream or several held
  together, how uncertain it is, and the growth and drawdowns at fractions
  of it (`allocation.kelly`). It REPORTS a leverage; it sizes nothing, and
  no other tool applies it.
- `apply_risk_overlay` -- a volatility target on lagged EWMA, CPPI or TIPP
  over one stream, the exposure at t decided from returns through t - 1
  (`allocation.overlays`); the overlaid stream and the exposure path are
  published at a `run_id`.
- `get_strategy_capacity_curve` -- net return and Sharpe ratio against AUM
  under spread plus square-root impact on the strategy's own weight
  changes (`allocation.capacity`).

STREAMS, NEVER TICKERS. Each takes dated values inline or a published
`returns_panel` reference, never a symbol, so nothing is fetched and the
same input gives the same bits. Inline values are read by the stream
helpers every `ReturnStream` is read by (`agent.runtimes.streams`), and
checked by the same `align_return_streams`; a `returns_ref` is read by
`returns_ref_streams`. The `ReturnStream` model itself is not an input
here because it also takes a `symbol`, which would make these tools
fetchers.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, ClassVar, Dict, List, Literal, Optional, Tuple

import pandas as pd
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from standard_quant_tools.agent.models import Seed
from standard_quant_tools.agent.runtimes._json_safe import (
    ExplainsNulls,
)
from standard_quant_tools.agent.runtimes._json_safe import (
    finite_or_none as _finite_or_none,
)
from standard_quant_tools.agent.runtimes.streams import (
    StreamAlignment,
    aligned_dated_streams,
    aligned_returns_ref,
    aligned_streams,
    columns_only_with_ref,
    dated_series,
    exactly_one_source,
    finite_dated_series,
    finite_dated_streams,
    returns_ref_streams,
)
from standard_quant_tools.allocation import capacity as capacity_lib
from standard_quant_tools.allocation import kelly as kelly_lib
from standard_quant_tools.allocation import overlays as overlay_lib
from standard_quant_tools.error import ValidationError

Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]

#: The most streams and instruments one call takes. Held in the validators
#: rather than the schema: a guard against a runaway payload, not something
#: a caller plans around.
MAX_STREAMS = 500
MAX_INSTRUMENTS = 5_000

# ── inputs ──────────────────────────────────────────────────────────────


class _OneStream(BaseModel):
    """One return stream: dated values inline, or a column of a panel."""

    model_config = ConfigDict(extra="forbid")

    #: The tool a refusal names.
    tool: ClassVar[str] = ""

    returns: Optional[Dict[str, float]] = Field(
        None, description="{ISO date: return}, the stream inline."
    )
    returns_ref: Optional[str] = Field(
        None, description="A returns_panel reference in place of returns."
    )
    column: Optional[str] = Field(
        None, description="returns_ref's column, if it has several."
    )
    periods_per_year: Optional[int] = Field(
        None, ge=1, le=100_000, description="Default: read off the dates."
    )

    @field_validator("returns")
    @classmethod
    def _finite_returns(cls, v: Optional[Dict[str, float]]) -> Optional[Dict]:
        if v is not None and not v:
            raise ValueError("returns is empty; give the stream's dated returns.")
        return finite_dated_series(v, "returns")

    @model_validator(mode="after")
    def _one_source(self) -> "_OneStream":
        exactly_one_source(self, "returns", "returns_ref", self.tool)
        columns_only_with_ref(self, "column", "returns_ref")
        return self


class KellyReportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="{stream: {ISO date: return}}: one stream, or several "
        "held together; aligned on the dates they share.",
    )
    returns_ref: Optional[str] = Field(
        None, description="A returns_panel reference in place of returns."
    )
    columns: Optional[List[str]] = Field(
        None, min_length=1, description="returns_ref's streams; default all."
    )
    risk_free_rate: float = Field(
        0.0, gt=-1, le=10, description="Annual decimal; leverage is financed at it."
    )
    periods_per_year: Optional[int] = Field(
        None, ge=1, le=100_000, description="Default: read off the dates."
    )
    fractions: List[Annotated[float, Field(gt=0, le=10)]] = Field(
        list(kelly_lib.DEFAULT_FRACTIONS),
        min_length=1,
        max_length=10,
        description="Fractions of Kelly to report.",
    )
    drawdown_levels: List[Annotated[float, Field(gt=0, lt=1)]] = Field(
        list(kelly_lib.DEFAULT_DRAWDOWN_LEVELS),
        min_length=1,
        max_length=10,
        description="Losses (fractions of wealth) whose odds are reported.",
    )
    confidence: float = Field(0.95, ge=0.5, lt=1)
    n_bootstrap: int = Field(1000, ge=0, le=10_000)
    block_size: Optional[int] = Field(None, ge=1, le=10_000)
    random_seed: Seed = 0

    @field_validator("returns")
    @classmethod
    def _finite_returns(cls, v: Optional[Dict]) -> Optional[Dict]:
        return finite_dated_streams(v, "returns", most=MAX_STREAMS)

    @model_validator(mode="after")
    def _one_source(self) -> "KellyReportInput":
        exactly_one_source(self, "returns", "returns_ref", "get_kelly_report")
        columns_only_with_ref(self, "columns", "returns_ref")
        return self


#: The fields each overlay reads; a field set for another method is refused.
_OVERLAY_FIELDS = {
    "vol_target": ("target_volatility", "ewma_lambda", "warmup"),
    "cppi": ("multiplier", "floor"),
    "tipp": ("multiplier", "floor"),
}


class RiskOverlayInput(_OneStream):
    tool: ClassVar[str] = "apply_risk_overlay"

    method: Literal["vol_target", "cppi", "tipp"] = "vol_target"
    target_volatility: float = Field(
        0.10, gt=0, le=10, description="vol_target: annual target."
    )
    ewma_lambda: float = Field(
        0.94, gt=0, lt=1, description="vol_target: EWMA decay (RiskMetrics)."
    )
    warmup: int = Field(
        20, ge=2, le=5_000, description="vol_target: returns seeding sigma."
    )
    multiplier: float = Field(
        3.0, gt=0, le=100, description="cppi/tipp: exposure per unit of cushion."
    )
    floor: float = Field(
        0.8,
        ge=0,
        lt=1,
        description="cppi: fraction of starting wealth; tipp: of peak wealth.",
    )
    max_leverage: float = Field(
        1.0, gt=0, le=100, description="Cap on exposure, a multiple of equity."
    )
    cash_rate: float = Field(
        0.0,
        gt=-1,
        le=10,
        description="Annual rate on the unexposed part (paid above 1x).",
    )
    cost_bps: float = Field(
        0.0, ge=0, le=10_000, description="Cost per unit change in exposure."
    )
    run_id: Optional[str] = Field(
        None,
        description="Publishes the overlaid stream and exposure; a used one "
        "is refused.",
    )

    @model_validator(mode="after")
    def _fields_of_the_method(self) -> "RiskOverlayInput":
        # Refused only when it differs from its default: a client that sends
        # every default it read off the schema has set nothing.
        read = set(_OVERLAY_FIELDS[self.method])
        stray = sorted(
            {
                f
                for fields in _OVERLAY_FIELDS.values()
                for f in fields
                if f not in read
                and f in self.model_fields_set
                and getattr(self, f) != type(self).model_fields[f].default
            }
        )
        if stray:
            raise ValueError(
                f"{', '.join(stray)} {'is' if len(stray) == 1 else 'are'} not "
                f"read by method={self.method!r}, which uses "
                f"{', '.join(sorted(read))}. Drop them, or choose the method "
                "that reads them."
            )
        return self


class StrategyCapacityInput(_OneStream):
    tool: ClassVar[str] = "get_strategy_capacity_curve"

    weights: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="{instrument: {ISO date: weight}}: target weights as "
        "fractions of AUM.",
    )
    weights_ref: Optional[str] = Field(
        None, description="A weight_panel reference in place of weights."
    )
    adv: Dict[str, float] = Field(
        ..., description="Instrument -> average daily volume, in currency."
    )
    daily_volatility: Dict[str, float] = Field(
        ..., description="Instrument -> daily return volatility, decimal."
    )
    spread_bps: Dict[str, float] = Field(
        ..., description="Instrument -> full quoted spread; half is paid a trade."
    )
    impact_coefficient: float = Field(
        capacity_lib.DEFAULT_IMPACT_COEFFICIENT,
        gt=0,
        le=100,
        description="k in k sigma sqrt(trade/ADV); 1 per Toth et al. (2011).",
    )
    aum: Optional[List[Annotated[float, Field(gt=0)]]] = Field(
        None,
        min_length=1,
        max_length=60,
        description="AUM grid, currency; default around the break-even.",
    )
    risk_free_rate: float = Field(0.0, gt=-1, le=10, description="Annual decimal.")

    @field_validator("weights")
    @classmethod
    def _finite_weights(cls, v: Optional[Dict]) -> Optional[Dict]:
        return finite_dated_streams(v, "weights", most=MAX_INSTRUMENTS)

    @field_validator("adv", "daily_volatility", "spread_bps")
    @classmethod
    def _finite_map(cls, v: Dict[str, float], info: Any) -> Dict[str, float]:
        if not v or len(v) > MAX_INSTRUMENTS:
            raise ValueError(
                f"{info.field_name} needs one figure per instrument (1 to "
                f"{MAX_INSTRUMENTS}); got {len(v)}."
            )
        bad = [k for k, x in v.items() if not (math.isfinite(x) and x >= 0)]
        if bad:
            raise ValueError(
                f"{info.field_name} is not a finite, non-negative number for "
                f"{bad[:5]}."
            )
        return v

    @model_validator(mode="after")
    def _one_panel(self) -> "StrategyCapacityInput":
        exactly_one_source(
            self, "weights", "weights_ref", "get_strategy_capacity_curve"
        )
        return self


# ── results ─────────────────────────────────────────────────────────────


class _Result(ExplainsNulls):
    """A result's caveats, and a reason beside each null number."""

    model_config = ConfigDict(extra="forbid")

    warnings: List[str] = Field(default_factory=list)


_NO_DRAWS = "n_bootstrap is 0, so no paths were drawn"
_NO_EDGE = (
    "the Sharpe ratio is zero: Kelly holds nothing, so there is no growth "
    "to share and nothing to lose"
)


class KellyStream(ExplainsNulls):
    """One stream's row, reported by the result that holds it."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "p_estimate_over_twice_truth": "the Kelly fraction is exactly zero",
    }

    kelly_fraction: Stat = Field(
        None,
        description="In-sample f*: the leverage on the stream, a multiple of "
        "equity. Negative: Kelly shorts it.",
    )
    standard_error: Stat = Field(
        None, description="1 / (sigma sqrt(T)); sqrt((Sigma^-1)_ii / T) for several."
    )
    interval: List[Stat] = Field(
        default_factory=list, description="f* -/+ z x standard_error at confidence."
    )
    p_estimate_over_twice_truth: Stat = Field(
        None,
        description="P(f-hat > 2 f*), the estimate taken as the truth: "
        "Phi(-|f*| / standard_error).",
    )
    shrunk_fraction: Stat = Field(None, description="shrink_factor x kelly_fraction.")
    mean_excess_return: Stat = Field(None, description="Annual, over the rate.")
    volatility: Stat = Field(None, description="Annual.")
    sharpe_ratio: Stat = None


class KellyFraction(ExplainsNulls):
    """One fraction of Kelly's row, reported by the result that holds it."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "share_of_optimal_growth": _NO_EDGE,
        "drawdown_probability": _NO_EDGE,
        "realized_excess_growth_rate": "a period of the sample loses "
        "everything at this leverage, so its log growth is minus infinity",
        "max_drawdown_median": _NO_DRAWS,
        "max_drawdown_worst_5pct": _NO_DRAWS,
        "share_of_paths_ruined": _NO_DRAWS,
    }

    fraction_of_kelly: Stat = None
    leverage: Dict[str, Stat] = Field(
        default_factory=dict, description="fraction x f*, per stream."
    )
    gross_leverage: Stat = None
    excess_growth_rate: Stat = Field(
        None,
        description="theta^2 (c - c^2/2): annual log growth over the rate if "
        "f* were the truth.",
    )
    excess_growth_rate_with_estimation_error: Stat = Field(
        None,
        description="Its expectation holding c f-hat: minus c^2 N / (2T).",
    )
    share_of_optimal_growth: Stat = Field(
        None, description="The line above over theta^2 / 2."
    )
    realized_excess_growth_rate: Stat = Field(
        None, description="In-sample: this sample's log growth at the leverage."
    )
    drawdown_probability: Dict[str, Stat] = Field(
        default_factory=dict,
        description="Loss d -> P(ever losing d of wealth) = (1-d)^(2/c-1) "
        "(Browne 1999; Thorp 2006), f* taken as the truth.",
    )
    max_drawdown_in_sample: Stat = Field(
        None, description="<= 0, as max_drawdown gives it, on the sample's path."
    )
    max_drawdown_median: Stat = Field(
        None, description="Over the bootstrap paths, each as long as the sample."
    )
    max_drawdown_worst_5pct: Stat = Field(
        None, description="The 5% quantile: 95% of paths draw down less."
    )
    share_of_paths_ruined: Stat = Field(
        None, description="Bootstrap paths with a period losing everything."
    )


class KellyReportResult(_Result):
    null_reasons = {
        "empirical_kelly_excess_growth_rate": "empirical_kelly has no maximum",
    }

    stream_names: List[str] = Field(default_factory=list)
    n_streams: int = 0
    n_periods: int = 0
    periods_per_year: int = 0
    periods_per_year_source: str = ""
    years: Stat = Field(None, description="T = n_periods / periods_per_year.")
    risk_free_rate: Stat = None
    confidence: Stat = None
    by_stream: Dict[str, KellyStream] = Field(default_factory=dict)
    sharpe_ratio: Stat = Field(
        None,
        description="In-sample theta = sqrt(mu' Sigma^-1 mu): the Kelly "
        "portfolio's Sharpe ratio.",
    )
    optimal_excess_growth_rate: Stat = Field(None, description="theta^2 / 2.")
    gross_kelly_leverage: Stat = None
    net_kelly_leverage: Stat = None
    p_estimate_over_twice_truth: Stat = Field(
        None,
        description="For the Kelly portfolio's scale: Phi(-theta sqrt(T)), "
        "the chance its growth is no better than cash.",
    )
    shrink_factor: Stat = Field(
        None,
        description="theta^2 T / (theta^2 T + N), Kan & Zhou (2007)'s plug-in: "
        "the share of f-hat that maximises expected growth.",
    )
    condition_number: Stat = None
    empirical_kelly: Optional[Dict[str, Stat]] = Field(
        None,
        description="In-sample: the leverages that grew this sample the most. "
        "Null when that has no maximum.",
    )
    empirical_kelly_excess_growth_rate: Stat = None
    by_fraction: List[KellyFraction] = Field(default_factory=list)
    n_bootstrap: int = 0
    block_size: int = 0
    random_seed: int = 0
    streams: Optional[StreamAlignment] = Field(
        None, description="The dates the streams share and what each lost."
    )


class RiskOverlayResult(_Result):
    null_reasons = {"worst_breach": "wealth never fell below the floor"}

    method: str = ""
    stream: str = ""
    n_periods: int = 0
    warmup_periods: int = Field(0, description="Returns before the overlay starts.")
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    periods_per_year: int = 0
    periods_per_year_source: str = ""
    target_volatility: Stat = Field(None, description="vol_target only.")
    realized_volatility: Stat = Field(None, description="Of the overlaid stream.")
    stream_volatility: Stat = Field(None, description="Of the stream, same dates.")
    realized_over_target: Stat = Field(None, description="vol_target only.")
    annualized_return_before: Stat = None
    annualized_return_after: Stat = None
    max_drawdown_before: Stat = None
    max_drawdown_after: Stat = None
    exposure_mean: Stat = None
    exposure_min: Stat = None
    exposure_max: Stat = None
    exposure_final: Stat = None
    share_at_max_leverage: Stat = None
    turnover: Stat = Field(
        None, description="sum |change in exposure|, the first entry included."
    )
    turnover_per_year: Stat = None
    cost_total: Stat = Field(None, description="Sum of each period's cost, a return.")
    cost_per_year: Stat = None
    floor_breaches: Optional[int] = Field(
        None, description="Times wealth crossed below the floor. Null: vol_target."
    )
    worst_breach: Stat = Field(None, description="min (V - F) / F below the floor.")
    periods_below_floor: Optional[int] = None
    cash_locked: Optional[bool] = Field(
        None,
        description="The cushion was gone and the exposure zero to the end. "
        "Null: vol_target.",
    )
    cash_locked_from: Optional[str] = None
    final_floor: Stat = Field(None, description="cppi and tipp only.")
    overlaid_returns_ref: Optional[str] = Field(
        None, description="The overlaid stream, a returns_panel. Null without run_id."
    )
    exposure_ref: Optional[str] = Field(
        None, description="The exposure path, an analytic_series. Null without run_id."
    )
    streams: Optional[StreamAlignment] = Field(
        None, description="With returns_ref: the column's dates and what it lost."
    )


class CapacityPoint(ExplainsNulls):
    """One AUM of the curve, reported by the result that holds it."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {"net_sharpe": "the net returns did not vary"}

    aum: Stat = None
    net_return: Stat = Field(None, description="Annual mean, after costs.")
    net_sharpe: Stat = None
    cost_per_year: Stat = None
    impact_cost_per_year: Stat = None
    max_participation: Stat = Field(
        None, description="The largest single trade's share of its ADV."
    )


class CapacityInstrument(ExplainsNulls):
    """One instrument's row, reported by the result that holds it."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "impact_share": "the strategy never trades, so there is no impact",
        "cost_share": "the strategy never trades, so there is no cost",
    }

    instrument: str = ""
    impact_share: Stat = Field(None, description="Of all impact cost, at any AUM.")
    cost_share: Stat = Field(None, description="Of all cost at reference_aum.")
    turnover_per_year: Stat = None
    max_participation: Stat = Field(None, description="At reference_aum.")


class StrategyCapacityResult(_Result):
    null_reasons = {
        "breakeven_aum": "no impact grows with AUM: nothing is traded, or "
        "every traded instrument has zero volatility",
        "half_sharpe_aum": "the gross Sharpe ratio is not positive, or no "
        "impact grows with AUM, so no AUM halves it",
        "gross_sharpe": "the gross returns did not vary",
    }

    stream: str = ""
    n_periods: int = 0
    n_instruments: int = 0
    n_weight_rows: int = 0
    periods_per_year: int = 0
    periods_per_year_source: str = ""
    impact_coefficient: Stat = None
    gross_return: Stat = Field(None, description="Annual mean, before costs.")
    gross_sharpe: Stat = None
    spread_cost_per_year: Stat = None
    turnover_per_year: Stat = Field(None, description="sum |change in weight| a year.")
    breakeven_aum: Stat = Field(
        None, description="Where mean net return is zero: ((m - a) / b)^2."
    )
    half_sharpe_aum: Stat = Field(None, description="Where net Sharpe is half gross.")
    reference_aum: Stat = Field(
        None, description="breakeven_aum, else the largest AUM of the curve."
    )
    curve: List[CapacityPoint] = Field(default_factory=list)
    binding: List[CapacityInstrument] = Field(
        default_factory=list, description="By share of impact cost, largest first."
    )
    streams: Optional[StreamAlignment] = Field(
        None, description="With returns_ref: the column's dates and what it lost."
    )


# ── stream resolution ───────────────────────────────────────────────────


def _one_stream(
    input_data: _OneStream, who: str, min_overlap: int
) -> Tuple[pd.Series, Optional[StreamAlignment], List[str]]:
    """The stream as a dated Series, checked as a return stream."""
    if input_data.returns_ref is not None:
        panel = returns_ref_streams(
            input_data.returns_ref,
            what=who,
            columns=None if input_data.column is None else [input_data.column],
        )
        if len(panel) != 1:
            raise ValidationError(
                f"{who}: returns_ref has {len(panel)} columns ({list(panel)[:12]}) "
                "and this tool reads ONE stream. Name it in column."
            )
        frame, alignment, notes = aligned_streams(
            panel, what=who, min_overlap=min_overlap
        )
        return frame.iloc[:, 0], alignment, notes
    assert input_data.returns is not None  # one of the two, by the model
    frame, _alignment, notes = aligned_streams(
        {"returns": dated_series(input_data.returns, "returns", who)},
        what=who,
        min_overlap=min_overlap,
    )
    return frame.iloc[:, 0], None, notes


def _periods(index: pd.Index, given: Optional[int], who: str) -> Tuple[int, str, List]:
    from standard_quant_tools.metrics.annualization import resolve_periods_per_year

    return resolve_periods_per_year(index, periods_per_year=given, where=who)


def _iso(stamp: Any) -> Optional[str]:
    if stamp is None:
        return None
    stamp = pd.Timestamp(stamp)
    return str(stamp.date()) if stamp == stamp.normalize() else stamp.isoformat()


# ── tools ───────────────────────────────────────────────────────────────


def get_kelly_report(input_data: KellyReportInput) -> KellyReportResult:
    who = "get_kelly_report"
    if input_data.returns_ref is not None:
        frame, alignment, notes = aligned_returns_ref(
            input_data.returns_ref,
            what=who,
            min_overlap=kelly_lib.MIN_PERIODS,
            columns=input_data.columns,
        )
    else:
        assert input_data.returns is not None  # one of the two, by the model
        frame, alignment, notes = aligned_dated_streams(
            input_data.returns, what=who, min_overlap=kelly_lib.MIN_PERIODS
        )
    periods, source, period_notes = _periods(
        frame.index, input_data.periods_per_year, who
    )
    raw = kelly_lib.kelly_report(
        frame,
        periods_per_year=periods,
        risk_free_rate=input_data.risk_free_rate,
        fractions=input_data.fractions,
        drawdown_levels=input_data.drawdown_levels,
        confidence=input_data.confidence,
        n_bootstrap=input_data.n_bootstrap,
        block_size=input_data.block_size,
        random_seed=input_data.random_seed,
    )
    raw["warnings"] = notes + period_notes + raw["warnings"]
    raw["by_stream"] = {k: KellyStream(**v) for k, v in raw["by_stream"].items()}
    raw["by_fraction"] = [KellyFraction(**row) for row in raw["by_fraction"]]
    return KellyReportResult(
        **raw,
        periods_per_year_source=source,
        streams=alignment,
    )


def apply_risk_overlay(input_data: RiskOverlayInput) -> RiskOverlayResult:
    who = "apply_risk_overlay"
    series, alignment, notes = _one_stream(input_data, who, 2)
    periods, source, period_notes = _periods(
        series.index, input_data.periods_per_year, who
    )
    read = {
        name: getattr(input_data, name) for name in _OVERLAY_FIELDS[input_data.method]
    }
    raw = overlay_lib.apply_overlay(
        series,
        method=input_data.method,
        periods_per_year=periods,
        max_leverage=input_data.max_leverage,
        cash_rate=input_data.cash_rate,
        cost_bps=input_data.cost_bps,
        **read,
    )
    overlaid = raw.pop("overlaid")
    exposure = raw.pop("exposure")
    refs: Dict[str, Optional[str]] = {
        "overlaid_returns_ref": None,
        "exposure_ref": None,
    }
    if input_data.run_id is not None:
        from standard_quant_tools.agent.runtimes import handoff

        refs["overlaid_returns_ref"] = handoff.publish(
            overlaid.to_frame(),
            kind="returns_panel",
            run_id=input_data.run_id,
            name="overlaid_returns",
            producer=who,
        )
        refs["exposure_ref"] = handoff.publish(
            exposure,
            kind="analytic_series",
            run_id=input_data.run_id,
            name="exposure",
            producer=who,
        )
    raw["warnings"] = notes + period_notes + raw["warnings"]
    for key in ("first_date", "last_date", "cash_locked_from"):
        raw[key] = _iso(raw[key])
    # A figure the method does not have is left unset, and its description
    # says which method has it; a null among the rest says why it is null.
    raw = {key: value for key, value in raw.items() if value is not None}
    return RiskOverlayResult(
        **raw,
        **refs,
        stream=str(series.name),
        periods_per_year_source=source,
        streams=alignment,
    )


def _weight_frame(input_data: StrategyCapacityInput, who: str) -> pd.DataFrame:
    """The weights as a date-by-instrument frame, from inline or the ref."""
    mapping = input_data.weights
    if input_data.weights_ref is not None:
        from standard_quant_tools.agent.runtimes import handoff

        try:
            mapping = handoff.resolve(input_data.weights_ref, expect="weight_panel")
        except Exception as exc:  # noqa: BLE001 -- one refusal, by name
            raise ValidationError(
                f"{who}: weights_ref could not be resolved as a weight_panel "
                f"-- {exc} construct_weights_from_scores publishes one."
            ) from exc
    assert mapping is not None  # one of the two, by the model
    columns = {
        str(name): dated_series(
            {str(k): float(v) for k, v in row.items()}, f"weights[{name!r}]", who
        )
        for name, row in mapping.items()
    }
    return pd.DataFrame(columns).sort_index()


def get_strategy_capacity_curve(
    input_data: StrategyCapacityInput,
) -> StrategyCapacityResult:
    who = "get_strategy_capacity_curve"
    series, alignment, notes = _one_stream(input_data, who, 3)
    periods, source, period_notes = _periods(
        series.index, input_data.periods_per_year, who
    )
    raw = capacity_lib.capacity_curve(
        series,
        _weight_frame(input_data, who),
        adv=input_data.adv,
        daily_volatility=input_data.daily_volatility,
        spread_bps=input_data.spread_bps,
        periods_per_year=periods,
        impact_coefficient=input_data.impact_coefficient,
        aum=input_data.aum,
        risk_free_rate=input_data.risk_free_rate,
    )
    raw["warnings"] = notes + period_notes + raw["warnings"]
    raw["curve"] = [CapacityPoint(**p) for p in raw["curve"]]
    raw["binding"] = [CapacityInstrument(**b) for b in raw["binding"]]
    return StrategyCapacityResult(
        **raw,
        stream=str(series.name),
        periods_per_year_source=source,
        streams=alignment,
    )


TOOL_DEFS = [
    (
        "get_kelly_report",
        "The Kelly leverage of one return stream (mu / sigma^2) or several "
        "held together (Sigma^-1 mu), its standard error and the chance the "
        "estimate is over twice the truth. Also the growth at fractions of "
        "Kelly with and without estimation error, the Kan-Zhou shrunk "
        "fraction, the in-sample empirical Kelly, the odds of ever losing a "
        "given share of wealth, and bootstrap maximum drawdowns. In-sample "
        "throughout; it reports a leverage and applies none.",
        KellyReportInput,
    ),
    (
        "apply_risk_overlay",
        "Runs a volatility-target (lagged EWMA), CPPI or TIPP overlay over one "
        "return stream: realized against target volatility, drawdown before "
        "and after, floor breaches and turnover. Exposure at t uses returns "
        "through t-1 only; the unexposed part earns cash_rate. A run_id "
        "publishes the overlaid stream (returns_panel) and the exposure path "
        "(analytic_series).",
        RiskOverlayInput,
    ),
    (
        "get_strategy_capacity_curve",
        "Net return and Sharpe ratio against AUM for a strategy's gross return "
        "stream and weight panel under spread plus square-root impact, with "
        "its break-even and half-Sharpe AUM. ADV, daily volatility and spread "
        "are the caller's per instrument; the instruments whose impact binds "
        "are listed.",
        StrategyCapacityInput,
    ),
]

TOOL_DISPATCH = {
    "get_kelly_report": (get_kelly_report, KellyReportInput),
    "apply_risk_overlay": (apply_risk_overlay, RiskOverlayInput),
    "get_strategy_capacity_curve": (
        get_strategy_capacity_curve,
        StrategyCapacityInput,
    ),
}

__all__ = [
    "KellyReportInput",
    "KellyReportResult",
    "RiskOverlayInput",
    "RiskOverlayResult",
    "StrategyCapacityInput",
    "StrategyCapacityResult",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "apply_risk_overlay",
    "get_kelly_report",
    "get_strategy_capacity_curve",
]
