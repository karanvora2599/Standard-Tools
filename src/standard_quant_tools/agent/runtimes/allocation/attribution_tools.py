"""
Where active return came from: `attribute_returns_brinson`, Brinson-Fachler
allocation, selection and interaction per segment and period, linked over
the periods by Carino or Menchero so they add up to the cumulative active
return (`allocation.attribution`).

Weights and returns are inline maps, {period: {segment: value}}, with the
periods as ISO dates (each period's end) and the segments as labels --
sectors, regions, sleeves, anything both sides are split by. Nothing is
fetched, and no benchmark is looked up: its weights and returns are the
caller's.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from standard_quant_tools.agent.runtimes._json_safe import ExplainsNulls
from standard_quant_tools.allocation import attribution as attribution_lib

from .leverage_tools import Stat, _Result

#: The most periods and segments one call takes, held in the validator.
MAX_PERIODS = 2_000
MAX_SEGMENTS = 500

Panel = Dict[str, Dict[str, float]]


class BrinsonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio_weights: Panel = Field(
        ...,
        description="{ISO period-end date: {segment: weight}}; each period's "
        "weights sum to 1.",
    )
    benchmark_weights: Panel = Field(
        ..., description="The benchmark's, same shape; each period sums to 1."
    )
    portfolio_returns: Panel = Field(
        ...,
        description="{period: {segment: return}}, decimals; needed where the "
        "portfolio holds the segment.",
    )
    benchmark_returns: Panel = Field(
        ..., description="The benchmark's, for every segment either side holds."
    )
    linking: Literal["carino", "menchero"] = Field(
        "carino", description="How periods compound into the cumulative effects."
    )

    @field_validator(
        "portfolio_weights",
        "benchmark_weights",
        "portfolio_returns",
        "benchmark_returns",
    )
    @classmethod
    def _finite(cls, v: Panel, info) -> Panel:
        field = info.field_name
        if not v:
            raise ValueError(f"{field} is empty; give at least one period.")
        if len(v) > MAX_PERIODS:
            raise ValueError(f"{field} has {len(v)} periods, more than {MAX_PERIODS}.")
        for period, row in v.items():
            if len(row) > MAX_SEGMENTS:
                raise ValueError(
                    f"{field}[{period!r}] has {len(row)} segments, more than "
                    f"{MAX_SEGMENTS}."
                )
            bad = [s for s, x in row.items() if not math.isfinite(x)]
            if bad:
                raise ValueError(
                    f"{field}[{period!r}] is not finite for {bad[:5]}; every "
                    "weight and return must be a number."
                )
        return v


class BrinsonEffects(ExplainsNulls):
    """One segment's effects, reported by the result that holds them."""

    model_config = ConfigDict(extra="forbid")

    allocation: Stat = Field(None, description="(w_p - w_b)(r_b - R_b).")
    selection: Stat = Field(None, description="w_b (r_p - r_b).")
    interaction: Stat = Field(None, description="(w_p - w_b)(r_p - r_b).")
    total: Stat = None


class BrinsonPeriod(ExplainsNulls):
    """One period, reported by the result that holds it."""

    model_config = ConfigDict(extra="forbid")

    period: str = ""
    portfolio_return: Stat = None
    benchmark_return: Stat = None
    active_return: Stat = None
    allocation: Stat = None
    selection: Stat = None
    interaction: Stat = None
    linking_coefficient: Stat = Field(
        None, description="What this period's effects are scaled by when linked."
    )
    segments: Dict[str, BrinsonEffects] = Field(default_factory=dict)


class BrinsonResult(_Result):
    linking: str = ""
    n_periods: int = 0
    n_segments: int = 0
    portfolio_return: Stat = Field(None, description="prod(1 + R_p,t) - 1.")
    benchmark_return: Stat = Field(None, description="prod(1 + R_b,t) - 1.")
    active_return: Stat = Field(
        None, description="The cumulative active return the effects sum to."
    )
    allocation: Stat = Field(None, description="Linked, over every segment.")
    selection: Stat = None
    interaction: Stat = None
    sum_of_effects: Stat = Field(None, description="active_return, to rounding.")
    by_segment: Dict[str, BrinsonEffects] = Field(
        default_factory=dict, description="Each segment's linked effects."
    )
    by_period: List[BrinsonPeriod] = Field(
        default_factory=list, description="Each period's own, unlinked, in order."
    )


def _periods_in_order(input_data: BrinsonInput, who: str) -> List[str]:
    """The periods as the caller wrote them, sorted by the date each names."""
    from standard_quant_tools.agent.runtimes._shared import parse_date_keys

    parsed = parse_date_keys(input_data.benchmark_weights, "benchmark_weights", who)
    order = sorted(zip(parsed, input_data.benchmark_weights), key=lambda p: p[0])
    for field in ("portfolio_weights", "portfolio_returns", "benchmark_returns"):
        parse_date_keys(getattr(input_data, field), field, who)
    return [label for _stamp, label in order]


def attribute_returns_brinson(input_data: BrinsonInput) -> BrinsonResult:
    who = "attribute_returns_brinson"
    raw = attribution_lib.brinson_fachler(
        _periods_in_order(input_data, who),
        input_data.portfolio_weights,
        input_data.benchmark_weights,
        input_data.portfolio_returns,
        input_data.benchmark_returns,
        linking=input_data.linking,
    )
    raw["by_segment"] = {s: BrinsonEffects(**e) for s, e in raw["by_segment"].items()}
    raw["by_period"] = [
        BrinsonPeriod(
            **{
                **row,
                "period": str(row["period"]),
                "segments": {
                    s: BrinsonEffects(**e) for s, e in row["segments"].items()
                },
            }
        )
        for row in raw["by_period"]
    ]
    return BrinsonResult(**raw)


TOOL_DEFS = [
    (
        "attribute_returns_brinson",
        "Brinson-Fachler allocation, selection and interaction per segment "
        "and period, linked by Carino or Menchero so the effects sum to the "
        "cumulative active return over a benchmark. Weights and returns are "
        "the caller's, {period: {segment: value}}; each period's weights sum "
        "to one.",
        BrinsonInput,
    ),
]

TOOL_DISPATCH = {
    "attribute_returns_brinson": (attribute_returns_brinson, BrinsonInput),
}

__all__ = [
    "BrinsonInput",
    "BrinsonResult",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "attribute_returns_brinson",
]
