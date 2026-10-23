"""
What a portfolio with deposits and withdrawals returned: the true
time-weighted return, linked Modified Dietz and the money-weighted return
(GIPS 2020), from dated valuations and dated external flows.

The library function is `allocation.book_returns.measure_book_returns`;
its module docstring has the formulas and the conventions. The inputs are inline maps
keyed by ISO date -- never a ticker or a fetched series -- so the same call
gives the same bits. The money-weighted return is an IRR found by
`finance.cash_flows.rate_roots`, null unless it is unique, as
`analyze_cash_flows`' IRR is.
"""

from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

from standard_quant_tools.agent.runtimes._json_safe import (
    ExplainsNulls,
    finite_or_none,
)
from standard_quant_tools.allocation import book_returns as returns_lib

Stat = Annotated[Optional[float], BeforeValidator(finite_or_none)]


class BookReturnsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valuations: Dict[str, float] = Field(
        ...,
        min_length=2,
        max_length=returns_lib.MAX_VALUATIONS,
        description="{ISO date: market value}: the start, the end, and one at "
        "each flow for a true TWR.",
    )
    flows: Optional[Dict[str, float]] = Field(
        None,
        max_length=returns_lib.MAX_FLOWS,
        description="{ISO date: external flow}: + contribution, - withdrawal.",
    )
    flows_in_valuation: bool = Field(
        False,
        description="True: a valuation on a flow's date includes the flow; "
        "False: it is taken before it.",
    )
    day_count: Literal["ACT/365F", "ACT/360", "30/360", "ACT/ACT"] = Field(
        "ACT/365F", description="Years for annualizing and the MWR's IRR."
    )
    large_flow_share: float = Field(
        0.10, gt=0.0, le=1.0, description="Flag a flow above this share of value."
    )

    @field_validator("valuations", "flows")
    @classmethod
    def _dated(cls, v: Optional[Dict[str, float]], info) -> Optional[Dict[str, float]]:
        if v is None:
            return v
        from standard_quant_tools.agent.runtimes._shared import parse_date_keys

        parse_date_keys(v, info.field_name, "measure_book_returns", finite=True)
        return v


class BookPeriod(ExplainsNulls):
    """One sub-period between two valuations; reported by its result."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "modified_dietz": "its average capital is not positive",
        "time_weighted": "a flow inside it has no valuation, or it opens "
        "with no capital",
    }

    start: str = ""
    end: str = ""
    start_value: Stat = None
    end_value: Stat = None
    net_flow: Stat = None
    modified_dietz: Stat = None
    time_weighted: Stat = None


class LargeFlow(BaseModel):
    """A flow above large_flow_share of the value before it."""

    model_config = ConfigDict(extra="forbid")

    date: str
    amount: float
    share_of_value: Stat = Field(None, description="Null when the value was 0.")
    valued: bool = Field(..., description="Whether its date has a valuation.")


_CUMULATIVE_NULL = "the return over the whole period is null"


class BookReturnsResult(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "time_weighted_return": "a flow date has no valuation -- "
        "missing_valuation_dates lists them; value the portfolio there, or "
        "read modified_dietz_return, which approximates it -- or a "
        "sub-period opens with no capital",
        "time_weighted_annualized": _CUMULATIVE_NULL,
        "modified_dietz_return": "a sub-period's average capital (opening "
        "value plus day-weighted flows) is not positive",
        "modified_dietz_annualized": _CUMULATIVE_NULL,
        "money_weighted_return": "the flows have no single IRR -- mwr_status is "
        "'none' when no rate equates them, 'multiple' when several do "
        "(mwr_roots), 'uncertified' past 20 sign changes",
        "money_weighted_annualized": "the money-weighted return is null",
        "mwr_roots": "the root's return is past a float's range",
        "share_of_value": "the value before the flow was zero",
    }

    start_date: str = ""
    end_date: str = ""
    years: Stat = Field(None, description="The period, in day_count years.")
    day_count: str = ""
    flows_in_valuation: bool = False
    n_valuations: int = 0
    n_flows: int = 0
    n_periods: int = 0
    start_value: Stat = None
    end_value: Stat = None
    net_flows: Stat = Field(None, description="Sum of the external flows.")
    time_weighted_return: Stat = Field(
        None, description="True TWR over the period: sub-period returns linked."
    )
    time_weighted_annualized: Stat = Field(None, description="Null under a year.")
    modified_dietz_return: Stat = Field(
        None, description="Modified Dietz per sub-period, linked geometrically."
    )
    modified_dietz_annualized: Stat = Field(None, description="Null under a year.")
    money_weighted_return: Stat = Field(
        None, description="(1 + IRR)^years - 1: the owner's return over the period."
    )
    money_weighted_annualized: Stat = Field(
        None, description="The IRR as an annual rate; null under a year."
    )
    mwr_status: Literal["unique", "none", "multiple", "uncertified"] = "none"
    mwr_roots: List[Stat] = Field(
        default_factory=list, description="Every root, as a return over the period."
    )
    timing_gap: Stat = Field(
        None,
        description="money_weighted_return - time_weighted_return: what the "
        "flows' timing added or cost the owner; null without both.",
    )
    missing_valuation_dates: List[str] = Field(default_factory=list)
    n_missing_valuation_dates: int = 0
    large_flows: List[LargeFlow] = Field(default_factory=list)
    periods: List[BookPeriod] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def measure_book_returns(input_data: BookReturnsInput) -> BookReturnsResult:
    from standard_quant_tools.agent.runtimes._shared import parse_date_keys

    tool = "measure_book_returns"
    valuations = parse_date_keys(input_data.valuations, "valuations", tool, finite=True)
    flows = (
        None
        if input_data.flows is None
        else parse_date_keys(input_data.flows, "flows", tool, finite=True)
    )
    out = returns_lib.measure_book_returns(
        valuations,
        flows,
        day_count=input_data.day_count,
        flows_in_valuation=input_data.flows_in_valuation,
        large_flow_share=input_data.large_flow_share,
    )
    # None is "does not apply here" (an annualized figure under a year) and
    # stays null without a line of its own; NaN is undefined and explained.
    return BookReturnsResult(**{k: v for k, v in out.items() if v is not None})


TOOL_DEFS = [
    (
        "measure_book_returns",
        "Time-weighted, linked Modified Dietz and money-weighted returns of "
        "a portfolio with deposits and withdrawals (GIPS 2020), from dated "
        "valuations and dated external flows. The true TWR needs a valuation "
        "on every flow date and is null, naming the dates, without one; the "
        "MWR is the IRR of the flows and the end value, null unless unique. "
        "Nothing under a year is annualized, and flows above a share of "
        "value are flagged.",
        BookReturnsInput,
    ),
]

TOOL_DISPATCH = {
    "measure_book_returns": (measure_book_returns, BookReturnsInput),
}

__all__ = [
    "BookPeriod",
    "BookReturnsInput",
    "BookReturnsResult",
    "LargeFlow",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "measure_book_returns",
]
