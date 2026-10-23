"""
The time value of money as three tools: what a stream of cash flows is
worth and returns, what capital costs, and what a forecast is worth.

- `analyze_cash_flows` -- NPV, every real IRR, MIRR, duration, convexity
  and DV01 of periodic or dated flows (`finance.cash_flows`).
- `estimate_cost_of_capital` -- CAPM, WACC and Hamada's betas
  (`finance.cost_of_capital`).
- `value_cash_flows` -- a DCF, its reverse and its sensitivity grid
  (`finance.valuation`).

EVERY RATE IS THE CALLER'S, and so is every day count: nothing here
fetches a rate, a beta or a premium. The inputs are inline numbers and
dated maps, so the same call gives the same bits.

AN IRR IS NULL UNLESS IT IS UNIQUE, with every root listed and the reason
in `irr_status`; a reverse DCF that two rates satisfy is null the same way.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from standard_quant_tools.agent.runtimes._json_safe import (
    ExplainsNulls,
    finite_or_none,
)
from standard_quant_tools.finance import cash_flows as flows_lib
from standard_quant_tools.finance import cost_of_capital as capital_lib
from standard_quant_tools.finance import valuation as valuation_lib

Stat = Annotated[Optional[float], BeforeValidator(finite_or_none)]

#: The day counts a dated stream can be timed by: `delta_one.daycount`'s.
DayCount = Literal["ACT/365F", "ACT/360", "30/360", "ACT/ACT"]


def _finite_list(values: Optional[List[float]], field: str) -> Optional[List[float]]:
    if values is None:
        return values
    bad = [i for i, v in enumerate(values) if not math.isfinite(v)]
    if bad:
        raise ValueError(
            f"{field} is not finite at positions {bad[:5]}; give every entry "
            "as a number, 0.0 where there is none."
        )
    return values


class _Result(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")

    warnings: List[str] = Field(default_factory=list)


def _asked(out: Dict[str, Any]) -> Dict[str, Any]:
    """A library result without its None entries. The libraries give None
    for what the call did not ask for (an NPV without a rate, the other
    terminal method's multiple) and NaN for what it asked for and is not
    defined; the first stays null quietly, the second is explained."""
    return {k: v for k, v in out.items() if v is not None}


# ── analyze_cash_flows ───────────────────────────────────────────────────


class CashFlowsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flows: Optional[List[float]] = Field(
        None,
        min_length=1,
        max_length=flows_lib.MAX_FLOWS,
        description="Amounts at periods 0, 1, 2, ...; flow 0 is now, undiscounted.",
    )
    periods_per_year: int = Field(
        1, ge=1, le=365, description="Flow spacing, and how 'periodic' compounds."
    )
    dated_flows: Optional[Dict[str, float]] = Field(
        None,
        min_length=1,
        max_length=flows_lib.MAX_FLOWS,
        description="{ISO date: amount} in place of flows; time runs from the "
        "first date.",
    )
    day_count: Optional[DayCount] = Field(
        None, description="Required with dated_flows."
    )
    compounding: Optional[Literal["periodic", "annual", "continuous"]] = Field(
        None,
        description="How every rate compounds; default periodic for flows, "
        "annual (XIRR's) for dated_flows.",
    )
    rate: Optional[float] = Field(
        None,
        gt=-1.0,
        le=10.0,
        description="Discount rate or yield: npv, durations, convexity, dv01.",
    )
    finance_rate: Optional[float] = Field(
        None, gt=-1.0, le=10.0, description="MIRR: negative flows discount at it."
    )
    reinvest_rate: Optional[float] = Field(
        None, gt=-1.0, le=10.0, description="MIRR: positive flows compound at it."
    )

    @field_validator("flows")
    @classmethod
    def _finite_flows(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        return _finite_list(v, "flows")

    @field_validator("dated_flows")
    @classmethod
    def _dated(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if v is None:
            return v
        from standard_quant_tools.agent.runtimes._shared import parse_date_keys

        parse_date_keys(v, "dated_flows", "analyze_cash_flows", finite=True)
        return v

    @model_validator(mode="after")
    def _one_stream(self) -> "CashFlowsInput":
        given = [n for n in ("flows", "dated_flows") if getattr(self, n) is not None]
        if len(given) != 1:
            raise ValueError(
                "analyze_cash_flows needs exactly one of flows / dated_flows; "
                f"got {given or 'neither'}. flows are periodic amounts, "
                "dated_flows {ISO date: amount}."
            )
        if self.flows is not None and self.day_count is not None:
            raise ValueError(
                f"day_count={self.day_count!r} applies to dated_flows; periodic "
                "flows are timed by periods_per_year. Drop day_count."
            )
        if self.dated_flows is not None and self.day_count is None:
            raise ValueError(
                "dated_flows needs a day_count: the same dates are different "
                "times under ACT/360 and ACT/365F, and a rate is quoted under "
                "one. XIRR uses 'ACT/365F'."
            )
        if (self.finance_rate is None) != (self.reinvest_rate is None):
            raise ValueError(
                "finance_rate and reinvest_rate come together: MIRR discounts "
                "the negative flows at one and compounds the positive at the "
                "other. Give both (the same rate twice if they are equal)."
            )
        return self


#: Why a duration-like number is null when a rate was given.
_NO_PRICE = (
    "the present value at the rate is zero, and a duration divides by it "
    "(a bond's price is not one of its flows), or it is past a float's range",
    "the discount factors overflow a float at this rate and horizon",
)


class CashFlowsResult(_Result):
    null_reasons = {
        "npv": "the discount factors overflow a float at this rate and horizon",
        "macaulay_duration": _NO_PRICE,
        "modified_duration": _NO_PRICE,
        "convexity": _NO_PRICE,
        "dv01": _NO_PRICE,
        "irr": "the flows do not have exactly one real IRR -- irr_status is "
        "'none' when no rate zeroes the NPV, 'multiple' when several do "
        "(irr_roots), 'uncertified' past 20 sign changes",
        "irr_roots": "the root's rate is past a float's range",
        "search_range": "the bound's rate is past a float's range",
        "mirr": "MIRR needs a negative and a positive flow at different "
        "times, and compounded flows within a float's range",
    }

    n_flows: int = 0
    horizon_years: Stat = Field(None, description="Time of the last flow.")
    total: Stat = Field(None, description="The flows' undiscounted sum.")
    compounding: str = ""
    periods_per_year: int = 1
    day_count: Optional[str] = None
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    rate: Optional[float] = None
    npv: Stat = Field(None, description="PV of every flow at rate; null without one.")
    irr: Stat = Field(
        None, description="The IRR, in the compounding's terms; null unless unique."
    )
    irr_status: Literal["unique", "none", "multiple", "uncertified"] = Field(
        "none",
        description="'none': no real root; 'multiple': several (irr_roots); "
        "'uncertified': over 20 sign changes, so roots were found on a grid.",
    )
    irr_roots: List[Stat] = Field(
        default_factory=list, description="Every real IRR found, ascending."
    )
    sign_changes: int = Field(
        0,
        description="Sign changes of the flows in time order: at most that many IRRs.",
    )
    cumulative_sign_changes: int = 0
    norstrom_holds: bool = Field(
        False,
        description="Norstrom (1972): the cumulative flows change sign exactly "
        "once and end non-zero, so exactly one IRR is above 0%.",
    )
    root_search: Literal["exact", "grid", "none"] = Field(
        "none",
        description="'exact': Rolle isolation of every root; 'grid': bracketed "
        "on a grid; 'none': no sign change to search.",
    )
    search_range: Optional[List[Stat]] = Field(
        None, description="The rates every root lies between (Cauchy bounds)."
    )
    macaulay_duration: Stat = Field(
        None, description="PV-weighted mean time of the flows, in years."
    )
    modified_duration: Stat = Field(
        None, description="-(1/P) dP/dy: Macaulay / (1 + y/m)."
    )
    convexity: Stat = Field(None, description="(1/P) d2P/dy2, in years squared.")
    dv01: Stat = Field(
        None, description="Value change for 1 bp of yield: modified x P x 1e-4."
    )
    mirr: Stat = Field(None, description="Null without finance and reinvest rates.")


def analyze_cash_flows(input_data: CashFlowsInput) -> CashFlowsResult:
    if input_data.flows is not None:
        schedule = flows_lib.periodic_cash_flows(
            input_data.flows,
            periods_per_year=input_data.periods_per_year,
            compounding=input_data.compounding,
        )
    else:
        from standard_quant_tools.agent.runtimes._shared import parse_date_keys

        assert input_data.dated_flows is not None  # one of the two, by the model
        schedule = flows_lib.dated_cash_flows(
            parse_date_keys(
                input_data.dated_flows,
                "dated_flows",
                "analyze_cash_flows",
                finite=True,
            ),
            day_count=input_data.day_count,
            compounding=input_data.compounding,
            periods_per_year=input_data.periods_per_year,
        )
    return CashFlowsResult(
        **_asked(
            flows_lib.analyze_cash_flows(
                schedule,
                rate=input_data.rate,
                finance_rate=input_data.finance_rate,
                reinvest_rate=input_data.reinvest_rate,
            )
        )
    )


# ── estimate_cost_of_capital ─────────────────────────────────────────────


class CostOfCapitalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_free_rate: float = Field(..., ge=-1.0, le=1.0, description="Decimal.")
    beta: float = Field(..., ge=-20.0, le=20.0, description="Levered equity beta.")
    equity_risk_premium: float = Field(..., ge=-1.0, le=1.0, description="Decimal.")
    beta_standard_error: Optional[float] = Field(
        None, ge=0.0, le=20.0, description="For the sensitivity at beta -/+ it."
    )
    equity_value: float = Field(..., description="Market value of equity, > 0.")
    debt_value: float = Field(0.0, ge=0.0, description="Market value of debt.")
    cost_of_debt: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Pre-tax yield on debt; needed with debt."
    )
    tax_rate: float = Field(..., description="Marginal tax rate, a decimal in [0, 1).")
    target_debt_to_equity: Optional[float] = Field(
        None, ge=0.0, le=1000.0, description="D/E to relever beta and WACC at."
    )

    @field_validator("equity_value")
    @classmethod
    def _positive_equity(cls, v: float) -> float:
        if not (math.isfinite(v) and v > 0.0):
            raise ValueError(
                f"equity_value={v!r} is not a positive number. WACC weights are "
                "market values, and equity worth nothing (or less) has no cost "
                "of equity and no D/E; pass the market capitalization."
            )
        return v

    @field_validator("tax_rate")
    @classmethod
    def _tax(cls, v: float) -> float:
        if not (math.isfinite(v) and 0.0 <= v < 1.0):
            raise ValueError(
                f"tax_rate={v!r} is outside [0, 1): it is a decimal share of "
                "income, so 21% is 0.21."
            )
        return v

    @field_validator("debt_value")
    @classmethod
    def _finite_debt(cls, v: float) -> float:
        if not math.isfinite(v) or v > 1e15:
            raise ValueError(
                f"debt_value={v!r} must be a finite market value within 1e15."
            )
        return v


class CapitalAtTarget(ExplainsNulls):
    """Hamada's beta relevered at the target D/E, and what it costs;
    reported by the result that holds it."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "wacc": "the target adds debt and no cost_of_debt was given to price it"
    }

    debt_to_equity: Stat = None
    debt_weight: Stat = Field(None, description="D/E / (1 + D/E).")
    beta: Stat = Field(None, description="unlevered_beta x (1 + (1-t) D/E).")
    cost_of_equity: Stat = None
    wacc: Stat = Field(None, description="cost_of_debt held as given.")


class BetaSensitivity(ExplainsNulls):
    """The cost of equity and WACC one standard error either side of beta."""

    model_config = ConfigDict(extra="forbid")

    beta_standard_error: Stat = None
    beta_low: Stat = None
    beta_high: Stat = None
    cost_of_equity_low: Stat = None
    cost_of_equity_high: Stat = None
    wacc_low: Stat = None
    wacc_high: Stat = None


class CostOfCapitalResult(_Result):
    cost_of_equity: Stat = Field(None, description="r_f + beta x ERP.")
    after_tax_cost_of_debt: Stat = Field(
        None, description="cost_of_debt x (1 - t); null without one."
    )
    equity_weight: Stat = Field(None, description="E / (D + E).")
    debt_weight: Stat = None
    debt_to_equity: Stat = None
    wacc: Stat = Field(None, description="E/V k_e + D/V k_d (1 - t).")
    unlevered_beta: Stat = Field(None, description="Hamada: beta / (1 + (1-t) D/E).")
    unlevered_cost_of_equity: Stat = Field(
        None, description="r_f + unlevered_beta x ERP."
    )
    at_target: Optional[CapitalAtTarget] = Field(
        None, description="At target_debt_to_equity; null without one."
    )
    beta_sensitivity: Optional[BetaSensitivity] = Field(
        None, description="At beta -/+ beta_standard_error; null without one."
    )


def estimate_cost_of_capital(input_data: CostOfCapitalInput) -> CostOfCapitalResult:
    return CostOfCapitalResult(
        **_asked(capital_lib.estimate_cost_of_capital(**input_data.model_dump()))
    )


# ── value_cash_flows ─────────────────────────────────────────────────────


class ValueCashFlowsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cash_flows: List[float] = Field(
        ...,
        min_length=1,
        max_length=valuation_lib.MAX_YEARS,
        description="Forecast free cash flow for years 1..N, each at year end.",
    )
    discount_rate: Optional[float] = Field(
        None, gt=-1.0, le=10.0, description="Annual; omit when solving for it."
    )
    terminal_method: Literal["gordon", "exit_multiple"] = "gordon"
    terminal_growth: Optional[float] = Field(
        None, gt=-1.0, le=1.0, description="Gordon growth after year N, < rate."
    )
    exit_multiple: Optional[float] = Field(
        None, gt=0.0, le=1000.0, description="Year-N value = this x terminal_metric."
    )
    terminal_metric: Optional[float] = Field(
        None, description="Year-N figure the multiple applies to; default FCF_N."
    )
    target_value: Optional[float] = Field(
        None, description="Reverse DCF: the value to explain."
    )
    solve_for: Optional[
        Literal["discount_rate", "terminal_growth", "exit_multiple"]
    ] = Field(None, description="Reverse DCF: the input target_value implies.")
    sensitivity_rates: Optional[List[float]] = Field(
        None, min_length=1, max_length=valuation_lib.MAX_GRID
    )
    sensitivity_terminal: Optional[List[float]] = Field(
        None,
        min_length=1,
        max_length=valuation_lib.MAX_GRID,
        description="Grid columns, growths or multiples; default around the input.",
    )

    @field_validator("cash_flows", "sensitivity_rates", "sensitivity_terminal")
    @classmethod
    def _finite(cls, v: Optional[List[float]], info) -> Optional[List[float]]:
        return _finite_list(v, info.field_name)

    @field_validator("terminal_metric", "target_value")
    @classmethod
    def _finite_amount(cls, v: Optional[float], info) -> Optional[float]:
        if v is not None and (not math.isfinite(v) or abs(v) > 1e15):
            raise ValueError(
                f"{info.field_name}={v!r} must be a finite amount within 1e15."
            )
        return v

    @model_validator(mode="after")
    def _consistent(self) -> "ValueCashFlowsInput":
        # The library's own rules, so a call is refused here by the same
        # message rather than after it reaches the computation.
        valuation_lib.check_dcf_inputs(
            terminal_method=self.terminal_method,
            discount_rate=self.discount_rate,
            terminal_growth=self.terminal_growth,
            exit_multiple=self.exit_multiple,
            terminal_metric=self.terminal_metric,
            target_value=self.target_value,
            solve_for=self.solve_for,
        )
        return self


_NOT_SOLVED = "the reverse DCF found no single value of solve_for"


class ValuationGrid(ExplainsNulls):
    """The value at each discount rate (row) and terminal assumption
    (column); reported by the result that holds it."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {"values": "the cell's growth is not below its rate"}

    discount_rates: List[float] = Field(default_factory=list)
    terminal: List[float] = Field(default_factory=list)
    terminal_axis: str = ""
    values: List[List[Stat]] = Field(default_factory=list)


class ValueCashFlowsResult(_Result):
    null_reasons = {
        "value": _NOT_SOLVED,
        "pv_forecast": _NOT_SOLVED,
        "terminal_value": _NOT_SOLVED,
        "pv_terminal": _NOT_SOLVED,
        "terminal_share": "the value is zero, or was not solved",
        "discount_rate": _NOT_SOLVED,
        "terminal_growth": _NOT_SOLVED,
        "exit_multiple": _NOT_SOLVED,
        "implied": "no single value of solve_for reaches target_value "
        "(implied_roots lists any found)",
        "implied_exit_multiple": "the last forecast flow is zero, or the value "
        "was not solved",
        "implied_terminal_growth": "terminal_value + FCF_N is zero, or the "
        "value was not solved",
    }

    n_years: int = 0
    terminal_method: str = ""
    discount_rate: Stat = Field(None, description="The rate used (solved, if so).")
    terminal_growth: Stat = Field(None, description="Gordon only.")
    exit_multiple: Stat = Field(None, description="Exit multiple only.")
    terminal_metric: Stat = Field(None, description="Exit multiple only.")
    value: Stat = Field(None, description="PV of the forecast plus PV of the TV.")
    pv_forecast: Stat = None
    terminal_value: Stat = Field(None, description="At year N, undiscounted.")
    pv_terminal: Stat = None
    terminal_share: Stat = Field(
        None, description="pv_terminal / value: how much is the terminal assumption."
    )
    pv_by_year: List[Stat] = Field(default_factory=list)
    implied_exit_multiple: Stat = Field(
        None, description="Gordon only: terminal_value / FCF_N."
    )
    implied_terminal_growth: Stat = Field(
        None, description="Exit multiple only: the perpetual growth it implies."
    )
    solve_for: Optional[str] = None
    target_value: Stat = None
    implied: Stat = Field(
        None, description="Reverse DCF: the solved input; null unless unique."
    )
    implied_roots: List[Stat] = Field(default_factory=list)
    implied_bracket: Optional[List[Stat]] = Field(
        None, description="The grid cell Brent's method solved in."
    )
    search_range: Optional[List[Stat]] = Field(
        None, description="The whole range scanned for a root."
    )
    sensitivity: Optional[ValuationGrid] = None


def value_cash_flows(input_data: ValueCashFlowsInput) -> ValueCashFlowsResult:
    return ValueCashFlowsResult(
        **_asked(valuation_lib.dcf_valuation(**input_data.model_dump()))
    )


# ── registration ─────────────────────────────────────────────────────────

TOOL_DEFS = [
    (
        "analyze_cash_flows",
        "NPV, IRR, MIRR, Macaulay and modified duration, convexity and DV01 "
        "of periodic or dated cash flows, listing every real IRR and giving "
        "the IRR only when it is unique. Flow 0 of a list is now and is not "
        "discounted (not Excel NPV's convention); dated flows run from the "
        "first date under the day count given, and annual compounding on "
        "ACT/365F is XIRR. Norstrom's condition is reported: cumulative "
        "flows that change sign once give exactly one IRR above 0%. A "
        "bond's duration takes its coupons and principal, not its price.",
        CashFlowsInput,
    ),
    (
        "estimate_cost_of_capital",
        "CAPM cost of equity, after-tax cost of debt and WACC on market "
        "values, with Hamada's unlevered beta and the beta and WACC "
        "relevered at a target debt-to-equity. Every rate and value is the "
        "caller's. With beta's standard error the cost of equity and WACC "
        "are also given at beta -/+ one error. Hamada takes debt as riskless "
        "and its amount as fixed; the relevered WACC holds the cost of debt "
        "as given.",
        CostOfCapitalInput,
    ),
    (
        "value_cash_flows",
        "DCF value of forecast free cash flows with a Gordon or exit-multiple "
        "terminal value, the terminal value's share of it, and a value grid "
        "by discount rate and growth or multiple. With "
        "target_value it is a reverse DCF: the discount rate, terminal growth "
        "or exit multiple the target implies, by a bracketed root with the "
        "bracket reported, null when two values fit. Gordon growth at or "
        "above the rate is refused.",
        ValueCashFlowsInput,
    ),
]

TOOL_DISPATCH = {
    "analyze_cash_flows": (analyze_cash_flows, CashFlowsInput),
    "estimate_cost_of_capital": (estimate_cost_of_capital, CostOfCapitalInput),
    "value_cash_flows": (value_cash_flows, ValueCashFlowsInput),
}

__all__ = [
    "CashFlowsInput",
    "CashFlowsResult",
    "CostOfCapitalInput",
    "CostOfCapitalResult",
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "ValuationGrid",
    "ValueCashFlowsInput",
    "ValueCashFlowsResult",
    "analyze_cash_flows",
    "estimate_cost_of_capital",
    "value_cash_flows",
]
