"""
What a set of positions costs to finance, the margin it needs, and how
long a stress path leaves before the first margin call.

- `estimate_financing_cost` -- the account's financing terms (the
  `FinancingSpec` every engine reads) as a calculator on positions over a
  horizon: debit interest on borrowed cash, credit on idle cash, each
  short's borrow, the rebate on segregated proceeds, the net cost, the
  cash drag and the effective rate on gross exposure and on equity.
- `compute_margin_requirement` -- Reg T initial and FINRA 4210 maintenance
  margin per position and for the account, each line naming its rule.
- `analyze_margin_runway` -- a stress path walked day by day: the first
  maintenance call and its amount, the day forced deleveraging starts if
  calls go unmet, and the cash buffer that avoids every call.

POSITIONS, NEVER TICKERS: the lines, prices, FX rates and every rate are
the caller's, so nothing is fetched and the same input gives the same
bits. The library functions are `finance.financing_cost.financing_cost`,
`finance.margin.margin_requirement` and `finance.margin.margin_runway`.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from standard_quant_tools.agent.runtimes._financing import financing_spec
from standard_quant_tools.agent.runtimes._shared import parse_iso_date
from standard_quant_tools.finance.financing_cost import financing_cost
from standard_quant_tools.finance.margin import margin_requirement, margin_runway

from .margin_models import (
    FinancingCostInput,
    FinancingCostResult,
    MarginRequirementInput,
    MarginRequirementResult,
    MarginRunwayInput,
    MarginRunwayResult,
)


def _fields(record: Any) -> Dict[str, Any]:
    """A library result as a mapping, its notes renamed to warnings."""
    data = asdict(record)
    data["warnings"] = list(data.pop("notes", ()))
    return data


def estimate_financing_cost(input_data: FinancingCostInput) -> FinancingCostResult:
    tool = "estimate_financing_cost"
    start = end = None
    if input_data.start_date is not None:
        start = parse_iso_date(input_data.start_date, "start_date", tool)
    if input_data.end_date is not None:
        end = parse_iso_date(input_data.end_date, "end_date", tool)
    result = financing_cost(
        input_data.positions,
        equity=input_data.equity,
        financing=financing_spec(input_data.financing, tool),
        base_currency=input_data.base_currency,
        fx_rates=input_data.fx_rates,
        days=input_data.days,
        start=start,
        end=end,
    )
    return FinancingCostResult(**_fields(result))


def compute_margin_requirement(
    input_data: MarginRequirementInput,
) -> MarginRequirementResult:
    result = margin_requirement(
        input_data.positions,
        input_data.base_currency,
        input_data.fx_rates,
        input_data.house_rates,
    )
    data = _fields(result)
    equity = input_data.equity
    data["equity"] = equity
    data["excess_over_initial"] = None if equity is None else equity - result.initial
    data["excess_over_maintenance"] = (
        None if equity is None else equity - result.maintenance
    )
    return MarginRequirementResult(**data)


def analyze_margin_runway(input_data: MarginRunwayInput) -> MarginRunwayResult:
    holdings = input_data.holdings
    result = margin_runway(
        input_data.equity,
        equity_changes=input_data.equity_changes,
        initial_margin=input_data.initial_margin,
        maintenance_margin=input_data.maintenance_margin,
        positions=None if holdings is None else holdings.positions,
        base_currency="USD" if holdings is None else holdings.base_currency,
        fx_rates=None if holdings is None else holdings.fx_rates,
        house_rates=None if holdings is None else holdings.house_rates,
        returns=input_data.returns,
        restore_to=input_data.restore_to,
        cure_days=input_data.cure_days,
    )
    return MarginRunwayResult(**_fields(result))


TOOL_DEFS = [
    (
        "estimate_financing_cost",
        "What a set of positions costs to finance over a horizon: debit and "
        "credit interest, short borrow, the rebate, the net cost, and its "
        "rate on gross exposure and on equity. Simple interest on today's "
        "balances; cash is equity less the lines' net, proceeds segregated or "
        "not. Leave futures and swaps out.",
        FinancingCostInput,
    ),
    (
        "compute_margin_requirement",
        "Reg T initial and FINRA 4210 maintenance margin per position and for "
        "the account, each line naming its rule: stocks, leveraged ETFs and "
        "uncovered equity options. Equity in USD; no offsets for hedges or "
        "spreads; a house rate below the rule's is refused.",
        MarginRequirementInput,
    ),
    (
        "analyze_margin_runway",
        "Walks a stress path you supply: the first maintenance call and its "
        "amount, the day forced deleveraging starts if calls go unmet, and "
        "the least cash today that avoids every call. The path is daily "
        "equity changes, or daily returns that revalue holdings.",
        MarginRunwayInput,
    ),
]

TOOL_DISPATCH = {
    "estimate_financing_cost": (estimate_financing_cost, FinancingCostInput),
    "compute_margin_requirement": (
        compute_margin_requirement,
        MarginRequirementInput,
    ),
    "analyze_margin_runway": (analyze_margin_runway, MarginRunwayInput),
}

__all__ = [
    "TOOL_DEFS",
    "TOOL_DISPATCH",
    "analyze_margin_runway",
    "compute_margin_requirement",
    "estimate_financing_cost",
]
