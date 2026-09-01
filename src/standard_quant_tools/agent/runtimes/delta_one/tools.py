"""
The `delta_one` runtime: which instrument is the cheapest way to hold this.

Carry and basis against a quoted future, the term structure and what
rolling along it costs, the translation from a portfolio beta into a
number of contracts, whether that hedge historically worked, a basket
against its index, and a normalized comparison of every way to express the
same exposure.

NOTHING HERE FETCHES. Curves arrive as lists of contracts, baskets as
lists of constituents, financing as a number. This library has no futures
data provider, no index-constituent source and no dividend calendar, and a
tool that pretended otherwise would compute a curve that does not exist --
the same call the derivatives runtime made about option chains, with the
same side benefit that these work on a hypothetical curve.

THE FUNCTIONS ARE THIN, and deliberately so. Every number comes from
`standard_quant_tools.delta_one`; this module converts JSON-shaped
arguments into what those expect and wraps the answer in a typed result.
It computes nothing, because a second implementation is a second thing to
keep correct.

RESULTS ARE TYPED rather than passed through as dicts. The MCP server
builds its structured-output schema from the return annotation, so an
untyped return means a client receives JSON it has no schema for -- and on
this surface the schema is carrying real information, like the fact that a
basis in POINTS and a basis in BPS are not convertible without the time to
expiry.
"""

from __future__ import annotations

import logging

from standard_quant_tools.delta_one import basis as _basis
from standard_quant_tools.delta_one import baskets as _baskets
from standard_quant_tools.delta_one import carry as _carry
from standard_quant_tools.delta_one import expressions as _expressions
from standard_quant_tools.delta_one import futures as _futures
from standard_quant_tools.delta_one import hedging as _hedging

from .models import (
    BasisHistoryInput,
    CashFuturesBasisInput,
    CompareExpressionsInput,
    FuturesCurveInput,
    FuturesHedgeInput,
    HedgeEffectivenessInput,
    IndexBasketInput,
    RollAnalysisInput,
    SolveForwardCarryInput,
)

# Imported at MODULE level, never under TYPE_CHECKING: mcp/catalog.py calls
# typing.get_type_hints() and `from __future__ import annotations` makes
# these strings that must resolve, or the tool loses its output schema.
from .results import (
    BasisHistoryResult,
    CashFuturesBasisResult,
    CompareExpressionsResult,
    FuturesCurveResult,
    FuturesHedgeResult,
    HedgeEffectivenessResult,
    IndexBasketResult,
    RollAnalysisResult,
    SolveForwardCarryResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "analyze_basis_history",
    "analyze_cash_futures_basis",
    "analyze_futures_curve",
    "analyze_hedge_effectiveness",
    "analyze_index_basket",
    "analyze_roll",
    "compare_delta_one_expressions",
    "size_futures_hedge",
    "solve_forward_carry",
]


def analyze_cash_futures_basis(
    input_data: CashFuturesBasisInput,
) -> CashFuturesBasisResult:
    return CashFuturesBasisResult(
        **_basis.cash_futures_basis(
            spot=input_data.spot,
            future_price=input_data.future_price,
            time_to_expiry=input_data.time_to_expiry,
            risk_free_rate=input_data.risk_free_rate,
            dividend_yield=input_data.dividend_yield,
            borrow_rate=input_data.borrow_rate,
            tolerance_bps=input_data.tolerance_bps,
        )
    )


def solve_forward_carry(
    input_data: SolveForwardCarryInput,
) -> SolveForwardCarryResult:
    return SolveForwardCarryResult(
        **_carry.solve_carry(
            spot=input_data.spot,
            forward=input_data.forward,
            time_to_expiry=input_data.time_to_expiry,
            solve_for=input_data.solve_for,
            risk_free_rate=input_data.risk_free_rate,
            dividend_yield=input_data.dividend_yield,
            borrow_rate=input_data.borrow_rate,
        )
    )


def analyze_basis_history(input_data: BasisHistoryInput) -> BasisHistoryResult:
    return BasisHistoryResult(
        **_basis.basis_history(
            spot=input_data.spot_prices,
            futures=input_data.futures_prices,
            window=input_data.window,
            time_to_expiry=input_data.time_to_expiry,
        )
    )


def analyze_futures_curve(input_data: FuturesCurveInput) -> FuturesCurveResult:
    contracts = [c.model_dump(exclude_none=True) for c in input_data.contracts]
    return FuturesCurveResult(**_futures.futures_curve(contracts, spot=input_data.spot))


def analyze_roll(input_data: RollAnalysisInput) -> RollAnalysisResult:
    return RollAnalysisResult(
        **_futures.roll_analysis(
            front_price=input_data.front_price,
            next_price=input_data.next_price,
            contracts_held=input_data.contracts_held,
            multiplier=input_data.multiplier,
            days_to_front_expiry=input_data.days_to_front_expiry,
            next_multiplier=input_data.next_multiplier,
            cost_per_contract=input_data.cost_per_contract,
            spread_ticks=input_data.spread_ticks,
            tick_value=input_data.tick_value,
        )
    )


def size_futures_hedge(input_data: FuturesHedgeInput) -> FuturesHedgeResult:
    return FuturesHedgeResult(
        **_hedging.futures_hedge(
            portfolio_value=input_data.portfolio_value,
            portfolio_beta=input_data.portfolio_beta,
            future_price=input_data.future_price,
            multiplier=input_data.multiplier,
            future_beta=input_data.future_beta,
            objective=input_data.objective,
            existing_contracts=input_data.existing_contracts,
        )
    )


def analyze_hedge_effectiveness(
    input_data: HedgeEffectivenessInput,
) -> HedgeEffectivenessResult:
    return HedgeEffectivenessResult(
        **_hedging.hedge_effectiveness(
            portfolio_returns=input_data.portfolio_returns,
            hedge_returns=input_data.hedge_returns,
            hedge_ratio=input_data.hedge_ratio,
            window=input_data.window,
            periods_per_year=input_data.periods_per_year,
        )
    )


def analyze_index_basket(input_data: IndexBasketInput) -> IndexBasketResult:
    constituents = [c.model_dump(exclude_none=True) for c in input_data.constituents]
    return IndexBasketResult(
        **_baskets.index_basket(
            constituents,
            index_level=input_data.index_level,
            divisor=input_data.divisor,
        )
    )


def compare_delta_one_expressions(
    input_data: CompareExpressionsInput,
) -> CompareExpressionsResult:
    expressions = [e.model_dump() for e in input_data.expressions]
    return CompareExpressionsResult(
        **_expressions.compare_expressions(
            expressions,
            notional=input_data.notional,
            horizon_years=input_data.horizon_years,
            direction=input_data.direction,
        )
    )
