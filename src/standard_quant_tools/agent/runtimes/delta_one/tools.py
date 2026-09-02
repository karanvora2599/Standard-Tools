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

import pandas as pd

from standard_quant_tools.delta_one import basis as _basis
from standard_quant_tools.delta_one import baskets as _baskets
from standard_quant_tools.delta_one import carry as _carry
from standard_quant_tools.delta_one import dividends as _dividends
from standard_quant_tools.delta_one import etf as _etf
from standard_quant_tools.delta_one import expressions as _expressions
from standard_quant_tools.delta_one import futures as _futures
from standard_quant_tools.delta_one import hedging as _hedging
from standard_quant_tools.delta_one import rebalance as _rebalance
from standard_quant_tools.delta_one import replication as _replication
from standard_quant_tools.delta_one import streaming as _streaming
from standard_quant_tools.delta_one import swaps as _swaps

from .models import (
    BasisDislocationInput,
    BasisHistoryInput,
    CashFuturesBasisInput,
    CompareExpressionsInput,
    DividendPointsInput,
    EtfFairValueInput,
    FuturesCurveInput,
    FuturesHedgeInput,
    HedgeEffectivenessInput,
    IndexBasketInput,
    IndexRebalanceInput,
    ReplicationBasketInput,
    RollAnalysisInput,
    SolveForwardCarryInput,
    SpreadMonitorInput,
    TotalReturnFutureInput,
    TotalReturnSwapInput,
)

# Imported at MODULE level, never under TYPE_CHECKING: mcp/catalog.py calls
# typing.get_type_hints() and `from __future__ import annotations` makes
# these strings that must resolve, or the tool loses its output schema.
from .results import (
    BasisDislocationResult,
    BasisHistoryResult,
    CashFuturesBasisResult,
    CompareExpressionsResult,
    DividendPointsResult,
    EtfFairValueResult,
    FuturesCurveResult,
    FuturesHedgeResult,
    HedgeEffectivenessResult,
    IndexBasketResult,
    IndexRebalanceResult,
    ReplicationBasketResult,
    RollAnalysisResult,
    SolveForwardCarryResult,
    SpreadMonitorResult,
    TotalReturnFutureResult,
    TotalReturnSwapResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "analyze_basis_history",
    "monitor_spread_stream",
    "detect_basis_dislocation",
    "analyze_cash_futures_basis",
    "analyze_dividend_points",
    "analyze_etf_fair_value",
    "analyze_futures_curve",
    "analyze_hedge_effectiveness",
    "analyze_index_basket",
    "analyze_index_rebalance",
    "analyze_roll",
    "analyze_total_return_future",
    "compare_delta_one_expressions",
    "optimize_replication_basket",
    "price_total_return_swap",
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
            days_between_expiries=input_data.days_between_expiries,
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


def optimize_replication_basket(
    input_data: ReplicationBasketInput,
) -> ReplicationBasketResult:
    return ReplicationBasketResult(
        **_replication.optimize_replication_basket(
            returns=pd.DataFrame(input_data.returns),
            benchmark_returns=pd.Series(input_data.benchmark_returns),
            max_names=input_data.max_names,
            long_only=input_data.long_only,
            max_weight=input_data.max_weight,
            weight_caps=input_data.weight_caps,
            covariance_method=input_data.covariance_method,
            periods_per_year=input_data.periods_per_year,
        )
    )


def analyze_etf_fair_value(input_data: EtfFairValueInput) -> EtfFairValueResult:
    return EtfFairValueResult(
        **_etf.etf_fair_value(
            etf_price=input_data.etf_price,
            nav=input_data.nav,
            nav_is_intraday=input_data.nav_is_intraday,
            basket_value=input_data.basket_value,
            cash_component=input_data.cash_component,
            creation_unit_shares=input_data.creation_unit_shares,
            creation_fee=input_data.creation_fee,
            etf_spread_bps=input_data.etf_spread_bps,
            basket_spread_bps=input_data.basket_spread_bps,
            tolerance_bps=input_data.tolerance_bps,
        )
    )


def price_total_return_swap(
    input_data: TotalReturnSwapInput,
) -> TotalReturnSwapResult:
    return TotalReturnSwapResult(
        **_swaps.price_total_return_swap(
            notional=input_data.notional,
            initial_price=input_data.initial_price,
            current_price=input_data.current_price,
            financing_rate=input_data.financing_rate,
            spread_bps=input_data.spread_bps,
            dividends=input_data.dividends,
            start_date=input_data.start_date,
            valuation_date=input_data.valuation_date,
            time_elapsed=input_data.time_elapsed,
            day_count=input_data.day_count,
            direction=input_data.direction,
        )
    )


def analyze_total_return_future(
    input_data: TotalReturnFutureInput,
) -> TotalReturnFutureResult:
    return TotalReturnFutureResult(
        **_swaps.total_return_future(
            quote=input_data.quote,
            quote_convention=input_data.quote_convention,
            underlying_price=input_data.underlying_price,
            time_to_expiry=input_data.time_to_expiry,
            reference_rate=input_data.reference_rate,
            dividend_yield=input_data.dividend_yield,
            comparison_spread_bps=input_data.comparison_spread_bps,
        )
    )


def analyze_dividend_points(
    input_data: DividendPointsInput,
) -> DividendPointsResult:
    constituents = [c.model_dump() for c in input_data.constituents]
    return DividendPointsResult(
        **_dividends.dividend_points(
            constituents,
            divisor=input_data.divisor,
            as_of=input_data.as_of,
            expiry=input_data.expiry,
            spot=input_data.spot,
            future_price=input_data.future_price,
            financing_rate=input_data.financing_rate,
            time_to_expiry=input_data.time_to_expiry,
        )
    )


def analyze_index_rebalance(
    input_data: IndexRebalanceInput,
) -> IndexRebalanceResult:
    return IndexRebalanceResult(
        **_rebalance.index_rebalance_flow(
            old_weights=input_data.old_weights,
            new_weights=input_data.new_weights,
            indexed_assets=input_data.indexed_assets,
            adv=input_data.adv,
            auction_fraction=input_data.auction_fraction,
        )
    )


def detect_basis_dislocation(
    input_data: BasisDislocationInput,
) -> BasisDislocationResult:
    return BasisDislocationResult(
        **_basis.detect_basis_dislocation(
            spot=input_data.spot_prices,
            futures=input_data.futures_prices,
            time_to_expiry=input_data.time_to_expiry,
            reference_fraction=input_data.reference_fraction,
            threshold=input_data.threshold,
            slack=input_data.slack,
            max_breaks=input_data.max_breaks,
        )
    )


def monitor_spread_stream(input_data: SpreadMonitorInput) -> SpreadMonitorResult:
    state = input_data.state
    if state is None:
        state = _streaming.new_spread_monitor(
            channel=input_data.channel,
            label=input_data.label,
            warmup=input_data.warmup,
            threshold=input_data.threshold,
            slack=input_data.slack,
        )
    elif input_data.reset:
        state = _streaming.reset_spread_monitor(
            state, keep_baseline=input_data.keep_baseline_on_reset
        )
    return SpreadMonitorResult(
        **_streaming.update_spread_monitor(
            state,
            primary=input_data.primary_prices,
            reference=input_data.reference_prices,
            time_to_expiry=input_data.time_to_expiry,
        )
    )
