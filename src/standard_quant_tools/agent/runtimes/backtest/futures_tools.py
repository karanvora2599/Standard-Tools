"""
The futures backtest, kept in its own file because it is its own account.

WHY NOT IN `tools.py` WITH THE OTHERS. Every other backtest in this runtime
runs through the shared-cash engine, whose whole model is shares against a
cash balance. A futures account is a different set of books -- margin
posted rather than notional paid, profit arriving as variation margin
rather than accruing in a position's value, and equity that deliberately
does not include the contracts. Putting it beside the equity backtests
would invite the two to share helpers that assume the identity the futures
engine exists to break.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools.agent.runtimes._json_safe import (
    finite_or_none as _finite_or_none,
)
from standard_quant_tools.backtest.futures_engine import run_futures_simulation
from standard_quant_tools.backtest.futures_hedge_backtest import (
    run_futures_hedge_backtest as _run_futures_hedge_backtest,
)

logger = logging.getLogger(__name__)
Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


def _publish_state(
    data: Any,
    run_id: Optional[str],
    name: str,
) -> Optional[str]:
    """
    Publish one per-bar curve this simulation built, or return None.

    Opt-in, like every state curve on this surface: the scalars answer the
    question and the curve is the evidence behind them, so a caller who
    wants the evidence passes a run_id and a caller who does not pays
    nothing. `overwrite=True` because the NAME belongs to this library
    rather than the caller -- re-running under one run_id replaces that
    run's own curve instead of colliding with another agent's artifact.
    """
    if run_id is None or data is None or len(data) == 0:
        return None
    from standard_quant_tools.agent.runtimes import handoff

    if getattr(data, "name", None) is None:
        # The leverage curve is a quotient of two differently named series,
        # so pandas hands it no name and the stored column would be
        # "value". Named after the artifact instead, so the frame a
        # consumer resolves says what it holds.
        data = data.rename(name)
    return handoff.publish(
        data,
        "analytic_series",
        run_id,
        name,
        producer="backtest.run_futures_backtest",
        overwrite=True,
    )


__all__ = [
    "FUTURES_TOOL_CATEGORY",
    "FUTURES_TOOL_DEFS",
    "FUTURES_TOOL_DISPATCH",
    "FuturesBacktestInput",
    "FuturesBacktestResult",
    "run_futures_backtest",
    "run_futures_hedge_backtest",
]


class FuturesBacktestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prices: Dict[str, float] = Field(
        ...,
        min_length=2,
        description="ISO date to the TRADEABLE price of the contract held. "
        "Not a back-adjusted continuous series -- that is not a price, and "
        "sizing from one sizes against a level nobody could transact at. "
        "build_continuous_futures_series publishes the tradeable map for this.",
    )
    target_contracts: Dict[str, float] = Field(
        ...,
        min_length=1,
        description="ISO date to the SIGNED position wanted. Dates between "
        "targets hold the last one; before the first the account is flat.",
    )
    multiplier: float = Field(
        ..., gt=0, description="Contract point value, e.g. 50 for ES."
    )
    initial_capital: float = Field(1_000_000.0, gt=0, description="Currency.")
    initial_margin: float = Field(
        0.0,
        ge=0,
        description="Per contract, in currency. Zero models an unmargined "
        "account, which is an idealization rather than a futures account.",
    )
    maintenance_margin: Optional[float] = Field(
        None,
        ge=0,
        description="Per contract. Defaults to initial_margin. Below it the "
        "position is reduced, which is what a broker does.",
    )
    commission_per_contract: float = Field(
        0.0, ge=0, description="Per contract, per side."
    )
    slippage_points: float = Field(
        0.0, ge=0, description="Price POINTS given up per contract, per side."
    )
    collateral_rate: float = Field(
        0.0,
        ge=-1,
        le=1,
        description="Annual rate earned on cash. For a futures account most "
        "of the balance is cash, unlike an equity book.",
    )
    contract_map: Optional[Dict[str, str]] = Field(
        None,
        description="ISO date to contract code. When it changes the position "
        "rolls, paying both legs. Omitting it models NO roll, which over any "
        "horizon past one expiry omits the largest recurring cost.",
    )
    allow_fractional: bool = Field(
        False, description="Contracts are integers unless this says otherwise."
    )
    roll_day_prior_prices: Optional[Dict[str, float]] = Field(
        None,
        description="The OLD contract's close on each roll day, keyed like "
        "prices. A single series cannot carry it, so without it the roll "
        "day's variation margin is skipped (and the result says so); with "
        "it the old contract's move is booked before the roll.",
    )
    run_id: Optional[str] = Field(
        None,
        description="Identifier for the saved artifacts. When supplied, the "
        "five curves this simulation builds per bar -- cash, posted margin, "
        "contracts held, economic exposure and leverage -- are published "
        "under it and returned as the *_ref fields. Omit it and only the "
        "inline summary comes back, which cannot tell a comfortably "
        "margined quarter from one spent a tick from a call.",
    )


class FuturesBacktestResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    initial_capital: Optional[float] = None
    final_equity: Optional[float] = None
    total_return_pct: Optional[float] = None
    max_drawdown: Optional[float] = Field(
        None,
        description="Worst peak-to-trough decline as a SIGNED FRACTION at "
        "most zero (-0.20 is a 20% drawdown) -- the spelling every other "
        "drawdown on this surface uses, including the stress test's "
        "max_drawdown_pct, which despite its name is also a fraction. Read "
        "this one.",
    )
    max_drawdown_pct: Optional[float] = Field(
        None,
        description="The same number as a PERCENTAGE (-20.0 for a 20% "
        "drawdown). DEPRECATED in favour of max_drawdown: the identically "
        "named field on the stress test returns a fraction, so one name "
        "meant two things 100x apart across one boundary. Kept so existing "
        "callers do not break.",
    )
    max_leverage: Optional[float] = Field(
        None,
        description="ECONOMIC EXPOSURE over equity. Not the gross-market-value "
        "ratio the equity engine reports -- a futures book is at zero on that "
        "definition and many times its equity on this one.",
    )
    peak_exposure: Optional[float] = None
    total_variation_margin: Optional[float] = Field(
        None, description="Where a futures position's profit actually arrives."
    )
    total_commission: Optional[float] = None
    total_slippage: Optional[float] = None
    total_collateral_interest: Optional[float] = None
    n_margin_calls: int = 0
    margin_calls: List[Dict[str, Any]] = Field(default_factory=list)
    n_rolls: int = 0
    rolls: List[Dict[str, Any]] = Field(default_factory=list)
    margin_limited_fills: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Bars on which the target exceeded what the account could "
        "post initial margin for, with the requested and filled sizes.",
    )
    equity_curve: Dict[str, float] = Field(
        default_factory=dict, description="Cash plus posted margin, by date."
    )
    min_margin_cushion: Stat = Field(
        None,
        description="The narrowest the account ever got to a margin call: "
        "min over bars of (equity - maintenance required) / equity, where "
        "the requirement is |contracts| x maintenance_margin. 0.05 means "
        "that at its worst the account was 5% of its equity from a forced "
        "reduction. n_margin_calls = 0 says only that the line was never "
        "crossed; this says by how much. None when maintenance margin is "
        "zero -- an unmargined account has no line to be near, and the "
        "warnings say so.",
    )
    cash_curve_ref: Optional[str] = Field(
        None,
        description="An 'analytic_series' reference to the cash balance per "
        "bar, which for a futures account is most of the equity. Published "
        "only when run_id was given.",
    )
    margin_curve_ref: Optional[str] = Field(
        None,
        description="An 'analytic_series' reference to margin POSTED per "
        "bar. Published only when run_id was given.",
    )
    position_curve_ref: Optional[str] = Field(
        None,
        description="An 'analytic_series' reference to signed contracts held "
        "per bar -- what the account actually carried, as opposed to what "
        "target_contracts asked for on the bars margin would not allow. "
        "Published only when run_id was given.",
    )
    exposure_curve_ref: Optional[str] = Field(
        None,
        description="An 'analytic_series' reference to economic exposure per "
        "bar, in currency; peak_exposure is one point of it. Published only "
        "when run_id was given.",
    )
    leverage_curve_ref: Optional[str] = Field(
        None,
        description="An 'analytic_series' reference to exposure over equity "
        "per bar; max_leverage is one point of it. Published only when "
        "run_id was given.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="What this result knows that the numbers do not say.",
    )


def run_futures_backtest(input_data: FuturesBacktestInput) -> FuturesBacktestResult:
    out = run_futures_simulation(
        prices=input_data.prices,
        target_contracts=input_data.target_contracts,
        multiplier=input_data.multiplier,
        initial_capital=input_data.initial_capital,
        initial_margin=input_data.initial_margin,
        maintenance_margin=input_data.maintenance_margin,
        commission_per_contract=input_data.commission_per_contract,
        slippage_points=input_data.slippage_points,
        collateral_rate=input_data.collateral_rate,
        contract_map=input_data.contract_map,
        allow_fractional=input_data.allow_fractional,
        roll_day_prior_prices=input_data.roll_day_prior_prices,
    )

    # How close the account ever came to a forced reduction. The engine
    # tests `equity < |contracts| * maintenance_margin` on every bar and
    # reports only whether that ever fired, so a run that spent a quarter a
    # tick above the line and a run that never went near it both reported
    # n_margin_calls = 0. maintenance_margin defaults to initial_margin,
    # exactly as the engine defaults it.
    maintenance = (
        input_data.initial_margin
        if input_data.maintenance_margin is None
        else input_data.maintenance_margin
    )
    min_margin_cushion: Optional[float] = None
    if maintenance > 0:
        equity = out["equity_curve"]
        required = out["position_curve"].abs() * maintenance
        # Only over bars where equity is positive: past that the account is
        # gone, and a ratio to a non-positive equity is not a cushion.
        solvent = equity[equity > 0]
        if not solvent.empty:
            cushion = (solvent - required.loc[solvent.index]) / solvent
            min_margin_cushion = float(cushion.min())

    run_id = input_data.run_id
    return FuturesBacktestResult(
        initial_capital=out["initial_capital"],
        final_equity=out["final_equity"],
        total_return_pct=out["total_return_pct"],
        # One number, both spellings. The engine reports a percentage; the
        # fraction is what the rest of the surface means by a drawdown.
        max_drawdown=out["max_drawdown_pct"] / 100.0,
        max_drawdown_pct=out["max_drawdown_pct"],
        max_leverage=out["max_leverage"],
        peak_exposure=out["peak_exposure"],
        total_variation_margin=out["total_variation_margin"],
        total_commission=out["total_commission"],
        total_slippage=out["total_slippage"],
        total_collateral_interest=out["total_collateral_interest"],
        n_margin_calls=out["n_margin_calls"],
        margin_calls=out["margin_calls"],
        n_rolls=out["n_rolls"],
        rolls=out["rolls"],
        margin_limited_fills=out["margin_limited_fills"],
        equity_curve={str(k.date()): float(v) for k, v in out["equity_curve"].items()},
        min_margin_cushion=min_margin_cushion,
        cash_curve_ref=_publish_state(out["cash_curve"], run_id, "cash_curve"),
        margin_curve_ref=_publish_state(out["margin_curve"], run_id, "margin_curve"),
        position_curve_ref=_publish_state(
            out["position_curve"], run_id, "position_curve"
        ),
        exposure_curve_ref=_publish_state(
            out["exposure_curve"], run_id, "exposure_curve"
        ),
        leverage_curve_ref=_publish_state(
            out["leverage_curve"], run_id, "leverage_curve"
        ),
        warnings=out["warnings"],
    )


class FuturesHedgeBacktestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio_values: Dict[str, float] = Field(
        ...,
        min_length=2,
        description="The book's MARK by date, not its returns. A hedge is "
        "sized off notional and a return series has thrown that away.",
    )
    future_prices: Dict[str, float] = Field(
        ..., min_length=2, description="Hedge instrument closes on the same dates."
    )
    multiplier: float = Field(
        ..., gt=0, le=1e6, description="Contract point value, e.g. 50 for ES."
    )
    portfolio_beta: float = Field(
        1.0,
        ge=-20.0,
        le=20.0,
        description="The book's beta to the hedge instrument. Nothing is "
        "estimated here -- a rolling beta's lookback is the most "
        "consequential choice in the simulation and it belongs to you.",
    )
    future_beta: float = Field(
        1.0, ge=-20.0, le=20.0, description="Hedge instrument's own beta, usually 1."
    )
    rehedge: Literal["daily", "weekly", "monthly", "drift"] = Field(
        "monthly",
        description="When to re-size. 'drift' re-hedges only when residual "
        "exposure leaves the band, which is what a desk runs, because every "
        "re-hedge costs two spreads and a commission.",
    )
    drift_band: float = Field(
        0.05,
        ge=0.0,
        le=1.0,
        description="Residual exposure, as a fraction of the book, that "
        "triggers a re-hedge under rehedge='drift'. Ignored otherwise.",
    )
    initial_margin: float = Field(0.0, ge=0, le=1e9)
    commission_per_contract: float = Field(0.0, ge=0, le=1e6)
    slippage_points: float = Field(0.0, ge=0, le=1e6)
    collateral_rate: float = Field(0.0, ge=-1.0, le=1.0)
    contract_map: Optional[Dict[str, str]] = Field(
        None, description="Date -> contract label, to charge the roll."
    )
    allow_fractional: bool = Field(False)


class FuturesHedgeBacktestResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_bars: int = 0
    rehedge_rule: str = ""
    n_rehedges: int = 0
    cash_pnl: Optional[float] = Field(
        None,
        description="The unhedged book's P&L. Reported SEPARATELY from the "
        "hedge on purpose: a hedged book that made money because the hedge "
        "lost less than the cash leg is a different outcome from one where "
        "the hedge worked, and a net number cannot tell them apart.",
    )
    hedge_pnl: Optional[float] = None
    combined_pnl: Optional[float] = None
    unhedged_volatility: Optional[float] = None
    hedged_volatility: Optional[float] = None
    volatility_reduction: Optional[float] = None
    residual_beta: Optional[float] = Field(
        None,
        description="Beta left after hedging. The number that says whether it worked.",
    )
    effective_hedge_ratio: Optional[float] = None
    peak_hedge_notional: Optional[float] = None
    hedge_variation_margin: Optional[float] = None
    hedge_margin_calls: int = 0
    total_commission: Optional[float] = None
    total_slippage: Optional[float] = None
    n_rolls: int = 0
    contracts_held: Dict[str, float] = Field(default_factory=dict)
    hedge_effectiveness: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


def run_futures_hedge_backtest(
    input_data: FuturesHedgeBacktestInput,
) -> FuturesHedgeBacktestResult:
    return FuturesHedgeBacktestResult(
        **_run_futures_hedge_backtest(
            portfolio_values=input_data.portfolio_values,
            future_prices=input_data.future_prices,
            multiplier=input_data.multiplier,
            portfolio_beta=input_data.portfolio_beta,
            future_beta=input_data.future_beta,
            rehedge=input_data.rehedge,
            drift_band=input_data.drift_band,
            initial_margin=input_data.initial_margin,
            commission_per_contract=input_data.commission_per_contract,
            slippage_points=input_data.slippage_points,
            collateral_rate=input_data.collateral_rate,
            contract_map=input_data.contract_map,
            allow_fractional=input_data.allow_fractional,
        )
    )


FUTURES_TOOL_DEFS = [
    (
        "run_futures_backtest",
        "Simulate a FUTURES account, whose books the shared-cash engine "
        "cannot keep. Buying ten ES at 6200 does not cost 10 x 6200 x 50 of "
        "cash, it costs margin; the position then has no market value, "
        "because its profit arrives as daily variation margin credited to "
        "cash; and a short future pays no borrow. Equity here is cash plus "
        "posted margin and the contracts contribute nothing, so the leverage "
        "reported is economic exposure over equity rather than the "
        "gross-market-value ratio, and the two are not comparable. Margin "
        "calls reduce the position rather than being financed away.",
        FuturesBacktestInput,
    ),
    (
        "run_futures_hedge_backtest",
        "Carry a cash book and its futures hedge together, bar by bar, and "
        "report the two P&L streams SEPARATELY. That separation is the "
        "point: a hedged book that made money because the hedge lost less "
        "than the cash leg is a different outcome from one where the hedge "
        "worked, and a net number cannot distinguish them. Re-hedges on a "
        "calendar or when residual exposure leaves a band -- the band is "
        "what a desk runs, since every re-hedge costs two spreads. Nothing "
        "estimates beta: the lookback is the most consequential choice in "
        "the simulation, so you supply it. Collateral is sized so margin "
        "never binds, because this measures a hedge, not a margin call.",
        FuturesHedgeBacktestInput,
    ),
]

FUTURES_TOOL_DISPATCH = {
    "run_futures_backtest": (run_futures_backtest, FuturesBacktestInput),
    "run_futures_hedge_backtest": (
        run_futures_hedge_backtest,
        FuturesHedgeBacktestInput,
    ),
}

FUTURES_TOOL_CATEGORY = {
    "run_futures_backtest": "backtest_execution",
    "run_futures_hedge_backtest": "backtest_execution",
}
