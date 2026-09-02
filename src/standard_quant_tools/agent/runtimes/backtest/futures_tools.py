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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from standard_quant_tools.backtest.futures_engine import run_futures_simulation

logger = logging.getLogger(__name__)

__all__ = [
    "FUTURES_TOOL_CATEGORY",
    "FUTURES_TOOL_DEFS",
    "FUTURES_TOOL_DISPATCH",
    "FuturesBacktestInput",
    "FuturesBacktestResult",
    "run_futures_backtest",
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


class FuturesBacktestResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    initial_capital: Optional[float] = None
    final_equity: Optional[float] = None
    total_return_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
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
    equity_curve: Dict[str, float] = Field(
        default_factory=dict, description="Cash plus posted margin, by date."
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
    )
    return FuturesBacktestResult(
        initial_capital=out["initial_capital"],
        final_equity=out["final_equity"],
        total_return_pct=out["total_return_pct"],
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
        equity_curve={str(k.date()): float(v) for k, v in out["equity_curve"].items()},
        warnings=out["warnings"],
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
]

FUTURES_TOOL_DISPATCH = {
    "run_futures_backtest": (run_futures_backtest, FuturesBacktestInput),
}

FUTURES_TOOL_CATEGORY = {"run_futures_backtest": "backtest_execution"}
