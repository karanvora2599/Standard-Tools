"""
A futures account, which is not a smaller version of a cash account.

WHY THE PORTFOLIO ENGINE CANNOT DO THIS. `portfolio_engine.py` rests on one
identity -- `position value == shares x price == the cash you paid` -- and a
futures position breaks all three parts of it:

  * Buying ten ES at 6200 does not cost 10 x 6200 x 50 of cash. It costs
    initial margin, which might be 6% of that.
  * The position then has no market VALUE. Its profit arrives as daily
    variation margin, credited to cash, and once that is credited the
    contract is worth zero again. Counting both would double-count.
  * A short future pays no borrow. Sign alone cannot select the financing
    model any more.

So equity here is `cash + margin posted`, and the contracts contribute
nothing to it directly. That is not a modelling shortcut; it is what a
futures account statement says.

LEVERAGE IS THE NUMBER THAT LOOKS WRONG. Under the cash engine's definition
-- gross market value over equity -- a futures book is at zero leverage and
simultaneously carries many times its equity in economic exposure. Both are
reported here, separately and by different names, because a risk limit
written against one and measured against the other is how a book that looks
flat turns out not to be.

MARGIN CALLS ARE MODELLED, AND THE CASH ENGINE SAYS IT DOES NOT MODEL THEM.
When equity falls below maintenance margin the account must post more or
reduce. This reduces, because that is what a broker does when nobody posts,
and it records the event -- a backtest that quietly financed an infinite
call is a backtest of a strategy nobody could have run.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

__all__ = ["run_futures_simulation"]


def run_futures_simulation(
    *,
    prices: Mapping[Any, float],
    target_contracts: Mapping[Any, float],
    multiplier: float,
    initial_capital: float = 1_000_000.0,
    initial_margin: float = 0.0,
    maintenance_margin: Optional[float] = None,
    commission_per_contract: float = 0.0,
    slippage_points: float = 0.0,
    collateral_rate: float = 0.0,
    contract_map: Optional[Mapping[Any, str]] = None,
    allow_fractional: bool = False,
) -> Dict[str, Any]:
    """
    Simulate a futures account bar by bar.

    `prices` are the TRADEABLE prices of the contract actually held -- not a
    back-adjusted continuous series, which is not a price and would size
    every position against a level nobody could transact at.
    `target_contracts` is the signed position wanted on each date; dates
    between targets hold the last one.

    `initial_margin` and `maintenance_margin` are per contract, in currency.
    Leaving initial margin at zero models an unmargined account, which is a
    useful idealization and is not a futures account -- the result says so.

    `contract_map` names which contract each date belongs to. When it
    changes, the position is rolled: closed in the old and reopened in the
    new, paying commission and slippage on both legs. Without it, no roll
    is modelled and the series is assumed to be one contract throughout.
    """
    m = _positive(multiplier, "multiplier")
    capital = _positive(initial_capital, "initial_capital")
    im = _non_negative(initial_margin, "initial_margin")
    mm = (
        im
        if maintenance_margin is None
        else _non_negative(maintenance_margin, "maintenance_margin")
    )
    if mm > im:
        raise ValidationError(
            f"maintenance_margin ({mm}) exceeds initial_margin ({im}). A "
            "position would be in call the moment it opened."
        )
    commission = _non_negative(commission_per_contract, "commission_per_contract")
    slippage = _non_negative(slippage_points, "slippage_points")

    price_series = _series(prices, "prices")
    if (price_series <= 0).any():
        raise ValidationError("prices contains a non-positive value.")
    targets = _series(target_contracts, "target_contracts").reindex(price_series.index)
    # Held forward: a date with no new target keeps the last one, which is
    # what a position does. Before the first target the account is flat.
    targets = targets.ffill().fillna(0.0)

    # Keyed by Timestamp, because `prices` was. A caller who passes ISO
    # strings for both -- which is what a JSON payload carries -- would
    # otherwise get every `contract_map.get(date)` returning None against a
    # Timestamp key, and NO ROLL WOULD EVER FIRE. Silently: the simulation
    # runs, the numbers look plausible, and the largest recurring cost of
    # holding a future is simply absent.
    rolls_by_date: Dict[Any, str] = {}
    if contract_map is not None:
        if not contract_map:
            raise ValidationError(
                "contract_map is empty. Pass None to model no roll at all, "
                "which says so in the warnings, rather than an empty map "
                "that looks like one and is not."
            )
        rolls_by_date = {
            pd.Timestamp(key): str(value) for key, value in contract_map.items()
        }

    dates = price_series.index
    n = len(dates)
    if n < 2:
        raise ValidationError(
            f"{n} price observation(s); a simulation needs at least two bars "
            "for variation margin to be defined."
        )

    contracts = 0.0
    cash = capital
    margin_posted = 0.0
    warnings: List[str] = []

    equity_curve = np.empty(n)
    cash_curve = np.empty(n)
    margin_curve = np.empty(n)
    position_curve = np.empty(n)
    exposure_curve = np.empty(n)

    total_commission = 0.0
    total_slippage = 0.0
    total_interest = 0.0
    total_variation = 0.0
    margin_calls: List[Dict[str, Any]] = []
    rolls: List[Dict[str, Any]] = []

    previous_price = float(price_series.iloc[0])
    previous_contract = rolls_by_date.get(dates[0])

    for i, date in enumerate(dates):
        price = float(price_series.iloc[i])

        # 0. Did the contract change since the last bar? This has to be
        #    answered BEFORE variation margin, not after. `price` is the
        #    NEW contract and `previous_price` is the OLD one, so on a roll
        #    day their difference is the calendar spread, not a market move
        #    -- and booking it as profit invented returns out of nothing.
        #    A dead-flat market rolling 39 times up a contango curve
        #    reported +5.85% with a maximum drawdown of 0.00%.
        current_contract = rolls_by_date.get(date)
        rolled = (
            i > 0
            and contract_map is not None
            and current_contract is not None
            and previous_contract is not None
            and current_contract != previous_contract
            and contracts != 0.0
        )

        # 1. Variation margin on the position carried IN, before any trade.
        #    This is where a futures position's profit actually arrives.
        #
        #    Skipped on a roll day. The correct figure is the OLD contract's
        #    move over that day, and a single price series does not contain
        #    it -- the old contract's last print here is yesterday's. Zero
        #    understates by at most one day's move on one bar; the spread
        #    was overstating by the whole width of the roll, with the wrong
        #    sign for a long in contango. Pass a back-adjusted series if you
        #    need that day, or supply the roll days as their own bars.
        if i > 0 and contracts != 0.0 and not rolled:
            variation = (price - previous_price) * contracts * m
            cash += variation
            total_variation += variation

        # 2. Interest on collateral. Paid on cash, which for a futures
        #    account is most of the balance -- unlike a cash equity book,
        #    where the money is in the positions.
        if i > 0 and collateral_rate and cash > 0:
            days = max((dates[i] - dates[i - 1]).days, 0)
            interest = cash * collateral_rate * days / 365.0
            cash += interest
            total_interest += interest

        # 3. Roll costs, for the roll detected in step 0.
        if rolled:
            legs = 2.0 * abs(contracts)
            cost = legs * commission + legs * slippage * m
            cash -= cost
            total_commission += legs * commission
            total_slippage += legs * slippage * m
            rolls.append(
                {
                    "date": str(date),
                    "from": previous_contract,
                    "to": current_contract,
                    "contracts": float(contracts),
                    "cost": float(cost),
                    "spread_points": float(price - previous_price),
                    "variation_margin_skipped": True,
                }
            )

        # 4. Trade to target.
        target = float(targets.iloc[i])
        if not allow_fractional:
            target = float(round(target))
        delta = target - contracts
        if abs(delta) > 1e-9:
            cost = abs(delta) * commission + abs(delta) * slippage * m
            cash -= cost
            total_commission += abs(delta) * commission
            total_slippage += abs(delta) * slippage * m
            contracts = target
            # Margin is posted against the position now held, released
            # against what was closed. It moves between cash and margin
            # rather than leaving the account.
            required = abs(contracts) * im
            cash -= required - margin_posted
            margin_posted = required

        equity = cash + margin_posted
        exposure = abs(contracts) * price * m

        # 5. Maintenance. Equity below the maintenance requirement is a call.
        required_maintenance = abs(contracts) * mm
        if mm > 0 and contracts != 0.0 and equity < required_maintenance:
            # Reduce to what the account can actually carry, which is what a
            # broker does when nobody answers the call.
            affordable = math.floor(equity / mm) if mm > 0 else 0.0
            affordable = max(affordable, 0.0)
            reduced_to = math.copysign(min(abs(contracts), affordable), contracts)
            closed = contracts - reduced_to
            if abs(closed) > 1e-9:
                cost = abs(closed) * commission + abs(closed) * slippage * m
                cash -= cost
                total_commission += abs(closed) * commission
                total_slippage += abs(closed) * slippage * m
                contracts = reduced_to
                required = abs(contracts) * im
                cash -= required - margin_posted
                margin_posted = required
                equity = cash + margin_posted
                exposure = abs(contracts) * price * m
                margin_calls.append(
                    {
                        "date": str(date),
                        "equity": float(equity),
                        "required": float(required_maintenance),
                        "contracts_closed": float(closed),
                    }
                )

        if equity <= 0:
            warnings.append(
                f"The account went to zero equity on {date}. Everything from "
                "that bar on is a simulation of an account that no longer "
                "exists; the curves are truncated there."
            )
            equity_curve[i:] = equity
            cash_curve[i:] = cash
            margin_curve[i:] = margin_posted
            position_curve[i:] = 0.0
            exposure_curve[i:] = 0.0
            break

        equity_curve[i] = equity
        cash_curve[i] = cash
        margin_curve[i] = margin_posted
        position_curve[i] = contracts
        exposure_curve[i] = exposure
        previous_price = price
        if current_contract is not None:
            previous_contract = current_contract

    equity = pd.Series(equity_curve, index=dates, name="equity")
    exposure = pd.Series(exposure_curve, index=dates, name="exposure")
    with np.errstate(divide="ignore", invalid="ignore"):
        leverage = (exposure / equity).replace([np.inf, -np.inf], np.nan)

    if im == 0.0:
        warnings.append(
            "initial_margin is zero, so this models an account that posts "
            "nothing to hold a position. That is a useful idealization and "
            "it is not a futures account -- leverage below is unbounded by "
            "construction and no margin call can ever fire."
        )
    if margin_calls:
        warnings.append(
            f"{len(margin_calls)} margin call(s) forced the position down. A "
            "backtest that financed those calls instead would be testing a "
            "strategy nobody could have run."
        )
    if contract_map is None:
        warnings.append(
            "No contract_map, so NO ROLL was modelled and these prices are "
            "assumed to be one contract throughout. Over any horizon longer "
            "than a single expiry that omits the largest recurring cost of "
            "holding a future."
        )
    warnings.append(
        "Equity is cash plus posted margin. The contracts contribute no "
        "market value, because their profit has already been credited to "
        "cash as variation margin -- counting both would double it. "
        "`leverage` is therefore ECONOMIC EXPOSURE over equity, not the "
        "gross-market-value ratio the cash engine reports, and the two are "
        "not comparable."
    )

    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    return {
        "equity_curve": equity,
        "cash_curve": pd.Series(cash_curve, index=dates, name="cash"),
        "margin_curve": pd.Series(margin_curve, index=dates, name="margin"),
        "position_curve": pd.Series(position_curve, index=dates, name="contracts"),
        "exposure_curve": exposure,
        "leverage_curve": leverage,
        "initial_capital": capital,
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": float((equity.iloc[-1] / capital - 1.0) * 100.0),
        # PERCENT here (x100), unlike the identically-named field in
        # `stress_test`, which returns a fraction. Same name, 100x apart,
        # both agent-reachable, and neither used to say which.
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "max_leverage": float(np.nanmax(leverage.to_numpy())),
        "peak_exposure": float(exposure.max()),
        "total_variation_margin": float(total_variation),
        "total_commission": float(total_commission),
        "total_slippage": float(total_slippage),
        "total_collateral_interest": float(total_interest),
        "n_margin_calls": len(margin_calls),
        "margin_calls": margin_calls,
        "n_rolls": len(rolls),
        "rolls": rolls,
        "warnings": warnings,
    }


# ── internals ───────────────────────────────────────────────────────────


def _series(mapping: Mapping[Any, float], name: str) -> pd.Series:
    if not mapping:
        raise ValidationError(f"{name} is empty.")
    try:
        series = pd.Series(dict(mapping), dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must map dates to numbers; {exc}") from None
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def _positive(value: Any, name: str) -> float:
    out = _finite(value, name)
    if out <= 0:
        raise ValidationError(f"{name} must be positive, got {value!r}")
    return out


def _non_negative(value: Any, name: str) -> float:
    out = _finite(value, name)
    if out < 0:
        raise ValidationError(f"{name} must not be negative, got {value!r}")
    return out


def _finite(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return out
