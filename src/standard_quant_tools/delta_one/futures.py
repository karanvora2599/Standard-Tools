"""
The futures curve, and what moving a position along it costs.

TWO QUESTIONS, KEPT APART. `futures_curve` asks what the term structure
looks like. `roll_analysis` asks what happens if I move my actual position
from this contract into that one. They sound adjacent and they are not: the
first is a description of the market and needs no position, the second is a
trade with a size, a cost and a break-even, and folding them together would
produce a tool that could not answer either cleanly.

THE VOL RUNTIME ALREADY HAS THIS SHAPE. `analyze_vol_term_structure`
computes forward volatilities between expiries and reports contango or
backwardation, and its central point applies here unchanged: a trader
seeing the near contract at one carry and the far one at another is not
being offered the far number for the period between them. They are being
offered the FORWARD carry, which is whatever makes the two consistent, and
trading off the quoted levels instead can reverse the sign of the position.
This module is the price-curve analogue of that one.

A WARNING ABOUT THE WORD "CONTANGO". In this library those two words
already mean the shape of an implied-VOLATILITY term structure, because
that is the only term structure it had. Here they mean the price curve. The
two are unrelated and a curve can be in contango on one and backwardation
on the other at the same time, so this module always says which.

ROLL YIELD IS NOT A RETURN. It is the price step you pay or collect for
moving between contracts, expressed as a rate. It is not money earned: a
position rolled up a contango curve loses that step if spot does not move,
and a backwardated curve does not pay you unless spot behaves. Reporting it
as "yield" without that sentence is how the term became misleading.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from standard_quant_tools.analysis.derivatives import _positive
from standard_quant_tools.error import ValidationError

from ._numbers import bounded, finite, non_negative, positive
from .carry import observed_carry_rate

__all__ = ["futures_curve", "roll_analysis"]


def futures_curve(
    contracts: Sequence[Mapping[str, Any]],
    *,
    spot: Optional[float] = None,
) -> Dict[str, Any]:
    """
    A term structure of futures prices, with the forward carry between them.

    `contracts` is a sequence of mappings carrying `time_to_expiry` (years)
    and `price`, plus an optional `label`. They are sorted by expiry here
    rather than trusted in order, because a curve given out of order
    produces calendar spreads with the wrong sign and every downstream
    number stays plausible.

    `spot` is optional and changes what can be computed. With it, each
    contract gets a basis and an implied carry against cash. Without it,
    only the relationships BETWEEN contracts are available -- which is
    still most of the curve, and is the honest answer when the cash index
    is not observable at the same instant as the futures.
    """
    rows = _parse_contracts(contracts)
    if len(rows) < 2:
        raise ValidationError(
            f"a curve needs at least two contracts, got {len(rows)}. For a "
            "single contract against cash, use cash_futures_basis."
        )

    warnings: List[str] = []
    s = positive(spot, "spot") if spot is not None else None

    points: List[Dict[str, Any]] = []
    for row in rows:
        point: Dict[str, Any] = {
            "label": row["label"],
            "time_to_expiry": row["time_to_expiry"],
            "price": row["price"],
            "basis_points": None,
            "implied_carry_rate": None,
            "annualized_basis_bps": None,
        }
        if s is not None:
            carry = observed_carry_rate(
                spot=s, forward=row["price"], time_to_expiry=row["time_to_expiry"]
            )
            point["basis_points"] = float(row["price"] - s)
            point["implied_carry_rate"] = float(carry)
            point["annualized_basis_bps"] = float(carry * 10_000.0)
        points.append(point)

    spreads: List[Dict[str, Any]] = []
    for near, far in zip(rows, rows[1:]):
        dt = far["time_to_expiry"] - near["time_to_expiry"]
        if dt <= 0:
            raise ValidationError(
                f"contracts {near['label']!r} and {far['label']!r} have the "
                "same time to expiry, so the forward carry between them is "
                "undefined (it divides by that gap)."
            )
        forward_carry = math.log(far["price"] / near["price"]) / dt
        spreads.append(
            {
                "near": near["label"],
                "far": far["label"],
                "calendar_spread_points": float(far["price"] - near["price"]),
                "years_between": float(dt),
                "forward_carry_rate": float(forward_carry),
                "forward_carry_bps": float(forward_carry * 10_000.0),
            }
        )

    prices = [row["price"] for row in rows]
    rising = all(b > a for a, b in zip(prices, prices[1:]))
    falling = all(b < a for a, b in zip(prices, prices[1:]))
    if rising:
        shape = "contango"
    elif falling:
        shape = "backwardation"
    else:
        shape = "mixed"
        warnings.append(
            "The curve is not monotonic, so it is neither in contango nor in "
            "backwardation as a whole. Read the calendar spreads "
            "individually -- a single label for a kinked curve hides the "
            "segment that is actually dislocated."
        )

    total_years = rows[-1]["time_to_expiry"] - rows[0]["time_to_expiry"]
    slope = (
        (math.log(prices[-1] / prices[0]) / total_years) if total_years > 0 else None
    )

    curvature = None
    if len(spreads) >= 2:
        # Second difference of forward carry: positive means the curve
        # steepens with maturity. Reported only with three contracts
        # because with two there is one segment and no bend to measure.
        carries = [row["forward_carry_rate"] for row in spreads]
        curvature = float(carries[-1] - 2.0 * carries[len(carries) // 2] + carries[0])

    if s is None:
        warnings.append(
            "No spot was given, so each contract's basis against cash is "
            "undefined and only the forward carries BETWEEN contracts are "
            "reported. Those are still the calendar-spread economics; what "
            "is missing is whether the whole curve is rich to cash."
        )
    warnings.append(
        "Contango and backwardation here describe the PRICE curve. In this "
        "library the same two words describe an implied-volatility term "
        "structure (analyze_vol_term_structure) and the two are unrelated -- "
        "a name can be in contango on one and backwardation on the other."
    )
    warnings.append(
        "The forward carry between two expiries is what a calendar spread "
        "actually prices, and it is not the far contract's own carry. "
        "Trading off the individual levels rather than the forward can "
        "reverse the sign of the position."
    )

    return {
        "n_contracts": len(rows),
        "shape": shape,
        "spot": float(s) if s is not None else None,
        "curve": points,
        "calendar_spreads": spreads,
        "curve_slope_rate": float(slope) if slope is not None else None,
        "curve_curvature": curvature,
        "front_label": rows[0]["label"],
        "back_label": rows[-1]["label"],
        "warnings": warnings,
    }


def roll_analysis(
    *,
    front_price: float,
    next_price: float,
    contracts_held: float,
    multiplier: float,
    days_to_front_expiry: float,
    next_multiplier: Optional[float] = None,
    cost_per_contract: float = 0.0,
    spread_ticks: float = 0.0,
    tick_value: float = 0.0,
) -> Dict[str, Any]:
    """
    What moving a position from the front contract into the next one costs.

    `contracts_held` is SIGNED: negative is a short position, and the sign
    decides which way the roll spread cuts. A short rolled up a contango
    curve collects the step that a long pays, and a function that took an
    absolute size would report the wrong sign for half its callers.

    The break-even is the number to read. A roll is not free and it is not
    a loss either; it is a cost that the position has to out-earn before
    the next expiry, and `breakeven_annualized_rate` is the rate of return
    on the position's notional that exactly repays it.
    """
    f0 = positive(front_price, "front_price")
    f1 = positive(next_price, "next_price")
    m0 = positive(multiplier, "multiplier")
    m1 = positive(next_multiplier, "next_multiplier") if next_multiplier else m0
    n = float(contracts_held)
    if not math.isfinite(n) or n == 0:
        raise ValidationError(
            f"contracts_held={contracts_held!r} is not a position. Pass a "
            "signed non-zero size -- negative for a short, whose roll "
            "economics are the opposite sign of a long's."
        )
    days = float(days_to_front_expiry)
    if not math.isfinite(days) or days <= 0:
        raise ValidationError(
            f"days_to_front_expiry={days_to_front_expiry!r} must be positive. "
            "A contract at or past expiry cannot be rolled, it can only be "
            "settled."
        )

    roll_spread = f1 - f0
    front_notional = abs(n) * f0 * m0
    # Sized to hold the same MONEY, not the same contract count. When the
    # two multipliers differ (a micro rolling into a full-size, an index
    # redenomination) equal counts are a different position, and the
    # difference is exactly the factor most likely to go unnoticed.
    next_contracts = n * (f0 * m0) / (f1 * m1)

    # Rolling a long means selling the front and buying the next, so a
    # positive roll spread is a cost to a long and a credit to a short.
    cash_impact = -n * roll_spread * m0

    execution = (
        abs(n) * cost_per_contract
        + abs(next_contracts) * cost_per_contract
        + (abs(n) + abs(next_contracts)) * spread_ticks * tick_value
    )
    net_cost = -cash_impact + execution

    years = days / 365.0
    roll_yield = math.log(f1 / f0) / years if years > 0 else float("nan")
    breakeven = (
        net_cost / front_notional / years if front_notional and years else float("nan")
    )

    warnings: List[str] = []
    if abs(m1 - m0) > 1e-12:
        warnings.append(
            f"The two contracts have different multipliers ({m0:g} and "
            f"{m1:g}), so the roll is NOT contract-for-contract. "
            f"{abs(next_contracts):.2f} of the next contract holds the same "
            f"money as {abs(n):.2f} of the front; rolling one-for-one would "
            f"change the position size by "
            f"{abs(abs(n) / abs(next_contracts) - 1) * 100:.1f}%."
        )
    if execution == 0.0:
        warnings.append(
            "No execution cost was given, so this is the theoretical roll. "
            "A real roll crosses two spreads and the calendar spread is "
            "usually the wider of the quotes -- the net cost below is a "
            "floor, not an estimate."
        )
    warnings.append(
        "Roll yield is a PRICE STEP expressed as a rate, not a return. A "
        "long rolling up a contango curve gives up this much if spot does "
        "not move, and a backwardated curve does not pay it unless spot "
        "behaves. It is what the position must overcome, not what it earns."
    )

    return {
        "front_price": float(f0),
        "next_price": float(f1),
        "roll_spread_points": float(roll_spread),
        "contracts_held": n,
        "next_contracts_exact": float(next_contracts),
        "next_contracts_rounded": float(round(next_contracts)),
        "front_notional": float(front_notional),
        "cash_impact": float(cash_impact),
        "execution_cost": float(execution),
        "net_roll_cost": float(net_cost),
        "net_roll_cost_bps": (
            float(net_cost / front_notional * 10_000.0)
            if front_notional
            else float("nan")
        ),
        "roll_yield_rate": float(roll_yield),
        "roll_yield_bps": float(roll_yield * 10_000.0),
        "days_to_front_expiry": days,
        "breakeven_annualized_rate": float(breakeven),
        "warnings": warnings,
    }


# ── internals ───────────────────────────────────────────────────────────


def _parse_contracts(contracts: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and sort a curve, naming the contract that is wrong."""
    if not contracts:
        raise ValidationError("contracts is empty; a curve needs at least two.")

    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(contracts):
        if not isinstance(item, Mapping):
            raise ValidationError(
                f"contracts[{index}] is {type(item).__name__}, not a mapping "
                "with `time_to_expiry` and `price`."
            )
        label = str(item.get("label") or f"contract_{index}")
        if "time_to_expiry" not in item or "price" not in item:
            raise ValidationError(
                f"contract {label!r} needs both `time_to_expiry` (in years) "
                f"and `price`; got keys {sorted(item)}."
            )
        rows.append(
            {
                "label": label,
                "time_to_expiry": positive(
                    item["time_to_expiry"], f"{label}.time_to_expiry"
                ),
                "price": positive(item["price"], f"{label}.price"),
            }
        )

    # Sorted rather than trusted: a curve handed over out of order yields
    # negative calendar spreads that look like backwardation.
    rows.sort(key=lambda row: row["time_to_expiry"])
    return rows
