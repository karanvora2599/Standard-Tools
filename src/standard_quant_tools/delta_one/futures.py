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

import datetime as _dt
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from standard_quant_tools.delta_one.daycount import (
    CONVENTIONS,
    DEFAULT_CONVENTION,
)
from standard_quant_tools.delta_one.daycount import _convention as _canonical_convention
from standard_quant_tools.delta_one.daycount import day_count as _day_count_parts
from standard_quant_tools.error import ValidationError

#: One ordinary (non-leap) calendar year, used once at import to ASK
#: `daycount` what each convention divides by rather than restating four
#: denominators here. The five inline `/ 365.0` sites that module exists to
#: remove included two in this file.
_ONE_YEAR = (_dt.date(2026, 1, 1), _dt.date(2027, 1, 1))

#: What a day count is divided by, per convention, from `daycount` itself.
#: A ROLL IS MEASURED IN DAYS, NOT DATES -- the caller passes "91 days
#: between the expiries", never the two expiry dates -- so ACT/ACT cannot
#: split the period at a year boundary here and divides by an ordinary
#: 365-day year. A roll whose period contains 29 February accrues
#: marginally less than a date-aware ACT/ACT would say; ACT/365F and ACT/360
#: have constant denominators and are exact.
#:
#: 30/360 IS NOT HERE, because it is a rule for counting the NUMERATOR from
#: dates -- every month 30 days, a 31st clamped to the 30th -- and a roll
#: arrives as a count of actual days. Its denominator alone divides that
#: count, which is ACT/360 reported under another name: a 20 March to
#: 19 June quarterly roll is 91 actual days and 89 counted ones, so the
#: roll yield came back 2.2% low and labelled "30/360". `roll_analysis`
#: refuses it and names the exact remedy.
DAY_COUNT_DENOMINATORS: Dict[str, float] = {
    name: _day_count_parts(*_ONE_YEAR, convention=name)[1]
    for name in CONVENTIONS
    if name != "30/360"
}

from ._numbers import finite, non_negative, positive
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
    # A step inside float noise of the price level is no step. Strict
    # comparisons classified three contracts all at 5000 as "mixed" and
    # warned about a kinked curve hiding a dislocated segment -- of a curve
    # with no segments at all. A flat step inside a rising curve does not
    # make it non-monotonic either, so the labels are WEAK: contango means
    # no step down and at least one up.
    tolerance = 1e-12 * max(prices)
    steps = [b - a for a, b in zip(prices, prices[1:])]
    any_up = any(step > tolerance for step in steps)
    any_down = any(step < -tolerance for step in steps)
    if not any_up and not any_down:
        shape = "flat"
    elif not any_down:
        shape = "contango"
    elif not any_up:
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
    carries = [row["forward_carry_rate"] for row in spreads]
    if len(carries) >= 3:
        # MEAN OF THE CONSECUTIVE SECOND DIFFERENCES, which uses every
        # carry and needs no midpoint.
        #
        # This was `carries[-1] - 2*carries[len//2] + carries[0]`, and
        # `len//2` is only the true middle when the count is odd. With two
        # carries it collapsed to `c0 - c1` -- the NEGATIVE of the first
        # difference -- so a curve steepening from 0.0067 to 0.0133
        # reported -0.0066 against a docstring saying positive means
        # steepening. Exactly three contracts, the case the field
        # description called out as the minimum, was the broken one; with
        # five it silently skipped the second carry.
        #
        # Three carries (four contracts) is the real minimum: a second
        # difference needs three points, and three contracts give only two.
        second = [
            carries[i + 1] - 2.0 * carries[i] + carries[i - 1]
            for i in range(1, len(carries) - 1)
        ]
        curvature = float(sum(second) / len(second))

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
    days_between_expiries: Optional[float] = None,
    next_multiplier: Optional[float] = None,
    cost_per_contract: float = 0.0,
    spread_ticks: float = 0.0,
    tick_value: float = 0.0,
    day_count: str = DEFAULT_CONVENTION,
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

    BOTH ANNUALIZED NUMBERS DEPEND ON `day_count`, and it is reported back
    rather than assumed. The roll yield and the break-even are a price step
    and a cost turned into RATES, so the convention moves each by a factor
    of 360/365: the same 91-day quarterly roll is 160.376 bp under ACT/365F
    and 158.179 under ACT/360, because those 91 days are a larger slice of
    a 360-day year and the same step spread over more of a year is a lower
    rate. (The familiar "ACT/360 accrues about 1.4% more" runs the other
    way and is about an accrual at a GIVEN rate, which is what
    `swaps.price_total_return_swap` computes.) This function divided by a
    hard-coded 365 and named the convention nowhere -- not in the result,
    not in a warning, not in its own signature -- so a book financing
    ACT/360 compared its repo against someone else's convention.

    30/360 IS REFUSED. It counts the days between two DATES, and this
    function is given the days already counted; see
    `DAY_COUNT_DENOMINATORS` for what it used to compute instead.
    """
    convention = _canonical_convention(day_count)
    if convention not in DAY_COUNT_DENOMINATORS:
        raise ValidationError(
            f"roll_analysis: day_count={day_count!r} counts the days between "
            "two DATES (every month 30 days, a 31st clamped), and a roll is "
            "given here as days already counted, so it cannot be honoured: "
            "dividing actual days by 360 is ACT/360 under another name. If "
            "the roll really accrues 30/360, count the period from its dates "
            "with daycount.day_count(start, end, convention='30/360')[0] and "
            "pass that count as the days with day_count='ACT/360' -- the "
            "denominators are the same 360. Otherwise use ACT/360 for a "
            "money-market comparison."
        )
    denominator = DAY_COUNT_DENOMINATORS[convention]
    f0 = positive(front_price, "front_price")
    f1 = positive(next_price, "next_price")
    m0 = positive(multiplier, "multiplier")
    m1 = positive(next_multiplier, "next_multiplier") if next_multiplier else m0
    n = finite(contracts_held, "contracts_held")
    if n == 0:
        raise ValidationError(
            f"contracts_held={contracts_held!r} is not a position. Pass a "
            "signed non-zero size -- negative for a short, whose roll "
            "economics are the opposite sign of a long's."
        )
    days = finite(days_to_front_expiry, "days_to_front_expiry")
    if days <= 0:
        raise ValidationError(
            f"days_to_front_expiry={days_to_front_expiry!r} must be positive. "
            "A contract at or past expiry cannot be rolled, it can only be "
            "settled."
        )
    # Costs, so none of them can be negative: a negative commission made
    # the roll a CREDIT for trading (-10 per contract reported an execution
    # cost of -199.01), and a NaN one passed every comparison below and
    # came back as a NaN cost with no error.
    cost_per_contract = non_negative(cost_per_contract, "cost_per_contract")
    spread_ticks = non_negative(spread_ticks, "spread_ticks")
    tick_value = non_negative(tick_value, "tick_value")

    if spread_ticks and not tick_value:
        # `spread_ticks * tick_value` with tick_value at its default of
        # zero is zero: 83% of the spread cost went missing in silence.
        raise ValidationError(
            f"roll_analysis: spread_ticks={spread_ticks} needs tick_value "
            "(the currency value of one tick) to become a cost; with "
            "tick_value=0 the bid-ask crossed on both legs would be "
            "charged as nothing. Pass tick_value, or spread_ticks=0."
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

    # THE DENOMINATOR IS THE GAP BETWEEN THE TWO EXPIRIES, not the time
    # left on the front. `log(f1/f0)` is a carry BETWEEN contracts, so
    # annualizing it by the front's remaining life makes the answer depend
    # on the roll DATE, which is not an economic variable. Same two prices,
    # rolling at 90 days vs 1 day, reported 101 bps vs 9,114 bps -- and
    # rolling the day before expiry, which is when you roll, claimed a 91%
    # annualized break-even on an unchanged $7,540 cost. `futures_curve`
    # 140 lines above uses the expiry gap and always did.
    carry_years = (
        finite(days_between_expiries, "days_between_expiries") / denominator
        if days_between_expiries is not None
        else None
    )
    if carry_years is not None and carry_years <= 0:
        raise ValidationError(
            f"days_between_expiries={days_between_expiries!r} must be "
            "positive: it is the gap between the two contracts' expiries, "
            "which is what the roll's carry is earned over."
        )
    roll_yield = math.log(f1 / f0) / carry_years if carry_years else None

    # The break-even is a cost over the HOLDING period, so this one really
    # is the front's remaining life. Same convention as the roll yield
    # above: two rates out of one function under two different day counts
    # would be a worse answer than either.
    years = days / denominator
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
        "day_count": convention,
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
        "roll_yield_rate": (float(roll_yield) if roll_yield is not None else None),
        "roll_yield_bps": (
            float(roll_yield * 10_000.0) if roll_yield is not None else None
        ),
        "days_to_front_expiry": days,
        "days_between_expiries": (
            float(days_between_expiries) if days_between_expiries is not None else None
        ),
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
