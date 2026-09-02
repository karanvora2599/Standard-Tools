"""
The cash-and-carry identity, forwards and backwards.

ONE IDENTITY, TWO DIRECTIONS. Everything here is `F = S * exp((r - q - b) * T)`
rearranged. Forwards, it prices a fair forward from three rates and is
already implemented -- `analysis.derivatives.implied_forward_price` does it
with the components broken out, and this module CALLS that rather than
writing a second copy of an equation the library already owns. Backwards,
it recovers whichever of the three rates a quoted price implies, and that
direction did not exist.

WHY THE INVERSE IS ONE FUNCTION AND NOT THREE. `get_implied_financing`,
`get_implied_dividend` and `get_implied_borrow` would be three names for
one rearrangement, three schemas to serve, and three near-identical
descriptions for a model to choose between -- and this library has already
been bitten once by a pair of tool names close enough that a model could
not tell them apart. `solve_for` makes the choice a parameter, where it
belongs, and the arithmetic is shared so the three answers cannot drift.

THE THING THE INVERSE CANNOT DO. The identity has one equation and three
unknowns, so recovering one rate requires the other two as inputs. There
is no way to decompose a single quoted future into financing AND dividend
AND borrow, and a caller asking for that is asking for something the math
does not contain. Every implied rate here is therefore conditional, and it
says so: an implied borrow of 340 bps means "340 bps IF the financing and
dividend you supplied are right", and it absorbs every error in both.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from standard_quant_tools.analysis.derivatives import (
    MAX_RATE,
    _bounded,
    _positive,
    implied_forward_price,
)
from standard_quant_tools.error import ValidationError

from ._numbers import bounded, finite, non_negative, positive

__all__ = ["SOLVE_TARGETS", "forward_price", "observed_carry_rate", "solve_carry"]

#: Which unknown the inverse solves for, and what the answer means. The
#: prose is here rather than in the runtime layer because it is the same
#: caveat whether a human or a model is reading it.
SOLVE_TARGETS: Dict[str, str] = {
    "financing_rate": (
        "The repo or funding rate the quoted price implies, given your "
        "dividend and borrow. Compare it to SOFR or ESTR: a future trading "
        "above fair on a correct dividend is usually funding, not edge."
    ),
    "dividend_yield": (
        "The dividend the quoted price implies, given your financing and "
        "borrow. Useful on an index whose forecast dividend is contested -- "
        "the future is a market view on it."
    ),
    "borrow_rate": (
        "The stock-loan rate the quoted price implies, given your financing "
        "and dividend. On a hard-to-borrow single name this is usually the "
        "term that is moving, which is why the library keeps it separate "
        "from the dividend everywhere else."
    ),
}


def forward_price(
    *,
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    borrow_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    The fair forward and its three components.

    A thin pass-through to `analysis.derivatives.implied_forward_price`,
    kept here so Delta One code has one import for carry and so the
    delegation is visible. It computes nothing of its own on purpose: a
    second implementation of `F = S exp((r-q-b)T)` is a second thing that
    can be wrong, and the two would be discovered to disagree by somebody
    pricing a trade.
    """
    return implied_forward_price(
        spot=spot,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        borrow_rate=borrow_rate,
    )


def observed_carry_rate(*, spot: float, forward: float, time_to_expiry: float) -> float:
    """
    The continuously-compounded net carry a quoted forward implies.

    `ln(F/S) / T`. This is the whole left-hand side of the identity, and it
    is a single number regardless of how the three components split -- the
    market quotes one price and the decomposition is always an assumption
    laid over it.
    """
    s = positive(spot, "spot")
    f = positive(forward, "forward")
    t = positive(time_to_expiry, "time_to_expiry")
    return math.log(f / s) / t


def solve_carry(
    *,
    spot: float,
    forward: float,
    time_to_expiry: float,
    solve_for: str,
    risk_free_rate: Optional[float] = None,
    dividend_yield: Optional[float] = None,
    borrow_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Recover one carry component from a quoted forward and the other two.

    Rearranging `ln(F/S)/T = r - q - b`:

        r = ln(F/S)/T + q + b
        q = r - b - ln(F/S)/T
        b = r - q - ln(F/S)/T

    The two supplied rates are required rather than defaulted to zero. A
    default would make the commonest mistake silent: solving for borrow
    with no dividend passed returns the borrow that would hold if the name
    paid nothing, which on a 3% yielder is wrong by 300 bps and looks
    entirely reasonable.
    """
    if solve_for not in SOLVE_TARGETS:
        raise ValidationError(
            f"solve_for={solve_for!r} is not one of {sorted(SOLVE_TARGETS)}. "
            "The identity has three terms and one equation, so exactly one "
            "of them can be recovered from a quoted price."
        )

    net_carry = observed_carry_rate(
        spot=spot, forward=forward, time_to_expiry=time_to_expiry
    )
    t = float(time_to_expiry)

    given = {
        "risk_free_rate": risk_free_rate,
        "dividend_yield": dividend_yield,
        "borrow_rate": borrow_rate,
    }
    target_arg = {
        "financing_rate": "risk_free_rate",
        "dividend_yield": "dividend_yield",
        "borrow_rate": "borrow_rate",
    }[solve_for]

    missing = [
        name for name, value in given.items() if name != target_arg and value is None
    ]
    if missing:
        raise ValidationError(
            f"solving for {solve_for!r} needs the other two components, and "
            f"{missing} {'were' if len(missing) > 1 else 'was'} not given. "
            "They are not defaulted to zero: an implied borrow computed "
            "against an assumed zero dividend is wrong by the whole "
            "dividend and looks perfectly plausible."
        )

    # Bound and unpack in one place. The `missing` check above already
    # guarantees the two non-target rates are present; pulling them into
    # locals here is what lets the arithmetic below read as arithmetic
    # rather than as three lines of float(x or 0) defensiveness.
    supplied = {
        name: _bounded(float(value), name, low=-MAX_RATE, high=MAX_RATE)
        for name, value in given.items()
        if name != target_arg and value is not None
    }

    if solve_for == "financing_rate":
        solved = net_carry + supplied["dividend_yield"] + supplied["borrow_rate"]
    elif solve_for == "dividend_yield":
        solved = supplied["risk_free_rate"] - supplied["borrow_rate"] - net_carry
    else:
        solved = supplied["risk_free_rate"] - supplied["dividend_yield"] - net_carry

    warnings: List[str] = []
    if abs(solved) > 1.0:
        warnings.append(
            f"The implied {solve_for} is {solved * 100:.0f}%, which is not a "
            "rate any market charges. Check the time to expiry first -- it "
            "is in YEARS here, and passing days is the usual cause; a "
            "quoted price on the wrong underlying is the other."
        )
    if solve_for == "borrow_rate" and solved < 0:
        warnings.append(
            f"A negative implied borrow ({solved * 10_000:.0f} bps) means the "
            "quote is cheaper than the other two components allow. It is a "
            "rebate only in the rarest cases; far more often the dividend "
            "assumption is too high or the future is genuinely cheap."
        )
    if solve_for == "dividend_yield" and solved < 0:
        warnings.append(
            f"A negative implied dividend ({solved * 100:.2f}%) is not a "
            "dividend. Either the financing or the borrow you supplied is "
            "too low, or the contract is not on the underlying you think."
        )

    warnings.append(
        f"CONDITIONAL on the two rates you supplied. The identity has one "
        f"equation and three unknowns, so this {solve_for} absorbs every "
        "error in the other two -- it is what makes the quote consistent, "
        "not an independently observed rate."
    )

    return {
        "solved_for": solve_for,
        "solved_rate": float(solved),
        "solved_rate_bps": float(solved * 10_000.0),
        "net_carry_rate": float(net_carry),
        "spot": float(spot),
        "forward": float(forward),
        "time_to_expiry": t,
        "assumed": dict(supplied),
        "meaning": SOLVE_TARGETS[solve_for],
        "warnings": warnings,
    }
