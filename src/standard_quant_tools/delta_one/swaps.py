"""
Total return swaps and total return futures: the same exposure, financed.

WHAT A TRS IS, ARITHMETICALLY. One side receives the underlying's total
return -- price change plus dividends -- and pays a financing rate plus a
spread on the notional. That is the whole product. The complexity is not in
the payoff, it is in the conventions: which day count the financing accrues
on, whether the notional resets, and whether dividends arrive gross or net
of withholding. Each of those changes the number and none of them is
visible in the result unless it is named, so all three are arguments here
and all three are reported back.

THE SPREAD IS THE PRODUCT. Financing is a reference rate anyone can look
up; what a desk actually quotes is the spread over it, and that is where
the whole economics of the trade lives. A TRS at SOFR+35 and one at
SOFR+95 are the same instrument at very different prices, and the second is
often still cheaper than holding the cash position once borrow and balance
sheet are counted -- which is what compare_delta_one_expressions is for.

WHY A TRF IS NOT AN OPTION AND SITS HERE. A total return future is an
exchange-listed instrument whose quoted level embeds a financing spread
rather than a price. It belongs with the swap it replicates, not with
convex products. Its one genuine subtlety is that quoting conventions
differ between exchanges and contracts -- some quote the spread in basis
points, some quote a level -- so `quote_convention` is required rather
than assumed. A tool that guessed would be wrong for half its callers and
silently.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from standard_quant_tools.analysis.derivatives import MAX_RATE
from standard_quant_tools.error import ValidationError

from ._numbers import bounded, finite, non_negative, positive
from .daycount import DEFAULT_CONVENTION, year_fraction

__all__ = ["QUOTE_CONVENTIONS", "price_total_return_swap", "total_return_future"]

#: How a TRF's quoted number should be read. Required rather than defaulted
#: because the two are not convertible without knowing which you have, and
#: a wrong guess misprices by the whole financing leg.
QUOTE_CONVENTIONS: Dict[str, str] = {
    "spread_bps": (
        "The quote IS the financing spread over the reference rate, in "
        "basis points. Common on European index TRFs."
    ),
    "index_level": (
        "The quote is a price level that embeds accrued financing, so the "
        "spread has to be backed out of it against the underlying."
    ),
}


def price_total_return_swap(
    *,
    notional: float,
    initial_price: float,
    current_price: float,
    financing_rate: float,
    spread_bps: float = 0.0,
    dividends: float = 0.0,
    start_date: Optional[str] = None,
    valuation_date: Optional[str] = None,
    time_elapsed: Optional[float] = None,
    day_count: str = DEFAULT_CONVENTION,
    direction: str = "receive",
) -> Dict[str, Any]:
    """
    Mark a total return swap, with the equity and financing legs separated.

        total return = (S_t - S_0 + D) / S_0
        financing    = (r + spread) * T
        P&L          = notional * (total return - financing)

    `direction` is which side of the swap this is. "receive" takes the total
    return and pays financing, which is the long-equivalent and the default;
    "pay" is the mirror, and it is a genuinely different position rather
    than a sign convention -- a payer earns the financing leg.

    Time comes either from two dates (with a day-count convention, since
    financing accrues on calendar terms) or from `time_elapsed` in years
    directly. The dates are preferred: ACT/360 and ACT/365F differ by about
    1.4% of the financing leg, which on a large notional is real money and
    is invisible if the convention is implicit.
    """
    if direction not in ("receive", "pay"):
        raise ValidationError(
            f"direction={direction!r} must be 'receive' or 'pay'. A payer "
            "earns the financing leg and owes the equity return, which is a "
            "different position rather than a sign flip on the same one."
        )
    n = finite(notional, "notional")
    s0 = positive(initial_price, "initial_price")
    s1 = positive(current_price, "current_price")
    div = finite(dividends, "dividends")

    t, source = _elapsed(start_date, valuation_date, time_elapsed, day_count)

    price_return = (s1 - s0) / s0
    dividend_return = div / s0
    total_return = price_return + dividend_return

    all_in_rate = float(financing_rate) + float(spread_bps) / 10_000.0
    financing = all_in_rate * t

    sign = 1.0 if direction == "receive" else -1.0
    equity_leg = sign * n * total_return
    financing_leg = -sign * n * financing
    net = equity_leg + financing_leg

    warnings: List[str] = []
    if div == 0.0:
        warnings.append(
            "No dividends were given, so this is a PRICE-return swap in "
            "everything but name. On a paying underlying the total return "
            "leg is understated by the whole dividend, which over a year on "
            "a 2% yielder is 200 bps of the notional."
        )
    if spread_bps == 0.0:
        warnings.append(
            "The financing spread is zero, so this prices at the reference "
            "rate flat. That is not a quote anyone receives -- the spread is "
            "the negotiated part and is where the product's economics are."
        )
    if t <= 0:
        warnings.append(
            f"The elapsed time is {t:.6f} years, so the financing leg is "
            "zero or negative and this is a valuation at or before "
            "inception rather than a mark."
        )
    warnings.append(
        f"Financing accrued on {source} over {t:.6f} years ({day_count}). "
        "ACT/360 and ACT/365F differ by about 1.4% of the financing leg, so "
        "this number is only right if that convention matches the contract."
    )

    return {
        "direction": direction,
        "notional": n,
        "initial_price": s0,
        "current_price": s1,
        "time_elapsed_years": float(t),
        "day_count": day_count,
        "price_return": float(price_return),
        "dividend_return": float(dividend_return),
        "total_return": float(total_return),
        "financing_rate": float(financing_rate),
        "spread_bps": float(spread_bps),
        "all_in_financing_rate": float(all_in_rate),
        "financing_accrued": float(financing),
        "equity_leg": float(equity_leg),
        "financing_leg": float(financing_leg),
        "net_pnl": float(net),
        "net_return_on_notional": float(net / n) if n else float("nan"),
        "warnings": warnings,
    }


def total_return_future(
    *,
    quote: float,
    quote_convention: str,
    underlying_price: float,
    time_to_expiry: float,
    reference_rate: float,
    dividend_yield: float = 0.0,
    comparison_spread_bps: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Read a TRF quote as a financing spread, and compare it to a swap's.

    This is the tool that answers "regular futures imply 50 bp of funding
    and the TRF implies 95 bp -- where does the 45 go", which is otherwise
    a question a model has to reconstruct contract economics to answer.

    `comparison_spread_bps` is what the same exposure costs somewhere else,
    typically the TRS quote. Supplying it turns the result from a
    measurement into a decision.
    """
    if quote_convention not in QUOTE_CONVENTIONS:
        raise ValidationError(
            f"quote_convention={quote_convention!r} must be one of "
            f"{sorted(QUOTE_CONVENTIONS)}. The two are not convertible "
            "without knowing which one a quote is, and assuming the wrong "
            "one misprices by the entire financing leg."
        )
    s = positive(underlying_price, "underlying_price")
    t = positive(time_to_expiry, "time_to_expiry")
    # Bounded before it reaches `exp`. Found by the adversarial sweep: an
    # unbounded spread of 1e12 bps left as a bare OverflowError from
    # math.exp rather than as a ValidationError naming the argument, and
    # -1e300 silently returned a level of exactly 0.0 -- a real number
    # claiming the contract is worthless. Same failure `_bounded` exists to
    # stop, and the same one `implied_forward_price` had in its dividend.
    rate = bounded(reference_rate, "reference_rate", low=-MAX_RATE, high=MAX_RATE)

    if quote_convention == "spread_bps":
        implied_spread_bps = bounded(
            quote,
            "quote",
            low=-MAX_RATE * 10_000.0,
            high=MAX_RATE * 10_000.0,
            unit=" bps",
        )
        bounded(
            rate + implied_spread_bps / 10_000.0,
            "reference_rate + quote",
            low=-MAX_RATE,
            high=MAX_RATE,
        )
        # A level consistent with that spread, so both conventions come back
        # populated whichever one was supplied.
        implied_level = s * math.exp((rate + implied_spread_bps / 10_000.0) * t)
    else:
        level = positive(quote, "quote")
        implied_level = level
        # ln(level / spot) / T is the all-in financing the level embeds; the
        # spread is whatever that is above the reference rate.
        all_in = math.log(level / s) / t
        implied_spread_bps = (all_in - rate) * 10_000.0

    all_in_rate = rate + implied_spread_bps / 10_000.0
    carry_cost = all_in_rate - float(dividend_yield)

    difference_bps = None
    if comparison_spread_bps is not None:
        difference_bps = implied_spread_bps - float(comparison_spread_bps)

    warnings: List[str] = []
    if abs(implied_spread_bps) > 1_000:
        warnings.append(
            f"The implied spread is {implied_spread_bps:.0f} bps, which is "
            "not a financing spread anyone quotes. Check quote_convention "
            "first -- reading a level as a spread, or the reverse, produces "
            "exactly this."
        )
    if difference_bps is not None and abs(difference_bps) > 25:
        warnings.append(
            f"The TRF implies {implied_spread_bps:.0f} bps against "
            f"{float(comparison_spread_bps):.0f} bps elsewhere, a "
            f"{difference_bps:+.0f} bp gap. Before treating it as relative "
            "value, account for what the two instruments do not share: "
            "margin versus posted collateral, exchange clearing against "
            "bilateral credit, and the balance-sheet cost of each."
        )
    warnings.append(
        "The implied spread is CONDITIONAL on the reference rate supplied. "
        "It absorbs any error in that rate one-for-one, so comparing two "
        "instruments requires quoting both against the same reference."
    )

    return {
        "quote": float(quote),
        "quote_convention": quote_convention,
        "convention_meaning": QUOTE_CONVENTIONS[quote_convention],
        "underlying_price": s,
        "time_to_expiry": t,
        "reference_rate": float(rate),
        "implied_spread_bps": float(implied_spread_bps),
        "implied_level": float(implied_level),
        "all_in_financing_rate": float(all_in_rate),
        "net_carry_rate": float(carry_cost),
        "comparison_spread_bps": (
            None if comparison_spread_bps is None else float(comparison_spread_bps)
        ),
        "difference_bps": difference_bps,
        "warnings": warnings,
    }


# ── internals ───────────────────────────────────────────────────────────


def _elapsed(start, end, elapsed, convention):
    """Years, from two dates or from a number, but not from neither."""
    if elapsed is not None:
        if start is not None or end is not None:
            raise ValidationError(
                "pass either time_elapsed OR start_date/valuation_date, not "
                "both -- two answers to the same question makes the "
                "precedence rule part of the contract."
            )
        value = finite(elapsed, "time_elapsed")
        return value, "the time_elapsed given"
    if start is None or end is None:
        raise ValidationError(
            "the financing leg needs a period: pass start_date and "
            "valuation_date, or time_elapsed in years."
        )
    return year_fraction(start, end, convention=convention), f"{start} to {end}"
