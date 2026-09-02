"""
An ETF against what it holds.

THE PREMIUM IS NOT THE ARBITRAGE. An ETF quoted 30 bps over its NAV looks
like 30 bps of free money and almost never is. Three things eat it before
anyone trades: the creation unit is a fixed block size, so the trade is not
divisible; creating requires assembling the basket at ITS spreads rather
than the ETF's; and the NAV being compared against is usually yesterday's
struck value, not a live one. This module reports the premium and then
reports what is left of it, because the second number is the one that
decides anything.

STRUCK NAV AND INTRADAY NAV ARE DIFFERENT NUMBERS. A fund's official NAV is
struck once, after the close, on the prices the administrator uses. An
intraday indicative value is a vendor's estimate published every fifteen
seconds off the last basket. Comparing a live ETF price to a struck NAV
measures mostly the market's move since the strike, which is why a
premium computed at 3pm against last night's NAV is not a mispricing and
gets flagged here as the stale comparison it is.

THE BASKET IS OPTIONAL AND CHANGES THE ANSWER. Without one this computes a
premium against a NAV the caller supplies. With one it values the creation
basket independently, which is the only way to see whether the discrepancy
is in the fund or in the NAV -- and for an ETF holding anything illiquid,
those are different problems with different fixes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from standard_quant_tools.error import ValidationError

from ._numbers import bounded, finite, non_negative, positive

__all__ = ["etf_fair_value"]

#: Below this the premium is inside the noise of a non-simultaneous print
#: and is not worth attributing. Matches the basis tolerance elsewhere in
#: this package for the same reason: two prices observed milliseconds apart
#: on a moving tape differ by about this much for no reason at all.
DEFAULT_TOLERANCE_BPS = 25.0


def etf_fair_value(
    *,
    etf_price: float,
    nav: float,
    nav_is_intraday: bool = False,
    basket_value: Optional[float] = None,
    cash_component: float = 0.0,
    creation_unit_shares: Optional[float] = None,
    creation_fee: float = 0.0,
    etf_spread_bps: float = 0.0,
    basket_spread_bps: float = 0.0,
    tolerance_bps: float = DEFAULT_TOLERANCE_BPS,
) -> Dict[str, Any]:
    """
    An ETF's premium or discount, and what survives the cost of capturing it.

    `nav` is the value the fund's shares are worth per share. `basket_value`
    plus `cash_component`, when supplied, is what one ETF share's worth of
    creation basket is independently worth -- supply it and the result
    separates a fund trading away from its holdings from a NAV that
    disagrees with them.

    Costs are charged as a ROUND TRIP on both legs, because an arbitrage is
    two trades in opposite instruments and a one-way figure understates it
    by half. The creation fee is charged once per unit, which is what makes
    the unit size matter: a fixed fee on a small unit is a large cost in
    basis points and on a large one it is nothing.
    """
    price = positive(etf_price, "etf_price")
    net_asset_value = positive(nav, "nav")

    premium = price - net_asset_value
    premium_pct = premium / net_asset_value * 100.0
    premium_bps = premium / net_asset_value * 10_000.0

    warnings: List[str] = []

    basket_total: Optional[float] = None
    basket_vs_nav_bps: Optional[float] = None
    if basket_value is not None:
        basket_total = positive(basket_value, "basket_value") + float(cash_component)
        if basket_total <= 0:
            raise ValidationError(
                f"basket_value + cash_component is {basket_total!r}, which is "
                "not a positive per-share value."
            )
        basket_vs_nav_bps = (basket_total / net_asset_value - 1.0) * 10_000.0

    # Both legs, both ways. An ETF arbitrage crosses the fund's spread and
    # the basket's, and unwinds through both again.
    execution_bps = 2.0 * (
        non_negative(etf_spread_bps, "etf_spread_bps")
        + non_negative(basket_spread_bps, "basket_spread_bps")
    )

    fee_bps = 0.0
    if creation_unit_shares is not None:
        unit_shares = positive(creation_unit_shares, "creation_unit_shares")
        unit_notional = unit_shares * price
        fee_bps = non_negative(creation_fee, "creation_fee") / unit_notional * 10_000.0
    elif creation_fee:
        warnings.append(
            "A creation fee was given with no creation_unit_shares, so it "
            "cannot be expressed in basis points and is NOT included in the "
            "net arbitrage below. A fee is per unit; without the unit size "
            "there is nothing to divide it by."
        )

    gross_bps = abs(premium_bps)
    net_bps = gross_bps - execution_bps - fee_bps

    if abs(premium_bps) <= tolerance_bps:
        classification = "fair"
    elif premium_bps > 0:
        classification = "premium"
    else:
        classification = "discount"

    # Which way the arbitrage runs. A premium means the fund is dear to what
    # it holds, so the trade is to CREATE -- buy the basket, deliver it, sell
    # the new shares. A discount reverses it.
    action = None
    if classification == "premium":
        action = "create"
    elif classification == "discount":
        action = "redeem"

    if nav_is_intraday is False and classification != "fair":
        warnings.append(
            "The NAV supplied is a STRUCK end-of-day value, so this premium "
            "includes every basket move since the strike. Intraday, that is "
            "mostly the market rather than a mispricing -- pass an indicative "
            "value with nav_is_intraday=True to compare like with like."
        )
    if classification != "fair" and net_bps <= 0:
        warnings.append(
            f"The {gross_bps:.0f} bp gross discrepancy does not survive "
            f"{execution_bps:.0f} bp of round-trip execution"
            + (f" and {fee_bps:.0f} bp of creation fee" if fee_bps else "")
            + ". This is the normal outcome and it is why a visible premium "
            "is not an opportunity."
        )
    if execution_bps == 0.0:
        warnings.append(
            "No spreads were given, so the net figure equals the gross one. "
            "That is a theoretical premium, not a tradeable edge -- an ETF "
            "arbitrage crosses the fund's spread and the basket's, twice."
        )
    if basket_vs_nav_bps is not None and abs(basket_vs_nav_bps) > tolerance_bps:
        warnings.append(
            f"The creation basket is worth {basket_vs_nav_bps:+.0f} bps "
            "against the NAV, so the fund and its own stated value disagree "
            "before the ETF price is considered. On a fund holding anything "
            "illiquid that is a stale-marks problem rather than an arbitrage."
        )

    return {
        "etf_price": float(price),
        "nav": float(net_asset_value),
        "nav_is_intraday": bool(nav_is_intraday),
        "premium_discount": float(premium),
        "premium_discount_pct": float(premium_pct),
        "premium_discount_bps": float(premium_bps),
        "classification": classification,
        "tolerance_bps": float(tolerance_bps),
        "basket_value_per_share": basket_total,
        "basket_vs_nav_bps": basket_vs_nav_bps,
        "gross_arbitrage_bps": float(gross_bps),
        "execution_bps": float(execution_bps),
        "creation_fee_bps": float(fee_bps),
        "net_arbitrage_bps": float(net_bps),
        "arbitrage_survives": bool(net_bps > 0 and classification != "fair"),
        "action": action,
        "warnings": warnings,
    }
