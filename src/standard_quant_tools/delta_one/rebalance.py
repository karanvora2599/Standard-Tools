"""
What an index change forces passive money to trade.

THE FLOW IS MECHANICAL AND THE DATE IS KNOWN. That combination is what
makes an index rebalance different from every other kind of order flow. A
name entering an index at 0.35% weight, against $800bn of assets tracking
it, obliges roughly $2.8bn of buying -- not because anyone formed a view,
but because a mandate says the tracker must hold the index. The size is
arithmetic and the deadline is published months in advance.

WHICH IS WHY THE ADV RATIO IS THE WHOLE ANSWER. $2.8bn is nothing against a
name trading $2bn a day and is a crisis against one trading $450m. The
second is six days of the entire market's volume that must clear in one
auction, and the price impact of that is what index-inclusion trades are
about. So this reports flow as a MULTIPLE OF DAILY VOLUME first and in
currency second.

WHAT IT DELIBERATELY DOES NOT DO. It does not predict the price move. That
depends on how much of the passive money is truly passive, how much
front-running has already happened, and how deep the closing auction is --
none of which is in the weights. Sizing the flow honestly and leaving the
price alone is more useful than a number with three unstated assumptions
inside it.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from standard_quant_tools.error import ValidationError

from ._numbers import bounded, finite, non_negative, positive

__all__ = ["index_rebalance_flow"]


def index_rebalance_flow(
    *,
    old_weights: Mapping[str, float],
    new_weights: Mapping[str, float],
    indexed_assets: float,
    adv: Optional[Mapping[str, float]] = None,
    auction_fraction: float = 1.0,
) -> Dict[str, Any]:
    """
    The buying and selling an index change forces, name by name.

    `indexed_assets` is the money tracking the index, in currency. That is
    the multiplier on every weight change and the number this whole
    calculation is most sensitive to -- it is also the least knowable, since
    published tracking figures exclude closet indexers who trade the same
    way. Treat the output as a lower bound.

    `auction_fraction` is the share of a name's daily volume that prints in
    the closing auction, where index trades are conventionally executed.
    Default 1.0 measures flow against the whole day, which understates the
    problem: if only 15% of volume is in the auction, pass 0.15 and the
    participation figures become the ones that actually bind.
    """
    aum = positive(indexed_assets, "indexed_assets")
    fraction = finite(auction_fraction, "auction_fraction")
    if not 0 < fraction <= 1:
        raise ValidationError(
            f"auction_fraction={auction_fraction!r} must be in (0, 1]. It is "
            "the share of daily volume printing in the auction, not a "
            "multiple of it."
        )

    old = _weights(old_weights, "old_weights")
    new = _weights(new_weights, "new_weights")
    universe = sorted(set(old) | set(new))

    rows: List[Dict[str, Any]] = []
    for symbol in universe:
        w_old = old.get(symbol, 0.0)
        w_new = new.get(symbol, 0.0)
        delta = w_new - w_old
        if abs(delta) < 1e-12:
            continue

        notional = delta * aum
        if symbol not in old:
            event = "addition"
        elif symbol not in new:
            event = "deletion"
        else:
            event = "increase" if delta > 0 else "decrease"

        row: Dict[str, Any] = {
            "symbol": symbol,
            "event": event,
            "old_weight": w_old,
            "new_weight": w_new,
            "weight_change": float(delta),
            "side": "buy" if delta > 0 else "sell",
            "notional": float(notional),
            "adv": None,
            "days_of_adv": None,
            "auction_participation": None,
        }
        if adv is not None and symbol in adv:
            volume = finite(adv[symbol], f"adv[{symbol!r}]")
            if volume > 0:
                row["adv"] = volume
                row["days_of_adv"] = float(abs(notional) / volume)
                row["auction_participation"] = float(
                    abs(notional) / (volume * fraction)
                )
        rows.append(row)

    buys = [r for r in rows if r["side"] == "buy"]
    sells = [r for r in rows if r["side"] == "sell"]
    buy_notional = float(sum(r["notional"] for r in buys))
    sell_notional = float(sum(-r["notional"] for r in sells))
    # One-way turnover: the fraction of the index that changes hands, which
    # is the figure index providers quote and is half the sum of absolute
    # weight changes.
    turnover_pct = float(sum(abs(r["weight_change"]) for r in rows) / 2.0 * 100.0)

    ranked = sorted(rows, key=lambda r: abs(r["notional"]), reverse=True)
    by_pressure = sorted(
        (r for r in rows if r["days_of_adv"] is not None),
        key=lambda r: r["days_of_adv"],
        reverse=True,
    )

    warnings: List[str] = []
    missing_adv = [r["symbol"] for r in rows if r["adv"] is None]
    if missing_adv:
        warnings.append(
            f"{len(missing_adv)} name(s) have no ADV, so their flow is sized "
            f"in currency but not in days of volume: {missing_adv[:5]}. "
            "Currency alone does not say whether a trade is difficult -- "
            "$2.8bn is nothing in one name and a week of volume in another."
        )
    if by_pressure and by_pressure[0]["days_of_adv"] > 1.0:
        worst = by_pressure[0]
        warnings.append(
            f"{worst['symbol']!r} needs {worst['days_of_adv']:.1f} days of "
            f"its entire average volume ({worst['side']} "
            f"{abs(worst['notional']):,.0f}). Index flow conventionally "
            "executes in one closing auction, so anything above about 0.2 "
            "days is a trade that cannot clear at the print."
        )
    if fraction == 1.0 and by_pressure:
        warnings.append(
            "auction_participation is measured against the FULL day's "
            "volume because auction_fraction was left at 1.0. Auctions are "
            "typically 10-20% of the day, so the binding participation "
            "figures are roughly five to ten times those below."
        )
    if abs(buy_notional - sell_notional) / max(buy_notional, sell_notional, 1.0) > 0.05:
        warnings.append(
            f"Buys ({buy_notional:,.0f}) and sells ({sell_notional:,.0f}) do "
            "not offset. For a fully-invested tracker they should net to "
            "about zero -- a large imbalance means the weights supplied are "
            "not two states of the same index, or one set does not sum to 1."
        )
    warnings.append(
        "Flow only. This deliberately does not predict a price move: that "
        "depends on how much of the indexed money is genuinely passive, how "
        "much has already been front-run, and how deep the auction is, none "
        "of which is in a weight."
    )

    return {
        "indexed_assets": aum,
        "auction_fraction": fraction,
        "n_changes": len(rows),
        "n_additions": sum(1 for r in rows if r["event"] == "addition"),
        "n_deletions": sum(1 for r in rows if r["event"] == "deletion"),
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "net_notional": float(buy_notional - sell_notional),
        "gross_notional": float(buy_notional + sell_notional),
        "turnover_pct": turnover_pct,
        "changes": ranked,
        "hardest_to_trade": [r["symbol"] for r in by_pressure[:5]],
        "largest_flow": ranked[0]["symbol"] if ranked else None,
        "warnings": warnings,
    }


# ── internals ───────────────────────────────────────────────────────────


def _weights(mapping: Mapping[str, float], name: str) -> Dict[str, float]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValidationError(
            f"{name} must be a non-empty mapping of symbol to weight."
        )
    out: Dict[str, float] = {}
    for symbol, value in mapping.items():
        out[str(symbol)] = finite(value, f"{name}[{symbol!r}]")
    total = sum(out.values())
    if total <= 0:
        raise ValidationError(
            f"{name} sums to {total:g}, which is not an index. Weights are "
            "expected to sum to about 1."
        )
    return out
