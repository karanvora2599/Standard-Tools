"""
A basket of names against the index it is supposed to be.

WHY THIS IS NOT PORTFOLIO OPTIMIZATION. The portfolio package asks what
weights are best -- maximum Sharpe, minimum variance, risk parity. This
asks something with a right answer already fixed: given these constituents
at these prices, what is the basket WORTH, and does that agree with where
the index is printing. There is nothing to optimize; there is a valuation
and a discrepancy, and the discrepancy is either an arbitrage or a stale
price, which is what the contributions are for.

TWO CONSTRUCTIONS, AND THEY ARE NOT INTERCHANGEABLE. A real index is
divisor-based: level = sum(shares * price) / divisor, where the divisor is
a maintained constant that absorbs corporate actions. A tracking basket is
weight-based: value = sum(weight * price relative), where weights sum to
one. Given a divisor this module uses the first, which reproduces the index
exactly; without one it uses the second, which reproduces its RETURNS but
not its level. Conflating them is how a basket comes out off by a constant
factor that nobody can find.

STALENESS IS THE COMMON ANSWER. A basket printing 40 bps away from its
index is far more often one constituent that has not traded than an
arbitrage across all of them. So the per-name contribution is returned
sorted, and a single name carrying most of the discrepancy is flagged --
because the fix for that is a fresher price, not a trade.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from standard_quant_tools.error import ValidationError

from ._numbers import bounded, finite, non_negative, positive

__all__ = ["index_basket"]


def index_basket(
    constituents: Sequence[Mapping[str, Any]],
    *,
    index_level: Optional[float] = None,
    divisor: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Value a basket and compare it to the index it replicates.

    Each constituent is a mapping with a `symbol`, a `price`, and either
    `shares` (with a `divisor`) or `weight` (without one). An optional
    `reference_price` enables the staleness check: it is the price the name
    was last known to trade at, and a constituent whose price has not moved
    while the rest of the basket has is the usual cause of a wide spread.

    With no `index_level` this is a valuation and nothing more. That is a
    legitimate call -- pricing a custom basket that has no index -- and the
    comparison fields come back None rather than zero, because zero would
    read as "the basket is exactly on top of its index".
    """
    rows = _parse_constituents(constituents, divisor=divisor)
    warnings: List[str] = []

    if divisor is not None:
        d = float(divisor)
        if not math.isfinite(d) or d <= 0:
            raise ValidationError(
                f"divisor={divisor!r} must be positive and finite. It is the "
                "maintained constant an index level is defined by; without a "
                "valid one, pass weights instead of shares."
            )
        market_cap = sum(row["shares"] * row["price"] for row in rows)
        basket_value = market_cap / d
        construction = "divisor"
        for row in rows:
            row["value"] = row["shares"] * row["price"]
            row["weight"] = row["value"] / market_cap if market_cap else float("nan")
            row["index_points"] = row["value"] / d
    else:
        total_weight = sum(row["weight"] for row in rows)
        if total_weight <= 0:
            raise ValidationError(
                "the constituent weights sum to zero or less, so the basket "
                "has no value. Weights are expected to sum to about 1."
            )
        basket_value = sum(row["weight"] * row["price"] for row in rows)
        construction = "weighted"
        for row in rows:
            row["value"] = row["weight"] * row["price"]
            row["index_points"] = row["value"]
        if abs(total_weight - 1.0) > 0.01:
            warnings.append(
                f"The weights sum to {total_weight:.4f} rather than 1. The "
                "basket value below is therefore scaled by that factor, and "
                "comparing it to an index level will be wrong by the same "
                "amount. Normalize the weights or pass shares and a divisor."
            )

    spread_points: Optional[float] = None
    spread_bps: Optional[float] = None
    if index_level is not None:
        level = float(index_level)
        if not math.isfinite(level) or level <= 0:
            raise ValidationError(
                f"index_level={index_level!r} must be positive and finite."
            )
        spread_points = basket_value - level
        spread_bps = (basket_value / level - 1.0) * 10_000.0

    # Sorted by absolute contribution so the name that is moving the basket
    # is first, which is the order somebody debugging a wide spread reads in.
    contributions = sorted(
        rows, key=lambda row: abs(row.get("index_points", 0.0)), reverse=True
    )

    stale = [row["symbol"] for row in rows if row.get("is_stale")]
    missing = [row["symbol"] for row in rows if row.get("price") is None]

    if stale:
        warnings.append(
            f"{len(stale)} constituent(s) have a price identical to their "
            f"reference price: {stale[:5]}. On a moving tape that usually "
            "means the print is stale rather than that the name is "
            "unchanged, and a stale constituent shows up as basket/index "
            "spread that no trade can capture."
        )
    if index_level is not None and spread_bps is not None and abs(spread_bps) > 25:
        top = contributions[0]
        share = (
            abs(top["index_points"]) / abs(basket_value) * 100.0
            if basket_value
            else 0.0
        )
        warnings.append(
            f"The basket is {spread_bps:+.0f} bps from the index. Check "
            f"{top['symbol']!r} first -- it carries {share:.1f}% of the "
            "basket value, and a single wrong or stale price on the largest "
            "weight explains most wide spreads."
        )
    warnings.append(
        "Basket and index prices must be observed at the SAME instant for "
        "this spread to mean anything. A basket priced on last trades "
        "against an index printed on a different clock produces a spread "
        "that is an artefact of the timestamps."
    )

    return {
        "construction": construction,
        "n_constituents": len(rows),
        "basket_value": float(basket_value),
        "index_level": float(index_level) if index_level is not None else None,
        "spread_points": spread_points,
        "spread_bps": spread_bps,
        "divisor": float(divisor) if divisor is not None else None,
        "constituents": [
            {
                "symbol": row["symbol"],
                "price": row["price"],
                "weight": float(row.get("weight", float("nan"))),
                "value": float(row["value"]),
                "index_points": float(row["index_points"]),
                "is_stale": bool(row.get("is_stale", False)),
            }
            for row in contributions
        ],
        "largest_contributor": contributions[0]["symbol"] if contributions else None,
        "stale_symbols": stale,
        "missing_symbols": missing,
        "warnings": warnings,
    }


# ── internals ───────────────────────────────────────────────────────────


def _parse_constituents(
    constituents: Sequence[Mapping[str, Any]], *, divisor: Optional[float]
) -> List[Dict[str, Any]]:
    """Validate a constituent list, naming the name that is wrong."""
    if not constituents:
        raise ValidationError("constituents is empty; a basket needs at least one.")

    need = "shares" if divisor is not None else "weight"
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(constituents):
        if not isinstance(item, Mapping):
            raise ValidationError(
                f"constituents[{index}] is {type(item).__name__}, not a "
                f"mapping with `symbol`, `price` and `{need}`."
            )
        symbol = str(item.get("symbol") or f"constituent_{index}")
        if "price" not in item:
            raise ValidationError(f"constituent {symbol!r} has no `price`.")
        price = positive(item["price"], f"{symbol}.price")

        if need not in item:
            raise ValidationError(
                f"constituent {symbol!r} has no `{need}`. A divisor was "
                f"{'given' if divisor is not None else 'not given'}, so this "
                f"basket is {'share' if divisor is not None else 'weight'}-"
                "based and every constituent needs that field."
            )
        quantity = finite(item[need], f"{symbol}.{need}")

        reference = item.get("reference_price")
        is_stale = (
            reference is not None
            and math.isfinite(float(reference))
            and float(reference) == price
        )

        row: Dict[str, Any] = {"symbol": symbol, "price": price, "is_stale": is_stale}
        row[need] = quantity
        rows.append(row)
    return rows
