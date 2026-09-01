"""
Five ways to hold the same exposure, priced on one basis.

THE QUESTION THIS ANSWERS. "I want $100m of SPX for six months" has at
least five answers -- the cash basket, SPY, ES futures, a synthetic forward
from options, a total return swap -- and they are economically equivalent
in payoff and nowhere near equivalent in cost. Choosing between them means
holding financing, dividend treatment, borrow, expense ratio, roll
schedule, bid-ask, commission and capital requirement in mind at once, in
different units, and comparing. That is a deterministic calculation and it
should not be done from memory.

THE HORIZON IS WHY THE RANKING CHANGES. This is the part that makes the
comparison worth computing rather than eyeballing. Carry accrues per year;
execution is paid once. So a 2 bp round trip is 24 bp/yr over a month and
1 bp/yr over two years, and an instrument that is cheapest to hold is
routinely not the cheapest to hold BRIEFLY. Everything below is normalized
to annualized basis points over a stated horizon, and the same five
expressions reorder when that horizon changes -- which is the answer, not
an artefact.

THE SIGN CONVENTION, ONCE. Every rate is quoted as a COST TO THE HOLDER:
positive means it reduces your return. `dividend_yield` is the exception
and is a RECEIPT, subtracted from the total, because that is how everyone
quotes it. For a short position the whole thing flips -- you earn the
financing and pay the dividend -- and `direction` does that flip rather
than asking the caller to negate eight numbers by hand, which is where
sign errors come from.

WHAT THIS DOES NOT DO. It does not fetch a financing rate, an expense
ratio or a swap spread; there is no provider for any of them and a made-up
one would produce a confident ranking of fictional costs. Every number
comes in as an argument. What the library contributes is the
normalization, the amortization and the comparison -- which is the part
that is easy to get wrong and impossible to check by eye.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

from standard_quant_tools.error import ValidationError

__all__ = ["EXPRESSION_KINDS", "compare_expressions"]

#: The instrument families this normalizes across, and the cost that
#: usually decides each one. The notes are not decoration -- they name the
#: term a caller most often forgets to supply for that kind, and a missing
#: term is silently zero.
EXPRESSION_KINDS: Dict[str, str] = {
    "cash": (
        "The underlying itself, or a basket of it. Fully funded, so the "
        "capital requirement is 100% and the financing rate is the "
        "opportunity cost of that cash. Receives dividends gross."
    ),
    "etf": (
        "A fund tracking the exposure. The term people forget is the "
        "expense ratio, which is small per year and is the whole difference "
        "over a long hold. Dividends arrive net of the fund's withholding."
    ),
    "future": (
        "Financing is EMBEDDED in the price rather than paid separately, so "
        "the term to supply is the basis over fair, not a funding rate. "
        "Capital is margin, and the unposted balance earns interest. Rolls."
    ),
    "forward": (
        "An OTC bilateral forward. Like a future without the roll or the "
        "exchange, and with counterparty credit priced into the rate."
    ),
    "synthetic": (
        "Long call plus short put at the same strike. Financing and "
        "dividend are implied by put-call parity rather than quoted, so "
        "solve_carry on the parity forward is what produces the rate here."
    ),
    "swap": (
        "A total return swap. Financing is an explicit reference rate plus "
        "a spread, and the spread is the negotiated part. No roll, minimal "
        "execution, and the capital requirement is whatever was posted."
    ),
}

_RATE_FIELDS = ("financing_rate", "borrow_rate", "fee_rate", "dividend_yield")
_EXECUTION_FIELDS = ("spread_bps", "commission_bps", "impact_bps")


def compare_expressions(
    expressions: Sequence[Mapping[str, Any]],
    *,
    notional: float,
    horizon_years: float,
    direction: str = "long",
) -> Dict[str, Any]:
    """
    Normalize several ways of holding one exposure onto annualized bps.

    Each expression is a mapping carrying a `label`, a `kind` from
    `EXPRESSION_KINDS`, and whichever cost terms apply -- all optional, all
    defaulting to zero. That default is the one real hazard here: an
    omitted expense ratio does not fail, it makes the ETF look free. The
    result therefore reports `terms_supplied` per expression so a
    suspiciously cheap row can be checked against what was actually priced.

    Execution costs are ROUND TRIP. A position is entered and exited, and
    comparing a one-way cost against a full year of carry understates every
    short-horizon holding by half its execution.
    """
    if direction not in ("long", "short"):
        raise ValidationError(
            f"direction={direction!r} must be 'long' or 'short'. It flips "
            "every carry sign, so it is not defaulted silently."
        )
    n = _number(notional, "notional", positive=True)
    horizon = _number(horizon_years, "horizon_years", positive=True)
    if not expressions:
        raise ValidationError(
            "expressions is empty; there is nothing to compare. Two or more "
            "is the point, but one is priced without complaint."
        )

    sign = 1.0 if direction == "long" else -1.0
    rows: List[Dict[str, Any]] = []

    for index, item in enumerate(expressions):
        if not isinstance(item, Mapping):
            raise ValidationError(
                f"expressions[{index}] is {type(item).__name__}, not a mapping."
            )
        label = str(item.get("label") or f"expression_{index}")
        kind = str(item.get("kind") or "cash")
        if kind not in EXPRESSION_KINDS:
            raise ValidationError(
                f"{label!r}: kind={kind!r} is not one of "
                f"{sorted(EXPRESSION_KINDS)}."
            )

        unknown = set(item) - {
            "label",
            "kind",
            "capital_requirement_pct",
            "rolls_per_year",
            "roll_cost_bps",
            *_RATE_FIELDS,
            *_EXECUTION_FIELDS,
        }
        if unknown:
            raise ValidationError(
                f"{label!r}: unknown fields {sorted(unknown)}. A silently "
                "ignored cost term would make this expression look cheaper "
                "than it is, which is the one error this tool exists to "
                "prevent."
            )

        supplied = sorted(k for k in item if k not in ("label", "kind"))

        financing = _number(item.get("financing_rate", 0.0), f"{label}.financing_rate")
        borrow = _number(item.get("borrow_rate", 0.0), f"{label}.borrow_rate")
        fee = _number(item.get("fee_rate", 0.0), f"{label}.fee_rate")
        dividend = _number(item.get("dividend_yield", 0.0), f"{label}.dividend_yield")

        # Costs positive, dividend a receipt. `sign` flips the whole carry
        # for a short, where you earn the funding and owe the dividend.
        carry_rate = sign * (financing + borrow + fee - dividend)
        carry_bps = carry_rate * 10_000.0

        one_way = sum(
            _number(item.get(field, 0.0), f"{label}.{field}", non_negative=True)
            for field in _EXECUTION_FIELDS
        )
        round_trip_bps = 2.0 * one_way
        # Amortized, which is the whole reason horizon is an argument.
        execution_bps = round_trip_bps / horizon

        rolls = _number(
            item.get("rolls_per_year", 0.0),
            f"{label}.rolls_per_year",
            non_negative=True,
        )
        roll_cost = _number(item.get("roll_cost_bps", 0.0), f"{label}.roll_cost_bps")
        roll_bps = rolls * roll_cost

        total_bps = carry_bps + execution_bps + roll_bps
        capital_pct = _number(
            item.get("capital_requirement_pct", 1.0),
            f"{label}.capital_requirement_pct",
            non_negative=True,
        )

        rows.append(
            {
                "label": label,
                "kind": kind,
                "carry_bps": float(carry_bps),
                "financing_bps": float(sign * financing * 10_000.0),
                "borrow_bps": float(sign * borrow * 10_000.0),
                "fee_bps": float(sign * fee * 10_000.0),
                "dividend_bps": float(-sign * dividend * 10_000.0),
                "execution_bps": float(execution_bps),
                "execution_round_trip_bps": float(round_trip_bps),
                "roll_bps": float(roll_bps),
                "total_annualized_bps": float(total_bps),
                "cost_over_horizon_bps": float(total_bps * horizon),
                "cost_over_horizon_currency": float(total_bps * horizon / 10_000.0 * n),
                "capital_requirement_pct": float(capital_pct),
                "capital_required": float(capital_pct * n),
                "terms_supplied": supplied,
            }
        )

    ranked = sorted(rows, key=lambda row: row["total_annualized_bps"])
    cheapest, dearest = ranked[0], ranked[-1]
    spread_bps = dearest["total_annualized_bps"] - cheapest["total_annualized_bps"]

    warnings: List[str] = []
    thin = [row["label"] for row in rows if len(row["terms_supplied"]) <= 1]
    if thin:
        warnings.append(
            f"{thin} were priced on one term or none, so they are close to "
            "free by construction rather than by economics. Every omitted "
            "cost defaults to zero -- check `terms_supplied` on any row that "
            "wins by a surprising margin."
        )
    if len(rows) > 1 and abs(spread_bps) < 5.0:
        warnings.append(
            f"The cheapest and dearest are {spread_bps:.1f} bps apart, which "
            "is inside the error of any of these inputs. Treat them as tied "
            "and choose on something this does not price -- operational "
            "burden, counterparty, tracking risk."
        )
    horizon_sensitive = [
        row["label"]
        for row in rows
        if row["execution_bps"] > 0.5 * abs(row["total_annualized_bps"])
    ]
    if horizon_sensitive:
        warnings.append(
            f"For {horizon_sensitive}, amortized execution is more than half "
            f"the total at a {horizon:.2f}-year horizon. That ranking is a "
            "statement about the holding period, not the instrument -- "
            "re-run with the horizon you will actually hold."
        )
    if any(row["kind"] == "future" for row in rows) and not any(
        row["kind"] == "future" and row["roll_bps"] for row in rows
    ):
        warnings.append(
            "A futures expression was priced with no roll cost. Over any "
            "horizon longer than one contract that is understated -- the "
            "roll is usually the largest single cost of holding a future."
        )
    warnings.append(
        "Costs only. These expressions are economically equivalent in "
        "PAYOFF and not in risk: a swap carries counterparty exposure, a "
        "future carries margin calls and basis risk, an ETF carries "
        "tracking error, and none of that is a basis point."
    )

    return {
        "direction": direction,
        "notional": n,
        "horizon_years": horizon,
        "n_expressions": len(rows),
        "expressions": ranked,
        "cheapest": cheapest["label"],
        "cheapest_total_bps": cheapest["total_annualized_bps"],
        "dearest": dearest["label"],
        "dearest_total_bps": dearest["total_annualized_bps"],
        "spread_bps": float(spread_bps),
        "spread_currency_over_horizon": float(spread_bps * horizon / 10_000.0 * n),
        "warnings": warnings,
    }


# ── internals ───────────────────────────────────────────────────────────


def _number(
    value: Any, name: str, *, positive: bool = False, non_negative: bool = False
) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    if positive and out <= 0:
        raise ValidationError(f"{name} must be positive, got {value!r}")
    if non_negative and out < 0:
        raise ValidationError(
            f"{name} must not be negative, got {value!r}. Execution and roll "
            "costs are quoted as costs; a rebate belongs in the rate terms."
        )
    return out
