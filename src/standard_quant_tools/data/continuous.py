"""
Stitching a futures curve into one series, and being honest about what that
series is.

TWO OUTPUTS, NOT ONE, AND THIS IS THE WHOLE POINT. A back-adjusted
continuous future is a research instrument. It is not a price. Adjusting
the history so the roll leaves no gap changes every historical level -- a
difference-adjusted series can go NEGATIVE on a contract that never traded
below zero -- so a backtest that computes `shares = capital / price` on one
is sizing against a number nobody could have transacted at, and a stop
placed at a historical level is placed at a level that did not exist.

So this returns both:

    research_series        one continuous series, adjusted, for indicators
    tradeable_contract_map which ACTUAL contract each date belongs to, and
                           what it actually traded at

Anything computing a signal reads the first. Anything computing a position
size, a cost, or a fill reads the second. Collapsing them into one series
is the error this module exists to make impossible, and it is the reason
`Documentation/19_runtimes.md` treats temporal integrity as a contract
rather than a convention.

THE ROLL RULE CHANGES THE ANSWER. Rolling on volume, on open interest, or a
fixed number of days before expiry produce different series from the same
contracts, and the differences are largest exactly where they matter -- a
volume roll moves late in a quiet expiry and early in a busy one. It is a
required argument for that reason.

WHAT ADJUSTMENT DOES TO A RETURN. Difference adjustment preserves the
absolute point change across a roll and distorts percentage returns.
Ratio adjustment preserves percentage returns and distorts point changes.
Neither is right in general: use difference for anything measured in
points, ratio for anything compounding. `none` leaves the jump in, which is
correct only if you are modelling the jump.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

__all__ = ["ADJUSTMENTS", "ROLL_RULES", "build_continuous_futures"]

#: How the active contract is chosen on each date, and what each rule is
#: sensitive to. None dominates: they disagree most in exactly the expiries
#: where liquidity moved unusually, which is where a backtest is most
#: likely to be lying to itself.
ROLL_RULES: Dict[str, str] = {
    "volume": (
        "Switch when the next contract's volume first exceeds the front's. "
        "Follows where the liquidity actually is, and moves late in a quiet "
        "expiry and early in a busy one."
    ),
    "open_interest": (
        "Switch when the next contract's open interest first exceeds the "
        "front's. Slower and steadier than volume -- open interest reflects "
        "positioning rather than turnover, so it is less jumpy on one busy "
        "day."
    ),
    "days_before_expiry": (
        "Switch a fixed number of calendar days before the front expires. "
        "Deterministic and knowable in advance, which is its whole "
        "advantage: it never depends on data published after the decision."
    ),
}

#: What the adjustment preserves, and what it therefore destroys.
ADJUSTMENTS: Dict[str, str] = {
    "none": (
        "Leave the roll gap in. The series is then a sequence of real prices "
        "with jumps at every roll, and a return computed across one is the "
        "contract change rather than a market move."
    ),
    "difference": (
        "Shift history so point changes are continuous. Preserves absolute "
        "moves and distorts percentage returns; can drive early history "
        "negative on a contract that never traded below zero."
    ),
    "ratio": (
        "Scale history so percentage returns are continuous. Preserves "
        "compounding and distorts point changes; cannot go negative."
    ),
}


def build_continuous_futures(
    contracts: Sequence[Mapping[str, Any]],
    *,
    roll_rule: str = "volume",
    adjustment: str = "ratio",
    days_before_expiry: int = 5,
) -> Dict[str, Any]:
    """
    One continuous series from a chain of contracts, plus what was tradeable.

    Each contract is a mapping with a `symbol`, an `expiry`, and a `prices`
    mapping of date to close. `volume` and `open_interest` mappings are
    required by the roll rules that read them -- a rule cannot be evaluated
    from data that was not supplied, and defaulting to the front contract
    would silently produce a series that never rolls.

    The returned `research_series` is adjusted. The returned
    `tradeable_contract_map` is not: it carries, per date, the contract that
    was active and the price it actually traded at.
    """
    if roll_rule not in ROLL_RULES:
        raise ValidationError(
            f"roll_rule={roll_rule!r} must be one of {sorted(ROLL_RULES)}. "
            "The three produce different series from the same contracts, so "
            "there is no safe default to fall back on."
        )
    if adjustment not in ADJUSTMENTS:
        raise ValidationError(
            f"adjustment={adjustment!r} must be one of {sorted(ADJUSTMENTS)}."
        )

    rows = _parse(contracts, roll_rule)
    if len(rows) < 2:
        raise ValidationError(
            f"a continuous series needs at least two contracts, got "
            f"{len(rows)}. One contract is already continuous."
        )

    warnings: List[str] = []
    active = _choose_active(rows, roll_rule, days_before_expiry, warnings)
    if active.empty:
        raise ValidationError(
            "no date had an active contract. Check that the contracts' price "
            "dates and expiries overlap."
        )

    # The raw, tradeable series: what the active contract actually printed.
    raw = pd.Series(
        [rows[symbol]["prices"].get(date, np.nan) for date, symbol in active.items()],
        index=active.index,
        dtype="float64",
        name="price",
    )
    if raw.isna().any():
        missing = int(raw.isna().sum())
        raise ValidationError(
            f"{missing} date(s) selected a contract with no price on that "
            "date. The roll chose a contract that was not trading, which "
            "means the volume or open-interest series disagrees with the "
            "price series."
        )

    rolls = _roll_dates(active)
    adjusted = _adjust(raw, active, rolls, rows, adjustment, warnings)

    if adjustment == "difference" and (adjusted <= 0).any():
        warnings.append(
            f"Difference adjustment drove {int((adjusted <= 0).sum())} "
            "historical value(s) to zero or below, on a contract that never "
            "traded there. That is the adjustment working as defined, not a "
            "bug -- and it is exactly why this series must not be used to "
            "size a position. Use tradeable_contract_map for that."
        )
    if adjustment == "none" and rolls:
        gaps = [
            abs(
                raw.loc[date]
                - rows[active.loc[:date].iloc[-2]]["prices"].get(date, raw.loc[date])
            )
            for date in rolls
            if date in raw.index
        ]
        if gaps:
            warnings.append(
                f"Unadjusted, so the series carries {len(rolls)} roll "
                "jump(s). A return computed across one measures the contract "
                "change rather than a market move."
            )

    warnings.append(
        "research_series is ADJUSTED and is not a price. Size positions, "
        "compute costs and place stops from tradeable_contract_map, which "
        "carries the contract that was actually active on each date and "
        "what it actually traded at."
    )
    warnings.append(
        f"Rolled on {roll_rule}: {ROLL_RULES[roll_rule]} A different rule "
        "produces a different series from the same contracts, and they "
        "disagree most in the expiries where liquidity moved unusually."
    )

    return {
        "roll_rule": roll_rule,
        "adjustment": adjustment,
        "n_contracts": len(rows),
        "n_observations": int(len(raw)),
        "start": str(active.index[0]),
        "end": str(active.index[-1]),
        "n_rolls": len(rolls),
        "roll_dates": [str(date) for date in rolls],
        "research_series": {str(k): float(v) for k, v in adjusted.items()},
        "tradeable_contract_map": {
            str(date): {"symbol": symbol, "price": float(raw.loc[date])}
            for date, symbol in active.items()
        },
        "contracts_used": sorted(set(active.to_list())),
        "warnings": warnings,
    }


# ── internals ───────────────────────────────────────────────────────────


def _parse(
    contracts: Sequence[Mapping[str, Any]], roll_rule: str
) -> Dict[str, Dict[str, Any]]:
    """Validate the chain, refusing the series a rule cannot be run on."""
    if not contracts:
        raise ValidationError("contracts is empty.")
    needs = {"volume": "volume", "open_interest": "open_interest"}.get(roll_rule)

    rows: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(contracts):
        if not isinstance(item, Mapping):
            raise ValidationError(
                f"contracts[{index}] is {type(item).__name__}, not a mapping."
            )
        symbol = str(item.get("symbol") or f"contract_{index}")
        if symbol in rows:
            raise ValidationError(f"contract {symbol!r} appears twice.")
        for field in ("expiry", "prices"):
            if field not in item:
                raise ValidationError(f"contract {symbol!r} has no `{field}`.")
        if needs and needs not in item:
            raise ValidationError(
                f"contract {symbol!r} has no `{needs}`, which roll_rule="
                f"{roll_rule!r} reads. A rule cannot be evaluated from data "
                "that was not supplied, and falling back to the front "
                "contract would produce a series that never rolls."
            )
        prices = {pd.Timestamp(k): float(v) for k, v in item["prices"].items()}
        if not prices:
            raise ValidationError(f"contract {symbol!r} has no prices.")
        rows[symbol] = {
            "symbol": symbol,
            "expiry": pd.Timestamp(item["expiry"]),
            "prices": prices,
            "volume": {
                pd.Timestamp(k): float(v) for k, v in (item.get("volume") or {}).items()
            },
            "open_interest": {
                pd.Timestamp(k): float(v)
                for k, v in (item.get("open_interest") or {}).items()
            },
        }
    return rows


def _choose_active(
    rows: Dict[str, Dict[str, Any]],
    roll_rule: str,
    days_before_expiry: int,
    warnings: List[str],
) -> pd.Series:
    """The active contract on each date, as a Series of symbol."""
    order = sorted(rows, key=lambda s: rows[s]["expiry"])
    dates = sorted({date for row in rows.values() for date in row["prices"]})

    chosen: List[Tuple[pd.Timestamp, str]] = []
    # Monotone: once rolled forward, never back. A rule that oscillates on a
    # noisy volume print would otherwise produce a series that rolls in and
    # out of the same contract, and every one of those transitions would be
    # charged as a real roll by anything downstream.
    floor = 0
    for date in dates:
        index = floor
        while index < len(order) - 1:
            front, nxt = rows[order[index]], rows[order[index + 1]]
            if date > front["expiry"]:
                index += 1
                continue
            if roll_rule == "days_before_expiry":
                if (front["expiry"] - date).days <= days_before_expiry:
                    index += 1
                    continue
            else:
                field = "volume" if roll_rule == "volume" else "open_interest"
                front_value = front[field].get(date)
                next_value = nxt[field].get(date)
                if (
                    front_value is not None
                    and next_value is not None
                    and next_value > front_value
                ):
                    index += 1
                    continue
            break
        # Skip a date whose chosen contract has already expired and has no
        # successor -- the chain has run out rather than the date being bad.
        if date > rows[order[index]]["expiry"]:
            continue
        floor = index
        chosen.append((date, order[index]))

    if not chosen:
        return pd.Series(dtype=object)
    index = pd.DatetimeIndex([d for d, _ in chosen])
    return pd.Series([s for _, s in chosen], index=index, name="contract")


def _roll_dates(active: pd.Series) -> List[pd.Timestamp]:
    """The dates on which the active contract changed."""
    changed = active != active.shift(1)
    changed.iloc[0] = False
    return list(active.index[changed])


def _adjust(
    raw: pd.Series,
    active: pd.Series,
    rolls: List[pd.Timestamp],
    rows: Dict[str, Dict[str, Any]],
    adjustment: str,
    warnings: List[str],
) -> pd.Series:
    """
    Back-adjust the history so the roll leaves no artificial jump.

    Walked BACKWARD from the most recent roll, so the newest segment keeps
    its true prices and every older one is shifted. The alternative --
    adjusting forward from the oldest -- leaves today's price wrong, which
    is the half of the series anyone actually compares against a screen.
    """
    out = raw.astype(float).copy()
    if adjustment == "none" or not rolls:
        return out

    for date in reversed(rolls):
        position = out.index.get_loc(date)
        if position == 0:
            continue
        previous_symbol = active.iloc[position - 1]
        new_symbol = active.iloc[position]
        # Both contracts' prices on the ROLL DATE. The gap between them is
        # the artefact; anything else on that date is a real move.
        old_price = rows[previous_symbol]["prices"].get(date)
        new_price = rows[new_symbol]["prices"].get(date)
        if old_price is None or new_price is None or old_price <= 0:
            warnings.append(
                f"No overlapping price on {date.date()} for both "
                f"{previous_symbol!r} and {new_symbol!r}, so that roll could "
                "not be adjusted and its jump remains in the series."
            )
            continue
        if adjustment == "difference":
            out.iloc[:position] = out.iloc[:position] + (new_price - old_price)
        else:
            out.iloc[:position] = out.iloc[:position] * (new_price / old_price)
    return out
