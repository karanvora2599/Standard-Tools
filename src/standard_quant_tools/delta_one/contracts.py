"""
What a futures contract IS, as opposed to what it costs.

WHY THIS IS NOT A TOOL. The rest of this library thinks in ticker, price,
shares and weight, and for cash equity that is complete -- a share of AAPL
at $190 is $190 of exposure and there is nothing else to know. A futures
contract is not self-describing in the same way. `ES at 6200` is $310,000
of exposure only because the multiplier is 50, and nothing in the number
6200 says so. Every Delta One calculation that turns a price into money
needs that missing fact, so it has to live somewhere, and a dataclass is
the right shape for it rather than an agent tool: an agent asking "what is
the multiplier for ES" wants a fact this library has no source for.

NO SHIPPED REGISTRY, ON PURPOSE. It would be easy to hard-code that ES is
50 and NQ is 20 and SX5E is 10, and it would be wrong within a year --
exchanges relist, split and redenominate contracts, and a stale multiplier
does not fail, it prices the position off by a constant factor and every
downstream number stays plausible. The caller passes the specification.
That is the same call the derivatives runtime made about option chains and
it has the same justification: a fact this library cannot verify should
not be a fact this library asserts.

THE INVARIANT WORTH KNOWING. `tick_value = multiplier * tick_size`, and it
is derived rather than stored. The two are quoted independently on every
exchange fact sheet and a caller who passes both usually has one of them
from a different contract; deriving it means they cannot disagree.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from standard_quant_tools.error import ValidationError

from .daycount import DateLike, to_date, year_fraction

__all__ = ["SETTLEMENT_TYPES", "ContractSpec"]

#: How a contract stops existing. It changes what the last day means, not
#: merely how it is booked: a cash-settled index future ends at a printed
#: settlement level and a physically-settled one obliges delivery, so a
#: position held to expiry has completely different consequences.
SETTLEMENT_TYPES = ("cash", "physical")


@dataclass(frozen=True)
class ContractSpec:
    """
    One futures contract's identity and unit economics.

    Frozen because a specification that changes under a calculation is a
    class of bug nobody debugs successfully -- half the numbers computed
    before the mutation and half after, with no error.
    """

    symbol: str
    multiplier: float
    expiry: Optional[_dt.date] = None
    root: Optional[str] = None
    tick_size: Optional[float] = None
    currency: str = "USD"
    settlement: str = "cash"
    exchange: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValidationError(
                f"symbol must be a non-empty string, got {self.symbol!r}."
            )
        _positive(self.multiplier, "multiplier")
        if self.tick_size is not None:
            _positive(self.tick_size, "tick_size")
        if self.settlement not in SETTLEMENT_TYPES:
            raise ValidationError(
                f"settlement={self.settlement!r} is not one of "
                f"{list(SETTLEMENT_TYPES)}. A cash-settled contract ends at "
                "a printed level and a physical one obliges delivery; the "
                "difference decides what holding to expiry means, so it is "
                "not defaulted silently."
            )
        if self.expiry is not None and not isinstance(self.expiry, _dt.date):
            # Normalizing in a frozen dataclass needs object.__setattr__.
            object.__setattr__(self, "expiry", to_date(self.expiry, "expiry"))

    # ── derived economics ───────────────────────────────────────────────

    @property
    def tick_value(self) -> Optional[float]:
        """
        Currency per tick. Derived, never stored -- see the module note.

        `None` when no tick size was given, which is honest: a caller who
        did not supply one cannot be told what a tick is worth, and a
        plausible-looking 0.0 would be read as a free contract.
        """
        if self.tick_size is None:
            return None
        return float(self.multiplier) * float(self.tick_size)

    def notional(self, price: float) -> float:
        """The money one contract represents at `price`."""
        return _finite(price, "price") * float(self.multiplier)

    def contracts_for(self, exposure: float, price: float) -> float:
        """
        How many contracts express `exposure` of money at `price`.

        UNROUNDED, deliberately. Rounding is a decision with a residual
        attached and it belongs to the caller who can report that residual;
        a function that quietly returned an integer here would hide the one
        number that says whether the hedge is finished.
        """
        return _finite(exposure, "exposure") / self.notional(price)

    def time_to_expiry(self, as_of: DateLike, *, convention: str = "ACT/365F") -> float:
        """
        Years from `as_of` to expiry, or a refusal if there is no expiry.

        Refuses rather than returning `None` because every caller of this
        multiplies the result by a rate, and `None` would surface as a
        TypeError inside an arithmetic expression three frames away.
        """
        if self.expiry is None:
            raise ValidationError(
                f"{self.symbol}: no expiry was given, so time to expiry "
                "cannot be computed. Pass `expiry` on the ContractSpec, or "
                "pass a time to expiry in years directly."
            )
        return year_fraction(as_of, self.expiry, convention=convention)

    # ── interop ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-shaped view, with the derived fields resolved."""
        return {
            "symbol": self.symbol,
            "root": self.root,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "multiplier": float(self.multiplier),
            "tick_size": float(self.tick_size) if self.tick_size else None,
            "tick_value": self.tick_value,
            "currency": self.currency,
            "settlement": self.settlement,
            "exchange": self.exchange,
        }

    @classmethod
    def from_mapping(cls, data: Dict[str, Any]) -> "ContractSpec":
        """
        Build from a plain dict, ignoring a `tick_value` if one is present.

        Ignored rather than refused because exchange fact sheets and vendor
        payloads carry it as a matter of course, and refusing would make
        the common case of pasting one in an error. It is derived from the
        other two; a supplied one that disagreed would have to lose anyway.
        """
        known = {
            "symbol",
            "multiplier",
            "expiry",
            "root",
            "tick_size",
            "currency",
            "settlement",
            "exchange",
        }
        unknown = set(data) - known - {"tick_value"}
        if unknown:
            raise ValidationError(
                f"unknown contract fields: {sorted(unknown)}. "
                f"A ContractSpec carries {sorted(known)}."
            )
        return cls(**{k: v for k, v in data.items() if k in known})


# ── internals ───────────────────────────────────────────────────────────


def _positive(value: Any, name: str) -> float:
    if value is None:
        raise ValidationError(f"{name} is required and was not given")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ValidationError(f"{name} must be positive and finite, got {value!r}")
    return value


def _finite(value: Any, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    return value
