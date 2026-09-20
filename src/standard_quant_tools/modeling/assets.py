"""
AssetKey: what an entity IS, beyond the string a provider resolves.

THE COLLISION. A universe was a list of symbols, and an entity was the
symbol. That is one string doing two jobs: the name a provider fetches
by, and the identity a panel row belongs to. The jobs come apart the
moment one symbol names two things -- `BHP` on the ASX and on the NYSE,
`ES` the future and `ES` a ticker somewhere else -- and when they come
apart silently, two instruments become one entity and every feature,
label and weight computed on it is computed on a blend. The review named
these cases; this module makes them spellable and refuses the blend.

THE KEY. `SYMBOL[@VENUE][~CLASS]`: the symbol a provider resolves, the
venue it trades on as an `exchange_calendars` code (`XNYS`, `XASX`,
`XCME`), and its asset class. `AAPL` and `AAPL~equity` are the same key;
`BHP.AX@XASX` and `BHP@XNYS` are two. The CANONICAL string is what the
panel's `entity` column carries and what a scoring universe names; the
SYMBOL is what reaches the provider. A venue shared by every key is the
dataset's calendar unless the spec names one, which is what makes an
intraday interval annualizable without a second field to keep in step.

WHAT IT DOES NOT DO. No shipped provider resolves a venue: it fetches
by symbol. So two keys that differ only by venue would fetch one series
twice and register it as two entities -- the collision this exists to
name -- and `fetch_plan` refuses that by name rather than making it.
Naming the venue is for the calendar and for keeping two instruments
apart in the panel; a venue-aware provider is what would make it a
routing decision, and that is a data problem rather than a modeling one.
"""

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict

from standard_quant_tools.error import ValidationError

AssetClass = Literal["equity", "etf", "index", "future", "fx", "crypto"]
ASSET_CLASSES = ("equity", "etf", "index", "future", "fx", "crypto")
#: A provider symbol as the shipped providers spell them: `BRK-B`, `BHP.AX`,
#: `^GSPC`, `ES=F`, `BTC-USD`. No spaces, and none of the two characters
#: the key grammar reserves.
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9^][A-Za-z0-9._=^-]*$")
#: An exchange code: upper case, the shape `exchange_calendars` uses.
_VENUE_RE = re.compile(r"^[A-Z0-9]{2,12}$")


class AssetKey(BaseModel):
    """One instrument: the symbol a provider resolves, where it trades,
    and what it is."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    venue: Optional[str] = None
    asset_class: AssetClass = "equity"

    @property
    def canonical(self) -> str:
        """`SYMBOL[@VENUE][~CLASS]`, the class omitted when it is equity."""
        text = self.symbol
        if self.venue:
            text += f"@{self.venue}"
        if self.asset_class != "equity":
            text += f"~{self.asset_class}"
        return text


def parse_asset_key(text: str) -> AssetKey:
    """`SYMBOL[@VENUE][~CLASS]` -> AssetKey, refused by name when malformed."""
    raw = str(text).strip()
    if not raw:
        raise ValidationError("an asset key cannot be empty")
    body, _, klass = raw.partition("~")
    symbol, _, venue = body.partition("@")
    if not _SYMBOL_RE.match(symbol):
        raise ValidationError(
            f"asset key {text!r}: {symbol!r} is not a symbol (letters and digits, "
            "with . _ - = ^ inside; no spaces; '@' and '~' are the key's own "
            "separators)."
        )
    if venue and not _VENUE_RE.match(venue):
        raise ValidationError(
            f"asset key {text!r}: venue {venue!r} should be an exchange code such "
            "as XNYS, XASX or XCME (upper case, as exchange_calendars names them)."
        )
    if klass and klass not in ASSET_CLASSES:
        raise ValidationError(
            f"asset key {text!r}: {klass!r} is not an asset class; one of "
            f"{list(ASSET_CLASSES)}."
        )
    return AssetKey(symbol=symbol, venue=venue or None, asset_class=klass or "equity")


def fetch_symbol(entity: str) -> str:
    """The string a provider is asked for."""
    return parse_asset_key(entity).symbol


def is_qualified(entity: str) -> bool:
    """Whether the string says more than a symbol. A string test, not a
    parse: an external panel may name entities however it likes, and
    this is asked of those names too."""
    text = str(entity)
    return "@" in text or "~" in text


def canonical_universe(universe: Sequence[str]) -> List[str]:
    return [parse_asset_key(entity).canonical for entity in universe]


def common_venue(universe: Sequence[str]) -> Optional[str]:
    """The one venue every key names, or None when any key names none or
    they disagree."""
    venues = {parse_asset_key(entity).venue for entity in universe}
    if len(venues) == 1:
        return venues.pop()
    return None


def fetch_plan(universe: Sequence[str]) -> Dict[str, str]:
    """
    entity -> the symbol fetched for it, in universe order.

    Two entities that resolve to one symbol are refused by name: no
    shipped provider resolves a venue, so the second fetch would return
    the first series under a second identity.
    """
    plan: Dict[str, str] = {}
    by_symbol: Dict[str, str] = {}
    for entity in universe:
        key = parse_asset_key(entity)
        if key.symbol in by_symbol and by_symbol[key.symbol] != key.canonical:
            raise ValidationError(
                f"{by_symbol[key.symbol]!r} and {key.canonical!r} both fetch as "
                f"{key.symbol!r}: the provider resolves a symbol without a venue, "
                "so they would be one series registered as two entities. Spell "
                "the venue-specific symbol the provider knows (e.g. 'BHP.AX@XASX' "
                "beside 'BHP@XNYS'), or keep one of them."
            )
        by_symbol[key.symbol] = key.canonical
        plan[key.canonical] = key.symbol
    return plan


__all__ = [
    "ASSET_CLASSES",
    "AssetClass",
    "AssetKey",
    "canonical_universe",
    "common_venue",
    "fetch_plan",
    "fetch_symbol",
    "is_qualified",
    "parse_asset_key",
]
