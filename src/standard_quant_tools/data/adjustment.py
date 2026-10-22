"""
What a bar frame says about its own adjustment, and what a consumer that
computes returns from several frames should say about theirs.

A provider stamps every OHLCV frame it serves with the two flags its
`get_metadata` states -- `split_adjusted` and `dividend_adjusted` -- and
the `return_basis` they imply, in `DataFrame.attrs`. pandas carries attrs
through slicing, column selection and Parquet, so a frame that crossed a
reference boundary still says what it is.

WHY A CONSUMER ECHOES IT. A total-return series and a price-return series
of the same stock differ by its dividend yield every year, and nothing in
the numbers says which one a backtest compounded. Two inputs on different
bases -- one name from a total-return feed and one from a split-only feed,
or a published reference beside a fresh fetch -- make a comparison between
them a comparison of dividend policy. So the consumers that compute
returns from bars (the backtest engines and the dataset build) report the
basis they ran on and say so when their inputs disagree.

The split flag is read separately, because the split screen asks a
different question: whether a -90% bar is a split. A frame stamped only
with the older `adjusted` attribute (Databento's, which is False: venue
prices, neither adjustment) is read at that word for both.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from standard_quant_tools.data.metadata import (
    PRICE_RETURN,
    RETURN_BASES,
    TOTAL_RETURN,
    UNKNOWN_RETURN_BASIS,
    return_basis_for,
)

#: The basis a consumer reports when its inputs disagree: some returns
#: carry dividends and some do not.
MIXED_RETURN_BASIS = "mixed"

SPLIT_KEY = "split_adjusted"
DIVIDEND_KEY = "dividend_adjusted"
BASIS_KEY = "return_basis"
LEGACY_KEY = "adjusted"

_DESCRIBED = {
    TOTAL_RETURN: "total_return (dividends in the return)",
    PRICE_RETURN: "price_return (dividends left out)",
}


def _flag(value: Any) -> Optional[bool]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _attrs(frame: Any) -> Mapping[str, Any]:
    attrs = getattr(frame, "attrs", None)
    return attrs if isinstance(attrs, Mapping) else {}


def stamp_adjustment(frame: Any, metadata: Any) -> Any:
    """Write `metadata`'s two flags and the return basis onto `frame.attrs`
    and return the frame. The older `adjusted` attribute is left alone."""
    split = _flag(getattr(metadata, "split_adjusted", None))
    dividend = _flag(getattr(metadata, "dividend_adjusted", None))
    frame.attrs[SPLIT_KEY] = split
    frame.attrs[DIVIDEND_KEY] = dividend
    frame.attrs[BASIS_KEY] = return_basis_for(dividend)
    return frame


def split_adjusted_of(frame: Any) -> Optional[bool]:
    """Whether a frame's bars are split-adjusted: its `split_adjusted`
    stamp, else the older `adjusted` one, else None."""
    attrs = _attrs(frame)
    flag = _flag(attrs.get(SPLIT_KEY))
    if flag is not None:
        return flag
    return _flag(attrs.get(LEGACY_KEY))


def return_basis_of(frame: Any) -> str:
    """The return basis a frame carries: its `return_basis` stamp, else the
    one its `dividend_adjusted` (or older `adjusted`) stamp implies, else
    `unknown`."""
    attrs = _attrs(frame)
    stamped = attrs.get(BASIS_KEY)
    if stamped in RETURN_BASES:
        return str(stamped)
    dividend = _flag(attrs.get(DIVIDEND_KEY))
    if dividend is None:
        dividend = _flag(attrs.get(LEGACY_KEY))
    return return_basis_for(dividend)


def combine_return_bases(
    bases: Mapping[str, str], *, what: str = "inputs"
) -> Tuple[str, List[str]]:
    """
    One basis for a result built from several inputs, and the warnings a
    disagreement earns.

    Every input on one known basis: that basis, no warning. Known bases
    that differ: `mixed`, and a warning naming which input is which. Some
    known and some not stated: `unknown`, and a warning naming the ones
    that said nothing. Nothing stated anywhere: `unknown`, silently --
    there is no disagreement to report, only an absence the basis names.
    """
    known: Dict[str, List[str]] = {}
    unstated: List[str] = []
    for label, basis in bases.items():
        if basis in (TOTAL_RETURN, PRICE_RETURN):
            known.setdefault(basis, []).append(str(label))
        else:
            unstated.append(str(label))
    if not known:
        return UNKNOWN_RETURN_BASIS, []
    if len(known) > 1:
        listing = "; ".join(
            f"{_DESCRIBED[basis]}: {_names(labels)}"
            for basis, labels in sorted(known.items())
        )
        tail = f"; no basis stated: {_names(unstated)}" if unstated else ""
        return MIXED_RETURN_BASIS, [
            f"RETURN BASIS MIXED across {what} -- {listing}{tail}. A return "
            "compounded from dividend-adjusted bars includes the dividend and "
            "one from split-only bars does not, so every comparison or sum "
            "across these differs by each name's dividend yield (about "
            "1-2%/yr for a broad equity index) for a reason that has nothing "
            "to do with the strategy. Fetch them from one provider, or adjust "
            "the price-return series for dividends first."
        ]
    (basis,) = known
    if unstated:
        return UNKNOWN_RETURN_BASIS, [
            f"RETURN BASIS NOT KNOWN for every input: {_names(known[basis])} "
            f"are {_DESCRIBED[basis]}, but {_names(unstated)} carry no "
            "statement of dividend adjustment, so whether they match cannot "
            "be told. If they do not, the gap is each name's dividend yield."
        ]
    return basis, []


def resolve_return_basis(
    metadata: Any, frames: Mapping[str, Any], *, what: str = "inputs"
) -> Tuple[str, List[str]]:
    """The basis a consumer of one provider's bars ran on: the provider's
    own statement when it makes one, else the frames' stamps combined."""
    stated = _flag(getattr(metadata, "dividend_adjusted", None))
    if metadata is not None and not hasattr(metadata, "dividend_adjusted"):
        # An object that predates the two flags speaks with `adjusted`
        # alone, read at its documented word: both adjustments or neither.
        stated = _flag(getattr(metadata, "adjusted", None))
    if stated is not None:
        return return_basis_for(stated), []
    return combine_return_bases(
        {label: return_basis_of(frame) for label, frame in frames.items()},
        what=what,
    )


def _names(labels: Iterable[str]) -> str:
    listed = list(labels)
    shown = ", ".join(listed[:8])
    more = f" and {len(listed) - 8} more" if len(listed) > 8 else ""
    return f"{shown}{more}"


__all__ = [
    "BASIS_KEY",
    "DIVIDEND_KEY",
    "MIXED_RETURN_BASIS",
    "SPLIT_KEY",
    "combine_return_bases",
    "resolve_return_basis",
    "return_basis_of",
    "split_adjusted_of",
    "stamp_adjustment",
]
