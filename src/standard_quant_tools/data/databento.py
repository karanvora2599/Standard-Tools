"""
Databento's wire shape, translated into the contract this library already has.

WHY TRANSLATE RATHER THAN WIDEN. `DataProvider.get_order_book` declared its
columns -- `timestamp`, then `bid_price_{i}`/`bid_size_{i}`/`ask_price_{i}`/
`ask_size_{i}` -- before any provider implemented it, and
`analysis/order_book.py` has read exactly those names ever since. Databento
is one vendor of several, so the boundary is the right place to absorb its
spelling. Widening the contract to accept both would put the vendor's
vocabulary into every consumer of a book, and the next vendor would add a
third spelling to the same `if`.

THE THREE THINGS THAT SILENTLY POISON A BOOK, and why this module exists at
all rather than a `rename()` at the call site:

1. PRICES ARE FIXED-POINT int64, scaled by 1e-9. A raw `bid_px_00` of
   100_010_000_000 is $100.01. Registered unscaled, every spread, microprice
   and depth slope is off by a factor of a billion -- and the numbers stay
   finite and ordered, so nothing looks broken. Whether a given extract is
   fixed-point depends on how it was written: `.to_df()` converts to float
   dollars, `.to_parquet()` on the raw store does not.

2. AN EMPTY LEVEL IS int64 MAX, not null. `UNDEF_PRICE` is
   9_223_372_036_854_775_807, which scaled becomes a $9.2 BILLION quote. A
   book with five real levels and five empty ones would report a depth slope
   computed against nine-figure prices. Sentinels are masked BEFORE scaling,
   because after scaling the sentinel is just a large float and no longer
   identifiable.

3. `ts_event` IS NOT `ts_recv`. The first is when the venue says it
   happened, the second when the capture point saw it. They differ by the
   network, and the difference is exactly the thing a latency-sensitive
   microstructure study is measuring. Defaulting silently to either one
   without saying so would make that study wrong in a way its author could
   not see, so the choice is reported every time.

WHAT THIS MODULE DOES NOT DO. Talk to the API. It is pure frame-in,
frame-out, so every one of the traps above is testable without a key, a
network or an entitlement -- which is the whole reason they are here rather
than inside a provider method that can only be exercised against live data.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

#: Databento fixed-point prices are integers of this many per currency unit.
FIXED_PRICE_SCALE = 1_000_000_000

#: Sentinels. These are values, not nulls, and they are all at the top of
#: their integer range -- which is why an unmasked one reads as an enormous
#: but perfectly well-formed price rather than as missing data.
UNDEF_PRICE = 9_223_372_036_854_775_807
UNDEF_ORDER_SIZE = 4_294_967_295
UNDEF_TIMESTAMP = 18_446_744_073_709_551_615

#: Record flag bits. Two of these are the vendor telling you its own data is
#: suspect, which is worth more than any check this library could invent.
F_PUBLISHER_SPECIFIC = 2
F_MAYBE_BAD_BOOK = 4
F_BAD_TS_RECV = 8
F_MBP = 16
F_SNAPSHOT = 32
F_TOB = 64
F_LAST = 128

FLAG_MEANINGS: Dict[int, str] = {
    F_MAYBE_BAD_BOOK: (
        "the venue signalled the book may be inconsistent at this update"
    ),
    F_BAD_TS_RECV: "the receive timestamp is unreliable on this record",
    F_SNAPSHOT: "a snapshot record rather than an incremental update",
    F_TOB: "top-of-book only, not a depth update",
}

#: Datasets, with the coverage facts that decide which one can answer a
#: question. Carried here rather than rediscovered per call site.
DATASET_CONSOLIDATED = "EQUS.MINI"
DATASET_NASDAQ_BASIC = "XNAS.BASIC"
DATASET_DEPTH = "XNAS.ITCH"

#: EQUS.MINI is the consolidated tape and is the best answer whenever it
#: covers the window -- but it does not exist before this date, and a
#: request that starts earlier has to fall through to a venue dataset.
CONSOLIDATED_START = pd.Timestamp("2023-03-28", tz="UTC")

#: schema -> the kind of thing it produces here.
SCHEMA_KINDS: Dict[str, str] = {
    "mbp-10": "order_book_panel",
    "mbp-1": "quote_panel",
    "bbo-1s": "quote_panel",
    "bbo-1m": "quote_panel",
    "tbbo": "quote_panel",
    "trades": "tick_tape",
    "mbo": "order_book_panel",
    "ohlcv-1s": "price_panel",
    "ohlcv-1m": "price_panel",
    "ohlcv-1h": "price_panel",
    "ohlcv-1d": "price_panel",
}

_LEVEL_RE = re.compile(r"^(?P<side>bid|ask)_(?P<field>px|sz|ct)_(?P<level>\d{2})$")

#: The columns `.to_df()` produces that this library's contract has no place
#: for. Kept, not dropped -- `action`, `side` and `flags` are what make
#: cancellation rate and book-quality filtering computable at all, and they
#: are precisely the fields a naive rename would throw away.
PASSTHROUGH = (
    "action",
    "side",
    "depth",
    "flags",
    "sequence",
    "instrument_id",
    "publisher_id",
    "symbol",
    "order_id",
    "channel_id",
    "ts_in_delta",
)

PRICE_SCALES = ("auto", "fixed", "float")
TIMESTAMP_SOURCES = ("auto", "ts_recv", "ts_event", "index")


def looks_like_databento(columns: Sequence[str]) -> bool:
    """Whether these columns are Databento's spelling rather than this one."""
    return any(_LEVEL_RE.match(str(c)) for c in columns) or (
        "ts_recv" in columns or "ts_event" in columns
    )


def book_depth(columns: Sequence[str]) -> int:
    """Complete Databento levels present, counting from 00 and stopping at a gap."""
    present = {str(c) for c in columns}
    level = 0
    while all(
        f"{side}_{field}_{level:02d}" in present
        for side in ("bid", "ask")
        for field in ("px", "sz")
    ):
        level += 1
    return level


def _decide_price_scale(
    frame: pd.DataFrame, columns: Sequence[str], requested: str
) -> Tuple[float, str]:
    """
    Fixed-point or already-dollars, and the sentence explaining the choice.

    `auto` reads the DTYPE first, because that is the fact rather than an
    inference: Databento's fixed-point prices are int64 and `.to_df()`
    produces float64 dollars. Magnitude is only a cross-check, and it is
    checked in both directions -- an integer column of plausible dollar
    magnitudes is far more likely a rounded export than nanodollars, and
    scaling it by a billion would be the very error this function exists to
    prevent.
    """
    if requested == "fixed":
        return 1.0 / FIXED_PRICE_SCALE, "price_scale='fixed' as requested"
    if requested == "float":
        return 1.0, "price_scale='float' as requested; prices left as they are"

    usable = [c for c in columns if c in frame.columns]
    if not usable:
        return 1.0, "no price columns to scale"

    sample = frame[usable[0]]
    integral = pd.api.types.is_integer_dtype(sample)
    values = pd.to_numeric(sample, errors="coerce").to_numpy(dtype="float64")
    finite = values[np.isfinite(values) & (values != float(UNDEF_PRICE))]
    typical = float(np.nanmedian(np.abs(finite))) if finite.size else float("nan")

    if integral:
        if np.isfinite(typical) and typical < 1e6:
            return (
                1.0,
                f"price_scale='auto': the column is integer but its median "
                f"magnitude is {typical:,.0f}, which is a plausible price "
                "already -- treated as whole units, NOT divided by 1e9. Pass "
                "price_scale='fixed' if these really are nanodollars.",
            )
        return (
            1.0 / FIXED_PRICE_SCALE,
            "price_scale='auto': integer prices, read as Databento "
            f"fixed-point and divided by {FIXED_PRICE_SCALE:,}",
        )

    if np.isfinite(typical) and typical > 1e6:
        return (
            1.0 / FIXED_PRICE_SCALE,
            f"price_scale='auto': the column is float but its median "
            f"magnitude is {typical:,.0f}, far above any real price -- read "
            "as fixed-point that was cast to float and divided by "
            f"{FIXED_PRICE_SCALE:,}",
        )
    return 1.0, "price_scale='auto': float prices, already in whole units"


def _mask_sentinels(
    frame: pd.DataFrame, price_columns: Sequence[str], size_columns: Sequence[str]
) -> int:
    """Sentinel -> NaN, BEFORE any scaling makes them unrecognizable."""
    masked = 0
    for column in price_columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        hit = values == float(UNDEF_PRICE)
        if hit.any():
            masked += int(hit.sum())
            frame[column] = values.mask(hit)
        else:
            frame[column] = values
    for column in size_columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        hit = values == float(UNDEF_ORDER_SIZE)
        if hit.any():
            masked += int(hit.sum())
            frame[column] = values.mask(hit)
        else:
            frame[column] = values
    return masked


def _resolve_timestamp(
    frame: pd.DataFrame, requested: str
) -> Tuple[Optional[pd.Series], str]:
    """Which column becomes `timestamp`, and the sentence saying so."""
    if requested == "index":
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValidationError(
                "timestamp='index' was asked for but the frame's index is "
                f"a {type(frame.index).__name__}, not a DatetimeIndex. "
                "`.to_df()` sets a datetime index; a Parquet round-trip "
                "often does not."
            )
        return pd.Series(frame.index, index=frame.index), (
            "timestamp taken from the frame's DatetimeIndex"
        )

    if requested in ("ts_recv", "ts_event"):
        if requested not in frame.columns:
            raise ValidationError(
                f"timestamp={requested!r} was asked for but the frame has no "
                f"{requested!r} column. It has: {list(frame.columns)[:12]}"
            )
        chosen = requested
    elif "ts_recv" in frame.columns:
        chosen = "ts_recv"
    elif "ts_event" in frame.columns:
        chosen = "ts_event"
    elif isinstance(frame.index, pd.DatetimeIndex):
        return pd.Series(frame.index, index=frame.index), (
            "timestamp taken from the frame's DatetimeIndex; neither "
            "ts_recv nor ts_event is present as a column"
        )
    else:
        return None, "no ts_recv, ts_event or datetime index found"

    stamps = frame[chosen]
    if pd.api.types.is_integer_dtype(stamps):
        stamps = stamps.mask(stamps == UNDEF_TIMESTAMP)
        converted = pd.to_datetime(stamps, unit="ns", errors="coerce", utc=True)
    else:
        converted = pd.to_datetime(stamps, errors="coerce", utc=True)

    if chosen == "ts_recv":
        note = (
            "timestamp taken from ts_recv, when the capture point SAW the "
            "update. ts_event (when the venue says it happened) is the other "
            "choice and they differ by the network -- which is the quantity "
            "a latency study measures, so pass timestamp='ts_event' if that "
            "is what you mean."
        )
    else:
        note = (
            "timestamp taken from ts_event, the venue's own event time. "
            "ts_recv was not available or was not chosen; note that ordering "
            "by ts_event can differ from the order the book was actually "
            "observed in."
        )
    return converted, note


def normalize_book(
    frame: pd.DataFrame,
    *,
    price_scale: str = "auto",
    timestamp: str = "auto",
    levels: Optional[int] = None,
    keep_empty_levels: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    An MBP-10 frame in Databento's spelling -> this library's book contract.

    Returns the converted frame and the notes explaining every judgement it
    made, because two of those judgements (which timestamp, what price
    scale) change the numbers and neither is inferable from the result.

    A TRAILING LEVEL THAT IS EMPTY IN EVERY SNAPSHOT IS DROPPED. An mbp-10
    subscription on a name that never shows ten levels returns four real
    ones and six columns of sentinel, and once those sentinels become null
    the columns are still THERE -- so the dataset would declare ten levels,
    `book_levels` would agree, and `depth_slope` would regress size against
    distance over two levels that hold nothing. Dropping them makes the
    declared depth the real depth. `keep_empty_levels=True` preserves the
    vendor's fixed width for a caller who is aligning several extracts and
    needs the column set stable across them.
    """
    if price_scale not in PRICE_SCALES:
        raise ValidationError(
            f"price_scale={price_scale!r}; expected one of {list(PRICE_SCALES)}"
        )
    if timestamp not in TIMESTAMP_SOURCES:
        raise ValidationError(
            f"timestamp={timestamp!r}; expected one of {list(TIMESTAMP_SOURCES)}"
        )
    if frame.empty:
        raise ValidationError(
            "cannot normalize an empty frame: there is no dtype or magnitude "
            "to read the price scale from, and guessing it is the one error "
            "this module exists to prevent."
        )

    available = book_depth(frame.columns)
    if available < 1:
        raise ValidationError(
            "this frame has no complete Databento book level. A level needs "
            "bid_px_00, bid_sz_00, ask_px_00 and ask_sz_00 (two-digit, "
            f"zero-padded). Columns present: {list(frame.columns)[:12]}"
        )
    depth = available if levels is None else min(int(levels), available)
    notes: List[str] = []
    if levels is not None and int(levels) > available:
        notes.append(
            f"NOTE: {levels} levels asked for, {available} complete ones "
            "present. A level is complete only with all four of its columns; "
            "counting stops at the first gap."
        )

    out = pd.DataFrame(index=frame.index)
    price_columns: List[str] = []
    size_columns: List[str] = []
    for level in range(depth):
        for side in ("bid", "ask"):
            out[f"{side}_price_{level}"] = frame[f"{side}_px_{level:02d}"]
            out[f"{side}_size_{level}"] = frame[f"{side}_sz_{level:02d}"]
            price_columns.append(f"{side}_price_{level}")
            size_columns.append(f"{side}_size_{level}")
            counts = f"{side}_ct_{level:02d}"
            if counts in frame.columns:
                # Order COUNT at a level. Not in the library's contract, and
                # kept anyway: it is the only queue-shaped quantity MBP-10
                # carries, and a queue-position proxy cannot be built later
                # from a frame this was dropped from.
                out[f"{side}_count_{level}"] = pd.to_numeric(
                    frame[counts], errors="coerce"
                )

    n_masked = _mask_sentinels(out, price_columns, size_columns)
    if n_masked:
        notes.append(
            f"NOTE: {n_masked} sentinel value(s) became null. Databento marks "
            f"an empty level with int64 max ({UNDEF_PRICE:,}), which is a "
            "VALUE and not a null -- left in and scaled it would have read "
            "as a $9.2 billion quote."
        )

    factor, scale_note = _decide_price_scale(out, price_columns, price_scale)
    notes.append(scale_note)
    if factor != 1.0:
        for column in price_columns:
            out[column] = out[column] * factor

    if not keep_empty_levels:
        empty = [
            level
            for level in range(depth)
            if out[f"bid_price_{level}"].isna().all()
            and out[f"ask_price_{level}"].isna().all()
        ]
        # Only a TRAILING run is dropped. A gap in the middle -- level 3
        # empty while level 4 has size -- is not a thin book, it is a
        # malformed one, and silently renumbering around it would hide that.
        trailing: List[int] = []
        for level in reversed(range(depth)):
            if level in empty:
                trailing.append(level)
            else:
                break
        if trailing:
            drop: List[str] = []
            for level in trailing:
                for side in ("bid", "ask"):
                    drop.extend(
                        [
                            f"{side}_price_{level}",
                            f"{side}_size_{level}",
                            f"{side}_count_{level}",
                        ]
                    )
            out = out.drop(columns=[c for c in drop if c in out.columns])
            kept = depth - len(trailing)
            notes.append(
                f"NOTE: levels {min(trailing)}-{max(trailing)} are empty in "
                f"every one of the {len(frame):,} snapshots, so this extract "
                f"is {kept} levels deep, not {depth}. Their columns were "
                "dropped -- left in, the dataset would DECLARE "
                f"{depth} levels and depth_slope would regress against "
                "levels holding nothing. Pass keep_empty_levels=True to "
                "preserve the vendor's fixed width."
            )
        inner = [level for level in empty if level not in trailing]
        if inner:
            notes.append(
                f"WARNING: level(s) {inner} are empty in every snapshot but "
                "sit BELOW a level that has size. That is a malformed book "
                "rather than a thin one, and the columns were kept so it "
                "stays visible instead of being renumbered away."
            )

    stamps, stamp_note = _resolve_timestamp(frame, timestamp)
    notes.append(stamp_note)
    if stamps is not None:
        out.insert(0, "timestamp", stamps.to_numpy())

    for column in PASSTHROUGH:
        if column in frame.columns:
            out[column] = frame[column].to_numpy()

    if "flags" in out.columns:
        notes.extend(flag_warnings(out["flags"]))

    return out, notes


def normalize_quotes(
    frame: pd.DataFrame, *, price_scale: str = "auto", timestamp: str = "auto"
) -> Tuple[pd.DataFrame, List[str]]:
    """An MBP-1 / TBBO / BBO frame -> the `quote_panel` contract."""
    book, notes = normalize_book(
        frame, price_scale=price_scale, timestamp=timestamp, levels=1
    )
    renamed = book.rename(
        columns={
            "bid_price_0": "bid_price",
            "ask_price_0": "ask_price",
            "bid_size_0": "bid_size",
            "ask_size_0": "ask_size",
            "bid_count_0": "bid_count",
            "ask_count_0": "ask_count",
        }
    )
    return renamed, notes


def normalize_trades(
    frame: pd.DataFrame, *, price_scale: str = "auto", timestamp: str = "auto"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    A `trades` frame -> the `tick_tape` contract.

    Databento already calls these `price` and `size`, which is what the
    microstructure tools want, so this is the scaling and the timestamp and
    almost nothing else. It still goes through a function because the
    fixed-point trap applies identically here and a caller who normalized
    their book and hand-renamed their tape would hit it on the tape.
    """
    if "price" not in frame.columns or "size" not in frame.columns:
        raise ValidationError(
            "a trades frame needs `price` and `size` columns; got "
            f"{list(frame.columns)[:12]}. Those exact names are what the "
            "microstructure tools refuse without."
        )
    out = pd.DataFrame(index=frame.index)
    out["price"] = pd.to_numeric(frame["price"], errors="coerce")
    out["size"] = pd.to_numeric(frame["size"], errors="coerce")

    masked = int((out["price"] == float(UNDEF_PRICE)).sum())
    out["price"] = out["price"].mask(out["price"] == float(UNDEF_PRICE))
    out["size"] = out["size"].mask(out["size"] == float(UNDEF_ORDER_SIZE))
    notes: List[str] = []
    if masked:
        notes.append(f"NOTE: {masked} sentinel price(s) became null.")

    factor, scale_note = _decide_price_scale(out, ["price"], price_scale)
    notes.append(scale_note)
    if factor != 1.0:
        out["price"] = out["price"] * factor

    stamps, stamp_note = _resolve_timestamp(frame, timestamp)
    notes.append(stamp_note)
    if stamps is not None:
        out.insert(0, "timestamp", stamps.to_numpy())

    for column in PASSTHROUGH:
        if column in frame.columns:
            out[column] = frame[column].to_numpy()
    if "flags" in out.columns:
        notes.extend(flag_warnings(out["flags"]))
    return out, notes


def normalize_mbo(
    frame: pd.DataFrame, *, price_scale: str = "auto", timestamp: str = "auto"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    An MBO (market-by-order) frame -> this library's order-event contract.

    THE SHAPE IS NOT A BOOK. Every row is one order's add, cancel, modify
    or fill, with the venue's own `order_id`. That is what makes queue
    position and a true cancellation rate computable at all, and it is a
    different object from the depth snapshots `normalize_book` produces --
    which is why it gets its own function rather than a flag.

    The same three traps apply and are handled the same way: fixed-point
    prices decided from the dtype, sentinels masked BEFORE scaling, and the
    timestamp choice reported rather than made silently.
    """
    required = ("action", "side", "order_id")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"an MBO frame needs {list(required)}; missing {missing}. Got "
            f"{list(frame.columns)[:12]}. Without `order_id` and `action` "
            "this is a depth or trade feed, not an order feed, and every "
            "order-level measure would be a depth measure under another name."
        )
    if frame.empty:
        raise ValidationError(
            "cannot normalize an empty MBO frame: there is no dtype or "
            "magnitude to read the price scale from."
        )

    out = pd.DataFrame(index=frame.index)
    notes: List[str] = []

    price = pd.to_numeric(frame.get("price"), errors="coerce")
    masked = int((price == float(UNDEF_PRICE)).sum())
    price = price.mask(price == float(UNDEF_PRICE))
    out["price"] = price
    if masked:
        notes.append(
            f"NOTE: {masked} sentinel price(s) became null. A CLEAR carries "
            "no price, so some of these are ordinary."
        )
    factor, scale_note = _decide_price_scale(out, ["price"], price_scale)
    notes.append(scale_note)
    if factor != 1.0:
        out["price"] = out["price"] * factor

    size = pd.to_numeric(frame.get("size"), errors="coerce")
    out["size"] = size.mask(size == float(UNDEF_ORDER_SIZE))

    # Kept as the venue's own letters. Translating them to words here would
    # put this module in the business of naming market events, and the
    # letters are what `analysis/order_events.py` reads and documents.
    out["action"] = frame["action"].astype(str).str.strip().str.upper().str[:1]
    out["side"] = frame["side"].astype(str).str.strip().str.upper().str[:1]
    out["order_id"] = frame["order_id"]

    stamps, stamp_note = _resolve_timestamp(frame, timestamp)
    notes.append(stamp_note)
    if stamps is not None:
        out.insert(0, "timestamp", stamps.to_numpy())

    for column in ("sequence", "flags", "channel_id", "instrument_id", "symbol"):
        if column in frame.columns:
            out[column] = frame[column].to_numpy()
    if "flags" in out.columns:
        notes.extend(flag_warnings(out["flags"]))
    return out, notes


def flag_warnings(flags: pd.Series) -> List[str]:
    """
    What the vendor's own flag bits say about this data.

    Worth more than any check invented here: `F_MAYBE_BAD_BOOK` is the venue
    reporting that it could not keep the book consistent, which no amount of
    downstream arithmetic could recover.
    """
    values = pd.to_numeric(flags, errors="coerce").fillna(0).astype("int64")
    total = int(values.size)
    notes: List[str] = []
    if not total:
        return notes
    for bit in (F_MAYBE_BAD_BOOK, F_BAD_TS_RECV):
        hit = int(((values & bit) != 0).sum())
        if hit:
            notes.append(
                f"WARNING: {hit} of {total} records set flag {bit} -- "
                f"{FLAG_MEANINGS[bit]}. This is the venue's own assessment, "
                "not an inference; consider excluding these rows."
            )
    snapshots = int(((values & F_SNAPSHOT) != 0).sum())
    if snapshots:
        notes.append(
            f"NOTE: {snapshots} of {total} records are snapshots "
            f"(flag {F_SNAPSHOT}), which repeat state rather than report a "
            "change. Counting them as updates overstates quote intensity."
        )
    return notes


__all__ = [
    "CONSOLIDATED_START",
    "DATASET_CONSOLIDATED",
    "DATASET_DEPTH",
    "DATASET_NASDAQ_BASIC",
    "FIXED_PRICE_SCALE",
    "FLAG_MEANINGS",
    "F_BAD_TS_RECV",
    "F_LAST",
    "F_MAYBE_BAD_BOOK",
    "F_MBP",
    "F_PUBLISHER_SPECIFIC",
    "F_SNAPSHOT",
    "F_TOB",
    "PRICE_SCALES",
    "SCHEMA_KINDS",
    "TIMESTAMP_SOURCES",
    "UNDEF_ORDER_SIZE",
    "UNDEF_PRICE",
    "UNDEF_TIMESTAMP",
    "book_depth",
    "flag_warnings",
    "looks_like_databento",
    "normalize_book",
    "normalize_mbo",
    "normalize_quotes",
    "normalize_trades",
]
