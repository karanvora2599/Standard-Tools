"""
The point-in-time join, and the temporal contract it enforces.

WHAT THIS FIXES. Everywhere else in this package, "was this available yet"
is answered by one timestamp: the bar's own date. For prices that is right —
a bar's close is known when the bar closes. For everything else it is
wrong, and wrongly in the direction that flatters a backtest:

    AAPL Q2 EPS
        period_end   2026-06-30      <- when the quarter ended
        reported_at  2026-07-29      <- when anyone could act on it
        revised_at   2026-08-14      <- when the number changed

A feature built at 2026-07-15 that joins on `period_end` reads a number
nobody had for another fortnight. Joining on `reported_at` but taking the
LATEST value reads a revision nobody had for another six weeks. Both look
like ordinary joins and both produce a model that cannot be traded.

So a record here carries an `available_time` distinct from its `event_time`,
and the join rule is:

    a feature at t may consume only rows with available_time <= t

not `event_time <= t`, which is the mistake this module exists to make
impossible to write by accident.

WHY THE MODALITIES ARE NOT HERE. This is the join primitive and the contract,
not a fundamentals feed. No shipped provider exposes point-in-time
fundamentals — `get_financial_ratios(symbol)` takes no `as_of` at all, and
every provider reports `point_in_time=False` — so a DataBundle carrying
fundamentals today would be an empty box with a correct label on it. What is
buildable and testable NOW is the join and its rules, so that when a PIT
source does arrive the leakage-critical part is already written and covered
rather than being invented under deadline.

REVISIONS. A record may be restated. `asof_join` therefore takes the latest
row whose `available_time <= t`, which for a revised value means the version
that was current AT t and not the final one. Reproducing a historical
decision means seeing the numbers as they were, mistakes included.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

#: When a record describes the world (the quarter end, the reference month).
EVENT_TIME = "event_time"
#: When the record could first be acted upon. THE join key.
AVAILABLE_TIME = "available_time"
#: Which entity it is about. Absent for a global series (CPI, Fed Funds).
ENTITY = "entity"

_RESERVED = (EVENT_TIME, AVAILABLE_TIME, ENTITY)


def validate_pit_frame(
    frame: pd.DataFrame, *, name: str = "record set", require_entity: bool = True
) -> pd.DataFrame:
    """
    Check a frame carries the temporal contract, and normalize its dtypes.

    Rejecting `available_time < event_time` is the substantive check. A record
    available BEFORE the period it describes has ended is not a tight
    reporting calendar, it is a column mix-up — and it is the one error that
    would silently make a backtest look prescient rather than merely wrong.
    """
    missing = [c for c in (EVENT_TIME, AVAILABLE_TIME) if c not in frame.columns]
    if require_entity and ENTITY not in frame.columns:
        missing.append(ENTITY)
    if missing:
        raise ValidationError(
            f"{name}: point-in-time records need column(s) {missing}. "
            f"`{EVENT_TIME}` is when the record describes the world; "
            f"`{AVAILABLE_TIME}` is when it could first be acted on, and is "
            "what the join uses. They are different columns because using the "
            "first as the second is the leak this module exists to prevent."
        )

    out = frame.copy()
    for column in (EVENT_TIME, AVAILABLE_TIME):
        out[column] = pd.to_datetime(out[column], errors="coerce")
        if out[column].isna().any():
            n_bad = int(out[column].isna().sum())
            raise ValidationError(
                f"{name}: {n_bad} row(s) have an unparseable or missing "
                f"`{column}`. A record with no availability time cannot be "
                "placed in time at all, so it can neither be used nor safely "
                "ignored — drop it deliberately or supply the timestamp."
            )

    early = out[AVAILABLE_TIME] < out[EVENT_TIME]
    if early.any():
        sample = out.loc[early, [EVENT_TIME, AVAILABLE_TIME]].head(3)
        raise ValidationError(
            f"{name}: {int(early.sum())} row(s) claim to have been available "
            f"BEFORE the period they describe ended, e.g.\n{sample.to_string()}\n"
            "That is almost always the two columns the wrong way round. Left "
            "alone it makes every model built on this data look prescient."
        )
    return out


def asof_join(
    panel: pd.DataFrame,
    records: pd.DataFrame,
    *,
    fields: Optional[Sequence[str]] = None,
    date_col: str = "date",
    entity_col: str = "entity",
    by_entity: bool = True,
    prefix: str = "",
    max_staleness: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """
    Attach each panel row the most recent record that was AVAILABLE by then.

    `panel` is the modeling panel — one row per (date, entity). `records` is
    a point-in-time record set carrying `available_time` (and `event_time`,
    and `entity` unless `by_entity=False`).

    The join is strictly backward and inclusive of `available_time == date`:
    a filing released before the close of that bar is usable on that bar. If
    your bars are intraday, or your availability timestamps are intraday, put
    the real timestamps in both and this stays correct — the comparison is on
    the timestamps you supply, not on calendar days.

    `by_entity=False` joins a GLOBAL series — CPI, Fed Funds, the VIX — to
    every entity on each date. The same rule applies: a release is usable
    from its release time, not from the month it describes.

    `max_staleness` bounds how old a record may be and still be used. Without
    it, a series that simply stops updating keeps supplying its last value
    forever, and the model learns from a number that stopped being a
    measurement years earlier. With it, rows older than the bound come back
    NaN, which alignment then handles like any other missing value.

    Returns `panel` with the requested fields added. Nothing is reordered and
    no row is dropped — a panel row with no record available yet gets NaN,
    which is the honest answer for "nobody knew this yet".
    """
    if date_col not in panel.columns:
        raise ValidationError(f"asof_join: panel has no {date_col!r} column")
    if by_entity and entity_col not in panel.columns:
        raise ValidationError(f"asof_join: panel has no {entity_col!r} column")

    records = validate_pit_frame(records, name="records", require_entity=by_entity)
    if fields is None:
        fields = [c for c in records.columns if c not in _RESERVED]
    fields = list(fields)
    unknown = [f for f in fields if f not in records.columns]
    if unknown:
        raise ValidationError(f"asof_join: records have no column(s) {unknown}")
    if not fields:
        raise ValidationError("asof_join: no value columns to join")

    collisions = [f"{prefix}{f}" for f in fields if f"{prefix}{f}" in panel.columns]
    if collisions:
        raise ValidationError(
            f"asof_join: would overwrite existing panel column(s) {collisions}. "
            "Pass `prefix` to namespace them."
        )

    left = panel.copy()
    left["_pit_row"] = np.arange(len(left), dtype=np.int64)
    left[date_col] = pd.to_datetime(left[date_col])

    right_columns = [AVAILABLE_TIME, EVENT_TIME] + fields
    if by_entity:
        right_columns.append(ENTITY)
    right = records[right_columns].copy()

    # merge_asof requires BOTH sides sorted by the join key, and sorts of the
    # `by` key are not enough -- an unsorted left side does not raise, it
    # produces wrong matches. Sorting here rather than trusting the caller.
    left_sorted = left.sort_values(date_col, kind="stable")
    right_sorted = right.sort_values(AVAILABLE_TIME, kind="stable")

    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=date_col,
        right_on=AVAILABLE_TIME,
        left_by=entity_col if by_entity else None,
        right_by=ENTITY if by_entity else None,
        direction="backward",
        allow_exact_matches=True,
    )

    if max_staleness is not None:
        if max_staleness <= pd.Timedelta(0):
            raise ValidationError(
                f"asof_join: max_staleness must be positive, got {max_staleness}"
            )
        stale = (merged[date_col] - merged[AVAILABLE_TIME]) > max_staleness
        merged.loc[stale, fields] = np.nan

    merged = merged.rename(columns={f: f"{prefix}{f}" for f in fields})
    keep = ["_pit_row"] + [f"{prefix}{f}" for f in fields]
    attached = merged[keep].set_index("_pit_row")

    out = panel.copy()
    for field in fields:
        name = f"{prefix}{field}"
        out[name] = attached[name].reindex(np.arange(len(out))).to_numpy()
    return out


def coverage_report(
    panel: pd.DataFrame, joined: pd.DataFrame, fields: Iterable[str]
) -> List[str]:
    """
    Warnings about what the join could not supply.

    A point-in-time join legitimately produces NaN at the start of a sample —
    nobody had the data yet — and that is not an error. It IS worth saying
    out loud, because the alternative is a caller discovering it as an
    unexplained drop in row count after alignment.
    """
    warnings: List[str] = []
    total = len(joined)
    if total == 0:
        return warnings
    for field in fields:
        if field not in joined.columns:
            continue
        missing = int(joined[field].isna().sum())
        if missing == total:
            warnings.append(
                f"WARNING: {field!r} was never available for any panel row. "
                "Check that the record set covers this date range and these "
                "entities, and that available_time is populated."
            )
        elif missing:
            warnings.append(
                f"NOTE: {field!r} was not yet available for {missing} of {total} "
                f"panel rows ({missing / total:.1%}). Alignment drops any row "
                "where a requested feature is missing, so this costs rows for "
                "every other feature too."
            )
    return warnings
