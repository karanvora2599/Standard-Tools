"""
Validating a dataset that will not fit in memory, a batch at a time.

WHY THIS IS NOT `validate_pit_records`. It is that tool's out-of-core
sibling and shares its actual checks -- `validate_pit_frame` is called here,
not reimplemented, because the substantive rule it enforces
(`available_time < event_time` is a column mix-up, not a tight reporting
calendar) is exactly as true of forty gigabytes as of forty rows. What could
not be shared is the delivery: `validate_pit_records` takes rows INLINE
through a tool call's JSON and is capped at 5,000 of them. An L2 tape is
five orders of magnitude past that cap.

A VERDICT, NOT AN EXCEPTION. Same reason `validate_data_bundle` returns one:
the answer is usually yes-with-caveats, and a caller who gets an exception
learns only that something was wrong with the first thing checked. A dataset
with three crossed books in nine million rows is fine and a dataset that is
one third crossed is not, and only a report that counts both can tell them
apart.

BOUNDED, AND HONEST ABOUT IT. Scanning stops at `scan_limit` rows. A partial
scan that says how far it got beats a full scan nobody waits for -- but a
clean verdict over the first two million rows of a two-billion-row tape is
evidence about the first two million rows, so `truncated` is on the report
and every count is reported against `rows_scanned` rather than against a
total the scan never reached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.data.external import (
    DEFAULT_BATCH_ROWS,
    DEFAULT_SCAN_LIMIT,
    ExternalDataset,
    book_levels,
    check_schema,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.point_in_time import (
    AVAILABLE_TIME,
    ENTITY,
    EVENT_TIME,
    validate_pit_frame,
)

#: A dataset this fraction crossed is not a book with a few bad prints.
#: Below it, crossed snapshots are noted and excluded the way
#: `order_book.book_metrics` already excludes them; above it, the file is
#: more likely to have its bid and ask columns the wrong way round.
CROSSED_BLOCKING_FRACTION = 0.05


@dataclass
class ExternalValidationReport:
    """What a bounded scan found, and whether it blocks modeling."""

    kind: str = ""
    rows_scanned: int = 0
    rows_total: Optional[int] = None
    batches: int = 0
    truncated: bool = False
    usable: bool = False
    blocking: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def coverage(self) -> Optional[float]:
        if not self.rows_total:
            return None
        return min(1.0, self.rows_scanned / float(self.rows_total))


def _finite_mask(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return np.isfinite(values)


class _Checker:
    """Per-kind state carried across batches."""

    def __init__(self, kind: str, columns: List[str]) -> None:
        self.kind = kind
        self.columns = columns
        self.counts: Dict[str, int] = {}
        self.last_timestamp: Any = None
        self.out_of_order = 0
        self.pit_failures: List[str] = []
        self.lag_sum = 0.0
        self.lag_n = 0
        self.lag_min: Optional[float] = None
        self.lag_max: Optional[float] = None

    def bump(self, key: str, n: int) -> None:
        if n:
            self.counts[key] = self.counts.get(key, 0) + int(n)

    # ---- time ordering, shared by every kind that stamps its rows ----
    def check_order(self, frame: pd.DataFrame, column: str) -> None:
        if column not in frame.columns:
            return
        stamps = pd.to_datetime(frame[column], errors="coerce")
        self.bump("unparseable_timestamps", int(stamps.isna().sum()))
        valid = stamps.dropna()
        if valid.empty:
            return
        within = int((valid.diff() < pd.Timedelta(0)).sum())
        across = 0
        if self.last_timestamp is not None and valid.iloc[0] < self.last_timestamp:
            across = 1
        self.out_of_order += within + across
        self.last_timestamp = valid.iloc[-1]

    def feed(self, frame: pd.DataFrame) -> None:
        handler = getattr(self, f"_check_{self.kind}", None)
        if handler is not None:
            handler(frame)

    # ---- order_book_panel ----
    def _check_order_book_panel(self, frame: pd.DataFrame) -> None:
        self.check_order(frame, "timestamp")
        bid = pd.to_numeric(frame["bid_price_0"], errors="coerce")
        ask = pd.to_numeric(frame["ask_price_0"], errors="coerce")
        bid_size = pd.to_numeric(frame["bid_size_0"], errors="coerce")
        ask_size = pd.to_numeric(frame["ask_size_0"], errors="coerce")

        both = np.isfinite(bid.to_numpy()) & np.isfinite(ask.to_numpy())
        self.bump("non_finite_touch_prices", int((~both).sum()))
        # `book_metrics` excludes crossed snapshots from its price stats, so
        # counting them here reports what that exclusion will silently cost.
        self.bump("crossed", int((bid.to_numpy()[both] >= ask.to_numpy()[both]).sum()))
        self.bump(
            "non_positive_touch_size",
            int(((bid_size.fillna(-1) <= 0) | (ask_size.fillna(-1) <= 0)).sum()),
        )
        self.bump("empty_book", int(((bid_size == 0) & (ask_size == 0)).sum()))

    # ---- event_panel ----
    def _check_event_panel(self, frame: pd.DataFrame) -> None:
        self.check_order(frame, AVAILABLE_TIME)
        has_entity = ENTITY in frame.columns
        try:
            checked = validate_pit_frame(
                frame, name="external dataset", require_entity=has_entity
            )
        except ValidationError as exc:
            # One message per distinct failure, not one per batch -- a
            # column mix-up repeats in every batch and would otherwise
            # produce a report that is the same sentence a hundred times.
            text = str(exc)
            if text not in self.pit_failures and len(self.pit_failures) < 5:
                self.pit_failures.append(text)
            return
        lag = (
            checked[AVAILABLE_TIME] - checked[EVENT_TIME]
        ).dt.total_seconds() / 86400.0
        finite = lag[np.isfinite(lag.to_numpy())]
        if not finite.empty:
            self.lag_sum += float(finite.sum())
            self.lag_n += int(finite.size)
            low, high = float(finite.min()), float(finite.max())
            self.lag_min = low if self.lag_min is None else min(self.lag_min, low)
            self.lag_max = high if self.lag_max is None else max(self.lag_max, high)

    # ---- tick_tape ----
    def _check_tick_tape(self, frame: pd.DataFrame) -> None:
        self.check_order(frame, "timestamp")
        price = pd.to_numeric(frame["price"], errors="coerce")
        size = pd.to_numeric(frame["size"], errors="coerce")
        self.bump("non_finite_price", int((~_finite_mask(price)).sum()))
        self.bump("non_positive_price", int((price.fillna(-1) <= 0).sum()))
        self.bump("non_positive_size", int((size.fillna(-1) <= 0).sum()))

    # ---- order_event_panel ----
    def _check_order_event_panel(self, frame: pd.DataFrame) -> None:
        """The kind that had no handler at all.

        `feed()` dispatches by name and no-ops when there is none, so an
        order-event panel would have been scanned and reported on without
        even its timestamps being checked for order.

        The vocabularies come from `analysis.order_events` rather than
        being respelled here: that module is what will read the panel, so
        an action letter this accepts and it rejects would be a
        disagreement between the validator and the consumer.
        """
        from standard_quant_tools.analysis.order_events import (
            ACTION_MEANINGS,
            ASK,
            BID,
        )

        self.check_order(frame, "timestamp")

        action = frame["action"].astype("string").str.strip().str.upper()
        self.bump("unknown_action", int((~action.isin(ACTION_MEANINGS)).sum()))

        side = frame["side"].astype("string").str.strip().str.upper()
        # A clear (R) carries no side, so an empty side is only counted
        # against events that should have one.
        sided = action != "R"
        self.bump(
            "unknown_side",
            int((sided & ~side.isin({BID, ASK})).sum()),
        )

        self.bump("missing_order_id", int(frame["order_id"].isna().sum()))

        size = pd.to_numeric(frame["size"], errors="coerce")
        price = pd.to_numeric(frame["price"], errors="coerce")
        # A cancel or clear legitimately carries no price; an ADD or a FILL
        # without one is a row `order_event_metrics` cannot use.
        priced = action.isin({"A", "M", "F", "T"})
        self.bump(
            "non_finite_price_on_priced_event",
            int((priced & ~_finite_mask(price)).sum()),
        )
        self.bump("non_positive_size", int((size.fillna(-1) <= 0).sum()))

    # ---- quote_panel ----
    def _check_quote_panel(self, frame: pd.DataFrame) -> None:
        self.check_order(frame, "timestamp")
        bid = pd.to_numeric(frame["bid_price"], errors="coerce")
        ask = pd.to_numeric(frame["ask_price"], errors="coerce")
        both = _finite_mask(bid) & _finite_mask(ask)
        self.bump("non_finite_quotes", int((~both).sum()))
        self.bump("crossed", int((bid.to_numpy()[both] >= ask.to_numpy()[both]).sum()))


def validate_external(
    handle: ExternalDataset,
    *,
    kind: Optional[str] = None,
    scan_limit: int = DEFAULT_SCAN_LIMIT,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> ExternalValidationReport:
    """
    Scan a registered dataset and report what would break modeling on it.

    The schema check runs first and alone: if a `quote_panel` has no
    `ask_price` there is nothing useful to say about its rows, and scanning
    two million of them to say it again per batch would be slower and no
    more informative.
    """
    kind = str(kind or handle.kind)
    report = ExternalValidationReport(kind=kind, rows_total=handle.rows)

    schema_problems = check_schema(kind, handle.columns)
    if schema_problems:
        report.blocking.extend(schema_problems)
        report.usable = False
        return report

    if kind == "order_book_panel":
        report.stats["levels"] = book_levels(handle.columns)

    checker = _Checker(kind, list(handle.columns))
    limit = max(1, int(scan_limit))
    for frame in handle.batches(batch_rows=batch_rows):
        if frame.empty:
            continue
        remaining = limit - report.rows_scanned
        if remaining <= 0:
            report.truncated = True
            break
        if len(frame) > remaining:
            frame = frame.iloc[:remaining]
            report.truncated = True
        checker.feed(frame)
        report.rows_scanned += len(frame)
        report.batches += 1
        if report.rows_scanned >= limit:
            report.truncated = True
            break

    if report.rows_scanned == 0:
        report.blocking.append(
            "the dataset has a valid schema but no rows. Nothing downstream "
            "can be computed from an empty extract."
        )
        report.usable = False
        return report

    _summarize(report, checker)
    report.usable = not report.blocking
    return report


def _summarize(report: ExternalValidationReport, checker: _Checker) -> None:
    scanned = report.rows_scanned
    counts = checker.counts
    report.stats.update({k: v for k, v in counts.items() if v})

    def fraction(key: str) -> float:
        return counts.get(key, 0) / float(scanned) if scanned else 0.0

    if checker.out_of_order:
        report.stats["rows_out_of_time_order"] = checker.out_of_order
        report.warnings.append(
            f"WARNING: {checker.out_of_order} row(s) of {scanned} scanned "
            "arrive earlier than the row before them. Anything that reads "
            "this as a time series -- a rolling window, an as-of join, a "
            "signed-flow rule -- assumes it is sorted and will not say so."
        )

    if counts.get("unparseable_timestamps"):
        n = counts["unparseable_timestamps"]
        report.blocking.append(
            f"{n} row(s) of {scanned} scanned have a timestamp that cannot "
            "be parsed. A row that cannot be placed in time can neither be "
            "used nor safely ignored."
        )

    if report.kind in ("order_book_panel", "quote_panel"):
        crossed = counts.get("crossed", 0)
        if crossed:
            share = fraction("crossed")
            message = (
                f"{crossed} of {scanned} scanned snapshots are crossed "
                f"(bid >= ask), {share:.2%}."
            )
            if share >= CROSSED_BLOCKING_FRACTION:
                report.blocking.append(
                    message
                    + " Above "
                    + f"{CROSSED_BLOCKING_FRACTION:.0%} this is more likely "
                    "the bid and ask columns the wrong way round than a "
                    "genuinely locked market."
                )
            else:
                report.warnings.append(
                    "NOTE: " + message + " book_metrics excludes these from its price "
                    "statistics, so they cost rows silently."
                )

    if report.kind == "order_book_panel":
        levels = int(report.stats.get("levels", 0))
        if levels == 1:
            report.warnings.append(
                "WARNING: this book has ONE complete level, so it is top of "
                "book. depth_slope is null on a one-level book and "
                "cumulative imbalance equals touch imbalance -- the depth "
                "measures will run and will tell you nothing depth-specific."
            )
        if counts.get("non_positive_touch_size"):
            n = counts["non_positive_touch_size"]
            report.warnings.append(
                f"NOTE: {n} of {scanned} scanned snapshots have a zero or "
                "negative size at the touch. A zero-size side makes the "
                "microprice lean fully to the other one."
            )

    if report.kind == "event_panel":
        for failure in checker.pit_failures:
            report.blocking.append(failure)
        if checker.lag_n:
            mean_lag = checker.lag_sum / checker.lag_n
            report.stats.update(
                {
                    "mean_publication_lag_days": round(mean_lag, 4),
                    "min_publication_lag_days": round(float(checker.lag_min or 0), 4),
                    "max_publication_lag_days": round(float(checker.lag_max or 0), 4),
                }
            )
            if checker.lag_max == 0 and checker.lag_min == 0:
                report.warnings.append(
                    "WARNING: every scanned row has available_time exactly "
                    "equal to event_time. That is a dataset with no "
                    "publication lag at all, which is right for a price and "
                    "almost never right for anything reported -- check the "
                    "extract did not simply copy one column into the other."
                )

    if report.kind == "tick_tape":
        for key, label in (
            ("non_positive_price", "a non-positive price"),
            ("non_positive_size", "a non-positive size"),
        ):
            if counts.get(key):
                report.blocking.append(
                    f"{counts[key]} of {scanned} scanned trades have {label}. "
                    "Signing rules and volume buckets both divide by these."
                )

    if report.truncated:
        total = report.rows_total
        seen = f"{scanned:,}"
        report.warnings.append(
            f"NOTE: the scan stopped after {seen} rows"
            + (f" of {total:,}" if total else "")
            + ". Every count above is over what was scanned, not over the "
            "whole dataset -- raise scan_limit to widen it."
        )


__all__ = [
    "CROSSED_BLOCKING_FRACTION",
    "ExternalValidationReport",
    "validate_external",
]
