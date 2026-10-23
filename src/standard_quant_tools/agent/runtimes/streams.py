"""
Return streams as tool inputs: what a stream is, how it resolves, and the
report a tool hands back about putting several on one calendar.

A `ReturnStream` is ONE dated return series from exactly one place:

    {"ref": "sqt://returns_panel/run/rets", "column": "AAPL"}
                                         a column of a published panel
    {"values": {"2024-01-02": 0.004, "2024-01-03": -0.011, ...}}
                                         the caller has it in hand
    {"symbol": "NVDA", "start_date": "2024-01-02", "end_date": "2025-12-31"}
                                         fetched, over that window

It is `DataSource` with one difference that is the point of it: inline
values carry their DATES. `DataSource.values` is an undated list, which is
fine for one series and wrong for several -- two equal-length lists from
different calendars look aligned and are not -- so a list is refused here
with that reason. The symbol keeps `DataSource`'s rules: both ends of the
window are required, and the provider is the caller's.

`ReturnStreams` is a mapping of name to stream. `resolve_return_streams`
turns it into dated Series, which `portfolio.streams.align_return_streams`
puts on the dates they share.

`returns_ref_streams` is the same thing for the existing tools that take a
`returns_panel` reference in place of inline returns: each column is a
stream, a blank cell a date that column was not observed on.

The tools that take streams but never a symbol -- `returns` as
{stream: {ISO date: return}}, or `returns_ref` with `columns` -- share the
rest: `finite_dated_streams` / `finite_dated_series` as their field
validators, `exactly_one_source` and `columns_only_with_ref` as their model
validators, and `dated_streams` / `aligned_dated_streams` to read the
inline form as `resolve_return_streams` reads a stream's values.

This module imports nothing from the tools at load time, so the shared
result models can name `StreamAlignment` without a cycle.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from standard_quant_tools.data.factory import ProviderName

__all__ = [
    "MAX_DATES",
    "ReturnStream",
    "ReturnStreams",
    "StreamAlignment",
    "StreamCoverage",
    "aligned_dated_streams",
    "aligned_returns_ref",
    "aligned_streams",
    "columns_only_with_ref",
    "dated_series",
    "dated_streams",
    "exactly_one_source",
    "finite_dated_series",
    "finite_dated_streams",
    "resolve_return_streams",
    "returns_ref_streams",
]

#: The most dates one inline stream takes. Held in the validators rather
#: than the schema: a guard against a runaway payload, not something a
#: caller plans around.
MAX_DATES = 200_000


class ReturnStream(BaseModel):
    """One dated return series: a reference, dated values, or a symbol."""

    model_config = ConfigDict(extra="forbid")

    ref: Optional[str] = Field(
        None,
        description="An `sqt://returns_panel/...` reference. Another kind is "
        "refused; an equity curve is converted to returns_panel first.",
    )
    column: Optional[str] = Field(
        None,
        description="With `ref`: the panel column this stream is. Required "
        "when the panel has more than one.",
    )
    values: Optional[Dict[str, float]] = Field(
        None,
        min_length=1,
        description="{ISO date: return}, the stream inline. Undated lists are "
        "refused: equal lengths are not the same dates.",
    )
    symbol: Optional[str] = Field(
        None,
        description="A ticker, fetched as daily close-to-close returns over "
        "start_date..end_date (both required with it).",
    )
    start_date: Optional[str] = Field(None, description="YYYY-MM-DD, inclusive.")
    end_date: Optional[str] = Field(None, description="YYYY-MM-DD, inclusive.")
    source: Optional[ProviderName] = Field(
        None, description="Provider for `symbol`; None uses the default."
    )

    @model_validator(mode="before")
    @classmethod
    def _undated_values_are_not_a_stream(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("values"), (list, tuple)):
            raise ValueError(
                "values is an undated list, and undated values are not a "
                "stream: two equal-length lists from different calendars "
                "look aligned and are not, and nothing could check them. Give "
                "the stream as {ISO date: return}, e.g. {'2024-01-02': 0.004}."
            )
        return data

    @model_validator(mode="after")
    def _exactly_one(self) -> "ReturnStream":
        given = [
            name
            for name, value in (
                ("ref", self.ref),
                ("values", self.values),
                ("symbol", self.symbol),
            )
            if value is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "a return stream needs exactly one of ref / values / symbol; "
                f"got {given or 'none'}. Two origins would make the precedence "
                "rule part of the contract."
            )
        if self.column is not None and self.ref is None:
            raise ValueError(
                "column picks a column of a `ref` panel and was given without "
                "one. Drop column, or pass the ref it belongs to."
            )
        window = [
            name
            for name, value in (
                ("start_date", self.start_date),
                ("end_date", self.end_date),
                ("source", self.source),
            )
            if value is not None
        ]
        if self.symbol is None and window:
            raise ValueError(
                f"{', '.join(window)} only apply to a `symbol`: a ref or dated "
                f"values already fix their own window and origin. Drop "
                f"{', '.join(window)}, or pass a symbol."
            )
        missing = [
            name
            for name, value in (
                ("start_date", self.start_date),
                ("end_date", self.end_date),
            )
            if value is None
        ]
        if self.symbol is not None and missing:
            # DataSource's rule: a symbol's window is the caller's, both ends.
            raise ValueError(
                f"symbol={self.symbol!r} needs {' and '.join(missing)}. A "
                "symbol fetched without a window answers differently on "
                "different days and for different history depths; name it, "
                'e.g. "start_date": "2024-01-02", "end_date": "2025-12-31".'
            )
        if self.values is not None:
            bad = [
                date
                for date, value in self.values.items()
                if not math.isfinite(float(value))
            ]
            if bad:
                raise ValueError(
                    f"values is not finite on {bad[:5]}; a gap is a date the "
                    "stream does not have, not a NaN on it. Drop those dates."
                )
            from standard_quant_tools.agent.runtimes._shared import parse_date_keys

            # The keys are dates or the stream is refused here, by name,
            # rather than when it is resolved.
            parse_date_keys(self.values, "values", "ReturnStream")
        return self


#: A mapping of stream name to stream: the input every multi-stream tool takes.
ReturnStreams = Dict[str, ReturnStream]


class StreamCoverage(BaseModel):
    """What one stream supplied, and what the common calendar cost it."""

    model_config = ConfigDict(extra="forbid")

    n_supplied: int = Field(..., description="Dated returns the stream has.")
    first_date: Optional[str] = Field(None, description="Its first date.")
    last_date: Optional[str] = Field(None, description="Its last date.")
    n_kept: int = Field(..., description="Of those, on a date every stream has.")
    n_dropped_before: int = Field(
        0, description="Before the window every stream covers."
    )
    n_dropped_after: int = Field(0, description="After that window.")
    n_dropped_inside: int = Field(
        0, description="Inside it, on a date some other stream lacks."
    )


class StreamAlignment(BaseModel):
    """How dated streams were put on one calendar: the dates they share."""

    model_config = ConfigDict(extra="forbid")

    n_streams: int = 0
    min_overlap: int = Field(
        1, description="The fewest shared dates the call accepted."
    )
    n_dates: int = Field(0, description="Dates every stream has: the rows used.")
    n_dates_any: int = Field(0, description="Dates at least one stream has.")
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    window_start: Optional[str] = Field(
        None, description="The latest first date: where every stream has begun."
    )
    window_end: Optional[str] = Field(
        None, description="The earliest last date: where the first stream ends."
    )
    start_set_by: List[str] = Field(default_factory=list)
    end_set_by: List[str] = Field(default_factory=list)
    by_stream: Dict[str, StreamCoverage] = Field(default_factory=dict)


def dated_series(values: Mapping[str, float], field: str, what: str) -> Any:
    """
    {ISO date: value} as a float Series on its dates, sorted: the keys
    parsed by `parse_date_keys`, every value finite, and a refusal that
    names `field` when either is not.
    """
    import pandas as pd

    from standard_quant_tools.agent.runtimes._shared import parse_date_keys

    parsed = parse_date_keys(values, field, what, finite=True)
    return pd.Series(
        list(parsed.values()), index=pd.DatetimeIndex(list(parsed)), dtype="float64"
    ).sort_index()


def dated_streams(
    values: Mapping[str, Mapping[str, float]], *, what: str, field: str = "returns"
) -> Dict[str, Any]:
    """{stream: {ISO date: value}} as dated Series by name, not yet aligned."""
    return {
        str(name): dated_series(row, f"{field}[{name!r}]", what)
        for name, row in values.items()
    }


def aligned_streams(
    streams: Mapping[str, Any], *, what: str, min_overlap: int
) -> Tuple[Any, StreamAlignment, List[str]]:
    """
    Dated streams on the dates they share: the aligned frame, the report,
    and a warning when aligning dropped anything.
    """
    from standard_quant_tools.portfolio.streams import (
        align_return_streams,
        alignment_notes,
    )

    frame, report = align_return_streams(streams, min_overlap=min_overlap, what=what)
    return frame, StreamAlignment(**report), alignment_notes(report, what)


def aligned_dated_streams(
    values: Mapping[str, Mapping[str, float]],
    *,
    what: str,
    min_overlap: int,
    field: str = "returns",
) -> Tuple[Any, StreamAlignment, List[str]]:
    """`aligned_returns_ref` for the inline form, {stream: {ISO date: return}}."""
    return aligned_streams(
        dated_streams(values, what=what, field=field),
        what=what,
        min_overlap=min_overlap,
    )


# ── validators for the inline form ──────────────────────────────────────
#
# They raise ValueError, so a pydantic validator that calls one refuses the
# input by name at the boundary.


def finite_dated_series(
    values: Optional[Mapping[str, float]], field: str, *, max_dates: int = MAX_DATES
) -> Optional[Mapping[str, float]]:
    """{ISO date: value} with at most `max_dates` dates, every value finite."""
    if values is None:
        return values
    if len(values) > max_dates:
        raise ValueError(f"{field} has {len(values)} dates, more than {max_dates}.")
    bad = [date for date, value in values.items() if not math.isfinite(value)]
    if bad:
        raise ValueError(
            f"{field} is not finite on {bad[:5]}; a gap is a date the stream "
            "does not have, not a NaN on it. Drop those dates."
        )
    return values


def finite_dated_streams(
    values: Optional[Mapping[str, Mapping[str, float]]],
    field: str,
    *,
    most: int,
    max_dates: int = MAX_DATES,
) -> Optional[Mapping[str, Mapping[str, float]]]:
    """{name: {ISO date: value}} with one to `most` entries, each a
    `finite_dated_series`; a refusal names the entry that is not."""
    if values is None:
        return values
    if not values:
        raise ValueError(f"{field} is empty; name at least one entry.")
    if len(values) > most:
        raise ValueError(f"{field} has {len(values)} entries, more than {most}.")
    for name, row in values.items():
        finite_dated_series(row, f"{field}[{name!r}]", max_dates=max_dates)
    return values


def exactly_one_source(model: BaseModel, inline: str, ref: str, who: str) -> None:
    """Exactly one of the inline field and the reference field is given."""
    given = [n for n in (inline, ref) if getattr(model, n) is not None]
    if len(given) != 1:
        raise ValueError(
            f"{who} needs exactly one of {inline} / {ref}; got "
            f"{given or 'neither'}. {inline} is the data inline, {ref} a "
            "published reference."
        )


def columns_only_with_ref(
    model: BaseModel, columns: str, ref: str, *, most: Optional[int] = None
) -> None:
    """
    `columns` -- one column name, or a list of them -- is given only with
    the reference it picks from, and a list names each column once and at
    most `most` of them. A column named twice is refused rather than read
    once: it is a caller's mistake, and dropping it would hide which.
    """
    picked = getattr(model, columns)
    if picked is None:
        return
    if getattr(model, ref) is None:
        what = "a column" if isinstance(picked, str) else "columns"
        raise ValueError(
            f"{columns} names {what} of {ref} and was given without it. "
            f"Drop {columns}, or pass {ref}."
        )
    if isinstance(picked, str):
        return
    repeated = sorted({c for c in picked if picked.count(c) > 1})
    if repeated:
        raise ValueError(
            f"{columns} names {repeated} more than once; each column is one "
            "stream, so name it once."
        )
    if most is not None and len(picked) > most:
        raise ValueError(
            f"{columns} names {len(picked)} columns, more than the {most} one "
            "call takes."
        )


def _panel(ref: str, what: str, field: str) -> Any:
    from standard_quant_tools.agent.runtimes import handoff

    return handoff.resolve_frame(
        ref,
        expect="returns_panel",
        who=f"{what}: {field}",
        produced_by=(
            "fetch_returns_panel publishes one from tickers; an equity curve is "
            "levels, and convert_reference turns it into a returns_panel."
        ),
    )


def resolve_return_streams(
    streams: Mapping[str, ReturnStream], *, what: str
) -> Dict[str, Any]:
    """
    Each stream as a dated pandas Series, in the mapping's order.

    A reference is resolved with `expect="returns_panel"`, so a weight panel
    or an equity curve is refused by its kind; a panel of several columns
    needs `column`. Dated values are parsed as ISO dates. A symbol is
    fetched by `resolve_source`, over its window from its provider. The
    Series are not yet aligned: `align_return_streams` does that, and
    refuses what is not returns.
    """
    from standard_quant_tools.agent.runtimes.data.models import (
        DataSource,
        resolve_source,
    )
    from standard_quant_tools.error import ValidationError
    from standard_quant_tools.portfolio.streams import streams_from_panel

    out: Dict[str, Any] = {}
    for key, stream in streams.items():
        name = str(key)
        if not isinstance(stream, ReturnStream):
            stream = ReturnStream.model_validate(stream)
        if stream.values is not None:
            out[name] = dated_series(stream.values, f"streams[{name!r}].values", what)
        elif stream.ref is not None:
            frame = _panel(stream.ref, what, f"streams[{name!r}].ref")
            columns = [str(c) for c in frame.columns]
            column = stream.column
            if column is None:
                if len(columns) != 1:
                    raise ValidationError(
                        f"{what}: streams[{name!r}].ref has {len(columns)} "
                        f"columns ({columns[:8]}); name the one this stream is "
                        "in `column`."
                    )
                column = columns[0]
            out[name] = streams_from_panel(
                frame, what=f"{what}: streams[{name!r}]", columns=[column]
            )[column].rename(name)
        else:
            out[name] = resolve_source(
                DataSource(
                    symbol=stream.symbol,
                    start_date=stream.start_date,
                    end_date=stream.end_date,
                    source=stream.source,
                ),
                what=f"{what}: streams[{name!r}]",
            ).rename(name)
    return out


def returns_ref_streams(
    ref: str,
    *,
    what: str,
    columns: Optional[Sequence[str]] = None,
    field: str = "returns_ref",
) -> Dict[str, Any]:
    """The columns of a `returns_panel` reference as dated streams."""
    from standard_quant_tools.portfolio.streams import streams_from_panel

    return streams_from_panel(_panel(ref, what, field), what=what, columns=columns)


def aligned_returns_ref(
    ref: str,
    *,
    what: str,
    min_overlap: int,
    columns: Optional[Sequence[str]] = None,
    field: str = "returns_ref",
) -> Tuple[Any, StreamAlignment, List[str]]:
    """
    A `returns_panel` reference on the dates its columns share: the aligned
    frame, the report, and a warning when aligning dropped anything.
    """
    return aligned_streams(
        returns_ref_streams(ref, what=what, columns=columns, field=field),
        what=what,
        min_overlap=min_overlap,
    )
