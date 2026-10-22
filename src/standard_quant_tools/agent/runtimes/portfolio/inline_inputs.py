"""
Two inline input shapes the construction tools share: a covariance matrix in
either form this library writes one, and return series that carry their
dates.

THE COVARIANCE. `estimate_covariance` answers with `matrix`, a map of asset
to {asset: covariance}, while the four tools that consume a covariance --
risk parity, maximum diversification, marginal risk contribution and the
scenario sigmas -- took only a nested list with the names beside it in
`assets`. The output of one could not be the input of the next: the matrix
handed over as it came back was refused as `list_type`, and an agent had to
rebuild the rows itself, in an order it had to get right. Both forms are
accepted now. A map is read BY NAME and laid out in `assets` order, and it
is refused when its names are not exactly `assets` or a row lacks a column.
Symmetry and finiteness are then the numeric contract's to check, for both
forms alike, in the library function the tool calls.

THE RETURNS. HRP took `Dict[str, List[float]]`: series with no dates, so
two equal-length series from different calendars -- one a fortnight later,
one with another market's holidays -- were accepted as aligned and
correlated row by row, which pairs each day with some other day. A series
may now be a date-keyed map. Dated series are checked: by default they must
carry the same dates, and `alignment="intersect"` keeps the dates all of
them share and reports, per series, what that dropped. The undated form
still works, and the result says its alignment could not be checked.

See the CHANGELOG entry of 2026-10-22.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from standard_quant_tools.agent.runtimes._shared import parse_date_keys
from standard_quant_tools.error import ValidationError

__all__ = [
    "NamedCovariance",
    "NamedCovarianceInput",
    "ReturnSeries",
    "ReturnsAlignment",
    "SeriesAlignment",
    "aligned_returns",
    "covariance_rows",
]

#: A covariance matrix as rows parallel to `assets`, or as the map of asset
#: to {asset: covariance} that `estimate_covariance` returns as `matrix`.
NamedCovariance = Union[List[List[float]], Dict[str, Dict[str, float]]]

#: One asset's returns: a date-keyed map {ISO date: return}, whose dates are
#: checked against the other series, or an undated list, whose are not.
ReturnSeries = Union[List[float], Dict[str, float]]

#: How many offending names or dates a refusal lists before counting the rest.
_SHOWN = 5


def _listed(items: Sequence[Any]) -> str:
    shown = ", ".join(repr(i) for i in items[:_SHOWN])
    more = len(items) - _SHOWN
    return f"[{shown}{f' and {more} more' if more > 0 else ''}]"


def covariance_rows(
    covariance: Mapping[str, Mapping[str, float]], assets: Sequence[str]
) -> List[List[float]]:
    """
    A name-keyed covariance laid out as rows in `assets` order, or a refusal
    naming every asset the map and `assets` disagree on.

    The map is read by name, never by position: `{"B": ..., "A": ...}` with
    `assets=["A", "B"]` gives A's row first. A row that lacks a column is
    refused rather than filled, because a missing covariance is not zero.
    """
    names = list(assets)
    expected = set(names)
    missing = [a for a in names if a not in covariance]
    extra = [str(k) for k in covariance if k not in expected]
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"no row for {_listed(missing)}")
        if extra:
            parts.append(f"rows for {_listed(extra)}, which are not in assets")
        raise ValueError(
            f"covariance is keyed by asset and its row names must be exactly "
            f"`assets`; it has {' and '.join(parts)}. Pass the `assets` and "
            "`matrix` estimate_covariance returned together, or name the same "
            "assets in both."
        )
    ragged = []
    for row_name in names:
        row = covariance[row_name]
        lacks = [a for a in names if a not in row]
        surplus = [str(k) for k in row if k not in expected]
        if lacks:
            ragged.append(f"covariance[{row_name!r}] lacks {_listed(lacks)}")
        if surplus:
            ragged.append(
                f"covariance[{row_name!r}] has {_listed(surplus)}, which are "
                "not in assets"
            )
    if ragged:
        raise ValueError(
            "covariance is not square under the asset names: "
            f"{'; '.join(ragged[:_SHOWN])}. Every row needs a value for every "
            "asset, the diagonal included -- a missing covariance is not zero."
        )
    return [[float(covariance[r][c]) for c in names] for r in names]


class NamedCovarianceInput(BaseModel):
    """
    The base of an input taking `assets` and a `covariance` in either form.

    A map is turned into rows once the fields are validated, so the tool
    behind the input always reads `covariance` as rows parallel to `assets`
    -- the form it has always read. When `assets` is absent (it is optional
    where the covariance is), the map is left for the tool's own refusal,
    which names the missing `assets`.
    """

    @model_validator(mode="after")
    def _covariance_by_name(self) -> "NamedCovarianceInput":
        covariance = getattr(self, "covariance", None)
        assets = getattr(self, "assets", None)
        if isinstance(covariance, dict) and assets:
            self.covariance = covariance_rows(covariance, assets)
        return self


class SeriesAlignment(BaseModel):
    """What aligning one dated series on the shared dates did to it."""

    model_config = ConfigDict(extra="forbid")

    n_given: int = Field(..., description="Dated returns supplied.")
    n_used: int = Field(..., description="Of those, on a date every series has.")
    n_dropped: int = Field(..., description="On a date some other series lacks.")
    first_date: Optional[str] = Field(None, description="First date supplied.")
    last_date: Optional[str] = Field(None, description="Last date supplied.")


class ReturnsAlignment(BaseModel):
    """How dated return series were put on one calendar."""

    model_config = ConfigDict(extra="forbid")

    alignment: Literal["exact", "intersect"]
    n_dates: int = Field(..., description="Dates every series has; the rows used.")
    first_date: Optional[str] = None
    last_date: Optional[str] = None
    by_series: Dict[str, SeriesAlignment] = Field(default_factory=dict)


def _iso(stamp: pd.Timestamp) -> str:
    """A date as the caller wrote it: 'YYYY-MM-DD' unless it has a time."""
    return str(stamp.date()) if stamp == stamp.normalize() else stamp.isoformat()


def aligned_returns(
    returns: Mapping[str, Union[Sequence[float], Mapping[str, float]]],
    alignment: str,
    tool: str,
) -> Tuple[pd.DataFrame, Optional[ReturnsAlignment], List[str]]:
    """
    The returns as one frame with a row per date, the alignment report
    (None for undated lists, whose alignment cannot be checked) and the
    warnings to carry onto the result.

    Refused: dated and undated series mixed; undated lists of different
    lengths; `alignment="intersect"` on undated lists; under the default
    `alignment="exact"`, dated series whose dates differ -- equal lengths
    included, which is the case undated lists cannot catch; and under
    `intersect`, series that share no date.
    """
    dated = [name for name, values in returns.items() if isinstance(values, Mapping)]
    undated = [name for name in returns if name not in set(dated)]
    if dated and undated:
        raise ValidationError(
            f"{tool}: returns mixes date-keyed maps ({_listed(dated)}) with "
            f"undated lists ({_listed(undated)}). A list cannot be placed on "
            "the maps' dates; give every series as {ISO date: return}."
        )

    if not dated:
        lengths = {name: len(values) for name, values in returns.items()}
        if len(set(lengths.values())) > 1:
            raise ValidationError(
                f"{tool}: the return series have different lengths "
                f"({lengths}). They must be aligned in time."
            )
        if alignment == "intersect":
            raise ValidationError(
                f"{tool}: alignment='intersect' needs dated series, and these "
                "are undated lists, which have no dates to intersect. Give "
                "each series as {ISO date: return}, or drop `alignment`."
            )
        frame = pd.DataFrame({name: list(values) for name, values in returns.items()})
        return (
            frame,
            None,
            [
                f"{tool}: the returns are undated lists, so whether they are "
                "aligned in time could not be checked; equal lengths were "
                "taken to mean the same dates. Two series from different "
                "calendars, or one shifted by a day, pass that test and "
                "correlate as noise. Give each series as {ISO date: return} "
                "to have its dates checked."
            ],
        )

    parsed: Dict[str, Dict[pd.Timestamp, float]] = {
        name: parse_date_keys(values, f"returns[{name!r}]", tool, finite=True)
        for name, values in returns.items()
    }
    common = set.intersection(*(set(series) for series in parsed.values()))
    union = set.union(*(set(series) for series in parsed.values()))
    if alignment == "exact" and common != union:
        lengths = {name: len(series) for name, series in parsed.items()}
        same_length = len(set(lengths.values())) == 1
        lacking = [
            f"{name!r} lacks {len(union) - len(series)} "
            f"(first {_iso(min(union - set(series)))})"
            for name, series in parsed.items()
            if len(series) < len(union)
        ]
        raise ValidationError(
            f"{tool}: the dated return series do not carry the same dates: "
            f"{len(union)} dates appear across them and {len(common)} are in "
            f"every one; {'; '.join(lacking[:_SHOWN])}."
            + (
                f" They are all {next(iter(lengths.values()))} long, which is "
                "exactly the misalignment equal-length lists hid."
                if same_length
                else ""
            )
            + " Pass alignment='intersect' to use the dates every series "
            "shares (each series' losses are reported), or align them first."
        )
    if not common:
        raise ValidationError(
            f"{tool}: the dated return series share no date, so there is no "
            "row on which all of them were observed. Check that they cover "
            "the same period."
        )

    index = sorted(common)
    frame = pd.DataFrame(
        {name: [series[d] for d in index] for name, series in parsed.items()},
        index=pd.DatetimeIndex(index),
    )
    report = ReturnsAlignment(
        alignment=alignment,  # type: ignore[arg-type]
        n_dates=len(index),
        first_date=_iso(index[0]),
        last_date=_iso(index[-1]),
        by_series={
            name: SeriesAlignment(
                n_given=len(series),
                n_used=len(index),
                n_dropped=len(series) - len(index),
                first_date=_iso(min(series)),
                last_date=_iso(max(series)),
            )
            for name, series in parsed.items()
        },
    )
    notes: List[str] = []
    dropped = {n: s.n_dropped for n, s in report.by_series.items() if s.n_dropped}
    if dropped:
        notes.append(
            f"{tool}: alignment='intersect' kept the {len(index)} dates all "
            f"{len(parsed)} series share and dropped {dropped} return(s) from "
            "the series named; by_series in `alignment` has the detail."
        )
    return frame, report, notes
