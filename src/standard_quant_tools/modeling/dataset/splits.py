"""
Splits in the bars a dataset is built from.

Databento serves the prices the venue published and says so
(`get_metadata().adjusted` is False): a 10:1 split is a -90% bar. Every
label and feature in this runtime is computed from raw Close, so a label
whose horizon spans that bar, and every feature whose window or smoother
reaches back across it, reads the split as a return. A live 2022-2026
Databento panel of 30 names spanned six splits: 25 labels and 1,131
feature rows read one, and about half of a ridge model's rank IC came
from those rows.

Two things live here, both run by `build_dataset`:

THE SCREEN. Every entity's Close, and the benchmark's when a feature
reads the benchmark, is screened for close-to-close moves beyond the
backtest's threshold (35%,
`constants.SPLIT_SCREEN_THRESHOLD`, the same object
`backtest.screens` reads), each named after the split ratio it is
consistent with (`data.quality.detect_split_like_moves`). For each move
the build counts the labels that span it and the panel rows carrying a
feature computed across it, using each feature's converged warm-up
(`features.params.resolved_warmup`) plus its deepest lag on the bars of
the series that moved. A universe-scope feature reads every entity's
returns, so a move in one entity counts every entity's rows; a move in
the benchmark counts every entity's rows for the features that read the
benchmark. The screen reads the bars; it never changes the panel.

THE DECLARED TABLE. `DatasetSpec.corporate_actions` lists splits the
caller knows about. Before any feature or label is computed, each listed
entity's bars before the ex-date are back-adjusted: Open, High, Low and
Close divided by the ratio, Volume multiplied by it. The screen then runs
on the adjusted bars, so a correctly declared split is no longer named.
Dividends are not adjusted by anything here.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.constants import SPLIT_SCREEN_THRESHOLD
from standard_quant_tools.data.quality import (
    SPLIT_RATIO_TOLERANCE,
    _stamp,
    split_like_moves_at,
    split_ratio_label,
)
from standard_quant_tools.error import ValidationError

from ..features.base import FeatureScope
from ..features.params import resolved_warmup
from .lags import lags_by_output_name

#: The columns a declared split divides by its ratio. Volume is multiplied.
PRICE_COLUMNS = ("Open", "High", "Low", "Close")

#: How many moves the build warning names before "and N more".
_LISTED = 6


def _count(n: int, noun: str) -> str:
    """'1 split', '6 splits'."""
    return f"{n:,} {noun}" + ("" if n == 1 else "s")


# ── The provider's own word on adjustment ───────────────────────────────


def _as_flag(value: Any) -> Optional[bool]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def bars_adjusted_flag(
    metadata: Optional[Any], frames: Iterable[pd.DataFrame]
) -> Optional[bool]:
    """
    Whether the bars are split-adjusted: the provider's metadata when it
    says, else the frames' `attrs["adjusted"]` when every frame carries one
    and they agree, else None (not known).

    The metadata comes first because it is the provider's statement about
    the feed; a frame's stamp is the fallback for a provider whose
    metadata is unavailable. A mocked provider's metadata is not a bool
    and is read as silence, not as an answer.
    """
    if metadata is not None:
        flag = _as_flag(getattr(metadata, "adjusted", None))
        if flag is not None:
            return flag
    stamps = [_as_flag(getattr(frame, "attrs", {}).get("adjusted")) for frame in frames]
    if stamps and all(stamp is not None for stamp in stamps):
        if len(set(stamps)) == 1:
            return stamps[0]
    return None


def refuse_declared_splits_on_adjusted_bars(
    actions: Sequence[Any], source: str
) -> None:
    """A declared split on bars the provider has already adjusted would
    divide the earlier prices a second time."""
    if not actions:
        return
    listing = ", ".join(
        f"{a.entity} on {a.ex_date} ({split_ratio_label(a.split_ratio)})"
        for a in actions
    )
    raise ValidationError(
        f"build_model_dataset: DatasetSpec.corporate_actions declares {listing}, "
        f"but the provider reports adjusted=True ({source}): its bars already "
        "have splits taken out, so adjusting them again would divide the "
        "earlier prices a second time. Remove corporate_actions, or build from "
        "a provider that serves unadjusted bars."
    )


# ── The declared table ──────────────────────────────────────────────────


def _ex_timestamp(index: pd.DatetimeIndex, ex_date: str) -> pd.Timestamp:
    """The ex-date on the bars' own clock: midnight in the index's zone."""
    stamp = pd.Timestamp(ex_date)
    if index.tz is not None:
        return stamp.tz_localize(index.tz)
    return stamp


def adjust_for_declared_splits(
    frame: pd.DataFrame, actions: Sequence[Any]
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    One entity's bars with its declared splits taken out, and a record of
    what each split did.

    Back-adjustment: every bar dated before an ex-date has its prices
    divided by that split's ratio and its volume multiplied by it, and two
    splits compound. A split whose ex-date is at or before the first bar
    has no earlier bar to adjust, and one after the last bar is not in
    these bars at all; neither adjusts anything. The frame is copied, never
    edited in place: a provider may hand back a cached object.
    """
    index = pd.DatetimeIndex(frame.index)
    close = pd.to_numeric(frame["Close"], errors="coerce").to_numpy(dtype=float)
    factor = np.ones(len(index), dtype=float)
    records: List[Dict[str, Any]] = []
    for action in actions:
        before = np.asarray(index < _ex_timestamp(index, action.ex_date))
        n_before = int(before.sum())
        record: Dict[str, Any] = {
            "entity": action.entity,
            "ex_date": action.ex_date,
            "split_ratio": float(action.split_ratio),
            "bars_adjusted": 0,
            "raw_close_move": None,
            "matches_raw_move": None,
        }
        if n_before == 0:
            record["status"] = "before_first_bar"
        elif n_before == len(index):
            record["status"] = "after_last_bar"
        else:
            first_after = int(np.flatnonzero(~before)[0])
            prior = close[first_after - 1] if first_after > 0 else np.nan
            current = close[first_after]
            if prior > 0 and current > 0:
                record["raw_close_move"] = round(float(current / prior - 1.0), 4)
                distance = abs(math.log((prior / current) / action.split_ratio))
                record["matches_raw_move"] = bool(distance <= SPLIT_RATIO_TOLERANCE)
            else:
                record["matches_raw_move"] = False
            record["status"] = "applied"
            record["first_bar_after"] = _stamp(index[first_after])
            record["bars_adjusted"] = n_before
            factor[before] *= float(action.split_ratio)
        records.append(record)
    if not np.any(factor != 1.0):
        return frame, records
    adjusted = frame.copy()
    for column in PRICE_COLUMNS:
        if column in adjusted.columns:
            adjusted[column] = (
                pd.to_numeric(adjusted[column], errors="coerce").to_numpy(dtype=float)
                / factor
            )
    if "Volume" in adjusted.columns:
        adjusted["Volume"] = (
            pd.to_numeric(adjusted["Volume"], errors="coerce").to_numpy(dtype=float)
            * factor
        )
    return adjusted, records


def apply_declared_splits(
    ohlcv_by_entity: Mapping[str, pd.DataFrame], actions: Sequence[Any]
) -> Tuple[Dict[str, pd.DataFrame], List[Dict[str, Any]]]:
    """Every entity's bars with its declared splits taken out, and one
    record per declared split, in declaration order. An entity with no
    declared split keeps its own frame object, untouched."""
    out = dict(ohlcv_by_entity)
    records: List[Dict[str, Any]] = []
    by_entity: Dict[str, List[Any]] = {}
    for action in actions:
        by_entity.setdefault(action.entity, []).append(action)
    for entity, listed in by_entity.items():
        if entity not in out:
            continue
        out[entity], entity_records = adjust_for_declared_splits(out[entity], listed)
        records.extend(entity_records)
    order = {(a.entity, a.ex_date): i for i, a in enumerate(actions)}
    records.sort(key=lambda r: order.get((r["entity"], r["ex_date"]), 0))
    return out, records


def declared_split_warnings(
    records: Sequence[Dict[str, Any]], bars_adjusted: Optional[bool]
) -> List[str]:
    """
    What the declared table did, one note for the splits applied and one
    warning per split the bars do not bear out.

    A declared split whose ex-date bar does not move by about its ratio is
    still applied -- the spec says what the dataset is, and a rebuild must
    produce the same panel -- and named, because a wrong date or ratio
    creates a jump of its own, which the price-jump screen then names.
    """
    if not records:
        return []
    out: List[str] = []
    applied = [r for r in records if r["status"] == "applied"]
    if applied:
        listing = ", ".join(
            f"{r['entity']} before {r['ex_date']} ({split_ratio_label(r['split_ratio'])}"
            + (
                f"; the raw close moved {r['raw_close_move']:+.0%} into "
                f"{r['first_bar_after']})"
                if r["raw_close_move"] is not None
                else ")"
            )
            for r in applied
        )
        out.append(
            f"DECLARED SPLITS: {_count(len(applied), 'split')} from "
            f"DatasetSpec.corporate_actions adjusted before any feature or label "
            f"was computed: {listing}. Open, High, Low and Close before each "
            "ex-date are divided by the ratio and Volume is multiplied by it; "
            "dividends are not adjusted, so a return target here is still a "
            "price return. The price-jump screen runs on the adjusted bars."
        )
    for r in records:
        label = split_ratio_label(r["split_ratio"])
        if r["status"] == "applied" and not r["matches_raw_move"]:
            implied = 1.0 / r["split_ratio"] - 1.0
            moved = (
                f"the raw close moved {r['raw_close_move']:+.1%} into "
                f"{r['first_bar_after']}"
                if r["raw_close_move"] is not None
                else f"the raw bars have no usable close around {r['first_bar_after']}"
            )
            cause = (
                "the date or the ratio may be wrong"
                if bars_adjusted is False
                else "the date or the ratio may be wrong, or the bars may already "
                "be adjusted (the provider does not say)"
            )
            out.append(
                f"DECLARED SPLIT NOT SEEN: {r['entity']} {label} on {r['ex_date']} -- "
                f"{moved}, not the {implied:+.0%} a {label} split gives, so "
                f"{cause}. It was applied as declared; a move it created is "
                "named by the price-jump screen."
            )
        elif r["status"] == "before_first_bar":
            out.append(
                f"DECLARED SPLIT OUTSIDE THE BARS: {r['entity']} {label} on "
                f"{r['ex_date']} is at or before {r['entity']}'s first bar, so no "
                "earlier bar exists to adjust; it changed nothing."
            )
        elif r["status"] == "after_last_bar":
            out.append(
                f"DECLARED SPLIT OUTSIDE THE BARS: {r['entity']} {label} on "
                f"{r['ex_date']} is after {r['entity']}'s last bar, so it is not "
                "in these bars; it changed nothing."
            )
    return out


# ── The screen ──────────────────────────────────────────────────────────

_BENCHMARK_READERS: Dict[Any, bool] = {}


def reads_benchmark(definition: Any) -> bool:
    """
    Whether a feature reads `FeatureContext.benchmark_close`: its own
    function's source names it. A feature whose source cannot be read is
    assumed to, so a count over it is an upper bound rather than a miss.
    """
    fn = definition.fn
    key = getattr(fn, "__code__", fn)
    if key not in _BENCHMARK_READERS:
        try:
            _BENCHMARK_READERS[key] = "benchmark_close" in inspect.getsource(fn)
        except (OSError, TypeError):
            _BENCHMARK_READERS[key] = True
    return _BENCHMARK_READERS[key]


@dataclass
class _Reach:
    """Bars, counted from the moved bar, through which a feature still
    reads it: its converged warm-up plus its deepest lag."""

    own: int = 0  # every bar-reading feature, on the entity that moved
    universe: int = 0  # universe-scope features, on every entity
    benchmark: int = 0  # features reading the benchmark, on every entity


def _feature_reach(
    feature_specs: Sequence[Any],
    feature_defs: Sequence[Any],
    resolved_params: Sequence[Dict[str, Any]],
) -> _Reach:
    lags = lags_by_output_name(feature_specs)
    reach = _Reach()
    for fs, definition, params in zip(feature_specs, feature_defs, resolved_params):
        if definition.scope == FeatureScope.POINT_IN_TIME:
            continue
        bars = int(resolved_warmup(definition, params)) + max(
            lags.get(fs.output_name) or [0]
        )
        reach.own = max(reach.own, bars)
        if definition.scope == FeatureScope.UNIVERSE:
            reach.universe = max(reach.universe, bars)
        if reads_benchmark(definition):
            reach.benchmark = max(reach.benchmark, bars)
    return reach


def _naive_ns(values: Any) -> np.ndarray:
    """Datetimes as UTC-naive int64 nanoseconds, so bars and panel rows on
    different zone conventions compare as instants."""
    index = pd.DatetimeIndex(values)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return index.as_unit("ns").asi8


@dataclass
class PriceJumpScreen:
    """What the screen found and what reads it.

    `jumps` is JSON-safe and chronological; `rows[i]` holds the panel
    index labels of the rows carrying a feature computed across `jumps[i]`
    (what `score_model` checks its scored rows against)."""

    jumps: List[Dict[str, Any]] = field(default_factory=list)
    rows: List[np.ndarray] = field(default_factory=list)
    n_labels: int = 0
    n_targets: int = 0
    n_feature_rows: int = 0
    n_rows: int = 0
    has_labels: bool = False
    # The dates of the spanning labels, and the other rows on them: what a
    # target computed across each date's entities (a rank, a market-
    # neutral return) moves along with the spanning labels.
    n_label_dates: int = 0
    n_rows_sharing_label_dates: int = 0


def _label_columns(panel: pd.DataFrame) -> List[str]:
    """One label-end column per distinct label: the per-horizon columns of
    a multi-horizon panel (the primary among them), else the primary."""
    extra = [c for c in panel.columns if str(c).startswith("label_end_date__")]
    if extra:
        return extra
    return ["label_end_date"] if "label_end_date" in panel.columns else []


def screen_price_jumps(
    ohlcv_by_entity: Mapping[str, pd.DataFrame],
    benchmark: Tuple[str, pd.DataFrame],
    feature_specs: Sequence[Any],
    feature_defs: Sequence[Any],
    resolved_params: Sequence[Dict[str, Any]],
    panel: pd.DataFrame,
    threshold: float = SPLIT_SCREEN_THRESHOLD,
) -> PriceJumpScreen:
    """
    Screen every entity's bars and the benchmark's, and count what reads
    each move. Reads the panel; never changes it.
    """
    reach = _feature_reach(feature_specs, feature_defs, resolved_params)
    n = len(panel)
    screen = PriceJumpScreen(n_rows=n)
    label_columns = _label_columns(panel)
    screen.has_labels = bool(label_columns)
    screen.n_targets = n * len(label_columns)

    dates = _naive_ns(panel["date"])
    labels_index = panel.index.to_numpy()
    groups = {
        str(entity): np.asarray(positions)
        for entity, positions in panel.groupby("entity", sort=False).indices.items()
    }
    by_entity: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for entity, positions in groups.items():
        order = np.argsort(dates[positions], kind="stable")
        by_entity[entity] = (dates[positions][order], positions[order])
    label_ends = {column: _naive_ns(panel[column]) for column in label_columns}

    feature_union = np.zeros(n, dtype=bool)
    label_union = {column: np.zeros(n, dtype=bool) for column in label_columns}

    def rows_between(entity: str, start: int, end: int) -> np.ndarray:
        if entity not in by_entity:
            return np.empty(0, dtype=np.int64)
        entity_dates, positions = by_entity[entity]
        lo = np.searchsorted(entity_dates, start, side="left")
        hi = np.searchsorted(entity_dates, end, side="right")
        return positions[lo:hi]

    def reach_end(bars: np.ndarray, position: int, length: int) -> Optional[int]:
        if length <= 0:
            return None
        return int(bars[min(position + length - 1, len(bars) - 1)])

    series: List[Tuple[str, str, pd.DataFrame]] = [
        (str(entity), "entity", frame) for entity, frame in ohlcv_by_entity.items()
    ]
    # The benchmark is fetched for every build, but only a feature that
    # reads it can carry its move into the panel; with none, a move in it
    # is in no row and is not listed.
    if reach.benchmark > 0:
        series.append((str(benchmark[0]), "benchmark", benchmark[1]))
    found: List[Tuple[int, str, Dict[str, Any], np.ndarray]] = []
    for name, role, frame in series:
        if "Close" not in frame.columns or len(frame) < 2:
            continue
        bars = _naive_ns(frame.index)
        for position, move, ratio, error in split_like_moves_at(
            frame["Close"], threshold
        ):
            at = int(bars[position])
            reading: List[np.ndarray] = []
            if role == "entity":
                length = max(reach.own, reach.universe)
                end = reach_end(bars, position, reach.own)
                if end is not None:
                    reading.append(rows_between(name, at, end))
                end = reach_end(bars, position, reach.universe)
                if end is not None:
                    reading.extend(rows_between(e, at, end) for e in by_entity)
            else:
                length = reach.benchmark
                end = reach_end(bars, position, reach.benchmark)
                if end is not None:
                    reading.extend(rows_between(e, at, end) for e in by_entity)
            rows = (
                np.unique(np.concatenate(reading))
                if reading
                else np.empty(0, dtype=np.int64)
            )
            feature_union[rows] = True
            n_labels = 0
            if role == "entity" and name in by_entity:
                entity_dates, positions = by_entity[name]
                earlier = positions[: np.searchsorted(entity_dates, at, side="left")]
                for column in label_columns:
                    spans = earlier[label_ends[column][earlier] >= at]
                    label_union[column][spans] = True
                    n_labels += int(len(spans))
            record = {
                "entity": name,
                "role": role,
                "date": _stamp(frame.index[position]),
                "close_move": round(float(move), 4),
                "split_ratio": None if ratio is None else float(ratio),
                "ratio_error": None if error is None else round(float(error), 4),
                "labels": n_labels,
                "feature_rows": int(len(rows)),
                "reach_bars": int(length),
            }
            found.append((at, name, record, labels_index[rows]))
    found.sort(key=lambda item: (item[0], item[1]))
    screen.jumps = [record for _, _, record, _ in found]
    screen.rows = [rows for _, _, _, rows in found]
    screen.n_feature_rows = int(feature_union.sum())
    screen.n_labels = int(sum(int(mask.sum()) for mask in label_union.values()))
    spanning = np.zeros(n, dtype=bool)
    for mask in label_union.values():
        spanning |= mask
    if spanning.any():
        label_dates = np.unique(dates[spanning])
        screen.n_label_dates = int(len(label_dates))
        screen.n_rows_sharing_label_dates = int(
            (np.isin(dates, label_dates) & ~spanning).sum()
        )
    return screen


# ── What the screen says ────────────────────────────────────────────────


def _within(error: float) -> int:
    """A log distance as a whole percent, at least 1."""
    return max(1, int(math.ceil(round((math.exp(error) - 1.0) * 100.0, 6))))


def _ratio_phrase(jump: Dict[str, Any]) -> str:
    ratio = jump["split_ratio"]
    if ratio is None:
        return "not near a split ratio"
    kind = "reverse split" if ratio < 1 else "split"
    return (
        f"within {_within(jump['ratio_error'])}% of a {split_ratio_label(ratio)} {kind}"
    )


def _series_name(jump: Dict[str, Any], benchmark: str = "(benchmark)") -> str:
    return jump["entity"] + (f" {benchmark}" if jump["role"] == "benchmark" else "")


def describe_jump(jump: Dict[str, Any]) -> str:
    """'NVDA 2024-06-10 (-90%, within 1% of a 10:1 split)'."""
    return (
        f"{_series_name(jump)} {jump['date']} "
        f"({jump['close_move']:+.0%}, {_ratio_phrase(jump)})"
    )


def price_jump_warnings(
    screen: PriceJumpScreen,
    bars_adjusted: Optional[bool],
    n_declared_applied: int = 0,
    threshold: float = SPLIT_SCREEN_THRESHOLD,
    cross_sectional_target: bool = False,
) -> List[str]:
    """The build's one PRICE JUMPS warning, or nothing when the screen
    found no move."""
    if not screen.jumps:
        return []
    jumps = screen.jumps
    listing = ", ".join(describe_jump(j) for j in jumps[:_LISTED])
    more = f" and {len(jumps) - _LISTED} more" if len(jumps) > _LISTED else ""
    parts = [
        f"PRICE JUMPS: {_count(len(jumps), 'close-to-close move')} beyond "
        f"{threshold:.0%} "
        f"in the bars this dataset was built from: {listing}{more}."
    ]
    if n_declared_applied:
        parts.append(
            "These are the moves left after adjusting the "
            f"{_count(n_declared_applied, 'split')} declared in "
            "DatasetSpec.corporate_actions."
        )
    if bars_adjusted is False:
        parts.append(
            "The provider reports adjusted=False, so a split is a price fall in "
            "these bars, and every label or feature that spans one reads it as a "
            "return."
        )
    elif bars_adjusted is True:
        parts.append(
            "The provider reports adjusted=True, so each is a genuine move or a "
            "bad print; check the bar before trusting the rows that read it."
        )
    else:
        parts.append(
            "Whether these bars are split-adjusted is not known here; if they are "
            "not, a label or feature that spans one reads a split as a return."
        )
    share = screen.n_feature_rows / screen.n_rows if screen.n_rows else 0.0
    if screen.has_labels:
        parts.append(
            f"In this panel {screen.n_labels:,} of {screen.n_targets:,} targets "
            f"span one of these bars and {screen.n_feature_rows:,} rows "
            f"({share:.1%}) carry a feature computed across one."
        )
        if cross_sectional_target and screen.n_rows_sharing_label_dates:
            parts.append(
                "The target is computed across each date's entities, so the "
                f"labels of the other {screen.n_rows_sharing_label_dates:,} rows "
                f"on those {_count(screen.n_label_dates, 'date')} can move too."
            )
    else:
        parts.append(
            f"In this panel {screen.n_feature_rows:,} of {screen.n_rows:,} rows "
            f"({share:.1%}) carry a feature computed across one."
        )
    if bars_adjusted is not True:
        parts.append(
            "Declare each split in DatasetSpec.corporate_actions (entity, "
            "ex_date, split_ratio) to adjust the bars before it, rebuild from "
            "adjusted bars, or drop the named entities or dates. A genuine move "
            "of the same size reads the same way, so check each date first."
        )
    parts.append("dataset_meta.json lists each one as price_jumps.")
    return [" ".join(parts)]


def scored_row_jump_warnings(
    jumps: Sequence[Dict[str, Any]],
    rows: Sequence[np.ndarray],
    scored_index: Any,
    bars_adjusted: Optional[bool],
) -> List[str]:
    """
    One warning per screened move that a scored row reads through one of
    its features: the score's own rows, not the window's.
    """
    scored = np.asarray(list(scored_index))
    out: List[str] = []
    for jump, reading in zip(jumps, rows):
        if len(reading) == 0 or not np.isin(reading, scored).any():
            continue
        readers = (
            "benchmark-reading features" if jump["role"] == "benchmark" else "features"
        )
        head = (
            f"{_series_name(jump, '(the benchmark)')}: a {jump['close_move']:+.0%} "
            f"close-to-close move on {jump['date']} ({_ratio_phrase(jump)}) is "
            f"inside the {jump['reach_bars']} bars this score's {readers} read"
        )
        if jump["split_ratio"] is None:
            tail = (
                ". It is not near a split ratio, so it is a genuine move or a bad "
                "print; check the bar before trusting the score."
            )
        elif bars_adjusted is False:
            tail = (
                ", and the provider reports adjusted=False, so the score reads the "
                "split as a return. Retrain from a spec that declares it in "
                "corporate_actions to score adjusted bars."
            )
        elif bars_adjusted is True:
            tail = (
                ", and the provider reports adjusted=True, so it is a genuine move "
                "or a bad print; check the bar before trusting the score."
            )
        else:
            tail = (
                "; whether these bars are split-adjusted is not known, and if they "
                "are not, the score reads the split as a return."
            )
        out.append(head + tail)
    return out


__all__ = [
    "PRICE_COLUMNS",
    "PriceJumpScreen",
    "adjust_for_declared_splits",
    "apply_declared_splits",
    "bars_adjusted_flag",
    "declared_split_warnings",
    "describe_jump",
    "price_jump_warnings",
    "reads_benchmark",
    "refuse_declared_splits_on_adjusted_bars",
    "scored_row_jump_warnings",
    "screen_price_jumps",
]
