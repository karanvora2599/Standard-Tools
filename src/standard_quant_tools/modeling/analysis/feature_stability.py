"""
Is this feature real, and does it stay real?

`feature_report.py` answers "what does this feature look like, over the
whole sample". Three questions it does not answer, and each of them has
talked somebody out of a strategy:

- **Drift.** A feature computed on 2015-2019 data and deployed in 2024 may
  no longer be the same measurement. The IC over the full sample averages
  across that, and an average across a break describes neither side of it.
- **Stability.** A mean IC of 0.04 is one thing if it is 0.04 every year and
  another if it is 0.30 in one year and -0.05 in the rest. The second is
  not a weaker version of the first; it is a different claim about the world.
- **Significance.** An IC of 0.03 on 60 dates and 20 entities is a number
  you can get from noise. The only honest way to know is to ask how often
  noise produces it, which is what the permutation test does.

NO SCIPY. It is not a declared dependency of this package, and
`feature_report.py` already treats "needs no scipy" as a reason to prefer
one implementation over another. The two statistics here that would usually
come from it -- PSI and the two-sample KS -- are a few lines of numpy each,
and writing them out keeps the dependency surface where it is.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.features.transforms import (
    HAS_CPP,
    _cpp_core,
)
from standard_quant_tools.modeling.validation.metrics import (
    check_ic_method,
)

from ..validation.metrics import _rank_rows, cross_sectional_ic
from .feature_report import _boundary_date, _panel_dates

logger = logging.getLogger(__name__)

#: PSI convention. These are the thresholds the credit-risk literature
#: settled on and they are conventions rather than tests -- there is no null
#: distribution behind them. Reported so an agent does not have to invent
#: its own cutoff, labelled so it does not mistake them for a p-value.
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25

#: Bins for the PSI histogram. Ten is the usual choice; the statistic is
#: mildly sensitive to it, which is why the bin count is reported.
_PSI_BINS = 10

#: Guards a PSI term against a bucket that is empty in one window. Without
#: it a single empty bucket sends the statistic to infinity, which says
#: "infinitely drifted" when it means "no observations here".
_PSI_FLOOR = 1e-6


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _require(panel: pd.DataFrame, feature: str) -> None:
    missing = [c for c in ("date", "entity", "target") if c not in panel.columns]
    if missing:
        raise ValidationError(f"panel is missing required columns: {missing}")
    if feature not in panel.columns:
        raise ValidationError(f"panel has no feature {feature!r}")


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = _PSI_BINS
) -> float:
    """
    How far `current` has moved from `reference`, in PSI.

    Bin edges come from the REFERENCE window's quantiles, not from the
    pooled sample. Pooling would let the current window move the edges it is
    being measured against, which mutes exactly the drift the statistic is
    for.
    """
    reference, current = _finite(reference), _finite(current)
    if reference.size == 0 or current.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        # A constant reference window has no distribution to drift from.
        return 0.0 if np.allclose(current, reference[0]) else float("nan")
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / reference.size
    cur_pct = np.histogram(current, bins=edges)[0] / current.size
    ref_pct = np.clip(ref_pct, _PSI_FLOOR, None)
    cur_pct = np.clip(cur_pct, _PSI_FLOOR, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """
    Two-sample Kolmogorov-Smirnov statistic: the largest gap between the two
    empirical CDFs.

    Written out rather than imported because scipy is not a dependency here.
    The implementation is the textbook one -- evaluate both CDFs at every
    observed value and take the maximum absolute difference.
    """
    reference, current = _finite(reference), _finite(current)
    if reference.size == 0 or current.size == 0:
        return float("nan")
    reference = np.sort(reference)
    current = np.sort(current)
    grid = np.concatenate([reference, current])
    cdf_ref = np.searchsorted(reference, grid, side="right") / reference.size
    cdf_cur = np.searchsorted(current, grid, side="right") / current.size
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def psi_verdict(psi: float) -> str:
    """
    The label that goes with a PSI, from the two thresholds above.

    One function rather than the comparison written out at each call site,
    so every reader of a PSI in this package draws the same line in the
    same place. A non-finite PSI compares false against both thresholds and
    comes back 'stable', which is what the single-split report has always
    said; callers that can distinguish "no drift" from "no measurement"
    check `np.isfinite` first and say so.
    """
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def _date_blocks(frame: pd.DataFrame, n_blocks: int, caller: str) -> List[np.ndarray]:
    """
    The panel's dates cut into `n_blocks` contiguous chunks.

    Shared by the block-IC report and the block-PSI curve so the two are
    the same split. A tool that shows an IC per block beside a PSI per
    block, computed from two different partitions of the dates, would be
    inviting a comparison that is not one.
    """
    dates = np.sort(_panel_dates(frame, caller).unique())
    if len(dates) < n_blocks:
        raise ValidationError(
            f"{len(dates)} dates cannot be split into {n_blocks} blocks"
        )
    return list(np.array_split(dates, n_blocks))


def feature_drift(
    panel: pd.DataFrame,
    feature: str,
    *,
    split_date: Optional[str] = None,
    method: str = "spearman",
) -> Dict[str, Any]:
    """
    How one feature's distribution -- and its IC -- differ either side of a
    date.

    Both halves matter and they fail differently. A feature can drift in
    distribution while keeping its IC (a rescaling), or hold its
    distribution while losing its IC (the relationship decayed). The first
    is a preprocessing problem; the second means the edge is gone. Reporting
    only one of them invites fixing the wrong thing.

    `split_date` defaults to the median date, which splits the panel into
    equal halves by TIME rather than by row count -- an entity that joins
    the universe late should not drag the boundary. When it is given, it is
    parsed by the rule the tool doors use: an unreadable string is refused
    by name instead of raising pandas' own parse error, and an empty one is
    refused rather than read as "use the median", which is what omitting it
    means.
    """
    check_ic_method(method, what="feature_drift")
    _require(panel, feature)
    frame = panel[["date", "entity", feature, "target"]].dropna(
        subset=["date", feature]
    )
    if frame.empty:
        raise ValidationError(f"feature {feature!r} has no observations")

    dates = _panel_dates(frame, "feature_drift")
    boundary = (
        _boundary_date(split_date, "split_date", "feature_drift", dates)
        if split_date is not None
        else dates.median()
    )
    before = frame[dates < boundary]
    after = frame[dates >= boundary]

    if before.empty or after.empty:
        raise ValidationError(
            f"split at {boundary.date()} leaves one side empty "
            f"({len(before)} before, {len(after)} after). Pick a split_date "
            "inside the panel's range."
        )

    ref = before[feature].to_numpy(dtype=float)
    cur = after[feature].to_numpy(dtype=float)
    psi = population_stability_index(ref, cur)

    def _ic(part: pd.DataFrame) -> float:
        usable = part.dropna(subset=["target"])
        if usable.empty:
            return float("nan")
        series = cross_sectional_ic(
            usable["target"].to_numpy(dtype=float),
            usable[feature].to_numpy(dtype=float),
            usable["date"].to_numpy(),
            method=method,
        )
        return float(series.mean()) if len(series) else float("nan")

    ic_before, ic_after = _ic(before), _ic(after)
    verdict = psi_verdict(psi)

    ic_flipped = (
        np.isfinite(ic_before)
        and np.isfinite(ic_after)
        and np.sign(ic_before) != np.sign(ic_after)
        and abs(ic_before) > 0.01
        and abs(ic_after) > 0.01
    )

    return {
        "feature": feature,
        "split_date": str(boundary.date()),
        "n_before": int(len(before)),
        "n_after": int(len(after)),
        "psi": psi,
        "psi_bins": _PSI_BINS,
        "psi_verdict": verdict,
        "ks_statistic": ks_statistic(ref, cur),
        "mean_before": float(np.nanmean(ref)) if ref.size else float("nan"),
        "mean_after": float(np.nanmean(cur)) if cur.size else float("nan"),
        "std_before": float(np.nanstd(ref)) if ref.size else float("nan"),
        "std_after": float(np.nanstd(cur)) if cur.size else float("nan"),
        "ic_before": ic_before,
        "ic_after": ic_after,
        "ic_flipped": bool(ic_flipped),
    }


def feature_stability(
    panel: pd.DataFrame,
    feature: str,
    *,
    n_blocks: int = 4,
    method: str = "spearman",
) -> Dict[str, Any]:
    """
    The feature's IC computed inside each of `n_blocks` contiguous time
    blocks.

    Contiguous, never shuffled. A feature's whole problem is usually that it
    worked in one regime, and randomly interleaved folds would average that
    away -- which is the failure this function exists to expose rather than
    to reproduce.

    `sign_consistency` is the number to read first: the fraction of blocks
    whose IC has the same sign as the full-sample IC. A mean IC of 0.04 at
    0.5 sign consistency is a coin flip with a good average.
    """
    check_ic_method(method, what="regime_stability")
    _require(panel, feature)
    if n_blocks < 2:
        raise ValidationError("n_blocks must be at least 2")

    frame = panel[["date", "entity", feature, "target"]].dropna(
        subset=["date", feature, "target"]
    )
    if frame.empty:
        raise ValidationError(f"feature {feature!r} has no usable observations")

    chunks = _date_blocks(frame, n_blocks, "feature_stability")

    blocks: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        part = frame[pd.to_datetime(frame["date"]).isin(chunk)]
        series = (
            cross_sectional_ic(
                part["target"].to_numpy(dtype=float),
                part[feature].to_numpy(dtype=float),
                part["date"].to_numpy(),
                method=method,
            )
            if not part.empty
            else pd.Series(dtype=float)
        )
        blocks.append(
            {
                "block": index,
                "start": str(pd.Timestamp(chunk[0]).date()),
                "end": str(pd.Timestamp(chunk[-1]).date()),
                "n_dates": int(len(chunk)),
                "ic_mean": float(series.mean()) if len(series) else float("nan"),
            }
        )

    ics = np.array([b["ic_mean"] for b in blocks], dtype=float)
    usable = _finite(ics)
    overall = cross_sectional_ic(
        frame["target"].to_numpy(dtype=float),
        frame[feature].to_numpy(dtype=float),
        frame["date"].to_numpy(),
        method=method,
    )
    overall_ic = float(overall.mean()) if len(overall) else float("nan")
    consistency = (
        float(np.mean(np.sign(usable) == np.sign(overall_ic)))
        if usable.size and np.isfinite(overall_ic)
        else float("nan")
    )

    return {
        "feature": feature,
        "n_blocks": n_blocks,
        "blocks": blocks,
        "ic_overall": overall_ic,
        "ic_block_mean": float(np.mean(usable)) if usable.size else float("nan"),
        "ic_block_std": float(np.std(usable)) if usable.size else float("nan"),
        "ic_block_min": float(np.min(usable)) if usable.size else float("nan"),
        "ic_block_max": float(np.max(usable)) if usable.size else float("nan"),
        "sign_consistency": consistency,
        "worst_block": (
            int(
                blocks[int(np.argmin(np.where(np.isfinite(ics), ics, np.inf)))]["block"]
            )
            if usable.size
            else None
        ),
    }


def psi_by_block(
    panel: pd.DataFrame,
    feature: str,
    *,
    n_blocks: int = 4,
    reference: str = "first",
) -> List[Dict[str, Any]]:
    """
    The feature's PSI in each contiguous time block, against an earlier one.

    `feature_drift` asks the same question across ONE boundary, which
    answers "has it moved" and not "when, and how fast". A feature that
    drifted once in 2021 and a feature that has been sliding every quarter
    since give the same single-split PSI, and they are not the same
    problem: the first is a break to split the sample at, the second is a
    feature whose definition is decaying and will keep decaying.

    `reference='first'` measures every block against the FIRST one, so a
    monotone slide accumulates and the curve rises. `reference='previous'`
    measures each block against the one before it, so the same slide reads
    flat and only a jump stands out. The two together separate a break from
    a drift; either alone cannot.

    Block 0 reports `psi=None`. It is the reference under `'first'` and has
    no predecessor under `'previous'`, and reporting 0.0 there would put a
    measurement where there is none.

    The blocks are the ones `feature_stability` reports, from the same
    split of the same rows, so the IC per block and the PSI per block can
    be read side by side.
    """
    _require(panel, feature)
    if n_blocks < 2:
        raise ValidationError("n_blocks must be at least 2")
    if reference not in ("first", "previous"):
        raise ValidationError(
            f"psi_by_block: reference={reference!r}; expected 'first' "
            "(every block against the earliest) or 'previous' (each block "
            "against the one before it)."
        )

    frame = panel[["date", "entity", feature, "target"]].dropna(
        subset=["date", feature, "target"]
    )
    if frame.empty:
        raise ValidationError(f"feature {feature!r} has no usable observations")

    chunks = _date_blocks(frame, n_blocks, "psi_by_block")
    dates = pd.to_datetime(frame["date"])

    windows: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        values = frame[dates.isin(chunk)][feature].to_numpy(dtype=float)
        windows.append(values)
        if index == 0:
            psi: Optional[float] = None
        else:
            against = windows[0] if reference == "first" else windows[index - 1]
            psi = population_stability_index(against, values)
        rows.append(
            {
                "block": index,
                "start": str(pd.Timestamp(chunk[0]).date()),
                "end": str(pd.Timestamp(chunk[-1]).date()),
                "n_dates": int(len(chunk)),
                "psi": psi,
                # None rather than 'stable' where there is no number: the
                # verdict is a reading of a PSI, and an absent PSI has not
                # been read.
                "psi_verdict": (
                    psi_verdict(psi) if psi is not None and np.isfinite(psi) else None
                ),
            }
        )
    return rows


def _null_distribution(
    target: np.ndarray,
    values: np.ndarray,
    dates: np.ndarray,
    n_permutations: int,
    method: str,
    random_seed: int,
) -> np.ndarray:
    """
    One mean IC per draw, from the kernel when it is present.

    The whole loop goes across the boundary rather than the correlation
    alone. Called from Python it is `n_permutations` shuffles, the same
    number of correlation calls, and the same number of round trips through
    pandas -- and that last part was about a third of the cost, for objects
    nobody looks at. Fusing also lets the ranking happen ONCE: shuffling
    values inside a date permutes their ranks, so for spearman the ranks can
    be shuffled directly instead of recomputed on every draw.

    Two numpy attempts at the same idea measured SLOWER than the loop they
    replaced (0.14x for a global lexsort, 0.6x for rank-once in numpy),
    because the existing kernel counting-sorts in O(n) and numpy has to
    sort. That is why this is C++ and not a rewrite.

    REPRODUCIBILITY IS WITHIN A BACKEND. The kernel uses its own generator,
    not a reimplementation of numpy's PCG64 bit stream, so the same
    `random_seed` gives different DRAWS with and without the extension --
    the contract `simulate_forward_paths` already states. The null they are
    drawn from is the same: measured against the analytic standard deviation
    of the mean IC under the null, the kernel lands at 1.001x and the Python
    path at 1.006x, and a permutation p-value is a property of that null
    rather than of any particular draw.
    """
    if HAS_CPP and hasattr(_cpp_core, "permutation_null_ic"):
        codes, uniques = pd.factorize(dates, sort=True)
        return np.asarray(
            _cpp_core.permutation_null_ic(
                np.ascontiguousarray(target, dtype=np.float64),
                np.ascontiguousarray(values, dtype=np.float64),
                np.ascontiguousarray(codes.astype(np.int64)),
                int(len(uniques)),
                int(n_permutations),
                int(random_seed) & 0xFFFFFFFFFFFFFFFF,
                method == "spearman",
            ),
            dtype=float,
        )

    # Row positions per date, computed once: the shuffle is the inner loop.
    groups = [np.flatnonzero(dates == d) for d in pd.unique(dates)]
    rng = np.random.default_rng(random_seed)
    null = np.empty(n_permutations, dtype=float)
    shuffled = values.copy()
    for i in range(n_permutations):
        for positions in groups:
            shuffled[positions] = rng.permutation(values[positions])
        series = cross_sectional_ic(target, shuffled, dates, method=method)
        null[i] = float(series.mean()) if len(series) else np.nan
    return null


def _circular_shift_null(
    target: np.ndarray,
    values: np.ndarray,
    dates: np.ndarray,
    entities: np.ndarray,
    n_permutations: int,
    method: str,
    random_seed: int,
) -> np.ndarray:
    """
    One mean IC per draw, each entity's feature series rolled in time by
    its own random offset.

    The roll keeps every entity's serial correlation and marginal
    distribution and destroys only the alignment with the target -- which
    is the null an autocorrelated feature actually lives under. A
    within-date shuffle also destroys the serial correlation, so its null
    ICs are independent across dates while the observed ones are not, and
    its p-values are too small by exactly that much: 27-35% rejections of a
    true null at phi 0.95-0.99 against an overlapping label (findings D10).
    """
    order = np.lexsort((dates, pd.factorize(entities)[0]))
    ordered_entities = entities[order]
    breaks = np.flatnonzero(ordered_entities[1:] != ordered_entities[:-1]) + 1
    groups = np.split(order, breaks)
    rng = np.random.default_rng(random_seed)
    null = np.empty(n_permutations, dtype=float)
    shifted = values.copy()
    for i in range(n_permutations):
        for positions in groups:
            m = len(positions)
            if m > 1:
                shifted[positions] = np.roll(values[positions], int(rng.integers(1, m)))
        series = cross_sectional_ic(target, shifted, dates, method=method)
        null[i] = float(series.mean()) if len(series) else np.nan
    return null


def _unit_rows(block: np.ndarray) -> np.ndarray:
    """Each row centred and scaled to unit length; a row with no spread
    (a constant cross-section) becomes zeros, so its correlation with
    anything is 0.0 -- the value `cross_sectional_ic` gives such a date."""
    centred = block - block.mean(axis=1, keepdims=True)
    norms = np.sqrt(np.einsum("ij,ij->i", centred, centred))
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = centred / norms[:, None]
    unit[~(np.isfinite(norms) & (norms > 0))] = 0.0
    return unit


def _entity_shuffle_null(
    target: np.ndarray,
    values: np.ndarray,
    dates: np.ndarray,
    entities: np.ndarray,
    n_permutations: int,
    method: str,
    random_seed: int,
) -> "tuple[np.ndarray, float]":
    """
    One mean IC per draw, each draw handing every entity's WHOLE feature
    series to another entity: one permutation of the entities, applied on
    every date. Returned with the observed assignment's mean IC computed by
    the same arithmetic, which is what a draw is compared against.

    What it keeps and what it breaks. Each series keeps its serial
    correlation and its own history, and every date keeps the same set of
    feature values; the only thing destroyed is which entity a series
    belongs to. That includes a static tilt -- a feature whose ranking of
    the entities barely moves (beta's cross-sectional rank correlates 0.94
    with itself 20 days later on the live panel) carries its average IC
    into a circular shift, because rolling a series in time leaves the
    entity's level where it was. Measured on that panel's selection window,
    beta_60's IC is +0.067, its circular-shift null centres at +0.060 (p
    0.075) and its entity-shuffle null at +0.002 (p 0.005). In simulation
    (741 dates, 30 entities, a five-bar overlapping label, AR(1) 0.98
    features) it rejected a true null 4-7% of the time at alpha 0.05 and
    kept its power on a static signal, where the circular shift had none.

    THE DRAW IS CHEAP ON COMPLETE DATES. On a date where every entity has a
    row, the per-date correlation of the reassigned series is a sum over
    entities of (unit-scaled feature rank of entity pi(e)) x (unit-scaled
    target rank of entity e), and ranking commutes with reassigning whole
    columns. Summed over those dates it is the trace of one
    entities-by-entities matrix under the permutation, built once -- so a
    draw costs a gather of `n_entities` numbers instead of a pass over the
    panel. Dates where some entity is missing are recomputed per draw
    through `cross_sectional_ic`, because which rows survive there depends
    on the permutation.

    Reproducible across machines and backends: the permutations come from
    numpy's `default_rng(random_seed)`, not from the native kernel's own
    generator, and the same seed draws the same permutations everywhere.
    Entities are ordered by name before the draw, so the row order of the
    panel does not change which series goes where.

    With E entities there are only E! assignments. At three entities, one
    draw in six is the observed assignment itself, so no p-value can fall
    much below 1/6 -- `select_features` warns when the panel is that small.
    """
    date_codes, date_index = pd.factorize(dates, sort=True)
    entity_codes, entity_index = pd.factorize(entities, sort=True)
    n_dates, n_entities = len(date_index), len(entity_index)
    if n_entities < 2:
        return np.full(n_permutations, np.nan), float("nan")
    cells = date_codes.astype(np.int64) * n_entities + entity_codes.astype(np.int64)
    if np.unique(cells).size != cells.size:
        raise ValidationError(
            "permutation_test_ic: null='entity_shuffle' reassigns each "
            "entity's series as a whole, so it needs one row per (date, "
            f"entity), and the panel has {cells.size - np.unique(cells).size} "
            "repeated pair(s). Drop the duplicate rows, or pass "
            "null='circular_shift'."
        )
    feature_grid = np.full(n_dates * n_entities, np.nan)
    target_grid = np.full(n_dates * n_entities, np.nan)
    feature_grid[cells] = values
    target_grid[cells] = target
    feature_grid = feature_grid.reshape(n_dates, n_entities)
    target_grid = target_grid.reshape(n_dates, n_entities)

    # Every row handed in has both values, so an empty cell is a missing
    # row and the two grids are empty in the same places.
    complete = ~np.isnan(feature_grid).any(axis=1)
    n_complete = int(complete.sum())
    cross = np.zeros((n_entities, n_entities))
    if n_complete:
        left = feature_grid[complete]
        right = target_grid[complete]
        if method == "spearman":
            left, right = _rank_rows(left), _rank_rows(right)
        # einsum rather than a matrix product: no BLAS call, so the sum
        # does not depend on the caller's thread count.
        cross = np.einsum("di,dj->ij", _unit_rows(left), _unit_rows(right))

    partial = np.flatnonzero(~complete)
    partial_feature = feature_grid[partial]
    partial_target = target_grid[partial]
    target_present = ~np.isnan(partial_target)
    partial_rows = np.broadcast_to(
        np.arange(partial.size)[:, None], partial_feature.shape
    )

    columns = np.arange(n_entities)

    def _draw(order: np.ndarray) -> float:
        # Entity e is handed the series of entity order[e].
        total = float(cross[order, columns].sum())
        count = n_complete
        if partial.size:
            shuffled = partial_feature[:, order]
            usable = target_present & ~np.isnan(shuffled)
            series = cross_sectional_ic(
                partial_target[usable],
                shuffled[usable],
                partial_rows[usable],
                method=method,
            )
            total += float(series.sum())
            count += len(series)
        return total / count if count else np.nan

    rng = np.random.default_rng(random_seed)
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        null[i] = _draw(rng.permutation(n_entities))
    # The observed assignment through the same arithmetic. A draw that
    # reproduces it -- the identity permutation, one draw in E! -- must tie
    # it exactly, and the trace above sums in a different order from
    # `cross_sectional_ic`, so the observed IC itself can sit a rounding
    # error away. Measured on three entities and 200 draws: 31 were the
    # identity, each 1e-16 below the observed IC, so against the observed
    # IC p came out at 0.005; against this it is 0.159.
    return null, _draw(columns)


#: The nulls `permutation_test_ic` draws from, in the order its refusal
#: names them.
PERMUTATION_NULLS = ("circular_shift", "within_date", "entity_shuffle")

#: Seconds one draw costs per panel row, by null: what a refusal quotes
#: before the first draw. Measured 2026-10-04 on panels of 5,000-250,000
#: rows (Python 3.11 and 3.12, native kernel present): circular_shift
#: 39-55 ns a row (1.3-1.4 ms a draw at 31,680 rows; 100-130 ns a row
#: under 5,000 rows, where fixed costs dominate), within_date 1-2 ns,
#: entity_shuffle 9-26 ns when one date in ten is missing an entity and
#: under 1 ns when every date is complete. A single quote of "about 1.6 ms
#: a draw" stood for every null and every panel size; on the live panel's
#: zoned dates the circular shift then cost 16.9 ms a draw, because each
#: draw re-factorized the dates (fixed in `permutation_test_ic`).
_DRAW_SECONDS_PER_ROW = {
    "circular_shift": 5e-8,
    "within_date": 2e-9,
    "entity_shuffle": 1e-8,
}


def estimate_draw_seconds(null: str, n_rows: int) -> float:
    """Roughly what one permutation draw costs on a panel of `n_rows`, under
    `null`. A budget refusal quotes it so a caller can weigh the wait; it is
    an order of magnitude, not a promise."""
    rate = _DRAW_SECONDS_PER_ROW.get(null, _DRAW_SECONDS_PER_ROW["circular_shift"])
    return rate * max(int(n_rows), 1)


def permutation_test_ic(
    panel: pd.DataFrame,
    feature: str,
    *,
    n_permutations: int = 200,
    method: str = "spearman",
    random_seed: int = 0,
    null: str = "circular_shift",
) -> Dict[str, Any]:
    """
    How often noise produces an IC this large.

    `null='circular_shift'` (default) rolls each entity's feature series in
    time by a random offset: the link to the target is destroyed and the
    feature's serial correlation is not, so the null's per-date ICs are as
    autocorrelated as the observed ones. `null='within_date'` shuffles the
    feature within each date -- the null as it was stated before, "no
    cross-sectional information within a date" -- which also destroys the
    serial correlation. On i.i.d. features the two agree; on the live
    panel, where every feature's per-date IC had lag-1 autocorrelation
    +0.6, the within-date null rejected a true null 27-35% of the time and
    called two features significant that a block bootstrap put at p=0.21
    and p=0.07 (findings D10). `ic_autocorrelation_lag1` in the result says
    which regime a feature is in.

    `null='entity_shuffle'` hands each entity's whole series to another
    entity, one permutation for every date (see `_entity_shuffle_null`).
    It keeps the serial correlation as the circular shift does, and it
    also breaks a static tilt the circular shift keeps: a feature whose
    ranking of the entities barely moves over time keeps most of its IC
    when rolled in time, so its circular-shift null is centred near its
    own IC rather than near zero. `null_mean` in the result shows where
    each null is centred.

    Returns a two-sided empirical p-value with the +1 correction in both
    numerator and denominator, so a p of exactly 0 is never reported --
    200 permutations cannot distinguish "p < 0.005" from "p = 0", and
    printing 0.0 claims a precision the sample size does not have. It
    counts draws whose |IC| is at least the observed |IC|, which is
    distance from zero: against a null not centred on zero it is not the
    distance from the null's centre.

    Rows with a NaN feature or target are dropped from both the observed IC
    and the null. A +/-inf feature or target is refused with a
    ValidationError rather than dropped; replace it with NaN to drop it.
    """
    check_ic_method(method, what="permutation_test")
    _require(panel, feature)
    if n_permutations < 1:
        raise ValidationError("n_permutations must be at least 1")
    if null not in PERMUTATION_NULLS:
        raise ValidationError(
            f"permutation_test_ic: null={null!r}; expected 'circular_shift', "
            "'within_date' or 'entity_shuffle'."
        )

    frame = panel[["date", "entity", feature, "target"]].dropna(
        subset=["date", feature, "target"]
    )
    if frame.empty:
        raise ValidationError(f"feature {feature!r} has no usable observations")

    # The dates as their sorted codes, computed once. `cross_sectional_ic`
    # factorizes whatever it is handed on every call, and a zoned date
    # column arrives as an object array of Timestamps: on the live panel
    # (31,680 rows) that factorization was 17 of the 18 ms one IC pass took,
    # paid again on every draw. Codes in sorted order factorize to
    # themselves, so every IC below -- observed and drawn -- is the same
    # number it was; only the index of the per-date series changes, and
    # nothing here reads it.
    dates = pd.factorize(frame["date"], sort=True)[0]
    target = frame["target"].to_numpy(dtype=float)
    values = frame[feature].to_numpy(dtype=float)

    # +/-inf is refused, not dropped. The observed IC and its null must be
    # computed on the same rows, and they were not: the observed IC kept an
    # inf row (ranked as the extreme for spearman; for pearson it made the
    # date's IC 0.0), while the native null dropped it. Measured with one
    # +inf per date, the spearman IC tested was 0.180 against 0.201 on the
    # rows the null used, and a pearson IC of 0.0 was tested against a null
    # of sd 0.044, so p was ~1 whatever the feature did. Masking both sides
    # would make them agree, but on a row set no other feature-lab IC uses
    # -- cross_sectional_ic ranks an inf -- so this test would silently
    # answer about a different sample from the IC reported beside it.
    for name, column in ((feature, values), ("target", target)):
        n_inf = int(np.isinf(column).sum())
        if n_inf:
            raise ValidationError(
                f"permutation_test_ic: {name!r} has {n_inf} infinite value(s). "
                "An infinity is a failed division upstream, not an "
                "observation, and the IC and its null would disagree about "
                "which rows it belongs to. Replace it with NaN to drop those "
                "rows, or fix the feature so it cannot divide by zero."
            )

    observed_series = cross_sectional_ic(target, values, dates, method=method)
    observed = float(observed_series.mean()) if len(observed_series) else float("nan")
    if not np.isfinite(observed):
        raise ValidationError(
            f"feature {feature!r} has no computable IC, so there is nothing "
            "to test for significance"
        )

    # What a draw is compared against: the observed IC, except under the
    # entity shuffle, whose null includes the observed assignment itself and
    # hands back that assignment's IC through the draws' own arithmetic (see
    # `_entity_shuffle_null`). The two agree to rounding; only exact ties
    # depend on which is used.
    reference = observed
    if null == "within_date":
        draws = _null_distribution(
            target, values, dates, n_permutations, method, random_seed
        )
    elif null == "entity_shuffle":
        draws, reference = _entity_shuffle_null(
            target,
            values,
            dates,
            frame["entity"].to_numpy(),
            n_permutations,
            method,
            random_seed,
        )
    else:
        draws = _circular_shift_null(
            target,
            values,
            dates,
            frame["entity"].to_numpy(),
            n_permutations,
            method,
            random_seed,
        )
    ic_autocorrelation = (
        float(observed_series.autocorr(1))
        if len(observed_series) >= 3
        else float("nan")
    )

    usable = _finite(draws)
    at_least_as_extreme = int(np.sum(np.abs(usable) >= abs(reference)))
    p_value = (at_least_as_extreme + 1) / (usable.size + 1)

    return {
        "feature": feature,
        "observed_ic": observed,
        "n_permutations": int(n_permutations),
        "n_usable_permutations": int(usable.size),
        "null_mean": float(np.mean(usable)) if usable.size else float("nan"),
        "null_std": float(np.std(usable)) if usable.size else float("nan"),
        "null_p95_abs": (
            float(np.quantile(np.abs(usable), 0.95)) if usable.size else float("nan")
        ),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "random_seed": int(random_seed),
        "null": null,
        "ic_autocorrelation_lag1": ic_autocorrelation,
    }


__all__ = [
    "PERMUTATION_NULLS",
    "PSI_MODERATE",
    "PSI_SIGNIFICANT",
    "estimate_draw_seconds",
    "feature_drift",
    "feature_stability",
    "ks_statistic",
    "permutation_test_ic",
    "population_stability_index",
    "psi_by_block",
    "psi_verdict",
]
