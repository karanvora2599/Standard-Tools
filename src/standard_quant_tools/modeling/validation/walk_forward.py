"""
WalkForwardSplit — no existing generic time-series splitter in this
codebase to reuse (checked backtest/: only strategy-level walk-forward
backtesting, not a sklearn-style splitter), so this is genuinely new.

Yields (train_positions, test_positions) over a sorted, unique date
array, walking forward one `test_window` at a time, with an `embargo` gap
between each fold's train and test window so a feature's lookback can't
bleed across the boundary. engine.py maps these date-positions back to
panel row masks (a caller with a long entity-stacked panel passes
`panel['date'].unique()` sorted here, not the panel itself).
"""

from typing import Any, Iterator, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError


class WalkForwardSplit:
    """
    Walk-forward folds over a sorted, unique date axis.

    `scheme` chooses what the training window does as the fold moves:

      'rolling' (default, and the original behaviour) — a fixed-length
        window that slides forward, so the model is always fit on exactly
        `train_window` dates and never sees anything older. The right
        choice when the relationship being estimated drifts, and the
        honest one when you want every fold trained on a comparable
        amount of data.

      'expanding' — an anchored window that starts at the beginning of the
        sample and grows, so each fold trains on everything available up
        to its embargo. On a short history the rolling window discards
        data that is perfectly usable; this keeps it. The trade is that
        later folds are fit on more data than earlier ones, so a
        performance trend across folds mixes "the model got better" with
        "the model got more data", and fold-to-fold comparison is no
        longer apples to apples.

    `train_window` remains the MINIMUM training length in both schemes, so
    an expanding run still refuses to fit its first fold on less history
    than a rolling run would have used.
    """

    def __init__(
        self,
        train_window: int,
        test_window: int,
        embargo: int = 0,
        scheme: str = "rolling",
    ):
        if train_window <= 0 or test_window <= 0:
            raise ValidationError(
                f"train_window and test_window must be > 0, got "
                f"({train_window}, {test_window})"
            )
        if embargo < 0:
            raise ValidationError(f"embargo must be >= 0, got {embargo}")
        if scheme not in ("rolling", "expanding"):
            raise ValidationError(
                f"scheme must be 'rolling' or 'expanding', got {scheme!r}"
            )
        self.train_window = train_window
        self.test_window = test_window
        self.embargo = embargo
        self.scheme = scheme

    def split(self, dates: pd.Index) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(dates)
        start = 0
        while start + self.train_window + self.embargo + self.test_window <= n:
            train_end = start + self.train_window
            test_start = train_end + self.embargo
            test_end = test_start + self.test_window
            # Expanding keeps the same fold BOUNDARIES as rolling — the
            # test windows are identical — and only anchors the training
            # start at 0, so the two schemes stay directly comparable.
            train_start = 0 if self.scheme == "expanding" else start
            yield np.arange(train_start, train_end), np.arange(test_start, test_end)
            start += self.test_window

    def n_splits(self, dates: pd.Index) -> int:
        """How many folds `split(dates)` will actually yield — engine.py
        uses this to raise a clear error before fitting anything if the
        dataset is too short for even one fold, rather than silently
        returning zero folds' worth of metrics."""
        return sum(1 for _ in self.split(dates))


class PurgedKFoldSplit:
    """
    K contiguous test blocks over the date axis, each with the training
    dates around it purged and embargoed.

    WHAT THIS BUYS OVER WALK-FORWARD. Walk-forward can only test a date
    using data before it, so the earliest `train_window` dates are never
    tested and every fold is evaluated on a different, later regime. Purged
    K-fold tests EVERY date exactly once, which makes far better use of a
    short history and gives a metric that is not dominated by whatever
    happened at the end of the sample.

    WHAT IT COSTS, STATED PLAINLY. Folds after the first train partly on
    data that comes AFTER their test block. That is not leakage in the
    label sense — the purge and embargo below remove the rows whose
    information actually touches the test window — but it is not a
    simulation of live trading either, because a live model cannot be
    fitted on next year's data. Use it to estimate whether a signal exists;
    use walk-forward to estimate what it would have earned.

    THE PURGE. A training row whose label resolves inside (or across the
    edge of) the test block shares bars with it, so it is dropped. That is
    done by the caller on the row's own recorded label end date — see
    engine.py — because entities are on different calendars and an integer
    offset against the global date axis is not equivalent. What this class
    contributes is the `embargo` band of dates removed on BOTH sides of the
    test block, which walk-forward only needs on one side.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0):
        if n_splits < 2:
            raise ValidationError(f"purged k-fold needs n_splits >= 2, got {n_splits}")
        if embargo < 0:
            raise ValidationError(f"embargo must be >= 0, got {embargo}")
        self._n_splits = n_splits
        self.embargo = embargo

    def split(self, dates: pd.Index) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(dates)
        if n < self._n_splits:
            return
        # Contiguous blocks in TIME, not the shuffled membership an
        # ordinary KFold would produce: a shuffled fold would scatter test
        # dates through the training window, and no purge could then
        # separate them.
        bounds = np.linspace(0, n, self._n_splits + 1).astype(int)
        for i in range(self._n_splits):
            test_start, test_end = int(bounds[i]), int(bounds[i + 1])
            if test_end <= test_start:
                continue
            test_positions = np.arange(test_start, test_end)
            keep = np.ones(n, dtype=bool)
            keep[
                max(0, test_start - self.embargo) : min(n, test_end + self.embargo)
            ] = False
            train_positions = np.flatnonzero(keep)
            if train_positions.size == 0:
                continue
            yield train_positions, test_positions

    def n_splits(self, dates: pd.Index) -> int:
        return sum(1 for _ in self.split(dates))


class CombinatorialPurgedSplit:
    """
    Every choice of `n_test_groups` blocks out of `n_groups` is a test set.

    WHAT THIS BUYS. Walk-forward yields one out-of-sample number per fold
    and one path through time; purged k-fold tests each date once. Neither
    gives the OOS metric a distribution. Here the date axis is cut into
    `n_groups` contiguous blocks and every combination of `n_test_groups`
    of them is a test set, so C(n, k) paths are scored -- 15 from six choose
    two -- and "how good is this model out of sample" has a spread, a fifth
    percentile and a median rather than a single draw. That is the number
    model SELECTION should be made on; a single walk-forward figure has no
    error bar at all.

    WHAT IT COSTS, STATED PLAINLY. Every path trains on blocks that come
    AFTER some of its test blocks, so like purged k-fold it is not a
    simulation of live trading: it answers "is there a signal here, and how
    sure are we", never "what would this have earned". A model validated
    this way is refused by the portfolio evaluation and the bridge by name.
    Use walk-forward for the number to quote a return from.

    THE PURGE IS PER BLOCK. A test set here is two or more disjoint blocks,
    and a training row between them must be purged only when its OWN label
    reaches the later block -- not for lying between the two, which is what
    a purge on [first test date, last test date] would have done to every
    row in the gap. The engine applies `label_overlap_mask` to each
    contiguous run of the test set and ORs the results. The `embargo` band
    is removed on both sides of every block, as purged k-fold does.

    `dates` is the SORTED, UNIQUE date axis; positions index into it.
    """

    def __init__(self, n_groups: int = 6, n_test_groups: int = 2, embargo: int = 0):
        if n_groups < 2:
            raise ValidationError(f"cpcv needs n_groups >= 2, got {n_groups}")
        if not 1 <= n_test_groups < n_groups:
            raise ValidationError(
                f"cpcv needs 1 <= n_test_groups < n_groups, got "
                f"n_test_groups={n_test_groups} for n_groups={n_groups}"
            )
        if embargo < 0:
            raise ValidationError(f"embargo must be >= 0, got {embargo}")
        self.n_groups = int(n_groups)
        self.n_test_groups = int(n_test_groups)
        self.embargo = int(embargo)

    @property
    def n_paths(self) -> int:
        from math import comb

        return comb(self.n_groups, self.n_test_groups)

    def split(self, dates: pd.Index) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        from itertools import combinations

        n = len(dates)
        if n < self.n_groups:
            return
        groups = np.array_split(np.arange(n), self.n_groups)
        for combination in combinations(range(self.n_groups), self.n_test_groups):
            keep = np.ones(n, dtype=bool)
            test_parts = []
            for g in combination:
                block = groups[g]
                if block.size == 0:
                    continue
                start, end = int(block[0]), int(block[-1]) + 1
                test_parts.append(block)
                keep[max(0, start - self.embargo) : min(n, end + self.embargo)] = False
            if not test_parts:
                continue
            test_positions = np.concatenate(test_parts)
            train_positions = np.flatnonzero(keep)
            if train_positions.size == 0:
                continue
            yield train_positions, test_positions

    def n_splits(self, dates: pd.Index) -> int:
        return sum(1 for _ in self.split(dates))


def contiguous_runs(positions: np.ndarray) -> "list[tuple[int, int]]":
    """(first, last) position of each contiguous run in a sorted position
    array -- the blocks a combinatorial test set is made of."""
    positions = np.asarray(positions)
    if positions.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(positions) != 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, positions.size - 1]
    return [(int(positions[s]), int(positions[e])) for s, e in zip(starts, ends)]


def block_label_end(
    row_dates: np.ndarray,
    row_label_end: np.ndarray,
    first_test_date: Any,
    last_test_date: Any,
) -> Any:
    """The last bar the TEST BLOCK's own labels span.

    A block is not finished when its last bar prints: a forward return on its
    final date resolves `horizon` bars later, and those bars belong to the
    block's labels. Read off the test rows' own recorded label ends rather
    than from a horizon, so entities on different calendars each contribute
    their real end.

    Falls back to the block's last date when no test row records one, which
    purges exactly what it purged before.
    """
    in_block = (row_dates >= first_test_date) & (row_dates <= last_test_date)
    if not in_block.any():
        return last_test_date
    ends = np.asarray(row_label_end)[in_block]
    ends = ends[~pd.isna(ends)]
    if not ends.size:
        return last_test_date
    latest = ends.max()
    return latest if latest > last_test_date else last_test_date


def label_overlap_mask(
    train_mask: np.ndarray,
    row_dates: np.ndarray,
    row_label_end: "np.ndarray | None",
    first_test_date: Any,
    last_test_date: Any,
) -> np.ndarray:
    """
    Which training rows share label bars with a test block.

    An overlap is between two intervals: the bars a training row's label
    spans, and the bars the TEST BLOCK's labels span. A row is purged when
    they intersect -- its label ends on or after the block starts, and the
    row itself begins on or before the block's labels END.

    THE SECOND INTERVAL USED TO BE THE BLOCK'S BARS, which is one side of
    the comparison only. A block's labels reach `horizon` bars past its last
    date, and a training row inside that reach has features built from bars
    the test labels are still resolving over. Under walk-forward it never
    arose -- training is entirely before the block, so the condition held
    either way -- but purged k-fold and cpcv put training rows on BOTH sides,
    and the rows just after a block were kept.

    MEASURED, 100 dates and a 5-bar label at embargo=0: purged_kfold kept 20
    such rows across its five folds -- `horizon` rows for every fold with
    training data after its block, which is all but the last -- and cpcv kept
    100 across fifteen folds. They are the rows nearest the block, so the
    most informative ones. walk_forward kept none, and its purge count did
    not move when `block_label_end` was introduced (35 either way), which is
    the check that this fix is a no-op there.

    This is why `ValidationSpec.embargo` does not have to cover the horizon:
    the purge covers it on both sides, from each row's own label end, which
    a fixed date count cannot do when entities sit on different calendars.
    The embargo is the extra gap for feature LOOKBACK, which is a different
    leak.

    ONE IMPLEMENTATION, for the outer fold loop and the inner
    hyperparameter search alike. The engine had this rule inline and the
    search had nothing -- its inner folds were cut on dates with an embargo
    of zero and no purge, so the candidate selection was scored on training
    rows whose labels reached into the inner test window. The outer OOS
    number stayed clean; what was leak-biased was WHICH parameters won.

    A row with no label end (NaT) compares False and is never purged: it
    has no resolved label to leak. `row_label_end=None` purges nothing,
    which is the honest behaviour for a panel that never recorded one.
    """
    if row_label_end is None:
        return np.zeros(np.asarray(train_mask).shape, dtype=bool)
    return (
        np.asarray(train_mask, dtype=bool)
        & (row_label_end >= first_test_date)
        & (
            row_dates
            <= block_label_end(
                row_dates, row_label_end, first_test_date, last_test_date
            )
        )
    )


def date_block_splits(
    row_dates: np.ndarray,
    row_label_end: "np.ndarray | None",
    *,
    n_folds: int,
    embargo: int = 0,
) -> "list[tuple[np.ndarray, np.ndarray]]":
    """
    `n_folds` contiguous date blocks, each with the rows that may be fitted
    while it is held out.

    ONE IMPLEMENTATION for every inner split that holds out a block of
    dates. The conformal calibration had this loop inline and the
    PROBABILITY calibration had none of it: `_calibrated` passed an integer
    to `CalibratedClassifierCV`, which means StratifiedKFold, which splits
    by ROW -- so the same date's other entities sat on both sides of the
    split, no row was purged for a label reaching into its block, and no
    embargo band was removed. Two spec fields are called
    `calibration_folds` and only `ConformalSpec`'s kept the promise.

    A block is held out with its `embargo` band of dates on either side,
    and every remaining row whose label bars intersect the block's is
    purged. Blocks left with nothing to fit on, or nothing to score, are
    dropped rather than returned empty, so the caller may get fewer than
    `n_folds` pairs and should say so if that matters.

    Masks, not indices: the engine's fold machinery is mask-based while
    sklearn's `cv` takes indices, so each caller converts at its own door.
    """
    row_dates = np.asarray(row_dates)
    # np.unique, not sorted(set(...).tolist()): .tolist() on a datetime64
    # array yields datetime.datetime objects, and a panel that recorded no
    # label ends carries an object array of None whose comparison against
    # datetime64 degrades to False (purge nothing, which is honest) but
    # against datetime.datetime raises. Keep the dtype the panel came with.
    dates = np.unique(row_dates)
    if len(dates) < n_folds:
        raise ValidationError(
            f"holding out {n_folds} date blocks needs at least {n_folds} "
            f"distinct dates in the window; this one has {len(dates)}."
        )
    date_code = np.searchsorted(dates, row_dates)
    out: "list[tuple[np.ndarray, np.ndarray]]" = []
    for block in np.array_split(np.arange(len(dates)), n_folds):
        if block.size == 0:
            continue
        first, last = int(block[0]), int(block[-1])
        in_test = np.zeros(len(dates), dtype=bool)
        in_test[block] = True
        banned = np.zeros(len(dates), dtype=bool)
        banned[max(0, first - embargo) : min(len(dates), last + embargo + 1)] = True
        test_mask = in_test[date_code]
        train_mask = ~banned[date_code]
        train_mask &= ~label_overlap_mask(
            train_mask, row_dates, row_label_end, dates[first], dates[last]
        )
        if not train_mask.any() or not test_mask.any():
            continue
        out.append((train_mask, test_mask))
    return out


class DateBlockCV:
    """
    A scikit-learn `cv` whose folds are contiguous DATE blocks, purged and
    embargoed -- `date_block_splits` behind the interface sklearn expects.

    WHY AN OBJECT AND NOT A LIST OF INDEX PAIRS. The estimator is wrapped
    in `CalibratedClassifierCV` before `_fit` runs, and `_fit` may hand the
    fit fewer rows than it was given: early stopping holds out the last
    dates of the window with an embargo before them, and
    `EarlyStoppingFit.apply` returns `X[fit_rows]`. Indices computed
    against the full window then point past the end -- an MLP fold of 5,900
    rows fitted on 5,200 and sklearn raised `index 5200 is out of bounds`.

    `fit_rows` is an index array and not necessarily a prefix, so this
    carries the DATES and is restricted by the same rows, positionally,
    rather than trusting an offset. `split` refuses a row count it was not
    built for instead of indexing into the wrong window, which is the
    failure that would otherwise be silent.
    """

    def __init__(
        self,
        row_dates: np.ndarray,
        row_label_end: "np.ndarray | None",
        *,
        n_folds: int,
        embargo: int = 0,
    ):
        self.row_dates = np.asarray(row_dates)
        self.row_label_end = (
            None if row_label_end is None else np.asarray(row_label_end)
        )
        self.n_folds = int(n_folds)
        self.embargo = int(embargo)

    def restricted_to(self, rows: np.ndarray) -> "DateBlockCV":
        """The same rule over a subset of the rows, taken positionally."""
        return DateBlockCV(
            self.row_dates[rows],
            None if self.row_label_end is None else self.row_label_end[rows],
            n_folds=self.n_folds,
            embargo=self.embargo,
        )

    def masks(self) -> "list[tuple[np.ndarray, np.ndarray]]":
        return date_block_splits(
            self.row_dates,
            self.row_label_end,
            n_folds=self.n_folds,
            embargo=self.embargo,
        )

    def split(self, X, y=None, groups=None):
        n = len(X)
        if n != len(self.row_dates):
            raise ValidationError(
                f"the calibration folds were cut on {len(self.row_dates)} rows "
                f"and the fit was handed {n}. The blocks would be read off the "
                "wrong window. If early stopping shortened it, the cv has to "
                "be restricted to the rows that remain (see _fit)."
            )
        pairs = self.masks()
        if len(pairs) < 2:
            raise ValidationError(
                f"the calibration map has {len(pairs)} usable date block(s) "
                f"over {len(self.row_dates)} rows at embargo={self.embargo}; a "
                "map averaged over one block is that block's map. Lower "
                "calibration_folds, lower the embargo, or widen train_window "
                "-- and note early stopping takes the last dates of the "
                "window before this is cut."
            )
        for train_mask, test_mask in pairs:
            yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.masks())


def restrict_date_block_cv(estimator: Any, rows: np.ndarray) -> None:
    """Cut a wrapped estimator's date-block folds on `rows` instead.

    A no-op for every estimator that is not calibrated on date blocks, so
    `_fit` can call it unconditionally.
    """
    cv = getattr(estimator, "cv", None)
    if isinstance(cv, DateBlockCV):
        estimator.cv = cv.restricted_to(rows)


def build_splitter(validation_spec: Any) -> Any:
    """Construct the splitter a ValidationSpec asks for."""
    if validation_spec.method == "cpcv":
        return CombinatorialPurgedSplit(
            n_groups=validation_spec.n_splits,
            n_test_groups=validation_spec.n_test_splits,
            embargo=validation_spec.embargo,
        )
    if validation_spec.method == "purged_kfold":
        return PurgedKFoldSplit(
            n_splits=validation_spec.n_splits, embargo=validation_spec.embargo
        )
    return WalkForwardSplit(
        train_window=validation_spec.train_window,
        test_window=validation_spec.test_window,
        embargo=validation_spec.embargo,
        scheme=validation_spec.scheme,
    )
