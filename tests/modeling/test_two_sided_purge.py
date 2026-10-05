"""
The label-overlap purge tests BOTH sides of a test block.

An overlap is between two intervals: the bars a training row's label spans,
and the bars the test block's labels span. The second used to be the block's
BARS, which is one side of the comparison -- a block's labels resolve
`horizon` bars past its last date, and training rows sitting in that reach
were kept.

Walk-forward never showed it, because training there is entirely before the
block. Purged k-fold and cpcv put training rows on both sides, and those are
the two methods the engine offers for a panel without a time order to
respect.

Measured at the time of the fix, 100 dates and a 5-bar label at embargo=0:
purged_kfold kept 20 leaked rows across five folds, cpcv kept 100 across
fifteen, walk_forward kept none and its purge count did not move. These
tests assert the zero, and assert the no-op, so neither can regress quietly.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling.validation.walk_forward import (
    CombinatorialPurgedSplit,
    PurgedKFoldSplit,
    WalkForwardSplit,
    block_label_end,
    contiguous_runs,
    label_overlap_mask,
)

HORIZON = 5
N_DATES = 100


@pytest.fixture(scope="module")
def panel():
    """Dates and a label that resolves exactly HORIZON bars ahead, so every
    row that reaches a given block is known before the test runs."""
    dates = pd.bdate_range("2021-01-04", periods=N_DATES)
    row_dates = dates.to_numpy()
    label_end = np.array(
        [dates[min(i + HORIZON, N_DATES - 1)] for i in range(N_DATES)],
        dtype="datetime64[ns]",
    )
    return row_dates, label_end


def _purge_and_leak(splitter, row_dates, label_end):
    """(folds, rows purged, rows left inside a block's label span)."""
    folds = purged = leaked = 0
    for train_pos, test_pos in splitter.split(pd.DatetimeIndex(row_dates)):
        folds += 1
        train_mask = np.zeros(len(row_dates), dtype=bool)
        train_mask[train_pos] = True
        overlaps = np.zeros(len(row_dates), dtype=bool)
        blocks = contiguous_runs(np.asarray(test_pos))
        for first, last in blocks:
            overlaps |= label_overlap_mask(
                train_mask, row_dates, label_end, row_dates[first], row_dates[last]
            )
        purged += int(overlaps.sum())
        surviving = train_mask & ~overlaps
        for first, last in blocks:
            span_end = label_end[first : last + 1].max()
            leaked += int(
                (
                    surviving
                    & (row_dates >= row_dates[first])
                    & (row_dates <= span_end)
                ).sum()
            )
    return folds, purged, leaked


@pytest.mark.parametrize(
    "name,splitter",
    [
        ("purged_kfold", PurgedKFoldSplit(n_splits=5, embargo=0)),
        ("cpcv", CombinatorialPurgedSplit(n_groups=6, n_test_groups=2, embargo=0)),
    ],
)
def test_no_training_row_survives_inside_a_block_label_span(name, splitter, panel):
    """The leak itself, on the two methods that train on both sides.

    At embargo=0, so the purge is the only defence — which is the condition
    the default spec ships with.
    """
    row_dates, label_end = panel
    folds, purged, leaked = _purge_and_leak(splitter, row_dates, label_end)
    assert folds > 1, f"{name}: the fixture produced no folds to check"
    assert purged > 0, f"{name}: nothing was purged, so the check is vacuous"
    assert leaked == 0, (
        f"{name}: {leaked} training rows sit inside a test block's label span. "
        "Their features are built from bars the test labels are still "
        "resolving over, and they are the rows nearest the block."
    )


def test_walk_forward_is_unchanged_by_the_fix(panel):
    """The no-op control.

    Training precedes testing here, so `row_dates <= block_label_end` holds
    for every training row whichever end is used. 35 is the count measured
    before `block_label_end` existed; if this moves, the fix stopped being a
    no-op for walk-forward and the two-sided reach is purging something it
    should not.
    """
    row_dates, label_end = panel
    folds, purged, leaked = _purge_and_leak(
        WalkForwardSplit(train_window=30, test_window=10, embargo=0),
        row_dates,
        label_end,
    )
    assert (folds, purged, leaked) == (7, 35, 0)


def test_block_label_end_reaches_past_the_block(panel):
    """The helper, directly: a block's labels end HORIZON bars past it."""
    row_dates, label_end = panel
    first, last = row_dates[20], row_dates[39]
    assert block_label_end(row_dates, label_end, first, last) == row_dates[44]


def test_block_label_end_falls_back_to_the_last_date(panel):
    """No recorded label end for the block's rows purges what it used to.

    A panel that never recorded label ends must not start purging more than
    the block itself — the honest behaviour is the old one.
    """
    row_dates, _ = panel
    nat = np.full(N_DATES, np.datetime64("NaT"), dtype="datetime64[ns]")
    assert (
        block_label_end(row_dates, nat, row_dates[20], row_dates[39]) == row_dates[39]
    )
    # And a block with no rows at all in the panel.
    empty = np.array([], dtype="datetime64[ns]")
    assert (
        block_label_end(empty, empty, row_dates[20], row_dates[39]) == row_dates[39]
    )


def test_a_row_whose_label_never_resolves_is_not_purged(panel):
    """NaT compares False: no resolved label, nothing to leak.

    This held before the fix and is asserted here because `block_label_end`
    now reads the same column and must not turn a NaT into a purge.
    """
    row_dates, label_end = panel
    ends = label_end.copy()
    ends[10] = np.datetime64("NaT")
    train_mask = np.zeros(N_DATES, dtype=bool)
    train_mask[:20] = True
    mask = label_overlap_mask(
        train_mask, row_dates, ends, row_dates[20], row_dates[39]
    )
    assert not mask[10]
