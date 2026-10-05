"""
The probability calibration map is fitted on purged, embargoed DATE blocks.

It used to pass an integer to `CalibratedClassifierCV`, which means
StratifiedKFold, which splits by ROW. Three things followed, and the first
is the one that hides best:

  * the same date's other entities sat on both sides of the split, so the
    map was fitted on rows contemporaneous with the rows it was mapping;
  * no training row was purged for a label reaching into its block;
  * no embargo band was removed.

Stratification is what it cost. A stratified row split guarantees a class
in every fold BY ignoring time, and date blocks cannot, so an imbalanced
label now refuses rather than calibrating on leaked folds. The refusal says
so.

`ConformalSpec.calibration_folds` already described this discipline for the
interval calibration; two fields carry that name and only one kept it. Both
go through `date_block_splits` now.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.engine import _calibrated
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.walk_forward import (
    DateBlockCV,
    date_block_splits,
    restrict_date_block_cv,
)

HORIZON = 5
PER_DATE = 10
N_DATES = 90
EMBARGO = 5


@pytest.fixture(scope="module")
def rows():
    """A cross-sectional panel: PER_DATE entities on each of N_DATES dates,
    with a label that resolves HORIZON bars ahead."""
    dates = pd.bdate_range("2021-01-04", periods=N_DATES)
    row_dates = np.repeat(dates.to_numpy(), PER_DATE)
    label_end = np.repeat(
        np.array(
            [dates[min(i + HORIZON, N_DATES - 1)] for i in range(N_DATES)],
            dtype="datetime64[ns]",
        ),
        PER_DATE,
    )
    return row_dates, label_end


def _spec(folds=3, embargo=EMBARGO, calibration="isotonic"):
    return ModelSpec(
        task="classification",
        estimator=EstimatorSpec(
            type="random_forest",
            calibration=calibration,
            calibration_folds=folds,
        ),
        validation=ValidationSpec(
            train_window=150, test_window=30, embargo=embargo
        ),
        random_seed=0,
    )


class TestTheFoldsRespectTime:
    def test_no_date_appears_on_both_sides_of_a_split(self, rows):
        """The leak itself. A date in both halves means the map was fitted
        on the same day's other entities."""
        row_dates, label_end = rows
        for train_mask, test_mask in date_block_splits(
            row_dates, label_end, n_folds=3, embargo=EMBARGO
        ):
            shared = set(row_dates[train_mask].tolist()) & set(
                row_dates[test_mask].tolist()
            )
            assert not shared, f"{len(shared)} dates on both sides of a split"

    def test_a_stratified_row_split_does_put_them_on_both_sides(self, rows):
        """The control: what `cv=3` did, so this file states the defect
        rather than only the fix."""
        from sklearn.model_selection import StratifiedKFold

        row_dates, _ = rows
        rng = np.random.default_rng(0)
        y = (rng.random(len(row_dates)) < 0.5).astype(int)
        shared_any = False
        for train_idx, test_idx in StratifiedKFold(n_splits=3).split(row_dates, y):
            if set(row_dates[train_idx].tolist()) & set(row_dates[test_idx].tolist()):
                shared_any = True
        assert shared_any, (
            "a stratified row split kept the dates apart on this panel, so "
            "the control no longer demonstrates what it was written for"
        )

    def test_the_embargo_band_is_removed(self, rows):
        """Dates within EMBARGO either side of a block are in neither half."""
        row_dates, label_end = rows
        axis = np.unique(row_dates)
        for train_mask, test_mask in date_block_splits(
            row_dates, label_end, n_folds=3, embargo=EMBARGO
        ):
            held = np.unique(row_dates[test_mask])
            first = int(np.searchsorted(axis, held[0]))
            last = int(np.searchsorted(axis, held[-1]))
            banned = set(
                axis[max(0, first - EMBARGO) : min(len(axis), last + EMBARGO + 1)]
                .tolist()
            )
            assert not (set(row_dates[train_mask].tolist()) & banned)

    def test_a_label_reaching_into_the_block_is_purged(self, rows):
        """With embargo=0 the purge is the only defence, and it still holds
        — the same two-sided rule the outer folds use."""
        row_dates, label_end = rows
        for train_mask, test_mask in date_block_splits(
            row_dates, label_end, n_folds=3, embargo=0
        ):
            held = np.unique(row_dates[test_mask])
            reach = label_end[test_mask].max()
            inside = (
                train_mask
                & (row_dates >= held[0])
                & (row_dates <= reach)
            )
            assert not inside.any(), (
                f"{int(inside.sum())} training rows sit inside the block's "
                "label span"
            )

    def test_fewer_distinct_dates_than_folds_is_refused(self):
        """Not an sklearn error from three frames down."""
        dates = np.repeat(pd.bdate_range("2021-01-04", periods=2).to_numpy(), 5)
        with pytest.raises(ValidationError, match="distinct dates"):
            date_block_splits(dates, None, n_folds=3)

    def test_no_label_ends_recorded_purges_nothing(self, rows):
        """A panel that never recorded label ends must still split, on the
        embargo alone — purging nothing is the honest behaviour."""
        row_dates, _ = rows
        splits = date_block_splits(row_dates, None, n_folds=3, embargo=EMBARGO)
        assert len(splits) == 3
        for train_mask, test_mask in splits:
            assert train_mask.any() and test_mask.any()


class TestWhatCalibratedBuilds:
    def test_the_cv_is_date_blocks_and_not_an_integer(self, rows):
        """The regression this file exists for."""
        from sklearn.ensemble import RandomForestClassifier

        row_dates, label_end = rows
        y = np.tile([0, 1], len(row_dates) // 2)
        wrapped = _calibrated(
            RandomForestClassifier(n_estimators=5, random_state=0),
            _spec(),
            len(y),
            y=y,
            row_dates=row_dates,
            row_label_end=label_end,
        )
        assert isinstance(wrapped.cv, DateBlockCV)
        assert wrapped.cv.embargo == EMBARGO
        assert wrapped.cv.get_n_splits() == 3

    def test_no_calibration_still_returns_the_estimator_untouched(self):
        """The no-op path takes no rows, so every existing spec is
        unaffected — including callers that pass none."""
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier()
        assert _calibrated(estimator, _spec(calibration="none"), 1000) is estimator

    def test_a_single_class_fold_is_refused_by_name(self, rows):
        """Date blocks cannot guarantee a class per fold. The refusal names
        the count and the trade rather than letting sklearn fail inside a
        joblib worker."""
        from sklearn.ensemble import RandomForestClassifier

        row_dates, label_end = rows
        # All of one class until late in the window, so an early block's
        # training rows are single-class.
        y = (row_dates >= np.unique(row_dates)[70]).astype(int)
        with pytest.raises(ValidationError, match="one class only"):
            _calibrated(
                RandomForestClassifier(n_estimators=5, random_state=0),
                _spec(),
                len(y),
                y=y,
                row_dates=row_dates,
                row_label_end=label_end,
            )


class TestTheEarlyStoppingInteraction:
    """Early stopping keeps the last dates of the window for itself, so the
    rows the fit sees are a subset and the blocks must be cut on those.

    Measured before this was handled: an MLP fold of 5,900 rows fitted on
    5,200 and sklearn raised `index 5200 is out of bounds for axis 0 with
    size 5200`.
    """

    def test_split_refuses_a_row_count_it_was_not_built_for(self, rows):
        row_dates, label_end = rows
        cv = DateBlockCV(row_dates, label_end, n_folds=3, embargo=EMBARGO)
        with pytest.raises(ValidationError, match="wrong window"):
            list(cv.split(np.zeros((len(row_dates) - 100, 2))))

    def test_restricting_takes_the_dates_by_the_same_rows(self, rows):
        """`fit_rows` is an index array, not a prefix, so the dates are
        taken by it rather than sliced to a length."""
        row_dates, label_end = rows
        cv = DateBlockCV(row_dates, label_end, n_folds=3, embargo=EMBARGO)
        keep = np.flatnonzero(row_dates < np.unique(row_dates)[60])
        narrowed = cv.restricted_to(keep)
        assert len(narrowed.row_dates) == len(keep)
        assert narrowed.row_dates[-1] == row_dates[keep][-1]
        assert narrowed.embargo == EMBARGO
        # And it still splits, on the window that remains.
        assert len(list(narrowed.split(np.zeros((len(keep), 2))))) >= 2

    def test_restricting_is_a_no_op_for_an_uncalibrated_estimator(self):
        """`_fit` calls it unconditionally."""
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier()
        restrict_date_block_cv(estimator, np.arange(5))  # must not raise
        assert not isinstance(getattr(estimator, "cv", None), DateBlockCV)
