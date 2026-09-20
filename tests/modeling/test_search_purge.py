"""
The inner hyperparameter search runs under the outer loop's leakage
discipline: the spec's embargo, and a purge of every training row whose
label reaches into the inner test window.

It did neither. `_inner_splitter` hardwired `embargo=0` and the inner
train/test selection never looked at `label_end_date`, so the candidate
that won was the one that scored best on training rows whose labels had
already seen the window they were scored against. The outer OOS metric was
never touched by this -- the outer fold loop has purged on the row's own
label end since the P0 leakage fixes -- but WHICH parameters the outer fold
was then fit with was decided on leaked information.

The tests plant the leak rather than hope to observe it: a panel whose
label ends are known lets every inner fold be checked for a training row
that reaches the test block, and a fit_predict spy records exactly what
each candidate was handed.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    SearchSpec,
    TargetSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.search import (
    _inner_splitter,
    search_best_params,
)
from standard_quant_tools.modeling.validation.walk_forward import (
    label_overlap_mask,
)

HORIZON = 5


def _frame(n_dates: int = 60, entities=("A", "B")) -> pd.DataFrame:
    """A training window whose label ends are exactly `HORIZON` bars ahead,
    so every row that leaks into a given test block is known in advance."""
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    rows = []
    for entity in entities:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "entity": entity,
                    "f": float(i),
                    "target": float(i),
                    "label_end_date": (
                        dates[i + HORIZON] if i + HORIZON < n_dates else pd.NaT
                    ),
                }
            )
    return pd.DataFrame(rows)


def _spy():
    """A fit_predict that records what it was handed and predicts zeros."""
    seen = []

    def fit_predict(params, inner_train, inner_test):
        seen.append((params, inner_train.copy(), inner_test.copy()))
        return np.zeros(len(inner_test)), None

    return fit_predict, seen


def _search(frame, *, embargo, label_end, inner_splits=2):
    fit_predict, seen = _spy()
    _params, report = search_best_params(
        task="regression",
        search_spec=SearchSpec(param_grid={"alpha": [1.0]}, inner_splits=inner_splits),
        base_params={},
        train_frame=frame,
        feature_ids=["f"],
        random_seed=0,
        fit_predict=fit_predict,
        embargo=embargo,
        label_end=label_end,
    )
    return report, seen


class TestTheOverlapRule:
    def test_purges_exactly_the_rows_whose_label_reaches_the_block(self):
        frame = _frame(n_dates=30, entities=("A",))
        row_dates = frame["date"].to_numpy()
        label_end = frame["label_end_date"].to_numpy()
        dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
        # Test block: dates 20..24. A row at position p has label end p+5,
        # so rows 15..19 reach into the block; rows before 15 do not.
        first, last = dates[20], dates[24]
        train_mask = (frame["date"] < first).to_numpy()
        purged = label_overlap_mask(train_mask, row_dates, label_end, first, last)
        assert np.flatnonzero(purged).tolist() == [15, 16, 17, 18, 19]

    def test_a_row_with_no_label_end_is_never_purged(self):
        frame = _frame(n_dates=30, entities=("A",))
        label_end = frame["label_end_date"].to_numpy()
        dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
        train_mask = np.ones(len(frame), dtype=bool)
        # The last HORIZON rows are NaT: they have no resolved label.
        purged = label_overlap_mask(
            train_mask, frame["date"].to_numpy(), label_end, dates[0], dates[-1]
        )
        assert not purged[-HORIZON:].any()

    def test_no_label_end_purges_nothing(self):
        frame = _frame(n_dates=30, entities=("A",))
        dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
        purged = label_overlap_mask(
            np.ones(len(frame), dtype=bool),
            frame["date"].to_numpy(),
            None,
            dates[20],
            dates[24],
        )
        assert not purged.any()


class TestInnerFoldsArePurgedAndEmbargoed:
    def test_no_training_label_reaches_any_inner_test_window(self):
        frame = _frame()
        report, seen = _search(
            frame, embargo=0, label_end=frame["label_end_date"].to_numpy()
        )
        assert report["searched"]
        assert seen, "the spy was never called"
        for _params, inner_train, inner_test in seen:
            first_test = inner_test["date"].min()
            reaching = inner_train["label_end_date"] >= first_test
            assert not reaching.any(), (
                f"{int(reaching.sum())} training rows carry labels that end "
                f"on or after the inner test window's first date {first_test.date()}"
            )

    def test_the_purge_is_reported_and_is_the_planted_count(self):
        """Both entities have HORIZON rows reaching each inner block, so the
        count is exact, not merely positive."""
        frame = _frame()
        report, _seen = _search(
            frame, embargo=0, label_end=frame["label_end_date"].to_numpy()
        )
        assert report["purged_on_label_end"] is True
        assert report["n_train_rows_purged_overlap"] == [2 * HORIZON, 2 * HORIZON]

    def test_the_embargo_separates_every_inner_train_and_test_window(self):
        frame = _frame()
        dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
        report, seen = _search(frame, embargo=3, label_end=None)
        assert report["embargo"] == 3
        for _params, inner_train, inner_test in seen:
            last_train = dates.get_loc(inner_train["date"].max())
            first_test = dates.get_loc(inner_test["date"].min())
            assert first_test - last_train - 1 == 3

    def test_without_label_ends_the_leak_is_reachable_and_reported_as_zero(self):
        """The null case: a panel with no label ends cannot be purged, and
        the report says so rather than claiming a purge that did not run."""
        frame = _frame()
        report, seen = _search(frame, embargo=0, label_end=None)
        assert report["purged_on_label_end"] is False
        assert report["n_train_rows_purged_overlap"] == [0, 0]
        leaked = any(
            (inner_train["label_end_date"] >= inner_test["date"].min()).any()
            for _p, inner_train, inner_test in seen
        )
        assert leaked, "the planted leak should be present when nothing purges it"

    def test_label_end_must_be_one_per_row(self):
        frame = _frame()
        with pytest.raises(ValidationError, match="one per row"):
            _search(frame, embargo=0, label_end=frame["label_end_date"].to_numpy()[:-1])


class TestTheInnerSplitterKeepsItsFoldCount:
    @pytest.mark.parametrize("embargo", [0, 1, 5])
    @pytest.mark.parametrize("inner_splits", [2, 3])
    def test_exactly_inner_splits_folds_with_the_embargo_applied(
        self, embargo, inner_splits
    ):
        n_dates = 60
        splitter = _inner_splitter(n_dates, inner_splits, embargo)
        assert splitter is not None
        folds = list(splitter.split(pd.RangeIndex(n_dates)))
        assert len(folds) == inner_splits
        for train_pos, test_pos in folds:
            assert test_pos[0] - train_pos[-1] - 1 == embargo
        # The last fold ends on the last date, so no dates are wasted.
        assert folds[-1][1][-1] == n_dates - 1

    def test_too_short_declines(self):
        assert _inner_splitter(4, inner_splits=3, embargo=2) is None


class TestEngineThreadsTheDisciplineThrough:
    def _dataset(self):
        return build_dataset(
            DatasetSpec(
                universe=["AAA", "BBB", "CCC"],
                start="2022-01-01",
                end="2023-12-31",
                features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
                target=TargetSpec(horizon=HORIZON),
                benchmark="SPY",
            )
        )

    def _spec(self, embargo):
        return ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={}),
            validation=ValidationSpec(train_window=200, test_window=40, embargo=embargo),
            search=SearchSpec(param_grid={"alpha": [0.1, 10.0]}, inner_splits=2),
            random_seed=1,
        )

    @pytest.mark.parametrize("embargo", [0, 4])
    def test_every_searched_fold_records_the_purge_and_the_embargo(
        self, patched_multi_factory, embargo
    ):
        result = run_experiment(self._dataset(), self._spec(embargo), "ds", register=False)
        reports = [
            r for r in result["validation_report"]["hyperparameter_search"] if r["searched"]
        ]
        assert reports
        for report in reports:
            assert report["embargo"] == embargo
            assert report["purged_on_label_end"] is True
            # Three entities, a five-bar label: every inner block has
            # training rows reaching into it.
            assert all(n > 0 for n in report["n_train_rows_purged_overlap"])
