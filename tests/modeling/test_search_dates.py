"""
The hyperparameter search reads a zoned panel's dates as instants (see the
CHANGELOG entry of 2026-10-04).

The run path read its date columns as UTC instants, but the search still
converted them with `to_numpy()`, which on a timezone-aware column builds a
`pd.Timestamp` per row: the row dates and label ends of every training
window it searched, and the test dates of every (candidate, inner fold)
score. A grid search on the live 31,680-row panel spent about 0.76 s of
its 1.9 s there. `search_best_params` now reads the dates once, as
instants, whatever form `label_end` arrives in.

Pinned here: no Timestamp is built per row, in the search or in a run that
searches; the search on a zoned panel reports exactly what it reports on
the same instants without a zone (best parameters, every candidate's score,
the purge counts), in UTC, New York and Tokyo, with `label_end` passed as
the column, as Timestamps, or as Timestamps in several zones; the frames
handed to `fit_predict` keep their zone; and a date column and label ends
of which only one carries a zone, or label ends mixing zoned and naive
values, are refused by name.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    SearchSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.search import search_best_params

ZONES = ["UTC", "America/New_York", "Asia/Tokyo"]
HORIZON = 5


def _panel(tz="UTC", n_dates=90, n_entities=6, seed=11):
    """A long panel stamped at 16:00 local time whose label ends are the
    date `HORIZON` rows ahead, NaT for the last `HORIZON` dates."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02 16:00", periods=n_dates)
    if tz is not None:
        dates = dates.tz_localize(tz)
    position = np.repeat(np.arange(n_dates), n_entities)
    ends = pd.Series(dates[np.minimum(position + HORIZON, n_dates - 1)])
    ends[position + HORIZON >= n_dates] = pd.NaT
    f = rng.normal(size=position.size)
    g = rng.normal(size=position.size)
    return pd.DataFrame(
        {
            "date": dates[position],
            "entity": np.tile([f"E{i}" for i in range(n_entities)], n_dates),
            "f": f,
            "g": g,
            "target": 0.4 * f - 0.2 * g + rng.normal(size=position.size),
            "label_end_date": ends.array,
        }
    )


def _naive(panel):
    """The same panel with its zone dropped: the UTC wall times."""
    out = panel.copy()
    for column in ("date", "label_end_date"):
        out[column] = out[column].dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def _search(frame, label_end, seen=None, scoring="cs_rank_ic"):
    """A grid whose candidates rank the rows differently, so every
    candidate's score is its own and depends on which rows share a date."""

    def fit_predict(params, inner_train, inner_test, fold_index):
        if seen is not None:
            seen.append((inner_train["date"].dtype, inner_test["date"].dtype))
        prediction = inner_test["f"] + params["alpha"] * inner_test["g"]
        return prediction.to_numpy(), None

    extra = {"turnover_penalty": 0.5} if scoring.endswith("turnover") else {}
    return search_best_params(
        task="regression",
        search_spec=SearchSpec(
            param_grid={"alpha": [-1.0, -0.25, 0.0, 0.5, 2.0]},
            inner_splits=3,
            scoring=scoring,
            **extra,
        ),
        base_params={},
        train_frame=frame,
        feature_ids=["f", "g"],
        random_seed=0,
        fit_predict=fit_predict,
        embargo=2,
        label_end=label_end,
    )


@pytest.fixture
def no_timestamps(monkeypatch):
    """Fail the moment a date array as long as a training window is boxed
    into Timestamps. A run's sorted date axis -- one entry per date, read
    for the fold labels -- may still be."""
    real = pd.arrays.DatetimeArray.__iter__

    def boxed(self):
        if len(self) > 200:
            raise AssertionError(f"a Timestamp was built for each of {len(self)} rows")
        return real(self)

    monkeypatch.setattr(pd.arrays.DatetimeArray, "__iter__", boxed)


class TestTheSearchBuildsNoTimestamp:
    def test_the_search_reads_a_zoned_window_as_instants(self, no_timestamps):
        frame = _panel("America/New_York")
        _params, report = _search(frame, frame["label_end_date"])
        assert report["searched"]
        assert min(report["n_train_rows_purged_overlap"]) > 0

    def test_a_run_that_searches_reads_its_folds_as_instants(self, no_timestamps):
        """The folds and the refit each search their training rows, and the
        engine hands the label-end column itself to the search."""
        panel = _panel("UTC", n_dates=120)
        spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge"),
            validation=ValidationSpec(train_window=60, test_window=20, embargo=2),
            search=SearchSpec(param_grid={"alpha": [0.1, 10.0]}, inner_splits=2),
        )
        dataset = {
            "panel": panel,
            "feature_ids": ["f", "g"],
            "target_id": "forward_return:5",
            "data_hash": "search_dates",
        }
        result = run_experiment(dataset, spec, "ds", register=False)
        searches = result["validation_report"]["hyperparameter_search"]
        assert searches and all(s["searched"] for s in searches)


class TestAZonedWindowSearchesAsItsUtcTwin:
    @pytest.mark.parametrize("tz", ZONES)
    @pytest.mark.parametrize("scoring", ["cs_rank_ic", "cs_rank_ic_net_of_turnover"])
    def test_every_score_and_purge_is_the_twin_s(self, tz, scoring):
        """The best parameters, every candidate's score to the bit, and the
        rows each inner fold purged. In Tokyo the 16:00 stamps are 07:00
        UTC the same day; in New York they are 21:00 UTC."""
        frame = _panel(tz)
        twin = _naive(frame)
        params, report = _search(frame, frame["label_end_date"], scoring=scoring)
        twin_params, twin_report = _search(
            twin, twin["label_end_date"].to_numpy(), scoring=scoring
        )
        assert min(report["n_train_rows_purged_overlap"]) > 0
        assert params == twin_params
        assert report == twin_report
        scores = [c["score"] for c in report["candidates"]]
        assert len(set(scores)) == len(scores)

    @pytest.mark.parametrize("tz", ZONES)
    def test_label_ends_as_timestamps_are_read_as_the_column(self, tz):
        """`to_numpy()` of a zoned column -- an object array of Timestamps,
        NaT included -- is what the engine passed and what a direct caller
        may still pass; it is read as the column is."""
        frame = _panel(tz)
        _p, from_column = _search(frame, frame["label_end_date"])
        objects = frame["label_end_date"].to_numpy()
        assert objects.dtype == object
        _p, from_objects = _search(frame, objects)
        _p, from_list = _search(frame, list(objects))
        assert from_objects == from_column
        assert from_list == from_column

    def test_label_ends_in_several_zones_are_read_as_their_instants(self):
        """Timestamps from different zones in one array, which pandas will
        not hold as one zoned column, compare as the instants they name."""
        frame = _panel("UTC")
        ends = frame["label_end_date"]
        mixed = np.array(
            [
                end.tz_convert("Asia/Tokyo") if i % 2 else end
                for i, end in enumerate(ends)
            ],
            dtype=object,
        )
        _p, from_mixed = _search(frame, mixed)
        _p, from_column = _search(frame, ends)
        assert from_mixed == from_column

    def test_fit_predict_is_handed_the_frames_in_their_zone(self):
        seen = []
        frame = _panel("Asia/Tokyo")
        _search(frame, frame["label_end_date"], seen=seen)
        assert seen
        for train_dtype, test_dtype in seen:
            assert str(train_dtype.tz) == str(test_dtype.tz) == "Asia/Tokyo"


class TestAZoneOnOneSideOnlyIsRefused:
    def test_zoned_dates_with_naive_label_ends(self):
        """Naive label ends -- UTC instants, or a local wall clock -- beside
        a zoned date column name no instant the purge can compare. pandas
        raised `TypeError: Cannot compare tz-naive and tz-aware` from the
        purge; compared as instants, a local wall clock would be read as
        UTC and a row whose label ends on an inner test date kept."""
        frame = _panel("America/New_York")
        naive_ends = _naive(frame)["label_end_date"].to_numpy()
        with pytest.raises(
            ValidationError, match="'date' column is timezone-aware and label_end"
        ):
            _search(frame, naive_ends)

    def test_naive_dates_with_zoned_label_ends(self):
        frame = _panel(None)
        zoned_ends = frame["label_end_date"].dt.tz_localize("UTC")
        with pytest.raises(
            ValidationError, match="timezone-naive and label_end is timezone-aware"
        ):
            _search(frame, zoned_ends)

    def test_label_ends_mixing_zoned_and_naive_values(self):
        frame = _panel("UTC")
        mixed = frame["label_end_date"].to_numpy().copy()
        mixed[3] = mixed[3].tz_localize(None)
        with pytest.raises(ValidationError, match="mixes timezone-aware and naive"):
            _search(frame, mixed)

    def test_label_ends_with_nothing_in_them_are_not_refused(self):
        """No label end at all names no zone, and purges nothing."""
        frame = _panel("UTC")
        nothing = np.array([pd.NaT] * len(frame), dtype=object)
        _params, report = _search(frame, nothing)
        assert report["n_train_rows_purged_overlap"] == [0, 0, 0]
