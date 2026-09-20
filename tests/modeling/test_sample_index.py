"""
The sample index: what a row IS, carried beside the arrays.

The weights, the adapters and the conformal calibration were reading
dates, entities and label ends off whichever frame slice was in hand.
`SampleIndex` is that metadata written once, in row order, taken and
reordered with the same masks as `X`. The tests plant a frame whose rows
are deliberately out of date order and check that the ranking adapter
reorders the index with the matrix, that the weights read off the index
are the weights read off the frame, and that every adapter declares the
one input kind that exists.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.adapters import (
    FitArrays,
    available_tasks,
    get_adapter,
)
from standard_quant_tools.modeling.capabilities import estimator_capabilities
from standard_quant_tools.modeling.engine import _fold_sample_weights
from standard_quant_tools.modeling.samples import SampleIndex
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
    WeightingSpec,
)
from standard_quant_tools.modeling.validation.weights import build_sample_weights


def _frame(n_dates=6, entities=("B", "A", "C")):
    dates = pd.bdate_range("2024-01-01", periods=n_dates)
    rows = []
    for date in dates:
        for entity in entities:
            rows.append({"date": date, "entity": entity, "f": np.random.rand()})
    frame = pd.DataFrame(rows)
    # Deliberately NOT in (date, entity) order: reversed.
    frame = frame.iloc[::-1].reset_index(drop=True)
    frame["label_end_date"] = frame["date"] + pd.Timedelta(days=3)
    frame["target"] = np.arange(len(frame), dtype=float)
    return frame


class TestTheIndex:
    def test_it_is_read_off_a_frame_in_row_order(self):
        frame = _frame()
        index = SampleIndex.from_frame(frame)
        assert len(index) == index.n == len(frame)
        np.testing.assert_array_equal(index.dates, frame["date"].to_numpy())
        np.testing.assert_array_equal(index.entities, frame["entity"].to_numpy())
        np.testing.assert_array_equal(
            index.label_end, frame["label_end_date"].to_numpy()
        )
        without = SampleIndex.from_frame(frame.drop(columns=["label_end_date"]))
        assert without.label_end is None
        assert list(index.context().dates) == list(index.dates)

    def test_take_selects_the_same_rows_a_mask_selects_from_x(self):
        frame = _frame()
        index = SampleIndex.from_frame(frame)
        mask = frame["entity"].to_numpy() == "A"
        taken = index.take(mask)
        assert len(taken) == int(mask.sum())
        assert set(taken.entities) == {"A"}
        order = np.argsort(index.dates, kind="stable")
        reordered = index.take(order)
        assert (
            np.diff(reordered.dates.astype("datetime64[ns]").astype(int)) >= 0
        ).all()

    def test_lengths_must_agree(self):
        with pytest.raises(ValidationError, match="one of each per row"):
            SampleIndex(dates=np.arange(3), entities=np.arange(2))
        with pytest.raises(ValidationError, match="one per row"):
            SampleIndex(
                dates=np.arange(3), entities=np.arange(3), label_end=np.arange(2)
            )
        with pytest.raises(ValidationError, match="date"):
            SampleIndex.from_frame(pd.DataFrame({"x": [1.0]}))


class TestTheAdaptersCarryIt:
    def test_the_base_adapter_passes_the_index_through_unchanged(self):
        frame = _frame()
        index = SampleIndex.from_frame(frame)
        arrays = get_adapter("regression").prepare(
            None, index, frame[["f"]], frame["target"].to_numpy(), None
        )
        assert isinstance(arrays, FitArrays) and arrays.index is index
        assert arrays.X.shape == (len(frame), 1)

    def test_the_ranking_adapter_reorders_the_index_with_the_matrix(self):
        frame = _frame()
        index = SampleIndex.from_frame(frame)
        spec = ModelSpec(
            task="ranking",
            estimator=EstimatorSpec(type="lightgbm_ranker"),
            validation=ValidationSpec(train_window=4, test_window=1, min_folds=1),
        )
        arrays = get_adapter("ranking").prepare(
            spec, index, frame[["f"]], frame["target"].to_numpy(), None
        )
        # Sorted by (date, entity), index and matrix alike.
        expected = frame.sort_values(["date", "entity"], kind="stable")
        np.testing.assert_array_equal(arrays.index.dates, expected["date"].to_numpy())
        np.testing.assert_array_equal(
            arrays.index.entities, expected["entity"].to_numpy()
        )
        np.testing.assert_array_equal(arrays.X[:, 0], expected["f"].to_numpy())
        assert arrays.group.sum() == len(frame)

    def test_every_adapter_declares_the_tabular_kind(self):
        assert all(get_adapter(t).input_kind == "tabular" for t in available_tasks())
        assert {e["input_kind"] for e in estimator_capabilities()} == {"tabular"}


class TestTheWeightsReadTheIndex:
    def test_weights_from_the_index_are_the_weights_from_the_frame(self):
        frame = _frame(n_dates=40)
        spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge"),
            validation=ValidationSpec(train_window=10, test_window=5, min_folds=1),
            weighting=WeightingSpec(
                method="uniqueness_and_time_decay", half_life_days=10
            ),
        )
        from_index = _fold_sample_weights(spec, SampleIndex.from_frame(frame))
        direct = build_sample_weights(
            "uniqueness_and_time_decay",
            frame["date"].to_numpy(),
            frame["label_end_date"].to_numpy(),
            frame["entity"].to_numpy(),
            10,
        )
        np.testing.assert_allclose(from_index, direct)
        unweighted = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge"),
            validation=ValidationSpec(train_window=10, test_window=5, min_folds=1),
        )
        assert _fold_sample_weights(unweighted, SampleIndex.from_frame(frame)) is None
