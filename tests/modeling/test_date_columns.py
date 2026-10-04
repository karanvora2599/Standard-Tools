"""
A timezone-aware date column is read as datetime64 instants, not boxed into
a Timestamp per row (see the CHANGELOG entry of 2026-10-04).

`Series.to_numpy()` on a `datetime64[ns, UTC]` column builds one
`pd.Timestamp` per row. A ridge walk-forward run on the live 31,680-row
panel made 58 such calls -- the sample index, the preprocessing context,
the plan and the fold loop each converting the panel or a slice of it --
and they were about half of the run. The rows are now read as their UTC
instants (`datetime_values`), which order, compare, group and subtract the
same as the zoned Timestamps did, so every number a run reports is the
same; the calendar labels are still read off the frame, in its own zone.

Pinned here: no Timestamp is built where the conversions were; the index,
the weights on both backends and a whole run on a zoned panel agree to the
bit with the same panel without its zone, in UTC, New York and Tokyo, on
[s] and [ns] columns and with label ends missing; the fold labels stay in
the panel's zone; and a date column and a label-end column of which only
one carries a zone are refused by name, where pandas raised a TypeError
from inside the purge.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.plan import plan_experiment
from standard_quant_tools.modeling.preprocessing import FoldContext
from standard_quant_tools.modeling.preprocessing.base import datetime_values
from standard_quant_tools.modeling.samples import SampleIndex
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
    WeightingSpec,
)
from standard_quant_tools.modeling.validation import weights as weights_module
from standard_quant_tools.modeling.validation.weights import build_sample_weights

ZONES = ["UTC", "America/New_York", "Asia/Tokyo"]


def _panel(
    tz=None, n_dates=120, n_entities=6, horizon=5, seed=3, missing_ends=0, gaps=0
):
    """A long panel in (date, entity) order whose label ends are the date
    `horizon` rows ahead: NaT for the last `missing_ends` dates, and for
    `gaps` rows drawn from anywhere in the sample."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_dates + horizon)
    if tz is not None:
        dates = dates.tz_localize(tz)
    position = np.repeat(np.arange(n_dates), n_entities)
    ends = pd.Series(dates[position + horizon])
    if missing_ends:
        ends[position >= n_dates - missing_ends] = pd.NaT
    if gaps:
        ends[rng.choice(position.size, size=gaps, replace=False)] = pd.NaT
    f1 = rng.normal(size=position.size)
    return pd.DataFrame(
        {
            "date": dates[position],
            "entity": np.tile([f"E{i}" for i in range(n_entities)], n_dates),
            "f1": f1,
            "f2": rng.normal(size=position.size),
            "target": 0.3 * f1 + rng.normal(size=position.size),
            "label_end_date": ends.array,
        }
    )


def _naive(panel):
    """The same panel with its zone dropped: the UTC wall times."""
    out = panel.copy()
    for column in ("date", "label_end_date"):
        if isinstance(out[column].dtype, pd.DatetimeTZDtype):
            out[column] = out[column].dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def _dataset(panel):
    return {
        "panel": panel,
        "feature_ids": ["f1", "f2"],
        "target_id": "forward_return:5",
        "data_hash": "dates",
    }


@pytest.fixture
def no_timestamps(monkeypatch):
    """Fail the moment a date array as long as a panel is boxed into
    Timestamps. The sorted date axis -- one entry per date, read for the
    fold labels -- may still be."""
    real = pd.arrays.DatetimeArray.__iter__

    def boxed(self):
        if len(self) > 200:
            raise AssertionError(f"a Timestamp was built for each of {len(self)} rows")
        return real(self)

    monkeypatch.setattr(pd.arrays.DatetimeArray, "__iter__", boxed)


class TestTheValues:
    @pytest.mark.parametrize("tz", ZONES)
    @pytest.mark.parametrize("unit", ["s", "ns"])
    def test_a_zoned_column_is_its_utc_instants(self, tz, unit):
        column = pd.Series(
            pd.date_range("2024-03-08", periods=72, freq=pd.Timedelta(hours=1), tz=tz)
        ).astype(f"datetime64[{unit}, {tz}]")
        values = datetime_values(column)
        assert values.dtype == np.dtype(f"datetime64[{unit}]")
        expected = column.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
        assert values.tobytes() == expected.tobytes()
        # A fresh array: writing to it does not reach the frame.
        values[0] = np.datetime64("2000-01-01")
        assert column.iloc[0].year == 2024

    def test_a_naive_column_is_what_to_numpy_returns(self):
        column = pd.Series(pd.bdate_range("2024-01-01", periods=5))
        assert datetime_values(column).tobytes() == column.to_numpy().tobytes()

    def test_reading_the_rows_builds_no_timestamp(self, no_timestamps):
        panel = _panel("UTC")
        index = SampleIndex.from_frame(panel)
        context = FoldContext.from_frame(panel)
        assert index.dates.dtype.kind == index.label_end.dtype.kind == "M"
        assert context.dates.dtype.kind == "M"
        dates = pd.DatetimeIndex(np.unique(index.dates)).tz_localize("UTC")
        spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge"),
            validation=ValidationSpec(train_window=40, test_window=20),
        )
        assert plan_experiment(spec, dates, panel=panel).n_purged > 0


class TestTheIndexAndTheWeights:
    @pytest.mark.parametrize("tz", ZONES)
    def test_the_index_of_a_zoned_panel_is_the_naive_panel_s(self, tz):
        panel = _panel(tz, missing_ends=3)
        zoned, naive = SampleIndex.from_frame(panel), SampleIndex.from_frame(
            _naive(panel)
        )
        assert zoned.dates.tobytes() == naive.dates.tobytes()
        assert zoned.label_end.tobytes() == naive.label_end.tobytes()

    @pytest.mark.parametrize("native", [True, False])
    @pytest.mark.parametrize("tz", ZONES)
    @pytest.mark.parametrize(
        "method", ["label_uniqueness", "time_decay", "uniqueness_and_time_decay"]
    )
    def test_the_weights_agree_to_the_bit_on_both_backends(
        self, monkeypatch, native, tz, method
    ):
        """With label ends missing at the end of the sample and inside it.
        The Python backend, handed the zoned column as Timestamps, searched
        an object array for each label end, and numpy's search, which
        narrows its range from the previous key, went wrong on the keys
        after a missing one: the weights drifted from the kernel's by up to
        1.36 (mean 1). On instants it agrees with the kernel to the bit."""
        if native and not weights_module.HAS_CPP:
            pytest.skip("native extension not built")
        monkeypatch.setattr(weights_module, "HAS_CPP", native)
        panel = _panel(tz, missing_ends=3, gaps=9).sample(frac=1.0, random_state=2)
        zoned = SampleIndex.from_frame(panel)
        naive = SampleIndex.from_frame(_naive(panel))
        args = (method,)
        got = build_sample_weights(
            *args, zoned.dates, zoned.label_end, zoned.entities, 30.0
        )
        want = build_sample_weights(
            *args, naive.dates, naive.label_end, naive.entities, 30.0
        )
        assert got.tobytes() == want.tobytes()


class TestARun:
    SPEC = ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge"),
        validation=ValidationSpec(train_window=60, test_window=20, embargo=2),
        weighting=WeightingSpec(method="uniqueness_and_time_decay", half_life_days=30),
    )

    @staticmethod
    def _labels(result):
        keys = ("train_start", "train_end", "test_start", "test_end")
        return [[f[k] for k in keys] for f in result["validation_report"]["folds"]]

    @pytest.mark.parametrize("tz", ZONES)
    def test_a_zoned_panel_runs_as_its_utc_twin(self, tz):
        """Every number the run reports, and the out-of-sample predictions
        to the bit, are those of the same instants without a zone. The
        fold labels are the panel's own calendar dates: in Tokyo a midnight
        is the previous day in UTC, and the labels did not move to it."""
        panel = _panel(tz)
        zoned = run_experiment(_dataset(panel), self.SPEC, "ds")
        twin = run_experiment(_dataset(_naive(panel)), self.SPEC, "ds")
        assert zoned["oos_metrics"] == twin["oos_metrics"]
        a = pd.read_parquet(zoned["oos_predictions_uri"])["prediction"].to_numpy()
        b = pd.read_parquet(twin["oos_predictions_uri"])["prediction"].to_numpy()
        assert a.tobytes() == b.tobytes()
        local = panel.assign(
            date=panel["date"].dt.tz_localize(None),
            label_end_date=panel["label_end_date"].dt.tz_localize(None),
        )
        in_zone = run_experiment(_dataset(local), self.SPEC, "ds", register=False)
        assert self._labels(zoned) == self._labels(in_zone)
        assert self._labels(zoned)[0][0] == "2023-01-02"

    def test_a_date_and_a_label_end_of_which_one_has_a_zone_are_refused(self):
        panel = _panel("UTC")
        panel["label_end_date"] = panel["label_end_date"].dt.tz_localize(None)
        with pytest.raises(ValidationError, match="timezone-aware .* timezone-naive"):
            run_experiment(_dataset(panel), self.SPEC, "ds", register=False)
