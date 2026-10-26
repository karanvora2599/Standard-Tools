"""
`attach_model_outcomes(keep_intervals=True)` carries a model's band and
quantiles beside the outcome, so the two tools that read them can be
reached from a registered model.

`score_prediction_intervals` refuses a frame with no outcome and tells the
caller to attach outcomes first, and `backtest_var_forecasts` reads a
model's lower quantile as its VaR against the outcome beside it. But
`attach_model_outcomes` published exactly four columns -- date, entity,
prediction, target -- so the frame it made had no band or quantile left
for either tool to read, and the advice led to a second refusal. Off,
the four columns are unchanged (`test_capability_gaps_bridge.py` pins
them); on, the columns ride along and each value is the model's own.

Planted here:

  * the carried columns are the OOS frame's values on the same (date,
    entity), and date, entity, prediction and target are the four-column
    frame's exactly;
  * `score_prediction_intervals` scores the attached frame, its 90%
    coverage the share of outcomes inside [q05, q95] counted directly;
  * `backtest_var_forecasts` tests the attached q05 as a 95% VaR, its
    exceptions the outcomes below q05 on every fifth date of each entity
    counted directly;
  * a published predictions reference carries its columns the same way;
  * asking for intervals a point model never wrote is refused by name.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.agent.runtimes.research.var_tools import (
    VarBacktestInput,
    backtest_var_forecasts,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.models import AttachModelOutcomesInput
from standard_quant_tools.modeling.agent.statistics_models import (
    ScorePredictionIntervalsInput,
)
from standard_quant_tools.modeling.agent.statistics_tools import (
    score_prediction_intervals,
)
from standard_quant_tools.modeling.agent.tools import attach_model_outcomes
from standard_quant_tools.modeling.ensemble import load_oos_predictions
from standard_quant_tools.modeling.specs import (
    ConformalSpec,
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
)

from .test_capability_gaps_bridge import _dataset, _trained

CARRIED = ["lower", "upper", "q05", "q50", "q95"]


def _quantile_spec() -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(
            type="quantile", params={"alpha": 0.0, "solver": "highs"}
        ),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        quantiles=[0.05, 0.5, 0.95],
        intervals=ConformalSpec(alpha=0.1),
        random_seed=1,
    )


def _attach(run_id: str, **kwargs):
    return attach_model_outcomes(
        AttachModelOutcomesInput(run_id=run_id, name="joined", **kwargs)
    )


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["entity"] = out["entity"].astype(str)
    return out.set_index(["date", "entity"]).sort_index()


class _Trained(NamedTuple):
    model_id: str
    dataset_id: str


@pytest.fixture
def quantile_model(patched_multi_factory) -> _Trained:
    dataset_id = _dataset()
    return _Trained(_trained(dataset_id, _quantile_spec()).model_id, dataset_id)


class TestTheBandRidesAlong:
    def test_the_carried_columns_are_the_models_own(self, quantile_model):
        """Each carried value is the OOS frame's on the same (date,
        entity), and the four shared columns are the plain frame's."""
        model_id = quantile_model.model_id
        plain = _attach("keep_plain", model_id=model_id)
        kept = _attach("keep_bands", model_id=model_id, keep_intervals=True)

        assert plain.columns == ["date", "entity", "prediction", "target"]
        assert kept.columns == ["date", "entity", "prediction", *CARRIED, "target"]
        assert kept.n_rows == plain.n_rows
        assert kept.horizon == plain.horizon

        four = _keyed(handoff.resolve(plain.ref, expect="predictions"))
        joined = _keyed(handoff.resolve(kept.ref, expect="predictions"))
        pd.testing.assert_frame_equal(joined[["prediction", "target"]], four)

        oos = _keyed(load_oos_predictions(model_id, keep_distribution=True))
        expected = oos.loc[joined.index, CARRIED]
        pd.testing.assert_frame_equal(joined[CARRIED], expected)

    def test_score_prediction_intervals_reads_the_attached_frame(self, quantile_model):
        """The refusal's own advice now works: the 90% coverage is the
        share of outcomes inside [q05, q95], counted here directly."""
        kept = _attach(
            "keep_scored", model_id=quantile_model.model_id, keep_intervals=True
        )
        scored = score_prediction_intervals(
            ScorePredictionIntervalsInput(predictions_ref=kept.ref)
        )
        frame = handoff.resolve(kept.ref, expect="predictions").dropna(
            subset=["target", *CARRIED]
        )
        y = frame["target"].to_numpy(dtype=float)
        inside = (y >= frame["q05"].to_numpy()) & (y <= frame["q95"].to_numpy())
        assert scored.quantile_coverage["quantile_coverage_90"] == pytest.approx(
            float(np.mean(inside)), abs=1e-12
        )
        band = (y >= frame["lower"].to_numpy()) & (y <= frame["upper"].to_numpy())
        assert scored.interval_coverage == pytest.approx(
            float(np.mean(band)), abs=1e-12
        )

    def test_backtest_var_forecasts_reads_the_lower_quantile(self, quantile_model):
        """q05 is a 95% VaR; on a five-bar target each entity is thinned to
        every fifth of its own dates, and an exception is an outcome below
        q05 -- counted here without the tool."""
        kept = _attach(
            "keep_var", model_id=quantile_model.model_id, keep_intervals=True
        )
        assert kept.horizon == 5
        result = backtest_var_forecasts(
            VarBacktestInput(
                predictions_ref=kept.ref, quantile_column="q05", horizon=kept.horizon
            )
        )
        frame = handoff.resolve(kept.ref, expect="predictions")
        exceptions = 0
        observations = 0
        for _entity, rows in frame.groupby("entity", sort=True):
            thinned = rows.sort_values("date").iloc[:: kept.horizon]
            thinned = thinned.dropna(subset=["target", "q05"])
            observations += len(thinned)
            exceptions += int((thinned["target"] < thinned["q05"]).sum())
        assert result.n_rows_before_thinning == kept.n_rows
        assert result.n_exceptions == exceptions
        assert result.n_observations == observations


class TestAReferenceCarriesItsColumnsToo:
    def test_a_published_frame_keeps_its_band(self, quantile_model):
        """The reference path carries the same columns as the model path,
        with the dataset named because a reference has no manifest."""
        model_id = quantile_model.model_id
        oos = load_oos_predictions(model_id, keep_distribution=True)
        ref = handoff.publish(
            oos, "predictions", "keep_ref", "published", producer="test.fixture"
        )
        from_model = _attach("keep_ref_model", model_id=model_id, keep_intervals=True)
        from_ref = _attach(
            "keep_ref_joined",
            predictions_ref=ref,
            dataset_id=quantile_model.dataset_id,
            keep_intervals=True,
        )
        assert from_ref.columns == from_model.columns
        pd.testing.assert_frame_equal(
            _keyed(handoff.resolve(from_ref.ref, expect="predictions")),
            _keyed(handoff.resolve(from_model.ref, expect="predictions")),
        )


class TestNothingToCarryIsRefused:
    def test_a_point_model_is_refused_by_name(self, patched_multi_factory):
        """A ridge model writes no band and no quantile; asking for them is
        refused rather than answered with the plain four columns."""
        model = _trained(_dataset())
        with pytest.raises(ValidationError, match="keep_intervals=True"):
            _attach("keep_point", model_id=model.model_id, keep_intervals=True)
