"""
Two things `score_predictions` claimed and could not do.

The effective sample size "adjusted for overlapping forward returns" was
computed with horizon=1, so it equalled the row count; a survival task fell
through to the ranking metrics, so a duration was scored with NDCG. The
horizon is now an input and survival has its own branch, both planted.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.models import ScorePredictionsInput
from standard_quant_tools.modeling.agent.tools import score_predictions


def _frame(n_entities=12, n_dates=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2023-01-02", periods=n_dates), n_entities)
    entities = np.tile([f"E{i:02d}" for i in range(n_entities)], n_dates)
    signal = rng.normal(size=n_entities * n_dates)
    return pd.DataFrame(
        {
            "date": dates,
            "entity": entities,
            "prediction": signal,
            "target": signal + rng.normal(scale=0.5, size=signal.size),
        }
    )


def _publish(frame, name):
    return handoff.publish(frame, kind="predictions", run_id="score_fixes", name=name)


class TestTheHorizon:
    def test_the_horizon_deflates_the_effective_sample_size(self):
        ref = _publish(_frame(), "horizon")
        one = score_predictions(
            ScorePredictionsInput(predictions_ref=ref, task="regression")
        )
        five = score_predictions(
            ScorePredictionsInput(predictions_ref=ref, task="regression", horizon=5)
        )
        assert one.effective_sample_size == pytest.approx(one.n_observations)
        assert five.effective_sample_size < one.effective_sample_size / 2
        assert any("NON-overlapping" in note for note in one.notes)
        assert not any("NON-overlapping" in note for note in five.notes)
        with pytest.raises(Exception):
            ScorePredictionsInput(predictions_ref=ref, task="regression", horizon=0)


class TestSurvival:
    def _survival_frame(self, seed=1):
        rng = np.random.default_rng(seed)
        frame = _frame(seed=seed)
        risk = frame["prediction"].to_numpy()
        # Higher risk, sooner event: the duration is an exponential whose
        # rate rises with the risk, censored by an independent clock.
        event_time = rng.exponential(1.0 / np.exp(risk))
        censor_time = rng.exponential(1.5, size=risk.size)
        frame["target"] = np.minimum(event_time, censor_time) + 1e-3
        frame["event"] = (event_time <= censor_time).astype(float)
        return frame

    def test_a_survival_task_is_scored_on_concordance_not_ndcg(self):
        ref = _publish(self._survival_frame(), "survival")
        result = score_predictions(
            ScorePredictionsInput(predictions_ref=ref, task="survival")
        )
        assert result.metrics["concordance"] > 0.6
        assert "cs_concordance_mean" in result.metrics
        assert not any(key.startswith("ndcg") for key in result.metrics)
        assert result.cross_sectional_ic == {}
        assert any("RISK" in note for note in result.notes)

    def test_the_event_column_is_required_and_binary(self):
        frame = self._survival_frame()
        ref = _publish(frame.drop(columns=["event"]), "no_event")
        with pytest.raises(ValidationError, match="event column"):
            score_predictions(
                ScorePredictionsInput(predictions_ref=ref, task="survival")
            )
        bad = frame.copy()
        bad.loc[bad.index[:3], "event"] = 2.0
        ref = _publish(bad, "bad_event")
        with pytest.raises(ValidationError, match="0 or 1"):
            score_predictions(
                ScorePredictionsInput(predictions_ref=ref, task="survival")
            )
        renamed = frame.rename(columns={"event": "observed"})
        ref = _publish(renamed, "renamed_event")
        result = score_predictions(
            ScorePredictionsInput(
                predictions_ref=ref, task="survival", event_column="observed"
            )
        )
        assert result.metrics["concordance"] > 0.6
