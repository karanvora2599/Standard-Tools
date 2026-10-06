"""
`score_predictions` tests the headline a run tests (the CHANGELOG entry of
2026-10-04).

Its only verdict was `beats_baseline`, r2 against the constant, and the
shipped reference prompt told an agent to stop when it was false. On a
ranked label that judges the predictions' scale, not their ordering: every
model on the live panel had a negative r2 beside a positive rank IC. The
result now carries `beats_null` and a `headline` block computed by the
run's own test. What these tests hold:

- a prediction that orders the names exactly but on the wrong scale loses
  to the baseline and beats the null, and the note says which is which;
- the block is the run's: on a registered model's predictions, attached to
  their outcomes and scored at the label's horizon, it equals
  `validation_report["headline"]`;
- noise, too few dates, one entity and a classifier each get the answer
  the run would give, worded for scored rather than out-of-sample dates;
- the reference prompt stops on `beats_null`, not `beats_baseline`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import handoff
from standard_quant_tools.modeling.adapters import get_adapter
from standard_quant_tools.modeling.agent.models import (
    AttachModelOutcomesInput,
    BuildModelDatasetInput,
    RunModelExperimentInput,
    ScorePredictionsInput,
)
from standard_quant_tools.modeling.agent.tools import (
    attach_model_outcomes,
    build_model_dataset,
    run_model_experiment,
    score_predictions,
)
from standard_quant_tools.modeling.engine import _headline_report
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.comparison import (
    headline_degrees_of_freedom,
)
from standard_quant_tools.modeling.validation.metrics import cross_sectional_ic

REPO = Path(__file__).resolve().parents[2]


def _frame(*, scale=1.0, signal=1.0, n_entities=12, n_dates=120, seed=0):
    """`prediction` is `scale` times a signal the target loads on by
    `signal`; scale 100 orders the names as well as scale 1 and is a
    hundred times too wide."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2023-01-02", periods=n_dates), n_entities)
    entities = np.tile([f"E{i:02d}" for i in range(n_entities)], n_dates)
    x = rng.normal(size=n_entities * n_dates)
    return pd.DataFrame(
        {
            "date": dates,
            "entity": entities,
            "prediction": scale * x,
            "target": signal * 0.3 * x + rng.normal(size=x.size),
        }
    )


def _score(frame, name, **kwargs):
    ref = handoff.publish(frame, kind="predictions", run_id="score_headline", name=name)
    kwargs.setdefault("task", "regression")
    return score_predictions(ScorePredictionsInput(predictions_ref=ref, **kwargs))


class TestTheTwoVerdicts:
    def test_an_ordering_on_the_wrong_scale_beats_the_null_not_the_baseline(self):
        result = _score(_frame(scale=100.0), "wide", horizon=5)
        assert result.metrics["r2"] < -100
        assert result.beats_baseline is False
        assert result.beats_null is True
        headline = result.headline
        assert headline["metric"] == "cs_rank_ic_mean"
        assert headline["value"] == result.metrics["cs_rank_ic_mean"]
        assert headline["n_dates"] == 120
        assert headline["hac_lag"] is None
        assert headline["hac_degrees_of_freedom"] == headline_degrees_of_freedom(120, 5)
        assert headline["t_stat"] > 2 and headline["p_value"] < 0.05
        note = next(n for n in result.notes if "does not beat baseline_r2" in n)
        assert "`beats_null`" in note
        assert not any("has not learned anything" in n for n in result.notes)

    def test_the_block_is_the_runs_test_on_these_predictions(self):
        frame = _frame(signal=0.2)
        result = _score(frame, "same_test", horizon=5)
        series = cross_sectional_ic(
            frame["target"].to_numpy(),
            frame["prediction"].to_numpy(),
            frame["date"].to_numpy(),
            "spearman",
        )
        expected, _warnings = _headline_report(
            get_adapter("regression"),
            "regression",
            {"cs_rank_ic_mean": result.metrics["cs_rank_ic_mean"]},
            series,
            5,
        )
        assert result.headline == expected

    def test_the_horizon_sets_the_degrees_of_freedom(self):
        """120 dates: floor(0.4 x 120^(2/3)) = 9 frequencies at horizon 1,
        and at most 120 / (3 x 5) = 8 for a 5-bar label."""
        frame = _frame(signal=0.0)
        one = _score(frame, "lag_one")
        five = _score(frame, "lag_five", horizon=5)
        assert one.headline["hac_degrees_of_freedom"] == 9
        assert five.headline["hac_degrees_of_freedom"] == 8
        assert one.headline["hac_lag"] is None and five.headline["hac_lag"] is None
        assert any(
            "reads floor(0.4 n^(2/3)) cosine frequencies of n dates at horizon=1" in n
            for n in one.notes
        )


class TestWhatTheRunWouldSay:
    def test_noise_is_not_distinguishable_and_the_dates_are_scored(self):
        result = _score(_frame(signal=0.0, seed=4), "noise", horizon=5)
        assert result.beats_null is False
        (line,) = [w for w in result.warnings if "distinguishable" in w]
        assert "over 120 scored dates and is not distinguishable from zero" in line
        assert "out-of-sample" not in line

    def test_too_few_dates_is_not_tested(self):
        result = _score(_frame(n_dates=6), "short")
        assert result.beats_null is None
        assert result.headline["n_dates"] == 6 and result.headline["t_stat"] is None
        assert any("6 scored date(s), fewer than the 10" in w for w in result.warnings)

    def test_one_entity_has_no_headline_to_test(self):
        result = _score(_frame(n_entities=1), "single")
        assert result.beats_null is None
        assert result.headline["value"] is None
        assert not any("scored date" in w for w in result.warnings)

    def test_a_classifier_is_compared_with_one_half(self):
        frame = _frame(signal=3.0)
        frame["prediction"] = 1.0 / (1.0 + np.exp(-frame["prediction"]))
        result = _score(frame, "classifier", task="classification")
        assert result.headline["metric"] == "auc"
        assert result.headline["null"] == 0.5
        assert result.headline["value"] == result.metrics["auc"]
        # Tested now, not compared: this asserted `t_stat is None` and
        # `beats_null is (auc > 0.5)`, which is the point comparison as a
        # contract. A True has to be backed by a p-value.
        assert result.headline["t_stat"] is not None
        assert result.beats_null is True
        assert result.headline["p_value"] < 0.05
        assert result.beats_baseline is None


class TestOnARegisteredModel:
    def test_it_reproduces_the_runs_own_headline(self, patched_multi_factory):
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(
                spec=DatasetSpec(
                    universe=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
                    start="2022-01-01",
                    end="2023-12-31",
                    features=[
                        FeatureSpec(id="technical.rsi"),
                        FeatureSpec(id="market.momentum"),
                    ],
                    target=TargetSpec(horizon=5),
                    benchmark="SPY",
                )
            )
        ).dataset_id
        run = run_model_experiment(
            RunModelExperimentInput(
                dataset_id=dataset_id,
                spec=ModelSpec(
                    task="regression",
                    estimator=EstimatorSpec(type="ridge", params={}),
                    validation=ValidationSpec(
                        train_window=150, test_window=30, embargo=5
                    ),
                    random_seed=1,
                ),
            )
        )
        attached = attach_model_outcomes(
            AttachModelOutcomesInput(
                model_id=run.model_id, run_id="score_headline", name="registered"
            )
        )
        scored = score_predictions(
            ScorePredictionsInput(
                predictions_ref=attached.ref,
                task="regression",
                horizon=attached.horizon,
            )
        )
        expected = run.validation_report["headline"]
        assert set(scored.headline) == set(expected)
        for key, value in expected.items():
            if isinstance(value, float):
                assert scored.headline[key] == pytest.approx(value, rel=1e-12), key
            else:
                assert scored.headline[key] == value, key


class TestTheReferencePrompt:
    @pytest.mark.parametrize(
        "path",
        [
            "Implementation/Agent_Model_Backtester.py",
            "Implementation/Anthropic/Agent_Model_Backtester.py",
            "Implementation/Gemini/Agent_Model_Backtester.py",
            "Implementation/OpenAI/Agent_Model_Backtester.py",
        ],
    )
    def test_it_stops_on_the_null_test(self, path):
        text = (REPO / path).read_text(encoding="utf-8")
        assert "beats_null. If this is false, stop and say so" in text
        assert "beats_baseline. If this is false, stop" not in text
