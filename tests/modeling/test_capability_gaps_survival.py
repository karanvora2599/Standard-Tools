"""
The survival curve a registered model computed and never returned.

Until now a survival model was a regression model with extra steps: it
trained end to end, reported a concordance and an integrated Brier score,
and the only number a caller could get back out of it was `predict` -- a
risk that ORDERS the cross-section. That answers who fills first. It
cannot answer how likely one particular order is to still be resting at
t, which is the number a desk sizes on, and which the fitted baseline
hazard persisted with every Cox-family model has been able to produce all
along.

Planted so the two questions stay separable. The panel's duration is
driven by the features through a known log-hazard, so the fitted model
has a real ordering to recover; the curves are then read at a grid of the
model's own baseline knots. Under proportional hazards the medians must
come out in the reverse order of the risks -- the highest-risk name
reaches 0.5 first -- and every curve must fall, never rise, and never
exceed 1. The null cases are here too: an estimator that only ranks is
refused rather than quietly handed back its risk, a grid that stops
before the first baseline knot returns S == 1 with no median and says the
curve was truncated, and a model whose task is not survival is refused by
name.

The factoring these tests also guard: the gates (training-information
cutoff, feature provenance, universe pins, one-date cross-section,
staleness) are now run once for both entry points, so score_model's own
answer -- its result keys, its effective date, its risk ordering -- must
be exactly what it was before the curve existed.
"""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling import scoring as _scoring
from standard_quant_tools.modeling.agent.survival_models import (
    PredictSurvivalCurveInput,
)
from standard_quant_tools.modeling.agent.survival_tools import predict_survival_curve
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.scoring import score_model, survival_curves
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec

#: Six names, because the claim under test is about the SPREAD of the
#: medians across a cross-section: with two rows a "highest risk" and a
#: "lowest risk" are the same pair however the model came out.
UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]

#: The date every curve and every score below is read as of. After the
#: planted panel's last bar, so the training-information cutoff is
#: satisfied the same way test_scoring's regression case satisfies it.
AS_OF = "2023-12-29"

#: The log-hazard planted on the two standardized features. Non-zero on
#: both and opposite in sign, so a model that learned nothing cannot
#: reproduce the ordering by accident.
PLANTED_LOG_HAZARD = (1.6, -0.9)

#: The result keys score_model returned BEFORE the gate-and-matrix build
#: was factored out into `_scoring_context`. Read off its return
#: statement and pinned here: the factoring was supposed to move code,
#: not the answer.
SCORE_MODEL_RESULT_KEYS = {
    "model_id",
    "as_of",
    "features_uri",
    "effective_score_date",
    "staleness_days",
    "predictions_uri",
    "predictions_hash",
    "n_entities",
    "missing_entities",
    "stale_entities",
    "warnings",
    "summary_stats",
    "interval_stats",
}


def _survival_dataset_spec() -> DatasetSpec:
    """The same shape test_scoring trains on -- two entity-scope features,
    a pooled transform, a provider this library can rebuild from -- so the
    curve path is exercised through the real scoring gates rather than
    around them. An externally registered panel cannot be scored at all,
    which is why the survival label is planted onto a built panel instead
    of arriving as one."""
    return DatasetSpec(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )


def _register_a_cox_model(
    dataset_id: str = "ds_planted_time_to_fill",
    seed: int = 7,
) -> str:
    """Build the panel, plant a censored duration on it that the features
    genuinely drive, and register a Cox model fitted on it.

    The duration is exponential with rate exp(beta . z) on the
    standardized features and is censored independently, so roughly three
    rows in ten end the window still waiting -- which is the condition the
    survival task exists for and the one a regression on the duration
    alone gets wrong.
    """
    spec = _survival_dataset_spec()
    built = build_dataset(spec)
    panel = built["panel"].copy()
    feature_ids = built["feature_ids"]

    values = panel[feature_ids].to_numpy(dtype=float)
    standardized = (values - values.mean(axis=0)) / values.std(axis=0)
    log_hazard = (
        PLANTED_LOG_HAZARD[0] * standardized[:, 0]
        + PLANTED_LOG_HAZARD[1] * standardized[:, 1]
    )
    rng = np.random.default_rng(seed)
    event_time = rng.exponential(1.0 / np.exp(log_hazard))
    censor_time = rng.exponential(3.0, size=event_time.size)
    # Shifted off zero so the first baseline knot is comfortably above
    # the grid the truncation case asks about -- and because a survival
    # estimator refuses a non-positive duration outright.
    panel["target"] = np.minimum(event_time, censor_time) + 0.5
    panel["event"] = (event_time <= censor_time).astype(float)

    panel_uri = _artifacts.save_artifact(panel, run_id=dataset_id, name="panel")
    _artifacts.save_json(Path(panel_uri).parent, "dataset_spec", spec.model_dump())

    dataset = {
        "panel": panel,
        "feature_ids": feature_ids,
        # The target id, not the dataset spec's buildable forward return,
        # is what pairs the label with the task: a duration whose event
        # indicator was recorded is a time_to_fill, and the engine refuses
        # to fit it as anything but survival.
        "target_id": "time_to_fill:5",
        "data_hash": built["data_hash"],
        "spec_hash": built["spec_hash"],
        "dataset_spec": spec.model_dump(),
    }
    model_spec = ModelSpec(
        task="survival",
        estimator=EstimatorSpec(type="cox_ph", params={"alpha": 0.0}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )
    return run_experiment(dataset, model_spec, dataset_id=dataset_id)["model_id"]


class _RanksButCannotDate:
    """A survival estimator that orders the cross-section and stops there.

    Stands in for any estimator registered for the survival task that has
    no `predict_survival_function`: it answers "who first" and has nothing
    to say about "by when".
    """

    def __init__(self, inner):
        self._inner = inner

    def predict(self, X):
        return self._inner.predict(X)


class TestTheCurveComesOut:
    def test_the_highest_risk_name_reaches_one_half_before_the_lowest(
        self, patched_multi_factory
    ):
        model_id = _register_a_cox_model()
        result = survival_curves(model_id, AS_OF, UNIVERSE)

        assert result["n_entities"] == len(UNIVERSE)
        assert result["missing_entities"] == []
        assert result["n_baseline_knots"] > 100
        assert len(result["times"]) == 32
        assert result["times"] == sorted(result["times"])

        ranked = sorted(result["per_entity"], key=lambda row: row["risk"])
        lowest, highest = ranked[0], ranked[-1]
        assert highest["risk"] > lowest["risk"]
        assert highest["median_survival"] is not None
        assert lowest["median_survival"] is not None
        # The whole claim of a proportional-hazards fit: one baseline,
        # scaled by the risk, so a higher hazard crosses 0.5 sooner.
        assert highest["median_survival"] < lowest["median_survival"]

        for row in result["per_entity"]:
            curve = np.asarray(row["survival_at_times"], dtype=float)
            assert curve.size == len(result["times"])
            assert (curve <= 1.0 + 1e-12).all()
            assert (curve >= 0.0).all()
            # Non-increasing, to floating tolerance: S(t) is the
            # probability of not having gone YET, so it can only fall.
            assert (np.diff(curve) <= 1e-12).all()
            median = row["median_survival"]
            if median is None:
                assert (curve > 0.5).all()
            else:
                crossing = result["times"].index(median)
                assert curve[crossing] <= 0.5
                assert (curve[:crossing] > 0.5).all()

        mean_curve = np.asarray(result["survival_mean_curve"], dtype=float)
        assert (np.diff(mean_curve) <= 1e-12).all()
        assert any("proportional hazards" in w for w in result["warnings"])

    def test_the_median_is_never_reported_as_the_grid_s_last_point(
        self, patched_multi_factory
    ):
        """A curve that has not crossed 0.5 by the end of the grid returns
        None, not the last time it was looked at: "we stopped here" and
        "it happened here" are opposite claims, and the truncation warning
        says which one this is."""
        model_id = _register_a_cox_model()
        result = survival_curves(model_id, AS_OF, UNIVERSE, times=[0.01, 0.05, 0.2])

        assert result["times"] == [0.01, 0.05, 0.2]
        for row in result["per_entity"]:
            # Every requested time is before the first observed event, so
            # the baseline cumulative hazard is still zero there.
            assert row["survival_at_times"] == [1.0, 1.0, 1.0]
            assert row["median_survival"] is None

        assert any("TRUNCATED" in w for w in result["warnings"])
        assert any("never reach 0.5" in w for w in result["warnings"])

    def test_a_grid_past_the_last_knot_says_the_baseline_is_being_held_flat(
        self, patched_multi_factory
    ):
        model_id = _register_a_cox_model()
        default = survival_curves(model_id, AS_OF, UNIVERSE)
        far = float(default["times"][-1]) * 10.0
        result = survival_curves(
            model_id, AS_OF, UNIVERSE, times=[default["times"][0], far]
        )
        assert any("extrapolation, not an estimate" in w for w in result["warnings"])


class TestWhatItRefuses:
    def test_an_estimator_that_only_ranks_is_refused_with_no_fallback(
        self, patched_multi_factory, monkeypatch
    ):
        model_id = _register_a_cox_model()
        real = _scoring.load_model(model_id)
        monkeypatch.setattr(
            _scoring, "load_model", lambda *a, **kw: _RanksButCannotDate(real)
        )

        with pytest.raises(ValidationError) as excinfo:
            survival_curves(model_id, AS_OF, UNIVERSE)
        message = str(excinfo.value)
        # Names the estimator that cannot, and the ones that can.
        assert "_RanksButCannotDate" in message
        assert "cox_ph" in message
        assert "xgboost_cox" in message and "xgboost_aft" in message
        # And does not offer the risk as a substitute for a probability.
        assert "NOT returned in its place" in message

    def test_an_externally_registered_panel_is_refused_naming_this_door(self, tmp_path):
        """The way a real survival label usually arrives is a panel
        registered from outside -- and such a panel carries no recipe, so
        neither door can rebuild its features at a later date. The
        refusal must name the door that was actually used, not the other
        one."""
        import pandas as pd

        from standard_quant_tools.modeling.agent.models import (
            RegisterExternalPanelInput,
            RunModelExperimentInput,
        )
        from standard_quant_tools.modeling.agent.tools import (
            register_external_panel,
            run_model_experiment,
        )

        n_entities, n_dates = 8, 200
        rng = np.random.default_rng(3)
        features = rng.normal(size=(n_entities * n_dates, 3))
        rate = np.exp(features @ np.array([1.0, -0.5, 0.0]))
        event_time = rng.exponential(1.0 / rate)
        censor_time = rng.exponential(1.5, size=event_time.size)
        frame = pd.DataFrame(
            {
                "date": np.repeat(
                    pd.bdate_range("2021-01-01", periods=n_dates), n_entities
                ),
                "entity": np.tile([f"E{i:02d}" for i in range(n_entities)], n_dates),
                "f1": features[:, 0],
                "f2": features[:, 1],
                "f3": features[:, 2],
                "ttf": np.minimum(event_time, censor_time) + 1e-3,
                "filled": (event_time <= censor_time).astype(float),
            }
        )
        path = tmp_path / "resting_orders.parquet"
        frame.to_parquet(path, index=False)

        registered = register_external_panel(
            RegisterExternalPanelInput(
                path=str(path),
                interval="1s",
                targets=[
                    {
                        "name": "ttf",
                        "column": "ttf",
                        "horizon": 50,
                        "target_type": "time_to_fill",
                        "event_column": "filled",
                    }
                ],
            )
        )
        trained = run_model_experiment(
            RunModelExperimentInput(
                dataset_id=registered.dataset_id,
                spec=ModelSpec(
                    task="survival",
                    estimator=EstimatorSpec(type="cox_ph"),
                    validation=ValidationSpec(
                        train_window=160, test_window=40, embargo=0, min_folds=1
                    ),
                    random_seed=2,
                ),
            )
        )

        with pytest.raises(ValidationError) as excinfo:
            survival_curves(
                trained.model_id,
                "2030-01-02",
                [f"E{i:02d}" for i in range(n_entities)],
            )
        message = str(excinfo.value)
        assert "externally registered" in message
        # The refusal names THIS entry point. It used to be written for
        # one caller only, so a shared gate would have told a curve
        # request to go and fix a score_model call it never made.
        assert "so survival_curves cannot run" in message
        assert "{caller}" not in message

    def test_a_regression_model_is_refused_naming_its_task(self, patched_multi_factory):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_regression_has_no_curve"
        )
        with pytest.raises(ValidationError) as excinfo:
            survival_curves(model_id, AS_OF, ["AAA", "BBB", "CCC"])
        message = str(excinfo.value)
        assert "regression" in message
        assert "score_model" in message

    def test_a_decreasing_time_grid_is_refused_by_the_schema(self):
        with pytest.raises(PydanticValidationError, match="strictly increasing"):
            PredictSurvivalCurveInput(
                model_id="mdl_whatever",
                as_of=AS_OF,
                universe=["AAA"],
                times=[30.0, 10.0, 60.0],
            )
        with pytest.raises(PydanticValidationError, match="non-negative"):
            PredictSurvivalCurveInput(
                model_id="mdl_whatever",
                as_of=AS_OF,
                universe=["AAA"],
                times=[-1.0, 10.0],
            )


class TestThroughTheTool:
    def test_the_curves_cross_the_wire_only_when_they_are_asked_for(
        self, patched_multi_factory
    ):
        model_id = _register_a_cox_model()
        base = dict(model_id=model_id, as_of=AS_OF, universe=UNIVERSE)

        summary = predict_survival_curve(PredictSurvivalCurveInput(**base))
        assert summary.n_entities == len(UNIVERSE)
        assert len(summary.times) == 32
        assert all(row.survival_at_times is None for row in summary.per_entity)
        assert all(row.median_survival is not None for row in summary.per_entity)
        assert len(summary.survival_mean_curve) == len(summary.times)

        full = predict_survival_curve(
            PredictSurvivalCurveInput(**base, include_matrix=True)
        )
        assert all(
            row.survival_at_times is not None
            and len(row.survival_at_times) == len(full.times)
            for row in full.per_entity
        )
        # Same call, same numbers: the switch controls what is RETURNED,
        # not what was computed.
        assert [row.median_survival for row in full.per_entity] == [
            row.median_survival for row in summary.per_entity
        ]

    def test_a_cell_budget_below_the_matrix_refuses_naming_the_product(
        self, patched_multi_factory
    ):
        model_id = _register_a_cox_model()
        base = dict(model_id=model_id, as_of=AS_OF, universe=UNIVERSE, n_times=32)

        with pytest.raises(ValidationError) as excinfo:
            predict_survival_curve(
                PredictSurvivalCurveInput(
                    **base, include_matrix=True, max_matrix_cells=100
                )
            )
        message = str(excinfo.value)
        assert f"{len(UNIVERSE)} entities x 32 times = {len(UNIVERSE) * 32}" in message
        assert "max_matrix_cells=100" in message

        # The same budget does not fire without the matrix: the medians
        # are the decision and they are a handful of numbers.
        capped = predict_survival_curve(
            PredictSurvivalCurveInput(**base, max_matrix_cells=100)
        )
        assert capped.n_entities == len(UNIVERSE)


class TestTheGatesDidNotMove:
    def test_score_model_agrees_on_the_risk_order_and_the_score_date(
        self, patched_multi_factory
    ):
        model_id = _register_a_cox_model()
        curves = survival_curves(model_id, AS_OF, UNIVERSE)
        scored = score_model(model_id=model_id, as_of=AS_OF, universe=UNIVERSE)

        assert curves["effective_score_date"] == scored["effective_score_date"]
        assert curves["n_entities"] == scored["n_entities"]
        assert curves["missing_entities"] == scored["missing_entities"]
        assert curves["stale_entities"] == scored["stale_entities"]

        predictions = _artifacts.load_artifact(scored["predictions_uri"])
        from_score = (
            predictions.sort_values("prediction", ascending=False)["entity"]
            .astype(str)
            .tolist()
        )
        from_curves = [
            row["entity"]
            for row in sorted(
                curves["per_entity"], key=lambda row: row["risk"], reverse=True
            )
        ]
        assert from_curves == from_score

    def test_score_model_s_result_keys_did_not_move(self, patched_multi_factory):
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_result_keys_snapshot"
        )
        result = score_model(
            model_id=model_id, as_of=AS_OF, universe=["AAA", "BBB", "CCC"]
        )
        assert set(result) == SCORE_MODEL_RESULT_KEYS
        assert result["n_entities"] == 3
        assert result["missing_entities"] == []

    def test_the_curve_path_enforces_the_same_staleness_limit(
        self, patched_multi_factory
    ):
        """One gate, spot-checked through the new door: the planted panel
        ends weeks before as_of, so a limit of one day must refuse here
        exactly as it refuses a score -- and name the new caller, not the
        old one."""
        model_id = _register_a_cox_model()
        with pytest.raises(ValidationError) as excinfo:
            survival_curves(model_id, AS_OF, UNIVERSE, max_staleness_days=1)
        assert "survival_curves:" in str(excinfo.value)
        with pytest.raises(ValidationError) as excinfo:
            score_model(
                model_id=model_id,
                as_of=AS_OF,
                universe=UNIVERSE,
                max_staleness_days=1,
            )
        assert "score_model:" in str(excinfo.value)
