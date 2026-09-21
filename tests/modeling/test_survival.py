"""
The survival task: a duration that may be censored, fitted as a risk and
judged on whether it ordered the durations right.

Planted with a known hazard. Each row's event time is exponential with
rate exp(beta . x), beta = (1, -0.5, 0), and a censoring time is drawn
independently, so the label is (min(event, censor), event <= censor). The
truth's own concordance -- the risk beta . x against the same censored
label -- is the ceiling a fitted model can reach, and the tests assert the
Cox fit recovers beta and lands within a few hundredths of it. The other
direction is planted too: a regression on the censored label is refused
by the task check, a panel registered without an event indicator is
refused by name, and an ensemble will not average a hazard with a return.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.adapters import available_tasks, get_adapter
from standard_quant_tools.modeling.agent.models import (
    RegisterExternalPanelInput,
    RunModelExperimentInput,
)
from standard_quant_tools.modeling.agent.tools import (
    register_external_panel,
    run_model_experiment,
)
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.ensemble import _check_tasks
from standard_quant_tools.modeling.estimators import survival as survival_estimators
from standard_quant_tools.modeling.estimators.registry import ESTIMATOR_REGISTRY
from standard_quant_tools.modeling.estimators.survival import CoxPHRegressor
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    SearchSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.targets.registry import (
    TARGET_KINDS,
    targets_for_task,
)
from standard_quant_tools.modeling.tasks import SCORE_TASKS, TASKS
from standard_quant_tools.modeling.validation.survival import (
    concordance_index,
    survival_labels,
    survival_metrics,
)

BETA = np.array([1.0, -0.5, 0.0])
requires_xgboost = pytest.mark.skipif(
    not survival_estimators.HAS_XGBOOST_SURVIVAL, reason="xgboost is not installed"
)


def _planted(n: int, seed: int = 0, censor_scale: float = 1.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    rate = np.exp(X @ BETA)
    event_time = rng.exponential(1.0 / rate)
    censor_time = rng.exponential(censor_scale, size=n)
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(float)
    return X, duration, event


def _planted_dataset(n_entities=25, n_dates=320, seed=1):
    X, duration, event = _planted(n_entities * n_dates, seed=seed)
    dates = np.repeat(pd.bdate_range("2021-01-01", periods=n_dates), n_entities)
    entities = np.tile([f"E{i:02d}" for i in range(n_entities)], n_dates)
    panel = pd.DataFrame(
        {
            "date": dates,
            "entity": entities,
            "f1": X[:, 0],
            "f2": X[:, 1],
            "f3": X[:, 2],
            "target": duration + 1e-3,
            "event": event,
        }
    )
    return {
        "panel": panel,
        "feature_ids": ["f1", "f2", "f3"],
        "target_id": "time_to_fill:50",
        "data_hash": f"survival-{seed}",
    }


def _spec(estimator="cox_ph", params=None, **overrides) -> ModelSpec:
    fields = dict(
        task="survival",
        estimator=EstimatorSpec(type=estimator, params=params or {}),
        validation=ValidationSpec(
            train_window=160, test_window=40, embargo=0, min_folds=1
        ),
        random_seed=2,
    )
    fields.update(overrides)
    return ModelSpec(**fields)


class TestTheTaskExists:
    def test_survival_is_a_task_a_score_task_and_an_adapter(self):
        assert "survival" in TASKS and "survival" in SCORE_TASKS
        assert "survival" in available_tasks()
        assert get_adapter("survival").score_has_scale is False

    def test_time_to_fill_is_a_censored_survival_label(self):
        assert TARGET_KINDS["time_to_fill"].tasks == ("survival",)
        assert targets_for_task("survival") == ("time_to_fill",)
        assert ("survival", "cox_ph") in ESTIMATOR_REGISTRY

    def test_a_regression_on_the_censored_label_is_refused_by_the_task_check(self):
        spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
            validation=ValidationSpec(train_window=160, test_window=40, min_folds=1),
        )
        with pytest.raises(ValidationError, match="time_to_fill"):
            run_experiment(_planted_dataset(), spec, "ds", register=False)

    def test_a_survival_search_scores_on_concordance_only(self):
        with pytest.raises(ValueError, match="concordance"):
            _spec(search=SearchSpec(param_grid={"alpha": [0.0, 1.0]}))
        with pytest.raises(ValueError, match="survival"):
            ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge"),
                validation=ValidationSpec(train_window=160, test_window=40),
                search=SearchSpec(param_grid={"alpha": [1.0]}, scoring="concordance"),
            )

    def test_an_ensemble_will_not_average_a_hazard_with_a_return(self):
        with pytest.raises(ValidationError, match="hazard"):
            _check_tasks({"a": "survival", "b": "regression"}, "rank_mean")
        _check_tasks({"a": "survival", "b": "survival"}, "rank_mean")


class TestTheMetric:
    def test_concordance_counts_comparable_pairs_the_right_way(self):
        # Durations 1 < 2 < 3; the middle row is censored, so it can only
        # be the LATER member of a pair.
        durations = np.array([1.0, 2.0, 3.0])
        events = np.array([1.0, 0.0, 1.0])
        c, pairs = concordance_index(durations, events, np.array([3.0, 2.0, 1.0]))
        assert (c, pairs) == (1.0, 2)  # (1,2) and (1,3); (2,3) is not comparable
        c, _ = concordance_index(durations, events, np.array([1.0, 2.0, 3.0]))
        assert c == 0.0
        c, _ = concordance_index(durations, events, np.array([1.0, 1.0, 1.0]))
        assert c == 0.5
        c, pairs = concordance_index(durations, np.zeros(3), np.array([3.0, 2.0, 1.0]))
        assert np.isnan(c) and pairs == 0

    def test_the_truth_scores_near_its_theoretical_ordering(self):
        X, duration, event = _planted(4000, seed=3)
        c, pairs = concordance_index(duration, event, X @ BETA)
        assert 0.70 < c < 0.85 and pairs > 1_000_000
        metrics = survival_metrics(np.column_stack([duration, event]), X @ BETA)
        assert metrics["concordance"] == pytest.approx(c)
        assert 0.3 < metrics["event_rate"] < 0.9

    def test_labels_need_the_event_column(self):
        frame = pd.DataFrame({"target": [1.0, 2.0]})
        with pytest.raises(ValidationError, match="event"):
            survival_labels(frame)
        frame["event"] = [1, 2]
        with pytest.raises(ValidationError, match="0 or 1"):
            survival_labels(frame)


class TestCoxProportionalHazards:
    def test_it_recovers_the_planted_coefficients(self):
        X, duration, event = _planted(6000, seed=5)
        model = CoxPHRegressor().fit(X, np.column_stack([duration, event]))
        np.testing.assert_allclose(model.coef_, BETA, atol=0.12)
        assert model.n_iter_ < 30
        # The risk is the log hazard ratio: monotone in beta . x.
        risk = model.predict(X)
        assert np.corrcoef(risk, X @ BETA)[0, 1] > 0.995

    def test_its_concordance_matches_the_truth_s_out_of_sample(self):
        X, duration, event = _planted(6000, seed=6)
        train, test = slice(0, 4000), slice(4000, 6000)
        model = CoxPHRegressor().fit(
            X[train], np.column_stack([duration[train], event[train]])
        )
        fitted, _ = concordance_index(
            duration[test], event[test], model.predict(X[test])
        )
        truth, _ = concordance_index(duration[test], event[test], X[test] @ BETA)
        assert abs(fitted - truth) < 0.02

    def test_weights_and_the_penalty_are_read(self):
        X, duration, event = _planted(2000, seed=7)
        y = np.column_stack([duration, event])
        shrunk = CoxPHRegressor(alpha=1e4).fit(X, y)
        assert np.abs(shrunk.coef_).max() < 0.1
        weighted = CoxPHRegressor().fit(X, y, sample_weight=np.full(2000, 2.0))
        plain = CoxPHRegressor().fit(X, y)
        np.testing.assert_allclose(weighted.coef_, plain.coef_, atol=1e-6)
        with pytest.raises(ValidationError, match="observed event"):
            CoxPHRegressor().fit(X, np.column_stack([duration, np.zeros(2000)]))
        with pytest.raises(ValidationError, match="positive"):
            CoxPHRegressor().fit(X, np.column_stack([duration - duration.max(), event]))


class TestThroughTheEngine:
    @pytest.fixture(scope="class")
    def dataset(self):
        return _planted_dataset()

    def _oracle(self, dataset, frame):
        joined = frame.merge(dataset["panel"], on=["date", "entity"])
        risk = joined[["f1", "f2", "f3"]].to_numpy() @ BETA
        c, _ = concordance_index(
            joined["target"].to_numpy(), joined["event"].to_numpy(), risk
        )
        return c

    def test_cox_ph_orders_the_out_of_sample_durations_like_the_truth(self, dataset):
        result = run_experiment(dataset, _spec(), "ds")
        metrics = result["oos_metrics"]
        assert 0.7 < metrics["concordance"] < 0.9
        assert metrics["cs_concordance_n_dates"] > 100
        assert 0.7 < metrics["cs_concordance_mean"] < 0.9
        assert "r2" not in metrics and "mae" not in metrics
        from standard_quant_tools.modeling import artifacts as _artifacts

        frame = _artifacts.load_artifact(result["oos_predictions_uri"])
        pooled, _ = concordance_index(
            frame.merge(dataset["panel"], on=["date", "entity"])["target"].to_numpy(),
            frame.merge(dataset["panel"], on=["date", "entity"])["event"].to_numpy(),
            frame.merge(dataset["panel"], on=["date", "entity"])[
                "prediction"
            ].to_numpy(),
        )
        assert abs(pooled - self._oracle(dataset, frame)) < 0.03
        manifest = load_manifest(result["model_id"])
        assert manifest.task == "survival"
        assert set(result["feature_importance_summary"]) == {"f1", "f2", "f3"}

    def test_a_panel_without_the_indicator_is_refused_before_any_fit(self, dataset):
        stripped = {**dataset, "panel": dataset["panel"].drop(columns=["event"])}
        with pytest.raises(ValidationError, match="event"):
            run_experiment(stripped, _spec(), "ds", register=False)

    def test_a_search_selects_on_concordance_over_purged_inner_folds(self, dataset):
        spec = _spec(
            search=SearchSpec(
                param_grid={"alpha": [0.0, 100.0]},
                inner_splits=2,
                scoring="concordance",
            )
        )
        result = run_experiment(dataset, spec, "ds", register=False)
        reports = [
            r
            for r in result["validation_report"]["hyperparameter_search"]
            if r["searched"]
        ]
        assert reports and all(r["scoring"] == "concordance" for r in reports)
        assert all(r["best_params"] == {"alpha": 0.0} for r in reports)

    @requires_xgboost
    @pytest.mark.parametrize("estimator", ["xgboost_cox", "xgboost_aft"])
    def test_the_xgboost_objectives_order_the_durations_too(self, dataset, estimator):
        spec = _spec(estimator, params={"n_estimators": 60, "max_depth": 2})
        result = run_experiment(dataset, spec, "ds", register=False)
        assert result["oos_metrics"]["concordance"] > 0.7
        assert set(result["feature_importance_summary"]) == {"f1", "f2", "f3"}


class TestAnExternalPanel:
    def _file(self, tmp_path, *, with_event=True, bad_event=False):
        dataset = _planted_dataset(n_entities=10, n_dates=200, seed=9)
        frame = dataset["panel"].rename(columns={"target": "ttf", "event": "filled"})
        if bad_event:
            frame["filled"] = frame["filled"] * 2 + 1
        if not with_event:
            frame = frame.drop(columns=["filled"])
        path = tmp_path / "fills.parquet"
        frame.to_parquet(path, index=False)
        return path

    def test_it_registers_with_its_event_column_and_fits_as_survival(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        registered = register_external_panel(
            RegisterExternalPanelInput(
                path=str(self._file(tmp_path)),
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
        assert registered.target_id == "time_to_fill:50"
        result = run_model_experiment(
            RunModelExperimentInput(dataset_id=registered.dataset_id, spec=_spec())
        )
        assert result.oos_metrics["concordance"] > 0.7
        with pytest.raises(ValidationError, match="time_to_fill"):
            run_model_experiment(
                RunModelExperimentInput(
                    dataset_id=registered.dataset_id,
                    spec=ModelSpec(
                        task="regression",
                        estimator=EstimatorSpec(type="ridge"),
                        validation=ValidationSpec(train_window=160, test_window=40),
                    ),
                )
            )

    def test_a_censored_label_without_an_indicator_is_refused(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        with pytest.raises(ValidationError, match="event_column"):
            register_external_panel(
                RegisterExternalPanelInput(
                    path=str(self._file(tmp_path, with_event=False)),
                    interval="1s",
                    targets=[
                        {
                            "name": "ttf",
                            "column": "ttf",
                            "horizon": 50,
                            "target_type": "time_to_fill",
                        }
                    ],
                )
            )
        with pytest.raises(ValidationError, match="0 or 1"):
            register_external_panel(
                RegisterExternalPanelInput(
                    path=str(self._file(tmp_path, bad_event=True)),
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
