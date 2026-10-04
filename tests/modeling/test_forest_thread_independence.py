"""
A random forest's numbers do not depend on the budget (see the CHANGELOG
entry of 2026-10-04).

A forest builds its trees from seeds drawn before any thread starts, so its
fit is the same at any n_jobs; its predictions were not. Above one job
scikit-learn adds the trees' outputs in whatever order its threads finish:
at a budget of 16 over 8 folds, 5,798 of 15,120 out-of-sample predictions
differed from the budget-1 run in the last bits, and two budget-16 runs
differed from each other. The deployed forest kept n_jobs at the budget, so
`score_model` depended on it too. Every forest is now put back on one thread
after its fit, before anything predicts with it or writes it down.
"""

import threading

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling import engine
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.specs import (
    ComputeBudgetSpec,
    EstimatorSpec,
    ModelSpec,
    SearchSpec,
    ValidationSpec,
)

RF = {"n_estimators": 16, "max_depth": 4}


def _dataset(n_entities=10, n_dates=300, seed=0, classification=False):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_dates), n_entities)
    X = rng.normal(size=(len(dates), 4))
    target = X @ np.array([0.4, -0.3, 0.2, 0.0]) + rng.normal(scale=0.8, size=len(X))
    if classification:
        target = (target > 0).astype(float)
    panel = pd.DataFrame(
        {
            "date": dates,
            "entity": np.tile([f"E{i:02d}" for i in range(n_entities)], n_dates),
            **{f"f{i}": X[:, i] for i in range(4)},
            "target": target,
        }
    )
    return {
        "panel": panel,
        "feature_ids": [f"f{i}" for i in range(4)],
        "target_id": "forward_direction:5" if classification else "forward_return:5",
        "data_hash": f"forest-{seed}-{classification}",
    }


def _spec(budget, task="regression", **kwargs):
    kwargs.setdefault(
        "validation",
        ValidationSpec(train_window=60, test_window=20, embargo=2, min_folds=2),
    )
    estimator = kwargs.pop("estimator", EstimatorSpec(type="random_forest", params=RF))
    return ModelSpec(
        task=task,
        estimator=estimator,
        budget=ComputeBudgetSpec(max_parallelism=budget),
        random_seed=5,
        **kwargs,
    )


def _deployed_predictions(model_id, X):
    path = _artifacts.run_dir(model_id) / "model.joblib"
    model = joblib.load(path)
    return model, np.asarray(model.predict(X))


def _outputs(result):
    oos = _artifacts.load_artifact(result["oos_predictions_uri"])
    return result["oos_metrics"], oos


class TestIdenticalAtEveryBudget:
    def test_one_auto_and_twice_the_folds(self, monkeypatch):
        """Budget 1; 'auto' resolving to 2 and to 8 threads; and an explicit
        budget at least twice the fold count, where each fold's forest fits
        on two or more threads. Out-of-sample predictions, metrics and the
        deployed forest's predictions are identical bit for bit."""
        dataset = _dataset()
        X = dataset["panel"][dataset["feature_ids"]].to_numpy()[:500]
        reference = run_experiment(dataset, _spec(1), "ds")
        n_folds = reference["n_folds"]
        assert n_folds >= 8
        ref_metrics, ref_oos = _outputs(reference)
        _model, ref_deployed = _deployed_predictions(reference["model_id"], X)

        runs = []
        for threads in ("2", "8"):
            monkeypatch.setenv("SQT_NUM_THREADS", threads)
            runs.append(run_experiment(dataset, _spec("auto"), "ds"))
        monkeypatch.delenv("SQT_NUM_THREADS")
        runs.append(run_experiment(dataset, _spec(min(64, 2 * n_folds + 2)), "ds"))

        for run in runs:
            metrics, oos = _outputs(run)
            assert metrics == ref_metrics
            pd.testing.assert_frame_equal(oos, ref_oos, check_exact=True)
            model, deployed = _deployed_predictions(run["model_id"], X)
            assert model.n_jobs is None
            assert np.array_equal(deployed, ref_deployed)
            hashes = load_manifest(run["model_id"]).content_hashes
            ref_hashes = load_manifest(reference["model_id"]).content_hashes
            for name in ("oos_predictions", "prediction_reference", "model.joblib"):
                assert hashes[name] == ref_hashes[name], name

    def test_the_calibrated_forests_inside_predict_on_one_thread(self):
        estimator = EstimatorSpec(
            type="random_forest", params=RF, calibration="sigmoid"
        )
        dataset = _dataset(classification=True)
        one = run_experiment(
            dataset,
            _spec(1, task="classification", estimator=estimator),
            "ds",
            register=False,
        )
        many = run_experiment(
            dataset,
            _spec(32, task="classification", estimator=estimator),
            "ds",
            register=False,
        )
        assert one["oos_metrics"] == many["oos_metrics"]


class TestWhoPredictsOnWhat:
    def test_every_forest_predicts_on_one_thread(self, monkeypatch):
        """Fits at the budget's share, predictions at n_jobs None."""
        fitted, predicted = [], []
        lock = threading.Lock()
        real_fit = RandomForestRegressor.fit
        real_predict = RandomForestRegressor.predict

        def fit(self, *args, **kwargs):
            with lock:
                fitted.append(self.n_jobs)
            return real_fit(self, *args, **kwargs)

        def predict(self, *args, **kwargs):
            with lock:
                predicted.append(self.n_jobs)
            return real_predict(self, *args, **kwargs)

        monkeypatch.setattr(RandomForestRegressor, "fit", fit)
        monkeypatch.setattr(RandomForestRegressor, "predict", predict)
        result = run_experiment(_dataset(), _spec(32), "ds")
        assert len(fitted) == result["n_folds"] + 1
        assert max(n or 1 for n in fitted) >= 2  # fitted on threads
        assert predicted and set(predicted) <= {None, 1}

    def test_the_deployed_forest_carries_no_budget(self):
        result = run_experiment(_dataset(), _spec(8), "ds")
        model, _ = _deployed_predictions(
            result["model_id"], np.zeros((3, 4), dtype=float)
        )
        assert isinstance(model, RandomForestRegressor)
        assert model.n_jobs is None

    def test_a_classifier_too(self):
        result = run_experiment(
            _dataset(classification=True), _spec(8, task="classification"), "ds"
        )
        model, _ = _deployed_predictions(
            result["model_id"], np.zeros((3, 4), dtype=float)
        )
        assert isinstance(model, RandomForestClassifier)
        assert model.n_jobs is None


class TestSearchCandidatesShareTheBudget:
    def test_a_pooled_grid_hands_each_candidate_its_share(self, monkeypatch):
        """A grid at budget 4 runs its candidates on four threads, each
        forest on one; it used to hand every candidate all four, sixteen
        tree builders in all."""
        seen = []
        lock = threading.Lock()
        real = engine._fit

        def spy(estimator, *args, **kwargs):
            with lock:
                seen.append(
                    (
                        threading.current_thread().name,
                        getattr(estimator, "n_jobs", None),
                    )
                )
            return real(estimator, *args, **kwargs)

        monkeypatch.setattr(engine, "_fit", spy)
        search = SearchSpec(param_grid={"max_depth": [2, 3, 4]}, inner_splits=2)
        run_experiment(
            _dataset(n_dates=200), _spec(4, search=search), "ds", register=False
        )
        pooled = [jobs for name, jobs in seen if not name.startswith("MainThread")]
        assert pooled and set(pooled) <= {None, 1}

    def test_the_rule_is_the_fold_pool_rule(self):
        from standard_quant_tools.modeling.validation.search import (
            search_pool_workers,
        )

        assert search_pool_workers("grid", 3, 2, 4) == 4
        # Two pairs left after the first candidate: two workers, each
        # candidate then getting half the budget.
        assert search_pool_workers("grid", 2, 2, 16) == 2
        assert search_pool_workers("grid", 5, 2, 1) == 1
        assert search_pool_workers("tpe", 5, 2, 8) == 1
        assert search_pool_workers("grid", 1, 2, 8) == 1


@pytest.mark.parametrize("budget", [1, 4])
def test_the_forest_allowlist_widened_with_defaults_unchanged(budget):
    """max_features, max_samples and min_samples_leaf may be set; a spec that
    sets none of them fits the forest it always did."""
    dataset = _dataset(n_dates=160)
    plain = run_experiment(dataset, _spec(budget), "ds", register=False)
    defaults = run_experiment(
        dataset,
        _spec(
            budget,
            estimator=EstimatorSpec(
                type="random_forest",
                params={
                    **RF,
                    "max_features": 1.0,
                    "max_samples": None,
                    "min_samples_leaf": 1,
                },
            ),
        ),
        "ds",
        register=False,
    )
    assert plain["oos_metrics"] == defaults["oos_metrics"]
    narrower = run_experiment(
        dataset,
        _spec(
            budget,
            estimator=EstimatorSpec(
                type="random_forest",
                params={**RF, "max_features": 0.5, "max_samples": 0.5},
            ),
        ),
        "ds",
        register=False,
    )
    assert narrower["oos_metrics"] != plain["oos_metrics"]
