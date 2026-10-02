"""
A `gradient_boosting` run on a large panel says that the histogram-binned
booster exists, and how much faster it measured -- and fits what it was
asked to fit.

scikit-learn's exact-split GradientBoosting is where the time of such a run
goes (see the CHANGELOG entry of 2026-10-01). hist_gradient_boosting is the
binned equivalent, but a different model, so the choice is the caller's:
the sentence is guidance, never a substitution.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from standard_quant_tools.modeling import engine
from standard_quant_tools.modeling.engine import (
    _GRADIENT_BOOSTING_ADVICE_ROWS,
    _gradient_boosting_advice,
    run_experiment,
)
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
)


def _spec(estimator, task="regression"):
    return ModelSpec(
        task=task,
        estimator=EstimatorSpec(type=estimator, params={}),
        validation=ValidationSpec(
            train_window=60, test_window=20, embargo=2, min_folds=2
        ),
        random_seed=1,
    )


def _dataset(classification=False):
    rng = np.random.default_rng(0)
    n_entities, n_dates = 6, 140
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_dates), n_entities)
    X = rng.normal(size=(len(dates), 3))
    target = X[:, 0] + rng.normal(size=len(dates))
    if classification:
        target = (target > 0).astype(float)
    panel = pd.DataFrame(
        {
            "date": dates,
            "entity": np.tile([f"E{i}" for i in range(n_entities)], n_dates),
            "a": X[:, 0],
            "b": X[:, 1],
            "c": X[:, 2],
            "target": target,
        }
    )
    return {
        "panel": panel,
        "feature_ids": ["a", "b", "c"],
        "target_id": "forward_direction:5" if classification else "forward_return:5",
        "data_hash": f"advice-{classification}",
    }


def _advice(warnings):
    return [w for w in warnings if "hist_gradient_boosting" in w]


class TestTheSentence:
    def test_said_from_the_threshold_up(self):
        spec = _spec("gradient_boosting")
        assert _gradient_boosting_advice(spec, _GRADIENT_BOOSTING_ADVICE_ROWS - 1) == []
        (line,) = _gradient_boosting_advice(spec, _GRADIENT_BOOSTING_ADVICE_ROWS)
        assert f"{_GRADIENT_BOOSTING_ADVICE_ROWS:,}-row panel" in line
        assert "hist_gradient_boosting" in line
        assert "faster" in line and "not substituted" in line

    def test_only_for_gradient_boosting(self):
        for estimator in ("hist_gradient_boosting", "random_forest", "ridge"):
            assert _gradient_boosting_advice(_spec(estimator), 10**7) == []


class TestInARun:
    def test_named_in_warnings_and_the_estimator_is_kept(self, monkeypatch):
        dataset = _dataset()
        monkeypatch.setattr(engine, "_GRADIENT_BOOSTING_ADVICE_ROWS", 100)
        built = []
        real = engine._instantiate

        def spy(cls, *args, **kwargs):
            built.append(cls)
            return real(cls, *args, **kwargs)

        monkeypatch.setattr(engine, "_instantiate", spy)
        result = run_experiment(dataset, _spec("gradient_boosting"), "ds")
        (line,) = _advice(result["warnings"])
        assert f"{len(dataset['panel']):,}-row panel" in line
        assert set(built) == {GradientBoostingRegressor}

    def test_classification_too(self, monkeypatch):
        monkeypatch.setattr(engine, "_GRADIENT_BOOSTING_ADVICE_ROWS", 100)
        built = []
        real = engine._instantiate

        def spy(cls, *args, **kwargs):
            built.append(cls)
            return real(cls, *args, **kwargs)

        monkeypatch.setattr(engine, "_instantiate", spy)
        result = run_experiment(
            _dataset(classification=True),
            _spec("gradient_boosting", task="classification"),
            "ds",
            register=False,
        )
        assert len(_advice(result["warnings"])) == 1
        assert set(built) == {GradientBoostingClassifier}

    def test_silent_below_the_threshold_and_for_other_estimators(self):
        dataset = _dataset()
        assert len(dataset["panel"]) < _GRADIENT_BOOSTING_ADVICE_ROWS
        for estimator in ("gradient_boosting", "hist_gradient_boosting"):
            result = run_experiment(dataset, _spec(estimator), "ds", register=False)
            assert _advice(result["warnings"]) == []
