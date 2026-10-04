"""
A `random_forest` run that will build every tree on one core says what a
budget would buy it (see the CHANGELOG entry of 2026-10-04).

A forest fitted at the old default budget of 1 built every tree on one
core and spent 97% of the run's time doing it. The budget now defaults to
'auto', so the sentence is for a spec that asked for 1, or a machine that
gives the process one CPU, on a panel of at least 10,000 rows. Guidance,
never a substitution: the run fits the forest it was asked to fit.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from standard_quant_tools.modeling import engine
from standard_quant_tools.modeling.engine import (
    _RANDOM_FOREST_ADVICE_ROWS,
    _random_forest_advice,
    run_experiment,
)
from standard_quant_tools.modeling.specs import (
    ComputeBudgetSpec,
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
)


def _spec(estimator="random_forest", budget=1):
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(
            type=estimator,
            params=(
                {"n_estimators": 5, "max_depth": 2}
                if estimator == "random_forest"
                else {}
            ),
        ),
        validation=ValidationSpec(
            train_window=60, test_window=20, embargo=2, min_folds=2
        ),
        budget=ComputeBudgetSpec(max_parallelism=budget),
        random_seed=1,
    )


def _dataset():
    rng = np.random.default_rng(0)
    n_entities, n_dates = 6, 140
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_dates), n_entities)
    X = rng.normal(size=(len(dates), 3))
    panel = pd.DataFrame(
        {
            "date": dates,
            "entity": np.tile([f"E{i}" for i in range(n_entities)], n_dates),
            "a": X[:, 0],
            "b": X[:, 1],
            "c": X[:, 2],
            "target": X[:, 0] + rng.normal(size=len(dates)),
        }
    )
    return {
        "panel": panel,
        "feature_ids": ["a", "b", "c"],
        "target_id": "forward_return:5",
        "data_hash": "forest-advice",
    }


def _advice(warnings):
    return [w for w in warnings if w.startswith("random_forest on a")]


class TestTheSentence:
    def test_said_from_the_threshold_up_at_a_budget_of_one(self):
        spec = _spec()
        assert _random_forest_advice(spec, _RANDOM_FOREST_ADVICE_ROWS - 1, 1) == []
        (line,) = _random_forest_advice(spec, _RANDOM_FOREST_ADVICE_ROWS, 1)
        assert f"{_RANDOM_FOREST_ADVICE_ROWS:,}-row panel" in line
        assert "budget.max_parallelism=1 builds every tree on one core" in line
        assert "hist_gradient_boosting" in line and "not substituted" in line

    def test_silent_when_the_budget_buys_threads(self):
        assert _random_forest_advice(_spec(budget=4), 10**6, 4) == []
        assert _random_forest_advice(_spec(budget="auto"), 10**6, 16) == []

    def test_auto_on_one_cpu_says_so(self):
        (line,) = _random_forest_advice(_spec(budget="auto"), 10**6, 1)
        assert "'auto', which is one thread on this machine" in line

    def test_only_for_random_forest(self):
        for estimator in ("ridge", "hist_gradient_boosting", "gradient_boosting"):
            assert _random_forest_advice(_spec(estimator), 10**6, 1) == []


class TestInARun:
    def test_named_in_warnings_and_the_forest_is_kept(self, monkeypatch):
        monkeypatch.setattr(engine, "_RANDOM_FOREST_ADVICE_ROWS", 100)
        built = []
        real = engine._instantiate

        def spy(cls, *args, **kwargs):
            built.append(cls)
            return real(cls, *args, **kwargs)

        monkeypatch.setattr(engine, "_instantiate", spy)
        dataset = _dataset()
        result = run_experiment(dataset, _spec(), "ds", register=False)
        (line,) = _advice(result["warnings"])
        assert f"{len(dataset['panel']):,}-row panel" in line
        assert set(built) == {RandomForestRegressor}

    def test_auto_resolving_to_one_thread_too(self, monkeypatch):
        monkeypatch.setattr(engine, "_RANDOM_FOREST_ADVICE_ROWS", 100)
        monkeypatch.setenv("SQT_NUM_THREADS", "1")
        result = run_experiment(_dataset(), _spec(budget="auto"), "ds", register=False)
        assert len(_advice(result["warnings"])) == 1

    def test_silent_below_the_threshold(self):
        dataset = _dataset()
        assert len(dataset["panel"]) < _RANDOM_FOREST_ADVICE_ROWS
        result = run_experiment(dataset, _spec(), "ds", register=False)
        assert _advice(result["warnings"]) == []
