"""
`budget.max_parallelism`: a knob that controls something.

It was not built in phase 4 because nothing ran in parallel and no
estimator read `n_jobs`, so it would have controlled nothing. It now
fans the grid search's candidates out over threads and hands `n_jobs` to
any estimator whose constructor accepts it. The property that matters is
that the RESULT does not depend on it: the same spec at 1 and at 4
threads selects the same parameters with the same candidate scores.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from standard_quant_tools.modeling.engine import _instantiate, run_experiment
from standard_quant_tools.modeling.specs import (
    ComputeBudgetSpec,
    EstimatorSpec,
    ModelSpec,
    SearchSpec,
    ValidationSpec,
)


def _dataset(n_entities=12, n_dates=240, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2021-01-01", periods=n_dates), n_entities)
    entities = np.tile([f"E{i:02d}" for i in range(n_entities)], n_dates)
    X = rng.normal(size=(n_entities * n_dates, 3))
    target = X @ np.array([0.5, -0.3, 0.1]) + rng.normal(scale=0.8, size=len(X))
    panel = pd.DataFrame(
        {
            "date": dates,
            "entity": entities,
            "f1": X[:, 0],
            "f2": X[:, 1],
            "f3": X[:, 2],
            "target": target,
        }
    )
    return {
        "panel": panel,
        "feature_ids": ["f1", "f2", "f3"],
        "target_id": "forward_return:5",
        "data_hash": f"parallel-{seed}",
    }


def _spec(max_parallelism: int) -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={}),
        validation=ValidationSpec(
            train_window=120, test_window=40, embargo=0, min_folds=1
        ),
        search=SearchSpec(
            param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}, inner_splits=2
        ),
        budget=ComputeBudgetSpec(max_parallelism=max_parallelism),
        random_seed=3,
    )


class TestTheKnob:
    def test_it_reaches_an_estimator_that_accepts_it_and_never_overrides_params(self):
        assert _instantiate(RandomForestRegressor, {}, 0, n_jobs=3).n_jobs == 3
        assert (
            _instantiate(RandomForestRegressor, {"n_jobs": 1}, 0, n_jobs=3).n_jobs == 1
        )
        assert _instantiate(RandomForestRegressor, {}, 0, n_jobs=1).n_jobs is None
        assert not hasattr(_instantiate(Ridge, {"alpha": 1.0}, 0, n_jobs=3), "n_jobs")

    def test_the_bounds_are_a_budget(self):
        assert ComputeBudgetSpec().max_parallelism == 1
        with pytest.raises(PydanticValidationError):
            ComputeBudgetSpec(max_parallelism=0)
        with pytest.raises(PydanticValidationError):
            ComputeBudgetSpec(max_parallelism=65)

    def test_the_result_does_not_depend_on_it(self):
        dataset = _dataset()
        sequential = run_experiment(dataset, _spec(1), "ds", register=False)
        threaded = run_experiment(dataset, _spec(4), "ds", register=False)
        for key, value in sequential["oos_metrics"].items():
            other = threaded["oos_metrics"][key]
            assert (np.isnan(value) and np.isnan(other)) or value == pytest.approx(
                other, rel=1e-9
            )
        for one, four in zip(
            sequential["validation_report"]["hyperparameter_search"],
            threaded["validation_report"]["hyperparameter_search"],
        ):
            assert one["best_params"] == four["best_params"]
            assert [c["params"] for c in one["candidates"]] == [
                c["params"] for c in four["candidates"]
            ]
            assert np.allclose(
                [c["score"] for c in one["candidates"]],
                [c["score"] for c in four["candidates"]],
            )
            assert one["max_parallelism"] == 1 and four["max_parallelism"] == 4
        assert sequential["validation_report"]["fits"]["max_parallelism"] == 1
        assert threaded["validation_report"]["fits"]["max_parallelism"] == 4
