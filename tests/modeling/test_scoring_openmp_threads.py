"""
A scored model predicts under the OpenMP limit the run fits under (see the
CHANGELOG entry of 2026-10-04).

Every fit and fold prediction of a run is held to its share of the budget;
`score_model` predicted with whatever the OpenMP runtime had, every logical
CPU by default. On a 30-row, 8-feature histogram-boosting score that was
a median 5.0-9.2 ms on sixteen threads against 0.75-0.93 ms on one, for
the same predictions. The deployed model now predicts at the count the run
would give one fit of the scored matrix on the model's budget -- one
thread under 'auto' below 2,000,000 cells, the budget when it is a number
-- and an estimator that does not run on OpenMP is left alone, as before.
"""

import contextlib
from types import SimpleNamespace

import pandas as pd
import pytest

from standard_quant_tools.modeling import scoring
from standard_quant_tools.modeling.scoring import (
    _prediction_thread_limit,
    score_model,
)
from standard_quant_tools.modeling.specs import (
    ComputeBudgetSpec,
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec

threadpoolctl = pytest.importorskip("threadpoolctl")
from threadpoolctl import threadpool_info  # noqa: E402

AS_OF = "2023-12-29"
UNIVERSE = ["AAA", "BBB", "CCC"]


def _openmp_threads():
    counts = [i["num_threads"] for i in threadpool_info() if i["user_api"] == "openmp"]
    return max(counts) if counts else None


@pytest.fixture
def openmp():
    """Skip where no OpenMP runtime is loaded to be limited."""
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401

    if _openmp_threads() is None:
        pytest.skip("no OpenMP runtime loaded")
    return _openmp_threads()


def _train(estimator: str, budget) -> str:
    params = {"max_iter": 15} if estimator == "hist_gradient_boosting" else {}
    return _train_a_model_with_spec(
        _dataset_spec(),
        model_spec=ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type=estimator, params=params),
            validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
            budget=ComputeBudgetSpec(max_parallelism=budget),
            random_seed=1,
        ),
    )


def _record_scoring_threads(monkeypatch):
    """The OpenMP count in force whenever the adapter scores."""
    seen = []
    real = scoring.get_adapter

    def spy(task):
        adapter = real(task)

        class Recording:
            def __getattr__(self, name):
                return getattr(adapter, name)

            def score(self, estimator, X):
                seen.append(_openmp_threads())
                return adapter.score(estimator, X)

        return Recording()

    monkeypatch.setattr(scoring, "get_adapter", spy)
    return seen


def _predictions(result) -> bytes:
    frame = pd.read_parquet(result["predictions_uri"])
    return frame["prediction"].to_numpy().tobytes()


class TestTheScoredPrediction:
    @pytest.mark.parametrize("budget, expected", [("auto", 1), (3, 3)])
    def test_runs_at_the_runs_share_and_the_count_comes_back(
        self, patched_multi_factory, monkeypatch, openmp, budget, expected
    ):
        model_id = _train("hist_gradient_boosting", budget)
        seen = _record_scoring_threads(monkeypatch)
        score_model(model_id=model_id, as_of=AS_OF, universe=UNIVERSE)
        assert seen == [expected]
        assert _openmp_threads() == openmp

    def test_an_estimator_off_openmp_is_left_alone(
        self, patched_multi_factory, monkeypatch, openmp
    ):
        model_id = _train("ridge", "auto")
        seen = _record_scoring_threads(monkeypatch)
        score_model(model_id=model_id, as_of=AS_OF, universe=UNIVERSE)
        assert seen == [openmp]

    def test_the_predictions_are_those_of_the_unlimited_runtime_to_the_bit(
        self, patched_multi_factory, monkeypatch, openmp
    ):
        model_id = _train("hist_gradient_boosting", "auto")
        limited = score_model(model_id=model_id, as_of=AS_OF, universe=UNIVERSE)
        monkeypatch.setattr(
            scoring,
            "_prediction_thread_limit",
            lambda *args: contextlib.nullcontext(),
        )
        unlimited = score_model(model_id=model_id, as_of=AS_OF, universe=UNIVERSE)
        assert _predictions(limited) == _predictions(unlimited)


class TestTheBudget:
    def test_it_is_the_budget_the_run_recorded(self, openmp):
        """Read off the manifest in hand -- `validation_report.fits` records
        the budget as the spec asked for it -- without loading the spec."""
        manifest = SimpleNamespace(
            task="regression",
            estimator_type="hist_gradient_boosting",
            validation_report={"fits": {"max_parallelism": 2}},
        )
        X = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
        with _prediction_thread_limit("mdl_not_registered", manifest, X):
            assert _openmp_threads() == 2
        assert _openmp_threads() == openmp

    def test_a_spec_that_does_not_load_leaves_the_default_auto(self, openmp):
        """The count decides how long a prediction takes and nothing it
        returns, so a model whose spec cannot be read is still scored: on
        the default budget, one thread for a small matrix."""
        manifest = SimpleNamespace(
            task="regression", estimator_type="hist_gradient_boosting"
        )
        X = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
        with _prediction_thread_limit("mdl_not_registered", manifest, X):
            assert _openmp_threads() == 1
        assert _openmp_threads() == openmp
