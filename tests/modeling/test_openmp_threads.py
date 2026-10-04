"""
OpenMP estimators run on their share of the budget (see the CHANGELOG entry
of 2026-10-04).

scikit-learn's histogram gradient boosting -- and LightGBM and XGBoost when
installed -- start an OpenMP team on every logical CPU for each fit and
prediction, whatever the budget said. Under the PASSIVE wait policy this
package sets on import, sixteen threads fitted a 15,000-row fold about five
times slower than one, with the same predictions. Each fit and prediction
now runs under a process-wide, reference-counted limit at the fit's share of
the budget: one thread under 'auto' below 2,000,000 training cells.
"""

import threading

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools import _blas
from standard_quant_tools.modeling import engine
from standard_quant_tools.modeling.engine import _fit_threads, run_experiment
from standard_quant_tools.modeling.specs import (
    ComputeBudgetSpec,
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
)

threadpoolctl = pytest.importorskip("threadpoolctl")
from threadpoolctl import threadpool_info  # noqa: E402


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


def _dataset(n_entities=10, n_dates=200, seed=0, classification=False):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_dates), n_entities)
    X = rng.normal(size=(len(dates), 4))
    target = X[:, 0] * 0.5 - X[:, 1] * X[:, 2] * 0.3 + rng.normal(size=len(dates))
    if classification:
        target = (target > 0).astype(float)
    panel = pd.DataFrame(
        {
            "date": dates,
            "entity": np.tile([f"E{i}" for i in range(n_entities)], n_dates),
            **{f"f{i}": X[:, i] for i in range(4)},
            "target": target,
        }
    )
    return {
        "panel": panel,
        "feature_ids": [f"f{i}" for i in range(4)],
        "target_id": "forward_direction:5" if classification else "forward_return:5",
        "data_hash": f"openmp-{seed}-{classification}",
    }


def _spec(budget, task="regression"):
    return ModelSpec(
        task=task,
        estimator=EstimatorSpec(type="hist_gradient_boosting", params={"max_iter": 15}),
        validation=ValidationSpec(
            train_window=60, test_window=20, embargo=2, min_folds=2
        ),
        budget=ComputeBudgetSpec(max_parallelism=budget),
        random_seed=4,
    )


def _record_fit_threads(monkeypatch):
    seen = []
    real = engine._fit

    def spy(estimator, *args, **kwargs):
        seen.append(_openmp_threads())
        return real(estimator, *args, **kwargs)

    monkeypatch.setattr(engine, "_fit", spy)
    return seen


class TestTheLimit:
    def test_set_inside_and_put_back_after(self, openmp):
        with _blas.openmp_thread_limit(1):
            assert _openmp_threads() == 1
        assert _openmp_threads() == openmp

    def test_none_leaves_the_runtime_alone(self, openmp):
        with _blas.openmp_thread_limit(None):
            assert _openmp_threads() == openmp

    def test_nested_users_run_on_the_count_in_force(self, openmp):
        with _blas.openmp_thread_limit(2):
            with _blas.openmp_thread_limit(5):
                assert _openmp_threads() == 2
            assert _openmp_threads() == 2
        assert _openmp_threads() == openmp

    def test_the_last_concurrent_user_restores(self, openmp):
        inside = threading.Barrier(2)
        leave = threading.Event()
        seen = []

        def user():
            with _blas.openmp_thread_limit(1):
                inside.wait()
                leave.wait()

        worker = threading.Thread(target=user)
        worker.start()
        with _blas.openmp_thread_limit(1):
            inside.wait()
            seen.append(_openmp_threads())
        # The worker is still inside: the limit holds.
        seen.append(_openmp_threads())
        leave.set()
        worker.join()
        seen.append(_openmp_threads())
        assert seen == [1, 1, openmp]


class TestTheShare:
    def test_the_rule(self):
        # Under 'auto', one thread below 2,000,000 training cells.
        assert _fit_threads("openmp", "auto", 16, 1_999_999) == (1, 1)
        assert _fit_threads("openmp", "auto", 16, 2_000_000) == (16, 16)
        # An explicit budget is taken as asked.
        assert _fit_threads("openmp", 4, 4, 100) == (4, 4)
        assert _fit_threads("openmp", 1, 1, 10**9) == (1, 1)
        # Everything else is handed its share as n_jobs and no limit.
        assert _fit_threads("budget", "auto", 8, 100) == (8, None)
        assert _fit_threads("one", 4, 4, 100) == (4, None)
        assert _fit_threads(None, 4, 4, 100) == (4, None)

    def test_auto_fits_a_small_panel_on_one_thread(self, monkeypatch, openmp):
        seen = _record_fit_threads(monkeypatch)
        result = run_experiment(_dataset(), _spec("auto"), "ds", register=False)
        assert len(seen) == result["n_folds"] + 1
        assert set(seen) == {1}
        assert _openmp_threads() == openmp

    def test_an_explicit_budget_is_the_share(self, monkeypatch, openmp):
        seen = _record_fit_threads(monkeypatch)
        run_experiment(_dataset(), _spec(3), "ds", register=False)
        assert set(seen) == {3}

    def test_a_large_matrix_under_auto_gets_the_budget(self, monkeypatch, openmp):
        monkeypatch.setattr(engine, "_OPENMP_ONE_THREAD_CELLS", 10)
        monkeypatch.setenv("SQT_NUM_THREADS", "2")
        seen = _record_fit_threads(monkeypatch)
        run_experiment(_dataset(), _spec("auto"), "ds", register=False)
        assert set(seen) == {2}


class TestTheBoostersAreHandedTheirShare:
    def test_lightgbm_is_told_its_thread_count_one_included(self, monkeypatch):
        """LightGBM reads an unset n_jobs as every core whatever the OpenMP
        runtime's count, so its share is passed explicitly."""
        pytest.importorskip("lightgbm")
        seen = []
        real = engine._fit

        def spy(estimator, *args, **kwargs):
            seen.append(estimator.get_params().get("n_jobs"))
            return real(estimator, *args, **kwargs)

        monkeypatch.setattr(engine, "_fit", spy)
        for budget, expected in (("auto", 1), (3, 3)):
            seen.clear()
            spec = _spec(budget).model_copy(
                update={
                    "estimator": EstimatorSpec(
                        type="lightgbm", params={"n_estimators": 10}
                    )
                }
            )
            run_experiment(_dataset(), spec, "ds", register=False)
            assert seen and set(seen) == {expected}


class TestTheNumbersDoNotDependOnIt:
    @pytest.mark.parametrize("classification", [False, True])
    def test_one_thread_and_four_agree_bit_for_bit(self, classification):
        """Every number a run reports; only the pickle differs, because the
        bin mapper records the thread count it was fitted with."""
        task = "classification" if classification else "regression"
        dataset = _dataset(classification=classification)
        one = run_experiment(dataset, _spec(1, task), "ds", register=False)
        four = run_experiment(dataset, _spec(4, task), "ds", register=False)
        assert one["oos_metrics"] == four["oos_metrics"] or all(
            (a == b) or (np.isnan(a) and np.isnan(b))
            for a, b in zip(one["oos_metrics"].values(), four["oos_metrics"].values())
        )
        assert [f["metrics"] for f in one["validation_report"]["folds"]] == [
            f["metrics"] for f in four["validation_report"]["folds"]
        ]
