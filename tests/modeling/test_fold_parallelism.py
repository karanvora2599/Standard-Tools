"""
Walk-forward folds fitted side by side, within `budget.max_parallelism`.

The folds of an experiment are independent, but they ran one at a time on
one thread, and the budget reached only an estimator's own `n_jobs` -- which
scikit-learn's gradient boosting does not have, so a gradient-boosting run
used one core whatever the budget said. For the estimators where nothing
recorded can depend on it (scikit-learn's gradient boosting and random
forest) the folds now run on up to `max_parallelism` threads, each fold's
estimator getting `max_parallelism // workers` jobs so the total never
exceeds the budget. See the CHANGELOG entry of 2026-10-01.

The bar is identity: every output a run records -- metrics, per-fold
records, importances, warnings, the OOS predictions file and the content
hashes in the manifest -- is the same at 1 and at N workers, and its order
does not depend on which fit finished first. Two things are excluded from
the comparison, each for a reason that has nothing to do with the folds:
`model_spec.json` carries the budget, which is the thing being varied, and
`model.skops` hashes differently between two identical sequential runs.
"""

import math
import threading
import time
from concurrent.futures import Future

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge

from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling import engine
from standard_quant_tools.modeling.engine import _fold_workers, run_experiment
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.specs import (
    ComputeBudgetSpec,
    ConformalSpec,
    EstimatorSpec,
    ModelSpec,
    SearchSpec,
    ValidationSpec,
)

GB = {"n_estimators": 12, "max_depth": 2}
RF = {"n_estimators": 12, "max_depth": 3}


def _dataset(n_entities=10, n_dates=300, seed=0, classification=False):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_dates), n_entities)
    entities = np.tile([f"E{i:02d}" for i in range(n_entities)], n_dates)
    X = rng.normal(size=(n_entities * n_dates, 4))
    signal = X @ np.array([0.4, -0.3, 0.2, 0.0]) + 0.3 * X[:, 0] * X[:, 1]
    target = signal + rng.normal(scale=0.8, size=len(X))
    if classification:
        target = (target > 0).astype(float)
    panel = pd.DataFrame(
        {"date": dates, "entity": entities, **{f"f{i}": X[:, i] for i in range(4)}}
    )
    panel["target"] = target
    return {
        "panel": panel,
        "feature_ids": [f"f{i}" for i in range(4)],
        "target_id": ("forward_direction:5" if classification else "forward_return:5"),
        "data_hash": f"folds-{seed}-{classification}",
    }


def _spec(estimator, params, max_parallelism, task="regression", **kwargs):
    kwargs.setdefault(
        "validation",
        ValidationSpec(train_window=60, test_window=20, embargo=2, min_folds=2),
    )
    return ModelSpec(
        task=task,
        estimator=EstimatorSpec(type=estimator, params=params),
        budget=ComputeBudgetSpec(max_parallelism=max_parallelism),
        random_seed=11,
        **kwargs,
    )


def _same(a, b, path="result"):
    """Deep, exact, NaN-aware equality that names the first difference."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), (path, list(a), list(b))
        for key in a:
            _same(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or a == b, (path, a, b)
    elif isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b, equal_nan=True), path
    else:
        assert a == b, (path, a, b)


def _recorded(result, drop_hashes=()):
    """Everything a registered run records, less what names the run itself
    and the budget value being varied."""
    out = {
        k: v for k, v in result.items() if k not in ("model_id", "oos_predictions_uri")
    }
    report = dict(out["validation_report"])
    report["fits"] = {k: v for k, v in report["fits"].items() if k != "max_parallelism"}
    out["validation_report"] = report
    out["oos_predictions"] = _artifacts.load_artifact(result["oos_predictions_uri"])
    hashes = dict(load_manifest(result["model_id"]).content_hashes)
    for name in ("model_spec.json", "model.skops", *drop_hashes):
        hashes.pop(name, None)
    out["content_hashes"] = hashes
    return out


def _fits_on_threads(monkeypatch):
    """Record, for every estimator fit, the thread it ran on and its n_jobs."""
    seen = []
    lock = threading.Lock()
    real = engine._fit

    def spy(estimator, *args, **kwargs):
        with lock:
            seen.append(
                (threading.current_thread().name, getattr(estimator, "n_jobs", None))
            )
        return real(estimator, *args, **kwargs)

    monkeypatch.setattr(engine, "_fit", spy)
    return seen


# ── Who runs side by side ────────────────────────────────────────────────


class TestFoldWorkers:
    def _workers(self, estimator_cls, budget, n_folds=10, **spec_kwargs):
        params = spec_kwargs.pop("params", {})
        spec = _spec("gradient_boosting", params, budget, **spec_kwargs)
        return _fold_workers(spec, estimator_cls, n_folds)

    def test_the_default_budget_is_the_sequential_loop(self):
        assert ComputeBudgetSpec().max_parallelism == 1
        assert self._workers(GradientBoostingRegressor, 1) == 1

    def test_boosting_and_forests_run_side_by_side_up_to_the_budget(self):
        assert self._workers(GradientBoostingRegressor, 4) == 4
        assert self._workers(RandomForestRegressor, 4) == 4
        # Never more workers than folds.
        assert self._workers(GradientBoostingRegressor, 16, n_folds=5) == 5
        assert self._workers(GradientBoostingRegressor, 8, n_folds=1) == 1

    def test_everything_else_keeps_its_folds_one_at_a_time(self):
        assert self._workers(HistGradientBoostingRegressor, 8) == 1
        assert self._workers(Ridge, 8) == 1

    def test_an_explicit_n_jobs_or_a_search_keeps_the_loop(self):
        assert self._workers(RandomForestRegressor, 8, params={"n_jobs": 2}) == 1
        search = SearchSpec(param_grid={"max_depth": [2, 3]}, inner_splits=2)
        assert self._workers(GradientBoostingRegressor, 8, search=search) == 1


# ── Identical at one and at N ────────────────────────────────────────────


class TestIdenticalAtOneAndN:
    def test_gradient_boosting_regression(self):
        dataset = _dataset()
        one = run_experiment(dataset, _spec("gradient_boosting", GB, 1), "ds")
        assert one["n_folds"] >= 8
        for workers in (3, 4, 32):
            other = run_experiment(
                dataset, _spec("gradient_boosting", GB, workers), "ds"
            )
            _same(_recorded(one), _recorded(other))

    def test_random_forest_regression(self):
        """At least as many folds as the budget, so each fold's forest runs at
        n_jobs=1, which sums its trees in order. The deployed forest is refit
        after the folds at n_jobs=max_parallelism, as it always was, so its
        pickle carries that parameter and is left out of the comparison."""
        dataset = _dataset()
        one = run_experiment(dataset, _spec("random_forest", RF, 1), "ds")
        four = run_experiment(dataset, _spec("random_forest", RF, 4), "ds")
        _same(
            _recorded(one, drop_hashes=("model.joblib",)),
            _recorded(four, drop_hashes=("model.joblib",)),
        )

    def test_calibrated_classification(self):
        dataset = _dataset(classification=True)
        estimator = EstimatorSpec(
            type="gradient_boosting", params=GB, calibration="sigmoid"
        )
        one = run_experiment(
            dataset,
            _spec("gradient_boosting", GB, 1, task="classification").model_copy(
                update={"estimator": estimator}
            ),
            "ds",
        )
        four = run_experiment(
            dataset,
            _spec("gradient_boosting", GB, 4, task="classification").model_copy(
                update={"estimator": estimator}
            ),
            "ds",
        )
        _same(_recorded(one), _recorded(four))

    def test_conformal_intervals(self):
        dataset = _dataset()
        intervals = ConformalSpec(alpha=0.2, calibration_folds=2)
        one = run_experiment(
            dataset, _spec("gradient_boosting", GB, 1, intervals=intervals), "ds"
        )
        four = run_experiment(
            dataset, _spec("gradient_boosting", GB, 4, intervals=intervals), "ds"
        )
        assert "lower" in _recorded(one)["oos_predictions"].columns
        _same(_recorded(one), _recorded(four))

    def test_combinatorial_paths_keep_their_numbers(self):
        dataset = _dataset(n_dates=240)
        cpcv = ValidationSpec(method="cpcv", n_splits=5, n_test_splits=2, embargo=2)
        one = run_experiment(
            dataset, _spec("gradient_boosting", GB, 1, validation=cpcv), "ds"
        )
        four = run_experiment(
            dataset, _spec("gradient_boosting", GB, 4, validation=cpcv), "ds"
        )
        assert sorted(_recorded(one)["oos_predictions"]["path"].unique()) == list(
            range(10)
        )
        _same(_recorded(one), _recorded(four))


# ── Order does not depend on completion ──────────────────────────────────


class _LazyFuture(Future):
    def __init__(self, pool):
        super().__init__()
        self._pool = pool

    def result(self, timeout=None):
        self._pool.drain()
        return super().result(timeout)


class _ReversedPool:
    """A stand-in executor that runs every submitted fit in REVERSE order,
    so the last fold finishes first -- deterministically."""

    instances = []

    def __init__(self, max_workers=None, thread_name_prefix=""):
        self.pending = []
        self.completed = []
        _ReversedPool.instances.append(self)

    def submit(self, fn, *args):
        future = _LazyFuture(self)
        self.pending.append((future, fn, args))
        return future

    def drain(self):
        while self.pending:
            future, fn, args = self.pending.pop()
            try:
                future.set_result(fn(*args))
            except Exception as exc:  # noqa: BLE001
                future.set_exception(exc)
            self.completed.append(args[0]["fold"].index)

    def shutdown(self, wait=True, cancel_futures=False):
        self.pending.clear()


class TestOrderDoesNotDependOnCompletion:
    def test_folds_finishing_last_to_first_record_first_to_last(self, monkeypatch):
        dataset = _dataset()
        one = run_experiment(dataset, _spec("gradient_boosting", GB, 1), "ds")
        _ReversedPool.instances.clear()
        monkeypatch.setattr(engine, "ThreadPoolExecutor", _ReversedPool)
        reversed_run = run_experiment(dataset, _spec("gradient_boosting", GB, 4), "ds")
        (pool,) = _ReversedPool.instances
        assert pool.completed == sorted(pool.completed, reverse=True)
        assert len(pool.completed) == one["n_folds"]
        _same(_recorded(one), _recorded(reversed_run))

    def test_jittered_threads(self, monkeypatch):
        """Real threads with random delays: whatever order they finish in."""
        dataset = _dataset(seed=3)
        one = run_experiment(dataset, _spec("gradient_boosting", GB, 1), "ds")
        rng = np.random.default_rng(0)
        delays = iter(rng.uniform(0.0, 0.05, size=1000).tolist())
        lock = threading.Lock()
        real = engine._predict_fold

        def jittered(*args, **kwargs):
            with lock:
                delay = next(delays)
            time.sleep(delay)
            return real(*args, **kwargs)

        monkeypatch.setattr(engine, "_predict_fold", jittered)
        for _ in range(2):
            other = run_experiment(dataset, _spec("gradient_boosting", GB, 6), "ds")
            _same(_recorded(one), _recorded(other))


# ── The budget is a ceiling ──────────────────────────────────────────────


class TestTheBudgetIsACeiling:
    def test_folds_run_on_worker_threads_at_one_job_each(self, monkeypatch):
        seen = _fits_on_threads(monkeypatch)
        result = run_experiment(_dataset(), _spec("random_forest", RF, 4), "ds")
        fold_fits = [s for s in seen if s[0].startswith("sqt-fold")]
        assert len(fold_fits) == result["n_folds"]
        assert len({name for name, _ in fold_fits}) <= 4
        assert {n_jobs for _, n_jobs in fold_fits} == {None}
        # The full-panel refit runs alone, after the folds, with the budget.
        assert seen[-1] == ("MainThread", 4)

    def test_fewer_folds_than_the_budget_share_it(self, monkeypatch):
        seen = _fits_on_threads(monkeypatch)
        validation = ValidationSpec(
            train_window=60, test_window=40, embargo=2, min_folds=2
        )
        result = run_experiment(
            _dataset(n_dates=200),
            _spec("random_forest", RF, 16, validation=validation),
            "ds",
        )
        n_folds = result["n_folds"]
        assert 2 <= n_folds < 16
        fold_fits = [s for s in seen if s[0].startswith("sqt-fold")]
        workers = len({name for name, _ in fold_fits})
        assert {n_jobs for _, n_jobs in fold_fits} == {16 // n_folds}
        assert workers * (16 // n_folds) <= 16

    def test_the_default_runs_every_fit_on_the_calling_thread(self, monkeypatch):
        seen = _fits_on_threads(monkeypatch)
        run_experiment(_dataset(), _spec("gradient_boosting", GB, 1), "ds")
        assert {name for name, _ in seen} == {"MainThread"}

    def test_an_ineligible_estimator_keeps_the_loop(self, monkeypatch):
        seen = _fits_on_threads(monkeypatch)
        run_experiment(_dataset(), _spec("ridge", {}, 4), "ds", register=False)
        assert {name for name, _ in seen} == {"MainThread"}


# ── Errors are the loop's ────────────────────────────────────────────────


def _fail_at(monkeypatch, *, fit_on=None, prepare_on=None):
    """Make the fit fail on the fold testing from `fit_on`, and the
    preparation fail on the fold testing from `prepare_on`."""
    real_predict, real_preprocess = engine._predict_fold, engine._preprocess

    def predict(adapter, model_spec, estimator, test_X, test_y, test_dates=None, **kw):
        if fit_on is not None and pd.Timestamp(test_dates[0]) == fit_on:
            raise RuntimeError(f"fit failed on the fold from {fit_on.date()}")
        return real_predict(
            adapter, model_spec, estimator, test_X, test_y, test_dates, **kw
        )

    def preprocess(model_spec, train_frame, test_frame, *args, **kwargs):
        if prepare_on is not None and test_frame["date"].iloc[0] == prepare_on:
            raise RuntimeError(
                f"preparation failed on the fold from {prepare_on.date()}"
            )
        return real_preprocess(model_spec, train_frame, test_frame, *args, **kwargs)

    monkeypatch.setattr(engine, "_predict_fold", predict)
    monkeypatch.setattr(engine, "_preprocess", preprocess)


def _fold_starts(dataset):
    result = run_experiment(
        dataset, _spec("gradient_boosting", GB, 1), "ds", register=False
    )
    return [pd.Timestamp(f["test_start"]) for f in result["validation_report"]["folds"]]


class TestErrorsAreTheLoops:
    @pytest.mark.parametrize(
        "fit_index, prepare_index, expected",
        [
            (1, 4, "fit"),
            (5, 2, "preparation"),
            (None, 3, "preparation"),
            (6, None, "fit"),
        ],
    )
    def test_the_first_failure_in_fold_order_is_raised(
        self, monkeypatch, fit_index, prepare_index, expected
    ):
        dataset = _dataset()
        starts = _fold_starts(dataset)
        _fail_at(
            monkeypatch,
            fit_on=None if fit_index is None else starts[fit_index],
            prepare_on=None if prepare_index is None else starts[prepare_index],
        )
        messages = []
        for budget in (1, 4):
            with pytest.raises(RuntimeError) as raised:
                run_experiment(dataset, _spec("gradient_boosting", GB, budget), "ds")
            messages.append(str(raised.value))
        assert messages[0] == messages[1]
        assert messages[0].startswith(expected)
