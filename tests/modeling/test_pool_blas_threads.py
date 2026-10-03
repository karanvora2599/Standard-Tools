"""
The modeling pools run each worker's fit on one BLAS thread (see the
CHANGELOG entry of 2026-10-02).

Walk-forward folds fitted side by side, and search candidates scored side by
side, each started their BLAS at one thread per logical CPU -- W workers, W
times over. Every fit on a pool now runs on one BLAS thread, and nothing
else changes: the default budget, a fold loop that runs alone and the
full-panel refit keep the caller's setting, and a run's numbers are the
same at a budget of 1 and of 4, now for an estimator wide enough to reach
LAPACK too.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools import _blas
from standard_quant_tools.modeling import engine
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.specs import (
    ComputeBudgetSpec,
    EstimatorSpec,
    ModelSpec,
    SearchSpec,
    ValidationSpec,
)

threadpoolctl = pytest.importorskip("threadpoolctl")
from threadpoolctl import threadpool_limits  # noqa: E402


def _dataset(n_features=4, n_entities=10, n_dates=240, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_dates), n_entities)
    entities = np.tile([f"E{i:02d}" for i in range(n_entities)], n_dates)
    X = rng.normal(size=(n_entities * n_dates, n_features))
    beta = rng.normal(size=n_features) / math.sqrt(n_features)
    target = X @ beta + rng.normal(scale=0.8, size=len(X))
    names = [f"f{i}" for i in range(n_features)]
    panel = pd.DataFrame(X, columns=names)
    panel.insert(0, "entity", entities)
    panel.insert(0, "date", dates)
    panel["target"] = target
    return {
        "panel": panel,
        "feature_ids": names,
        "target_id": "forward_return:5",
        "data_hash": f"pool-blas-{seed}-{n_features}",
    }


def _spec(estimator, params, max_parallelism, search=None):
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type=estimator, params=params),
        validation=ValidationSpec(
            train_window=100, test_window=40, embargo=0, min_folds=2
        ),
        search=search,
        budget=ComputeBudgetSpec(max_parallelism=max_parallelism),
        random_seed=7,
    )


GRID = SearchSpec(param_grid={"alpha": [0.01, 0.1, 1.0, 10.0]}, inner_splits=2)


@pytest.fixture
def caller_threads():
    """The caller's BLAS setting for the run: up to four threads, so that
    one thread inside a pool's fit is told from the caller's."""
    if _blas._get_controller() is None:
        pytest.skip("no controllable BLAS in this environment")
    with threadpool_limits(limits=4, user_api="blas"):
        threads = max(
            lib.get_num_threads() for lib in _blas._controller.lib_controllers
        )
        if threads < 2:
            pytest.skip("BLAS runs one thread here, so a limit of one is invisible")
        yield threads


def _record_fits(monkeypatch):
    """For every estimator fit: whether a search was running, the thread,
    and the BLAS thread count the fit ran under."""
    seen = []
    lock = threading.Lock()
    in_search = threading.Event()
    real_fit, real_search = engine._fit, engine.search_best_params

    def fit(estimator, *args, **kwargs):
        threads = max(
            lib.get_num_threads() for lib in _blas._controller.lib_controllers
        )
        with lock:
            seen.append((in_search.is_set(), threading.current_thread().name, threads))
        return real_fit(estimator, *args, **kwargs)

    def search(**kwargs):
        in_search.set()
        try:
            return real_search(**kwargs)
        finally:
            in_search.clear()

    monkeypatch.setattr(engine, "_fit", fit)
    monkeypatch.setattr(engine, "search_best_params", search)
    return seen


class TestWhoRunsOnOneThread:
    def test_folds_side_by_side(self, monkeypatch, caller_threads):
        seen = _record_fits(monkeypatch)
        params = {"n_estimators": 8, "max_depth": 2}
        run_experiment(
            _dataset(), _spec("gradient_boosting", params, 4), "ds", register=False
        )
        folds = [s for s in seen if s[1].startswith("sqt-fold")]
        alone = [s for s in seen if not s[1].startswith("sqt-fold")]
        assert len(folds) >= 2 and alone  # the folds, then the refit
        assert {s[2] for s in folds} == {1}
        assert {s[2] for s in alone} == {caller_threads}

    def test_the_default_budget_keeps_the_callers_setting(
        self, monkeypatch, caller_threads
    ):
        seen = _record_fits(monkeypatch)
        params = {"n_estimators": 8, "max_depth": 2}
        run_experiment(
            _dataset(), _spec("gradient_boosting", params, 1), "ds", register=False
        )
        assert seen and {s[2] for s in seen} == {caller_threads}

    def test_search_candidates_side_by_side(self, monkeypatch, caller_threads):
        """Every candidate's fit -- the first, which runs alone to fill the
        cache, included -- on one thread; the outer folds' fits and the
        refit, which run alone, on the caller's."""
        seen = _record_fits(monkeypatch)
        run_experiment(
            _dataset(), _spec("ridge", {}, 4, search=GRID), "ds", register=False
        )
        searched = [s for s in seen if s[0]]
        outer = [s for s in seen if not s[0]]
        assert len({s[1] for s in searched}) > 1  # a pool really ran
        assert {s[2] for s in searched} == {1}
        assert outer and {s[2] for s in outer} == {caller_threads}

    def test_a_search_without_a_pool_keeps_the_callers_setting(
        self, monkeypatch, caller_threads
    ):
        seen = _record_fits(monkeypatch)
        run_experiment(
            _dataset(), _spec("ridge", {}, 1, search=GRID), "ds", register=False
        )
        assert {s[2] for s in seen} == {caller_threads}
        seen.clear()
        one = SearchSpec(param_grid={"alpha": [1.0]}, inner_splits=2)
        run_experiment(
            _dataset(), _spec("ridge", {}, 4, search=one), "ds", register=False
        )
        assert {s[2] for s in seen} == {caller_threads}


def _numbers(result):
    """The run's numbers and choices, less the budget that is varied."""
    report = dict(result["validation_report"])
    report.pop("fits", None)
    searches = [
        {k: v for k, v in s.items() if k != "max_parallelism"}
        for s in report.pop("hyperparameter_search", None) or []
    ]
    final = report.pop("final_search", None)
    if isinstance(final, dict):
        final = {k: v for k, v in final.items() if k != "max_parallelism"}
    return report, searches, final, result["oos_metrics"]


def _same(a, b, path="result"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and list(a) == list(b), path
        for key in a:
            _same(a[key], b[key], f"{path}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or a == b, (path, a, b)
    else:
        assert a == b, (path, a, b)


class TestTheNumbersDoNotDependOnTheBudget:
    def test_a_wide_ridge_search_at_one_and_four(self):
        """150 features: each candidate solves a 150x150 system. Budget 1
        runs the solves at the process's BLAS setting, budget 4 on one
        thread each, and every number agrees exactly."""
        data = _dataset(n_features=150, n_entities=20)
        one = run_experiment(
            data, _spec("ridge", {}, 1, search=GRID), "ds", register=False
        )
        four = run_experiment(
            data, _spec("ridge", {}, 4, search=GRID), "ds", register=False
        )
        searches = one["validation_report"]["hyperparameter_search"]
        assert len(searches) >= 2 and all(s["searched"] for s in searches)
        _same(_numbers(one), _numbers(four))
