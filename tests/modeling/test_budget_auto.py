"""
`budget.max_parallelism` defaults to 'auto' (see the CHANGELOG entry of
2026-10-04).

'auto' is SQT_NUM_THREADS when set, else the CPUs the process may run on,
at most 64, read when the run starts. A run reports the budget as asked --
'auto', never the machine's count -- because a recorded call is replayed by
comparing its output on whatever machine checks it; the count it resolved to
is kept in the manifest's environment. The report also says how many folds
ran side by side and what limited it, and the cache block names projections
only where a shared cache could have made one.
"""

import os

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from standard_quant_tools.audit.hashing import hash_payload
from standard_quant_tools.audit.replay import normalize_identifiers
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.cache import FoldCache
from standard_quant_tools.modeling.engine import (
    FOLD_LIMIT_BUDGET,
    FOLD_LIMIT_ESTIMATOR,
    FOLD_LIMIT_N_JOBS,
    FOLD_LIMIT_ONE_FOLD,
    FOLD_LIMIT_SEARCH,
    _fold_schedule,
    run_experiment,
)
from standard_quant_tools.modeling.registry.environment import (
    THREAD_VARIABLES,
    environment_fingerprint,
)
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.specs import (
    ComputeBudgetSpec,
    EstimatorSpec,
    ModelSpec,
    SearchSpec,
    ValidationSpec,
    auto_parallelism,
)


def _dataset(n_entities=8, n_dates=240, seed=0):
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_dates), n_entities)
    X = rng.normal(size=(len(dates), 3))
    panel = pd.DataFrame(
        {
            "date": dates,
            "entity": np.tile([f"E{i}" for i in range(n_entities)], n_dates),
            "a": X[:, 0],
            "b": X[:, 1],
            "c": X[:, 2],
            "target": 0.4 * X[:, 0] + rng.normal(size=len(dates)),
        }
    )
    return {
        "panel": panel,
        "feature_ids": ["a", "b", "c"],
        "target_id": "forward_return:5",
        "data_hash": f"auto-{seed}",
    }


def _spec(estimator="random_forest", budget="auto", params=None, **kwargs):
    kwargs.setdefault(
        "validation",
        ValidationSpec(train_window=60, test_window=20, embargo=2, min_folds=2),
    )
    if params is None:
        params = {"n_estimators": 8, "max_depth": 3} if "forest" in estimator else {}
    budget_spec = (
        ComputeBudgetSpec()
        if budget is None
        else ComputeBudgetSpec(max_parallelism=budget)
    )
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type=estimator, params=params),
        budget=budget_spec,
        random_seed=2,
        **kwargs,
    )


class TestWhatAutoMeans:
    def test_the_default(self):
        assert ComputeBudgetSpec().max_parallelism == "auto"
        assert _spec(budget=None).budget.max_parallelism == "auto"

    def test_sqt_num_threads_first(self, monkeypatch):
        monkeypatch.setenv("SQT_NUM_THREADS", "3")
        assert auto_parallelism() == 3
        assert ComputeBudgetSpec().resolved_max_parallelism() == 3
        monkeypatch.setenv("SQT_NUM_THREADS", "500")
        assert auto_parallelism() == 64

    def test_otherwise_the_cpus_the_process_may_use(self, monkeypatch):
        for unset in ("", "0"):
            monkeypatch.setenv("SQT_NUM_THREADS", unset)
            try:
                expected = len(os.sched_getaffinity(0))
            except AttributeError:
                expected = os.cpu_count() or 1
            assert auto_parallelism() == min(expected, 64)

    def test_an_unusable_setting_is_refused_by_name(self, monkeypatch):
        monkeypatch.setenv("SQT_NUM_THREADS", "many")
        with pytest.raises(ValidationError, match="SQT_NUM_THREADS"):
            auto_parallelism()

    def test_an_explicit_number_keeps_its_meaning(self):
        assert ComputeBudgetSpec(max_parallelism=6).resolved_max_parallelism() == 6
        for bad in (0, 65, "all"):
            with pytest.raises(PydanticValidationError):
                ComputeBudgetSpec(max_parallelism=bad)


class TestWhatTheRunReports:
    def test_auto_is_reported_as_asked_and_resolved_in_the_manifest(self, monkeypatch):
        monkeypatch.setenv("SQT_NUM_THREADS", "4")
        result = run_experiment(_dataset(), _spec(), "ds")
        fits = result["validation_report"]["fits"]
        assert fits["max_parallelism"] == "auto"
        assert fits["fold_workers"] == "auto"
        assert fits["fold_parallel_limit"] == FOLD_LIMIT_BUDGET
        threads = load_manifest(result["model_id"]).environment["threads"]
        assert threads["auto_parallelism"] == 4
        assert threads["SQT_NUM_THREADS"] == "4"

    def test_a_search_reports_auto_too(self, monkeypatch):
        monkeypatch.setenv("SQT_NUM_THREADS", "4")
        search = SearchSpec(param_grid={"alpha": [0.1, 1.0, 10.0]}, inner_splits=2)
        result = run_experiment(
            _dataset(), _spec("ridge", search=search), "ds", register=False
        )
        reports = result["validation_report"]["hyperparameter_search"]
        assert reports and {r["max_parallelism"] for r in reports} == {"auto"}
        assert result["validation_report"]["final_search"]["max_parallelism"] == (
            "auto"
        )

    def test_an_explicit_budget_reports_its_numbers(self):
        result = run_experiment(_dataset(), _spec(budget=4), "ds", register=False)
        fits = result["validation_report"]["fits"]
        assert fits["max_parallelism"] == 4
        assert fits["fold_workers"] == 4
        assert fits["fold_parallel_limit"] == FOLD_LIMIT_BUDGET
        every = run_experiment(_dataset(), _spec(budget=64), "ds", register=False)
        assert every["validation_report"]["fits"]["fold_workers"] == every["n_folds"]
        assert every["validation_report"]["fits"]["fold_parallel_limit"] is None

    def test_the_output_does_not_depend_on_the_machine(self, monkeypatch):
        """What a replay compares: the tool output with run ids normalized,
        at 'auto' on a two-CPU and a sixteen-CPU process."""
        from standard_quant_tools.modeling.agent.models import (
            RunModelExperimentResult,
        )

        dataset = _dataset()
        outputs = []
        for threads in ("2", "16"):
            monkeypatch.setenv("SQT_NUM_THREADS", threads)
            result = run_experiment(dataset, _spec(), "ds")
            result["oos_predictions_ref"] = None
            outputs.append(
                hash_payload(
                    normalize_identifiers(
                        RunModelExperimentResult(**result).model_dump(mode="json")
                    )
                )
            )
        assert outputs[0] == outputs[1]


class TestTheFoldSchedule:
    def test_each_reason(self):
        assert _fold_schedule(_spec("ridge", 8), Ridge, 10, 8) == (
            1,
            FOLD_LIMIT_ESTIMATOR,
        )
        assert _fold_schedule(_spec(), RandomForestRegressor, 1, 8) == (
            1,
            FOLD_LIMIT_ONE_FOLD,
        )
        assert _fold_schedule(
            _spec(params={"n_jobs": 2}), RandomForestRegressor, 10, 8
        ) == (1, FOLD_LIMIT_N_JOBS)
        search = SearchSpec(param_grid={"max_depth": [2, 3]}, inner_splits=2)
        assert _fold_schedule(
            _spec("gradient_boosting", search=search), GradientBoostingRegressor, 10, 8
        ) == (1, FOLD_LIMIT_SEARCH)
        assert _fold_schedule(_spec(), RandomForestRegressor, 10, 1) == (
            1,
            FOLD_LIMIT_BUDGET,
        )
        assert _fold_schedule(_spec(), RandomForestRegressor, 10, 4) == (
            4,
            FOLD_LIMIT_BUDGET,
        )
        assert _fold_schedule(_spec(), RandomForestRegressor, 10, 16) == (10, None)

    def test_a_limit_the_machine_does_not_decide_is_reported_under_auto(self):
        result = run_experiment(_dataset(), _spec("ridge"), "ds", register=False)
        fits = result["validation_report"]["fits"]
        assert fits["fold_workers"] == 1
        assert fits["fold_parallel_limit"] == FOLD_LIMIT_ESTIMATOR


class TestTheCacheBlock:
    def test_a_private_cache_names_no_projection(self):
        result = run_experiment(_dataset(), _spec("ridge"), "ds", register=False)
        cache = result["validation_report"]["cache"]
        assert set(cache) == {"hits", "misses", "shared"}
        assert cache["shared"] is False

    def test_a_shared_cache_keeps_both(self):
        result = run_experiment(
            _dataset(), _spec("ridge"), "ds", register=False, fold_cache=FoldCache()
        )
        cache = result["validation_report"]["cache"]
        assert set(cache) == {"hits", "misses", "shared", "projections", "projectable"}


class TestTheEnvironmentRecordsTheWaitPolicy:
    def test_omp_wait_policy_is_a_thread_variable(self, monkeypatch):
        assert "OMP_WAIT_POLICY" in THREAD_VARIABLES
        monkeypatch.setenv("OMP_WAIT_POLICY", "PASSIVE")
        threads = environment_fingerprint()["threads"]
        assert threads["OMP_WAIT_POLICY"] == "PASSIVE"
        assert threads["auto_parallelism"] == auto_parallelism()

    def test_an_unusable_setting_does_not_break_the_fingerprint(self, monkeypatch):
        monkeypatch.setenv("SQT_NUM_THREADS", "many")
        assert environment_fingerprint()["threads"]["auto_parallelism"] is None
