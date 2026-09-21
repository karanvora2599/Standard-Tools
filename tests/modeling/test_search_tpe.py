"""
The tpe search backend: optuna's sampler choosing the candidates, and the
same purged, embargoed inner folds scoring them that grid and random use.

Planted. A fit_predict whose error is smallest at alpha = 3 lets the grid
find 3.0 exactly and the sampler land within half a unit of it over a
continuous range -- on frames that are, fold for fold, the ones the grid
was handed. The other direction is planted too: the spec refuses the tpe
fields under the other methods, a name on both kinds of axis, and a tpe
spec on a machine without optuna, before any fit.
"""

import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.models import ValidateModelSpecInput
from standard_quant_tools.modeling.agent.tools import validate_model_spec
from standard_quant_tools.modeling.capabilities import modeling_capabilities
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    ParamRange,
    SearchSpec,
    TargetSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation import search as search_module
from standard_quant_tools.modeling.validation.search import (
    n_search_candidates,
    search_best_params,
    search_candidates,
)

from .test_search_purge import _frame

requires_optuna = pytest.mark.skipif(
    not search_module.optuna_available(), reason="optuna is not installed"
)


def _dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )


def _tpe_spec(validation=None, **search) -> ModelSpec:
    fields = dict(
        method="tpe",
        param_ranges={"alpha": ParamRange(low=0.01, high=100.0, log=True)},
        max_trials=5,
        inner_splits=2,
    )
    fields.update(search)
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={}),
        validation=validation
        or ValidationSpec(train_window=150, test_window=30, embargo=5, min_folds=1),
        search=SearchSpec(**fields),
        random_seed=1,
    )


class TestTheSpec:
    def test_tpe_takes_continuous_axes_and_the_other_methods_refuse_the_tpe_fields(
        self,
    ):
        spec = SearchSpec(
            method="tpe",
            param_ranges={"alpha": ParamRange(low=0.01, high=100.0, log=True)},
            max_trials=10,
        )
        assert spec.param_grid == {} and n_search_candidates(spec) == 10
        with pytest.raises(PydanticValidationError, match="tpe"):
            SearchSpec(
                param_grid={"alpha": [1.0]},
                param_ranges={"l1_ratio": ParamRange(low=0.0, high=1.0)},
            )
        with pytest.raises(PydanticValidationError, match="tpe"):
            SearchSpec(param_grid={"alpha": [1.0]}, max_trials=5)
        with pytest.raises(PydanticValidationError, match="tpe"):
            SearchSpec(method="random", param_grid={"alpha": [1.0]}, early_pruning=True)
        with pytest.raises(PydanticValidationError, match="at least one axis"):
            SearchSpec(method="tpe")
        with pytest.raises(PydanticValidationError, match="at least one parameter"):
            SearchSpec(param_grid={})

    def test_a_range_is_ordered_a_log_axis_positive_an_integer_axis_integral(self):
        ParamRange(low=0.0, high=1.0)
        ParamRange(low=2, high=64, integer=True, log=True)
        with pytest.raises(PydanticValidationError, match="low < high"):
            ParamRange(low=1.0, high=1.0)
        with pytest.raises(PydanticValidationError, match="log"):
            ParamRange(low=0.0, high=1.0, log=True)
        with pytest.raises(PydanticValidationError, match="integer"):
            ParamRange(low=0.5, high=3.0, integer=True)

    def test_a_name_on_both_kinds_of_axis_is_refused(self):
        with pytest.raises(PydanticValidationError, match="both"):
            SearchSpec(
                method="tpe",
                param_grid={"alpha": [1.0]},
                param_ranges={"alpha": ParamRange(low=0.0, high=1.0)},
            )

    def test_tpe_candidates_are_counted_not_enumerated(self):
        spec = SearchSpec(
            method="tpe",
            param_ranges={"alpha": ParamRange(low=0.0, high=1.0)},
            max_trials=7,
        )
        assert n_search_candidates(spec) == 7
        with pytest.raises(ValidationError, match="enumerat"):
            search_candidates(spec, 0)
        grid = SearchSpec(param_grid={"a": [1, 2, 3], "b": [True, False]})
        assert n_search_candidates(grid) == 6 == len(search_candidates(grid, 0))
        sampled = SearchSpec(
            method="random", param_grid={"a": [1, 2, 3], "b": [True, False]}, n_iter=4
        )
        assert n_search_candidates(sampled) == 4 == len(search_candidates(sampled, 0))


def _planted(optimum: float):
    """A fit_predict whose predictions are the target scaled by how close
    alpha is to `optimum`, so neg_mae is largest exactly there. Records
    what it was handed, fold index included."""
    seen = []

    def fit_predict(params, inner_train, inner_test, fold_index):
        seen.append((dict(params), inner_train, inner_test, fold_index))
        scale = 1.0 - min(abs(float(params["alpha"]) - optimum) / 10.0, 0.99)
        return inner_test["target"].to_numpy() * scale, None

    return fit_predict, seen


def _search(frame, spec, fit_predict, seed=0):
    return search_best_params(
        task="regression",
        search_spec=spec,
        base_params={},
        train_frame=frame,
        feature_ids=["f"],
        random_seed=seed,
        fit_predict=fit_predict,
        embargo=1,
        label_end=frame["label_end_date"].to_numpy(),
    )


@requires_optuna
class TestTheSearch:
    def test_tpe_finds_the_planted_optimum_on_the_folds_the_grid_was_handed(self):
        frame = _frame(n_dates=80)
        grid_fit, grid_seen = _planted(3.0)
        _params, grid_report = _search(
            frame,
            SearchSpec(
                param_grid={"alpha": [0.0, 1.5, 3.0, 4.5, 6.0]},
                inner_splits=3,
                scoring="neg_mae",
            ),
            grid_fit,
        )
        tpe_fit, tpe_seen = _planted(3.0)
        params, report = _search(
            frame,
            SearchSpec(
                method="tpe",
                param_ranges={"alpha": ParamRange(low=0.0, high=6.0)},
                max_trials=40,
                inner_splits=3,
                scoring="neg_mae",
            ),
            tpe_fit,
        )
        assert grid_report["best_params"] == {"alpha": 3.0}
        assert report["searched"] and report["method"] == "tpe"
        assert report["n_candidates"] == 40 == len(report["candidates"])
        assert report["n_trials_pruned"] == 0
        assert abs(params["alpha"] - 3.0) < 0.5
        assert report["best_score"] == max(c["score"] for c in report["candidates"])
        # The same discipline: the same purge counts, the same embargo,
        # and fold for fold the same frames the grid candidates saw.
        assert (
            report["n_train_rows_purged_overlap"]
            == grid_report["n_train_rows_purged_overlap"]
        )
        assert report["embargo"] == 1 and report["purged_on_label_end"]
        grid_folds = {i: (train, test) for _p, train, test, i in grid_seen}
        assert len(grid_folds) == 3
        for _p, train, test, i in tpe_seen:
            pd.testing.assert_frame_equal(train, grid_folds[i][0])
            pd.testing.assert_frame_equal(test, grid_folds[i][1])
            assert not (train["label_end_date"] >= test["date"].min()).any()

    def test_the_search_is_reproducible_from_the_seed(self):
        frame = _frame(n_dates=60)
        spec = SearchSpec(
            method="tpe",
            param_ranges={"alpha": ParamRange(low=0.0, high=6.0)},
            max_trials=8,
            inner_splits=2,
            scoring="neg_mae",
        )
        first = _search(frame, spec, _planted(3.0)[0], seed=5)[1]
        second = _search(frame, spec, _planted(3.0)[0], seed=5)[1]
        other = _search(frame, spec, _planted(3.0)[0], seed=6)[1]
        assert first["candidates"] == second["candidates"]
        assert [c["params"] for c in first["candidates"]] != [
            c["params"] for c in other["candidates"]
        ]

    def test_categorical_axes_come_from_the_grid(self):
        frame = _frame(n_dates=60)
        fit, seen = _planted(3.0)
        _p, report = _search(
            frame,
            SearchSpec(
                method="tpe",
                param_grid={"fit_intercept": [True, False]},
                param_ranges={"alpha": ParamRange(low=0.0, high=6.0)},
                max_trials=6,
                inner_splits=2,
                scoring="neg_mae",
            ),
            fit,
        )
        assert report["n_candidates"] == 6
        for params, *_ in seen:
            assert params["fit_intercept"] in (True, False)
            assert 0.0 <= params["alpha"] <= 6.0

    def test_pruning_stops_trials_and_says_so(self):
        frame = _frame(n_dates=80)
        fit, seen = _planted(3.0)
        _p, report = _search(
            frame,
            SearchSpec(
                method="tpe",
                param_ranges={"alpha": ParamRange(low=0.0, high=6.0)},
                max_trials=30,
                inner_splits=3,
                scoring="neg_mae",
                early_pruning=True,
            ),
            fit,
        )
        assert report["n_candidates"] == 30
        assert report["n_trials_pruned"] >= 1
        for candidate in report["candidates"]:
            if candidate["pruned"]:
                assert candidate["n_folds_scored"] < 3
            else:
                assert candidate["n_folds_scored"] == 3
        # Fewer fits than the full budget, which is the point.
        assert len(seen) < 30 * 3
        # And the best trial was never a pruned one.
        best = report["candidates"][0]
        assert not best["pruned"]


class TestWithoutOptuna:
    def test_a_tpe_spec_is_refused_by_name_before_any_fit(
        self, monkeypatch, patched_multi_factory
    ):
        monkeypatch.setattr(search_module, "optuna_available", lambda: False)
        dataset = build_dataset(_dataset_spec())
        with pytest.raises(ValidationError, match="optuna"):
            run_experiment(dataset, _tpe_spec(), "ds", register=False)
        result = validate_model_spec(ValidateModelSpecInput(spec=_tpe_spec()))
        assert not result.valid
        assert any(
            p.where == "search.method" and "optuna" in p.problem
            for p in result.problems
        )
        assert modeling_capabilities()["optional_dependencies"]["optuna"] is False


@requires_optuna
class TestThroughTheEngine:
    @pytest.fixture
    def dataset(self, patched_multi_factory):
        return build_dataset(_dataset_spec())

    def test_it_runs_and_reports_every_trial(self, dataset):
        result = run_experiment(dataset, _tpe_spec(), "ds", register=False)
        reports = [
            r
            for r in result["validation_report"]["hyperparameter_search"]
            if r["searched"]
        ]
        assert reports
        for report in reports:
            assert report["method"] == "tpe"
            assert report["n_candidates"] == 5 == len(report["candidates"])
            assert 0.01 <= report["best_params"]["alpha"] <= 100.0
            assert report["n_inner_folds"] == 2
        assert result["validation_report"]["fits"]["candidates_per_fold"] == 5

    def test_the_estimate_and_the_capabilities_know_it(self):
        spec = _tpe_spec(validation=ValidationSpec(method="purged_kfold", n_splits=4))
        result = validate_model_spec(ValidateModelSpecInput(spec=spec))
        assert result.valid
        assert result.estimated_fits == 4 * (1 + 5 * 2) + 1 + 5 * 2
        capabilities = modeling_capabilities()
        assert capabilities["hyperparameter_search"] == ["grid", "random", "tpe"]
        assert capabilities["optional_dependencies"]["optuna"] is True

    def test_a_range_end_outside_the_estimator_bound_is_a_problem(self):
        spec = _tpe_spec(param_ranges={"alpha": ParamRange(low=-1.0, high=1.0)})
        result = validate_model_spec(ValidateModelSpecInput(spec=spec))
        assert not result.valid
        assert [p.where for p in result.problems] == ["search.param_ranges"]
