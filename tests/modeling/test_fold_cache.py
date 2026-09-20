"""
The fold cache: a pipeline fitted once per fold, and read for every run
and every candidate that needs the same matrices.

Two claims are planted. That a column-wise pipeline's matrices for a
feature subset are the wider run's columns EXACTLY -- a run that projected
them reports the same out-of-sample numbers, to the bit, as a run that
fitted them -- and that the reuse is counted where it happens: the inner
search preprocesses each inner fold once rather than once per candidate,
and an ablation fits each fold's pipeline once for the baseline and never
again. The other direction is planted too: a pipeline with a step that is
not column-wise projects nothing and refits.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import engine
from standard_quant_tools.modeling.agent.feature_models import FeatureAblationInput
from standard_quant_tools.modeling.agent.feature_tools import run_feature_ablation
from standard_quant_tools.modeling.agent.models import BuildModelDatasetInput
from standard_quant_tools.modeling.agent.tools import build_model_dataset
from standard_quant_tools.modeling.cache import FoldCache, column_wise_pipeline
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    PreprocessingSpec,
    SearchSpec,
    StepSpec,
    TargetSpec,
    ValidationSpec,
)

UNIVERSE = ["AAA", "BBB", "CCC"]
FEATURES = ["technical.rsi", "market.momentum", "risk.realized_volatility"]


def _dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id=f) for f in FEATURES],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )


def _ridge(**overrides) -> ModelSpec:
    fields = dict(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(
            train_window=120, test_window=40, embargo=2, min_folds=1
        ),
        random_seed=3,
    )
    fields.update(overrides)
    return ModelSpec(**fields)


def _frames(columns, n=6):
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.normal(size=(n, len(columns))), columns=columns)
    test = pd.DataFrame(rng.normal(size=(n // 2, len(columns))), columns=columns)
    return train, test


class TestTheCacheAlone:
    def test_an_exact_match_is_a_hit_and_the_same_object(self):
        cache = FoldCache()
        train, test = _frames(["a", "b"])
        assert cache.lookup("k", ["a", "b"]) is None
        cache.store("k", ["a", "b"], train, test, projectable=True)
        got = cache.lookup("k", ["a", "b"])
        assert got is not None and got[0] is train and got[1] is test
        assert cache.stats() == {"hits": 1, "misses": 1, "projections": 0, "entries": 1}

    def test_a_column_wise_entry_projects_a_subset_exactly(self):
        cache = FoldCache()
        train, test = _frames(["a", "b", "c"])
        cache.store("k", ["a", "b", "c"], train, test, projectable=True)
        got = cache.lookup("k", ["c", "a"])
        assert got is not None
        pd.testing.assert_frame_equal(got[0], train[["c", "a"]])
        pd.testing.assert_frame_equal(got[1], test[["c", "a"]])
        assert cache.projections == 1 and cache.misses == 0
        # A column the entry never had is a miss, not a partial answer.
        assert cache.lookup("k", ["a", "z"]) is None
        assert cache.misses == 1

    def test_an_entry_that_is_not_column_wise_never_projects(self):
        cache = FoldCache()
        train, test = _frames(["a", "b"])
        cache.store("k", ["a", "b"], train, test, projectable=False)
        assert cache.lookup("k", ["a"]) is None
        assert cache.lookup("k", ["a", "b"]) is not None

    def test_an_entry_whose_output_columns_moved_never_projects(self):
        # A missingness indicator is column-wise and ADDS a column, so its
        # output cannot be read for a feature subset by name.
        cache = FoldCache()
        train, test = _frames(["a", "b", "b__missing"])
        cache.store("k", ["a", "b"], train, test, projectable=True)
        assert cache.lookup("k", ["a"]) is None
        assert cache.lookup("k", ["a", "b"]) is not None

    def test_a_narrower_store_does_not_replace_a_wider_projectable_entry(self):
        cache = FoldCache()
        wide_train, wide_test = _frames(["a", "b", "c"])
        cache.store("k", ["a", "b", "c"], wide_train, wide_test, projectable=True)
        narrow_train, narrow_test = _frames(["a"])
        cache.store("k", ["a"], narrow_train, narrow_test, projectable=True)
        got = cache.lookup("k", ["b"])
        assert got is not None and got[0].shape[1] == 1
        assert len(cache) == 1

    def test_column_wise_is_read_off_the_registry(self):
        default = PreprocessingSpec().resolved_steps
        assert column_wise_pipeline(default)
        assert column_wise_pipeline(["winsorize", "zscore", "impute"])
        assert not column_wise_pipeline(
            [StepSpec(type="zscore"), StepSpec(type="pca_whiten")]
        )
        assert not column_wise_pipeline([{"type": "pca_whiten"}])


class TestTheEngineReuses:
    @pytest.fixture
    def dataset(self, patched_multi_factory):
        return build_dataset(_dataset_spec())

    def test_a_shared_cache_projects_a_feature_subset_and_the_numbers_are_identical(
        self, dataset
    ):
        cache = FoldCache()
        spec = _ridge()
        full = run_experiment(dataset, spec, "ds", register=False, fold_cache=cache)
        n_folds = full["n_folds"]
        assert full["validation_report"]["cache"] == {
            "hits": 0,
            "misses": n_folds,
            "projections": 0,
            "shared": True,
            "projectable": True,
        }
        subset = {**dataset, "feature_ids": dataset["feature_ids"][:2]}
        projected = run_experiment(subset, spec, "ds", register=False, fold_cache=cache)
        assert projected["validation_report"]["cache"] == {
            "hits": 0,
            "misses": 0,
            "projections": n_folds,
            "shared": True,
            "projectable": True,
        }
        # Exact: the same numbers a run that fitted the pipeline reports.
        fitted = run_experiment(subset, spec, "ds", register=False)
        assert projected["oos_metrics"] == fitted["oos_metrics"]
        assert fitted["validation_report"]["cache"]["misses"] == n_folds
        assert fitted["validation_report"]["cache"]["shared"] is False
        # And a fresh private cache sees the same node hashes, because the
        # cache changed nothing about what was fitted.
        assert [f["node_hash"] for f in projected["validation_report"]["folds"]] == [
            f["node_hash"] for f in fitted["validation_report"]["folds"]
        ]

    def test_a_step_that_is_not_column_wise_refits_the_subset(self, dataset):
        cache = FoldCache()
        spec = _ridge(
            preprocessing=PreprocessingSpec(
                steps=[
                    StepSpec(type="zscore"),
                    StepSpec(type="pca_whiten", params={"n_components": 2}),
                ]
            )
        )
        full = run_experiment(dataset, spec, "ds", register=False, fold_cache=cache)
        subset = {**dataset, "feature_ids": dataset["feature_ids"][:2]}
        again = run_experiment(subset, spec, "ds", register=False, fold_cache=cache)
        assert full["validation_report"]["cache"]["projectable"] is False
        assert again["validation_report"]["cache"]["projections"] == 0
        assert again["validation_report"]["cache"]["misses"] == again["n_folds"]
        # The same features are still an exact hit.
        third = run_experiment(dataset, spec, "ds", register=False, fold_cache=cache)
        assert third["validation_report"]["cache"]["hits"] == third["n_folds"]

    def test_the_inner_search_preprocesses_each_inner_fold_once(
        self, dataset, monkeypatch
    ):
        calls = []
        original = engine._preprocess

        def counting(*args, **kwargs):
            calls.append(len(args[1]))
            return original(*args, **kwargs)

        monkeypatch.setattr(engine, "_preprocess", counting)
        spec = _ridge(
            search=SearchSpec(param_grid={"alpha": [0.1, 1.0, 10.0]}, inner_splits=2)
        )
        result = run_experiment(dataset, spec, "ds", register=False)
        n_folds = result["n_folds"]
        searched = [
            r
            for r in result["validation_report"]["hyperparameter_search"]
            if r["searched"]
        ]
        assert searched
        # One per outer fold, plus one per inner fold of every fold that
        # searched -- not one per candidate per inner fold.
        assert len(calls) == n_folds + 2 * len(searched)
        assert len(calls) < n_folds * (1 + 3 * 2)
        report = result["validation_report"]["cache"]
        assert report["misses"] == len(calls)
        assert report["hits"] == 2 * 2 * len(
            searched
        )  # two more candidates x two folds

    def test_a_shared_cache_needs_a_dataset_hash(self, dataset):
        unhashed = {k: v for k, v in dataset.items() if k != "data_hash"}
        with pytest.raises(ValidationError, match="data_hash"):
            run_experiment(
                unhashed, _ridge(), "ds", register=False, fold_cache=FoldCache()
            )
        # Without a shared cache the run does not need one.
        run_experiment(unhashed, _ridge(), "ds", register=False)


class TestTheAblationReuses:
    def test_only_the_baseline_fits_the_pipeline(self, patched_multi_factory):
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_dataset_spec())
        ).dataset_id
        result = run_feature_ablation(
            FeatureAblationInput(dataset_id=dataset_id, spec=_ridge())
        )
        assert result.n_features == len(FEATURES)
        assert result.preprocessing_fitted == result.n_folds
        assert result.preprocessing_reused == len(FEATURES) * result.n_folds
        assert result.n_fits == (len(FEATURES) + 1) * result.n_folds
