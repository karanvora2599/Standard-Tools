"""
Regressions on the modeling tool surface found by reading it against the
runtime it fronts (the modeling runtime plan's defect list; CHANGELOG,
phase 0, 2026-09-20).

Each class pins one finding. The shape of every finding was the same: a
tool that was registered, dispatchable and documented, and whose answer was
either wrong or unreachable in a way no existing test could see.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import (
    BuildEnsembleInput,
    BuildModelDatasetInput,
    CheckLeakageInput,
    ListDatasetsInput,
    RunModelExperimentInput,
    ScoreModelInput,
    ValidateModelSpecInput,
)
from standard_quant_tools.modeling.agent.tools import (
    _headline,
    build_model_dataset,
    build_model_ensemble,
    check_leakage,
    list_datasets,
    run_model_experiment,
    score_model,
    validate_model_spec,
)
from standard_quant_tools.modeling.estimators.registry import ESTIMATOR_REGISTRY
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    SearchSpec,
    TargetSpec,
    ValidationSpec,
)

from .test_scoring import _dataset_spec, _train_a_model_with_spec

RANKERS = [name for task, name in ESTIMATOR_REGISTRY if task == "ranking"]
requires_ranker = pytest.mark.skipif(
    not RANKERS, reason="neither lightgbm nor xgboost is installed"
)

UNIVERSE = ["AAA", "BBB", "CCC"]


def _tool_dataset_spec(target: "TargetSpec | None" = None) -> DatasetSpec:
    return DatasetSpec(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=target or TargetSpec(horizon=5),
        benchmark="SPY",
    )


def _ridge(**overrides) -> ModelSpec:
    fields = dict(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )
    fields.update(overrides)
    return ModelSpec(**fields)


@pytest.fixture
def built_dataset_id(patched_multi_factory) -> str:
    return build_model_dataset(
        BuildModelDatasetInput(spec=_tool_dataset_spec())
    ).dataset_id


# ── F2: scoring goes through the task's adapter ─────────────────────────


class TestScoringGoesThroughTheAdapter:
    """
    `score_model` branched `task == "regression"` -> predict, else
    positive_class_proba, written when those were the only two tasks. A
    ranker is neither and has no predict_proba, so a registered ranking
    model trained, validated and then failed inside scoring.
    """

    @requires_ranker
    def test_a_ranking_model_scores(self, patched_multi_factory):
        model_id = _train_a_model_with_spec(
            _dataset_spec(),
            dataset_id="ds_rank_scoring",
            model_spec=ModelSpec(
                task="ranking",
                estimator=EstimatorSpec(type=RANKERS[0], params={"n_estimators": 20}),
                validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
                random_seed=1,
            ),
        )
        from standard_quant_tools.modeling.scoring import score_model as _score

        result = _score(model_id, as_of="2023-12-29", universe=UNIVERSE)
        assert result["n_entities"] == len(UNIVERSE)
        frame = _artifacts.load_artifact(result["predictions_uri"])
        # A ranker's score is an ordering, so the only thing to assert is
        # that every entity got one and they are not all the same.
        assert frame["prediction"].notna().all()
        assert frame["prediction"].nunique() > 1

    def test_a_classifier_still_scores_a_probability(self, patched_multi_factory):
        """The other branch must survive the routing change: a classifier's
        score is the positive-class probability, bounded in [0, 1]."""
        model_id = _train_a_model_with_spec(
            _dataset_spec(target=TargetSpec(type="forward_direction", horizon=5)),
            dataset_id="ds_clf_scoring",
            model_spec=ModelSpec(
                task="classification",
                estimator=EstimatorSpec(type="logistic", params={"C": 1.0}),
                validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
                random_seed=1,
            ),
        )
        from standard_quant_tools.modeling.scoring import score_model as _score

        result = _score(model_id, as_of="2023-12-29", universe=UNIVERSE)
        frame = _artifacts.load_artifact(result["predictions_uri"])
        assert ((frame["prediction"] >= 0.0) & (frame["prediction"] <= 1.0)).all()


# ── F5: the dataset metadata records what the listing reads ─────────────


class TestDatasetListingReportsWhatWasBuilt:
    """
    `list_datasets` read `rows`, `start_date` and `end_date` from
    dataset_meta.json from the day it was written, and nothing wrote them:
    every dataset listed with no row count and no span, sorted on a key
    that was always None.
    """

    def test_a_built_dataset_lists_with_its_extent(self, built_dataset_id):
        listing = list_datasets(ListDatasetsInput())
        entry = next(d for d in listing.datasets if d.dataset_id == built_dataset_id)
        meta = _artifacts.load_json(
            str(_artifacts.run_dir(built_dataset_id) / "dataset_meta.json")
        )
        panel = _artifacts.load_artifact(
            str(_artifacts.run_dir(built_dataset_id) / "panel.parquet")
        )
        # Planted against the panel itself, not against the metadata that
        # the listing reads -- the two agreeing with each other is not the
        # claim; agreeing with the data is.
        assert entry.rows == len(panel) == meta["rows"]
        assert entry.n_dates == panel["date"].nunique()
        assert entry.start_date == str(pd.Timestamp(panel["date"].min()).date())
        assert entry.end_date == str(pd.Timestamp(panel["date"].max()).date())
        assert entry.entities == len(UNIVERSE)
        assert entry.features == 2
        assert entry.provider == "yfinance"
        assert entry.interval == "1d"
        assert entry.target_id == "forward_return:5"

    def test_a_dataset_built_before_the_keys_existed_still_lists(self, tmp_path):
        """The honest answer for an older dataset is None, not a crash."""
        directory = _artifacts.run_dir("ds_legacy_listing")
        directory.mkdir(parents=True, exist_ok=True)
        _artifacts.save_json(
            directory,
            "dataset_meta",
            {"feature_ids": ["technical.rsi"], "target_id": "forward_return:5"},
        )
        listing = list_datasets(ListDatasetsInput())
        entry = next(d for d in listing.datasets if d.dataset_id == "ds_legacy_listing")
        assert entry.rows is None
        assert entry.n_dates is None
        assert entry.start_date is None

    def test_check_leakage_reports_the_recorded_coverage(self, built_dataset_id):
        result = check_leakage(
            CheckLeakageInput(
                feature_ids=["technical.rsi"], dataset_id=built_dataset_id
            )
        )
        assert result.dataset_coverage["rows"] > 0
        assert (
            result.dataset_coverage["start_date"] < result.dataset_coverage["end_date"]
        )


# ── F4: validate_model_spec estimates the work that will actually run ────


class TestValidateModelSpecEstimatesRealWork:
    """
    The fold count was read off `validation.n_splits`, which every spec
    carries at its default of 5 whether or not the method is purged
    k-fold -- so a walk-forward spec was always "5 folds" regardless of its
    windows or the dataset. And the dataset branch compared the dataset's
    feature list to itself, so it could never fail.
    """

    def test_walk_forward_fold_count_matches_what_the_experiment_runs(
        self, built_dataset_id
    ):
        spec = _ridge()
        estimate = validate_model_spec(
            ValidateModelSpecInput(spec=spec, dataset_id=built_dataset_id)
        )
        actual = run_model_experiment(
            RunModelExperimentInput(dataset_id=built_dataset_id, spec=spec)
        )
        expected = actual.validation_report["n_folds_expected"]
        assert estimate.estimated_folds == expected
        # One fit per fold, plus the refit on the full panel that follows
        # them -- the plan the experiment executed, which it reports back.
        assert estimate.estimated_fits == expected + 1
        assert actual.validation_report["fits"]["planned"] == expected + 1
        # The old answer, which no walk-forward spec over this dataset
        # produces: the windows yield far more than five folds here.
        assert expected != 5

    def test_a_search_grid_multiplies_through_the_real_fold_count(
        self, built_dataset_id
    ):
        spec = _ridge(
            search=SearchSpec(
                method="grid",
                param_grid={"alpha": [0.1, 1.0, 10.0]},
                inner_splits=2,
            )
        )
        result = validate_model_spec(
            ValidateModelSpecInput(spec=spec, dataset_id=built_dataset_id)
        )
        folds = result.estimated_folds
        assert folds and folds > 0
        # Against what RAN, not a formula: every fold that searched fitted
        # three candidates on two inner folds, every fold fitted once, and
        # the full panel was refit once.
        actual = run_model_experiment(
            RunModelExperimentInput(dataset_id=built_dataset_id, spec=spec)
        )
        searched = [
            r
            for r in actual.validation_report["hyperparameter_search"]
            if r["searched"]
        ]
        assert searched
        # ... plus the full-panel search that chooses the deployed parameters.
        assert result.estimated_fits == folds + 1 + 3 * 2 * len(searched) + 3 * 2
        assert result.estimated_fits == actual.validation_report["fits"]["planned"]

    def test_without_a_dataset_a_walk_forward_count_is_unknown_not_guessed(self):
        result = validate_model_spec(ValidateModelSpecInput(spec=_ridge()))
        assert result.valid
        assert result.estimated_folds is None
        assert result.estimated_fits is None
        assert any("dataset_id" in note for note in result.notes)

    def test_without_a_dataset_purged_kfold_reports_its_own_n_splits(self):
        spec = _ridge(validation=ValidationSpec(method="purged_kfold", n_splits=7))
        result = validate_model_spec(ValidateModelSpecInput(spec=spec))
        assert result.estimated_folds == 7
        assert result.estimated_fits == 7 + 1  # plus the full-panel refit

    def test_a_dataset_recorded_without_n_dates_is_reported_as_unknown(self):
        directory = _artifacts.run_dir("ds_legacy_validate")
        directory.mkdir(parents=True, exist_ok=True)
        _artifacts.save_json(
            directory,
            "dataset_meta",
            {"feature_ids": ["technical.rsi"], "target_id": "forward_return:5"},
        )
        result = validate_model_spec(
            ValidateModelSpecInput(spec=_ridge(), dataset_id="ds_legacy_validate")
        )
        assert result.valid
        assert result.estimated_folds is None

    def test_a_task_the_label_cannot_serve_is_refused_before_any_fit(
        self, built_dataset_id
    ):
        """This used to surface only inside run_model_experiment, after the
        panel had been loaded and hashed."""
        spec = ModelSpec(
            task="classification",
            estimator=EstimatorSpec(type="logistic"),
            validation=ValidationSpec(train_window=150, test_window=30),
        )
        result = validate_model_spec(
            ValidateModelSpecInput(spec=spec, dataset_id=built_dataset_id)
        )
        assert not result.valid
        assert [p.where for p in result.problems] == ["target"]
        assert "forward_return" in result.problems[0].problem

    def test_a_compatible_task_passes_the_same_check(self, built_dataset_id):
        result = validate_model_spec(
            ValidateModelSpecInput(spec=_ridge(), dataset_id=built_dataset_id)
        )
        assert result.valid, result.problems

    def test_a_misspelled_grid_axis_is_a_problem_not_a_silent_search(self):
        spec = _ridge(search=SearchSpec(param_grid={"alpah": [0.1, 1.0]}))
        result = validate_model_spec(ValidateModelSpecInput(spec=spec))
        assert not result.valid
        assert [p.where for p in result.problems] == ["search.param_grid"]
        assert "alpah" in result.problems[0].problem

    def test_a_grid_value_outside_the_bound_is_a_problem(self):
        spec = _ridge(search=SearchSpec(param_grid={"alpha": [0.1, -1.0]}))
        result = validate_model_spec(ValidateModelSpecInput(spec=spec))
        assert not result.valid
        assert result.problems[0].where == "search.param_grid"

    def test_a_grid_inside_the_bounds_is_not_a_problem(self):
        spec = _ridge(search=SearchSpec(param_grid={"alpha": [0.1, 1.0, 10.0]}))
        result = validate_model_spec(ValidateModelSpecInput(spec=spec))
        assert result.valid, result.problems


# ── F6: the headline metric is the one the report leads with ────────────


class TestHeadlineMetricIsCrossSectional:
    """
    Regression models were ranked by the POOLED Pearson `ic`, the metric the
    modeling guide's own "What the metrics mean" section says conflates
    cross-sectional skill with tracking the market's level.
    """

    def test_regression_leads_with_cross_sectional_rank_ic(self):
        metrics = {"ic": 0.9, "rank_ic": 0.8, "r2": 0.5, "cs_rank_ic_mean": 0.03}
        assert _headline("regression", metrics) == ("cs_rank_ic_mean", 0.03)

    def test_ranking_leads_with_the_same_metric(self):
        metrics = {"ndcg_at_10": 0.7, "cs_rank_ic_mean": 0.04}
        assert _headline("ranking", metrics) == ("cs_rank_ic_mean", 0.04)

    def test_classification_is_unchanged(self):
        assert _headline("classification", {"auc": 0.6, "accuracy": 0.55}) == (
            "auc",
            0.6,
        )

    def test_a_manifest_from_before_the_family_existed_still_ranks(self):
        """The fallbacks are for older manifests, not a second opinion."""
        assert _headline("regression", {"ic": 0.9, "r2": 0.5}) == ("ic", 0.9)

    def test_an_explicit_metric_still_wins(self):
        metrics = {"ic": 0.9, "cs_rank_ic_mean": 0.03}
        assert _headline("regression", metrics, "ic") == ("ic", 0.9)


# ── F8: the ensemble tool can run at all ────────────────────────────────


class TestTheEnsembleToolRuns:
    """
    `build_model_ensemble` called a bare `publish` that nothing in its
    module defined, so it raised NameError on every call that got past
    loading its models. Nothing got that far: the only test naming it
    checked that it was registered.
    """

    def test_two_models_combine_into_a_resolvable_reference(self, built_dataset_id):
        from standard_quant_tools.agent.runtimes import handoff

        ids = [
            run_model_experiment(
                RunModelExperimentInput(
                    dataset_id=built_dataset_id, spec=_ridge(random_seed=s)
                )
            ).model_id
            for s in (1, 2)
        ]
        result = build_model_ensemble(
            BuildEnsembleInput(model_ids=ids, run_id="run_ensemble", name="combined")
        )
        assert result.ref.startswith("sqt://")
        frame = handoff.resolve(result.ref, expect="predictions")
        assert set(["date", "entity", "prediction"]) <= set(frame.columns)
        assert len(frame) == result.n_rows > 0
        assert result.model_ids == ids
