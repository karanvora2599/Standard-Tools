"""
The three dry runs: the schedule, the weights, the transform.

Each of these reports something the engine computes on every call and
keeps to itself, so every test here is planted against the engine's own
answer rather than against the preview's arithmetic:

  * the fit count the plan reports is the fit count the experiment then
    runs, and the per-fold purge is the purge the panel implies -- which
    the spec validator's panel-free estimate cannot see at all;
  * a training window too short for its inner search is priced at one fit
    and named, a tpe search is counted and not enumerated, and an
    over-budget plan is reported rather than refused;
  * every weighting method normalizes to mean 1.0, a half-life equal to
    the panel's span spreads the weights by exactly two, and the composite
    is not the two halves multiplied;
  * the two preprocessing traps refuse with the steps' own messages, the
    indicator step doubles the width, and the default pair returns the
    fused state the engine persists.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent import tools as modeling_tools
from standard_quant_tools.modeling.agent.models import (
    BuildModelDatasetInput,
    RunModelExperimentInput,
    ValidateModelSpecInput,
)
from standard_quant_tools.modeling.agent.preview_models import (
    PlanModelExperimentInput,
    PreviewPreprocessingInput,
    PreviewSampleWeightsInput,
)
from standard_quant_tools.modeling.agent.preview_tools import (
    plan_model_experiment,
    preview_preprocessing,
    preview_sample_weights,
)
from standard_quant_tools.modeling.agent.tools import (
    _load_dataset_meta,
    _load_dataset_panel,
    build_model_dataset,
    run_model_experiment,
    validate_model_spec,
)
from standard_quant_tools.modeling.features.transforms import fit_preprocessing
from standard_quant_tools.modeling.plan import plan_experiment
from standard_quant_tools.modeling.preprocessing import (
    STATE_VERSION,
    FoldContext,
    fit_and_apply_pipeline,
    step_types,
)
from standard_quant_tools.modeling.specs import (
    ComputeBudgetSpec,
    ConformalSpec,
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    MissingDataSpec,
    ModelSpec,
    ParamRange,
    PreprocessingSpec,
    SearchSpec,
    StepSpec,
    TargetSpec,
    ValidationSpec,
    WeightingSpec,
)
from standard_quant_tools.modeling.validation.search import search_candidates

FEATURES = ["technical.rsi", "risk.atr_pct", "market.momentum"]
UNIVERSE = ["AAA", "BBB", "CCC"]
HORIZON = 5


def _dataset_spec(**kwargs) -> DatasetSpec:
    return DatasetSpec(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2030-01-01",
        features=[FeatureSpec(id=f) for f in FEATURES],
        target=TargetSpec(horizon=HORIZON),
        benchmark="SPY",
        **kwargs,
    )


def _ridge(**kwargs) -> ModelSpec:
    kwargs.setdefault("task", "regression")
    kwargs.setdefault("estimator", EstimatorSpec(type="ridge", params={"alpha": 1.0}))
    kwargs.setdefault(
        "validation",
        ValidationSpec(
            method="walk_forward", train_window=150, test_window=75, embargo=2
        ),
    )
    kwargs.setdefault("random_seed", 1)
    return ModelSpec(**kwargs)


def _unbudgeted(**kwargs) -> ModelSpec:
    """A spec whose ceiling is out of the way, so a test about the search
    is not also a test about the budget."""
    kwargs.setdefault("budget", ComputeBudgetSpec(max_fits=100_000))
    return _ridge(**kwargs)


@pytest.fixture
def complete_panel(patched_multi_factory) -> str:
    """A built dataset under the default complete-case policy: no holes,
    three feature columns, one five-bar label."""
    return build_model_dataset(BuildModelDatasetInput(spec=_dataset_spec())).dataset_id


@pytest.fixture
def panel_with_holes(patched_multi_factory) -> str:
    """The same dataset built with missing.policy='keep', so the feature
    warm-up NaN reaches the engine instead of costing the rows."""
    return build_model_dataset(
        BuildModelDatasetInput(
            spec=_dataset_spec(missing=MissingDataSpec(policy="keep"))
        )
    ).dataset_id


def _panel_span_days(dataset_id: str) -> float:
    meta, _directory = _load_dataset_meta(dataset_id)
    return float(
        (pd.Timestamp(meta["end_date"]) - pd.Timestamp(meta["start_date"])).days
    )


# ── plan_model_experiment ───────────────────────────────────────────────


class TestThePlanIsTheScheduleThatRuns:
    def test_the_planned_fit_count_is_the_one_the_experiment_reports_running(
        self, complete_panel
    ):
        spec = _ridge()
        planned = plan_model_experiment(
            PlanModelExperimentInput(dataset_id=complete_panel, spec=spec)
        )
        result = run_model_experiment(
            RunModelExperimentInput(dataset_id=complete_panel, spec=spec)
        )
        fits = result.validation_report["fits"]
        assert planned.n_fits == fits["planned"] > 0
        assert planned.n_fits_folds == fits["folds"]
        assert planned.n_fits_refit == fits["refit"]
        assert planned.n_candidates == fits["candidates_per_fold"]
        assert planned.max_fits == fits["max_fits"]
        assert planned.n_folds == len(result.validation_report["folds"])
        # And the per-fold schedule is the one the engine recorded, hash
        # for hash -- the plan is not a parallel estimate of the run.
        for fold, record in zip(planned.folds, result.validation_report["folds"]):
            assert fold.node_hash == record["node_hash"]
            assert fold.n_train_rows == record["n_train_rows"]
            assert fold.test_start == record["test_start"]

    def test_a_window_too_short_for_its_inner_search_is_priced_at_one_fit(
        self, complete_panel
    ):
        spec = _unbudgeted(
            validation=ValidationSpec(
                method="walk_forward", train_window=8, test_window=5, embargo=0
            ),
            search=SearchSpec(
                method="grid", param_grid={"alpha": [0.1, 1.0]}, inner_splits=3
            ),
        )
        result = plan_model_experiment(
            PlanModelExperimentInput(dataset_id=complete_panel, spec=spec)
        )
        assert result.fits_per_fit == 1
        assert result.n_candidates == 2
        assert {fold.n_inner_folds for fold in result.folds} == {0}
        assert {fold.n_fits for fold in result.folds} == {result.fits_per_fit}
        assert any("n_inner_folds=0" in w for w in result.warnings)
        # The search is still counted where it DOES run: once, on the full
        # date axis, choosing the parameters the refit deploys.
        assert result.n_inner_final == 3
        assert result.n_fits_final_search == 2 * 3

    def test_three_quantiles_and_five_conformal_blocks_cost_nine_fits_each(
        self, complete_panel
    ):
        spec = _ridge(
            estimator=EstimatorSpec(type="gradient_boosting"),
            quantiles=[0.05, 0.5, 0.95],
            intervals=ConformalSpec(calibration_folds=5),
        )
        result = plan_model_experiment(
            PlanModelExperimentInput(dataset_id=complete_panel, spec=spec)
        )
        assert result.fits_per_fit == 9
        assert {fold.n_fits for fold in result.folds} == {9}
        assert result.n_fits == 9 * result.n_folds + 9

    def test_the_panel_counts_the_purge_the_date_count_cannot(self, complete_panel):
        spec = _ridge()
        result = plan_model_experiment(
            PlanModelExperimentInput(dataset_id=complete_panel, spec=spec)
        )
        assert result.has_panel
        assert result.n_purged is not None and result.n_purged > 0
        assert all(fold.n_purged is not None for fold in result.folds)
        assert sum(fold.n_purged for fold in result.folds) == result.n_purged

        # The contrast is the point: the spec validator plans over a date
        # COUNT, which has no rows to purge, so the same spec reports None
        # on every fold and an n_fits that cannot see a shortened window.
        meta, _directory = _load_dataset_meta(complete_panel)
        without_panel = plan_experiment(spec, pd.RangeIndex(int(meta["n_dates"])))
        assert without_panel.n_purged is None
        assert all(fold.n_purged is None for fold in without_panel.folds)
        assert all(fold.n_train_rows is None for fold in without_panel.folds)
        estimate = validate_model_spec(
            ValidateModelSpecInput(spec=spec, dataset_id=complete_panel)
        )
        assert estimate.estimated_fits == without_panel.n_fits

    def test_a_tpe_search_is_counted_and_not_enumerated(self, complete_panel):
        spec = _unbudgeted(
            search=SearchSpec(
                method="tpe",
                param_ranges={"alpha": ParamRange(low=0.1, high=10.0, log=True)},
                max_trials=12,
            )
        )
        result = plan_model_experiment(
            PlanModelExperimentInput(
                dataset_id=complete_panel, spec=spec, include_candidates=True
            )
        )
        assert result.candidates is None
        assert result.n_candidates == 12
        assert any("n_search_candidates" in w for w in result.warnings)

    def test_a_grid_search_returns_the_combinations_it_will_score(self, complete_panel):
        search = SearchSpec(
            method="grid",
            param_grid={"alpha": [0.1, 1.0, 10.0], "fit_intercept": [True, False]},
            inner_splits=2,
        )
        spec = _unbudgeted(search=search)
        result = plan_model_experiment(
            PlanModelExperimentInput(
                dataset_id=complete_panel, spec=spec, include_candidates=True
            )
        )
        assert result.candidates == search_candidates(search, spec.random_seed)
        assert len(result.candidates) == result.n_candidates == 6
        # Off by default, and off is not "no candidates": the count stands.
        quiet = plan_model_experiment(
            PlanModelExperimentInput(dataset_id=complete_panel, spec=spec)
        )
        assert quiet.candidates is None and quiet.n_candidates == 6

    def test_an_over_budget_plan_is_reported_rather_than_refused(self, complete_panel):
        spec = _ridge(
            search=SearchSpec(
                method="grid", param_grid={"alpha": [0.1, 1.0, 10.0]}, inner_splits=2
            ),
            budget=ComputeBudgetSpec(max_fits=3),
        )
        result = plan_model_experiment(
            PlanModelExperimentInput(dataset_id=complete_panel, spec=spec)
        )
        assert result.within_budget is False
        assert result.max_fits == 3 and result.n_fits > 3
        assert any("budget.max_fits=3" in w for w in result.warnings)
        assert any(f"budget.max_fits={result.n_fits}" in w for w in result.warnings)
        # The dry run answered; the experiment still refuses the same spec.
        with pytest.raises(ValidationError, match=r"budget\.max_fits=3"):
            run_model_experiment(
                RunModelExperimentInput(dataset_id=complete_panel, spec=spec)
            )

    def test_the_fold_hash_moves_with_the_estimator_and_not_with_the_ceiling(
        self, complete_panel
    ):
        def _hashes(spec: ModelSpec):
            result = plan_model_experiment(
                PlanModelExperimentInput(dataset_id=complete_panel, spec=spec)
            )
            return [fold.node_hash for fold in result.folds]

        base = _hashes(_ridge())
        assert len(set(base)) == len(base) > 1
        assert _hashes(_ridge()) == base
        assert _hashes(_ridge(estimator=EstimatorSpec(type="lasso"))) != base
        assert (
            _hashes(
                _ridge(estimator=EstimatorSpec(type="ridge", params={"alpha": 2.0}))
            )
            != base
        )
        # budget.max_fits decides whether the fits happen, not what they
        # are: a fold cached under this hash is the same fold either way.
        assert _hashes(_ridge(budget=ComputeBudgetSpec(max_fits=499))) == base

    def test_the_fold_schedule_can_be_left_out_of_the_answer(self, complete_panel):
        spec = _ridge()
        result = plan_model_experiment(
            PlanModelExperimentInput(
                dataset_id=complete_panel, spec=spec, include_folds=False
            )
        )
        assert result.folds == []
        assert result.n_folds > 1 and result.n_fits > 1


# ── preview_sample_weights ──────────────────────────────────────────────


class TestTheWeightsAreDescribedBeforeTheyAreApplied:
    @pytest.mark.parametrize(
        "method",
        ["none", "label_uniqueness", "time_decay", "uniqueness_and_time_decay"],
    )
    def test_every_method_normalizes_to_mean_one(self, complete_panel, method):
        result = preview_sample_weights(
            PreviewSampleWeightsInput(
                dataset_id=complete_panel, weighting=WeightingSpec(method=method)
            )
        )
        assert result.method == method
        assert result.mean == pytest.approx(1.0, abs=1e-12)
        assert result.n_rows > 0
        assert result.min <= result.median <= result.max
        assert result.n_zero_weight == 0
        # The two sizes are always reported together and always with the
        # sentence that says they are not the same quantity.
        assert result.effective_sample_size_kish is not None
        assert result.effective_sample_size is not None
        assert any("not two" in w for w in result.warnings)

    def test_a_half_life_equal_to_the_panel_span_spreads_the_weights_by_two(
        self, complete_panel
    ):
        span = _panel_span_days(complete_panel)
        result = preview_sample_weights(
            PreviewSampleWeightsInput(
                dataset_id=complete_panel,
                weighting=WeightingSpec(method="time_decay", half_life_days=span),
            )
        )
        assert result.half_life_days == span
        assert result.ratio_max_min == pytest.approx(2.0, rel=1e-9)
        assert result.weight_share_newest_decile > 0.1

    def test_the_kish_size_is_bounded_by_the_rows_and_falls_with_the_half_life(
        self, complete_panel
    ):
        span = _panel_span_days(complete_panel)
        sizes = []
        for divisor in (1.0, 4.0, 16.0):
            result = preview_sample_weights(
                PreviewSampleWeightsInput(
                    dataset_id=complete_panel,
                    weighting=WeightingSpec(
                        method="time_decay", half_life_days=span / divisor
                    ),
                )
            )
            assert result.effective_sample_size_kish <= result.n_rows
            sizes.append(result.effective_sample_size_kish)
        assert sizes[0] > sizes[1] > sizes[2]

    def test_a_wide_spread_and_a_short_half_life_are_both_named(self, complete_panel):
        span = _panel_span_days(complete_panel)
        wide = preview_sample_weights(
            PreviewSampleWeightsInput(
                dataset_id=complete_panel,
                weighting=WeightingSpec(method="time_decay", half_life_days=span / 4.0),
            )
        )
        assert wide.ratio_max_min > 10
        assert any("times the lightest" in w for w in wide.warnings)
        short = preview_sample_weights(
            PreviewSampleWeightsInput(
                dataset_id=complete_panel,
                weighting=WeightingSpec(
                    method="time_decay", half_life_days=span / 20.0
                ),
            )
        )
        assert any("half_life_days" in w for w in short.warnings)

    def test_the_composite_is_not_its_two_halves_multiplied(self, complete_panel):
        span = _panel_span_days(complete_panel)

        def _summary(method):
            return preview_sample_weights(
                PreviewSampleWeightsInput(
                    dataset_id=complete_panel,
                    weighting=WeightingSpec(method=method, half_life_days=span),
                )
            )

        uniqueness = _summary("label_uniqueness")
        decay = _summary("time_decay")
        both = _summary("uniqueness_and_time_decay")
        # The weights compose; their SUMMARIES do not. max/min of a
        # product is the product of max/min only when the extremes fall on
        # the same rows, and here the oldest row is not the most unique.
        assert both.ratio_max_min != pytest.approx(
            uniqueness.ratio_max_min * decay.ratio_max_min, rel=1e-3
        )
        assert both.ratio_max_min != pytest.approx(uniqueness.ratio_max_min, rel=1e-3)
        assert both.ratio_max_min != pytest.approx(decay.ratio_max_min, rel=1e-3)
        assert both.effective_sample_size_kish < min(
            uniqueness.effective_sample_size_kish, decay.effective_sample_size_kish
        )
        # The overlap-based count is a property of the LABEL, so it is the
        # same number under every weighting -- which is why it is reported
        # beside the Kish size rather than instead of it.
        assert (
            both.effective_sample_size
            == uniqueness.effective_sample_size
            == decay.effective_sample_size
        )

    def test_method_none_is_described_as_flat_weights_and_not_refused(
        self, complete_panel
    ):
        result = preview_sample_weights(
            PreviewSampleWeightsInput(
                dataset_id=complete_panel, weighting=WeightingSpec(method="none")
            )
        )
        assert result.min == result.max == result.median == pytest.approx(1.0)
        assert result.ratio_max_min == pytest.approx(1.0)
        assert result.std == pytest.approx(0.0, abs=1e-12)
        assert result.effective_sample_size_kish == pytest.approx(result.n_rows)
        assert result.half_life_days is None
        assert any("every row enters the fit at weight 1" in w for w in result.warnings)

    def test_uniqueness_without_label_end_dates_gets_the_library_refusal(
        self, monkeypatch
    ):
        """A panel built before label end dates were recorded carries no
        column for the purge or the uniqueness to read, and the remedy is
        the library's own: rebuild, or weight by time alone."""
        panel = pd.DataFrame(
            {
                "date": np.repeat(pd.date_range("2022-01-03", periods=20), 2),
                "entity": np.tile(["AAA", "BBB"], 20),
                "technical.rsi": np.linspace(0.0, 1.0, 40),
                "target": np.linspace(-0.1, 0.1, 40),
            }
        )
        meta = {
            "feature_ids": ["technical.rsi"],
            "target_id": "forward_return:5",
            "data_hash": "unused",
        }
        monkeypatch.setattr(
            modeling_tools,
            "_load_dataset_panel",
            lambda dataset_id: (panel, meta, None),
        )
        with pytest.raises(ValidationError, match="label end date"):
            preview_sample_weights(
                PreviewSampleWeightsInput(
                    dataset_id="ds_no_label_end",
                    weighting=WeightingSpec(method="label_uniqueness"),
                )
            )
        # And the method that needs only the dates still answers on it.
        result = preview_sample_weights(
            PreviewSampleWeightsInput(
                dataset_id="ds_no_label_end",
                weighting=WeightingSpec(method="time_decay", half_life_days=30.0),
            )
        )
        assert result.n_rows == 40 and result.mean == pytest.approx(1.0)


# ── preview_preprocessing ───────────────────────────────────────────────


class TestThePipelineIsFittedBeforeTheExperimentRunsIt:
    def test_pca_refuses_a_panel_narrower_than_its_component_count(
        self, complete_panel
    ):
        with pytest.raises(ValidationError, match="n_components=8 exceeds the 3"):
            preview_preprocessing(
                PreviewPreprocessingInput(
                    dataset_id=complete_panel,
                    preprocessing=PreprocessingSpec(
                        steps=[StepSpec(type="pca_whiten")]
                    ),
                )
            )

    def test_pca_refuses_a_panel_with_holes(self, panel_with_holes):
        with pytest.raises(ValidationError, match="cannot fit on missing values"):
            preview_preprocessing(
                PreviewPreprocessingInput(
                    dataset_id=panel_with_holes,
                    preprocessing=PreprocessingSpec(
                        steps=[StepSpec(type="pca_whiten", params={"n_components": 2})]
                    ),
                )
            )

    def test_the_indicator_step_doubles_the_width_and_says_so(self, complete_panel):
        result = preview_preprocessing(
            PreviewPreprocessingInput(
                dataset_id=complete_panel,
                preprocessing=PreprocessingSpec(
                    steps=[StepSpec(type="missing_indicator")]
                ),
            )
        )
        assert result.n_columns_in == 3
        assert result.n_columns_out == 6 == 2 * result.n_columns_in
        assert result.steps[0].columns_added == [f"{f}__missing" for f in FEATURES]
        assert result.steps[0].stateless is True
        assert any("doubled the width from 3 to 6" in w for w in result.warnings)

    def test_a_winsorize_and_zscore_pipeline_leaves_the_width_alone(
        self, complete_panel
    ):
        result = preview_preprocessing(
            PreviewPreprocessingInput(
                dataset_id=complete_panel,
                preprocessing=PreprocessingSpec(
                    steps=[
                        StepSpec(
                            type="winsorize", params={"lower": 0.05, "upper": 0.95}
                        ),
                        StepSpec(type="zscore"),
                    ]
                ),
            )
        )
        assert result.n_columns_in == result.n_columns_out == 3
        assert result.output_columns == FEATURES
        assert [step.type for step in result.steps] == ["winsorize", "zscore"]
        assert all(step.column_wise for step in result.steps)
        assert all(
            step.columns_added == [] and step.columns_removed == []
            for step in result.steps
        )
        assert result.explained_variance_ratio is None
        assert result.n_nan_after == 0
        assert result.warnings == []
        # The transform did something: the output is centred and scaled
        # where the input was neither.
        after = result.per_column_after[FEATURES[0]]
        assert abs(after.mean) < 0.5 and 0.5 < after.std < 1.5

    def test_the_default_pair_is_the_fused_state_the_engine_persists(
        self, complete_panel
    ):
        # The engine's own shape, on the frame the native-kernel suite
        # fits: version, columns, and one entry per step carrying its type,
        # its resolved parameters and its statistics.
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            rng.normal(0, 1, (100, 3)), columns=[f"f{i}" for i in range(3)]
        )
        ctx = FoldContext(dates=np.repeat(pd.date_range("2022-01-03", periods=50), 2))
        train, test = frame.iloc[:70], frame.iloc[70:]
        state, _train_out, _test_out = fit_and_apply_pipeline(
            PreprocessingSpec().resolved_steps, train, test, ctx, ctx
        )
        assert state["version"] == STATE_VERSION
        assert step_types(state) == ["winsorize", "zscore"]
        assert set(state["steps"][0]["state"]) == {"lo", "hi"}
        assert set(state["steps"][1]["state"]) == {"mean", "std"}
        oracle = fit_preprocessing(train)
        for column in train.columns:
            assert state["steps"][0]["state"]["lo"][column] == pytest.approx(
                oracle[column]["lo"], abs=1e-12
            )
            assert state["steps"][1]["state"]["std"][column] == pytest.approx(
                oracle[column]["std"], abs=1e-12
            )

        # And the preview of the DEFAULT spec reports that state, step for
        # step and parameter for parameter, on a real dataset.
        result = preview_preprocessing(
            PreviewPreprocessingInput(
                dataset_id=complete_panel, preprocessing=PreprocessingSpec()
            )
        )
        assert [step.type for step in result.steps] == step_types(state)
        assert [step.params for step in result.steps] == [
            entry["params"] for entry in state["steps"]
        ]
        assert result.n_columns_in == result.n_columns_out == 3

    def test_the_sample_is_split_by_date_and_never_by_row(self, complete_panel):
        panel, _meta, _directory = _load_dataset_panel(complete_panel)
        result = preview_preprocessing(
            PreviewPreprocessingInput(
                dataset_id=complete_panel,
                preprocessing=PreprocessingSpec(),
                split_fraction=0.7,
            )
        )
        assert result.n_train_rows + result.n_test_rows == result.n_rows_sampled
        # No date is on both sides: the first date the state is applied to
        # is strictly later than the last date it was fitted on. A row
        # split would put the same cross-section in both halves.
        assert pd.Timestamp(result.test_start) > pd.Timestamp(result.train_end)
        train_dates = panel.loc[
            panel["date"] <= pd.Timestamp(result.train_end), "date"
        ].nunique()
        assert train_dates == round(int(panel["date"].nunique()) * 0.7)

        later = preview_preprocessing(
            PreviewPreprocessingInput(
                dataset_id=complete_panel,
                preprocessing=PreprocessingSpec(),
                split_fraction=0.9,
            )
        )
        assert pd.Timestamp(later.train_end) > pd.Timestamp(result.train_end)
        assert later.n_train_rows > result.n_train_rows

    def test_the_explained_variance_is_reported_only_where_there_is_a_pca(
        self, complete_panel
    ):
        pca = preview_preprocessing(
            PreviewPreprocessingInput(
                dataset_id=complete_panel,
                preprocessing=PreprocessingSpec(
                    steps=[StepSpec(type="pca_whiten", params={"n_components": 2})]
                ),
            )
        )
        assert pca.explained_variance_ratio is not None
        assert len(pca.explained_variance_ratio) == 2
        assert 0.0 < sum(pca.explained_variance_ratio) <= 1.0 + 1e-9
        assert pca.n_columns_in == 3 and pca.n_columns_out == 2
        assert pca.output_columns == ["pc1", "pc2"]
        assert pca.steps[0].column_wise is False
        assert pca.steps[0].columns_removed == FEATURES
        assert any("run_feature_ablation" in w for w in pca.warnings)
        assert any("no longer feature names" in w for w in pca.warnings)

        plain = preview_preprocessing(
            PreviewPreprocessingInput(
                dataset_id=complete_panel, preprocessing=PreprocessingSpec()
            )
        )
        assert plain.explained_variance_ratio is None

    def test_holes_the_engine_would_refuse_are_counted_here_instead(
        self, panel_with_holes
    ):
        result = preview_preprocessing(
            PreviewPreprocessingInput(
                dataset_id=panel_with_holes,
                preprocessing=PreprocessingSpec(
                    steps=[StepSpec(type="missing_indicator")]
                ),
            )
        )
        assert result.n_nan_after > 0
        assert any("missing value(s) remain" in w for w in result.warnings)
        # The indicator that would carry the information through an impute
        # step is there, and it is not all zero on this panel.
        assert result.per_column_after[f"{FEATURES[0]}__missing"].max == 1.0
