"""
The experiment plan: what run_experiment will do, counted before it does
it, and refused before the first fit when it costs more than the spec
allows.

Planted in both directions. The fit count is checked against the
splitter's own fold count and the search's own candidate list, and against
what an experiment then reports it ran; the purge the plan records per
fold is checked against a row-wise rule written independently of the
engine's mask; the node hashes move with exactly the things that determine
a fit and with nothing else; and an over-budget spec is refused with a spy
on the fit that proves nothing was fitted.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import engine
from standard_quant_tools.modeling.agent.models import (
    BuildModelDatasetInput,
    ValidateModelSpecInput,
)
from standard_quant_tools.modeling.agent.tools import (
    _load_dataset_meta,
    build_model_dataset,
    validate_model_spec,
)
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.plan import (
    fit_count,
    fits_per_estimator,
    plan_experiment,
)
from standard_quant_tools.modeling.specs import (
    ComputeBudgetSpec,
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    PreprocessingSpec,
    SearchSpec,
    TargetSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.search import search_candidates
from standard_quant_tools.modeling.validation.walk_forward import build_splitter

UNIVERSE = ["AAA", "BBB", "CCC"]
HORIZON = 5


def _dataset_spec(target: TargetSpec | None = None) -> DatasetSpec:
    return DatasetSpec(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=target or TargetSpec(horizon=HORIZON),
        benchmark="SPY",
    )


def _walk_forward(train_window=40, test_window=10, embargo=0, **kw) -> ValidationSpec:
    return ValidationSpec(
        method="walk_forward",
        train_window=train_window,
        test_window=test_window,
        embargo=embargo,
        min_folds=1,
        **kw,
    )


def _ridge(*, validation=None, search=None, budget=None, **kw) -> ModelSpec:
    fields = dict(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=validation or _walk_forward(),
        random_seed=1,
    )
    if search is not None:
        fields["search"] = search
    if budget is not None:
        fields["budget"] = budget
    fields.update(kw)
    return ModelSpec(**fields)


def _grid(n_alphas: int, inner_splits: int = 3, **kw) -> SearchSpec:
    return SearchSpec(
        method="grid",
        param_grid={"alpha": [float(i + 1) for i in range(n_alphas)]},
        inner_splits=inner_splits,
        **kw,
    )


class TestTheScheduleWithoutAPanel:
    dates = pd.bdate_range("2022-01-03", periods=120)

    def test_the_folds_are_the_splitter_s_and_each_costs_one_fit_plus_the_refit(self):
        spec = _ridge(validation=_walk_forward(40, 10, embargo=2))
        plan = plan_experiment(spec, self.dates)
        folds = list(build_splitter(spec.validation).split(self.dates))
        assert len(plan.folds) == len(folds) > 1
        for fold, (train_pos, test_pos) in zip(plan.folds, folds):
            assert fold.train_start == str(self.dates[train_pos[0]].date())
            assert fold.train_end == str(self.dates[train_pos[-1]].date())
            assert fold.test_start == str(self.dates[test_pos[0]].date())
            assert fold.test_end == str(self.dates[test_pos[-1]].date())
            assert fold.n_train_dates == len(train_pos)
            assert fold.n_inner_folds == 0 and fold.n_candidates == 0
            assert fold.n_fits == 1
            assert fold.purged_rows is None and fold.n_purged is None
        assert plan.n_fits == len(folds) + 1
        assert plan.n_fits_refit == 1
        assert not plan.has_panel and plan.n_purged is None
        assert plan.within_budget and plan.max_fits == 500

    def test_a_search_multiplies_only_through_folds_whose_window_can_search(self):
        # Three inner splits need at least four usable dates in the
        # training window: a 3-date window searches nothing and costs one
        # fit; a 40-date window scores every candidate on three folds.
        search = _grid(4, inner_splits=3)
        short = plan_experiment(
            _ridge(validation=_walk_forward(3, 2), search=search), self.dates
        )
        long = plan_experiment(
            _ridge(validation=_walk_forward(40, 10), search=search), self.dates
        )
        assert all(f.n_inner_folds == 0 and f.n_fits == 1 for f in short.folds)
        assert short.n_fits == len(short.folds) + 1
        assert all(f.n_inner_folds == 3 and f.n_fits == 1 + 4 * 3 for f in long.folds)
        assert long.n_fits == 13 * len(long.folds) + 1
        assert long.n_candidates == 4
        # The spec-only count assumes every fold searches: exact where
        # that is true, an upper bound where it is not.
        long_spec = _ridge(validation=_walk_forward(40, 10), search=search)
        short_spec = _ridge(validation=_walk_forward(3, 2), search=search)
        assert fit_count(long_spec, len(long.folds)) == long.n_fits
        assert fit_count(short_spec, len(short.folds)) > short.n_fits

    def test_a_random_search_counts_the_candidates_it_will_sample(self):
        search = SearchSpec(
            method="random",
            param_grid={"alpha": [0.1, 1.0, 10.0], "fit_intercept": [True, False]},
            n_iter=5,
            inner_splits=2,
        )
        spec = _ridge(validation=_walk_forward(40, 10), search=search)
        plan = plan_experiment(spec, self.dates)
        assert (
            plan.n_candidates == 5 == len(search_candidates(search, spec.random_seed))
        )
        assert plan.n_fits == (1 + 5 * 2) * len(plan.folds) + 1

    def test_calibration_multiplies_the_outer_fit_and_the_refit_but_not_the_search(
        self,
    ):
        spec = ModelSpec(
            task="classification",
            estimator=EstimatorSpec(
                type="logistic", calibration="isotonic", calibration_folds=4
            ),
            validation=_walk_forward(40, 10),
            search=SearchSpec(param_grid={"C": [0.1, 1.0]}, inner_splits=2),
        )
        assert fits_per_estimator(spec) == 4
        plan = plan_experiment(spec, self.dates)
        assert all(f.n_fits == 4 + 2 * 2 for f in plan.folds)
        assert plan.n_fits_refit == 4
        assert plan.n_fits == 8 * len(plan.folds) + 4

    def test_no_fold_is_a_refusal_not_an_empty_plan(self):
        with pytest.raises(ValidationError, match="no fold"):
            plan_experiment(_ridge(validation=_walk_forward(200, 50)), self.dates)


def _oracle_purge(panel: pd.DataFrame, dates: pd.Index, train_pos, test_pos) -> int:
    """Row-wise and independent of the engine: a training row is purged
    when its label reaches into a test block it does not follow."""
    in_train = panel["date"].isin(dates[train_pos]).to_numpy()
    runs, start = [], test_pos[0]
    for prev, cur in zip(test_pos[:-1], test_pos[1:]):
        if cur != prev + 1:
            runs.append((start, prev))
            start = cur
    runs.append((start, test_pos[-1]))
    label_end = panel["label_end_date"].to_numpy()
    row_date = panel["date"].to_numpy()
    purged = np.zeros(len(panel), dtype=bool)
    for a, b in runs:
        first, last = dates[a], dates[b]
        purged |= in_train & (label_end >= first) & (row_date <= last)
    return int(purged.sum())


class TestThePlanWithAPanel:
    @pytest.fixture
    def dataset(self, patched_multi_factory):
        return build_dataset(_dataset_spec())

    def test_the_purge_the_plan_records_is_the_purge_that_runs(self, dataset):
        spec = _ridge(validation=ValidationSpec(method="purged_kfold", n_splits=4))
        dates = pd.Index(sorted(dataset["panel"]["date"].unique()))
        plan = plan_experiment(
            spec,
            dates,
            panel=dataset["panel"],
            dataset_hash=dataset["data_hash"],
            feature_ids=dataset["feature_ids"],
        )
        assert plan.has_panel and plan.n_purged is not None and plan.n_purged > 0
        for fold, (train_pos, test_pos) in zip(
            plan.folds, build_splitter(spec.validation).split(dates)
        ):
            assert fold.n_purged == _oracle_purge(
                dataset["panel"], dates, train_pos, test_pos
            )
            assert fold.n_purged == len(fold.purged_rows)
        result = run_experiment(dataset, spec, "ds", register=False)
        report = result["validation_report"]
        assert result["n_train_rows_purged_overlap"] == plan.n_purged
        assert report["fits"]["planned"] == plan.n_fits == len(plan.folds) + 1
        assert report["fits"]["max_fits"] == 500
        for fold, record in zip(plan.folds, report["folds"]):
            assert record["n_train_rows"] == fold.n_train_rows
            assert record["n_test_rows"] == fold.n_test_rows
            assert record["node_hash"] == fold.node_hash
            assert record["test_start"] == fold.test_start
            assert record["test_end"] == fold.test_end

    def test_inner_folds_are_counted_on_the_dates_that_survive_the_purge(self, dataset):
        # A training window of 8 dates supports three inner splits on the
        # schedule (four usable dates is the floor), but the purge removes
        # the last HORIZON dates before the test block, and on the three
        # that are left the search cannot run.
        spec = _ridge(validation=_walk_forward(8, 5), search=_grid(2, inner_splits=3))
        dates = pd.Index(sorted(dataset["panel"]["date"].unique()))
        scheduled = plan_experiment(spec, dates)
        actual = plan_experiment(spec, dates, panel=dataset["panel"])
        assert all(f.n_inner_folds == 3 for f in scheduled.folds)
        assert all(f.n_train_dates == 8 - HORIZON for f in actual.folds)
        assert all(f.n_inner_folds == 0 for f in actual.folds)
        assert actual.n_fits < scheduled.n_fits
        result = run_experiment(dataset, spec, "ds", register=False)
        assert result["validation_report"]["fits"]["planned"] == actual.n_fits
        assert not any(
            r["searched"] for r in result["validation_report"]["hyperparameter_search"]
        )


class TestTheNodeHashes:
    dates = pd.bdate_range("2022-01-03", periods=120)

    def _hashes(self, spec, **kw):
        plan = plan_experiment(
            spec, self.dates, dataset_hash=kw.pop("dataset_hash", "d1"), **kw
        )
        return (
            [f.node_hash for f in plan.folds],
            [f.preprocessing_hash for f in plan.folds],
        )

    def test_the_hash_moves_with_what_determines_the_fit_and_with_nothing_else(self):
        base = _ridge()
        node, prep = self._hashes(base, feature_ids=["a", "b"])
        assert len(set(node)) == len(node) and len(set(prep)) == len(prep)
        assert self._hashes(base, feature_ids=["a", "b"]) == (node, prep)

        # The estimator, its parameters and the seed: the fit, not the matrices.
        other_params = _ridge(
            estimator=EstimatorSpec(type="ridge", params={"alpha": 2.0})
        )
        node2, prep2 = self._hashes(other_params, feature_ids=["a", "b"])
        assert node2 != node and prep2 == prep
        node3, prep3 = self._hashes(_ridge(random_seed=2), feature_ids=["a", "b"])
        assert node3 != node and prep3 == prep
        node4, prep4 = self._hashes(
            _ridge(estimator=EstimatorSpec(type="lasso", params={"alpha": 1.0})),
            feature_ids=["a", "b"],
        )
        assert node4 != node and prep4 == prep

        # The feature set: the fit, and NOT the preprocessing key -- a
        # column-wise pipeline's matrices for a subset are a projection of
        # the full set's, which is what the fold cache relies on.
        node5, prep5 = self._hashes(base, feature_ids=["a"])
        assert node5 != node and prep5 == prep

        # The pipeline and the dataset: both.
        node6, prep6 = self._hashes(
            _ridge(preprocessing=PreprocessingSpec(normalization="cross_sectional")),
            feature_ids=["a", "b"],
        )
        assert node6 != node and prep6 != prep
        node7, prep7 = self._hashes(base, feature_ids=["a", "b"], dataset_hash="d2")
        assert node7 != node and prep7 != prep


class TestTheBudget:
    @pytest.fixture
    def dataset(self, patched_multi_factory):
        return build_dataset(_dataset_spec())

    def test_the_default_ceiling_and_its_bounds(self):
        assert _ridge().budget.max_fits == 500
        with pytest.raises(PydanticValidationError):
            ComputeBudgetSpec(max_fits=0)
        with pytest.raises(PydanticValidationError):
            ComputeBudgetSpec(max_fits=100_001)
        with pytest.raises(PydanticValidationError):
            ComputeBudgetSpec(max_fit=10)

    def test_an_over_budget_spec_is_refused_before_the_first_fit(
        self, dataset, monkeypatch
    ):
        spec = _ridge(
            validation=_walk_forward(40, 10),
            search=_grid(3, inner_splits=2),
            budget=ComputeBudgetSpec(max_fits=3),
        )

        def _never(*_args, **_kwargs):
            pytest.fail("an over-budget experiment fitted something")

        monkeypatch.setattr(engine, "_fit", _never)
        with pytest.raises(ValidationError, match=r"budget\.max_fits=3") as info:
            run_experiment(dataset, spec, "ds", register=False)
        dates = pd.Index(sorted(dataset["panel"]["date"].unique()))
        planned = plan_experiment(spec, dates, panel=dataset["panel"]).n_fits
        assert f"{planned:,} estimator fits" in str(info.value)
        assert f"budget.max_fits={planned}" in str(info.value)

    def test_raising_the_ceiling_to_the_count_runs_it(self, dataset):
        spec = _ridge(validation=_walk_forward(40, 10), search=_grid(3, inner_splits=2))
        dates = pd.Index(sorted(dataset["panel"]["date"].unique()))
        planned = plan_experiment(spec, dates, panel=dataset["panel"]).n_fits
        assert planned > 3
        exact = spec.model_copy(update={"budget": ComputeBudgetSpec(max_fits=planned)})
        result = run_experiment(dataset, exact, "ds", register=False)
        assert result["validation_report"]["fits"] == {
            "planned": planned,
            "folds": planned - 1,
            "refit": 1,
            "candidates_per_fold": 3,
            "max_fits": planned,
            "max_parallelism": 1,
        }

    def test_validate_model_spec_reports_the_plan_and_the_ceiling(
        self, patched_multi_factory
    ):
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_dataset_spec())
        ).dataset_id
        tight = _ridge(
            validation=_walk_forward(40, 10),
            search=_grid(3, inner_splits=2),
            budget=ComputeBudgetSpec(max_fits=3),
        )
        result = validate_model_spec(
            ValidateModelSpecInput(spec=tight, dataset_id=dataset_id)
        )
        assert not result.valid
        assert [p.where for p in result.problems] == ["budget"]
        assert result.max_fits == 3 and result.within_budget is False
        # The tool plans over the dataset's date COUNT, which its metadata
        # records, so the panel is never loaded to answer this.
        meta, _directory = _load_dataset_meta(dataset_id)
        assert (
            result.estimated_fits
            == plan_experiment(tight, pd.RangeIndex(int(meta["n_dates"]))).n_fits
        )
        assert f"budget.max_fits={result.estimated_fits}" in (
            result.problems[0].suggestion or ""
        )
        loose = tight.model_copy(update={"budget": ComputeBudgetSpec()})
        result = validate_model_spec(
            ValidateModelSpecInput(spec=loose, dataset_id=dataset_id)
        )
        assert result.valid and result.within_budget is True
        assert result.max_fits == 500

    def test_without_a_dataset_the_estimate_is_the_spec_only_count(self):
        spec = _ridge(
            validation=ValidationSpec(method="purged_kfold", n_splits=7),
            search=_grid(3, inner_splits=2),
        )
        result = validate_model_spec(ValidateModelSpecInput(spec=spec))
        assert result.estimated_folds == 7
        assert result.estimated_fits == fit_count(spec, 7) == 7 * (1 + 3 * 2) + 1
        assert result.within_budget is True
        # Walk-forward without a date axis: unknown, and the budget with it.
        result = validate_model_spec(ValidateModelSpecInput(spec=_ridge()))
        assert result.estimated_fits is None and result.within_budget is None
        assert result.max_fits == 500
