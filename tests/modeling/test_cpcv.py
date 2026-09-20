"""
Combinatorial purged cross-validation: every choice of `n_test_splits`
blocks out of `n_splits` is a test set, so the out-of-sample number has a
distribution rather than a single draw.

Two things are planted. The splitter's combinatorics -- fifteen paths from
six groups choose two, every date tested exactly five times, an embargo
that removes exactly the dates within `embargo` of each block on both
sides. And the engine's purge under a test set made of two blocks: a
training row between the blocks is purged only when its own label reaches
the LATER block, never for lying between them, which a purge on
[first test date, last test date] would have done to every row in the gap.
The expected count is recomputed in the test by a row-wise rule written
independently of the engine's mask.
"""

import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.models import (
    BuildModelDatasetInput,
    EvaluateModelPortfolioInput,
    RunModelExperimentInput,
    ValidateModelSpecInput,
)
from standard_quant_tools.modeling.agent.tools import (
    build_model_dataset,
    evaluate_model_portfolio,
    run_model_experiment,
    validate_model_spec,
)
from standard_quant_tools.modeling.bridge import oos_predictions_to_signal_panel
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.ensemble import load_oos_predictions
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.walk_forward import (
    CombinatorialPurgedSplit,
    build_splitter,
)

UNIVERSE = ["AAA", "BBB", "CCC"]
HORIZON = 5


def _dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=TargetSpec(horizon=HORIZON),
        benchmark="SPY",
    )


def _cpcv(n_splits=6, n_test_splits=2, embargo=0, **kw) -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(
            method="cpcv",
            n_splits=n_splits,
            n_test_splits=n_test_splits,
            embargo=embargo,
            **kw,
        ),
        random_seed=1,
    )


class TestTheSplitter:
    def test_six_choose_two_is_fifteen_paths_and_every_date_is_tested_five_times(self):
        dates = pd.RangeIndex(120)
        splitter = CombinatorialPurgedSplit(n_groups=6, n_test_groups=2)
        folds = list(splitter.split(dates))
        assert len(folds) == 15 == splitter.n_splits(dates) == splitter.n_paths
        tested = np.zeros(120, dtype=int)
        for train_pos, test_pos in folds:
            tested[test_pos] += 1
            assert not set(train_pos) & set(test_pos)
            assert np.all(np.diff(test_pos) >= 1)
        # Each date's group sits in C(5, 1) = 5 of the C(6, 2) combinations.
        assert np.all(tested == 5)

    def test_the_embargo_removes_exactly_the_band_around_each_block(self):
        dates = pd.RangeIndex(120)
        folds = list(CombinatorialPurgedSplit(6, 2, embargo=3).split(dates))
        groups = np.array_split(np.arange(120), 6)
        for (train_pos, test_pos), combo in zip(folds, combinations(range(6), 2)):
            banned = set()
            for g in combo:
                start, end = groups[g][0], groups[g][-1] + 1
                banned |= set(range(max(0, start - 3), min(120, end + 3)))
            assert not set(train_pos) & banned
            assert set(train_pos) == set(range(120)) - banned

    def test_bad_shapes_are_refused(self):
        with pytest.raises(ValidationError):
            CombinatorialPurgedSplit(n_groups=3, n_test_groups=3)
        with pytest.raises(ValidationError):
            CombinatorialPurgedSplit(n_groups=1, n_test_groups=1)
        with pytest.raises(ValidationError):
            CombinatorialPurgedSplit(n_groups=6, n_test_groups=2, embargo=-1)

    def test_the_spec_builds_it_and_bounds_the_path_count(self):
        spec = ValidationSpec(method="cpcv", n_splits=6, n_test_splits=2)
        assert isinstance(build_splitter(spec), CombinatorialPurgedSplit)
        with pytest.raises(PydanticValidationError, match="paths"):
            ValidationSpec(method="cpcv", n_splits=14, n_test_splits=7)
        with pytest.raises(PydanticValidationError, match="fewer than"):
            ValidationSpec(method="cpcv", n_splits=4, n_test_splits=4)
        with pytest.raises(PydanticValidationError, match="cpcv"):
            ValidationSpec(
                method="walk_forward", train_window=10, test_window=5, n_test_splits=3
            )


def _expected_purge(panel: pd.DataFrame, dates: pd.Index, folds) -> int:
    """Row-wise, independently of the engine: a training row is purged
    when its label reaches into ANY test block it precedes."""
    purged = 0
    date_pos = {d: i for i, d in enumerate(dates)}
    for train_pos, test_pos in folds:
        train_set = set(train_pos.tolist())
        # Contiguous runs of test positions.
        runs, start = [], test_pos[0]
        for prev, cur in zip(test_pos[:-1], test_pos[1:]):
            if cur != prev + 1:
                runs.append((start, prev))
                start = cur
        runs.append((start, test_pos[-1]))
        blocks = [(dates[a], dates[b]) for a, b in runs]
        for row_date, label_end in zip(panel["date"], panel["label_end_date"]):
            if date_pos[row_date] not in train_set:
                continue
            if any(label_end >= first and row_date <= last for first, last in blocks):
                purged += 1
    return purged


class TestTheEngineUnderCpcv:
    @pytest.fixture
    def dataset(self, patched_multi_factory):
        return build_dataset(_dataset_spec())

    def test_the_purge_is_per_block_not_per_span(self, dataset):
        spec = _cpcv(n_splits=4, n_test_splits=2, embargo=0)
        result = run_experiment(dataset, spec, "ds", register=False)
        dates = pd.Index(sorted(dataset["panel"]["date"].unique()))
        folds = list(build_splitter(spec.validation).split(dates))
        expected = _expected_purge(dataset["panel"], dates, folds)
        assert result["n_train_rows_purged_overlap"] == expected
        # And the span rule would have purged far more: every training row
        # lying between two test blocks, not only the five before each.
        span = 0
        for train_pos, test_pos in folds:
            first, last = dates[test_pos[0]], dates[test_pos[-1]]
            in_train = dataset["panel"]["date"].isin(dates[train_pos])
            span += int(
                (
                    in_train
                    & (dataset["panel"]["label_end_date"] >= first)
                    & (dataset["panel"]["date"] <= last)
                ).sum()
            )
        assert span > expected

    def test_every_row_is_predicted_once_per_path_it_was_tested_in(self, dataset):
        result = run_experiment(dataset, _cpcv(), "ds")
        frame = load_oos_predictions(result["model_id"], keep_path=True)
        assert "path" in frame.columns
        counts = frame.groupby(["date", "entity"]).size()
        assert counts.min() == counts.max() == 5  # C(5, 1) of C(6, 2)
        assert result["n_folds"] == 15
        report = result["validation_report"]
        assert report["method"] == "cpcv"
        assert report["paths"]["n_paths"] == 15
        distribution = report["paths"]["metric_distribution"]["cs_rank_ic_mean"]
        assert distribution["p05"] <= distribution["p50"] <= distribution["p95"]
        assert distribution["n"] == 15
        # The effective sample counts each row once, not once per path.
        assert result["oos_metrics"]["n_oos_rows"] == float(len(counts))

    def test_the_pooled_ic_averages_each_date_across_paths_first(self, dataset):
        result = run_experiment(dataset, _cpcv(), "ds", register=False)
        assert (
            result["validation_report"]["paths"]["ic_pooling"]
            == "mean per date across paths"
        )
        assert result["oos_metrics"]["cs_rank_ic_n_dates"] == float(
            dataset["panel"]["date"].nunique()
        )


class TestWhatACpcvModelIsNotFor:
    @pytest.fixture
    def cpcv_model(self, patched_multi_factory):
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_dataset_spec())
        ).dataset_id
        return run_model_experiment(
            RunModelExperimentInput(dataset_id=dataset_id, spec=_cpcv())
        ).model_id

    def test_the_portfolio_evaluation_refuses_it(self, cpcv_model):
        with pytest.raises(ValidationError, match="walk_forward"):
            evaluate_model_portfolio(EvaluateModelPortfolioInput(model_id=cpcv_model))

    def test_the_bridge_refuses_it(self, cpcv_model):
        with pytest.raises(ValidationError, match="walk_forward"):
            oos_predictions_to_signal_panel(model_id=cpcv_model)

    def test_the_manifest_records_the_method(self, cpcv_model):
        assert load_manifest(cpcv_model).validation_method == "cpcv"


class TestTheFitEstimate:
    def test_the_path_count_is_known_without_a_dataset(self):
        result = validate_model_spec(ValidateModelSpecInput(spec=_cpcv()))
        assert result.estimated_folds == math.comb(6, 2) == 15
        assert result.estimated_fits == 15

    def test_and_exact_with_one(self, patched_multi_factory):
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_dataset_spec())
        ).dataset_id
        result = validate_model_spec(
            ValidateModelSpecInput(spec=_cpcv(), dataset_id=dataset_id)
        )
        assert result.estimated_folds == 15
