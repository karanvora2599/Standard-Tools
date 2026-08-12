"""
Regression tests for the P1 validation/financial-statistics findings.

The headline numbers a user reads were distorted in ways that make a model
look better (or merely different) than it is: pooled IC conflating
cross-sectional skill with market timing, equal-weighted fold averaging,
a raw row count that ignores label overlap, and no baseline to compare
against. These tests pin the corrected behavior and, where possible,
demonstrate the distortion directly.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.metrics import (
    average_fold_metrics,
    baseline_regression_metrics,
    cross_sectional_ic,
    effective_sample_size,
    regression_metrics,
    summarize_cross_sectional_ic,
)


class TestCrossSectionalIC:
    def test_pooled_ic_can_be_strong_with_zero_cross_sectional_skill(self):
        """
        The distortion being fixed, constructed explicitly.

        Every entity on a given date gets the SAME prediction (the market
        move) and realized returns are that market move plus pure noise.
        The model has no ability to rank names against each other — the
        cross-sectional IC is undefined/zero by construction — but pooling
        all (entity, date) rows into one correlation shows a large IC
        purely because both series move with the market.
        """
        rng = np.random.default_rng(0)
        n_dates, n_entities = 120, 8
        dates, y, p = [], [], []
        for d in range(n_dates):
            market = rng.normal(0, 0.02)
            for _ in range(n_entities):
                dates.append(d)
                p.append(market)  # identical across the cross-section
                y.append(market + rng.normal(0, 0.001))
        dates = np.array(dates)
        y = np.array(y)
        p = np.array(p)

        pooled = float(pd.Series(y).corr(pd.Series(p), method="pearson"))
        cs = cross_sectional_ic(y, p, dates, method="pearson")

        assert pooled > 0.9, "pooled IC looks like a great model"
        # Within a date every prediction is identical -> no ranking
        # information -> correlation is degenerate, reported as 0.0.
        assert abs(float(cs.mean())) < 1e-9, "cross-sectionally there is no skill"

    def test_cross_sectional_ic_detects_real_ranking_skill(self):
        rng = np.random.default_rng(1)
        n_dates, n_entities = 150, 10
        dates, y, p = [], [], []
        for d in range(n_dates):
            signal = rng.normal(0, 1, n_entities)
            realized = signal * 0.01 + rng.normal(0, 0.002, n_entities)
            dates.extend([d] * n_entities)
            p.extend(signal)
            y.extend(realized)
        cs = cross_sectional_ic(np.array(y), np.array(p), np.array(dates))
        assert float(cs.mean()) > 0.7

    def test_single_entity_dates_are_dropped_not_zeroed(self):
        """A correlation over one point is undefined, not zero."""
        cs = cross_sectional_ic(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([0, 0, 1]),  # date 1 has a single entity
        )
        assert list(cs.index) == [0]

    def test_icir_distinguishes_consistent_from_lucky(self):
        """
        Two IC series with the SAME mean but very different reliability.
        A mean alone cannot tell them apart; ICIR can.
        """
        consistent = pd.Series([0.03, 0.031, 0.029, 0.030, 0.03])
        lucky = pd.Series([-0.10, 0.20, -0.12, 0.19, -0.02])
        assert np.isclose(consistent.mean(), lucky.mean(), atol=5e-3)

        c = summarize_cross_sectional_ic(consistent, "cs_ic")
        b = summarize_cross_sectional_ic(lucky, "cs_ic")
        assert c["cs_ic_icir"] > 10 * b["cs_ic_icir"]
        assert c["cs_ic_hit_rate"] == 1.0
        assert b["cs_ic_hit_rate"] < 0.5

    def test_empty_ic_series_is_nan_not_zero(self):
        out = summarize_cross_sectional_ic(pd.Series(dtype=float), "cs_ic")
        assert np.isnan(out["cs_ic_mean"])
        assert out["cs_ic_n_dates"] == 0.0


class TestFoldWeighting:
    def test_large_folds_dominate_the_average(self):
        """
        Equal weighting let a 30-prediction fold count as much as a
        3000-prediction one.
        """
        folds = [{"ic": 0.5}, {"ic": 0.0}]
        equal = average_fold_metrics(folds)
        weighted = average_fold_metrics(folds, [10.0, 990.0])
        assert np.isclose(equal["ic"], 0.25)
        assert weighted["ic"] < 0.02, "the tiny fold should barely matter"

    def test_nan_folds_are_ignored_not_propagated(self):
        folds = [{"auc": float("nan")}, {"auc": 0.6}]
        assert np.isclose(average_fold_metrics(folds, [100.0, 100.0])["auc"], 0.6)

    def test_all_nan_metric_stays_nan(self):
        folds = [{"auc": float("nan")}, {"auc": float("nan")}]
        assert np.isnan(average_fold_metrics(folds, [1.0, 1.0])["auc"])

    def test_date_counts_are_summed_not_averaged(self):
        folds = [{"cs_ic_n_dates": 30.0}, {"cs_ic_n_dates": 30.0}]
        assert average_fold_metrics(folds, [1.0, 1.0])["cs_ic_n_dates"] == 60.0


class TestEffectiveSampleSize:
    def test_overlap_discounts_the_row_count(self):
        """2000 daily rows of a 20-day forward return are not 2000
        independent observations."""
        assert effective_sample_size(2000, horizon=20, n_entities=1) == 100.0

    def test_scales_with_entities(self):
        """Overlap applies along time, not across entities."""
        assert effective_sample_size(2000, horizon=20, n_entities=4) == 100.0

    def test_horizon_one_is_unchanged(self):
        assert effective_sample_size(500, horizon=1) == 500.0

    def test_non_positive_horizon_falls_back_to_raw_count(self):
        assert effective_sample_size(500, horizon=0) == 500.0


class TestBaselineComparison:
    def test_mean_predictor_baseline_is_reported(self):
        y = np.array([0.01, -0.02, 0.03, 0.00])
        base = baseline_regression_metrics(y)
        assert base["baseline_r2"] == 0.0
        assert base["baseline_mae"] > 0

    def test_regression_metrics_include_the_baseline(self):
        y = np.array([0.01, -0.02, 0.03, 0.00])
        out = regression_metrics(y, y * 0.5)
        assert "baseline_mae" in out
        # A useful model should beat predicting the mean.
        assert out["mae"] < out["baseline_mae"]


def _spec(**overrides) -> DatasetSpec:
    base = dict(
        universe=["AAA", "BBB"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=TargetSpec(horizon=5),
    )
    base.update(overrides)
    return DatasetSpec(**base)


def _model_spec(**validation) -> ModelSpec:
    params = dict(train_window=150, test_window=30, embargo=5)
    params.update(validation)
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(**params),
        random_seed=1,
    )


def _dataset(spec: DatasetSpec) -> dict:
    built = build_dataset(spec)
    return {
        "panel": built["panel"],
        "feature_ids": built["feature_ids"],
        "target_id": built["target_id"],
        "data_hash": built["data_hash"],
        "spec_hash": built["spec_hash"],
        "dataset_spec": spec.model_dump(),
    }


class TestFoldAccounting:
    def test_validation_report_records_expected_and_completed(
        self, patched_multi_factory
    ):
        result = run_experiment(_dataset(_spec()), _model_spec(), dataset_id="ds_vr")
        report = result["validation_report"]
        assert report["n_folds_expected"] >= report["n_folds_completed"] >= 2
        assert report["n_folds_skipped"] == len(report["skipped_folds"])
        assert 0 < report["fold_coverage"] <= 1.0

    def test_per_fold_metrics_are_retained(self, patched_multi_factory):
        """Averages alone cannot show performance decay across time."""
        result = run_experiment(_dataset(_spec()), _model_spec(), dataset_id="ds_vr2")
        folds = result["validation_report"]["folds"]
        assert len(folds) == result["n_folds"]
        first = folds[0]
        assert {"train_start", "train_end", "test_start", "test_end", "metrics"} <= set(
            first
        )
        assert "r2" in first["metrics"]

    def test_train_end_is_the_range_actually_fit_not_the_scheduled_one(
        self, patched_multi_factory
    ):
        """
        `train_end` reported the SCHEDULED window end. Label-overlap
        purging removes the tail of the training window -- exactly
        `horizon` bars of it -- so a fold whose last week was entirely
        purged still claimed to have trained through it. Lineage described
        the split that was planned rather than the one that ran.

        Both are now recorded, and their difference is the purge extent.
        """
        # embargo=0 < horizon=5, so the last 5 training dates carry labels
        # that finish inside the test window and are purged. With the
        # default embargo=5 the gap already covers the horizon and nothing
        # is purged -- the bug is invisible exactly when it does not matter.
        result = run_experiment(
            _dataset(_spec()),
            _model_spec(embargo=0),
            dataset_id="ds_vr_te",
        )
        folds = result["validation_report"]["folds"]
        assert folds, "need at least one fold to assert on"
        for fold in folds:
            assert "scheduled_train_end" in fold
            actual = pd.Timestamp(fold["train_end"])
            scheduled = pd.Timestamp(fold["scheduled_train_end"])
            # Purging can only ever move the end EARLIER.
            assert actual <= scheduled
        # With horizon=5 and daily bars, purging is not hypothetical: at
        # least one fold must have lost training tail to it.
        assert any(
            pd.Timestamp(f["train_end"]) < pd.Timestamp(f["scheduled_train_end"])
            for f in folds
        ), "expected label-overlap purging to shorten at least one training window"

    def test_target_horizon_and_purge_count_reported(self, patched_multi_factory):
        result = run_experiment(_dataset(_spec()), _model_spec(), dataset_id="ds_vr3")
        report = result["validation_report"]
        assert report["target_horizon"] == 5
        assert report["n_train_rows_purged_overlap"] >= 0

    def test_min_folds_rejects_a_single_split(self, patched_multi_factory):
        """
        A model used to be registered after ONE surviving fold, which is a
        single train/test split rather than walk-forward validation.
        """
        # A train window long enough that only one fold fits.
        spec = _model_spec(train_window=400, test_window=60, embargo=5)
        with pytest.raises(ValidationError, match="below min_folds"):
            run_experiment(_dataset(_spec()), spec, dataset_id="ds_vr4")

    def test_min_folds_can_be_lowered_deliberately(self, patched_multi_factory):
        spec = _model_spec(train_window=400, test_window=60, embargo=5, min_folds=1)
        result = run_experiment(_dataset(_spec()), spec, dataset_id="ds_vr5")
        assert result["n_folds"] == 1


class TestTrainingOnlyBaseline:
    """
    baseline_regression_metrics built its constant from the TEST fold's own
    mean. That is an oracle: at prediction time nobody knows the future
    window's average realized return, so `model MAE vs baseline MAE` was not
    a valid comparison. It never contaminated the trained model, only the
    number the model was judged against.
    """

    def test_constant_comes_from_training_not_test(self):
        from standard_quant_tools.modeling.validation.metrics import (
            baseline_regression_metrics,
        )

        train_y = np.array([1.0, 1.0, 1.0, 1.0])
        test_y = np.array([5.0, 5.0, 5.0, 5.0])

        oracle = baseline_regression_metrics(test_y)
        honest = baseline_regression_metrics(test_y, train_y)

        # The oracle predicts 5.0 and is perfect; the honest baseline
        # predicts 1.0 and is off by 4.0 on every row.
        assert oracle["baseline_mae"] == pytest.approx(0.0)
        assert honest["baseline_mae"] == pytest.approx(4.0)
        assert honest["baseline_is_oracle"] == 0.0
        assert oracle["baseline_is_oracle"] == 1.0

    def test_engine_uses_the_training_baseline(self, patched_multi_factory):
        """End to end: a registered model's metrics must not be scored
        against an oracle."""
        from standard_quant_tools.modeling.registry.model_registry import load_manifest

        from .test_scoring import _dataset_spec, _train_a_model_with_spec

        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_baseline")
        metrics = load_manifest(model_id).oos_metrics
        assert metrics.get("baseline_is_oracle") == 0.0


class TestPooledCrossSectionalIC:
    """
    average_fold_metrics computes a weighted mean across folds. That is
    correct for cs_ic_mean but wrong for the statistics built on it:

        mean(fold stds)   != std(all OOS daily ICs)
        mean(fold ICIRs)  != mean(all ICs) / std(all ICs)

    Averaging folds' stds discards the BETWEEN-fold variation entirely —
    exactly the variation ICIR exists to measure — so a model whose IC was
    stable inside each fold but swung between them scored as dependable.
    """

    def test_pooled_std_differs_from_averaged_fold_stds(self):
        from standard_quant_tools.modeling.validation.metrics import (
            aggregate_cross_sectional_ic,
            summarize_cross_sectional_ic,
        )

        # Two folds, each internally tight, but centred far apart.
        fold_a = pd.Series(
            [0.20, 0.21, 0.19], index=pd.date_range("2024-01-01", periods=3)
        )
        fold_b = pd.Series(
            [-0.20, -0.21, -0.19], index=pd.date_range("2024-02-01", periods=3)
        )

        per_fold = [
            summarize_cross_sectional_ic(fold_a, "cs_ic"),
            summarize_cross_sectional_ic(fold_b, "cs_ic"),
        ]
        averaged_std = np.mean([m["cs_ic_std"] for m in per_fold])
        pooled = aggregate_cross_sectional_ic([fold_a, fold_b], "cs_ic")

        # Each fold looks rock steady; pooled, the model is not.
        assert averaged_std < 0.02
        assert pooled["cs_ic_std"] > 0.15
        assert pooled["cs_ic_std"] > 10 * averaged_std

    def test_pooled_icir_is_not_the_average_of_fold_icirs(self):
        from standard_quant_tools.modeling.validation.metrics import (
            aggregate_cross_sectional_ic,
            summarize_cross_sectional_ic,
        )

        fold_a = pd.Series(
            [0.20, 0.21, 0.19], index=pd.date_range("2024-01-01", periods=3)
        )
        fold_b = pd.Series(
            [-0.20, -0.21, -0.19], index=pd.date_range("2024-02-01", periods=3)
        )

        fold_icirs = [
            summarize_cross_sectional_ic(f, "cs_ic")["cs_ic_icir"]
            for f in (fold_a, fold_b)
        ]
        pooled = aggregate_cross_sectional_ic([fold_a, fold_b], "cs_ic")
        # Mean IC is ~0 pooled, so ICIR must be ~0 -- not the average of two
        # large-magnitude opposite-signed fold ICIRs.
        assert abs(pooled["cs_ic_icir"]) < 0.5
        assert max(abs(v) for v in fold_icirs) > 10

    def test_n_dates_is_the_total_oos_dates(self):
        from standard_quant_tools.modeling.validation.metrics import (
            aggregate_cross_sectional_ic,
        )

        fold_a = pd.Series([0.1, 0.2], index=pd.date_range("2024-01-01", periods=2))
        fold_b = pd.Series([0.3], index=pd.date_range("2024-02-01", periods=1))
        pooled = aggregate_cross_sectional_ic([fold_a, fold_b], "cs_ic")
        assert pooled["cs_ic_n_dates"] == 3.0

    def test_empty_input_is_handled(self):
        from standard_quant_tools.modeling.validation.metrics import (
            aggregate_cross_sectional_ic,
        )

        out = aggregate_cross_sectional_ic([], "cs_ic")
        assert out["cs_ic_n_dates"] == 0.0
