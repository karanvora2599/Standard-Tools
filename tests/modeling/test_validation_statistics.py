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


def _reference_cross_sectional_ic(y_true, y_pred, dates, method="spearman"):
    """
    The per-date implementation cross_sectional_ic replaced, kept verbatim
    as the oracle. The vectorized version exists purely for speed, so the
    only thing worth asserting about it is that it computes the same
    numbers — an independent reimplementation would test whether the two
    agree on a definition, which is not the question.
    """
    frame = pd.DataFrame({"date": dates, "y": y_true, "p": y_pred})
    per_date = {}
    for date, group in frame.groupby("date", sort=True):
        if len(group) < 2:
            continue
        value = float(group["y"].corr(group["p"], method=method))
        per_date[date] = 0.0 if np.isnan(value) else value
    return pd.Series(per_date, dtype=float)


def _assert_matches_reference(y, p, dates, tol=1e-12):
    for method in ("spearman", "pearson"):
        expected = _reference_cross_sectional_ic(y, p, dates, method)
        actual = cross_sectional_ic(y, p, dates, method)
        assert list(actual.index) == list(expected.index), method
        assert len(actual) == len(expected), method
        if len(expected):
            np.testing.assert_allclose(
                actual.to_numpy(), expected.to_numpy(), rtol=0, atol=tol
            )


class TestCrossSectionalICVectorization:
    """
    cross_sectional_ic was 72% of a measured ridge walk-forward run, so it
    was rewritten from a per-date groupby into a small number of array
    passes. It has two internal layouts — a balanced-panel path that
    reshapes to (n_dates, n_entities), and a segment-reduction fallback for
    ragged panels — and both must reproduce the previous numbers exactly.
    """

    @pytest.mark.parametrize("seed", range(6))
    def test_balanced_panel_matches_reference(self, seed):
        rng = np.random.default_rng(seed)
        n_dates, n_entities = 30, 14
        dates = np.repeat(pd.date_range("2020-01-01", periods=n_dates), n_entities)
        y = rng.normal(0, 1, n_dates * n_entities)
        p = 0.3 * y + rng.normal(0, 1, n_dates * n_entities)
        _assert_matches_reference(y, p, dates)

    def test_ties_are_averaged_the_way_pandas_averages_them(self):
        """
        Rounding to one decimal forces many equal values per cross-section.
        Tie handling is the one part of a hand-written Spearman that
        silently disagrees rather than failing loudly: every member of a
        tied run must take the run's MEAN ordinal.
        """
        rng = np.random.default_rng(11)
        dates = np.repeat(pd.date_range("2020-01-01", periods=25), 20)
        y = np.round(rng.normal(0, 1, 500), 1)
        p = np.round(0.4 * y + rng.normal(0, 1, 500), 1)
        assert len(np.unique(y)) < 100, "test data is not actually tied"
        _assert_matches_reference(y, p, dates)

    def test_return_scale_data_does_not_lose_precision(self):
        """
        Real inputs are daily returns (~1e-2), where the textbook
        n*Sxy - Sx*Sy shortcut differences two nearly equal large numbers
        and loses most of its significant digits. Pinning a tight tolerance
        here is what keeps the implementation on the centered form.
        """
        rng = np.random.default_rng(3)
        dates = np.repeat(pd.date_range("2020-01-01", periods=40), 60)
        y = rng.normal(0.0004, 0.012, 2400)
        p = 0.25 * y + rng.normal(0.0004, 0.012, 2400)
        _assert_matches_reference(y, p, dates, tol=1e-13)

    def test_constant_cross_section_scores_zero_not_nan(self):
        """A date where every prediction is identical has undefined
        correlation; the contract is 0.0, not NaN."""
        dates = np.repeat(pd.date_range("2020-01-01", periods=4), 6)
        rng = np.random.default_rng(5)
        y = rng.normal(0, 1, 24)
        p = rng.normal(0, 1, 24)
        p[:6] = 7.0
        out = cross_sectional_ic(y, p, dates, "pearson")
        assert out.iloc[0] == 0.0
        assert not out.isna().any()
        _assert_matches_reference(y, p, dates)

    def test_ragged_panel_matches_reference(self):
        """Entities entering and leaving makes the panel non-rectangular,
        which routes through the segment-reduction fallback."""
        rng = np.random.default_rng(17)
        rows = []
        for date in pd.date_range("2021-01-04", periods=45):
            for _ in range(int(rng.integers(1, 22))):
                rows.append((date, rng.normal(), rng.normal()))
        frame = pd.DataFrame(rows, columns=["date", "y", "p"])
        widths = frame.groupby("date").size()
        assert widths.nunique() > 1, "test panel is not actually ragged"
        _assert_matches_reference(
            frame["y"].to_numpy(), frame["p"].to_numpy(), frame["date"].to_numpy()
        )

    def test_both_internal_paths_agree_on_the_same_data(self):
        """
        A balanced panel with one row deleted is the same data seen through
        the other code path. The two layouts must not disagree, or an
        entity's IPO date would silently change every earlier date's score.
        """
        rng = np.random.default_rng(23)
        n_dates, n_entities = 20, 9
        dates = np.repeat(pd.date_range("2020-01-01", periods=n_dates), n_entities)
        y = rng.normal(0, 1, n_dates * n_entities)
        p = 0.5 * y + rng.normal(0, 1, n_dates * n_entities)

        balanced = cross_sectional_ic(y, p, dates, "spearman")
        # Drop one row from the LAST date only; every earlier date is
        # untouched and must score identically through the ragged path.
        ragged = cross_sectional_ic(y[:-1], p[:-1], dates[:-1], "spearman")
        np.testing.assert_allclose(
            balanced.to_numpy()[:-1], ragged.to_numpy()[:-1], rtol=0, atol=1e-12
        )

    def test_nan_rows_are_dropped_pairwise_like_pandas(self):
        rng = np.random.default_rng(29)
        dates = np.repeat(pd.date_range("2022-02-01", periods=25), 12)
        y = rng.normal(0, 1, 300)
        p = 0.4 * y + rng.normal(0, 1, 300)
        y[rng.random(300) < 0.25] = np.nan
        p[rng.random(300) < 0.15] = np.nan
        _assert_matches_reference(y, p, dates)

    def test_all_nan_date_is_emitted_as_zero_not_dropped(self):
        """
        Pinning a quirk that was deliberately preserved. The row-count gate
        runs BEFORE the NaN drop, so a date with two all-NaN rows is
        reported as exactly 0.0 rather than omitted. Arguably wrong — a
        date with no usable data is not a date with zero IC — but the
        vectorization was a speed change and had no business moving a
        reported metric. Changing it is a separate decision; this test is
        what would catch it happening by accident.
        """
        dates = np.repeat(pd.date_range("2020-05-05", periods=2), 4)
        y = np.array([np.nan] * 4 + [1.0, 2.0, 3.0, 4.0])
        p = np.array([np.nan] * 4 + [2.0, 1.0, 4.0, 3.0])
        out = cross_sectional_ic(y, p, dates, "pearson")
        assert len(out) == 2
        assert out.iloc[0] == 0.0
        _assert_matches_reference(y, p, dates)

    def test_infinities_are_not_treated_as_missing(self):
        """pandas only treats NaN as missing, so inf must flow through:
        it ranks as an extreme for spearman and voids pearson to 0.0."""
        dates = np.repeat(pd.date_range("2023-01-02", periods=5), 8)
        rng = np.random.default_rng(31)
        y = rng.normal(0, 1, 40)
        p = rng.normal(0, 1, 40)
        p[3] = np.inf
        y[20] = -np.inf
        _assert_matches_reference(y, p, dates)

    @pytest.mark.parametrize(
        "y,p,dates",
        [
            (np.array([]), np.array([]), np.array([], dtype="datetime64[ns]")),
            (
                np.array([1.0]),
                np.array([2.0]),
                np.array(["2020-01-01"], dtype="datetime64[ns]"),
            ),
            (
                np.arange(5.0),
                np.arange(5.0)[::-1],
                pd.date_range("2020-01-01", periods=5).to_numpy(),
            ),
        ],
        ids=["empty", "single-row", "every-date-has-one-entity"],
    )
    def test_degenerate_shapes_return_an_empty_float_series(self, y, p, dates):
        out = cross_sectional_ic(y, p, dates, "spearman")
        assert out.empty
        assert out.dtype == float
        _assert_matches_reference(y, p, dates)
