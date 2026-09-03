"""
Where a model is wrong, and the four ways a breakdown lies about it.

WHAT THESE PIN.

  1. Residual autocorrelation computed on the STACKED panel measures the
     row ordering, not the series. Consecutive rows in a long panel are
     different entities on the same date. A panel whose stacked residuals
     alternate perfectly has a within-entity autocorrelation of zero, and
     these two numbers must not be confused.
  2. Calibration is a separate question from accuracy. Predictions spread
     twice as wide as their outcomes rank perfectly and are still wrong to
     size from, which no R2 or IC shows.
  3. The worst bucket of any breakdown is almost always the emptiest one.
     A finding computed without a row-count floor reports noise.
  4. The conditional breakdown is the one that pays: a model that is fine
     in the low deciles of a feature and hopeless in the high ones has an
     unremarkable aggregate score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.models import AnalyzeModelErrorsInput
from standard_quant_tools.modeling.agent.tools import analyze_model_errors
from standard_quant_tools.modeling.diagnostics import (
    MIN_BUCKET_ROWS,
    N_BUCKETS,
    _bucket_report,
    calibration,
    error_attribution,
    heteroskedasticity,
    residual_autocorrelation,
    residual_summary,
    worst_buckets,
)


def _joined(residuals_by_entity, predicted=None, **columns) -> pd.DataFrame:
    """A joined predictions/panel frame, one entity per key."""
    rows = []
    for entity, residuals in residuals_by_entity.items():
        dates = pd.date_range("2024-01-01", periods=len(residuals), freq="B")
        for i, (date, residual) in enumerate(zip(dates, residuals)):
            p = 0.0 if predicted is None else float(predicted(entity, i))
            row = {
                "date": date,
                "entity": entity,
                "_predicted": p,
                "_residual": float(residual),
                "_actual": p + float(residual),
            }
            for name, fn in columns.items():
                row[name] = float(fn(entity, i))
            rows.append(row)
    return pd.DataFrame(rows)


class TestAutocorrelationIsMeasuredWithinEntity:
    def test_a_perfectly_alternating_stack_is_not_autocorrelation(self) -> None:
        """
        THE DEFECT THIS EXISTS TO CATCH. Entity A is biased high throughout
        and entity B low, so the panel stacked date-major alternates
        +1, -1, +1, -1 and has a lag-1 correlation near MINUS ONE. Neither
        entity's own residual series is autocorrelated at all.
        """
        rng = np.random.default_rng(0)
        n = 200
        frame = _joined(
            {
                "AAA": 1.0 + rng.normal(0, 0.1, n),
                "BBB": -1.0 + rng.normal(0, 0.1, n),
            }
        )
        # What the naive version would have reported, computed here so the
        # test states the wrong answer rather than alluding to it.
        stacked = frame.sort_values(["date", "entity"])["_residual"].to_numpy()
        naive = np.corrcoef(stacked[:-1], stacked[1:])[0, 1]
        assert naive < -0.9

        assert abs(residual_autocorrelation(frame)) < 0.2

    def test_a_genuinely_persistent_residual_is_reported(self) -> None:
        """An overlapping label really does autocorrelate, and must show."""
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 1, 300)
        # A 5-bar rolling mean: consecutive values share 4 of 5 terms, which
        # is exactly what an overlapping forward return does.
        overlapping = pd.Series(noise).rolling(5).mean().dropna().to_numpy()
        frame = _joined({"AAA": overlapping})
        assert residual_autocorrelation(frame) > 0.6

    def test_a_constant_residual_series_contributes_nothing(self) -> None:
        """Zero variance has no correlation; it must not become 0.0."""
        assert residual_autocorrelation(_joined({"AAA": [2.0] * 50})) is None


class TestCalibrationIsNotAccuracy:
    def test_predictions_spread_twice_as_wide_have_slope_one_half(self) -> None:
        """
        Ranks perfectly, correlates at 1.0, and is still wrong to size
        from -- the case an R2 or an IC cannot show you.
        """
        rng = np.random.default_rng(2)
        actual = rng.normal(0, 0.01, 500)
        predicted = actual * 2.0
        report = calibration(actual, predicted, "regression")
        assert report["slope"] == pytest.approx(0.5, abs=1e-6)
        assert report["intercept"] == pytest.approx(0.0, abs=1e-6)
        assert report["dispersion_ratio"] == pytest.approx(2.0, abs=1e-6)

    def test_a_calibrated_regression_has_slope_one(self) -> None:
        rng = np.random.default_rng(3)
        predicted = rng.normal(0, 0.01, 500)
        actual = predicted + rng.normal(0, 0.002, 500)
        report = calibration(actual, predicted, "regression")
        assert report["slope"] == pytest.approx(1.0, abs=0.05)

    def test_identical_predictions_have_no_slope_rather_than_zero(self) -> None:
        report = calibration(np.arange(50.0), np.full(50, 3.0), "regression")
        assert report["slope"] is None
        assert "no slope is defined" in report["note"]

    def test_an_overconfident_classifier_shows_in_ece(self) -> None:
        """Says 0.9, right 60% of the time. Brier alone buries this."""
        rng = np.random.default_rng(4)
        predicted = np.full(1000, 0.9)
        actual = (rng.uniform(size=1000) < 0.6).astype(float)
        report = calibration(actual, predicted, "classification")
        assert report["expected_calibration_error"] == pytest.approx(0.3, abs=0.05)
        assert len(report["reliability"]) == 1

    def test_a_calibrated_classifier_has_small_ece(self) -> None:
        rng = np.random.default_rng(5)
        predicted = rng.uniform(0.05, 0.95, 4000)
        actual = (rng.uniform(size=4000) < predicted).astype(float)
        report = calibration(actual, predicted, "classification")
        assert report["expected_calibration_error"] < 0.05
        assert report["brier_score"] < 0.25


class TestHeteroskedasticity:
    def test_error_growing_with_confidence_is_positive(self) -> None:
        """The direction that costs money: least reliable where most sure."""
        rng = np.random.default_rng(6)
        predicted = rng.uniform(0, 1, 800)
        actual = predicted + rng.normal(0, 1, 800) * predicted
        assert heteroskedasticity(actual, predicted) > 0.4

    def test_uniform_error_is_near_zero(self) -> None:
        rng = np.random.default_rng(7)
        predicted = rng.uniform(0, 1, 800)
        actual = predicted + rng.normal(0, 0.1, 800)
        assert abs(heteroskedasticity(actual, predicted)) < 0.15


class TestResidualSummary:
    def test_bias_is_the_mean_and_is_not_the_mae(self) -> None:
        report = residual_summary(np.full(100, 1.5), np.zeros(100))
        assert report["mean_error"] == pytest.approx(1.5)
        assert report["rmse"] == pytest.approx(1.5)

    def test_a_symmetric_error_has_no_bias_and_plenty_of_mae(self) -> None:
        actual = np.tile([1.0, -1.0], 50)
        report = residual_summary(actual, np.zeros(100))
        assert report["mean_error"] == pytest.approx(0.0)
        assert report["mean_absolute_error"] == pytest.approx(1.0)

    def test_a_fat_tail_shows_in_excess_kurtosis(self) -> None:
        rng = np.random.default_rng(8)
        actual = rng.standard_t(3, 3000)
        assert residual_summary(actual, np.zeros(3000))["excess_kurtosis"] > 2

    def test_one_usable_row_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="fewer than two rows"):
            residual_summary(np.array([1.0, np.nan]), np.array([0.0, 0.0]))


class TestAttributionFindsWhereNotJustHowMuch:
    def test_one_bad_entity_is_visible_and_the_aggregate_is_not(self) -> None:
        rng = np.random.default_rng(9)
        frame = _joined(
            {
                "AAA": rng.normal(0, 0.1, 100),
                "BBB": rng.normal(0, 0.1, 100),
                "CCC": rng.normal(0, 1.0, 100),  # ten times the error
            }
        )
        report = error_attribution(frame)
        by_entity = {r["entity"]: r for r in report["by_entity"]}
        assert by_entity["CCC"]["rmse"] > 5 * by_entity["AAA"]["rmse"]
        assert any("entity CCC" in line for line in worst_buckets(report))

    def test_a_feature_decile_finds_a_conditional_failure(self) -> None:
        """
        The breakdown that turns "mediocre" into something actionable: the
        model is fine in a narrow book and hopeless in a wide one.
        """
        rng = np.random.default_rng(10)
        spread = rng.uniform(0, 1, 400)
        frame = _joined(
            {"AAA": rng.normal(0, 1, 400) * spread},
            spread=lambda _e, i: spread[i],
        )
        report = error_attribution(frame, feature="spread")
        deciles = sorted(report["by_feature_decile"], key=lambda r: int(r["decile"]))
        assert deciles[-1]["rmse"] > 5 * deciles[0]["rmse"]
        assert any("feature decile" in line for line in worst_buckets(report))

    def test_an_unknown_feature_names_the_panel_columns(self) -> None:
        frame = _joined({"AAA": [0.1] * 40}, spread=lambda _e, _i: 1.0)
        with pytest.raises(ValidationError, match="not a column"):
            error_attribution(frame, feature="volatility")

    def test_a_flag_is_refused_rather_than_called_deciles(self) -> None:
        """
        qcut(duplicates="drop") does NOT raise on a two-valued column -- it
        returns two buckets, which would have been reported as deciles.
        """
        frame = _joined({"AAA": [0.1] * 40}, flag=lambda _e, i: float(i % 2))
        with pytest.raises(ValidationError, match="too few distinct values"):
            error_attribution(frame, feature="flag")

    def test_a_coarse_feature_is_bucketed_but_says_it_is_coarse(self) -> None:
        """
        Four equally-frequent values do not give four buckets -- the 25/50/75
        quantile edges land ON value boundaries and collapse, leaving three.
        The buckets are still usable; calling them deciles is what is not.
        """
        rng = np.random.default_rng(15)
        frame = _joined({"AAA": rng.normal(0, 1, 200)}, tier=lambda _e, i: float(i % 4))
        report = error_attribution(frame, feature="tier")
        buckets = report["by_feature_decile"]
        assert 3 <= len(buckets) < N_BUCKETS
        assert sum(r["n"] for r in buckets) == 200
        assert "not tenths of the sample" in report["feature_note"]

    def test_period_granularity_changes_the_buckets(self) -> None:
        rng = np.random.default_rng(11)
        frame = _joined({"AAA": rng.normal(0, 1, 260)})
        monthly = error_attribution(frame, period="M")["by_period"]
        yearly = error_attribution(frame, period="Y")["by_period"]
        assert len(monthly) > len(yearly)
        assert sum(r["n"] for r in monthly) == sum(r["n"] for r in yearly) == 260


class TestThinBucketsDoNotBecomeFindings:
    def test_the_emptiest_bucket_is_not_reported_as_the_worst(self) -> None:
        """
        WHY THIS MATTERS. A breakdown ranked purely by RMSE reports whichever
        bucket had the fewest rows, essentially always. `DDD` has five rows
        and the largest errors in the panel, and is excluded from the
        findings and marked `thin` in the table -- present, not promoted.
        """
        rng = np.random.default_rng(12)
        frame = _joined(
            {
                "AAA": rng.normal(0, 0.1, 200),
                "BBB": rng.normal(0, 0.1, 200),
                "CCC": rng.normal(0, 0.1, 200),
                "DDD": [50.0, -50.0, 50.0, -50.0, 50.0],
            }
        )
        report = error_attribution(frame)
        rows = {r["entity"]: r for r in report["by_entity"]}
        assert rows["DDD"]["thin"] is True
        assert rows["DDD"]["rmse"] > 100 * rows["AAA"]["rmse"]
        assert rows["AAA"]["thin"] is False

        assert not any("DDD" in line for line in worst_buckets(report))

    def test_the_floor_is_the_documented_one(self) -> None:
        rng = np.random.default_rng(13)
        frame = _joined(
            {
                "AT": rng.normal(0, 1, MIN_BUCKET_ROWS),
                "BELOW": rng.normal(0, 1, MIN_BUCKET_ROWS - 1),
            }
        )
        rows = {r["entity"]: r for r in error_attribution(frame)["by_entity"]}
        assert rows["AT"]["thin"] is False
        assert rows["BELOW"]["thin"] is True


class TestEvenErrorsProduceNoFindings:
    def test_a_uniformly_mediocre_model_says_nothing(self) -> None:
        """Silence is the honest answer, not a reason to invent a headline."""
        rng = np.random.default_rng(14)
        frame = _joined({name: rng.normal(0, 1, 200) for name in ("AAA", "BBB", "CCC")})
        assert worst_buckets(error_attribution(frame)) == []


class TestThroughTheToolOnARealModel:
    """
    The join these diagnostics depend on. The persisted out-of-sample frame
    carries date, entity and prediction and NO outcome, so every number here
    exists only because the actuals are fetched back from the dataset panel
    the model was fit on -- which is also the only reason a breakdown by
    feature decile is possible at all.
    """

    @staticmethod
    def _fit(patched_multi_factory):
        from standard_quant_tools.modeling.agent import (
            BuildModelDatasetInput,
            RunModelExperimentInput,
            build_model_dataset,
            run_model_experiment,
        )
        from standard_quant_tools.modeling.specs import (
            DatasetSpec,
            EstimatorSpec,
            FeatureSpec,
            ModelSpec,
            TargetSpec,
            ValidationSpec,
        )

        dataset = build_model_dataset(
            BuildModelDatasetInput(
                spec=DatasetSpec(
                    universe=["AAA", "BBB", "CCC"],
                    start="2022-01-01",
                    end="2023-12-31",
                    features=[
                        FeatureSpec(id="technical.rsi"),
                        FeatureSpec(id="risk.rolling_beta"),
                    ],
                    target=TargetSpec(horizon=5),
                    benchmark="SPY",
                )
            )
        )
        experiment = run_model_experiment(
            RunModelExperimentInput(
                dataset_id=dataset.dataset_id,
                spec=ModelSpec(
                    task="regression",
                    estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                    validation=ValidationSpec(
                        train_window=150, test_window=30, embargo=5
                    ),
                    random_seed=11,
                ),
            )
        )
        return experiment.model_id

    def test_a_registered_model_gets_a_full_error_report(
        self, patched_multi_factory
    ) -> None:
        model_id = self._fit(patched_multi_factory)
        result = analyze_model_errors(AnalyzeModelErrorsInput(model_id=model_id))

        assert result.n_rows > 0
        assert result.task == "regression"
        # Actuals came from the panel; without the join every one of these
        # would be undefined.
        assert result.residuals["n"] == result.n_rows
        assert result.residuals["rmse"] > 0
        assert result.calibration["slope"] is not None
        # Three entities, so three entity buckets and ten prediction ones.
        assert {r["entity"] for r in result.by_entity} == {"AAA", "BBB", "CCC"}
        assert len(result.by_prediction_decile) == 10
        assert result.by_feature_decile == []

    def test_a_feature_decile_breakdown_uses_the_panels_own_columns(
        self, patched_multi_factory
    ) -> None:
        model_id = self._fit(patched_multi_factory)
        result = analyze_model_errors(
            AnalyzeModelErrorsInput(model_id=model_id, feature="technical.rsi")
        )
        assert len(result.by_feature_decile) >= 3
        assert sum(r["n"] for r in result.by_feature_decile) == result.n_rows

    def test_an_unknown_feature_lists_what_the_panel_actually_carries(
        self, patched_multi_factory
    ) -> None:
        model_id = self._fit(patched_multi_factory)
        with pytest.raises(ValidationError, match="technical.rsi"):
            analyze_model_errors(
                AnalyzeModelErrorsInput(model_id=model_id, feature="not_a_feature")
            )

    def test_top_n_trims_the_table_and_counts_what_it_dropped(
        self, patched_multi_factory
    ) -> None:
        """A 500-name universe would otherwise return 500 rows of table."""
        model_id = self._fit(patched_multi_factory)
        result = analyze_model_errors(
            AnalyzeModelErrorsInput(model_id=model_id, top_n=2)
        )
        assert len(result.by_prediction_decile) == 4
        assert result.buckets_omitted["by_prediction_decile"] == 6
        # Worst first, so the head of the table is the finding.
        rmses = [r["rmse"] for r in result.by_prediction_decile]
        assert rmses[0] == max(rmses)

    def test_a_coarser_period_makes_fewer_buckets(self, patched_multi_factory) -> None:
        model_id = self._fit(patched_multi_factory)
        monthly = analyze_model_errors(
            AnalyzeModelErrorsInput(model_id=model_id, period="M", top_n=50)
        )
        yearly = analyze_model_errors(
            AnalyzeModelErrorsInput(model_id=model_id, period="Y", top_n=50)
        )
        assert len(monthly.by_period) > len(yearly.by_period)
        assert sum(r["n"] for r in yearly.by_period) == monthly.n_rows

    def test_an_unregistered_model_is_refused(self) -> None:
        with pytest.raises(Exception):
            analyze_model_errors(AnalyzeModelErrorsInput(model_id="mdl_nope"))


class TestABucketIsNotAPlaceToHideRows:
    """
    Two defects that made a breakdown quietly wrong rather than loud.
    """

    def test_rows_with_no_feature_value_are_counted_not_bucketed(self) -> None:
        """
        THE DEFECT. `qcut` returns NA for a row whose feature is missing,
        and those rows have perfectly good residuals -- so they grouped
        into a bucket labelled "<NA>" that was reported alongside the real
        deciles and could be named the worst one. A feature with a warm-up
        window produces those rows on every real panel.
        """
        rng = np.random.default_rng(20)
        n = 300
        feature = rng.normal(0, 1, n)
        feature[:60] = np.nan  # a warm-up window, as any rolling feature has
        frame = _joined(
            {"AAA": rng.normal(0, 1, n)},
            spread=lambda _e, i: feature[i],
        )
        report = error_attribution(frame, feature="spread")

        labels = [row["decile"] for row in report["by_feature_decile"]]
        assert "<NA>" not in labels
        assert all(label.isdigit() for label in labels)
        # Counted rather than silently dropped: analysing fewer rows than
        # the caller believes is its own kind of wrong answer.
        assert report["rows_without_feature"] == 60
        assert sum(r["n"] for r in report["by_feature_decile"]) == n - 60

    def test_a_null_bucket_cannot_become_a_finding(self) -> None:
        """The "<NA>" bucket was large and bad enough to be reported."""
        rng = np.random.default_rng(21)
        n = 400
        feature = rng.normal(0, 1, n)
        feature[:150] = np.nan
        residuals = rng.normal(0, 0.2, n)
        residuals[:150] = rng.normal(0, 5.0, 150)  # and much worse
        frame = _joined({"AAA": residuals}, spread=lambda _e, i: feature[i])
        report = error_attribution(frame, feature="spread")
        assert not any("NA" in line for line in worst_buckets(report))

    def test_numeric_buckets_are_ordered_by_value_not_as_strings(self) -> None:
        """
        Decile labels are strings so one table can hold entities, periods
        and numbers. Sorted as strings, bucket 10 lands between 1 and 2 --
        a trap set for whoever raises N_BUCKETS, since deciles are 0-9
        today.
        """
        rng = np.random.default_rng(22)
        frame = _joined({"AAA": rng.normal(0, 1, 120)})
        labels = pd.Series([str(i % 12) for i in range(120)], index=frame.index)
        rows = _bucket_report(frame, labels, "bucket")
        assert [r["bucket"] for r in rows] == [str(i) for i in range(12)]

    def test_non_numeric_buckets_still_sort_lexicographically(self) -> None:
        rng = np.random.default_rng(23)
        frame = _joined({name: rng.normal(0, 1, 40) for name in ("CCC", "AAA", "BBB")})
        rows = error_attribution(frame)["by_entity"]
        assert [r["entity"] for r in rows] == ["AAA", "BBB", "CCC"]
