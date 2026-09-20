"""
Distributional predictions: quantiles fitted beside the point, a
split-conformal interval around it, and both judged by whether the
outcome fell where they said it would.

Planted on a Gaussian panel: target = 0.5 f1 - 0.3 f2 + N(0, sigma), so
the true 90% band is 2 x 1.645 sigma wide and a 90% interval of any kind
should cover about 90% of out-of-sample outcomes. The tolerances are the
sampling error of a few thousand rows plus the estimation error of a
linear quantile fit, not a hedge. The other direction is planted too: an
estimator without a quantile parameter is refused by name, a
classification spec cannot ask for either, and `prediction` is bitwise
what it was before the distribution was requested.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import (
    EvaluateModelPortfolioInput,
    ValidateModelSpecInput,
)
from standard_quant_tools.modeling.agent.tools import (
    evaluate_model_portfolio,
    validate_model_spec,
)
from standard_quant_tools.modeling.capabilities import estimator_capabilities
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.ensemble import load_oos_predictions
from standard_quant_tools.modeling.estimators import boosting
from standard_quant_tools.modeling.estimators.registry import (
    quantile_estimators,
    quantile_support,
)
from standard_quant_tools.modeling.plan import fits_per_estimator, plan_experiment
from standard_quant_tools.modeling.portfolio_eval import (
    scale_by_uncertainty,
    transform_predictions_to_weights,
)
from standard_quant_tools.modeling.registry.model_registry import (
    load_distribution,
    load_manifest,
)
from standard_quant_tools.modeling.scoring import score_model
from standard_quant_tools.modeling.specs import (
    ConformalSpec,
    EstimatorSpec,
    ModelSpec,
    PredictionTransformSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.conformal import conformal_radius
from standard_quant_tools.modeling.validation.distributional import (
    distributional_metrics,
    pinball_loss,
    quantile_column,
)

from .test_portfolio_eval import _score_panel
from .test_scoring import _dataset_spec, _train_a_model_with_spec

SIGMA = 1.0
TRUE_WIDTH_90 = 2 * 1.6449 * SIGMA


def _gaussian_dataset(n_entities=30, n_dates=400, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    frames = []
    for entity in range(n_entities):
        f = rng.normal(size=(n_dates, 3))
        y = 0.5 * f[:, 0] - 0.3 * f[:, 1] + rng.normal(scale=SIGMA, size=n_dates)
        frame = pd.DataFrame(f, columns=["f1", "f2", "f3"])
        frame["date"] = dates
        frame["entity"] = f"E{entity:02d}"
        frame["target"] = y
        frame["label_end_date"] = list(dates[1:]) + [pd.NaT]
        frames.append(frame.iloc[:-1])
    panel = pd.concat(frames).sort_values(["date", "entity"]).reset_index(drop=True)
    return {
        "panel": panel,
        "feature_ids": ["f1", "f2", "f3"],
        "target_id": "forward_return:1",
        "data_hash": f"gauss-{seed}",
    }


def _spec(estimator="quantile", quantiles=(), intervals=None, **overrides) -> ModelSpec:
    params = (
        {"alpha": 0.0, "solver": "highs"} if estimator == "quantile" else {"alpha": 1.0}
    )
    fields = dict(
        task="regression",
        estimator=EstimatorSpec(type=estimator, params=params),
        validation=ValidationSpec(
            train_window=200, test_window=50, embargo=1, min_folds=1
        ),
        quantiles=list(quantiles),
        intervals=intervals,
        random_seed=3,
    )
    fields.update(overrides)
    return ModelSpec(**fields)


class TestTheSpec:
    def test_quantiles_are_sorted_unique_levels_inside_the_unit_interval(self):
        spec = _spec(quantiles=(0.95, 0.05, 0.5, 0.5))
        assert spec.quantiles == [0.05, 0.5, 0.95]
        with pytest.raises(ValueError, match="inside"):
            _spec(quantiles=(0.0, 0.5))
        with pytest.raises(ValueError, match="inside"):
            _spec(quantiles=(0.5, 1.0))
        assert quantile_column(0.05) == "q05"
        assert quantile_column(0.5) == "q50"
        assert quantile_column(0.975) == "q97.5"

    def test_a_distribution_is_a_regression_question(self):
        with pytest.raises(ValueError, match="regression"):
            ModelSpec(
                task="classification",
                estimator=EstimatorSpec(type="logistic"),
                validation=ValidationSpec(train_window=100, test_window=20),
                quantiles=[0.5],
            )
        with pytest.raises(ValueError, match="regression"):
            ModelSpec(
                task="classification",
                estimator=EstimatorSpec(type="logistic"),
                validation=ValidationSpec(train_window=100, test_window=20),
                intervals=ConformalSpec(),
            )
        with pytest.raises(ValueError):
            ConformalSpec(alpha=0.0)
        with pytest.raises(ValueError):
            ConformalSpec(calibration_folds=1)

    def test_the_plan_counts_every_extra_fit(self):
        spec = _spec(
            quantiles=(0.05, 0.5, 0.95), intervals=ConformalSpec(calibration_folds=3)
        )
        assert fits_per_estimator(spec) == 1 + 3 + 3
        dates = pd.bdate_range("2020-01-01", periods=399)
        plan = plan_experiment(spec, dates)
        assert plan.n_fits == 7 * len(plan.folds) + 7
        assert fits_per_estimator(_spec()) == 1


class TestTheRegistry:
    def test_the_registry_says_which_estimators_fit_a_quantile_and_how(self):
        assert quantile_support("regression", "quantile").param == "quantile"
        assert (
            quantile_support("regression", "quantile_gradient_boosting").param
            == "alpha"
        )
        assert quantile_support("regression", "ridge") is None
        assert {"quantile", "quantile_gradient_boosting"} <= set(
            quantile_estimators("regression")
        )
        if boosting.HAS_LIGHTGBM:
            support = quantile_support("regression", "lightgbm")
            assert support.param == "alpha" and support.fixed == {
                "objective": "quantile"
            }
        if boosting.HAS_XGBOOST:
            support = quantile_support("regression", "xgboost")
            assert support.param == "quantile_alpha"
            assert support.fixed == {"objective": "reg:quantileerror"}
        by_name = {(e["task"], e["name"]): e for e in estimator_capabilities()}
        assert by_name[("regression", "quantile")]["quantile_param"] == "quantile"
        assert by_name[("regression", "ridge")]["quantile_param"] is None

    def test_an_estimator_without_a_quantile_parameter_is_refused_by_name(self):
        spec = _spec(estimator="ridge", quantiles=(0.1, 0.9))
        with pytest.raises(ValidationError, match="quantile_gradient_boosting"):
            run_experiment(_gaussian_dataset(), spec, "ds", register=False)
        result = validate_model_spec(ValidateModelSpecInput(spec=spec))
        assert not result.valid
        assert [p.where for p in result.problems] == ["quantiles"]


class TestTheMetricsAlone:
    def test_pinball_at_the_median_is_half_the_absolute_error(self):
        y = np.array([1.0, 2.0, 4.0])
        pred = np.array([0.0, 3.0, 4.0])
        assert pinball_loss(y, pred, 0.5) == pytest.approx(
            0.5 * np.mean([1.0, 1.0, 0.0])
        )
        # An over-prediction costs (1 - q), an under-prediction q.
        assert pinball_loss(np.array([0.0]), np.array([1.0]), 0.9) == pytest.approx(0.1)
        assert pinball_loss(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)

    def test_coverage_width_and_crossings_are_what_they_say(self):
        y = np.array([0.0, 0.0, 0.0, 10.0])
        q = {
            0.05: np.array([-1.0, -1.0, 1.0, -1.0]),
            0.5: np.array([0.0, 0.0, 0.0, 0.0]),
            0.95: np.array([1.0, 1.0, 0.5, 1.0]),
        }
        out = distributional_metrics(
            y, q, lower=np.array([-1.0] * 4), upper=np.array([1.0] * 4), alpha=0.1
        )
        assert out["quantile_coverage_90"] == pytest.approx(0.5)
        assert out["quantile_width_90"] == pytest.approx(np.mean([2.0, 2.0, -0.5, 2.0]))
        assert out["quantile_crossing_rate"] == pytest.approx(0.25)
        assert out["interval_coverage"] == pytest.approx(0.75)
        assert out["interval_width"] == pytest.approx(2.0)
        assert out["interval_nominal_coverage"] == pytest.approx(0.9)
        assert set(out) >= {"pinball_q05", "pinball_q50", "pinball_q95"}

    def test_the_conformal_radius_is_the_corrected_quantile(self):
        residuals = np.arange(1, 20, dtype=float)  # 1..19, n = 19
        # ceil(20 x 0.9) = 18th smallest.
        assert conformal_radius(residuals, 0.1) == 18.0
        # Too few residuals for the level: the largest stands.
        assert conformal_radius(np.array([1.0, 2.0]), 0.01) == 2.0


class TestQuantilesThroughTheEngine:
    @pytest.fixture(scope="class")
    def dataset(self):
        return _gaussian_dataset()

    def test_the_columns_the_coverage_and_the_untouched_point(self, dataset):
        spec = _spec(quantiles=(0.05, 0.5, 0.95))
        result = run_experiment(dataset, spec, "ds")
        frame = _artifacts.load_artifact(result["oos_predictions_uri"])
        assert {"q05", "q50", "q95", "prediction"} <= set(frame.columns)
        metrics = result["oos_metrics"]
        assert 0.87 <= metrics["quantile_coverage_90"] <= 0.93
        assert abs(metrics["quantile_width_90"] - TRUE_WIDTH_90) < 0.15 * TRUE_WIDTH_90
        assert metrics["quantile_crossing_rate"] < 0.05
        assert (frame["q05"] <= frame["q95"]).mean() > 0.95
        # The base fit of a quantile regressor IS the median, so the point
        # column and q50 agree and the pinball loss at 0.5 is half the MAE.
        np.testing.assert_allclose(frame["prediction"], frame["q50"], atol=1e-6)
        assert metrics["pinball_q50"] == pytest.approx(0.5 * metrics["mae"], rel=1e-6)
        # `prediction` is bitwise what a point-only run produces.
        point = run_experiment(dataset, _spec(), "ds", register=False)
        assert result["n_folds"] == point["n_folds"]
        assert metrics["mae"] == point["oos_metrics"]["mae"]
        # Every consumer that reads the three canonical columns still can.
        assert list(load_oos_predictions(result["model_id"]).columns) == [
            "date",
            "entity",
            "prediction",
        ]
        manifest = load_manifest(result["model_id"])
        assert manifest.distribution["quantiles"] == [0.05, 0.5, 0.95]
        assert manifest.distribution["columns"] == {
            "q05": 0.05,
            "q50": 0.5,
            "q95": 0.95,
        }
        assert manifest.distribution["conformal"] is None
        state, models = load_distribution(result["model_id"])
        assert set(models) == {"q05", "q50", "q95"} and state["quantiles"] == [
            0.05,
            0.5,
            0.95,
        ]
        assert "quantile_models.joblib" in manifest.content_hashes

    def test_a_point_only_model_has_no_distribution(self, dataset):
        result = run_experiment(dataset, _spec(), "ds")
        assert load_manifest(result["model_id"]).distribution == {}
        assert load_distribution(result["model_id"]) == ({}, {})
        assert "quantile_crossing_rate" not in result["oos_metrics"]


class TestConformalThroughTheEngine:
    @pytest.fixture(scope="class")
    def dataset(self):
        return _gaussian_dataset(seed=11)

    def test_a_ridge_gets_an_interval_that_covers_what_it_claims(self, dataset):
        spec = _spec(
            estimator="ridge", intervals=ConformalSpec(alpha=0.1, calibration_folds=3)
        )
        result = run_experiment(dataset, spec, "ds")
        frame = _artifacts.load_artifact(result["oos_predictions_uri"])
        assert {"lower", "upper"} <= set(frame.columns)
        assert "q50" not in frame.columns
        metrics = result["oos_metrics"]
        assert 0.87 <= metrics["interval_coverage"] <= 0.93
        assert abs(metrics["interval_width"] - TRUE_WIDTH_90) < 0.15 * TRUE_WIDTH_90
        assert metrics["interval_nominal_coverage"] == pytest.approx(0.9)
        np.testing.assert_allclose(
            (frame["upper"] - frame["lower"]) / 2, frame["upper"] - frame["prediction"]
        )
        conformal = load_manifest(result["model_id"]).distribution["conformal"]
        assert conformal["method"] == "split" and conformal["alpha"] == 0.1
        assert conformal["radius"] > 0 and conformal["n_calibration"] > 100
        # One fit, plus three refits for the calibration, per fold and once more.
        fits = result["validation_report"]["fits"]
        assert fits["planned"] == 4 * result["n_folds"] + 4

    def test_quantiles_and_an_interval_together(self, dataset):
        spec = _spec(quantiles=(0.1, 0.9), intervals=ConformalSpec(alpha=0.2))
        result = run_experiment(dataset, spec, "ds", register=False)
        metrics = result["oos_metrics"]
        assert 0.77 <= metrics["quantile_coverage_80"] <= 0.83
        assert 0.77 <= metrics["interval_coverage"] <= 0.83

    def test_too_few_dates_to_calibrate_is_refused_by_name(self, dataset):
        spec = _spec(
            estimator="ridge",
            intervals=ConformalSpec(calibration_folds=10),
            validation=ValidationSpec(
                train_window=8, test_window=50, embargo=0, min_folds=1
            ),
        )
        with pytest.raises(ValidationError, match="calibration"):
            run_experiment(dataset, spec, "ds", register=False)


class TestScoringEmitsTheDistribution:
    def test_the_scored_frame_carries_the_same_columns(self, patched_multi_factory):
        spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(
                type="quantile", params={"alpha": 0.0, "solver": "highs"}
            ),
            validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
            quantiles=[0.05, 0.5, 0.95],
            intervals=ConformalSpec(alpha=0.1),
            random_seed=1,
        )
        model_id = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_distribution", model_spec=spec
        )
        result = score_model(
            model_id, as_of="2023-12-29", universe=["AAA", "BBB", "CCC"]
        )
        frame = _artifacts.load_artifact(result["predictions_uri"])
        assert {
            "entity",
            "date",
            "prediction",
            "q05",
            "q50",
            "q95",
            "lower",
            "upper",
        } <= set(frame.columns)
        radius = load_manifest(model_id).distribution["conformal"]["radius"]
        np.testing.assert_allclose(frame["upper"] - frame["lower"], 2 * radius)
        np.testing.assert_allclose(frame["prediction"], frame["q50"], atol=1e-6)

    def test_a_point_only_model_scores_exactly_as_before(self, patched_multi_factory):
        model_id = _train_a_model_with_spec(_dataset_spec(), dataset_id="ds_point")
        result = score_model(model_id, as_of="2023-12-29", universe=["AAA", "BBB"])
        frame = _artifacts.load_artifact(result["predictions_uri"])
        assert list(frame.columns) == ["entity", "date", "prediction"]


class TestUncertaintyScaledTransform:
    def test_it_hits_the_exposure_targets_like_any_other_method(self):
        panel = _score_panel(n_dates=8, n_entities=20, seed=1)
        spec = PredictionTransformSpec(
            method="uncertainty_scaled",
            gross_exposure=1.0,
            net_exposure=0.0,
            max_position_weight=0.5,
        )
        weights, diag = transform_predictions_to_weights(panel, spec)
        assert np.allclose(weights.abs().sum(axis=1), 1.0)
        assert np.allclose(weights.sum(axis=1), 0.0, atol=1e-9)
        assert diag["n_dates_below_target_gross"] == 0

    def test_scaling_divides_by_the_width_and_refuses_without_one(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "entity": ["A", "B"],
                "prediction": [0.02, 0.02],
                "lower": [0.01, -0.08],
                "upper": [0.03, 0.12],
            }
        )
        scaled = scale_by_uncertainty(frame, "test")
        # The same +2%, one with a +/-1% band and one with a +/-10% band.
        assert scaled["prediction"].tolist() == pytest.approx([1.0, 0.1])
        with pytest.raises(ValidationError, match="ModelSpec.intervals"):
            scale_by_uncertainty(frame.drop(columns=["lower"]), "test")
        degenerate = frame.assign(upper=frame["lower"])
        with pytest.raises(ValidationError, match="width"):
            scale_by_uncertainty(degenerate, "test")

    def test_the_evaluator_reads_the_interval_and_refuses_a_point_model(
        self, patched_multi_factory
    ):
        conformal = _train_a_model_with_spec(
            _dataset_spec(),
            dataset_id="ds_uncertainty",
            model_spec=ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
                intervals=ConformalSpec(alpha=0.1),
                random_seed=1,
            ),
        )
        transform = PredictionTransformSpec(method="uncertainty_scaled")
        result = evaluate_model_portfolio(
            EvaluateModelPortfolioInput(model_id=conformal, transform=transform)
        )
        assert result.model_id == conformal
        assert np.isfinite(result.metrics["sharpe_ratio"])
        point = _train_a_model_with_spec(
            _dataset_spec(), dataset_id="ds_uncertainty_pt"
        )
        with pytest.raises(ValidationError, match="ModelSpec.intervals"):
            evaluate_model_portfolio(
                EvaluateModelPortfolioInput(model_id=point, transform=transform)
            )
