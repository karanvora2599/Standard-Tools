"""
A run says what its numbers mean (see the CHANGELOG entry of 2026-10-04).

Every model fitted to a 30-name daily equity panel reported a positive
cross-sectional rank IC, a negative r2 and, for the tree models, a null
importance block, and its `warnings` were empty. None of the sixteen runs' rank ICs was distinguishable
from zero; the r2 sat below a baseline that is zero by construction on a
ranked label; and the null importances were the estimators' own, not a
failed fit. A run now tests its headline against the null and warns when it
does not beat it, records the test in `validation_report["headline"]`, and
explains the other two in a new `notes` field. The numbers themselves --
metrics, predictions, importances -- are untouched.
"""

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling.adapters import (
    ClassificationAdapter,
    RankingAdapter,
    RegressionAdapter,
    SurvivalAdapter,
    headline_metrics,
)
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.specs import (
    ComputeBudgetSpec,
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.validation.comparison import (
    MIN_HEADLINE_DATES,
    andrews_lag,
    cosine_variance,
    headline_degrees_of_freedom,
    headline_lag,
    mean_vs_null_test,
    newey_west_variance,
)

HEADLINE_KEYS = {
    "metric",
    "null",
    "value",
    "n_dates",
    "t_stat",
    "t_stat_uncorrected",
    "p_value",
    "hac_lag",
    "hac_degrees_of_freedom",
    "ic_autocorrelation_lag1",
    "beats_null",
}


def _dataset(signal=0.0, n_entities=12, n_dates=260, seed=0, classification=False):
    """A panel whose label is `signal` times feature a plus noise; zero
    signal is a panel no model can beat the null on."""
    rng = np.random.default_rng(seed)
    dates = np.repeat(pd.bdate_range("2020-01-01", periods=n_dates), n_entities)
    X = rng.normal(size=(len(dates), 3))
    target = signal * X[:, 0] + rng.normal(size=len(dates))
    if classification:
        target = (target > 0).astype(float)
    panel = pd.DataFrame(
        {
            "date": dates,
            "entity": np.tile([f"E{i:02d}" for i in range(n_entities)], n_dates),
            "a": X[:, 0],
            "b": X[:, 1],
            "c": X[:, 2],
            "target": target,
        }
    )
    return {
        "panel": panel,
        "feature_ids": ["a", "b", "c"],
        "target_id": "forward_direction:5" if classification else "forward_return:5",
        "data_hash": f"headline-{signal}-{seed}-{classification}",
    }


def _spec(estimator="ridge", task="regression", params=None, **kwargs):
    kwargs.setdefault(
        "validation",
        ValidationSpec(train_window=60, test_window=20, embargo=2, min_folds=2),
    )
    return ModelSpec(
        task=task,
        estimator=EstimatorSpec(type=estimator, params=params or {}),
        budget=ComputeBudgetSpec(max_parallelism=1),
        random_seed=3,
        **kwargs,
    )


def _headline_warnings(result):
    return [w for w in result["warnings"] if "out-of-sample date" in w or "0.5 a" in w]


# ── The pure helpers ─────────────────────────────────────────────────────


class TestTheLag:
    def test_andrews(self):
        assert andrews_lag(100) == 4
        assert andrews_lag(504) == 5
        assert andrews_lag(0) == 0

    def test_twice_the_horizon_when_it_is_longer(self):
        """A 5-day label over 504 dates: 2h = 10 beats Andrews' 5. At h - 1
        Bartlett weights recover 68% of the overlap's long-run variance."""
        assert headline_lag(504, 5) == 10
        assert headline_lag(504, 1) == 5
        assert headline_lag(504, None) == 5

    def test_capped_below_the_sample(self):
        assert headline_lag(8, 20) == 7
        assert headline_lag(1, 5) == 0

    def test_bartlett_at_h_minus_one_recovers_68_percent(self):
        """The arithmetic the lag rule rests on: an h-day overlap's ICs have
        autocorrelation 1 - k/h, whose long-run variance is h times the daily
        one; Bartlett weights to h - 1 keep 1 + 2 sum (1 - k/h)^2 of it."""
        h = 5
        kept = 1 + 2 * sum((1 - k / h) ** 2 for k in range(1, h))
        assert kept / h == pytest.approx(0.68)
        at_2h = 1 + 2 * sum((1 - k / (2 * h + 1)) * (1 - k / h) for k in range(1, h))
        assert at_2h / h == pytest.approx(0.8545, abs=1e-4)


class TestTheMeanAgainstTheNull:
    def test_unless_a_lag_is_named_the_t_reads_the_cosine_variance(self):
        """The headline's test since the CHANGELOG entry of 2026-10-04: the
        variance from the series' lowest cosine frequencies, and the
        p-value from Student's t with as many degrees of freedom."""
        from scipy import stats

        rng = np.random.default_rng(0)
        x = 0.1 + rng.normal(size=400)
        out = mean_vs_null_test(x, null=0.0, horizon=5)
        degrees = headline_degrees_of_freedom(400, 5)
        assert out["lag"] is None and out["degrees_of_freedom"] == degrees == 21
        assert out["t_stat"] == pytest.approx(
            x.mean() / math.sqrt(cosine_variance(x, degrees)), rel=1e-12
        )
        assert out["p_value"] == pytest.approx(
            2 * stats.t.sf(abs(out["t_stat"]), degrees), rel=1e-9
        )
        assert out["t_stat_uncorrected"] == pytest.approx(
            x.mean() / math.sqrt(newey_west_variance(x, 0))
        )

    def test_the_t_is_the_newey_west_one(self):
        rng = np.random.default_rng(0)
        x = 0.1 + rng.normal(size=400)
        out = mean_vs_null_test(x, null=0.0, lag=7)
        assert out["n"] == 400 and out["lag"] == 7
        assert out["degrees_of_freedom"] is None
        assert out["t_stat"] == pytest.approx(
            x.mean() / math.sqrt(newey_west_variance(x, 7))
        )
        assert out["t_stat_uncorrected"] == pytest.approx(
            x.mean() / math.sqrt(newey_west_variance(x, 0))
        )
        assert out["p_value"] == pytest.approx(
            math.erfc(abs(out["t_stat"]) / math.sqrt(2))
        )

    def test_a_persistent_series_is_corrected_down(self):
        """The case the correction exists for: positively autocorrelated
        values make the uncorrected t too large."""
        rng = np.random.default_rng(1)
        e = rng.normal(size=600)
        x = np.convolve(e, np.ones(5) / 5, mode="valid") + 0.05
        for out in (
            mean_vs_null_test(x, horizon=5),
            mean_vs_null_test(x, lag=headline_lag(len(x), 5)),
        ):
            assert out["autocorrelation_lag1"] > 0.6
            assert abs(out["t_stat"]) < abs(out["t_stat_uncorrected"])

    def test_nan_dropped_and_degenerate_series_are_nan(self):
        out = mean_vs_null_test([1.0, np.nan, 1.0, 1.0])
        assert out["n"] == 3 and math.isnan(out["t_stat"])
        assert math.isnan(mean_vs_null_test([0.2])["t_stat"])


class TestOneHeadlineMap:
    def test_the_adapters_carry_it(self):
        assert RegressionAdapter.headline == "cs_rank_ic_mean"
        assert headline_metrics("regression") == (
            "cs_rank_ic_mean",
            "rank_ic",
            "ic",
            "r2",
        )
        assert RankingAdapter.headline == "cs_rank_ic_mean"
        assert ClassificationAdapter.headline == "auc"
        assert SurvivalAdapter.headline == "cs_concordance_mean"
        assert headline_metrics("no_such_task") == ()

    def test_the_tool_layer_reads_the_same_map(self):
        from standard_quant_tools.modeling.agent.tools import _HEADLINE_METRIC

        for task in ("regression", "classification", "ranking", "survival"):
            assert _HEADLINE_METRIC[task] == headline_metrics(task)


# ── In a run ─────────────────────────────────────────────────────────────


class TestTheHeadlineInARun:
    def test_noise_is_not_distinguishable_from_zero_and_says_so(self):
        result = run_experiment(_dataset(0.0), _spec(), "ds", register=False)
        headline = result["validation_report"]["headline"]
        assert set(headline) == HEADLINE_KEYS
        assert headline["metric"] == "cs_rank_ic_mean"
        assert headline["value"] == result["oos_metrics"]["cs_rank_ic_mean"]
        assert headline["n_dates"] == result["oos_metrics"]["cs_rank_ic_n_dates"]
        degrees = headline_degrees_of_freedom(headline["n_dates"], 5)
        assert headline["hac_lag"] is None
        assert headline["hac_degrees_of_freedom"] == degrees
        assert headline["beats_null"] is False
        (line,) = _headline_warnings(result)
        assert "is not distinguishable from zero" in line
        assert (
            f"on Student's t with {degrees} degrees of freedom, from a long-run "
            f"variance over the series' {degrees} lowest cosine frequencies"
        ) in line
        assert "read as independent, the same series gives t =" in line
        assert "rank regression models by" in line

    def test_a_planted_signal_beats_it_quietly(self):
        result = run_experiment(_dataset(0.5), _spec(), "ds", register=False)
        headline = result["validation_report"]["headline"]
        assert headline["beats_null"] is True
        assert headline["t_stat"] > 2 and headline["p_value"] < 0.05
        assert _headline_warnings(result) == []

    def test_a_reversed_ordering_is_named(self, monkeypatch):
        """Significantly below zero: the predictions order the names in
        reverse, and the sentence says what the negated ordering scored."""
        from standard_quant_tools.modeling import adapters

        real = adapters.ModelAdapter.score
        monkeypatch.setattr(
            adapters.ModelAdapter,
            "score",
            lambda self, estimator, X: -real(self, estimator, X),
        )
        result = run_experiment(_dataset(0.5), _spec(), "ds", register=False)
        headline = result["validation_report"]["headline"]
        assert headline["beats_null"] is False and headline["t_stat"] < -2
        (line,) = _headline_warnings(result)
        assert "below zero by more than noise explains" in line
        assert f"{-headline['value']:+.4f}" in line

    def test_too_few_dates_is_not_tested(self):
        validation = ValidationSpec(
            train_window=60, test_window=4, embargo=2, min_folds=2
        )
        result = run_experiment(
            _dataset(0.0, n_dates=70),
            _spec(validation=validation),
            "ds",
            register=False,
        )
        headline = result["validation_report"]["headline"]
        assert headline["n_dates"] < MIN_HEADLINE_DATES
        assert headline["beats_null"] is None and headline["t_stat"] is None
        (line,) = _headline_warnings(result)
        assert "was not tested" in line

    def test_combinatorial_paths_test_each_date_once(self):
        cpcv = ValidationSpec(method="cpcv", n_splits=5, n_test_splits=2, embargo=2)
        dataset = _dataset(0.0, n_dates=200)
        result = run_experiment(dataset, _spec(validation=cpcv), "ds", register=False)
        headline = result["validation_report"]["headline"]
        assert headline["n_dates"] == result["oos_metrics"]["cs_rank_ic_n_dates"]
        assert headline["n_dates"] <= dataset["panel"]["date"].nunique()

    def test_classification_is_compared_with_one_half(self):
        noise = run_experiment(
            _dataset(0.0, classification=True),
            _spec("logistic", task="classification"),
            "ds",
            register=False,
        )
        headline = noise["validation_report"]["headline"]
        assert headline["metric"] == "auc" and headline["null"] == 0.5
        # An AUC now carries the Hanley-McNeil z and p, from the class
        # counts `oos_metrics` already records. This asserted
        # `beats_null == (value > 0.5)` with `t_stat is None` -- the point
        # comparison written down as the contract, under which an AUC of
        # 0.5000001 was a win.
        assert headline["t_stat"] is not None
        assert headline["n_dates"] is not None
        if headline["beats_null"] is True:
            assert headline["p_value"] < 0.05
        signal = run_experiment(
            _dataset(1.0, classification=True),
            _spec("logistic", task="classification"),
            "ds",
            register=False,
        )
        strong = signal["validation_report"]["headline"]
        # A real signal still reads as one -- the test has to be passable.
        assert strong["beats_null"] is True and strong["p_value"] < 0.05
        assert not [w for w in signal["warnings"] if w.startswith("auc is")]


class TestThePointComparisons:
    def test_survival_is_compared_and_NOT_tested(self):
        """A concordance mean has no standard error in `oos_metrics`, so
        there is nothing to test it with.

        This asserted `beats_null is False` at 0.48 and `is True` at 0.6 --
        a bare `value > null` recorded as a verdict. Both are None now,
        which is the third state meaning "no test was made": it refuses a
        deployment stage rather than opening one, and does not pretend the
        difference was shown to be real. Inventing a standard error here
        would repeat in a new place the mistake this change removes.
        """
        from standard_quant_tools.modeling.engine import _headline_report

        for value in (0.48, 0.6):
            block, warnings = _headline_report(
                SurvivalAdapter(),
                "survival",
                {"cs_concordance_mean": value},
                None,
                5,
            )
            assert block["null"] == 0.5
            assert block["beats_null"] is None
            (line,) = warnings
            assert "COMPARISON rather than a test" in line
            assert "beats_null is null, not false" in line

    def test_a_missing_or_undefined_headline_is_not_judged(self):
        """A three-class AUC is NaN; nothing to compare, nothing said."""
        from standard_quant_tools.modeling.engine import _headline_report

        for metrics in ({}, {"auc": float("nan")}):
            block, warnings = _headline_report(
                ClassificationAdapter(), "classification", metrics, None, 5
            )
            assert block["beats_null"] is None and warnings == []


class TestTheNotes:
    def test_r2_below_its_baseline_is_explained(self):
        result = run_experiment(
            _dataset(0.0), _spec("random_forest", params={"n_estimators": 5}), "ds"
        )
        metrics = result["oos_metrics"]
        assert metrics["r2"] < metrics["baseline_r2"]
        (note,) = [n for n in result["notes"] if n.startswith("r2 is")]
        folds = result["validation_report"]["folds"]
        below = sum(f["metrics"]["r2"] < f["metrics"]["baseline_r2"] for f in folds)
        assert f"below it in {below} of {len(folds)} folds" in note
        ceiling = np.average(
            [f["metrics"]["ic"] ** 2 for f in folds],
            weights=[f["n_test_rows"] for f in folds],
        )
        assert f"{ceiling:.4f} averaged over these folds" in note
        assert "measured by cs_rank_ic_mean, not by r2" in note

    def test_an_r2_above_its_baseline_needs_no_note(self):
        result = run_experiment(_dataset(1.0), _spec(), "ds", register=False)
        assert result["oos_metrics"]["r2"] > result["oos_metrics"]["baseline_r2"]
        assert not [n for n in result["notes"] if n.startswith("r2 is")]

    def test_histogram_boosting_says_its_importances_do_not_exist(self):
        result = run_experiment(
            _dataset(0.5),
            _spec("hist_gradient_boosting", params={"max_iter": 10}),
            "ds",
            register=False,
        )
        assert result["validation_report"]["importance_source"] == "none"
        (note,) = [n for n in result["notes"] if "feature_importance_summary" in n]
        assert "null for every feature by construction" in note
        assert "exposes_feature_importance false" in note
        assert "metric='cs_rank_ic_mean'" in note

    def test_a_forest_says_its_importances_have_no_sign(self):
        result = run_experiment(
            _dataset(0.5),
            _spec("random_forest", params={"n_estimators": 5}),
            "ds",
            register=False,
        )
        assert result["validation_report"]["importance_source"] == (
            "feature_importances"
        )
        (note,) = [n for n in result["notes"] if "feature_importance_summary" in n]
        assert "null signed_mean, signed_std and sign_consistency" in note
        assert "exposes_feature_importance true" in note

    def test_coefficients_need_no_note(self):
        result = run_experiment(_dataset(1.0), _spec(), "ds", register=False)
        assert result["validation_report"]["importance_source"] == "coefficients"
        assert result["notes"] == []

    def test_a_calibrated_run_keeps_its_warning_and_adds_no_note(self):
        result = run_experiment(
            _dataset(0.5, classification=True),
            _spec(
                "random_forest", task="classification", params={"n_estimators": 5}
            ).model_copy(
                update={
                    "estimator": EstimatorSpec(
                        type="random_forest",
                        params={"n_estimators": 5},
                        calibration="sigmoid",
                    )
                }
            ),
            "ds",
            register=False,
        )
        assert result["validation_report"]["importance_source"] == "none"
        assert [w for w in result["warnings"] if "CalibratedClassifierCV" in w]
        assert not [n for n in result["notes"] if "feature_importance" in n]

    def test_both_return_paths_carry_the_field(self):
        registered = run_experiment(_dataset(1.0), _spec(), "ds")
        unregistered = run_experiment(_dataset(1.0), _spec(), "ds", register=False)
        assert registered["notes"] == unregistered["notes"] == []


class TestTheManifestKeepsIt:
    def test_inspect_model_shows_the_headline_and_the_source(self):
        from standard_quant_tools.modeling.agent.models import InspectModelInput
        from standard_quant_tools.modeling.agent.tools import inspect_model

        result = run_experiment(_dataset(0.0), _spec(), "ds")
        data = inspect_model(
            InspectModelInput(model_id=result["model_id"], view="validation")
        ).data
        report = data["validation_report"]
        assert report["headline"]["beats_null"] is False
        assert report["importance_source"] == "coefficients"


class TestTheAblationDefault:
    def test_regression_compares_the_rank_ic(self):
        """The default was the alphabetically first metric, which for a
        regression is the 0/1 flag `baseline_is_oracle`: every contribution
        0.0."""
        from standard_quant_tools.modeling.agent.feature_tools import (
            _first_numeric_metric,
        )

        metrics = {"baseline_is_oracle": 0.0, "cs_rank_ic_mean": 0.02, "r2": -0.1}
        assert _first_numeric_metric(metrics, "regression") == "cs_rank_ic_mean"
        assert _first_numeric_metric(
            {"accuracy": 0.6, "auc": 0.7}, "classification"
        ) == ("auc")
        # A headline missing or NaN falls back to the sorted rule.
        assert (
            _first_numeric_metric(
                {"cs_rank_ic_mean": float("nan"), "mae": 1.0}, "ranking"
            )
            == "mae"
        )
        assert _first_numeric_metric(metrics) == "baseline_is_oracle"
