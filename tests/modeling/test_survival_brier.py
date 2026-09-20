"""
The survival function behind the risk, and the Brier score read from it.

Concordance says whether a model orders the durations; it cannot say
whether "this row has a 30% chance of having gone by day 10" is right.
That takes a survival function per row and the Brier score under
censoring, which needs the censoring distribution estimated the right
way round. Planted both ways: the baseline with no covariates is the
Nelson-Aalen estimator by hand, the censoring estimate is a reverse
Kaplan-Meier by hand, and the whole score agrees with scikit-survival's
to the last digit when that library is installed.
"""

import numpy as np
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.estimators.survival import (
    HAS_XGBOOST_SURVIVAL,
    CoxPHRegressor,
    aft_survival_function,
    breslow_baseline,
    cox_survival_function,
    cumulative_hazard_at,
)
from standard_quant_tools.modeling.validation.survival import (
    brier_scores,
    brier_time_grid,
    censoring_distribution,
    integrated_brier_score,
    survival_metrics,
)

from .test_survival import _planted, _planted_dataset, _spec


def _sksurv_available() -> bool:
    try:
        import sksurv.metrics  # noqa: F401
    except ImportError:
        return False
    return True


def _labels(duration, event):
    return np.column_stack([duration, event])


class TestTheBaseline:
    def test_with_no_covariates_it_is_nelson_aalen_by_hand(self):
        durations = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        events = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
        times, cumhaz = breslow_baseline(durations, events, np.zeros(5))
        assert times.tolist() == [1.0, 2.0, 3.0]
        # 1/5 at t=1, then 1/4 of the four still at risk, then 1/2.
        assert np.allclose(cumhaz, [0.2, 0.45, 0.95])
        assert np.allclose(
            cumulative_hazard_at([0.5, 1.0, 2.5, 9.0], times, cumhaz),
            [0, 0.2, 0.45, 0.95],
        )

    def test_a_higher_risk_row_is_less_likely_to_still_be_going(self):
        X, duration, event = _planted(400, seed=3)
        model = CoxPHRegressor().fit(X, _labels(duration, event))
        grid = np.quantile(duration, [0.1, 0.3, 0.5, 0.7, 0.9])
        S = model.predict_survival_function(X, grid)
        assert S.shape == (400, 5)
        assert np.all(S <= 1.0) and np.all(S >= 0.0)
        assert np.all(np.diff(S, axis=1) <= 1e-12)
        risk = model.predict(X)
        high, low = np.argmax(risk), np.argmin(risk)
        assert np.all(S[high] < S[low])
        before_first = model.predict_survival_function(
            X, [model.baseline_times_[0] / 2]
        )
        assert np.allclose(before_first, 1.0)

    def test_the_aft_curves_are_curves_under_every_distribution(self):
        grid = np.array([0.5, 1.0, 2.0, 4.0])
        for distribution in ("normal", "logistic", "extreme"):
            S = aft_survival_function(np.log([1.0, 3.0]), grid, distribution, 0.7)
            assert S.shape == (2, 4)
            assert np.all(np.diff(S, axis=1) < 0)
            assert np.all(S[1] > S[0])  # a later predicted time survives longer
        with pytest.raises(ValidationError, match="unknown AFT"):
            aft_survival_function([0.0], grid, "cauchy", 1.0)


class TestTheBrierScore:
    def test_the_censoring_distribution_is_a_reverse_kaplan_meier_by_hand(self):
        durations = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
        events = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
        knots, G = censoring_distribution(durations, events)
        assert knots.tolist() == [1.0, 2.0, 3.0, 4.0]
        # At t=2 three are at risk once the tied event is taken out, one is
        # censored: 2/3. At t=4 the last row is censored: 0.
        assert np.allclose(G, [1.0, 2.0 / 3.0, 2.0 / 3.0, 0.0])

    def test_a_grid_stays_inside_both_follow_ups(self):
        train = _labels([1.0, 5.0, 9.0], [1, 1, 1])
        test = _labels([0.5, 2.0, 3.0, 8.0, 12.0], [1, 1, 1, 1, 1])
        grid = brier_time_grid(train, test)
        assert grid.tolist() == [2.0, 3.0, 8.0]  # inside [1, 9), never 12 or 0.5
        assert brier_time_grid(train, _labels([2.0, 3.0], [0, 1])).size == 0

    @pytest.mark.skipif(not _sksurv_available(), reason="scikit-survival not installed")
    def test_it_agrees_with_scikit_survival_to_the_last_digit(self):
        from sksurv.metrics import brier_score as sk_brier
        from sksurv.metrics import integrated_brier_score as sk_ibs
        from sksurv.util import Surv

        X, duration, event = _planted(700, seed=11)
        # The training half must span the test half's follow-up, which is
        # what scikit-survival requires of the censoring estimate.
        order = np.argsort(duration)
        test_idx = order[1:-1][::2][:200]
        train_idx = np.setdiff1d(np.arange(700), test_idx)
        train_y = _labels(duration[train_idx], event[train_idx])
        test_y = _labels(duration[test_idx], event[test_idx])
        model = CoxPHRegressor().fit(X[train_idx], train_y)
        grid = brier_time_grid(train_y, test_y)
        S = model.predict_survival_function(X[test_idx], grid)
        ours = brier_scores(train_y, test_y, S, grid)
        ours_ibs = integrated_brier_score(ours, grid)
        train_s = Surv.from_arrays(train_y[:, 1].astype(bool), train_y[:, 0])
        test_s = Surv.from_arrays(test_y[:, 1].astype(bool), test_y[:, 0])
        _times, theirs = sk_brier(train_s, test_s, S, grid)
        assert np.allclose(ours, theirs, atol=1e-12)
        assert abs(ours_ibs - sk_ibs(train_s, test_s, S, grid)) < 1e-12

    def test_an_informative_model_beats_a_constant_risk(self):
        X, duration, event = _planted(600, seed=5)
        train_y, test_y = _labels(duration[:400], event[:400]), _labels(
            duration[400:], event[400:]
        )
        model = CoxPHRegressor().fit(X[:400], train_y)
        grid = brier_time_grid(train_y, test_y)
        informed = integrated_brier_score(
            brier_scores(
                train_y, test_y, model.predict_survival_function(X[400:], grid), grid
            ),
            grid,
        )
        times, cumhaz = breslow_baseline(duration[:400], event[:400], np.zeros(400))
        flat = cox_survival_function(np.zeros(200), grid, times, cumhaz)
        constant = integrated_brier_score(
            brier_scores(train_y, test_y, flat, grid), grid
        )
        assert 0.0 < informed < constant < 0.5

    def test_the_metrics_report_it_only_when_they_can(self):
        X, duration, event = _planted(300, seed=7)
        train_y, test_y = _labels(duration[:200], event[:200]), _labels(
            duration[200:], event[200:]
        )
        model = CoxPHRegressor().fit(X[:200], train_y)
        risk = model.predict(X[200:])
        without = survival_metrics(test_y, risk)
        assert "integrated_brier" not in without
        with_curve = survival_metrics(
            test_y,
            risk,
            train_y=train_y,
            survival_function=lambda t: model.predict_survival_function(X[200:], t),
        )
        assert 0.0 < with_curve["integrated_brier"] < 0.5
        assert with_curve["brier_n_times"] >= 2
        assert with_curve["brier_horizon_min"] < with_curve["brier_horizon_max"]
        with pytest.raises(ValidationError, match="n_test, n_times"):
            brier_scores(train_y, test_y, np.ones((3, 3)), np.array([1.0, 2.0, 3.0]))


class TestThroughTheEngine:
    def test_cox_reports_an_integrated_brier_out_of_sample(self):
        result = run_experiment(_planted_dataset(), _spec(), "ds", register=False)
        metrics = result["oos_metrics"]
        assert 0.0 < metrics["integrated_brier"] < 0.25
        assert metrics["concordance"] > 0.6

    @pytest.mark.skipif(not HAS_XGBOOST_SURVIVAL, reason="xgboost not installed")
    @pytest.mark.parametrize("estimator", ["xgboost_cox", "xgboost_aft"])
    def test_the_xgboost_objectives_report_it_too(self, estimator):
        spec = _spec(estimator, {"n_estimators": 60, "max_depth": 2})
        result = run_experiment(_planted_dataset(), spec, "ds", register=False)
        assert 0.0 < result["oos_metrics"]["integrated_brier"] < 0.3
