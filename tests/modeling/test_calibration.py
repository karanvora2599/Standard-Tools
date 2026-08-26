"""
Calibration, and the live path it fixes.

`convert_reference(proba_threshold=T)` turns a classifier's probabilities
into a signal panel. The threshold only means something if "probability
above T" really means "wins about T of the time" — and for a tree ensemble
it does not, because averaging trees compresses probabilities toward the
middle.

Measured on a 6,000-row synthetic panel with a real but noisy signal, asking
the question a caller actually asks — if I set `proba_threshold=T`, what
fraction of the rows I select actually win:

    T      raw forest         isotonic-calibrated
           realized    n      realized    n
    0.5      0.740    943       0.741   909
    0.7      0.875    353       0.829   532
    0.9        --       0       0.912   194

The bottom row is the failure worth fixing. The raw forest never emits a
probability above 0.9, so a caller who asks for 0.9 gets an EMPTY signal
panel — not a wrong answer, no answer, and no error anywhere to say why.

WHAT THESE TESTS PIN. Not that calibration improves a score, which is a
claim about a dataset. That the reachable probability RANGE widens, which is
what makes a high threshold usable at all, and that the three ways of asking
for calibration wrongly are refused by name.
"""

import numpy as np
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.engine import _calibrated
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    ValidationSpec,
)


def _spec(task="classification", calibration="none", folds=3):
    return ModelSpec(
        task=task,
        estimator=EstimatorSpec(
            type="random_forest" if task == "classification" else "ridge",
            calibration=calibration,
            calibration_folds=folds,
        ),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=0,
    )


def _panel(n=6000, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    p = 1 / (1 + np.exp(-(1.2 * X[:, 0] + 0.8 * X[:, 1] - 0.5 * X[:, 2])))
    y = (rng.random(n) < p).astype(int)
    return X, y


class TestTheDefaultChangesNothing:
    def test_no_calibration_returns_the_estimator_untouched(self):
        """Every existing spec has to behave exactly as before."""
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier()
        assert _calibrated(estimator, _spec(), 1000) is estimator

    def test_the_field_defaults_to_none(self):
        assert EstimatorSpec(type="random_forest").calibration == "none"


class TestItRefusesTheThreeWrongWays:
    def test_calibrating_a_regressor_is_refused_by_name(self):
        """There are no probabilities to map. A spec asking for it is a
        mistake worth naming rather than a request to ignore."""
        from sklearn.linear_model import Ridge

        with pytest.raises(ValidationError) as exc:
            _calibrated(Ridge(), _spec(task="regression", calibration="isotonic"), 1000)
        message = str(exc.value)
        assert "regression" in message
        assert "only classification" in message

    def test_too_few_rows_for_the_folds_is_refused_before_sklearn_gets_there(self):
        """sklearn's own error arrives three frames down talking about
        n_splits, which is not a clue anybody can act on."""
        from sklearn.ensemble import RandomForestClassifier

        with pytest.raises(ValidationError) as exc:
            _calibrated(
                RandomForestClassifier(), _spec(calibration="isotonic", folds=5), 4
            )
        message = str(exc.value)
        assert "calibration_folds" in message
        assert "train_window" in message

    def test_an_unknown_method_is_rejected_by_the_schema(self):
        with pytest.raises(Exception):
            EstimatorSpec(type="random_forest", calibration="platt")

    def test_the_fold_count_is_bounded_in_the_schema(self):
        with pytest.raises(Exception):
            EstimatorSpec(
                type="random_forest", calibration="isotonic", calibration_folds=1
            )


class TestItWidensTheReachableRange:
    """The property that makes a high threshold usable, and the one that
    actually failed."""

    @staticmethod
    def _fit_both():
        from sklearn.ensemble import RandomForestClassifier

        X, y = _panel()
        Xtr, ytr, Xte = X[:4000], y[:4000], X[4000:]
        raw = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0).fit(
            Xtr, ytr
        )
        calibrated = _calibrated(
            RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0),
            _spec(calibration="isotonic"),
            len(ytr),
        ).fit(Xtr, ytr)
        return raw.predict_proba(Xte)[:, 1], calibrated.predict_proba(Xte)[:, 1]

    def test_the_raw_forest_cannot_reach_a_high_threshold(self):
        """The bug, stated as a test. Not a wrong answer -- no answer."""
        raw, _cal = self._fit_both()
        assert (raw > 0.9).sum() == 0, (
            "the raw forest reached 0.9 on this data, so this test no longer "
            "demonstrates the failure it was written for"
        )

    def test_calibration_makes_that_threshold_reachable(self):
        _raw, calibrated = self._fit_both()
        assert (calibrated > 0.9).sum() > 0

    def test_a_caller_at_that_threshold_gets_what_they_asked_for(self):
        """The end of the live path: rows selected at 0.9 should win about
        0.9 of the time."""
        from sklearn.ensemble import RandomForestClassifier

        X, y = _panel()
        Xtr, ytr, Xte, yte = X[:4000], y[:4000], X[4000:], y[4000:]
        calibrated = _calibrated(
            RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0),
            _spec(calibration="isotonic"),
            len(ytr),
        ).fit(Xtr, ytr)
        selected = calibrated.predict_proba(Xte)[:, 1] > 0.9
        assert selected.sum() > 20
        assert yte[selected].mean() > 0.8, (
            "rows selected at proba_threshold=0.9 won "
            f"{yte[selected].mean():.2f} of the time, which is not what the "
            "threshold promises"
        )

    def test_sigmoid_also_widens_it(self):
        """Platt is the safer choice on a short history, so it has to work
        too rather than being a documented-but-untested option."""
        from sklearn.ensemble import RandomForestClassifier

        X, y = _panel()
        calibrated = _calibrated(
            RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0),
            _spec(calibration="sigmoid"),
            4000,
        ).fit(X[:4000], y[:4000])
        assert calibrated.predict_proba(X[4000:])[:, 1].max() > 0.9


class TestHuber:
    def test_it_is_registered_for_regression(self):
        from standard_quant_tools.modeling.estimators.registry import (
            ESTIMATOR_REGISTRY,
        )

        assert ("regression", "huber") in ESTIMATOR_REGISTRY

    def test_it_is_discoverable_without_a_new_tool(self):
        """The mechanism working as designed: list_modeling_capabilities
        reports the registry, so a new estimator is discoverable the moment
        it is registered. That is why modeling can gain capabilities without
        gaining tools."""
        from standard_quant_tools.modeling.agent.models import (
            ListModelingCapabilitiesInput,
        )
        from standard_quant_tools.modeling.agent.tools import (
            list_modeling_capabilities,
        )

        result = list_modeling_capabilities(ListModelingCapabilitiesInput())
        assert "huber" in str(result.model_dump())

    def test_it_is_less_moved_by_one_outlier_than_least_squares(self):
        """The reason it exists. An 8-sigma day contributes 64 times what a
        1-sigma day does to a squared loss, and financial targets have
        those."""
        from sklearn.linear_model import HuberRegressor, LinearRegression

        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 1))
        y = 2.0 * X[:, 0] + rng.normal(0, 0.1, 200)
        clean_ols = LinearRegression().fit(X, y).coef_[0]
        clean_huber = HuberRegressor().fit(X, y).coef_[0]

        y_dirty = y.copy()
        y_dirty[0] += 50.0  # one bad print
        dirty_ols = LinearRegression().fit(X, y_dirty).coef_[0]
        dirty_huber = HuberRegressor().fit(X, y_dirty).coef_[0]

        assert abs(dirty_huber - clean_huber) < abs(dirty_ols - clean_ols), (
            "Huber moved at least as much as least squares under a single "
            "outlier, which is the one thing it is for"
        )

    def test_its_epsilon_bound_is_declared(self):
        """A parameter without a declared bound is one an agent can set to
        anything, and epsilon below 1.0 is not a valid Huber loss."""
        from standard_quant_tools.modeling.estimators.registry import (
            _PARAM_SCHEMAS,
            allowed_params,
        )

        assert "epsilon" in allowed_params("regression", "huber")
        bound = _PARAM_SCHEMAS[("regression", "huber")].bounds["epsilon"]
        assert bound.minimum == 1.0
        assert bound.note, "a bound without a note explains nothing"


class TestNoNewTools:
    def test_phase_4_added_no_tools(self):
        """The plan's own instruction: all of this lands in the registry and
        the spec, and `list_modeling_capabilities` already reports the
        registry. A dozen capabilities without a dozen tools is the
        mechanism working, not a shortcut."""
        from standard_quant_tools.modeling.agent import MODELING_TOOL_DISPATCH

        assert "calibrate_model" not in MODELING_TOOL_DISPATCH
        assert "fit_huber" not in MODELING_TOOL_DISPATCH
        assert len(MODELING_TOOL_DISPATCH) == 16
