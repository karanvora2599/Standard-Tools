"""
The model adapters, and two bugs the audit that produced them turned up.

The adapters exist because adding rankers put `if task == "ranking"` in four
places in `run_experiment`. What is worth testing is not the dispatch — the
rest of the suite exercises that end to end — but the contract each adapter
promises: what it hands the estimator, what score it produces, and which
metrics it says are meaningful. A capability report that disagrees with what
a run actually does is worse than no report.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.adapters import (
    ClassificationAdapter,
    RankingAdapter,
    RegressionAdapter,
    available_tasks,
    get_adapter,
)
from standard_quant_tools.modeling.estimators.registry import ESTIMATOR_REGISTRY
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    RankingSpec,
    ValidationSpec,
)


def _model_spec(task="ranking"):
    estimator = "lightgbm_ranker" if task == "ranking" else "ridge"
    return ModelSpec(
        task=task,
        estimator=EstimatorSpec(type=estimator, params={}),
        validation=ValidationSpec(train_window=10, test_window=5),
        ranking=RankingSpec(n_grades=4),
        random_seed=1,
    )


def _frame(n_dates=6, n_entities=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for date in pd.date_range("2020-01-01", periods=n_dates):
        for entity in range(n_entities):
            rows.append((date, f"E{entity:02d}", rng.normal(), rng.normal()))
    return pd.DataFrame(rows, columns=["date", "entity", "f1", "f2"])


class TestAdapterRegistry:
    def test_every_task_has_an_adapter(self):
        """A task in the estimator registry with no adapter would fail only
        when someone tried to run it."""
        tasks = {task for task, _ in ESTIMATOR_REGISTRY}
        assert tasks <= set(available_tasks()), tasks - set(available_tasks())

    def test_unknown_task_is_rejected_clearly(self):
        with pytest.raises(ValidationError, match="no model adapter"):
            get_adapter("sideways")

    def test_each_adapter_reports_its_own_task(self):
        for task in available_tasks():
            assert get_adapter(task).task == task


class TestPrepare:
    def test_the_default_adapter_passes_data_through_unchanged(self):
        frame = _frame()
        X = frame[["f1", "f2"]]
        y = np.arange(len(frame), dtype=float)
        arrays = RegressionAdapter().prepare(
            _model_spec("regression"), frame, X, y, None
        )
        np.testing.assert_array_equal(arrays.X, X.to_numpy())
        np.testing.assert_array_equal(arrays.y, y)
        assert arrays.group is None

    def test_ranking_grades_the_target_and_builds_groups(self):
        frame = _frame(n_dates=6, n_entities=8)
        X = frame[["f1", "f2"]]
        y = np.random.default_rng(1).normal(0, 1, len(frame))
        arrays = RankingAdapter().prepare(_model_spec(), frame, X, y, None)
        assert arrays.y.dtype.kind == "i", "ranker labels must be integer grades"
        assert arrays.group is not None
        assert arrays.group.sum() == len(frame)
        assert list(arrays.group) == [8] * 6

    def test_ranking_sort_is_independent_of_input_order(self):
        """
        Both libraries break histogram ties by row order and neither checks
        the grouping, so a shuffled panel must produce identical fit arrays —
        sorting on (date, entity) rather than date alone is what makes the
        fit a function of the data instead of its arrival order.
        """
        frame = _frame(n_dates=5, n_entities=6, seed=2)
        y = np.random.default_rng(3).normal(0, 1, len(frame))
        frame = frame.assign(_y=y)
        adapter, spec = RankingAdapter(), _model_spec()

        ordered = adapter.prepare(
            spec, frame, frame[["f1", "f2"]], frame["_y"].to_numpy(), None
        )
        shuffled = frame.iloc[np.random.default_rng(4).permutation(len(frame))]
        out = adapter.prepare(
            spec, shuffled, shuffled[["f1", "f2"]], shuffled["_y"].to_numpy(), None
        )
        np.testing.assert_allclose(ordered.X, out.X)
        np.testing.assert_array_equal(ordered.y, out.y)
        np.testing.assert_array_equal(ordered.group, out.group)

    def test_ranking_reorders_sample_weights_with_the_rows(self):
        """A weight left in the caller's order would be applied to the wrong
        row after sorting — silent, and wrong in exactly the way weighting
        exists to fix."""
        frame = _frame(n_dates=4, n_entities=5, seed=5)
        shuffled = frame.iloc[::-1].reset_index(drop=True)
        weights = np.arange(len(shuffled), dtype=float)
        arrays = RankingAdapter().prepare(
            _model_spec(),
            shuffled,
            shuffled[["f1", "f2"]],
            np.random.default_rng(6).normal(0, 1, len(shuffled)),
            weights,
        )
        assert arrays.sample_weight is not None
        # Same multiset, different order — i.e. permuted, not dropped.
        np.testing.assert_array_equal(np.sort(arrays.sample_weight), np.sort(weights))
        assert not np.array_equal(arrays.sample_weight, weights)


class TestCapabilities:
    def test_linear_models_are_reported_as_exposing_coefficients(self):
        """
        Regression test for a real bug. `hasattr(cls, "coef_")` is the
        obvious check and is wrong: sklearn sets coef_ during fit, so it does
        not exist on the class and EVERY linear model was reported as having
        no coefficients — while fold_feature_importance would go on to find
        one at fit time. The capability report has to agree with what the run
        actually produces.
        """
        from sklearn.linear_model import LogisticRegression, Ridge

        assert RegressionAdapter().capabilities(Ridge)["exposes_coefficients"]
        assert ClassificationAdapter().capabilities(LogisticRegression)[
            "exposes_coefficients"
        ]

    def test_trees_expose_importance_and_not_coefficients(self):
        from sklearn.ensemble import RandomForestRegressor

        caps = RegressionAdapter().capabilities(RandomForestRegressor)
        assert caps["exposes_feature_importance"]
        assert not caps["exposes_coefficients"]

    def test_hist_gradient_boosting_exposes_neither(self):
        """Not an oversight — it genuinely has neither, which is why
        fold_feature_importance reports NaN for it."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        caps = RegressionAdapter().capabilities(HistGradientBoostingRegressor)
        assert not caps["exposes_feature_importance"]
        assert not caps["exposes_coefficients"]

    def test_ranking_declares_groups_and_an_unscaled_score(self):
        if ("ranking", "lightgbm_ranker") not in ESTIMATOR_REGISTRY:
            pytest.skip("lightgbm not installed")
        caps = RankingAdapter().capabilities(
            ESTIMATOR_REGISTRY[("ranking", "lightgbm_ranker")]
        )
        assert caps["needs_groups"]
        assert not caps["score_has_scale"]

    def test_probability_support_is_detected(self):
        from sklearn.linear_model import LogisticRegression, Ridge

        assert ClassificationAdapter().capabilities(LogisticRegression)[
            "supports_probability"
        ]
        assert not RegressionAdapter().capabilities(Ridge)["supports_probability"]


class TestCapabilityReport:
    def test_reports_the_live_registries(self):
        from standard_quant_tools.modeling.capabilities import modeling_capabilities

        caps = modeling_capabilities()
        assert set(caps["tasks"]) == set(available_tasks())
        assert len(caps["estimators"]) == len(ESTIMATOR_REGISTRY)
        # `targets` reports buildability now rather than a flat name list:
        # six of the eighteen can be built from a price series and the
        # report used to advertise all eighteen the same way.
        assert "forward_return" in caps["targets"]["all"]
        assert "forward_return" in caps["targets"]["buildable"]
        assert "future_mid_return" in caps["targets"]["external_only"]
        assert "purged_kfold" in caps["validation"]["methods"]

    def test_optional_dependencies_are_reported_explicitly(self):
        """An estimator list that is silently shorter is much harder to act
        on than a stated absence."""
        from standard_quant_tools.modeling.capabilities import modeling_capabilities

        deps = modeling_capabilities()["optional_dependencies"]
        # The three it always reported, plus the ones whose absence an agent
        # could not otherwise learn about -- it would discover a missing
        # blpapi or cvxpy by making a call that failed, which is exactly the
        # "silently shorter" failure this test was written against.
        assert {"lightgbm", "xgboost", "native_extension"} <= set(deps)
        assert {"scipy", "numba", "polars", "blpapi", "cvxpy"} <= set(deps)
        assert all(isinstance(v, bool) for v in deps.values())

    def test_is_json_safe(self):
        import json

        from standard_quant_tools._jsonsafe import sanitize_for_json
        from standard_quant_tools.modeling.capabilities import modeling_capabilities

        json.dumps(sanitize_for_json(modeling_capabilities()))


class TestAuditFixes:
    """Two findings from the sweep that produced the adapters."""

    def test_clip_sigma_is_rejected_identically_with_and_without_the_kernel(self):
        """
        A backend divergence: the native kernel rejected a negative
        clip_sigma and the Python path silently skipped clipping, so the same
        call raised on a machine with the extension built and succeeded on one
        without it. Both must now refuse it the same way.
        """
        from standard_quant_tools.modeling.features import transforms

        frame = pd.DataFrame({"a": [1.0, 5.0, 9.0]})
        dates = np.array(["d", "d", "d"])
        original = transforms.HAS_CPP
        try:
            for flag in (True, False):
                transforms.HAS_CPP = flag
                with pytest.raises(ValidationError, match="clip_sigma"):
                    transforms.standardize_cross_sectional(frame, dates, -1.0)
        finally:
            transforms.HAS_CPP = original

    def test_zero_clip_sigma_still_means_no_clipping(self):
        """The guard rejects negatives without breaking the documented way to
        turn clipping off."""
        from standard_quant_tools.modeling.features.transforms import (
            standardize_cross_sectional,
        )

        frame = pd.DataFrame({"a": [1.0, 5.0, 9.0, 100.0]})
        dates = np.array(["d"] * 4)
        out = standardize_cross_sectional(frame, dates, 0.0)
        assert out["a"].abs().max() > 1.4

    def test_average_fold_metrics_on_an_empty_list(self):
        """Unreachable through run_experiment, which guards twice before
        calling it — but it is an importable helper, and an IndexError is not
        something a caller can act on."""
        from standard_quant_tools.modeling.validation.metrics import (
            average_fold_metrics,
        )

        assert average_fold_metrics([]) == {}
