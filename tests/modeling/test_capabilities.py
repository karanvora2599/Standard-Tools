"""
The capabilities added in the modeling feature pass.

Grouped by the gap each closes: cross-sectional normalization, sample
weighting, richer targets, expanding/purged validation, hyperparameter
search, and the faster estimators. Every one of these CHANGES model output
when switched on, so the tests here care about two things — that the
default is unchanged, and that the new behaviour is the behaviour claimed
rather than merely a different number.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.dataset.target import (
    apply_cross_sectional_target,
    build_target,
)
from standard_quant_tools.modeling.engine import _fit, run_experiment
from standard_quant_tools.modeling.estimators.registry import ESTIMATOR_REGISTRY
from standard_quant_tools.modeling.features.transforms import (
    standardize_cross_sectional,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    PreprocessingSpec,
    SearchSpec,
    TargetSpec,
    ValidationSpec,
    WeightingSpec,
)
from standard_quant_tools.modeling.validation.walk_forward import (
    PurgedKFoldSplit,
    WalkForwardSplit,
)
from standard_quant_tools.modeling.validation.weights import (
    label_uniqueness_weights,
    time_decay_weights,
)

FEATURES = ["technical.rsi", "risk.atr_pct", "market.momentum"]
UNIVERSE = ["AAA", "BBB", "CCC", "DDD"]


def _dataset(target_type="forward_return", **target_kwargs):
    return build_dataset(
        DatasetSpec(
            universe=UNIVERSE,
            start="2022-01-01",
            end="2030-01-01",
            features=[FeatureSpec(id=f) for f in FEATURES],
            target=TargetSpec(type=target_type, horizon=5, **target_kwargs),
            benchmark="SPY",
        )
    )


def _model(**kwargs):
    kwargs.setdefault("task", "regression")
    kwargs.setdefault("estimator", EstimatorSpec(type="ridge", params={"alpha": 1.0}))
    kwargs.setdefault(
        "validation", ValidationSpec(train_window=150, test_window=75, embargo=2)
    )
    kwargs.setdefault("random_seed", 1)
    return ModelSpec(**kwargs)


class TestCrossSectionalNormalization:
    def test_removes_the_common_component_by_construction(self):
        """
        The whole point: a day on which every entity's feature is high
        together carries no cross-sectional information, and after
        standardizing within the date it must carry none.
        """
        rng = np.random.default_rng(4)
        dates = np.repeat(pd.date_range("2020-01-01", periods=30), 8)
        # A large per-date "market" level plus a small per-entity signal.
        market = np.repeat(rng.normal(0, 10, 30), 8)
        frame = pd.DataFrame({"f": market + rng.normal(0, 1, 240)})

        out = standardize_cross_sectional(frame, dates)
        per_date_mean = out.groupby(dates)["f"].mean()
        assert np.allclose(per_date_mean.to_numpy(), 0.0, atol=1e-12)
        # And the market level, which dominated the raw column, is gone:
        # the standardized column no longer correlates with it at all.
        assert abs(np.corrcoef(out["f"].to_numpy(), market)[0, 1]) < 1e-10

    def test_matches_a_plain_groupby_implementation(self):
        rng = np.random.default_rng(9)
        dates = np.repeat(pd.date_range("2020-01-01", periods=25), 7)
        frame = pd.DataFrame({"a": rng.normal(0, 1, 175), "b": rng.normal(5, 3, 175)})
        fast = standardize_cross_sectional(frame, dates, clip_sigma=0.0)
        for column in frame.columns:
            grouped = frame[column].groupby(dates)
            expected = (frame[column] - grouped.transform("mean")) / grouped.transform(
                "std"
            )
            np.testing.assert_allclose(
                fast[column].to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12
            )

    def test_flat_cross_section_becomes_zero_not_nan(self):
        """Every entity at the mean is exactly what standardizing says
        they are; NaN here would silently drop the whole date."""
        dates = np.repeat(pd.date_range("2020-01-01", periods=3), 4)
        frame = pd.DataFrame({"f": [7.0] * 4 + list(range(4)) + [2.0] * 4})
        out = standardize_cross_sectional(frame, dates)
        assert not out["f"].isna().any()
        assert out["f"].iloc[:4].eq(0.0).all()
        assert out["f"].iloc[8:].eq(0.0).all()

    def test_clipping_bounds_an_outlier(self):
        dates = np.repeat(pd.date_range("2020-01-01", periods=1), 20)
        values = np.r_[np.zeros(19), 1000.0]
        out = standardize_cross_sectional(
            pd.DataFrame({"f": values}), dates, clip_sigma=3.0
        )
        assert out["f"].max() == pytest.approx(3.0)

    def test_ragged_cross_sections(self):
        """Entities entering and leaving must not break the segmentation."""
        rows = []
        rng = np.random.default_rng(12)
        for i, date in enumerate(pd.date_range("2020-01-01", periods=15)):
            for _ in range(2 + i % 5):
                rows.append((date, rng.normal()))
        frame = pd.DataFrame(rows, columns=["date", "f"])
        out = standardize_cross_sectional(frame[["f"]], frame["date"].to_numpy())
        means = out["f"].groupby(frame["date"].to_numpy()).mean()
        assert np.allclose(means.to_numpy(), 0.0, atol=1e-12)

    def test_default_is_unchanged(self, patched_multi_factory):
        assert _model().preprocessing.normalization == "pooled"

    def test_engine_accepts_it_end_to_end(self, patched_multi_factory):
        dataset = _dataset()
        pooled = run_experiment(dataset, _model(), "ds")
        cross = run_experiment(
            dataset,
            _model(preprocessing=PreprocessingSpec(normalization="cross_sectional")),
            "ds",
        )
        assert cross["n_folds"] == pooled["n_folds"]
        assert cross["validation_report"]["normalization"] == "cross_sectional"
        # It must actually do something different, or the option is a lie.
        assert cross["oos_metrics"]["cs_rank_ic_mean"] != pytest.approx(
            pooled["oos_metrics"]["cs_rank_ic_mean"], abs=1e-12
        )


class TestSampleWeighting:
    def test_uniqueness_is_higher_where_fewer_labels_overlap(self):
        """
        Interior rows sit under `horizon` concurrent labels; rows at the
        edges of the window sit under fewer and are therefore more
        informative per row. That ordering is the entire content of the
        weighting, so it is what gets asserted.
        """
        dates = pd.date_range("2020-01-01", periods=40).to_numpy()
        label_end = np.r_[dates[5:], np.repeat(np.datetime64("NaT"), 5)]
        entities = np.array(["AAA"] * 40)
        weights = label_uniqueness_weights(dates, label_end, entities)
        assert weights.shape == (40,)
        assert np.isclose(weights.mean(), 1.0)
        # The first row's label overlaps fewer earlier labels than a row in
        # the middle of the window, so it is weighted more heavily.
        assert weights[0] > weights[20]

    def test_uniqueness_is_computed_within_each_entity(self):
        """Two entities' labels are different series and do not make each
        other redundant, so adding an entity must not change weights."""
        dates = pd.date_range("2020-01-01", periods=30).to_numpy()
        label_end = np.r_[dates[5:], np.repeat(np.datetime64("NaT"), 5)]
        solo = label_uniqueness_weights(dates, label_end, np.array(["AAA"] * 30))
        both = label_uniqueness_weights(
            np.r_[dates, dates],
            np.r_[label_end, label_end],
            np.array(["AAA"] * 30 + ["BBB"] * 30),
        )
        np.testing.assert_allclose(both[:30], solo, rtol=0, atol=1e-12)

    def test_time_decay_halves_at_the_half_life(self):
        dates = pd.to_datetime(["2020-01-01", "2020-12-31", "2021-12-31"]).to_numpy()
        weights = time_decay_weights(dates, half_life=365.0)
        # Ratios, not absolute values, since the series is normalized.
        assert weights[2] / weights[1] == pytest.approx(2.0, rel=1e-3)
        assert weights[1] / weights[0] == pytest.approx(2.0, rel=1e-2)
        assert np.isclose(weights.mean(), 1.0)

    def test_zero_half_life_rejected(self):
        with pytest.raises(ValidationError, match="half_life"):
            time_decay_weights(pd.date_range("2020-01-01", periods=3).to_numpy(), 0.0)

    def test_estimator_without_sample_weight_raises_rather_than_ignoring(self):
        """
        Silently dropping the weights would leave a model that LOOKS like
        it corrected for label overlap and did not — worse than an error.
        """

        class NoWeights:
            def fit(self, X, y):
                return self

        with pytest.raises(ValidationError, match="does not accept sample_weight"):
            _fit(NoWeights(), np.zeros((3, 2)), np.zeros(3), np.ones(3))

    def test_default_is_unweighted(self):
        assert _model().weighting.method == "none"

    @pytest.mark.parametrize(
        "method", ["label_uniqueness", "time_decay", "uniqueness_and_time_decay"]
    )
    def test_engine_applies_each_method(self, patched_multi_factory, method):
        dataset = _dataset()
        result = run_experiment(
            dataset, _model(weighting=WeightingSpec(method=method)), "ds"
        )
        assert result["n_folds"] > 0
        assert result["validation_report"]["weighting"] == method


class TestTargets:
    def test_vol_scaled_divides_by_trailing_volatility(self):
        rng = np.random.default_rng(3)
        index = pd.date_range("2020-01-01", periods=300, freq="B")
        close = pd.Series(
            100 * np.cumprod(1 + rng.normal(0.0004, 0.012, 300)), index=index
        )
        spec = TargetSpec(type="forward_return_vol_scaled", horizon=5, vol_window=20)
        scaled = build_target(close, spec)
        raw = build_target(close, TargetSpec(type="forward_return", horizon=5))
        # Scaling by volatility should bring the target near unit variance,
        # which is the point of it.
        assert 0.5 < scaled.std() < 2.5
        assert raw.std() < 0.1
        # And it is the raw return divided by something positive, so signs
        # are preserved wherever both are defined.
        both = scaled.notna() & raw.notna()
        assert np.all(np.sign(scaled[both]) == np.sign(raw[both]))

    def test_vol_scaled_denominator_uses_no_future_data(self):
        """
        The divisor for row t must be knowable at t. Changing prices
        strictly AFTER t must therefore leave row t's target denominator
        alone — checked by confirming the volatility series itself does not
        move when the tail is altered.
        """
        rng = np.random.default_rng(6)
        index = pd.date_range("2020-01-01", periods=200, freq="B")
        close = pd.Series(
            100 * np.cumprod(1 + rng.normal(0.0004, 0.012, 200)), index=index
        )
        altered = close.copy()
        altered.iloc[150:] *= 3.0
        from standard_quant_tools.modeling.dataset.target import _horizon_volatility

        spec = TargetSpec(type="forward_return_vol_scaled", horizon=5, vol_window=20)
        a = _horizon_volatility(close, spec).iloc[:150]
        b = _horizon_volatility(altered, spec).iloc[:150]
        pd.testing.assert_series_equal(a, b)

    def test_triple_barrier_has_three_outcomes(self):
        rng = np.random.default_rng(8)
        index = pd.date_range("2020-01-01", periods=400, freq="B")
        close = pd.Series(
            100 * np.cumprod(1 + rng.normal(0.0, 0.015, 400)), index=index
        )
        labels = build_target(
            close, TargetSpec(type="triple_barrier", horizon=10, barrier=0.05)
        )
        assert set(labels.dropna().unique()) <= {0.0, 1.0, 2.0}
        assert labels.dropna().nunique() == 3

    def test_triple_barrier_labels_a_known_path(self):
        """A monotonically rising series can only ever touch the upper
        barrier, and a falling one only the lower."""
        index = pd.date_range("2020-01-01", periods=30, freq="B")
        rising = pd.Series(100 * 1.02 ** np.arange(30), index=index)
        falling = pd.Series(100 * 0.98 ** np.arange(30), index=index)
        spec = TargetSpec(type="triple_barrier", horizon=5, barrier=0.05)
        assert set(build_target(rising, spec).dropna().unique()) == {1.0}
        assert set(build_target(falling, spec).dropna().unique()) == {0.0}

    def test_triple_barrier_flat_series_touches_nothing(self):
        index = pd.date_range("2020-01-01", periods=30, freq="B")
        flat = pd.Series(100.0, index=index)
        labels = build_target(
            flat, TargetSpec(type="triple_barrier", horizon=5, barrier=0.05)
        )
        assert set(labels.dropna().unique()) == {2.0}

    def test_rank_target_is_centered_and_bounded(self):
        rng = np.random.default_rng(2)
        rows = []
        for date in pd.date_range("2020-01-01", periods=20):
            for entity in range(6):
                rows.append((date, f"E{entity}", rng.normal(0, 0.02)))
        panel = pd.DataFrame(rows, columns=["date", "entity", "target"])
        out = apply_cross_sectional_target(
            panel, TargetSpec(type="forward_return_rank", horizon=5)
        )
        assert out["target"].min() == pytest.approx(-0.5)
        assert out["target"].max() == pytest.approx(0.5)
        per_date = out.groupby("date")["target"].mean()
        assert np.allclose(per_date.to_numpy(), 0.0, atol=1e-12)

    def test_market_neutral_target_removes_the_date_mean(self):
        rng = np.random.default_rng(5)
        rows = []
        for date in pd.date_range("2020-01-01", periods=15):
            level = rng.normal(0, 0.05)  # a big common move
            for entity in range(5):
                rows.append((date, f"E{entity}", level + rng.normal(0, 0.005)))
        panel = pd.DataFrame(rows, columns=["date", "entity", "target"])
        out = apply_cross_sectional_target(
            panel, TargetSpec(type="forward_return_market_neutral", horizon=5)
        )
        per_date = out.groupby("date")["target"].mean()
        assert np.allclose(per_date.to_numpy(), 0.0, atol=1e-15)
        # The common component dominated the raw target and must be gone.
        assert out["target"].std() < panel["target"].std() / 5

    def test_single_entity_date_has_no_cross_section(self):
        panel = pd.DataFrame(
            [(pd.Timestamp("2020-01-01"), "AAA", 0.01)],
            columns=["date", "entity", "target"],
        )
        for target_type in ("forward_return_rank", "forward_return_market_neutral"):
            out = apply_cross_sectional_target(
                panel, TargetSpec(type=target_type, horizon=5)
            )
            assert out["target"].isna().all()

    @pytest.mark.parametrize(
        "target_type",
        [
            "forward_return_vol_scaled",
            "forward_return_rank",
            "forward_return_market_neutral",
        ],
    )
    def test_regression_runs_against_each_continuous_target(
        self, patched_multi_factory, target_type
    ):
        result = run_experiment(_dataset(target_type), _model(), "ds")
        assert result["n_folds"] > 0

    def test_classification_runs_against_triple_barrier(self, patched_multi_factory):
        dataset = _dataset("triple_barrier", barrier=0.03)
        result = run_experiment(
            dataset,
            _model(
                task="classification",
                estimator=EstimatorSpec(type="logistic", params={}),
            ),
            "ds",
        )
        assert result["n_folds"] > 0

    def test_task_target_mismatch_still_rejected(self, patched_multi_factory):
        with pytest.raises(ValidationError, match="expects one of"):
            run_experiment(_dataset("triple_barrier", barrier=0.03), _model(), "ds")


class TestValidationSchemes:
    def test_expanding_grows_the_training_window(self):
        dates = pd.Index(pd.date_range("2020-01-01", periods=60))
        rolling = list(WalkForwardSplit(20, 10, 2, "rolling").split(dates))
        expanding = list(WalkForwardSplit(20, 10, 2, "expanding").split(dates))
        assert len(rolling) == len(expanding)
        # Identical test windows, so the two are directly comparable.
        for (_, a), (_, b) in zip(rolling, expanding):
            np.testing.assert_array_equal(a, b)
        assert [len(t) for t, _ in rolling] == [20] * len(rolling)
        assert [len(t) for t, _ in expanding] == sorted(len(t) for t, _ in expanding)
        assert len(expanding[-1][0]) > len(expanding[0][0])

    def test_purged_kfold_tests_every_date_exactly_once(self):
        dates = pd.Index(pd.date_range("2020-01-01", periods=60))
        folds = list(PurgedKFoldSplit(n_splits=5, embargo=3).split(dates))
        assert len(folds) == 5
        tested = np.concatenate([test for _, test in folds])
        np.testing.assert_array_equal(np.sort(tested), np.arange(60))

    def test_purged_kfold_embargoes_both_sides(self):
        dates = pd.Index(pd.date_range("2020-01-01", periods=60))
        folds = list(PurgedKFoldSplit(n_splits=3, embargo=4).split(dates))
        # The middle fold has training data on both sides, and neither side
        # may come within `embargo` dates of the test block.
        train, test = folds[1]
        assert train.size > 0
        assert train.max() >= test.max()  # there IS training data after it
        assert np.min(np.abs(train[:, None] - test[None, :])) > 4 - 1

    def test_purged_kfold_rejects_a_single_split(self):
        with pytest.raises(ValidationError, match="n_splits >= 2"):
            PurgedKFoldSplit(n_splits=1)

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValidationError, match="scheme"):
            WalkForwardSplit(10, 5, 0, "sideways")

    def test_walk_forward_default_is_rolling(self):
        assert ValidationSpec(train_window=10, test_window=5).scheme == "rolling"

    def test_walk_forward_requires_windows(self):
        with pytest.raises(ValueError, match="requires"):
            ValidationSpec(method="walk_forward")

    def test_purged_kfold_does_not_require_windows(self):
        assert ValidationSpec(method="purged_kfold").n_splits == 5

    @pytest.mark.parametrize(
        "validation",
        [
            ValidationSpec(train_window=150, test_window=75, embargo=2),
            ValidationSpec(
                train_window=150, test_window=75, embargo=2, scheme="expanding"
            ),
            ValidationSpec(method="purged_kfold", n_splits=3, embargo=2),
        ],
        ids=["rolling", "expanding", "purged_kfold"],
    )
    def test_engine_runs_each_scheme(self, patched_multi_factory, validation):
        result = run_experiment(_dataset(), _model(validation=validation), "ds")
        assert result["n_folds"] > 0
        assert result["validation_report"]["method"] == validation.method


class TestHyperparameterSearch:
    def test_search_selects_and_reports_per_fold(self, patched_multi_factory):
        result = run_experiment(
            _dataset(),
            _model(
                estimator=EstimatorSpec(type="ridge", params={}),
                search=SearchSpec(
                    param_grid={"alpha": [0.01, 1.0, 100.0]}, inner_splits=2
                ),
            ),
            "ds",
        )
        reports = result["validation_report"]["hyperparameter_search"]
        assert reports is not None and len(reports) == result["n_folds"]
        for report in reports:
            if report["searched"]:
                assert report["best_params"]["alpha"] in (0.01, 1.0, 100.0)
                # Every candidate is kept, so a reader can see how flat the
                # surface was rather than only which value won.
                assert len(report["candidates"]) == 3

    def test_no_search_reports_none(self, patched_multi_factory):
        result = run_experiment(_dataset(), _model(), "ds")
        assert result["validation_report"]["hyperparameter_search"] is None

    def test_random_search_respects_its_budget(self, patched_multi_factory):
        result = run_experiment(
            _dataset(),
            _model(
                estimator=EstimatorSpec(type="ridge", params={}),
                search=SearchSpec(
                    method="random",
                    param_grid={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
                    n_iter=3,
                    inner_splits=2,
                ),
            ),
            "ds",
        )
        reports = result["validation_report"]["hyperparameter_search"]
        searched = [r for r in reports if r["searched"]]
        assert searched, "expected at least one fold to run a search"
        for report in searched:
            assert report["n_candidates"] == 3

    def test_training_window_too_short_declines_rather_than_guessing(self):
        """A search on two dates selects on noise, which is worse than not
        searching — so it must decline and say why."""
        from standard_quant_tools.modeling.validation.search import (
            search_best_params,
        )

        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                "target": [0.1, 0.2],
                "f": [1.0, 2.0],
            }
        )
        params, report = search_best_params(
            task="regression",
            search_spec=SearchSpec(param_grid={"alpha": [1.0]}, inner_splits=5),
            base_params={"alpha": 7.0},
            train_frame=frame,
            feature_ids=["f"],
            random_seed=1,
            fit_predict=lambda *a: (np.zeros(1), None),
        )
        assert report["searched"] is False
        assert "too few" in report["reason"]
        assert params == {"alpha": 7.0}

    def test_empty_grid_rejected(self):
        with pytest.raises(ValueError, match="at least one parameter"):
            SearchSpec(param_grid={})
        with pytest.raises(ValueError, match="no candidate values"):
            SearchSpec(param_grid={"alpha": []})


class TestFasterEstimators:
    def test_quantile_estimators_are_registered(self):
        assert ("regression", "quantile") in ESTIMATOR_REGISTRY
        assert ("regression", "quantile_gradient_boosting") in ESTIMATOR_REGISTRY

    def test_quantile_gradient_boosting_obeys_the_sklearn_param_contract(self):
        """
        Regression test for a real bug. An sklearn estimator's __init__ must
        name every parameter explicitly, because get_params() reads them off
        the signature; the first version of this class forwarded **kwargs
        and silently lost random_state, which then failed inside fit.
        """
        from standard_quant_tools.modeling.estimators.boosting import (
            QuantileGradientBoostingRegressor,
        )

        params = QuantileGradientBoostingRegressor().get_params()
        for name in (
            "alpha",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "random_state",
        ):
            assert name in params, name
        assert QuantileGradientBoostingRegressor(alpha=0.9).alpha == 0.9

    @pytest.mark.parametrize("name", ["lightgbm", "xgboost"])
    def test_optional_boosters_register_or_are_absent_cleanly(self, name):
        """
        Neither is a declared dependency, so the contract is: present and
        usable, or absent with the registry's ordinary error — never a
        broken import.
        """
        from standard_quant_tools.modeling.estimators import boosting

        available = {
            "lightgbm": boosting.HAS_LIGHTGBM,
            "xgboost": boosting.HAS_XGBOOST,
        }[name]
        registered = ("regression", name) in ESTIMATOR_REGISTRY
        assert available == registered

    def test_registered_boosters_fit_through_the_engine(self, patched_multi_factory):
        from standard_quant_tools.modeling.estimators import boosting

        names = [
            n
            for n, ok in (
                ("lightgbm", boosting.HAS_LIGHTGBM),
                ("xgboost", boosting.HAS_XGBOOST),
            )
            if ok
        ]
        if not names:
            pytest.skip("neither lightgbm nor xgboost is installed")
        dataset = _dataset()
        for name in names:
            result = run_experiment(
                dataset,
                _model(estimator=EstimatorSpec(type=name, params={"n_estimators": 20})),
                "ds",
            )
            assert result["n_folds"] > 0, name

    def test_param_ceilings_still_apply(self):
        from standard_quant_tools.modeling.estimators import boosting
        from standard_quant_tools.modeling.estimators.registry import validate_params

        if not boosting.HAS_LIGHTGBM:
            pytest.skip("lightgbm not installed")
        with pytest.raises(ValidationError, match="exceeds the maximum"):
            validate_params("regression", "lightgbm", {"n_estimators": 10_000_000})
        with pytest.raises(ValidationError, match="does not accept"):
            validate_params("regression", "lightgbm", {"not_a_param": 1})
