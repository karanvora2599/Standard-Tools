"""
The five built-in steps added after the registry: robust scaling, the
quantile (rank-gauss) transform, the missingness indicator, imputation and
PCA whitening. Every oracle is planted -- a median and MAD computed by
hand, a covariance the whitened output must flatten, a test-fold NaN that
must receive the TRAINING median -- and two of the steps change the column
set the estimator sees, so the engine's labelling of importance by the
pipeline's output is pinned end to end.
"""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.capabilities import modeling_capabilities
from standard_quant_tools.modeling.preprocessing import (
    FoldContext,
    apply_pipeline,
    build_step,
    fit_and_apply_pipeline,
    fit_pipeline,
    get_preprocessor,
)
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.scoring import score_model
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    PreprocessingSpec,
    StepSpec,
    ValidationSpec,
)

from .test_deployed_pipeline import ALPHA, UNIVERSE, _dataset_spec, _register

CTX = FoldContext(dates=np.array([]), entities=None)


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame({k: np.asarray(v, dtype=np.float64) for k, v in columns.items()})


class TestRobustScale:
    def test_median_and_mad_by_hand(self):
        """[1, 2, 3, 4, 100]: median 3, |x - 3| = [2, 1, 0, 1, 97], MAD 1."""
        train = _frame(f=[1, 2, 3, 4, 100])
        step = build_step("robust_scale", {})
        state = step.fit(train, CTX)
        assert state["center"]["f"] == 3.0
        assert state["scale"]["f"] == pytest.approx(1.4826)
        out = step.transform(_frame(f=[3.0, 4.0]), state, CTX)
        assert out["f"].tolist() == pytest.approx([0.0, 1.0 / 1.4826])

    def test_without_the_normal_factor_the_scale_is_the_raw_mad(self):
        state = build_step("robust_scale", {"scale_to_normal": False}).fit(
            _frame(f=[1, 2, 3, 4, 100]), CTX
        )
        assert state["scale"]["f"] == 1.0

    def test_an_outlier_that_moves_the_zscore_scale_does_not_move_this_one(self):
        base = _frame(f=np.linspace(-1, 1, 201))
        spiked = base.copy()
        spiked.loc[0, "f"] = 1e6
        robust = build_step("robust_scale", {})
        z = build_step("zscore", {})
        assert robust.fit(spiked, CTX)["scale"]["f"] == pytest.approx(
            robust.fit(base, CTX)["scale"]["f"], rel=1e-2
        )
        assert z.fit(spiked, CTX)["std"]["f"] > 100 * z.fit(base, CTX)["std"]["f"]

    def test_no_dispersion_scales_by_one(self):
        state = build_step("robust_scale", {}).fit(_frame(f=[2.0, 2.0, 2.0]), CTX)
        assert state["scale"]["f"] == 1.0


class TestQuantileTransform:
    def test_a_uniform_training_column_maps_to_a_standard_normal(self):
        rng = np.random.default_rng(0)
        train = _frame(f=rng.uniform(size=20_000))
        step = build_step("quantile_transform", {})
        state = step.fit(train, CTX)
        out = step.transform(train, state, CTX)["f"].to_numpy()
        assert abs(out.mean()) < 0.02
        assert abs(out.std() - 1.0) < 0.02
        # A known point: the training 97.5th percentile lands near 1.96.
        point = step.transform(_frame(f=[np.quantile(train["f"], 0.975)]), state, CTX)
        assert point["f"].iloc[0] == pytest.approx(norm.ppf(0.975), abs=0.03)

    def test_monotone_and_nan_preserving(self):
        rng = np.random.default_rng(1)
        train = _frame(f=rng.standard_t(2, size=5000))
        step = build_step("quantile_transform", {})
        state = step.fit(train, CTX)
        probe = _frame(f=[-50.0, -1.0, 0.0, 1.0, np.nan, 50.0])
        out = step.transform(probe, state, CTX)["f"]
        finite = out.drop(index=4).to_numpy()
        assert np.all(np.diff(finite) >= 0)
        assert np.isnan(out.iloc[4])
        # Beyond the training range maps to the ends, and the ends are finite.
        assert np.isfinite(finite).all()
        assert finite[0] == out.min() and finite[-1] == out.max()

    def test_uniform_output_is_a_cdf(self):
        rng = np.random.default_rng(2)
        train = _frame(f=rng.normal(size=5000))
        step = build_step("quantile_transform", {"output": "uniform"})
        state = step.fit(train, CTX)
        out = step.transform(train, state, CTX)["f"].to_numpy()
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert abs(out.mean() - 0.5) < 0.01

    def test_a_flat_training_column_maps_to_the_middle(self):
        step = build_step("quantile_transform", {})
        state = step.fit(_frame(f=[5.0] * 50), CTX)
        assert len(state["quantiles"]["f"]) == 1
        out = step.transform(_frame(f=[5.0, 7.0]), state, CTX)["f"]
        assert out.tolist() == pytest.approx([0.0, 0.0])

    def test_the_grid_is_bounded_and_ties_are_collapsed(self):
        train = _frame(f=np.repeat([1.0, 2.0, 3.0], 100))
        state = build_step("quantile_transform", {"n_quantiles": 50}).fit(train, CTX)
        assert state["quantiles"]["f"] == [1.0, 2.0, 3.0]
        assert len(state["probabilities"]["f"]) == 3
        assert np.all(np.diff(state["probabilities"]["f"]) > 0)


class TestImpute:
    def test_a_missing_test_value_receives_the_training_median(self):
        """Planted: the training median is 10 and the test fold's own would
        be 100. Only one of those is a statistic the model may see."""
        train = _frame(f=[9.0, 10.0, 11.0])
        test = _frame(f=[100.0, np.nan, 100.0])
        state, _train_out, test_out = fit_and_apply_pipeline(
            [StepSpec(type="impute")], train, test, CTX, CTX
        )
        assert state["steps"][0]["state"]["fill"]["f"] == 10.0
        assert test_out["f"].tolist() == [100.0, 10.0, 100.0]

    def test_mean_and_constant(self):
        train = _frame(f=[1.0, 2.0, 6.0])
        assert build_step("impute", {"strategy": "mean"}).fit(train, CTX)["fill"]["f"] == 3.0
        assert (
            build_step("impute", {"strategy": "constant", "fill_value": -1.0}).fit(train, CTX)[
                "fill"
            ]["f"]
            == -1.0
        )

    def test_an_all_missing_training_column_falls_back_to_the_constant(self):
        state = build_step("impute", {"fill_value": 7.0}).fit(_frame(f=[np.nan, np.nan]), CTX)
        assert state["fill"]["f"] == 7.0

    def test_a_complete_column_is_untouched(self):
        train = _frame(f=[1.0, 2.0, 3.0])
        step = build_step("impute", {})
        pd.testing.assert_frame_equal(step.transform(train, step.fit(train, CTX), CTX), train)


class TestMissingIndicator:
    def test_one_indicator_per_input_column_marking_exactly_the_nan(self):
        frame = _frame(a=[1.0, np.nan, 3.0], b=[np.nan, np.nan, 1.0])
        step = build_step("missing_indicator", {})
        out = step.transform(frame, step.fit(frame, CTX), CTX)
        assert list(out.columns) == ["a", "b", "a__missing", "b__missing"]
        assert out["a__missing"].tolist() == [0.0, 1.0, 0.0]
        assert out["b__missing"].tolist() == [1.0, 1.0, 0.0]
        pd.testing.assert_frame_equal(out[["a", "b"]], frame)

    def test_the_column_set_does_not_depend_on_the_fold(self):
        """A complete training fold and a holed test fold produce the same
        columns; only the values differ."""
        train = _frame(a=[1.0, 2.0], b=[3.0, 4.0])
        test = _frame(a=[np.nan, 2.0], b=[3.0, 4.0])
        _state, train_out, test_out = fit_and_apply_pipeline(
            [StepSpec(type="missing_indicator")], train, test, CTX, CTX
        )
        assert list(train_out.columns) == list(test_out.columns)
        assert train_out["a__missing"].sum() == 0.0
        assert test_out["a__missing"].sum() == 1.0

    def test_it_is_stateless(self):
        assert get_preprocessor("missing_indicator").stateless is True


class TestPCAWhiten:
    def _correlated(self, n: int = 4000, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        cov = np.array([[4.0, 2.0, 0.0], [2.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
        values = rng.multivariate_normal(np.zeros(3), cov, size=n)
        return pd.DataFrame(values, columns=["x", "y", "z"])

    def test_whitened_output_has_identity_covariance(self):
        train = self._correlated()
        step = build_step("pca_whiten", {"n_components": 3})
        out = step.transform(train, step.fit(train, CTX), CTX)
        assert list(out.columns) == ["pc1", "pc2", "pc3"]
        np.testing.assert_allclose(np.cov(out.to_numpy().T), np.eye(3), atol=0.05)

    def test_fewer_components_keep_the_leading_variance(self):
        train = self._correlated()
        step = build_step("pca_whiten", {"n_components": 2, "whiten": False})
        state = step.fit(train, CTX)
        out = step.transform(train, state, CTX)
        assert list(out.columns) == ["pc1", "pc2"]
        assert out["pc1"].var() > out["pc2"].var()
        ratios = state["explained_variance_ratio"]
        assert ratios[0] > ratios[1] > 0 and sum(ratios) < 1.0

    def test_the_basis_is_fitted_on_train_and_applied_to_test(self):
        train = self._correlated(seed=0)
        test = self._correlated(seed=1) * 10.0  # a very different test covariance
        state, _train_out, test_out = fit_and_apply_pipeline(
            [StepSpec(type="pca_whiten", params={"n_components": 3})], train, test, CTX, CTX
        )
        # Same rotation and scale, so the test output is ten times as
        # dispersed as unit -- the training basis, not a test-fold one.
        assert test_out.to_numpy().std() == pytest.approx(10.0, rel=0.1)
        np.testing.assert_allclose(
            np.asarray(state["steps"][0]["state"]["components"]),
            np.asarray(build_step("pca_whiten", {"n_components": 3}).fit(train, CTX)["components"]),
        )

    def test_signs_are_reproducible(self):
        train = self._correlated()
        components = build_step("pca_whiten", {"n_components": 3}).fit(train, CTX)["components"]
        for row in components:
            assert row[int(np.argmax(np.abs(row)))] > 0

    def test_too_many_components_and_nan_are_refused(self):
        train = self._correlated()
        with pytest.raises(ValidationError, match="exceeds the 3"):
            build_step("pca_whiten", {"n_components": 4}).fit(train, CTX)
        holed = train.copy()
        holed.iloc[0, 0] = np.nan
        with pytest.raises(ValidationError, match="impute"):
            build_step("pca_whiten", {"n_components": 2}).fit(holed, CTX)

    def test_not_column_wise(self):
        assert get_preprocessor("pca_whiten").column_wise is False

    def test_the_state_round_trips(self):
        train = self._correlated()
        state, out = fit_pipeline(
            [StepSpec(type="pca_whiten", params={"n_components": 2})], train, CTX
        )
        again = apply_pipeline(json.loads(json.dumps(state)), train, CTX)
        pd.testing.assert_frame_equal(again, out)


class TestTheEngineLabelsByThePipelineOutput:
    def _spec(self, steps):
        return ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={"alpha": ALPHA}),
            validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
            preprocessing=PreprocessingSpec(steps=steps),
            random_seed=1,
        )

    def test_a_pca_model_is_fitted_on_components_and_says_so(self, patched_multi_factory):
        model_id, dataset = _register(
            _dataset_spec(),
            self._spec(
                [
                    StepSpec(type="zscore"),
                    StepSpec(type="pca_whiten", params={"n_components": 2}),
                ]
            ),
            "ds_pca",
        )
        manifest = load_manifest(model_id)
        assert manifest.feature_ids == dataset["feature_ids"]
        assert manifest.model_input_columns == ["pc1", "pc2"]
        assert set(manifest.feature_importance_summary) == {"pc1", "pc2"}
        result = score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)
        predictions = _artifacts.load_artifact(result["predictions_uri"])["prediction"]
        assert result["n_entities"] == len(UNIVERSE)
        assert np.isfinite(predictions.to_numpy()).all()

    def test_an_indicator_model_doubles_its_input_columns(self, patched_multi_factory):
        model_id, dataset = _register(
            _dataset_spec(),
            self._spec([StepSpec(type="missing_indicator"), StepSpec(type="zscore")]),
            "ds_indicator",
        )
        manifest = load_manifest(model_id)
        features = dataset["feature_ids"]
        assert manifest.model_input_columns == features + [f"{c}__missing" for c in features]
        assert set(manifest.feature_importance_summary) == set(manifest.model_input_columns)

    def test_a_column_preserving_model_records_the_same_columns(self, patched_multi_factory):
        model_id, dataset = _register(
            _dataset_spec(),
            self._spec([StepSpec(type="robust_scale")]),
            "ds_robust",
        )
        manifest = load_manifest(model_id)
        assert manifest.model_input_columns == dataset["feature_ids"]


class TestTheCatalog:
    def test_all_eight_steps_are_registered_with_their_flags(self):
        reported = {e["id"]: e for e in modeling_capabilities()["preprocessing"]["steps"]}
        assert set(reported) == {
            "winsorize",
            "zscore",
            "cross_sectional_standardize",
            "robust_scale",
            "quantile_transform",
            "missing_indicator",
            "impute",
            "pca_whiten",
        }
        assert reported["pca_whiten"]["column_wise"] is False
        assert reported["missing_indicator"]["stateless"] is True
        assert reported["impute"]["params"] == ["fill_value", "strategy"]
