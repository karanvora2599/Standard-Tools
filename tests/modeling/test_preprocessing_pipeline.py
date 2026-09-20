"""
Preprocessing as a registry of steps and a pipeline of fitted state.

The invariants, in the order the tests plant them:

  * the default spec resolves to today's transform, byte for byte -- the
    generic steps equal `fit_preprocessing`/`apply_preprocessing` and the
    fused path equals the generic one;
  * a state is fitted on TRAINING rows and applied to test rows, and a
    planted test-only outlier cannot move it;
  * the state is plain JSON and reproduces the transform after a round trip;
  * the engine runs the resolved pipeline on every fold and on the refit,
    persists the state, and scoring applies it -- with a legacy model
    still scoring through the statistics file;
  * the registry refuses what every other registry refuses.
"""

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError
from sklearn.linear_model import Ridge

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.capabilities import modeling_capabilities
from standard_quant_tools.modeling.estimators.bounds import EstimatorParamSchema
from standard_quant_tools.modeling.features import transforms
from standard_quant_tools.modeling.features.transforms import (
    apply_preprocessing,
    fit_preprocessing,
    standardize_cross_sectional,
)
from standard_quant_tools.modeling.preprocessing import (
    PREPROCESSOR_REGISTRY,
    FoldContext,
    Preprocessor,
    PreprocessorDefinition,
    apply_pipeline,
    build_step,
    fit_and_apply_pipeline,
    fit_pipeline,
    legacy_stats,
    register_preprocessor,
    step_types,
)
from standard_quant_tools.modeling.registry.model_registry import (
    load_manifest,
    load_model,
    load_preprocessing_state,
    load_preprocessing_stats,
)
from standard_quant_tools.modeling.scoring import score_model
from standard_quant_tools.modeling.specs import (
    EstimatorSpec,
    ModelSpec,
    PreprocessingSpec,
    StepSpec,
    ValidationSpec,
)

from .test_deployed_pipeline import ALPHA, UNIVERSE, _dataset_spec, _register


def _panel(n_rows: int = 400, n_cols: int = 4, seed: int = 0) -> pd.DataFrame:
    """A long panel with fat tails, a few NaN and dates shared across
    entities, so every step has something to do."""
    rng = np.random.default_rng(seed)
    values = rng.standard_t(3, size=(n_rows, n_cols)) * [1.0, 100.0, 0.01, 5.0]
    values[rng.random(values.shape) < 0.02] = np.nan
    dates = np.repeat(pd.date_range("2021-01-04", periods=n_rows // 4), 4)
    frame = pd.DataFrame(values, columns=[f"f{i}" for i in range(n_cols)])
    frame.index = pd.RangeIndex(n_rows)
    return frame, FoldContext(dates=dates, entities=np.tile(np.arange(4), n_rows // 4))


def _ctx(frame: pd.DataFrame, n_entities: int = 4) -> FoldContext:
    n = len(frame)
    return FoldContext(
        dates=np.repeat(pd.date_range("2021-01-04", periods=n // n_entities), n_entities),
        entities=np.tile(np.arange(n_entities), n // n_entities),
    )


# ── The spec resolves to what ran before ────────────────────────────────


class TestTheSpecResolves:
    def test_default_is_the_pooled_pair(self):
        steps = PreprocessingSpec().resolved_steps
        assert [(s.type, s.params) for s in steps] == [
            ("winsorize", {"lower": 0.01, "upper": 0.99}),
            ("zscore", {}),
        ]

    def test_cross_sectional_is_one_stateless_step(self):
        steps = PreprocessingSpec(normalization="cross_sectional", clip_sigma=2.5).resolved_steps
        assert [(s.type, s.params) for s in steps] == [
            ("cross_sectional_standardize", {"clip_sigma": 2.5})
        ]

    def test_explicit_steps_win(self):
        spec = PreprocessingSpec(steps=[StepSpec(type="zscore")])
        assert step_types(spec.resolved_steps) == ["zscore"]

    def test_steps_beside_a_non_default_scheme_are_refused(self):
        with pytest.raises(PydanticValidationError, match="second claim"):
            PreprocessingSpec(
                steps=[StepSpec(type="zscore")], normalization="cross_sectional"
            )
        with pytest.raises(PydanticValidationError, match="clip_sigma"):
            PreprocessingSpec(steps=[StepSpec(type="zscore")], clip_sigma=2.0)

    def test_a_steps_spec_survives_its_own_serialization(self):
        spec = PreprocessingSpec(
            steps=[
                StepSpec(type="winsorize", params={"lower": 0.05, "upper": 0.95}),
                StepSpec(type="cross_sectional_standardize", params={"clip_sigma": 2.0}),
            ]
        )
        again = PreprocessingSpec(**spec.model_dump())
        assert again == spec
        assert again.resolved_dump()["steps"] == [s.model_dump() for s in spec.steps]

    def test_the_manifest_form_carries_the_resolved_pipeline(self):
        dumped = PreprocessingSpec(normalization="cross_sectional").resolved_dump()
        assert dumped["normalization"] == "cross_sectional"
        assert [s["type"] for s in dumped["steps"]] == ["cross_sectional_standardize"]

    def test_an_unknown_step_is_refused_at_the_boundary(self):
        with pytest.raises(PydanticValidationError, match="unknown preprocessing step"):
            StepSpec(type="robust_scale_x")

    def test_a_parameter_outside_its_bound_is_refused(self):
        with pytest.raises(PydanticValidationError, match="exceeds the maximum"):
            StepSpec(type="winsorize", params={"upper": 1.5})
        with pytest.raises(PydanticValidationError, match="does not accept"):
            StepSpec(type="zscore", params={"ddof": 0})

    def test_a_pipeline_is_bounded_in_length(self):
        with pytest.raises(PydanticValidationError):
            PreprocessingSpec(steps=[StepSpec(type="zscore")] * 17)


# ── The generic steps ARE the old functions ─────────────────────────────


class TestTheStepsReproduceTheOldTransform:
    @pytest.mark.parametrize("native", [True, False])
    def test_winsorize_then_zscore_equals_fit_and_apply_preprocessing(
        self, monkeypatch, native
    ):
        if native and not transforms.HAS_CPP:
            pytest.skip("native extension not built")
        monkeypatch.setattr(transforms, "HAS_CPP", native)
        frame, ctx = _panel()
        train, test = frame.iloc[:300], frame.iloc[300:]
        stats = fit_preprocessing(train)
        expected_train = apply_preprocessing(train, stats)
        expected_test = apply_preprocessing(test, stats)

        # The generic steps, built directly so the fused shortcut is not in
        # the picture.
        winsor = build_step("winsorize", {"lower": 0.01, "upper": 0.99})
        zscore = build_step("zscore", {})
        w_state = winsor.fit(train, ctx)
        w_train = winsor.transform(train, w_state, ctx)
        z_state = zscore.fit(w_train, ctx)
        got_train = zscore.transform(w_train, z_state, ctx)
        got_test = zscore.transform(winsor.transform(test, w_state, ctx), z_state, ctx)

        pd.testing.assert_frame_equal(got_train, expected_train, atol=1e-12, rtol=0)
        pd.testing.assert_frame_equal(got_test, expected_test, atol=1e-12, rtol=0)
        for column in train.columns:
            assert w_state["lo"][column] == pytest.approx(stats[column]["lo"], abs=1e-12)
            assert z_state["std"][column] == pytest.approx(stats[column]["std"], abs=1e-12)

    def test_the_fused_pipeline_equals_the_generic_one(self):
        frame, ctx = _panel(seed=3)
        train, test = frame.iloc[:300], frame.iloc[300:]
        default = PreprocessingSpec().resolved_steps
        state, fused_train, fused_test = fit_and_apply_pipeline(default, train, test, ctx, ctx)
        assert step_types(state) == ["winsorize", "zscore"]

        # A non-default winsorize bound forces the generic path; set it to
        # the default values by hand through the generic classes instead.
        winsor = build_step("winsorize", {"lower": 0.01, "upper": 0.99})
        zscore = build_step("zscore", {})
        w_state = winsor.fit(train, ctx)
        z_state = zscore.fit(winsor.transform(train, w_state, ctx), ctx)
        generic_test = zscore.transform(winsor.transform(test, w_state, ctx), z_state, ctx)
        pd.testing.assert_frame_equal(fused_test, generic_test, atol=1e-12, rtol=0)

    def test_cross_sectional_step_is_the_function(self):
        frame, ctx = _panel(seed=5)
        state, out = fit_pipeline(
            PreprocessingSpec(normalization="cross_sectional").resolved_steps, frame, ctx
        )
        assert state["steps"][0]["state"] == {}
        pd.testing.assert_frame_equal(
            out, standardize_cross_sectional(frame, ctx.dates, 3.0), atol=1e-12, rtol=0
        )

    def test_legacy_stats_are_the_old_file_for_the_default_and_empty_otherwise(self):
        frame, ctx = _panel(seed=7)
        state, _ = fit_pipeline(PreprocessingSpec().resolved_steps, frame, ctx)
        assert legacy_stats(state) == fit_preprocessing(frame)
        state, _ = fit_pipeline(
            [StepSpec(type="winsorize", params={"lower": 0.05, "upper": 0.95})], frame, ctx
        )
        assert legacy_stats(state) == {}


# ── The state: fitted on train, applied anywhere, carried as JSON ────────


class TestTheStateIsTheArtifact:
    def test_a_test_only_outlier_cannot_move_the_fitted_bounds(self):
        frame, ctx = _panel(seed=11)
        train, test = frame.iloc[:300].copy(), frame.iloc[300:].copy()
        test.iloc[0, 0] = 1e6  # planted, far beyond anything in train
        state, _train_out, test_out = fit_and_apply_pipeline(
            [StepSpec(type="winsorize", params={"lower": 0.05, "upper": 0.95})],
            train,
            test,
            ctx,
            ctx,
        )
        hi = state["steps"][0]["state"]["hi"]["f0"]
        assert hi == pytest.approx(float(train["f0"].quantile(0.95)))
        assert test_out.iloc[0, 0] == hi

    def test_the_state_round_trips_through_json(self):
        frame, ctx = _panel(seed=13)
        steps = [
            StepSpec(type="winsorize", params={"lower": 0.02, "upper": 0.98}),
            StepSpec(type="zscore"),
        ]
        state, out = fit_pipeline(steps, frame, ctx)
        reloaded = json.loads(json.dumps(state))
        pd.testing.assert_frame_equal(apply_pipeline(reloaded, frame, ctx), out)

    def test_columns_must_match_the_fitted_ones(self):
        frame, ctx = _panel(seed=17)
        state, _ = fit_pipeline(PreprocessingSpec().resolved_steps, frame, ctx)
        with pytest.raises(ValidationError, match="different columns"):
            apply_pipeline(state, frame.rename(columns={"f0": "g0"}), ctx)
        with pytest.raises(ValidationError, match="different columns"):
            apply_pipeline(state, frame[list(reversed(frame.columns))], ctx)

    def test_an_unknown_state_version_is_refused(self):
        frame, ctx = _panel(seed=19)
        state, _ = fit_pipeline(PreprocessingSpec().resolved_steps, frame, ctx)
        state["version"] = 99
        with pytest.raises(ValidationError, match="version"):
            apply_pipeline(state, frame, ctx)

    def test_nan_survives_every_built_in_step(self):
        frame, ctx = _panel(seed=23)
        missing = frame.isna()
        for steps in (
            PreprocessingSpec().resolved_steps,
            PreprocessingSpec(normalization="cross_sectional").resolved_steps,
        ):
            _state, out = fit_pipeline(steps, frame, ctx)
            pd.testing.assert_frame_equal(out.isna(), missing)


# ── The engine and the registry run it ──────────────────────────────────


class TestTheEngineRunsThePipeline:
    @pytest.fixture
    def steps_model(self, patched_multi_factory):
        spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={"alpha": ALPHA}),
            validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
            preprocessing=PreprocessingSpec(
                steps=[
                    StepSpec(type="winsorize", params={"lower": 0.05, "upper": 0.95}),
                    StepSpec(type="zscore"),
                ]
            ),
            random_seed=1,
        )
        return _register(_dataset_spec(), spec, "ds_steps")

    def test_explicit_steps_run_and_are_reported(self, steps_model):
        model_id, _dataset = steps_model
        manifest = load_manifest(model_id)
        assert manifest.validation_report["preprocessing_steps"] == ["winsorize", "zscore"]
        assert [s["type"] for s in manifest.preprocessing["steps"]] == ["winsorize", "zscore"]
        assert manifest.preprocessing["steps"][0]["params"] == {"lower": 0.05, "upper": 0.95}
        assert "preprocessing_state.json" in manifest.content_hashes

    def test_the_deployed_estimator_is_fitted_on_the_persisted_state(self, steps_model):
        """Planted oracle: refit the same ridge by hand on the state applied
        to the panel and compare coefficients exactly."""
        model_id, dataset = steps_model
        panel, features = dataset["panel"], dataset["feature_ids"]
        state = load_preprocessing_state(model_id)
        expected = Ridge(alpha=ALPHA).fit(
            apply_pipeline(state, panel[features], FoldContext.from_frame(panel)).to_numpy(),
            panel["target"].to_numpy(),
        )
        np.testing.assert_allclose(load_model(model_id).coef_, expected.coef_, atol=1e-12)
        # And the 5th percentile bound really is the 5th, not the default 1st.
        assert state["steps"][0]["state"]["lo"][features[0]] == pytest.approx(
            float(panel[features[0]].quantile(0.05))
        )
        assert load_preprocessing_stats(model_id) == {}

    def test_the_default_model_still_writes_the_legacy_statistics(self, patched_multi_factory):
        model_id, dataset = _register(
            _dataset_spec(),
            ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": ALPHA}),
                validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
                random_seed=1,
            ),
            "ds_default",
        )
        panel, features = dataset["panel"], dataset["feature_ids"]
        assert load_preprocessing_stats(model_id) == fit_preprocessing(panel[features])
        assert step_types(load_preprocessing_state(model_id)) == ["winsorize", "zscore"]

    def test_scoring_applies_the_state(self, steps_model):
        model_id, _dataset = steps_model
        result = score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)
        assert result["n_entities"] == len(UNIVERSE)
        predictions = _artifacts.load_artifact(result["predictions_uri"])["prediction"]
        assert np.isfinite(predictions.to_numpy()).all()

    def test_an_edited_state_is_refused(self, steps_model):
        model_id, _dataset = steps_model
        path = _artifacts.run_dir(model_id) / "preprocessing_state.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        state["steps"][1]["state"]["mean"] = {
            k: v + 1.0 for k, v in state["steps"][1]["state"]["mean"].items()
        }
        path.write_text(json.dumps(state), encoding="utf-8")
        with pytest.raises(ValidationError, match="preprocessing_state.json has changed"):
            score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)

    def test_a_model_without_a_state_file_scores_through_the_statistics(
        self, patched_multi_factory
    ):
        """The shape every model registered before the pipeline existed
        has: a statistics file and a manifest scheme, no state file."""
        model_id, _dataset = _register(
            _dataset_spec(),
            ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": ALPHA}),
                validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
                random_seed=1,
            ),
            "ds_legacy_state",
        )
        directory = _artifacts.run_dir(model_id)
        (directory / "preprocessing_state.json").unlink()
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["content_hashes"]["preprocessing_state.json"]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        assert load_preprocessing_state(model_id) is None
        result = score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)
        assert result["n_entities"] == len(UNIVERSE)


# ── The registry behaves like the other registries ──────────────────────


class _Noop(Preprocessor):
    id = "test_noop"
    stateless = True

    def fit(self, X, ctx):
        return {}

    def transform(self, X, state, ctx):
        return X


class TestTheRegistry:
    def test_duplicate_id_refused_without_overwrite(self):
        definition = PreprocessorDefinition(
            id="test_noop", description="nothing", cls=_Noop, schema=EstimatorParamSchema()
        )
        register_preprocessor(definition)
        try:
            with pytest.raises(ValidationError, match="already registered"):
                register_preprocessor(definition)
            register_preprocessor(definition, overwrite=True)
        finally:
            PREPROCESSOR_REGISTRY.pop("test_noop", None)

    def test_the_class_must_declare_the_same_id(self):
        with pytest.raises(ValidationError, match="must agree"):
            register_preprocessor(
                PreprocessorDefinition(
                    id="another_name", description="", cls=_Noop, schema=EstimatorParamSchema()
                )
            )

    def test_capabilities_list_the_steps_from_the_registry(self):
        reported = modeling_capabilities()["preprocessing"]
        assert reported["normalization"] == ["pooled", "cross_sectional"]
        by_id = {entry["id"]: entry for entry in reported["steps"]}
        assert set(by_id) >= {"winsorize", "zscore", "cross_sectional_standardize"}
        assert by_id["cross_sectional_standardize"]["stateless"] is True
        assert by_id["winsorize"]["params"] == ["lower", "upper"]
