"""
The deployed estimator is fitted under the transform the folds were
validated under.

It was not. `_preprocess` standardizes each fold within its date when
`preprocessing.normalization='cross_sectional'` is asked for; the full-panel
refit fitted the pooled winsorize/zscore statistics regardless, persisted
them, and score_model applied them. Measured on a six-entity panel, ridge,
three features: the deployed estimator's predictions under the two
transforms agreed at Spearman 0.84, and the manifest recorded nothing that
would let a reader notice.

Every test here plants its oracle. The refit is deterministic, so "fitted
under the fold transform" is checked by refitting the same estimator by
hand on that transform and comparing coefficients exactly -- not by asking
whether the predictions look reasonable.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.agent.models import InspectModelInput
from standard_quant_tools.modeling.agent.tools import inspect_model
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.features.transforms import (
    apply_preprocessing,
    fit_preprocessing,
    standardize_cross_sectional,
)
from standard_quant_tools.modeling.registry.model_registry import (
    load_manifest,
    load_model,
    load_preprocessing_stats,
)
from standard_quant_tools.modeling.scoring import score_model
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    PreprocessingSpec,
    TargetSpec,
    ValidationSpec,
)

# Six entities: with n <= 9 names the largest attainable |z| within a date
# is (n - 1) / sqrt(n) < 3, so the default 3-sigma clip never fires and the
# cross-sectional transform is exactly a per-date standardization. That is
# what makes the mean-prediction identity below exact.
UNIVERSE = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
ALPHA = 1.0


def _dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=[
            FeatureSpec(id="technical.rsi"),
            FeatureSpec(id="market.momentum"),
            FeatureSpec(id="risk.realized_volatility"),
        ],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )


def _model_spec(normalization: str) -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": ALPHA}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        preprocessing=PreprocessingSpec(normalization=normalization),
        random_seed=1,
    )


def _register(spec: DatasetSpec, model_spec: ModelSpec, dataset_id: str):
    """build + run the way the agent tool does, persisting the spec the
    scoring path needs; returns (model_id, dataset dict)."""
    built = build_dataset(spec)
    panel_uri = _artifacts.save_artifact(
        built["panel"], run_id=dataset_id, name="panel"
    )
    _artifacts.save_json(
        _artifacts.run_dir(dataset_id), "dataset_spec", spec.model_dump()
    )
    dataset = {
        "panel": built["panel"],
        "feature_ids": built["feature_ids"],
        "target_id": built["target_id"],
        "data_hash": built["data_hash"],
        "spec_hash": built["spec_hash"],
        "dataset_spec": spec.model_dump(),
    }
    del panel_uri
    return (
        run_experiment(dataset, model_spec, dataset_id=dataset_id)["model_id"],
        dataset,
    )


@pytest.fixture
def cross_sectional_model(patched_multi_factory):
    return _register(_dataset_spec(), _model_spec("cross_sectional"), "ds_cs")


@pytest.fixture
def pooled_model(patched_multi_factory):
    return _register(_dataset_spec(), _model_spec("pooled"), "ds_pooled")


class TestTheRefitHonoursTheValidatedTransform:
    def test_cross_sectional_model_is_fitted_on_the_cross_sectional_panel(
        self, cross_sectional_model
    ):
        model_id, dataset = cross_sectional_model
        panel, features = dataset["panel"], dataset["feature_ids"]
        expected = Ridge(alpha=ALPHA).fit(
            standardize_cross_sectional(
                panel[features], panel["date"].to_numpy(), 3.0
            ).to_numpy(),
            panel["target"].to_numpy(),
        )
        deployed = load_model(model_id)
        np.testing.assert_allclose(deployed.coef_, expected.coef_, rtol=0, atol=1e-12)
        np.testing.assert_allclose(deployed.intercept_, expected.intercept_, atol=1e-12)

    def test_and_not_on_the_pooled_panel(self, cross_sectional_model):
        """The other fit, which is what the refit used to produce. Kept so
        the test above cannot pass vacuously on a panel where the two
        transforms happen to coincide."""
        model_id, dataset = cross_sectional_model
        panel, features = dataset["panel"], dataset["feature_ids"]
        pooled = Ridge(alpha=ALPHA).fit(
            apply_preprocessing(
                panel[features], fit_preprocessing(panel[features])
            ).to_numpy(),
            panel["target"].to_numpy(),
        )
        assert not np.allclose(load_model(model_id).coef_, pooled.coef_, atol=1e-9)

    def test_pooled_model_is_fitted_on_the_pooled_panel(self, pooled_model):
        model_id, dataset = pooled_model
        panel, features = dataset["panel"], dataset["feature_ids"]
        stats = fit_preprocessing(panel[features])
        expected = Ridge(alpha=ALPHA).fit(
            apply_preprocessing(panel[features], stats).to_numpy(),
            panel["target"].to_numpy(),
        )
        np.testing.assert_allclose(
            load_model(model_id).coef_, expected.coef_, atol=1e-12
        )
        assert load_preprocessing_stats(model_id) == stats

    def test_the_manifest_says_which_transform_the_estimator_expects(
        self, cross_sectional_model, pooled_model
    ):
        cs = load_manifest(cross_sectional_model[0])
        assert cs.preprocessing["normalization"] == "cross_sectional"
        # The RESOLVED pipeline, so a reader sees what ran.
        assert [s["type"] for s in cs.preprocessing["steps"]] == [
            "cross_sectional_standardize"
        ]
        assert cs.preprocessing["steps"][0]["params"] == {"clip_sigma": 3.0}
        # The legacy file cannot express this pipeline and says so, rather
        # than writing `{}`, which apply_preprocessing read as the identity.
        legacy = load_preprocessing_stats(cross_sectional_model[0])
        assert legacy["legacy"] is False and "preprocessing_state" in legacy["note"]
        pooled = load_manifest(pooled_model[0])
        assert pooled.preprocessing["normalization"] == "pooled"
        assert [s["type"] for s in pooled.preprocessing["steps"]] == [
            "winsorize",
            "zscore",
        ]
        view = inspect_model(
            InspectModelInput(model_id=cross_sectional_model[0], view="validation")
        )
        assert view.data["preprocessing"]["normalization"] == "cross_sectional"


class TestScoringAppliesTheValidatedTransform:
    def test_cross_sectional_predictions_average_to_the_intercept(
        self, cross_sectional_model
    ):
        """
        A linear model on features standardized within the date predicts,
        on average over that date's entities, exactly its intercept: the
        standardized features have mean zero across the cross-section, so
        the coefficient terms cancel. That identity holds ONLY if scoring
        standardized within the date -- under the pooled statistics the
        cross-section's mean feature is whatever the day happened to be.
        """
        model_id, _dataset = cross_sectional_model
        result = score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)
        assert result["n_entities"] == len(UNIVERSE)
        predictions = _artifacts.load_artifact(result["predictions_uri"])["prediction"]
        deployed = load_model(model_id)
        assert abs(float(predictions.mean()) - float(deployed.intercept_)) < 1e-9

    def test_the_pooled_path_does_not_satisfy_that_identity(self, pooled_model):
        """The null: for a pooled model the identity is not expected, and
        if it held by accident the test above would be proving nothing."""
        model_id, _dataset = pooled_model
        result = score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)
        predictions = _artifacts.load_artifact(result["predictions_uri"])["prediction"]
        deployed = load_model(model_id)
        assert abs(float(predictions.mean()) - float(deployed.intercept_)) > 1e-6

    @staticmethod
    def _make_legacy(model_id: str) -> None:
        """
        The shape a model registered before the stop-gap had: a statistics
        file, no `preprocessing` field in the manifest, and -- since the
        pipeline registry -- no `preprocessing_state.json` either. The
        manifest is not self-hashed, so the field can be removed; the
        state file's hash is removed with it so the manifest stays
        consistent with the directory.
        """
        directory = _artifacts.run_dir(model_id)
        (directory / "preprocessing_state.json").unlink()
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        del manifest["preprocessing"]
        del manifest["content_hashes"]["preprocessing_state.json"]
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_a_legacy_cross_sectional_model_is_refused_not_guessed(
        self, cross_sectional_model
    ):
        """
        A manifest from before the field existed, whose bundled spec says
        cross_sectional, describes an estimator that was refit on the
        pooled statistics: no transform applied now reproduces a validated
        pipeline.
        """
        model_id, _dataset = cross_sectional_model
        self._make_legacy(model_id)
        with pytest.raises(ValidationError, match="predates the refit"):
            score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)

    def test_a_legacy_pooled_model_still_scores(self, pooled_model):
        model_id, _dataset = pooled_model
        self._make_legacy(model_id)
        result = score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)
        assert result["n_entities"] == len(UNIVERSE)
