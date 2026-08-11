"""
Regression tests for FeatureSpec.alias and FeatureDefinition.requires
enforcement.

Two feature-system gaps: the same feature could not be requested at two
parameter settings (momentum(20) + momentum(252) is a completely ordinary
multi-horizon spec, and it was rejected outright because the panel keyed one
column per feature id), and `requires` was informational only, so a provider
frame missing a needed column surfaced as a raw KeyError from inside
whichever feature touched it first.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.registry.model_registry import load_manifest
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)


def _spec(features) -> DatasetSpec:
    return DatasetSpec(
        universe=["AAA", "BBB"],
        start="2022-01-01",
        end="2023-12-31",
        features=features,
        target=TargetSpec(horizon=5),
    )


class TestMultiHorizonFeature:
    def test_same_feature_at_two_horizons(self, patched_multi_factory):
        built = build_dataset(
            _spec(
                [
                    FeatureSpec(
                        id="market.momentum", params={"lookback": 20}, alias="mom_20"
                    ),
                    FeatureSpec(
                        id="market.momentum", params={"lookback": 60}, alias="mom_60"
                    ),
                ]
            )
        )
        assert built["feature_ids"] == ["mom_20", "mom_60"]
        assert {"mom_20", "mom_60"} <= set(built["panel"].columns)

    def test_the_two_columns_hold_different_values(self, patched_multi_factory):
        """
        The point of the feature, not just the plumbing: each alias must
        carry its OWN parameterization, not two copies of one computation.
        """
        built = build_dataset(
            _spec(
                [
                    FeatureSpec(
                        id="market.momentum", params={"lookback": 20}, alias="mom_20"
                    ),
                    FeatureSpec(
                        id="market.momentum", params={"lookback": 60}, alias="mom_60"
                    ),
                ]
            )
        )
        panel = built["panel"]
        assert not np.allclose(panel["mom_20"], panel["mom_60"])

    def test_alias_flows_into_the_trained_model(self, patched_multi_factory):
        built = build_dataset(
            _spec(
                [
                    FeatureSpec(
                        id="market.momentum", params={"lookback": 20}, alias="mom_20"
                    ),
                    FeatureSpec(
                        id="market.momentum", params={"lookback": 60}, alias="mom_60"
                    ),
                ]
            )
        )
        model_spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
            validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
            random_seed=1,
        )
        result = run_experiment(built, model_spec, dataset_id="ds_alias")
        manifest = load_manifest(result["model_id"])
        assert manifest.feature_ids == ["mom_20", "mom_60"]
        # Importance is reported per alias, so the two horizons are
        # distinguishable in the output rather than collapsed.
        assert set(result["feature_importance_summary"]) == {"mom_20", "mom_60"}

    def test_unaliased_feature_keeps_its_id_as_the_column(self, patched_multi_factory):
        """Backwards compatibility: an ordinary single-use spec is
        unchanged."""
        built = build_dataset(_spec([FeatureSpec(id="technical.rsi")]))
        assert built["feature_ids"] == ["technical.rsi"]

    def test_mixed_aliased_and_plain(self, patched_multi_factory):
        built = build_dataset(
            _spec(
                [
                    FeatureSpec(id="technical.rsi"),
                    FeatureSpec(
                        id="market.momentum", params={"lookback": 20}, alias="mom_20"
                    ),
                    FeatureSpec(
                        id="market.momentum", params={"lookback": 60}, alias="mom_60"
                    ),
                ]
            )
        )
        assert built["feature_ids"] == ["technical.rsi", "mom_20", "mom_60"]


def _ohlcv_without(column: str, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(n, 1e6),
        },
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )
    return frame.drop(columns=[column])


class TestRequiresEnforcement:
    """
    FeatureDefinition.requires was informational. A provider returning a
    frame without 'Volume' produced a raw KeyError from inside whichever
    feature touched it first — naming the column, but not the feature, the
    symbol, or the fact that the provider was at fault.
    """

    @pytest.fixture
    def provider_missing_volume(self, monkeypatch) -> MagicMock:
        provider = MagicMock()
        provider.get_ohlcv.side_effect = lambda s, start, end: _ohlcv_without("Volume")
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
        return provider

    def test_missing_required_column_named_clearly(self, provider_missing_volume):
        with pytest.raises(ValidationError) as excinfo:
            build_dataset(_spec([FeatureSpec(id="volume.mfi")]))
        message = str(excinfo.value)
        assert "missing column(s)" in message
        assert "Volume" in message
        assert "volume.mfi" in message, "the FEATURE must be named, not just the column"

    def test_error_names_the_alias_when_one_is_used(self, provider_missing_volume):
        with pytest.raises(ValidationError, match="my_mfi"):
            build_dataset(_spec([FeatureSpec(id="volume.mfi", alias="my_mfi")]))

    def test_features_not_needing_the_column_are_unaffected(
        self, provider_missing_volume
    ):
        """Only features that actually declare the column should fail."""
        built = build_dataset(_spec([FeatureSpec(id="market.momentum")]))
        assert not built["panel"].empty
