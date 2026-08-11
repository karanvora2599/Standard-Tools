"""Tests for modeling.scoring.score_model: missing_entities reporting,
as_of validation, and that the reconstructed scoring DatasetSpec is
actually re-validated (not silently bypassed via model_copy)."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.scoring import score_model
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)

from .conftest import make_ohlcv


def _dataset_spec(**overrides) -> DatasetSpec:
    defaults = dict(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )
    defaults.update(overrides)
    return DatasetSpec(**defaults)


def _train_a_model_with_spec(
    spec: DatasetSpec, dataset_id: str = "ds_scoring_test"
) -> str:
    """Builds+trains a model exactly the way build_model_dataset +
    run_model_experiment would, but persists dataset_spec.json by hand
    (the agent tool normally does this; scoring.py depends on it
    existing) -- returns the trained model_id."""
    from pathlib import Path

    from standard_quant_tools.modeling import artifacts as _artifacts

    built = build_dataset(spec)
    panel_uri = _artifacts.save_artifact(
        built["panel"], run_id=dataset_id, name="panel"
    )
    directory = Path(panel_uri).parent
    _artifacts.save_json(directory, "dataset_spec", spec.model_dump())

    model_spec = ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )
    dataset = {
        "panel": built["panel"],
        "feature_ids": built["feature_ids"],
        "target_id": built["target_id"],
        "data_hash": built["data_hash"],
        # Mirrors what the run_model_experiment agent tool passes, so the
        # registered model bundles (and content-verifies) its own copy of
        # the training spec rather than reading the dataset directory's.
        "spec_hash": built["spec_hash"],
        "dataset_spec": spec.model_dump(),
    }
    result = run_experiment(dataset, model_spec, dataset_id=dataset_id)
    return result["model_id"]


def _train_a_model(patched_multi_factory) -> str:
    return _train_a_model_with_spec(_dataset_spec())


class TestScoreModelReliability:
    def test_missing_entities_reported_not_silently_dropped(self, monkeypatch):
        """CCC gets only 8 bars of OHLCV -- not enough for RSI's 14-bar
        lookback to ever produce a non-NaN value, so CCC contributes zero
        rows to both the training panel and the scoring snapshot. It
        must show up in missing_entities, not be silently absent from a
        result that otherwise looks like it succeeded for everyone."""

        def _get_ohlcv(symbol, start, end):
            if symbol == "CCC":
                return make_ohlcv(symbol, n=8)
            return make_ohlcv(symbol)

        provider = MagicMock()
        provider.get_ohlcv.side_effect = _get_ohlcv
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        spec = _dataset_spec(
            features=[FeatureSpec(id="technical.rsi")], universe=["AAA", "BBB"]
        )
        model_id = _train_a_model_with_spec(spec)

        result = score_model(
            model_id=model_id, as_of="2023-12-29", universe=["AAA", "BBB", "CCC"]
        )
        assert result["missing_entities"] == ["CCC"]
        assert result["n_entities"] == 2

    def test_malformed_as_of_raises_validation_error_not_raw_pandas_error(
        self, patched_multi_factory
    ):
        model_id = _train_a_model(patched_multi_factory)
        with pytest.raises(ValidationError, match="not a valid date"):
            score_model(model_id=model_id, as_of="not-a-date", universe=["AAA"])

    def test_scoring_spec_reconstruction_is_actually_validated(
        self, patched_multi_factory
    ):
        """Regression test for the model_copy(update=...) bug: passing a
        universe with duplicate symbols to score_model must be rejected
        by DatasetSpec's validator (raised as a pydantic ValidationError,
        the same as any other invalid DatasetSpec construction), not
        silently accepted the way model_copy(update=...) would have."""
        model_id = _train_a_model(patched_multi_factory)
        with pytest.raises(PydanticValidationError, match="duplicate symbols"):
            score_model(model_id=model_id, as_of="2023-12-29", universe=["AAA", "AAA"])

    def test_successful_score_has_no_missing_entities(self, patched_multi_factory):
        model_id = _train_a_model(patched_multi_factory)
        result = score_model(
            model_id=model_id, as_of="2023-12-29", universe=["AAA", "BBB", "CCC"]
        )
        assert result["missing_entities"] == []
        assert result["n_entities"] == 3
