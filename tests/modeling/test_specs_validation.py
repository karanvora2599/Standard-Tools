"""Tests for modeling.specs' Pydantic-boundary validators (duplicate
universe/feature ids, start-before-end, date parsing) and
modeling.agent.models.ScoreModelInput's mirrored checks."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.modeling.agent.models import ScoreModelInput
from standard_quant_tools.modeling.specs import DatasetSpec, FeatureSpec, TargetSpec


def _spec(**overrides):
    defaults = dict(
        universe=["AAA", "BBB"],
        start="2022-01-01",
        end="2023-01-01",
        features=[FeatureSpec(id="technical.rsi")],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )
    defaults.update(overrides)
    return DatasetSpec(**defaults)


class TestDatasetSpecValidation:
    def test_valid_spec_constructs(self):
        _spec()  # must not raise

    def test_duplicate_universe_symbols_rejected(self):
        with pytest.raises(PydanticValidationError, match="duplicate symbols"):
            _spec(universe=["AAA", "AAA", "BBB"])

    def test_repeated_feature_id_without_alias_rejected(self):
        """
        Still rejected — but because the two would produce the SAME panel
        column, not because repeating a feature is forbidden. The message
        points at `alias` rather than declaring multi-horizon specs
        unsupported.
        """
        with pytest.raises(PydanticValidationError, match="duplicate panel column"):
            _spec(
                features=[
                    FeatureSpec(id="technical.rsi", params={"period": 14}),
                    FeatureSpec(id="technical.rsi", params={"period": 30}),
                ]
            )

    def test_repeated_feature_id_with_aliases_accepted(self):
        """momentum(20) + momentum(252) — an ordinary multi-horizon model
        spec that was previously impossible to express."""
        spec = _spec(
            features=[
                FeatureSpec(
                    id="market.momentum", params={"lookback": 20}, alias="mom_20"
                ),
                FeatureSpec(
                    id="market.momentum", params={"lookback": 252}, alias="mom_252"
                ),
            ]
        )
        assert [f.output_name for f in spec.features] == ["mom_20", "mom_252"]

    def test_colliding_aliases_rejected(self):
        with pytest.raises(PydanticValidationError, match="duplicate panel column"):
            _spec(
                features=[
                    FeatureSpec(id="technical.rsi", alias="x"),
                    FeatureSpec(id="market.momentum", alias="x"),
                ]
            )

    def test_alias_colliding_with_a_plain_feature_id_rejected(self):
        with pytest.raises(PydanticValidationError, match="duplicate panel column"):
            _spec(
                features=[
                    FeatureSpec(id="technical.rsi"),
                    FeatureSpec(id="market.momentum", alias="technical.rsi"),
                ]
            )

    @pytest.mark.parametrize("reserved", ["date", "entity", "target", "label_end_date"])
    def test_reserved_alias_rejected(self, reserved):
        """An alias matching a panel-schema column would overwrite it."""
        with pytest.raises(PydanticValidationError, match="reserved"):
            FeatureSpec(id="technical.rsi", alias=reserved)

    def test_blank_alias_rejected(self):
        with pytest.raises(PydanticValidationError, match="non-empty"):
            FeatureSpec(id="technical.rsi", alias="   ")

    def test_output_name_defaults_to_id(self):
        assert FeatureSpec(id="technical.rsi").output_name == "technical.rsi"

    def test_start_after_end_rejected(self):
        with pytest.raises(PydanticValidationError, match="must be before"):
            _spec(start="2023-01-01", end="2022-01-01")

    def test_start_equal_end_rejected(self):
        with pytest.raises(PydanticValidationError, match="must be before"):
            _spec(start="2022-01-01", end="2022-01-01")

    def test_malformed_start_date_rejected(self):
        with pytest.raises(PydanticValidationError, match="not a valid date"):
            _spec(start="not-a-date")

    def test_empty_benchmark_rejected(self):
        with pytest.raises(PydanticValidationError):
            _spec(benchmark="")


class TestScoreModelInputValidation:
    def test_valid_input_constructs(self):
        ScoreModelInput(model_id="mdl_abc", as_of="2024-01-01", universe=["AAA", "BBB"])

    def test_malformed_as_of_rejected(self):
        with pytest.raises(PydanticValidationError, match="not a valid date"):
            ScoreModelInput(model_id="mdl_abc", as_of="not-a-date", universe=["AAA"])

    def test_duplicate_universe_rejected(self):
        with pytest.raises(PydanticValidationError, match="duplicate symbols"):
            ScoreModelInput(
                model_id="mdl_abc", as_of="2024-01-01", universe=["AAA", "AAA"]
            )
