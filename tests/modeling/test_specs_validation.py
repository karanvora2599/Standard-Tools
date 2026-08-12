"""Tests for modeling.specs' Pydantic-boundary validators (duplicate
universe/feature ids, start-before-end, date parsing) and
modeling.agent.models.ScoreModelInput's mirrored checks."""

import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.models import ScoreModelInput
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)


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


class TestParameterAndResourceBounds:
    """
    Estimator parameters already carried compute ceilings because one tool
    call otherwise becomes a resource-exhaustion path. The same reasoning
    had not been applied to feature parameters, request sizes or the RNG
    seed.
    """

    def test_integer_default_requires_integer_value(self):
        """
        `refit_every` is not in the window-name vocabulary, so a float
        passed the generic finite-number check and reached
        range(window, n+1, refit_every), which raises a raw TypeError from
        inside Python rather than a modeling error naming the feature.
        Integer-ness now comes from the DEFAULT's type, not the name.
        """
        from standard_quant_tools.modeling.features.params import resolve_params
        from standard_quant_tools.modeling.features.registry import get_feature

        definition = get_feature("factors.pca_loading")
        assert isinstance(definition.default_params["refit_every"], int)
        with pytest.raises(ValidationError, match="whole number"):
            resolve_params(definition, {"refit_every": 1.5})

    def test_integer_valued_float_is_accepted_and_coerced(self):
        from standard_quant_tools.modeling.features.params import resolve_params
        from standard_quant_tools.modeling.features.registry import get_feature

        resolved = resolve_params(
            get_feature("factors.pca_loading"), {"refit_every": 5.0}
        )
        assert resolved["refit_every"] == 5
        assert isinstance(resolved["refit_every"], int)

    def test_absurd_window_is_rejected(self):
        from standard_quant_tools.modeling.features.params import resolve_params
        from standard_quant_tools.modeling.features.registry import get_feature

        with pytest.raises(ValidationError, match="maximum supported window"):
            resolve_params(get_feature("technical.rsi"), {"period": 10_000_000})

    def test_universe_size_is_bounded(self):
        with pytest.raises(PydanticValidationError):
            DatasetSpec(
                universe=[f"S{i}" for i in range(1001)],
                start="2022-01-01",
                end="2023-01-01",
                features=[FeatureSpec(id="technical.rsi")],
                target=TargetSpec(horizon=5),
            )

    @pytest.mark.parametrize("seed", [-1, 2**32, 2**63])
    def test_random_seed_bounded_to_rng_range(self, seed):
        with pytest.raises(PydanticValidationError):
            ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={}),
                validation=ValidationSpec(train_window=100, test_window=20, embargo=5),
                random_seed=seed,
            )

    def test_valid_seed_still_accepted(self):
        spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={}),
            validation=ValidationSpec(train_window=100, test_window=20, embargo=5),
            random_seed=2**32 - 1,
        )
        assert spec.random_seed == 2**32 - 1


class TestReservedPanelColumnNames:
    """
    FeatureSpec.alias rejected reserved names because an alias becomes the
    panel's column. A feature id with NO alias is equally the output column
    name, and nothing checked it — so a custom feature registered as
    id="target" produced a column shadowing the panel's supervised target.
    """

    @pytest.mark.parametrize("reserved", ["date", "entity", "target", "label_end_date"])
    def test_reserved_feature_id_is_rejected(self, reserved):
        from standard_quant_tools.modeling.features.base import (
            FeatureDefinition,
            FeatureScope,
            TemporalSupport,
        )
        from standard_quant_tools.modeling.features.registry import register_feature

        with pytest.raises(ValidationError, match="reserved"):
            register_feature(
                FeatureDefinition(
                    id=reserved,
                    description="x",
                    fn=lambda ohlcv, context: ohlcv["Close"],
                    default_params={},
                    temporal_support=TemporalSupport.PIT_SAFE,
                    scope=FeatureScope.ENTITY,
                    requires=["Close"],
                    lookback=0,
                )
            )

    def test_alias_and_id_share_one_reserved_list(self):
        """Two independent copies is how the id path drifted from the alias
        path in the first place."""
        from standard_quant_tools.modeling.features.base import RESERVED_PANEL_COLUMNS

        assert RESERVED_PANEL_COLUMNS == {"date", "entity", "target", "label_end_date"}
        with pytest.raises(PydanticValidationError, match="reserved"):
            FeatureSpec(id="technical.rsi", alias="target")
