"""
A feature can be a function of another feature.

`FeatureSpec` had no field for it and `requires` names OHLCV columns only,
in all 29 occurrences — so "momentum per unit of volatility", "the spread
between two horizons of the same feature", "this one net of that one" were
unexpressible. Both escape hatches were bad: custom Python needs a code
deploy and loses warm-up resolution, and an external panel is second-class
("there is no feature definition to price and no warm-up this library can
know").

THE OPERATORS ARE POINTWISE, AND THAT IS A LEAKAGE DECISION. Row-wise
functions of columns on the same row have no state, no window and nothing
fitted, so a derived column costs no warm-up beyond its inputs' and cannot
leak — there is nothing to fit, so nothing can be fitted on the wrong rows.
`residual_ols` is refused by name for exactly that reason and the test for
it says so.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.derived import (
    FITTED_OPERATORS,
    OPERATORS,
    apply_derived,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    DerivedFeatureSpec,
    FeatureSpec,
    TargetSpec,
)


def _frame():
    return pd.DataFrame({"mom": [2.0, 4.0, 6.0], "vol": [1.0, 2.0, 0.0]})


def _spec(name, op, inputs, lags=None):
    return DerivedFeatureSpec(
        name=name, op=op, inputs=inputs, lags=lags or []
    )


class TestTheOperators:
    @pytest.mark.parametrize(
        "op,expected",
        [
            ("difference", [1.0, 2.0, 6.0]),
            ("product", [2.0, 8.0, 0.0]),
            ("sum", [3.0, 6.0, 6.0]),
        ],
    )
    def test_the_pointwise_arithmetic(self, op, expected):
        out = apply_derived(_frame(), [_spec("d", op, ["mom", "vol"])])
        assert list(out["d"]) == pytest.approx(expected)

    def test_a_zero_denominator_is_nan_not_an_infinity(self):
        """An infinity is the quieter failure: it survives into the panel
        and most imputations turn it into a very large real number the
        model then fits. NaN is what the rest of the pipeline talks
        about."""
        out = apply_derived(_frame(), [_spec("d", "ratio", ["mom", "vol"])])
        assert list(out["d"][:2]) == pytest.approx([2.0, 2.0])
        assert pd.isna(out["d"].iloc[2])
        assert not np.isinf(out["d"].to_numpy(dtype=float)).any()

    def test_the_inputs_are_untouched(self):
        frame = _frame()
        apply_derived(frame, [_spec("d", "sum", ["mom", "vol"])])
        assert "d" not in frame.columns


class TestItChains:
    def test_a_derived_feature_can_read_an_earlier_one(self):
        out = apply_derived(
            _frame(),
            [
                _spec("first", "sum", ["mom", "vol"]),
                _spec("second", "product", ["first", "mom"]),
            ],
        )
        assert list(out["second"]) == pytest.approx([6.0, 24.0, 36.0])

    def test_a_forward_reference_is_refused_by_the_spec(self):
        """Inputs may name only what already exists, which is what makes a
        cycle inexpressible rather than detected."""
        with pytest.raises(PydanticValidationError, match="not available where it is defined"):
            DatasetSpec(
                universe=["AAA"],
                start="2022-01-01",
                end="2023-01-01",
                features=[FeatureSpec(id="technical.rsi")],
                derived=[
                    _spec("a", "sum", ["technical.rsi", "b"]),
                    _spec("b", "sum", ["technical.rsi", "technical.rsi"]),
                ],
                target=TargetSpec(horizon=5),
            )

    def test_a_name_collision_is_refused(self):
        with pytest.raises(PydanticValidationError, match="collides with a column"):
            DatasetSpec(
                universe=["AAA"],
                start="2022-01-01",
                end="2023-01-01",
                features=[FeatureSpec(id="technical.rsi")],
                derived=[_spec("technical.rsi", "sum", ["technical.rsi", "technical.rsi"])],
                target=TargetSpec(horizon=5),
            )


class TestAFittedDerivationIsRefused:
    def test_residual_ols_is_named_and_explained(self):
        """Not merely absent. A residual is fitted, and fitting it over an
        entity's whole history at build time fits on the test window too."""
        assert "residual_ols" in FITTED_OPERATORS
        assert "residual_ols" not in OPERATORS

    def test_the_refusal_points_at_preprocessing(self):
        class _Fitted:
            name = "d"
            op = "residual_ols"
            inputs = ["mom", "vol"]

        with pytest.raises(ValidationError, match="preprocessing"):
            apply_derived(_frame(), [_Fitted()])

    def test_the_spec_will_not_even_build_one(self):
        with pytest.raises(PydanticValidationError):
            DerivedFeatureSpec(name="d", op="residual_ols", inputs=["a", "b"])


class TestItCostsNoWarmUp:
    def test_the_value_appears_exactly_where_both_inputs_do(self):
        """The property behind "pointwise": no window, so no extra history
        and nothing for a warm-up calculation to learn."""
        frame = pd.DataFrame({"a": [np.nan, 1.0, 2.0], "b": [1.0, np.nan, 4.0]})
        out = apply_derived(frame, [_spec("d", "sum", ["a", "b"])])
        assert pd.isna(out["d"].iloc[0])
        assert pd.isna(out["d"].iloc[1])
        assert out["d"].iloc[2] == pytest.approx(6.0)


class TestThroughTheBuilder:
    def test_the_panel_carries_it_and_the_dataset_names_it(
        self, patched_multi_factory
    ):
        from standard_quant_tools.modeling.dataset.builder import build_dataset

        from .test_scoring import _dataset_spec

        spec = _dataset_spec(
            derived=[
                _spec(
                    "mom_per_rsi",
                    "ratio",
                    ["market.momentum", "technical.rsi"],
                )
            ]
        )
        built = build_dataset(spec)
        assert "mom_per_rsi" in built["feature_ids"]
        assert "mom_per_rsi" in built["panel"].columns

    def test_a_derived_column_can_be_lagged_like_any_other(
        self, patched_multi_factory
    ):
        """The derivation runs before the lag expansion, so the lag is of
        the DERIVED value and not of its inputs."""
        from standard_quant_tools.modeling.dataset.builder import build_dataset

        from .test_scoring import _dataset_spec

        spec = _dataset_spec(
            derived=[
                _spec(
                    "spread",
                    "difference",
                    ["market.momentum", "technical.rsi"],
                    lags=[1],
                )
            ]
        )
        built = build_dataset(spec)
        assert "spread" in built["feature_ids"]
        assert "spread__lag1" in built["feature_ids"]
        panel = built["panel"].sort_values(["entity", "date"])
        checked = 0
        for _entity, rows in panel.groupby("entity"):
            shifted = rows["spread"].shift(1)
            both = rows["spread__lag1"].notna() & shifted.notna()
            assert np.allclose(
                rows.loc[both, "spread__lag1"].to_numpy(dtype=float),
                shifted[both].to_numpy(dtype=float),
            )
            checked += int(both.sum())
        assert checked > 0, "nothing was compared, so the lag is unverified"

    def test_a_model_fits_on_a_derived_feature(self, patched_multi_factory):
        """The end of the road: the thing was unexpressible, and now a
        model is fitted on it."""
        from standard_quant_tools.modeling.dataset.builder import build_dataset
        from standard_quant_tools.modeling.engine import run_experiment
        from standard_quant_tools.modeling.specs import (
            EstimatorSpec,
            ModelSpec,
            ValidationSpec,
        )

        from .test_scoring import _dataset_spec

        dataset = build_dataset(
            _dataset_spec(
                derived=[
                    _spec(
                        "mom_per_rsi",
                        "ratio",
                        ["market.momentum", "technical.rsi"],
                    )
                ]
            )
        )
        result = run_experiment(
            dataset,
            ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge"),
                validation=ValidationSpec(
                    method="walk_forward",
                    train_window=120,
                    test_window=40,
                    min_folds=1,
                ),
                random_seed=0,
            ),
            "ds_derived",
            register=False,
        )
        assert "mom_per_rsi" in result["feature_importance_summary"]
