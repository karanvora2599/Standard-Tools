"""
A label registered from outside the library builds, purges on its own
label ends, carries its own bounded parameters, and is refused by the
tasks it did not name -- without `specs.py`, `dataset/target.py` or
`engine.py` being edited.

Two labels are planted. A residual return reads the benchmark through the
feature context and takes a bounded `beta`; a next-bar sign resolves one
bar ahead and says so through its own `label_end_builder`, which the
engine's purge honours -- measured as exactly one purged row per entity
per fold at embargo zero, against five for a five-bar horizon.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.models import ValidateModelSpecInput
from standard_quant_tools.modeling.agent.tools import validate_model_spec
from standard_quant_tools.modeling.capabilities import modeling_capabilities
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.dataset.target import (
    build_label_end_dates,
    build_target,
)
from standard_quant_tools.modeling.engine import (
    _check_task_target_compatibility,
    run_experiment,
)
from standard_quant_tools.modeling.estimators.bounds import (
    EstimatorParamSchema,
    ParamBound,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)
from standard_quant_tools.modeling.targets import (
    TARGET_REGISTRY,
    TargetDefinition,
    register_target,
)

from .conftest import make_ohlcv

UNIVERSE = ["AAA", "BBB", "CCC"]


def _forward(close: pd.Series, horizon: int) -> pd.Series:
    return close.pct_change(periods=horizon, fill_method=None).shift(-horizon)


def _residual_return(ohlcv, spec, context):
    """Forward return minus beta times the benchmark's, over the same bars."""
    close = ohlcv["Close"]
    bench = context.benchmark_close.reindex(close.index)
    beta = float(spec.resolved_params["beta"])
    return _forward(close, spec.horizon) - beta * _forward(bench, spec.horizon)


def _next_bar_sign(ohlcv, spec, context):
    """1.0 if the next bar closes up, else 0.0 -- resolved ONE bar ahead
    whatever `horizon` says."""
    step = _forward(ohlcv["Close"], 1)
    return (step > 0).astype(float).where(step.notna())


def _next_bar_end(ohlcv, spec, context):
    close = ohlcv["Close"]
    ends = pd.Series(pd.NaT, index=close.index, dtype="datetime64[ns]")
    ends.iloc[:-1] = close.index[1:]
    return ends


def _high_low_range(ohlcv, spec, context):
    return (ohlcv["High"] - ohlcv["Low"]) / ohlcv["Close"]


@pytest.fixture(autouse=True)
def _planted_labels():
    register_target(
        TargetDefinition(
            id="test.residual_return",
            description="Forward return minus beta times the benchmark's forward return.",
            tasks=("regression", "ranking"),
            buildable=True,
            continuous=True,
            builder=_residual_return,
            param_schema=EstimatorParamSchema(bounds={"beta": ParamBound("float", 0.0, 3.0)}),
            default_params={"beta": 1.0},
        ),
        overwrite=True,
    )
    register_target(
        TargetDefinition(
            id="test.next_bar_sign",
            description="Whether the next bar closes up; resolves one bar ahead.",
            tasks=("classification",),
            buildable=True,
            continuous=False,
            builder=_next_bar_sign,
            label_end_builder=_next_bar_end,
        ),
        overwrite=True,
    )
    register_target(
        TargetDefinition(
            id="test.range",
            description="The bar's high-low range as a fraction of its close.",
            tasks=("regression",),
            buildable=True,
            continuous=True,
            requires=["High", "Low", "Close"],
            builder=_high_low_range,
        ),
        overwrite=True,
    )
    yield
    for name in ("test.residual_return", "test.next_bar_sign", "test.range"):
        TARGET_REGISTRY.pop(name, None)


def _spec(target: TargetSpec) -> DatasetSpec:
    return DatasetSpec(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi")],
        target=target,
        benchmark="SPY",
    )


class TestRegistration:
    def test_the_label_is_seen_everywhere_it_should_be(self):
        assert TargetSpec.model_json_schema()["properties"]["type"]["enum"] == sorted(
            TARGET_REGISTRY
        )
        assert "test.residual_return" in modeling_capabilities()["targets"]["buildable"]
        assert "test.residual_return" in modeling_capabilities()["targets"]["all"]

    def test_a_duplicate_is_refused_without_overwrite(self):
        with pytest.raises(ValidationError, match="already registered"):
            register_target(TARGET_REGISTRY["test.range"])

    def test_a_buildable_label_needs_a_builder(self):
        with pytest.raises(ValidationError, match="has no builder"):
            register_target(
                TargetDefinition(
                    id="test.no_builder",
                    description="Declared buildable with nothing to build it.",
                    tasks=("regression",),
                    buildable=True,
                    continuous=True,
                )
            )

    def test_an_external_label_must_not_have_one(self):
        with pytest.raises(ValidationError, match="external-only and carries a builder"):
            register_target(
                TargetDefinition(
                    id="test.fake_fill",
                    description="A fill probability approximated from bars, refused.",
                    tasks=("classification",),
                    buildable=False,
                    continuous=False,
                    builder=_next_bar_sign,
                )
            )

    def test_an_unknown_task_and_a_thin_description_are_refused(self):
        with pytest.raises(ValidationError, match="do not exist"):
            register_target(
                TargetDefinition(
                    id="test.bad_task",
                    description="Names a task that the library does not fit.",
                    tasks=("survival",),
                    buildable=True,
                    continuous=True,
                    builder=_next_bar_sign,
                )
            )
        with pytest.raises(ValidationError, match="too short"):
            register_target(
                TargetDefinition(
                    id="test.terse",
                    description="short",
                    tasks=("regression",),
                    buildable=True,
                    continuous=True,
                    builder=_next_bar_sign,
                )
            )


class TestParameters:
    def test_defaults_merge_under_overrides(self):
        assert TargetSpec(type="test.residual_return", horizon=20).resolved_params == {
            "beta": 1.0
        }
        assert TargetSpec(
            type="test.residual_return", horizon=20, params={"beta": 0.5}
        ).resolved_params == {"beta": 0.5}

    def test_a_value_outside_the_bound_is_refused_at_the_spec(self):
        with pytest.raises(PydanticValidationError, match="exceeds the maximum"):
            TargetSpec(type="test.residual_return", horizon=20, params={"beta": 5.0})

    def test_an_unknown_parameter_is_refused(self):
        with pytest.raises(PydanticValidationError, match="does not accept"):
            TargetSpec(type="test.residual_return", horizon=20, params={"gamma": 1.0})

    def test_a_built_in_takes_no_params(self):
        with pytest.raises(PydanticValidationError, match="does not accept"):
            TargetSpec(type="forward_return", horizon=5, params={"beta": 1.0})

    def test_params_survive_a_round_trip(self):
        spec = TargetSpec(type="test.residual_return", horizon=20, params={"beta": 0.5})
        assert TargetSpec(**spec.model_dump()) == spec


class TestBuilding:
    def test_the_residual_return_reads_the_benchmark_through_the_context(
        self, patched_multi_factory
    ):
        built = build_dataset(
            _spec(TargetSpec(type="test.residual_return", horizon=20, params={"beta": 0.5}))
        )
        panel = built["panel"]
        rows = panel[panel["entity"] == "AAA"].set_index("date")
        close = make_ohlcv("AAA")["Close"]
        bench = make_ohlcv("SPY")["Close"]
        expected = (_forward(close, 20) - 0.5 * _forward(bench, 20)).dropna()
        common = rows.index.intersection(expected.index)
        assert len(common) > 100
        np.testing.assert_allclose(
            rows.loc[common, "target"].to_numpy(), expected.loc[common].to_numpy(), atol=1e-12
        )
        assert built["target_id"] == "test.residual_return:20"

    def test_a_label_that_needs_high_and_low_is_refused_a_bare_close(self):
        close = make_ohlcv("AAA")["Close"]
        with pytest.raises(ValidationError, match=r"reads column\(s\) \['High', 'Low'\]"):
            build_target(close, TargetSpec(type="test.range", horizon=1))
        # And builds from the frame.
        series = build_target(make_ohlcv("AAA"), TargetSpec(type="test.range", horizon=1))
        assert series.notna().all()

    def test_the_default_label_end_is_the_horizon(self):
        ohlcv = make_ohlcv("AAA")
        ends = build_label_end_dates(ohlcv, TargetSpec(type="test.residual_return", horizon=20))
        assert ends.iloc[0] == ohlcv.index[20]
        assert ends.iloc[-20:].isna().all()

    def test_a_custom_label_end_is_honoured(self, patched_multi_factory):
        built = build_dataset(_spec(TargetSpec(type="test.next_bar_sign", horizon=5)))
        rows = built["panel"][built["panel"]["entity"] == "AAA"].set_index("date")
        dates = make_ohlcv("AAA").index
        position = dates.get_loc(rows.index[50])
        assert rows["label_end_date"].iloc[50] == dates[position + 1]


class TestTheEngine:
    def test_the_purge_reads_the_custom_label_end(self, patched_multi_factory):
        """
        Planted arithmetic. At embargo zero, a label that resolves one bar
        ahead reaches the test block from exactly the last training bar of
        each entity: one purged row per entity per fold. A five-bar horizon
        reaches from the last five.
        """
        validation = ValidationSpec(train_window=150, test_window=30, embargo=0)
        custom = run_experiment(
            build_dataset(_spec(TargetSpec(type="test.next_bar_sign", horizon=5))),
            ModelSpec(
                task="classification",
                estimator=EstimatorSpec(type="logistic"),
                validation=validation,
                random_seed=1,
            ),
            "ds",
            register=False,
        )
        default = run_experiment(
            build_dataset(_spec(TargetSpec(type="forward_direction", horizon=5))),
            ModelSpec(
                task="classification",
                estimator=EstimatorSpec(type="logistic"),
                validation=validation,
                random_seed=1,
            ),
            "ds",
            register=False,
        )
        folds = custom["validation_report"]["n_folds_completed"]
        assert folds == default["validation_report"]["n_folds_completed"] >= 2
        assert custom["n_train_rows_purged_overlap"] == 1 * len(UNIVERSE) * folds
        assert default["n_train_rows_purged_overlap"] == 5 * len(UNIVERSE) * folds

    def test_a_task_the_label_did_not_name_is_refused(self, patched_multi_factory):
        with pytest.raises(ValidationError, match="expects one of"):
            _check_task_target_compatibility("classification", "test.residual_return:20")
        _check_task_target_compatibility("ranking", "test.residual_return:20")

    def test_validate_model_spec_reads_the_registry(self, patched_multi_factory):
        from standard_quant_tools.modeling.agent.models import BuildModelDatasetInput
        from standard_quant_tools.modeling.agent.tools import build_model_dataset

        dataset_id = build_model_dataset(
            BuildModelDatasetInput(
                spec=_spec(TargetSpec(type="test.residual_return", horizon=20))
            )
        ).dataset_id
        refused = validate_model_spec(
            ValidateModelSpecInput(
                spec=ModelSpec(
                    task="classification",
                    estimator=EstimatorSpec(type="logistic"),
                    validation=ValidationSpec(train_window=150, test_window=30),
                ),
                dataset_id=dataset_id,
            )
        )
        assert not refused.valid and refused.problems[0].where == "target"
        accepted = validate_model_spec(
            ValidateModelSpecInput(
                spec=ModelSpec(
                    task="regression",
                    estimator=EstimatorSpec(type="ridge"),
                    validation=ValidationSpec(train_window=150, test_window=30),
                ),
                dataset_id=dataset_id,
            )
        )
        assert accepted.valid, accepted.problems

    def test_a_custom_label_trains_end_to_end(self, patched_multi_factory):
        result = run_experiment(
            build_dataset(
                _spec(TargetSpec(type="test.residual_return", horizon=20, params={"beta": 0.5}))
            ),
            ModelSpec(
                task="regression",
                estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
                random_seed=1,
            ),
            "ds",
        )
        assert result["n_folds"] >= 2
        assert result["validation_report"]["target_horizon"] == 20
