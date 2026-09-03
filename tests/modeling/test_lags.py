"""
History as columns, and the two ways it silently becomes wrong.

WHAT THESE PIN.

  1. A lag applied after the entities are stacked reaches the PREVIOUS
     ENTITY'S row, not the previous bar. The panel that results is
     perfectly well-formed and every aggregate statistic downstream looks
     normal, so nothing catches it later.
  2. A negative lag is a shift FORWARD -- a future value on today's row. It
     is refused at the spec boundary rather than clamped, because every
     leakage check in this library reasons about the target and none of
     them would see it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.lags import (
    MAX_EXPANDED_COLUMNS,
    MAX_LAG,
    MAX_LAGS_PER_FEATURE,
    deepest_lag,
    expand_lags,
    expanded_feature_ids,
    lag_column_name,
    lags_by_output_name,
    parse_lag_column,
    validate_lags,
)
from standard_quant_tools.modeling.specs import FeatureSpec


class TestLagsAreBackwardOnly:
    def test_a_negative_lag_is_refused_with_the_reason(self) -> None:
        with pytest.raises(ValidationError, match="NEGATIVE lag"):
            validate_lags([-1])

    def test_the_refusal_names_the_value_the_caller_probably_meant(self) -> None:
        with pytest.raises(ValidationError, match=r"pass 3"):
            validate_lags([-3])

    def test_zero_is_refused_as_the_column_that_already_exists(self) -> None:
        with pytest.raises(ValidationError, match="already in the panel"):
            validate_lags([0])

    def test_the_spec_refuses_it_too_not_only_the_helper(self) -> None:
        """The boundary that matters: an agent writes the spec, not a call."""
        with pytest.raises(Exception, match="NEGATIVE lag"):
            FeatureSpec(id="technical.rsi", lags=[-1])


class TestLagsAreNormalized:
    def test_order_and_duplicates_do_not_make_a_second_dataset(self) -> None:
        """Two specs meaning the same thing must hash to one dataset."""
        assert validate_lags([3, 1, 2, 1]) == [1, 2, 3]
        assert (
            FeatureSpec(id="x", lags=[2, 1]).lags
            == FeatureSpec(id="x", lags=[1, 2]).lags
        )

    def test_the_depth_limit_is_enforced(self) -> None:
        assert validate_lags([MAX_LAG]) == [MAX_LAG]
        with pytest.raises(ValidationError, match="deeper than"):
            validate_lags([MAX_LAG + 1])

    def test_the_count_limit_is_enforced(self) -> None:
        with pytest.raises(ValidationError, match="more than the"):
            validate_lags(list(range(1, MAX_LAGS_PER_FEATURE + 2)))


class TestTheColumnNameRoundTrips:
    def test_a_lag_column_parses_back_to_its_feature_and_depth(self) -> None:
        name = lag_column_name("technical.rsi", 3)
        assert name == "technical.rsi__lag3"
        assert parse_lag_column(name) == ("technical.rsi", 3)

    def test_an_ordinary_feature_is_not_mistaken_for_one(self) -> None:
        assert parse_lag_column("technical.rsi") is None
        assert parse_lag_column("risk.rolling_beta") is None


class TestExpansionStaysInsideTheEntity:
    def test_a_lag_reaches_the_previous_bar_not_the_previous_entity(self) -> None:
        """
        THE DEFECT THIS EXISTS TO CATCH. Expanding after `stack_long` shifts
        across the entity boundary: AAA's first row would carry BBB's last
        value. Here each entity is expanded on its OWN frame, so AAA's lag1
        is AAA's own previous bar and its first row is NaN.
        """
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        aaa = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
        bbb = pd.DataFrame({"f": [100.0, 200.0, 300.0, 400.0, 500.0]}, index=dates)

        aaa_out = expand_lags(aaa, {"f": [1]})
        bbb_out = expand_lags(bbb, {"f": [1]})

        assert np.isnan(aaa_out["f__lag1"].iloc[0])
        assert aaa_out["f__lag1"].tolist()[1:] == [1.0, 2.0, 3.0, 4.0]
        # Nothing from BBB's scale can appear in AAA's lag column.
        assert aaa_out["f__lag1"].max() < 100.0
        assert np.isnan(bbb_out["f__lag1"].iloc[0])

        # And the wrong way round, stated explicitly: stacking first and
        # shifting once gives AAA's value to BBB's first row.
        stacked = pd.concat(
            [aaa.assign(entity="AAA"), bbb.assign(entity="BBB")]
        ).reset_index(drop=True)
        naive = stacked["f"].shift(1)
        assert naive.iloc[5] == 5.0  # BBB's first row, holding AAA's last

    def test_several_lags_and_features_expand_together(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="B")
        frame = pd.DataFrame(
            {"a": np.arange(6.0), "b": np.arange(6.0) * 10}, index=dates
        )
        out = expand_lags(frame, {"a": [1, 2], "b": [1]})
        assert list(out.columns) == ["a", "b", "a__lag1", "a__lag2", "b__lag1"]
        assert out["a__lag2"].tolist()[2:] == [0.0, 1.0, 2.0, 3.0]
        assert out["b__lag1"].tolist()[1:] == [0.0, 10.0, 20.0, 30.0, 40.0]

    def test_an_unrequested_feature_is_left_alone(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        assert expand_lags(frame, {}) is frame
        assert list(expand_lags(frame, {"missing": [1]}).columns) == ["a"]


class TestTheExpandedColumnList:
    def test_a_feature_is_followed_by_its_own_lags(self) -> None:
        specs = [
            FeatureSpec(id="technical.rsi", lags=[1, 2]),
            FeatureSpec(id="risk.rolling_beta"),
        ]
        assert expanded_feature_ids(specs) == [
            "technical.rsi",
            "technical.rsi__lag1",
            "technical.rsi__lag2",
            "risk.rolling_beta",
        ]

    def test_an_alias_colliding_with_a_lag_column_is_refused(self) -> None:
        """Silently overwriting one feature with another's history."""
        specs = [
            FeatureSpec(id="technical.rsi", lags=[1]),
            FeatureSpec(id="risk.rolling_beta", alias="technical.rsi__lag1"),
        ]
        with pytest.raises(ValidationError, match="more than once"):
            expanded_feature_ids(specs)

    def test_the_total_width_is_bounded(self) -> None:
        specs = [
            FeatureSpec(id=f"f{i}", alias=f"f{i}", lags=list(range(1, 21)))
            for i in range(30)
        ]
        with pytest.raises(ValidationError, match=str(MAX_EXPANDED_COLUMNS)):
            expanded_feature_ids(specs)

    def test_the_helpers_agree_with_the_specs(self) -> None:
        specs = [
            FeatureSpec(id="a", alias="a", lags=[1, 5]),
            FeatureSpec(id="b", alias="b"),
        ]
        assert lags_by_output_name(specs) == {"a": [1, 5]}
        assert deepest_lag(specs) == 5
        assert deepest_lag([FeatureSpec(id="b", alias="b")]) == 0


class TestThroughTheBuilder:
    def test_a_built_panel_carries_the_lag_columns(self, patched_multi_factory) -> None:
        from standard_quant_tools.modeling.agent import (
            BuildModelDatasetInput,
            build_model_dataset,
        )
        from standard_quant_tools.modeling.specs import DatasetSpec, TargetSpec

        result = build_model_dataset(
            BuildModelDatasetInput(
                spec=DatasetSpec(
                    universe=["AAA", "BBB", "CCC"],
                    start="2022-01-01",
                    end="2023-12-31",
                    features=[
                        FeatureSpec(id="technical.rsi", lags=[1, 2]),
                        FeatureSpec(id="risk.rolling_beta"),
                    ],
                    target=TargetSpec(horizon=5),
                    benchmark="SPY",
                )
            )
        )
        assert result.feature_ids == [
            "technical.rsi",
            "technical.rsi__lag1",
            "technical.rsi__lag2",
            "risk.rolling_beta",
        ]

    def test_the_lag_column_is_the_entitys_own_earlier_value(
        self, patched_multi_factory
    ) -> None:
        """
        End-to-end, on a real panel: for every entity, lag1 on row i equals
        that entity's own feature on row i-1. If the expansion ran after
        stacking, the first row of every entity but the first would hold
        another entity's value instead of being dropped.
        """
        from standard_quant_tools.modeling.agent import (
            BuildModelDatasetInput,
            build_model_dataset,
        )
        from standard_quant_tools.modeling.agent.tools import _load_dataset_panel
        from standard_quant_tools.modeling.specs import DatasetSpec, TargetSpec

        built = build_model_dataset(
            BuildModelDatasetInput(
                spec=DatasetSpec(
                    universe=["AAA", "BBB", "CCC"],
                    start="2022-01-01",
                    end="2023-12-31",
                    features=[FeatureSpec(id="technical.rsi", lags=[1])],
                    target=TargetSpec(horizon=5),
                    benchmark="SPY",
                )
            )
        )
        panel, _meta, _dir = _load_dataset_panel(built.dataset_id)
        for entity, group in panel.groupby("entity"):
            ordered = group.sort_values("date")
            expected = ordered["technical.rsi"].shift(1).to_numpy()[1:]
            actual = ordered["technical.rsi__lag1"].to_numpy()[1:]
            assert np.allclose(actual, expected, equal_nan=True), entity

    def test_a_model_trains_on_the_expanded_panel(self, patched_multi_factory) -> None:
        from standard_quant_tools.modeling.agent import (
            BuildModelDatasetInput,
            RunModelExperimentInput,
            build_model_dataset,
            run_model_experiment,
        )
        from standard_quant_tools.modeling.specs import (
            DatasetSpec,
            EstimatorSpec,
            ModelSpec,
            TargetSpec,
            ValidationSpec,
        )

        built = build_model_dataset(
            BuildModelDatasetInput(
                spec=DatasetSpec(
                    universe=["AAA", "BBB", "CCC"],
                    start="2022-01-01",
                    end="2023-12-31",
                    features=[FeatureSpec(id="technical.rsi", lags=[1, 2, 3])],
                    target=TargetSpec(horizon=5),
                    benchmark="SPY",
                )
            )
        )
        experiment = run_model_experiment(
            RunModelExperimentInput(
                dataset_id=built.dataset_id,
                spec=ModelSpec(
                    task="regression",
                    estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
                    validation=ValidationSpec(
                        train_window=150, test_window=30, embargo=5
                    ),
                    random_seed=11,
                ),
            )
        )
        # The lags reached the fit: the registered model records them as
        # its own features, and importance is reported per column.
        from standard_quant_tools.modeling.registry.model_registry import (
            load_manifest,
        )

        manifest = load_manifest(experiment.model_id)
        assert set(manifest.feature_ids) == {
            "technical.rsi",
            "technical.rsi__lag1",
            "technical.rsi__lag2",
            "technical.rsi__lag3",
        }
        assert set(manifest.feature_importance_summary) <= set(manifest.feature_ids)
