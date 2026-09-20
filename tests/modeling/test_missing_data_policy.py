"""
The dataset-level missing-data policy: `drop` (complete-case alignment, the
only behaviour there was), `forward_fill_bounded` (carry a named feature's
last value within the entity for at most a stated number of bars) and
`keep` (keep the row, keep the NaN, let the fold layer handle it).

The gap is PLANTED: a registered test feature copies Close and blanks bars
100 to 103 of every entity plus the final bar, so every assertion below
names the exact rows a policy should recover, drop or fill, and the exact
value a fill must carry -- bar 99's, from the same entity.
"""

import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import artifacts as _artifacts
from standard_quant_tools.modeling.adapters import accepts_missing
from standard_quant_tools.modeling.agent.dataset_tools import (
    ExplainRowLossInput,
    explain_dataset_row_loss,
)
from standard_quant_tools.modeling.agent.models import BuildModelDatasetInput
from standard_quant_tools.modeling.agent.tools import build_model_dataset
from standard_quant_tools.modeling.capabilities import modeling_capabilities
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.estimators.registry import get_estimator_class
from standard_quant_tools.modeling.features.base import (
    FeatureDefinition,
    FeatureScope,
    TemporalSupport,
)
from standard_quant_tools.modeling.features.registry import (
    FEATURE_REGISTRY,
    register_feature,
)
from standard_quant_tools.modeling.registry.model_registry import (
    load_preprocessing_state,
)
from standard_quant_tools.modeling.scoring import score_model
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    MissingDataSpec,
    ModelSpec,
    PreprocessingSpec,
    StepSpec,
    TargetSpec,
    ValidationSpec,
)

from .conftest import make_ohlcv
from .test_deployed_pipeline import _register

UNIVERSE = ["AAA", "BBB", "CCC"]
GAP = slice(100, 104)


def _gappy(ohlcv, context, **params):
    """Close, with bars 100..103 and the final bar blanked."""
    series = ohlcv["Close"].copy()
    series.iloc[GAP] = np.nan
    series.iloc[-1] = np.nan
    return series


def _infinite(ohlcv, context, **params):
    series = ohlcv["Close"].copy()
    series.iloc[50] = np.inf
    return series


@pytest.fixture(autouse=True)
def _planted_features():
    for feature_id, fn in (("test.gappy", _gappy), ("test.inf", _infinite)):
        register_feature(
            FeatureDefinition(
                id=feature_id,
                description="planted",
                fn=fn,
                temporal_support=TemporalSupport.PIT_SAFE,
                scope=FeatureScope.ENTITY,
                lookback=0,
            ),
            overwrite=True,
        )
    yield
    FEATURE_REGISTRY.pop("test.gappy", None)
    FEATURE_REGISTRY.pop("test.inf", None)


def _spec(missing: "MissingDataSpec | None" = None, features=None) -> DatasetSpec:
    return DatasetSpec(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=features
        or [FeatureSpec(id="technical.rsi"), FeatureSpec(id="test.gappy", lags=[1])],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
        missing=missing or MissingDataSpec(),
    )


def _bar_dates():
    return make_ohlcv("AAA").index


def _entity_rows(panel: pd.DataFrame, entity: str) -> pd.DataFrame:
    return panel[panel["entity"] == entity].set_index("date")


# ── The spec ────────────────────────────────────────────────────────────


class TestTheSpec:
    def test_the_default_is_drop_and_is_absent_from_the_hash(self):
        spec = _spec()
        assert spec.missing.policy == "drop"
        assert "missing" not in json.loads(spec.model_dump_json(exclude_defaults=True))

    def test_forward_fill_needs_a_bound_and_an_allowlist(self):
        with pytest.raises(PydanticValidationError, match="max_staleness_bars >= 1"):
            MissingDataSpec(policy="forward_fill_bounded", features=["x"])
        with pytest.raises(PydanticValidationError, match="needs `features`"):
            MissingDataSpec(policy="forward_fill_bounded", max_staleness_bars=3)

    def test_the_other_policies_refuse_the_fill_fields(self):
        with pytest.raises(PydanticValidationError, match="belong to"):
            MissingDataSpec(policy="keep", max_staleness_bars=2)
        with pytest.raises(PydanticValidationError, match="belong to"):
            MissingDataSpec(policy="drop", features=["x"])

    def test_a_filled_feature_must_be_one_the_spec_produces(self):
        with pytest.raises(PydanticValidationError, match="does not produce"):
            _spec(
                MissingDataSpec(
                    policy="forward_fill_bounded", max_staleness_bars=2, features=["nope"]
                )
            )

    def test_a_fill_spec_survives_its_own_serialization(self):
        spec = _spec(
            MissingDataSpec(
                policy="forward_fill_bounded", max_staleness_bars=3, features=["test.gappy"]
            )
        )
        assert DatasetSpec(**spec.model_dump()) == spec


# ── The builder ─────────────────────────────────────────────────────────


class TestDropIsUnchanged:
    def test_the_gap_and_its_lag_shadow_are_absent(self, patched_multi_factory):
        built = build_dataset(_spec())
        dates = _bar_dates()
        rows = _entity_rows(built["panel"], "AAA")
        # The four blank bars, and bar 104 whose lag-1 reads bar 103.
        assert not set(dates[100:105]) & set(rows.index)
        assert dates[99] in rows.index and dates[105] in rows.index
        assert built["drop_attribution"]["policy"] == "drop"
        assert not built["panel"][built["feature_ids"]].isna().any().any()


class TestForwardFillBounded:
    def _built(self, staleness: int):
        return build_dataset(
            _spec(
                MissingDataSpec(
                    policy="forward_fill_bounded",
                    max_staleness_bars=staleness,
                    features=["test.gappy"],
                )
            )
        )

    def test_a_wide_enough_bound_carries_bar_99_across_the_gap(self, patched_multi_factory):
        built = self._built(5)
        dates = _bar_dates()
        for entity in UNIVERSE:
            rows = _entity_rows(built["panel"], entity)
            expected = float(make_ohlcv(entity)["Close"].iloc[99])
            for bar in range(100, 104):
                assert rows.loc[dates[bar], "test.gappy"] == expected
            # The lag of a filled value is the filled value.
            assert rows.loc[dates[101], "test.gappy__lag1"] == expected
            assert rows.loc[dates[104], "test.gappy__lag1"] == expected

    def test_the_fill_does_not_cross_entities(self, patched_multi_factory):
        built = self._built(5)
        dates = _bar_dates()
        a = _entity_rows(built["panel"], "AAA").loc[dates[102], "test.gappy"]
        b = _entity_rows(built["panel"], "BBB").loc[dates[102], "test.gappy"]
        assert a == float(make_ohlcv("AAA")["Close"].iloc[99])
        assert b == float(make_ohlcv("BBB")["Close"].iloc[99])
        assert a != b

    def test_a_tight_bound_fills_exactly_that_many_bars(self, patched_multi_factory):
        built = self._built(2)
        dates = _bar_dates()
        rows = _entity_rows(built["panel"], "AAA")
        expected = float(make_ohlcv("AAA")["Close"].iloc[99])
        assert rows.loc[dates[100], "test.gappy"] == expected
        assert rows.loc[dates[101], "test.gappy"] == expected
        assert dates[102] not in rows.index and dates[103] not in rows.index

    def test_the_fill_is_reported_with_its_count(self, patched_multi_factory):
        built = self._built(5)
        # Four gap bars plus the final bar, per entity.
        assert any("test.gappy 15" in w and "forward-filled" in w for w in built["warnings"])

    def test_a_feature_not_named_is_not_filled(self, patched_multi_factory):
        """Only test.gappy is allowlisted; the RSI warm-up is a leading NaN
        with nothing to carry, and stays a warm-up."""
        built = self._built(5)
        dates = _bar_dates()
        rows = _entity_rows(built["panel"], "AAA")
        assert rows.index.min() >= dates[14]

    def test_a_fill_that_recovers_nothing_says_so(self, patched_multi_factory):
        built = build_dataset(
            _spec(
                MissingDataSpec(
                    policy="forward_fill_bounded",
                    max_staleness_bars=3,
                    features=["technical.rsi"],
                )
            )
        )
        assert any("filled nothing" in w for w in built["warnings"])


class TestKeep:
    def test_the_gap_rows_survive_with_their_hole(self, patched_multi_factory):
        built = build_dataset(_spec(MissingDataSpec(policy="keep")))
        dates = _bar_dates()
        rows = _entity_rows(built["panel"], "AAA")
        for bar in range(100, 104):
            assert dates[bar] in rows.index
            assert np.isnan(rows.loc[dates[bar], "test.gappy"])
        assert rows.loc[dates[102], "technical.rsi"] == pytest.approx(
            rows.loc[dates[102], "technical.rsi"]
        )

    def test_rows_are_still_dropped_on_the_target(self, patched_multi_factory):
        built = build_dataset(_spec(MissingDataSpec(policy="keep")))
        dates = _bar_dates()
        rows = _entity_rows(built["panel"], "AAA")
        assert not set(dates[-5:]) & set(rows.index)
        assert not built["panel"]["target"].isna().any()

    def test_the_attribution_says_what_drop_would_have_removed(self, patched_multi_factory):
        kept = build_dataset(_spec(MissingDataSpec(policy="keep")))
        dropped = build_dataset(_spec())
        attribution = kept["drop_attribution"]
        assert attribution["policy"] == "keep"
        assert attribution["rows_after_alignment"] == len(kept["panel"])
        assert attribution["rows_that_drop_would_remove"] == dropped["drop_attribution"]["rows_dropped"]
        assert attribution["rows_with_missing_features"] == int(
            kept["panel"][kept["feature_ids"]].isna().any(axis=1).sum()
        )
        assert any("missing.policy='keep'" in w for w in kept["warnings"])

    def test_an_infinity_is_still_refused(self, patched_multi_factory):
        with pytest.raises(ValidationError, match="infinite"):
            build_dataset(
                _spec(
                    MissingDataSpec(policy="keep"),
                    features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="test.inf")],
                )
            )

    def test_the_row_loss_tool_says_the_rows_are_present(self, patched_multi_factory):
        dataset_id = build_model_dataset(
            BuildModelDatasetInput(spec=_spec(MissingDataSpec(policy="keep")))
        ).dataset_id
        result = explain_dataset_row_loss(ExplainRowLossInput(dataset_id=dataset_id))
        assert any("PRESENT" in w for w in result.warnings)


# ── The engine and scoring ──────────────────────────────────────────────


def _model(estimator: str, steps=None, **params) -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type=estimator, params=params),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        preprocessing=PreprocessingSpec(steps=steps) if steps else PreprocessingSpec(),
        random_seed=1,
    )


class TestTheEngineUnderKeep:
    @pytest.fixture
    def kept(self, patched_multi_factory):
        return build_dataset(_spec(MissingDataSpec(policy="keep")))

    def test_an_estimator_that_cannot_take_nan_is_refused_by_name(self, kept):
        with pytest.raises(ValidationError, match="impute"):
            run_experiment(kept, _model("ridge", alpha=1.0), "ds", register=False)

    def test_an_impute_step_closes_the_hole_with_the_training_median(self, kept):
        result = run_experiment(
            kept,
            _model("ridge", steps=[StepSpec(type="impute"), StepSpec(type="zscore")], alpha=1.0),
            "ds",
        )
        state = load_preprocessing_state(result["model_id"])
        fill = state["steps"][0]["state"]["fill"]["test.gappy"]
        assert fill == pytest.approx(float(kept["panel"]["test.gappy"].median()))
        assert result["n_folds"] >= 2

    def test_an_estimator_that_accepts_nan_needs_no_impute(self, kept):
        assert accepts_missing(get_estimator_class("regression", "hist_gradient_boosting"))
        result = run_experiment(
            kept, _model("hist_gradient_boosting", max_iter=20), "ds", register=False
        )
        assert result["n_folds"] >= 2

    def test_the_capability_report_says_who_accepts_missing(self):
        by_name = {
            (e["task"], e["name"]): e["accepts_missing"]
            for e in modeling_capabilities()["estimators"]
        }
        assert by_name[("regression", "ridge")] is False
        assert by_name[("regression", "hist_gradient_boosting")] is True


class TestScoringUnderKeep:
    def test_the_latest_bar_with_a_hole_is_scored_through_impute(self, patched_multi_factory):
        """The planted feature is NaN on the final bar of every entity, so
        under `keep` the row scoring wants to score carries a hole."""
        model_id, _dataset = _register(
            _spec(MissingDataSpec(policy="keep")),
            _model("ridge", steps=[StepSpec(type="impute"), StepSpec(type="zscore")], alpha=1.0),
            "ds_keep_score",
        )
        result = score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)
        assert result["n_entities"] == len(UNIVERSE)
        assert result["stale_entities"] == {}
        predictions = _artifacts.load_artifact(result["predictions_uri"])["prediction"]
        assert np.isfinite(predictions.to_numpy()).all()

    def test_a_nan_tolerant_estimator_scores_the_hole_directly(self, patched_multi_factory):
        model_id, _dataset = _register(
            _spec(MissingDataSpec(policy="keep")),
            _model("hist_gradient_boosting", max_iter=20),
            "ds_keep_hgb",
        )
        result = score_model(model_id, as_of="2023-12-29", universe=UNIVERSE)
        assert result["n_entities"] == len(UNIVERSE)
