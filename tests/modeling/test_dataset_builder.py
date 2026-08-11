"""Tests for modeling.dataset.builder.build_dataset: multi-symbol panel
shape/alignment, forward-return target correctness, and the
point-in-time safety check."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.dataset.leakage import PointInTimeViolation
from standard_quant_tools.modeling.dataset.target import build_target
from standard_quant_tools.modeling.features.base import (
    FeatureDefinition,
    FeatureScope,
    TemporalSupport,
)
from standard_quant_tools.modeling.features.registry import (
    FEATURE_REGISTRY,
    register_feature,
)
from standard_quant_tools.modeling.specs import DatasetSpec, FeatureSpec, TargetSpec

from .conftest import make_ohlcv


def _spec(**overrides) -> DatasetSpec:
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


class TestBuildDatasetPanelShape:
    def test_panel_has_expected_columns(self, patched_multi_factory):
        built = build_dataset(_spec())
        panel = built["panel"]
        assert list(panel.columns) == [
            "date",
            "entity",
            "technical.rsi",
            "market.momentum",
            "target",
            # Per-row date the forward-return label finishes observing.
            # Carried in the panel so walk-forward validation can purge
            # training rows whose label overlaps the test window (see
            # engine.py) -- an integer embargo cannot express this once
            # entities sit on different calendars.
            "label_end_date",
        ]

    def test_every_universe_entity_present(self, patched_multi_factory):
        built = build_dataset(_spec())
        assert set(built["panel"]["entity"].unique()) == {"AAA", "BBB", "CCC"}
        assert built["entities"] == ["AAA", "BBB", "CCC"]

    def test_no_nan_feature_or_target_rows(self, patched_multi_factory):
        built = build_dataset(_spec())
        panel = built["panel"]
        assert (
            not panel[["technical.rsi", "market.momentum", "target"]].isna().any().any()
        )

    def test_data_hash_deterministic_for_same_inputs(self, patched_multi_factory):
        built_1 = build_dataset(_spec())
        built_2 = build_dataset(_spec())
        assert built_1["data_hash"] == built_2["data_hash"]

    def test_include_target_false_has_no_target_column_and_more_rows(
        self, patched_multi_factory
    ):
        """Skipping target construction must not drop the most recent
        `horizon` bars the way training-mode's target-based dropna would."""
        with_target = build_dataset(_spec())
        without_target = build_dataset(_spec(), include_target=False)
        assert "target" not in without_target["panel"].columns
        assert without_target["target_id"] is None
        assert len(without_target["panel"]) > len(with_target["panel"])


class TestBuildTarget:
    def test_forward_return_matches_hand_computed(self):
        close = pd.Series([100.0, 102.0, 101.0, 105.0, 110.0, 108.0])
        target = build_target(close, TargetSpec(horizon=2))
        # value at t=0 should be (close[2]-close[0])/close[0]
        assert target.iloc[0] == pytest.approx((101.0 - 100.0) / 100.0)
        assert target.iloc[1] == pytest.approx((105.0 - 102.0) / 102.0)
        # last 2 rows have no future data -> NaN
        assert target.iloc[-1] != target.iloc[-1]  # NaN check
        assert target.iloc[-2] != target.iloc[-2]


class TestPointInTimeSafety:
    def test_current_only_feature_rejected(self, patched_multi_factory):
        register_feature(
            FeatureDefinition(
                id="test.current_only_dummy",
                description="d",
                fn=lambda ohlcv, context, **p: ohlcv["Close"] * 0.0,
                temporal_support=TemporalSupport.CURRENT_ONLY,
                scope=FeatureScope.ENTITY,
                lookback=0,
            )
        )
        try:
            spec = _spec(features=[FeatureSpec(id="test.current_only_dummy")])
            with pytest.raises(PointInTimeViolation):
                build_dataset(spec)
        finally:
            del FEATURE_REGISTRY["test.current_only_dummy"]

    def test_pit_violation_is_a_validation_error(self):
        assert issubclass(PointInTimeViolation, ValidationError)


class TestBuildDatasetValidation:
    def test_unknown_feature_id_raises(self, patched_multi_factory):
        with pytest.raises(ValidationError, match="unknown feature id"):
            build_dataset(_spec(features=[FeatureSpec(id="nonexistent.feature")]))

    def test_pca_feature_works_across_universe(self, patched_multi_factory):
        """Universe-scope features (factors.*) must be computed once over
        the shared panel and broadcast back into each entity's columns —
        not crash, not raise, not silently produce all-NaN columns."""
        spec = _spec(
            features=[
                FeatureSpec(id="technical.rsi"),
                FeatureSpec(id="factors.pca_loading"),
            ]
        )
        built = build_dataset(spec)
        assert "factors.pca_loading" in built["panel"].columns
        assert not built["panel"]["factors.pca_loading"].isna().all()


class TestFetchFailureHandling:
    """A raw provider exception used to propagate uncaught, with no
    indication of which symbol in a multi-symbol universe caused it."""

    def test_universe_symbol_fetch_failure_wrapped_with_symbol_context(
        self, monkeypatch
    ):
        provider = MagicMock()

        def _get_ohlcv(symbol, start, end):
            if symbol == "BBB":
                raise ConnectionError("network blip")
            return make_ohlcv(symbol)

        provider.get_ohlcv.side_effect = _get_ohlcv
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        with pytest.raises(ValidationError, match="'BBB'"):
            build_dataset(_spec())

    def test_universe_symbol_empty_data_raises_named_error(self, monkeypatch):
        provider = MagicMock()

        def _get_ohlcv(symbol, start, end):
            if symbol == "CCC":
                return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
            return make_ohlcv(symbol)

        provider.get_ohlcv.side_effect = _get_ohlcv
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        with pytest.raises(ValidationError, match="'CCC'"):
            build_dataset(_spec())

    def test_benchmark_fetch_failure_wrapped_with_context(self, monkeypatch):
        provider = MagicMock()

        def _get_ohlcv(symbol, start, end):
            if symbol == "SPY":
                raise ConnectionError("network blip")
            return make_ohlcv(symbol)

        provider.get_ohlcv.side_effect = _get_ohlcv
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        with pytest.raises(ValidationError, match="'SPY'"):
            build_dataset(_spec())

    def test_benchmark_empty_data_raises(self, monkeypatch):
        provider = MagicMock()

        def _get_ohlcv(symbol, start, end):
            if symbol == "SPY":
                return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
            return make_ohlcv(symbol)

        provider.get_ohlcv.side_effect = _get_ohlcv
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        with pytest.raises(ValidationError, match="'SPY'"):
            build_dataset(_spec())


class TestFiniteValueGuard:
    """dropna() removes NaN but not +/-inf -- a degenerate feature must
    be caught before it reaches sklearn, not silently accepted."""

    def test_inf_feature_value_rejected(self, patched_multi_factory):
        register_feature(
            FeatureDefinition(
                id="test.inf_dummy",
                description="d",
                fn=lambda ohlcv, context, **p: pd.Series(np.inf, index=ohlcv.index),
                temporal_support=TemporalSupport.PIT_SAFE,
                scope=FeatureScope.ENTITY,
                lookback=0,
            )
        )
        try:
            spec = _spec(features=[FeatureSpec(id="test.inf_dummy")])
            with pytest.raises(ValidationError, match="test.inf_dummy"):
                build_dataset(spec)
        finally:
            del FEATURE_REGISTRY["test.inf_dummy"]
