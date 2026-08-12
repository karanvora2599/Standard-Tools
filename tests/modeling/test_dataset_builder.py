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

from .conftest import make_ohlcv, make_provider_mock


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
        def _fetch(symbol):
            if symbol == "BBB":
                raise ConnectionError("network blip")
            return make_ohlcv(symbol)

        provider = make_provider_mock(_fetch)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        with pytest.raises(ValidationError, match="'BBB'") as excinfo:
            build_dataset(_spec())
        # The originating exception type survives into the message: a
        # ConnectionError and an unknown ticker are different problems with
        # different fixes, and the wrapper must not flatten them together.
        assert "ConnectionError" in str(excinfo.value)
        assert "network blip" in str(excinfo.value)

    def test_universe_symbol_empty_data_raises_named_error(self, monkeypatch):
        def _fetch(symbol):
            if symbol == "CCC":
                return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
            return make_ohlcv(symbol)

        provider = make_provider_mock(_fetch)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        with pytest.raises(ValidationError, match="'CCC'"):
            build_dataset(_spec())

    def test_every_failing_symbol_is_reported_not_just_the_first(self, monkeypatch):
        """asyncio.gather propagates only the FIRST exception and abandons
        the remaining tasks, so a universe with several bad tickers would
        surface one of them — in completion order, i.e. nondeterministically
        — and the caller would fix their universe one symbol per run."""

        def _fetch(symbol):
            if symbol in {"AAA", "CCC"}:
                raise ConnectionError(f"{symbol} is down")
            return make_ohlcv(symbol)

        provider = make_provider_mock(_fetch)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        with pytest.raises(ValidationError) as excinfo:
            build_dataset(_spec())
        message = str(excinfo.value)
        assert "'AAA'" in message and "'CCC'" in message
        assert "2 of 3" in message

    def test_benchmark_fetch_failure_wrapped_with_context(self, monkeypatch):
        def _fetch(symbol):
            if symbol == "SPY":
                raise ConnectionError("network blip")
            return make_ohlcv(symbol)

        provider = make_provider_mock(_fetch)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        with pytest.raises(ValidationError, match="'SPY'"):
            build_dataset(_spec())

    def test_benchmark_empty_data_raises(self, monkeypatch):
        def _fetch(symbol):
            if symbol == "SPY":
                return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
            return make_ohlcv(symbol)

        provider = make_provider_mock(_fetch)
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


class TestCustomFeatureOutputContract:
    """
    The ENTITY/UNIVERSE output contracts were documented for custom feature
    authors but never enforced: the return value went straight into a
    DataFrame constructor, so a wrong type or a foreign index surfaced as a
    generic pandas error several frames away with no mention of which
    feature produced it.
    """

    def _register(self, feature_id, fn, scope=FeatureScope.ENTITY):
        register_feature(
            FeatureDefinition(
                id=feature_id,
                description="d",
                fn=fn,
                temporal_support=TemporalSupport.PIT_SAFE,
                scope=scope,
                lookback=0,
            )
        )
        return feature_id

    def test_entity_feature_returning_ndarray_names_the_feature(
        self, patched_multi_factory
    ):
        fid = self._register(
            "test.bad_type_dummy",
            lambda ohlcv, context, **p: ohlcv["Close"].to_numpy(),
        )
        try:
            with pytest.raises(ValidationError, match=fid) as exc:
                build_dataset(_spec(features=[FeatureSpec(id=fid)]))
            assert "ndarray" in str(exc.value)
        finally:
            del FEATURE_REGISTRY[fid]

    def test_entity_feature_with_foreign_index_rejected(self, patched_multi_factory):
        """
        Labels the entity does not have either invent rows or mean the
        feature was computed against different data entirely.
        """
        fid = self._register(
            "test.foreign_index_dummy",
            lambda ohlcv, context, **p: pd.Series(
                1.0, index=pd.date_range("1999-01-01", periods=len(ohlcv), freq="D")
            ),
        )
        try:
            with pytest.raises(ValidationError, match="not in"):
                build_dataset(_spec(features=[FeatureSpec(id=fid)]))
        finally:
            del FEATURE_REGISTRY[fid]

    def test_shorter_output_is_allowed(self, patched_multi_factory):
        """
        The contract is SUBSET, not equality. A feature legitimately
        produces fewer rows than it consumes -- risk.rolling_beta works
        from returns, which lose the first bar to pct_change. Panel
        assembly is index-aligned, so the absent bars become NaN and the
        existing alignment step handles them.
        """
        fid = self._register(
            "test.short_output_dummy",
            lambda ohlcv, context, **p: ohlcv["Close"].iloc[5:] * 0.0 + 1.0,
        )
        try:
            result = build_dataset(_spec(features=[FeatureSpec(id=fid)]))
            assert fid in result["panel"].columns
        finally:
            del FEATURE_REGISTRY[fid]

    def test_universe_feature_missing_an_entity_names_it(self, patched_multi_factory):
        """Previously a bare KeyError from the per-entity assembly loop."""
        fid = self._register(
            "test.partial_universe_dummy",
            lambda returns, context, **p: returns.iloc[:, :1] * 0.0,
            scope=FeatureScope.UNIVERSE,
        )
        try:
            with pytest.raises(ValidationError, match="no values for"):
                build_dataset(_spec(features=[FeatureSpec(id=fid)]))
        finally:
            del FEATURE_REGISTRY[fid]

    def test_universe_feature_returning_series_rejected(self, patched_multi_factory):
        fid = self._register(
            "test.series_universe_dummy",
            lambda returns, context, **p: returns.iloc[:, 0],
            scope=FeatureScope.UNIVERSE,
        )
        try:
            with pytest.raises(ValidationError, match="one column per entity"):
                build_dataset(_spec(features=[FeatureSpec(id=fid)]))
        finally:
            del FEATURE_REGISTRY[fid]
