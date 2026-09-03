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


class TestTheTemporalContractIsCarriedAsABundle:
    """
    `DataBundle` pairs each frame with what its source can say about WHEN
    its rows became knowable, and `validate_bundle` turns that into a
    verdict.

    Both were built, tested and documented, and then called from nowhere.
    Repo-wide search found `DataBundle(` only in its own module and its own
    tests -- an abstraction in the state where it exists, passes, and
    nobody knows why. `build_dataset` is the right home for it: this is
    where frames and contracts first meet, and `join_point_in_time` later
    attaches records to exactly this panel.

    These tests pin the wiring rather than the verdict. The verdict itself
    is `data/bundle.py`'s to test; what matters here is that the call
    happens at all, because the failure mode being guarded against is
    silence.
    """

    def test_build_dataset_validates_a_bundle(self, patched_multi_factory):
        import standard_quant_tools.modeling.dataset.builder as builder

        seen = {}
        original = builder.validate_bundle

        def _spy(bundle, **kwargs):
            seen["bundle"] = bundle
            seen["kwargs"] = kwargs
            return original(bundle, **kwargs)

        builder.validate_bundle = _spy
        try:
            build_dataset(_spec())
        finally:
            builder.validate_bundle = original

        assert seen, (
            "build_dataset no longer validates a DataBundle. That call is "
            "the only production use of the abstraction -- without it, "
            "DataBundle is an orphan again."
        )
        assert "bars" in seen["bundle"].kinds

    def test_the_pit_requirement_is_relaxed_deliberately(self, patched_multi_factory):
        """
        No shipped provider reports `point_in_time=True`, so requiring it
        here would refuse every build. The relaxation has to be explicit
        and visible, not a default that drifted.
        """
        import standard_quant_tools.modeling.dataset.builder as builder

        seen = {}
        original = builder.validate_bundle

        def _spy(bundle, **kwargs):
            seen.update(kwargs)
            return original(bundle, **kwargs)

        builder.validate_bundle = _spy
        try:
            build_dataset(_spec())
        finally:
            builder.validate_bundle = original

        assert seen.get("require_pit") is False


class TestSeveralHorizonsFromOneBuild:
    """
    `horizons` computes the features ONCE and labels them at several
    distances. The alternative -- one dataset per horizon -- recomputes the
    same feature matrix N times and, worse, aligns each separately, so the
    resulting models are no longer comparable.
    """

    def test_a_single_horizon_panel_is_unchanged(self, patched_multi_factory):
        """
        Purely additive. Every spec written before `horizons` existed has
        one horizon, and emitting a duplicate `target__h5` beside `target`
        for those would change the shape of every dataset in existence to
        no purpose.
        """
        built = build_dataset(_spec())
        extra = [c for c in built["panel"].columns if c.startswith("target__")]
        assert extra == []
        assert built["targets"] == []

    def test_every_horizon_becomes_a_column(self, patched_multi_factory):
        built = build_dataset(_spec(target=TargetSpec(horizons=[1, 5, 20])))
        panel = built["panel"]
        for name in ("h1", "h5", "h20"):
            assert f"target__{name}" in panel.columns
            assert f"label_end_date__{name}" in panel.columns
        assert [t["name"] for t in built["targets"]] == ["h1", "h5", "h20"]

    def test_the_primary_is_the_shortest_and_is_the_plain_target(
        self, patched_multi_factory
    ):
        built = build_dataset(_spec(target=TargetSpec(horizons=[20, 1, 5])))
        assert built["target_id"] == "forward_return:1"
        panel = built["panel"]
        assert panel["target"].equals(panel["target__h1"])

    def test_each_column_is_that_horizons_forward_return(self, patched_multi_factory):
        """
        The columns must differ, and differ CORRECTLY. A loop that rebuilt
        the same horizon three times would produce three identical columns
        and every test above would still pass.
        """
        built = build_dataset(_spec(target=TargetSpec(horizons=[1, 5])))
        panel = built["panel"]
        assert not panel["target__h1"].equals(panel["target__h5"])
        # A 5-bar forward return is wider than a 1-bar one on the same bars.
        assert panel["target__h5"].std() > panel["target__h1"].std()

    def test_a_longer_horizon_keeps_its_own_unclosed_rows(self, patched_multi_factory):
        """
        Alignment drops on the PRIMARY label only. The 20-bar label has no
        value on the last 20 bars per entity, and those rows survive as
        NaN rather than costing the 1-bar model its data.
        """
        built = build_dataset(_spec(target=TargetSpec(horizons=[1, 20])))
        panel = built["panel"]
        assert panel["target__h1"].notna().all()
        assert panel["target__h20"].isna().any()

    def test_the_label_end_dates_differ_by_horizon(self, patched_multi_factory):
        built = build_dataset(_spec(target=TargetSpec(horizons=[1, 20])))
        panel = built["panel"].dropna(
            subset=["label_end_date__h1", "label_end_date__h20"]
        )
        assert (panel["label_end_date__h20"] > panel["label_end_date__h1"]).all(), (
            "a 20-bar label must finish observing after a 1-bar one, or the "
            "walk-forward purge is computed against the wrong window"
        )

    def test_the_spec_round_trips(self, patched_multi_factory):
        """`dataset_spec_hash` rebuilds a spec from its own dump to re-derive
        the hash, so a spec that cannot survive that is unusable."""
        from standard_quant_tools.modeling.dataset.builder import dataset_spec_hash

        spec = _spec(target=TargetSpec(horizons=[1, 5, 20]))
        rebuilt = DatasetSpec(**spec.model_dump())
        assert dataset_spec_hash(rebuilt) == dataset_spec_hash(spec)
        assert rebuilt.target.horizons == [1, 5, 20]

    def test_too_many_horizons_is_refused(self):
        with pytest.raises(Exception):
            TargetSpec(horizons=list(range(1, 20)))
