"""
The panel feature fast path must be invisible in the output.

build_dataset routes the technical features through one whole-universe
native call when every entity shares a bar index. That is a speed change
only, so the bar it has to clear is not "close enough" but bit-identical
panels AND an identical dataset content hash — the hash is what a stored
provenance record is compared against, so a fast path that moved it would
invalidate every prior record for no reason.

The guard is the interesting half. technical_indicators_panel stacks the
universe onto the INTERSECTION of every ticker's bars, and the indicators
involved are path-dependent (Wilder smoothing, EMAs), so a truncated
history changes values rather than merely coverage. On a ragged universe
the fast path must therefore decline to engage at all.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling.dataset import builder as builder_module
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.dataset.panel_features import (
    compute_panel_features,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    FeatureSpec,
    TargetSpec,
)

from .conftest import make_ohlcv, make_provider_mock

# Every feature the fast path knows how to batch.
PANEL_FEATURE_IDS = [
    "technical.rsi",
    "technical.adx",
    "technical.stochastic_k",
    "risk.atr_pct",
    "risk.bollinger_pct_b",
]


def _spec(universe, features):
    return DatasetSpec(
        universe=universe,
        start="2022-01-01",
        end="2030-01-01",
        features=features,
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )


def _ids(universe, feature_ids):
    return _spec(universe, [FeatureSpec(id=f) for f in feature_ids])


def _build_without_fast_path(spec, monkeypatch):
    monkeypatch.setattr(builder_module, "compute_panel_features", lambda *a, **k: {})
    try:
        return build_dataset(spec)
    finally:
        monkeypatch.undo()


def _assert_identical(fast, slow):
    fast_panel, slow_panel = fast["panel"], slow["panel"]
    assert list(fast_panel.columns) == list(slow_panel.columns)
    assert fast_panel.shape == slow_panel.shape
    for column in fast_panel.columns:
        left, right = fast_panel[column], slow_panel[column]
        if left.dtype.kind in "fc":
            left_values, right_values = left.to_numpy(), right.to_numpy()
            # Bit-identical, not approximately equal: the two paths run the
            # same kernels on the same bars, so any difference at all means
            # they are not actually equivalent.
            np.testing.assert_array_equal(np.isnan(left_values), np.isnan(right_values))
            finite = ~np.isnan(left_values)
            np.testing.assert_array_equal(left_values[finite], right_values[finite])
        else:
            assert left.equals(right), column
    assert fast["data_hash"] == slow["data_hash"]


class TestPanelFastPathIsExact:
    @pytest.mark.parametrize("n_entities", [2, 3, 6])
    def test_matches_the_per_entity_loop(
        self, patched_multi_factory, monkeypatch, n_entities
    ):
        universe = [f"SYM{i}" for i in range(n_entities)]
        spec = _ids(universe, PANEL_FEATURE_IDS)
        fast = build_dataset(spec)
        slow = _build_without_fast_path(spec, monkeypatch)
        _assert_identical(fast, slow)

    def test_mixed_panel_and_non_panel_features(
        self, patched_multi_factory, monkeypatch
    ):
        """Features the fast path does not know (macd_histogram, momentum)
        must still be computed per entity, in the same frame."""
        spec = _ids(
            ["AAA", "BBB", "CCC"],
            PANEL_FEATURE_IDS + ["technical.macd_histogram", "market.momentum"],
        )
        fast = build_dataset(spec)
        slow = _build_without_fast_path(spec, monkeypatch)
        _assert_identical(fast, slow)

    def test_non_default_parameters_are_routed_not_ignored(
        self, patched_multi_factory, monkeypatch
    ):
        """
        The panel call takes per-indicator keywords (rsi_period, ...). A
        parameter that failed to be translated would silently compute the
        DEFAULT instead — a wrong number, not an error — so this pins that
        custom parameters survive the trip.
        """
        spec = _spec(
            ["AAA", "BBB", "CCC"],
            [
                FeatureSpec(id="technical.rsi", params={"period": 7}),
                FeatureSpec(id="technical.adx", params={"period": 21}),
                FeatureSpec(
                    id="risk.bollinger_pct_b", params={"period": 30, "num_std": 1.5}
                ),
            ],
        )
        fast = build_dataset(spec)
        slow = _build_without_fast_path(spec, monkeypatch)
        _assert_identical(fast, slow)

    def test_same_feature_twice_at_different_periods(
        self, patched_multi_factory, monkeypatch
    ):
        """Two aliases of one indicator cannot share a single panel call,
        since the period is a per-indicator keyword. They must be split
        across calls rather than one silently overwriting the other."""
        spec = _spec(
            ["AAA", "BBB", "CCC"],
            [
                FeatureSpec(id="technical.rsi", params={"period": 7}, alias="rsi_fast"),
                FeatureSpec(
                    id="technical.rsi", params={"period": 28}, alias="rsi_slow"
                ),
            ],
        )
        fast = build_dataset(spec)
        slow = _build_without_fast_path(spec, monkeypatch)
        _assert_identical(fast, slow)
        # And the two aliases must genuinely differ, or the test above
        # would pass on two identical columns.
        panel = fast["panel"]
        assert not np.allclose(
            panel["rsi_fast"].to_numpy(), panel["rsi_slow"].to_numpy()
        )


class TestPanelFastPathGuard:
    def test_declines_on_a_ragged_universe(self, monkeypatch):
        """
        A late-listing entity makes the panel intersection shorter than
        some entities' own history. Because the indicators are
        path-dependent, computing on the truncated index would change
        values — so the fast path must decline entirely.
        """

        def fetch(symbol):
            frame = make_ohlcv(symbol)
            return frame.iloc[120:] if symbol == "LATE" else frame

        provider = make_provider_mock(fetch)
        monkeypatch.setattr(
            "standard_quant_tools.data.factory.DataFactory.get_provider",
            lambda *a, **kw: provider,
        )
        spec = _ids(["AAA", "BBB", "LATE"], PANEL_FEATURE_IDS)

        calls = {}
        real = builder_module.compute_panel_features

        def spy(*args, **kwargs):
            out = real(*args, **kwargs)
            calls["n_features"] = len(out)
            return out

        monkeypatch.setattr(builder_module, "compute_panel_features", spy)
        build_dataset(spec)
        assert calls["n_features"] == 0

    def test_declines_for_a_single_entity(self, patched_multi_factory):
        """Stacking a one-ticker 'panel' costs more than it saves."""
        from standard_quant_tools.modeling.features.registry import get_feature

        spec = _ids(["AAA"], PANEL_FEATURE_IDS)
        feature_defs = [get_feature(fs.id) for fs in spec.features]
        params = [dict(d.default_params) for d in feature_defs]
        ohlcv = {"AAA": make_ohlcv("AAA")}
        assert compute_panel_features(spec.features, feature_defs, params, ohlcv) == {}

    def test_unmappable_parameter_disqualifies_the_feature(self, patched_multi_factory):
        """
        A parameter with no panel keyword must make the feature fall back,
        not be dropped from the call and computed at its default.
        """
        from standard_quant_tools.modeling.features.registry import get_feature

        spec = _ids(["AAA", "BBB"], ["technical.rsi"])
        definition = get_feature("technical.rsi")
        ohlcv = {s: make_ohlcv(s) for s in ("AAA", "BBB")}
        out = compute_panel_features(
            spec.features,
            [definition],
            [{"period": 14, "not_a_panel_kwarg": 3}],
            ohlcv,
        )
        assert out == {}

    def test_missing_ohlc_column_disqualifies(self, patched_multi_factory):
        """The stacker needs High/Low/Close for every ticker even when the
        requested indicator only reads Close."""
        from standard_quant_tools.modeling.features.registry import get_feature

        spec = _ids(["AAA", "BBB"], ["technical.rsi"])
        definition = get_feature("technical.rsi")
        ohlcv = {s: make_ohlcv(s).drop(columns=["High"]) for s in ("AAA", "BBB")}
        assert (
            compute_panel_features(spec.features, [definition], [{"period": 14}], ohlcv)
            == {}
        )
