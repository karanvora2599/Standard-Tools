"""
A ragged universe is served group by group, and nothing moves.

Before the CHANGELOG entry of 2026-10-01 the panel fast path was all or
nothing: one entity whose bar index differed from the rest -- a late
listing, a delisting, a different holiday calendar -- sent EVERY entity to
the per-entity loop. Now the entities that share an index are stacked, one
panel call per such group, and only the entities that share it with nobody
are computed per entity.

The function below is the implementation that was replaced, verbatim, kept
as the reference. On an aligned universe the new one must return exactly
what it returned. On a ragged one -- where it returned {} -- every entity the
new one serves must equal that entity's own feature function, which is what
the per-entity loop computes, bit for bit.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.panel import HAS_CPP, technical_indicators_panel
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.dataset.panel_features import (
    _PANEL_FEATURES,
    _REQUIRED_COLUMNS,
    _aligned_groups,
    _batch,
    _extract,
    compute_panel_features,
)
from standard_quant_tools.modeling.features.base import FeatureContext
from standard_quant_tools.modeling.features.params import resolve_params
from standard_quant_tools.modeling.features.registry import get_feature
from standard_quant_tools.modeling.specs import FeatureSpec

from .conftest import make_ohlcv, make_provider_mock
from .test_panel_features import (
    PANEL_FEATURE_IDS,
    _assert_identical,
    _build_without_fast_path,
    _ids,
)

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="the fast path is native-only")


# ── The replaced implementation, kept as the reference ─────────────────────


def _reference_indices_identical(ohlcv_by_entity: Mapping[str, pd.DataFrame]) -> bool:
    reference: Optional[pd.Index] = None
    for frame in ohlcv_by_entity.values():
        if reference is None:
            reference = frame.index
        elif not reference.equals(frame.index):
            return False
    return reference is not None


def _reference_compute_panel_features(
    feature_specs: Sequence[Any],
    feature_defs: Sequence[Any],
    resolved_params: Sequence[Dict[str, Any]],
    ohlcv_by_entity: Mapping[str, pd.DataFrame],
) -> Dict[str, Dict[str, pd.Series]]:
    if not HAS_CPP:
        return {}
    if len(ohlcv_by_entity) < 2:
        return {}
    requests: List[Tuple[str, str, Dict[str, Any], Optional[str]]] = []
    for fs, definition, params in zip(feature_specs, feature_defs, resolved_params):
        mapping = _PANEL_FEATURES.get(definition.id)
        if mapping is None:
            continue
        indicator, param_map, field = mapping
        if set(params) - set(param_map):
            continue
        kwargs = {param_map[key]: value for key, value in params.items()}
        requests.append((fs.output_name, indicator, kwargs, field))
    if not requests:
        return {}
    if not _reference_indices_identical(ohlcv_by_entity):
        return {}
    if any(
        column not in frame.columns
        for frame in ohlcv_by_entity.values()
        for column in _REQUIRED_COLUMNS
    ):
        return {}
    feature_ids = {
        fs.output_name: definition.id
        for fs, definition in zip(feature_specs, feature_defs)
    }
    symbols = list(ohlcv_by_entity)
    out: Dict[str, Dict[str, pd.Series]] = {}
    for batch in _batch(requests):
        kwargs: Dict[str, Any] = {}
        for _, _, batch_kwargs, _ in batch:
            kwargs.update(batch_kwargs)
        indicators = [indicator for _, indicator, _, _ in batch]
        panel = technical_indicators_panel(
            ohlcv_by_entity, indicators=indicators, **kwargs
        )
        for output_name, indicator, _, field in batch:
            frame = panel[indicator]
            out[output_name] = {
                symbol: _extract(
                    feature_ids[output_name],
                    field,
                    frame,
                    symbol,
                    ohlcv_by_entity[symbol]["Close"],
                )
                for symbol in symbols
            }
    return out


# ── Helpers ──────────────────────────────────────────────────────────────


def _resolved(features):
    specs = [f if isinstance(f, FeatureSpec) else FeatureSpec(id=f) for f in features]
    defs = [get_feature(fs.id) for fs in specs]
    params = [resolve_params(d, fs.params) for fs, d in zip(specs, defs)]
    return specs, defs, params


def _loop(specs, defs, params, ohlcv):
    """Every feature of every entity through its own feature function --
    what the per-entity loop in build_dataset computes."""
    context = FeatureContext(interval="1d")
    return {
        fs.output_name: {s: d.fn(frame, context, **p) for s, frame in ohlcv.items()}
        for fs, d, p in zip(specs, defs, params)
    }


def _same_series(left: pd.Series, right: pd.Series) -> None:
    assert left.index.equals(right.index)
    a, b = left.to_numpy(dtype=float), right.to_numpy(dtype=float)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])


def _assert_served_exactly(out, specs, defs, params, ohlcv) -> None:
    served = {symbol for by_symbol in out.values() for symbol in by_symbol}
    truth = _loop(specs, defs, params, {s: ohlcv[s] for s in ohlcv if s in served})
    for name, by_symbol in out.items():
        for symbol, series in by_symbol.items():
            _same_series(series, truth[name][symbol])


def _served(out) -> Dict[str, List[str]]:
    return {name: sorted(by_symbol) for name, by_symbol in out.items()}


def _ragged(n_aligned=6, n=320):
    """`n_aligned` entities on one index, plus two late listings that share
    a start date, one late listing of its own, and one early delisting."""
    ohlcv = {f"A{i}": make_ohlcv(f"A{i}", n) for i in range(n_aligned)}
    ohlcv["LATE1"] = make_ohlcv("LATE1", n).iloc[60:]
    ohlcv["LATE2"] = make_ohlcv("LATE2", n).iloc[60:]
    ohlcv["IPO"] = make_ohlcv("IPO", n).iloc[97:]
    ohlcv["GONE"] = make_ohlcv("GONE", n).iloc[:-41]
    return ohlcv


# ── Tests ────────────────────────────────────────────────────────────────


class TestGrouping:
    def test_groups_follow_the_index_and_keep_universe_order(self):
        ohlcv = _ragged(n_aligned=3)
        assert _aligned_groups(ohlcv) == [
            ["A0", "A1", "A2"],
            ["LATE1", "LATE2"],
            ["IPO"],
            ["GONE"],
        ]
        interleaved = {k: ohlcv[k] for k in ("LATE1", "A0", "IPO", "A1", "LATE2")}
        assert _aligned_groups(interleaved) == [
            ["LATE1", "LATE2"],
            ["A0", "A1"],
            ["IPO"],
        ]

    def test_same_ends_and_length_are_not_enough(self):
        """Two histories with the same first bar, last bar and length but a
        different bar in the middle (a holiday) are different indices."""
        a = make_ohlcv("HOL1", 200)
        b = make_ohlcv("HOL2", 200)
        dates = b.index.to_list()
        dates[100] = dates[100] + pd.Timedelta(hours=12)
        b.index = pd.DatetimeIndex(dates)
        assert len(a) == len(b)
        assert (a.index[0], a.index[-1]) == (b.index[0], b.index[-1])
        assert _aligned_groups({"HOL1": a, "HOL2": b}) == [["HOL1"], ["HOL2"]]
        c = make_ohlcv("HOL3", 200)
        assert _aligned_groups({"HOL1": a, "HOL2": b, "HOL3": c}) == [
            ["HOL1", "HOL3"],
            ["HOL2"],
        ]


class TestRaggedUniverse:
    def test_shared_indices_are_served_and_the_rest_left_to_the_loop(self):
        """Planted: six aligned names and two late listings on one date are
        served; the lone late listing and the delisting are not."""
        ohlcv = _ragged()
        specs, defs, params = _resolved(PANEL_FEATURE_IDS)
        assert _reference_compute_panel_features(specs, defs, params, ohlcv) == {}
        out = compute_panel_features(specs, defs, params, ohlcv)
        served = sorted([f"A{i}" for i in range(6)] + ["LATE1", "LATE2"])
        assert _served(out) == {fs.output_name: served for fs in specs}
        _assert_served_exactly(out, specs, defs, params, ohlcv)

    @pytest.mark.parametrize("seed", range(4))
    def test_random_universes_match_the_reference_and_the_loop(self, seed):
        """Random raggedness and parameters, with one indicator at two
        periods so it needs two panel calls per group."""
        rng = np.random.default_rng(seed)
        n = 260
        ohlcv = {}
        for i in range(int(rng.integers(3, 9))):
            frame = make_ohlcv(f"R{seed}{i}", n)
            cut = int(rng.choice([0, 0, 0, 25, 25, 70]))
            if cut and rng.random() < 0.5:
                frame = frame.iloc[cut:]
            elif cut:
                frame = frame.iloc[: n - cut]
            ohlcv[f"R{seed}{i}"] = frame
        features = [
            FeatureSpec(
                id="technical.rsi", params={"period": int(rng.integers(5, 30))}
            ),
            FeatureSpec(id="technical.rsi", params={"period": 40}, alias="rsi_slow"),
            FeatureSpec(
                id="technical.adx", params={"period": int(rng.integers(7, 21))}
            ),
            FeatureSpec(id="technical.stochastic_k"),
            FeatureSpec(id="risk.atr_pct"),
            FeatureSpec(
                id="risk.bollinger_pct_b",
                params={"period": int(rng.integers(10, 30)), "num_std": 1.5},
            ),
        ]
        specs, defs, params = _resolved(features)
        out = compute_panel_features(specs, defs, params, ohlcv)
        _assert_served_exactly(out, specs, defs, params, ohlcv)
        reference = _reference_compute_panel_features(specs, defs, params, ohlcv)
        for name, by_symbol in reference.items():
            for symbol, series in by_symbol.items():
                _same_series(out[name][symbol], series)
        groups = [g for g in _aligned_groups(ohlcv) if len(g) >= 2]
        expected = sorted(s for g in groups for s in g)
        assert all(served == expected for served in _served(out).values())
        if reference:
            assert _served(out) == _served(reference)

    def test_none_ragged_is_the_reference_exactly(self):
        ohlcv = {s: make_ohlcv(s) for s in ("AAA", "BBB", "CCC", "DDD")}
        specs, defs, params = _resolved(PANEL_FEATURE_IDS)
        out = compute_panel_features(specs, defs, params, ohlcv)
        reference = _reference_compute_panel_features(specs, defs, params, ohlcv)
        assert reference and _served(out) == _served(reference)
        for name, by_symbol in reference.items():
            assert list(out[name]) == list(by_symbol)
            for symbol, series in by_symbol.items():
                _same_series(out[name][symbol], series)

    def test_all_ragged_serves_nothing(self):
        """The null case: no two entities share an index."""
        ohlcv = {f"S{i}": make_ohlcv(f"S{i}").iloc[10 * i :] for i in range(5)}
        specs, defs, params = _resolved(PANEL_FEATURE_IDS)
        assert compute_panel_features(specs, defs, params, ohlcv) == {}

    def test_one_entity_serves_nothing(self):
        specs, defs, params = _resolved(PANEL_FEATURE_IDS)
        assert compute_panel_features(specs, defs, params, {"A": make_ohlcv("A")}) == {}

    def test_no_panel_feature_serves_nothing(self):
        specs, defs, params = _resolved(["market.momentum"])
        assert compute_panel_features(specs, defs, params, _ragged()) == {}
        assert compute_panel_features([], [], [], _ragged()) == {}

    @pytest.mark.parametrize("n_bars", [2, 5, 15, 30])
    def test_histories_shorter_than_the_longest_window(self, n_bars):
        """ADX needs about twice its period before its first value; a group
        shorter than that must still come out exactly as its loop would."""
        ohlcv = {s: make_ohlcv(s, n_bars) for s in ("SHORT1", "SHORT2")}
        ohlcv.update({s: make_ohlcv(s) for s in ("LONG1", "LONG2")})
        specs, defs, params = _resolved(PANEL_FEATURE_IDS)
        out = compute_panel_features(specs, defs, params, ohlcv)
        assert all(
            served == ["LONG1", "LONG2", "SHORT1", "SHORT2"]
            for served in _served(out).values()
        )
        _assert_served_exactly(out, specs, defs, params, ohlcv)

    def test_an_entity_missing_a_stacked_column_is_left_to_the_loop(self):
        ohlcv = {s: make_ohlcv(s) for s in ("AAA", "BBB", "CCC")}
        ohlcv["CCC"] = ohlcv["CCC"].drop(columns=["High"])
        specs, defs, params = _resolved(["technical.rsi"])
        assert _reference_compute_panel_features(specs, defs, params, ohlcv) == {}
        out = compute_panel_features(specs, defs, params, ohlcv)
        assert _served(out) == {"technical.rsi": ["AAA", "BBB"]}
        _assert_served_exactly(out, specs, defs, params, ohlcv)

    def test_a_refused_group_is_handed_back_to_the_loop(self):
        """
        An infinity in an aligned entity of a RAGGED universe: the panel
        refuses that group, and the per-entity loop -- which computed every
        entity of such a universe before -- answers for it instead, with its
        own refusal. Stacked whole, an aligned universe meets the panel's
        refusal, as it always did.
        """
        ohlcv = _ragged(n_aligned=3)
        bad = ohlcv["A1"].copy()
        bad.iloc[100, bad.columns.get_loc("High")] = np.inf
        ohlcv["A1"] = bad
        specs, defs, params = _resolved(PANEL_FEATURE_IDS)
        out = compute_panel_features(specs, defs, params, ohlcv)
        assert all(served == ["LATE1", "LATE2"] for served in _served(out).values())
        _assert_served_exactly(out, specs, defs, params, ohlcv)
        # The loop is what answers for the refused group, and it refuses too.
        with pytest.raises(ValidationError, match="infinite"):
            _loop(specs, defs, params, {"A1": ohlcv["A1"]})

        aligned = {k: ohlcv[k] for k in ("A0", "A1", "A2")}
        with pytest.raises(ValidationError, match="technical_indicators_panel"):
            compute_panel_features(specs, defs, params, aligned)
        with pytest.raises(ValidationError, match="technical_indicators_panel"):
            _reference_compute_panel_features(specs, defs, params, aligned)

    def test_a_ragged_build_is_identical_to_the_loop(self, monkeypatch):
        """The whole dataset -- panel and content hash -- over a universe
        with late listings, a shared late start and a delisting."""
        universe = _ragged(n_aligned=4, n=500)
        provider = make_provider_mock(
            lambda symbol: universe.get(symbol, make_ohlcv(symbol, 500))
        )
        monkeypatch.setattr(
            "standard_quant_tools.data.factory.DataFactory.get_provider",
            lambda *a, **kw: provider,
        )
        spec = _ids(list(universe), PANEL_FEATURE_IDS + ["market.momentum"])
        fast = build_dataset(spec)
        _assert_identical(fast, _build_without_fast_path(spec, monkeypatch))

    def test_a_ragged_build_refuses_what_the_loop_refused(self, monkeypatch):
        """An infinity in an aligned entity of a ragged universe: the build
        raises the per-entity loop's error, word for word, as it did when
        the whole universe went to the loop."""
        universe = _ragged(n_aligned=3, n=500)
        bad = universe["A1"].copy()
        bad.iloc[250, bad.columns.get_loc("High")] = np.inf
        universe["A1"] = bad
        provider = make_provider_mock(
            lambda symbol: universe.get(symbol, make_ohlcv(symbol, 500))
        )
        monkeypatch.setattr(
            "standard_quant_tools.data.factory.DataFactory.get_provider",
            lambda *a, **kw: provider,
        )
        spec = _ids(list(universe), PANEL_FEATURE_IDS)
        with pytest.raises(Exception) as fast:
            build_dataset(spec)
        with pytest.raises(Exception) as slow:
            _build_without_fast_path(spec, monkeypatch)
        assert type(fast.value) is type(slow.value)
        assert str(fast.value) == str(slow.value)
