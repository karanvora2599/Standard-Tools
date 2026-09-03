"""
Where an entity sits in its universe's correlation graph.

WHAT THESE PIN.

  1. MST degree measures LOCAL topology and is not recoverable from PC1.
     A star universe -- one hub, spokes independent of each other -- has a
     hub of degree n-1 and spokes of degree 1, which is the structure the
     feature exists to report.
  2. The edge weight is a DISTANCE, sqrt(2(1-rho)), not a correlation. A
     tree built on a non-metric weight spans nothing in particular.
  3. Mean correlation excludes the diagonal. Including the self-correlation
     of 1 drags every entity's value toward it and compresses the spread
     the feature exists to show.
  4. Both are point-in-time: a value at bar t is estimated from bars at or
     before t, never carried backward from a later refit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.features.network import (
    MIN_WINDOW,
    _avg_correlation,
    _avg_correlation_at,
    _mst_degree,
    _mst_degree_at,
)


def _star_universe(n_spokes=4, n=400, seed=0) -> pd.DataFrame:
    """A hub every spoke correlates with, and spokes that do not correlate
    with each other except through the hub."""
    rng = np.random.default_rng(seed)
    hub = rng.normal(0, 1, n)
    columns = {"HUB": hub}
    for i in range(n_spokes):
        columns[f"S{i}"] = 0.9 * hub + 0.44 * rng.normal(0, 1, n)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame(columns, index=dates)


class TestMSTDegreeFindsTopology:
    def test_a_star_universe_has_one_hub(self) -> None:
        """
        The structure the feature exists to report. Every spoke's nearest
        neighbour is the hub, so the tree is a star and the hub's degree is
        the number of spokes.
        """
        panel = _star_universe(n_spokes=4)
        degrees = _mst_degree_at(panel)
        assert degrees["HUB"] == 4.0
        assert all(degrees[f"S{i}"] == 1.0 for i in range(4))

    def test_a_chain_universe_has_no_hub(self) -> None:
        """
        Each name correlates with its neighbours and not across the chain,
        so the tree is a path: interior degree 2, ends degree 1, and NO
        entity stands out. A star and a chain can share a PC1.
        """
        rng = np.random.default_rng(1)
        n = 600
        base = rng.normal(0, 1, (n, 6))
        columns = {}
        previous = base[:, 0]
        for i in range(6):
            # Each link is mostly its predecessor plus fresh noise, so
            # correlation decays with distance along the chain.
            previous = 0.85 * previous + 0.53 * base[:, i]
            columns[f"N{i}"] = previous.copy()
        panel = pd.DataFrame(
            columns, index=pd.date_range("2022-01-01", periods=n, freq="B")
        )
        degrees = _mst_degree_at(panel)
        assert degrees.max() <= 2.0
        # A tree over k nodes has k-1 edges, so the degrees sum to 2(k-1).
        assert degrees.sum() == 2.0 * (len(degrees) - 1)

    def test_every_tree_has_exactly_k_minus_one_edges(self) -> None:
        panel = _star_universe(n_spokes=6)
        assert _mst_degree_at(panel).sum() == 2.0 * 6

    def test_an_unestimable_pair_does_not_pull_the_tree(self) -> None:
        """
        A NaN correlation becomes the MAXIMUM distance, not zero. Zero would
        read as perfect correlation and route the tree through a pair that
        was never measured.
        """
        panel = _star_universe(n_spokes=3)
        panel["DEAD"] = np.nan
        degrees = _mst_degree_at(panel)
        # The all-NaN column is dropped before the matrix is built, so it
        # is absent rather than a spurious hub.
        assert "DEAD" not in degrees.index
        assert degrees["HUB"] == 3.0


class TestAverageCorrelation:
    def test_the_diagonal_is_excluded(self) -> None:
        """
        Including the self-correlation of 1.0 pulls every entity toward it.
        Two independent series have a mean correlation near ZERO, not near
        0.5, which is what the diagonal would produce for a pair.
        """
        rng = np.random.default_rng(2)
        panel = pd.DataFrame(
            {"A": rng.normal(0, 1, 500), "B": rng.normal(0, 1, 500)},
            index=pd.date_range("2022-01-01", periods=500, freq="B"),
        )
        values = _avg_correlation_at(panel)
        assert abs(values["A"]) < 0.15
        assert abs(values["B"]) < 0.15

    def test_a_correlated_pair_scores_high_and_a_loner_scores_low(self) -> None:
        rng = np.random.default_rng(3)
        shared = rng.normal(0, 1, 500)
        panel = pd.DataFrame(
            {
                "A": shared + 0.1 * rng.normal(0, 1, 500),
                "B": shared + 0.1 * rng.normal(0, 1, 500),
                "LONER": rng.normal(0, 1, 500),
            },
            index=pd.date_range("2022-01-01", periods=500, freq="B"),
        )
        values = _avg_correlation_at(panel)
        assert values["A"] > 0.4 and values["B"] > 0.4
        assert values["LONER"] < 0.2

    def test_it_is_scale_free_unlike_a_pc1_loading(self) -> None:
        """
        The reason this is not the same feature as factors.pca_loading:
        multiplying one series by 100 changes its variance contribution
        entirely and its mean correlation not at all.
        """
        rng = np.random.default_rng(4)
        shared = rng.normal(0, 1, 500)
        index = pd.date_range("2022-01-01", periods=500, freq="B")
        base = pd.DataFrame(
            {
                "A": shared + 0.3 * rng.normal(0, 1, 500),
                "B": shared + 0.3 * rng.normal(0, 1, 500),
                "C": shared + 0.3 * rng.normal(0, 1, 500),
            },
            index=index,
        )
        loud = base.copy()
        loud["A"] = loud["A"] * 100.0
        assert _avg_correlation_at(base)["A"] == pytest.approx(
            _avg_correlation_at(loud)["A"], abs=1e-9
        )


class TestTheRollingRefit:
    def test_a_value_uses_only_bars_at_or_before_it(self) -> None:
        """
        Point-in-time: truncating the panel after bar t must not change the
        value reported AT bar t. A refit that wrote forward would fail this.
        """
        panel = _star_universe(n_spokes=3, n=400)
        full = _mst_degree(panel, None, window=126, refit_every=21)
        cut = 200
        truncated = _mst_degree(panel.iloc[: cut + 1], None, window=126, refit_every=21)
        assert np.allclose(
            full["HUB"].to_numpy()[: cut + 1],
            truncated["HUB"].to_numpy(),
            equal_nan=True,
        )

    def test_the_warmup_is_the_window(self) -> None:
        panel = _star_universe(n_spokes=3, n=300)
        out = _avg_correlation(panel, None, window=126, refit_every=21)
        assert out["HUB"].iloc[:125].isna().all()
        assert out["HUB"].iloc[125:].notna().all()

    def test_values_are_held_between_refits_not_interpolated(self) -> None:
        panel = _star_universe(n_spokes=3, n=300)
        out = _avg_correlation(panel, None, window=126, refit_every=21)
        held = out["HUB"].iloc[125:146]
        assert held.nunique() == 1, "a held value must not drift between refits"

    def test_the_output_covers_every_entity_and_bar(self) -> None:
        panel = _star_universe(n_spokes=3, n=300)
        out = _mst_degree(panel, None, window=126, refit_every=21)
        assert list(out.columns) == list(panel.columns)
        assert len(out) == len(panel)


class TestTheWindowIsChecked:
    def test_a_window_too_short_to_estimate_is_refused(self) -> None:
        panel = _star_universe(n_spokes=3, n=100)
        with pytest.raises(ValidationError, match="sampling noise"):
            _avg_correlation(panel, None, window=MIN_WINDOW - 1, refit_every=5)

    def test_a_zero_refit_step_is_refused_not_a_range_error(self) -> None:
        panel = _star_universe(n_spokes=3, n=100)
        with pytest.raises(ValidationError, match="refit_every"):
            _mst_degree(panel, None, window=50, refit_every=0)


class TestRegisteredAndUsable:
    def test_both_are_in_the_catalog_as_universe_scope(self) -> None:
        from standard_quant_tools.modeling.features.base import FeatureScope
        from standard_quant_tools.modeling.features.registry import get_feature

        for feature_id in ("network.avg_correlation", "network.mst_degree"):
            definition = get_feature(feature_id)
            assert definition.scope is FeatureScope.UNIVERSE
            assert definition.lookback == 126

    def test_a_model_can_be_built_on_them(self, patched_multi_factory) -> None:
        from standard_quant_tools.modeling.agent import (
            BuildModelDatasetInput,
            build_model_dataset,
        )
        from standard_quant_tools.modeling.specs import (
            DatasetSpec,
            FeatureSpec,
            TargetSpec,
        )

        result = build_model_dataset(
            BuildModelDatasetInput(
                spec=DatasetSpec(
                    universe=["AAA", "BBB", "CCC"],
                    start="2022-01-01",
                    end="2023-12-31",
                    features=[
                        FeatureSpec(
                            id="network.avg_correlation",
                            params={"window": 60, "refit_every": 21},
                        ),
                        FeatureSpec(
                            id="network.mst_degree",
                            params={"window": 60, "refit_every": 21},
                        ),
                    ],
                    target=TargetSpec(horizon=5),
                    benchmark="SPY",
                )
            )
        )
        assert result.rows > 0
        assert result.feature_ids == [
            "network.avg_correlation",
            "network.mst_degree",
        ]
