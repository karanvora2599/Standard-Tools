"""
Regression tests for per-feature drop attribution.

Feature/target alignment drops rows — every feature consumes its lookback
window and a forward-return target consumes its horizon — and that loss
was reported only as a final row count. A count cannot separate "this is
the warm-up I asked for" from "one feature is silently costing me two
thirds of my panel", and it cannot say which feature.

The pair of counts is the point. `n_missing` alone is misleading: warm-up
windows overlap, so the per-feature figures sum to far more than the rows
actually lost, and a short-lookback feature sitting entirely inside a
longer one looks equally guilty. `n_sole_missing` is what removing that
one feature would give back.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.models import BuildModelDatasetInput
from standard_quant_tools.modeling.agent.tools import build_model_dataset
from standard_quant_tools.modeling.dataset.alignment import (
    attribute_drops,
    stack_features_only,
    stack_long,
)
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.dataset.coverage import alignment_warnings
from standard_quant_tools.modeling.specs import DatasetSpec, FeatureSpec, TargetSpec

from .conftest import make_ohlcv, make_provider_mock


def _spec(features, universe=("AAA", "BBB"), horizon=5) -> DatasetSpec:
    return DatasetSpec(
        universe=list(universe),
        start="2022-01-01",
        end="2023-12-31",
        features=features,
        target=TargetSpec(horizon=horizon),
    )


def _patch(monkeypatch, fetch):
    from standard_quant_tools.data.factory import DataFactory

    provider = make_provider_mock(fetch)
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return provider


# ── attribute_drops in isolation ───────────────────────────────────────


class TestAttributeDrops:
    def _panel(self):
        # 6 rows. short_feature is NaN in rows 0-1, long_feature in rows
        # 0-3, target in row 5. So rows 0-1 have two causes, rows 2-3 have
        # long_feature alone, row 5 has the target alone.
        return pd.DataFrame(
            {
                "entity": ["AAA"] * 6,
                "short_feature": [np.nan, np.nan, 1.0, 1.0, 1.0, 1.0],
                "long_feature": [np.nan, np.nan, np.nan, np.nan, 1.0, 1.0],
                "target": [0.1] * 5 + [np.nan],
            }
        )

    def test_row_counts(self):
        result = attribute_drops(
            self._panel(), ["short_feature", "long_feature"], "target"
        )
        assert result["rows_before_alignment"] == 6
        assert result["rows_dropped"] == 5
        assert result["rows_after_alignment"] == 1

    def test_n_missing_counts_every_nan(self):
        result = attribute_drops(
            self._panel(), ["short_feature", "long_feature"], "target"
        )
        assert result["per_feature"]["short_feature"]["n_missing"] == 2
        assert result["per_feature"]["long_feature"]["n_missing"] == 4
        assert result["per_feature"]["target"]["n_missing"] == 1

    def test_sole_missing_is_what_removing_that_column_recovers(self):
        """The discriminating number. short_feature is NaN in two rows, but
        long_feature is NaN in both of them too — dropping short_feature
        from the spec would recover nothing."""
        result = attribute_drops(
            self._panel(), ["short_feature", "long_feature"], "target"
        )
        assert result["per_feature"]["short_feature"]["n_sole_missing"] == 0
        assert result["per_feature"]["long_feature"]["n_sole_missing"] == 2
        assert result["per_feature"]["target"]["n_sole_missing"] == 1

    def test_sole_missing_never_exceeds_n_missing(self):
        result = attribute_drops(
            self._panel(), ["short_feature", "long_feature"], "target"
        )
        for counts in result["per_feature"].values():
            assert counts["n_sole_missing"] <= counts["n_missing"]

    def test_sole_missing_totals_cannot_exceed_rows_dropped(self):
        """Sole causes are disjoint by construction: a row with one missing
        column is attributed to exactly one column."""
        result = attribute_drops(
            self._panel(), ["short_feature", "long_feature"], "target"
        )
        total_sole = sum(c["n_sole_missing"] for c in result["per_feature"].values())
        assert total_sole <= result["rows_dropped"]

    def test_per_entity_drops_are_counted(self):
        panel = pd.DataFrame(
            {
                "entity": ["AAA", "AAA", "BBB"],
                "f": [np.nan, 1.0, np.nan],
                "target": [0.1, 0.1, 0.1],
            }
        )
        result = attribute_drops(panel, ["f"], "target")
        assert result["per_entity_rows_dropped"] == {"AAA": 1, "BBB": 1}

    def test_a_complete_panel_reports_no_drops(self):
        panel = pd.DataFrame(
            {"entity": ["AAA"] * 3, "f": [1.0, 2.0, 3.0], "target": [0.1, 0.2, 0.3]}
        )
        result = attribute_drops(panel, ["f"], "target")
        assert result["rows_dropped"] == 0
        assert all(c["n_missing"] == 0 for c in result["per_feature"].values())

    def test_empty_panel_does_not_raise(self):
        result = attribute_drops(pd.DataFrame(), ["f"], "target")
        assert result["rows_dropped"] == 0


# ── The stackers return it ─────────────────────────────────────────────


class TestStackersReturnAttribution:
    def _features(self):
        index = pd.date_range("2022-01-01", periods=5, freq="B")
        return {
            "AAA": pd.DataFrame({"f": [np.nan, 1.0, 2.0, 3.0, 4.0]}, index=index),
        }

    def test_stack_long_returns_panel_and_attribution(self):
        index = pd.date_range("2022-01-01", periods=5, freq="B")
        panel, attribution = stack_long(
            self._features(),
            {"AAA": pd.Series([0.1, 0.1, 0.1, 0.1, np.nan], index=index)},
        )
        assert len(panel) == 3
        assert attribution["rows_before_alignment"] == 5
        assert attribution["per_feature"]["f"]["n_sole_missing"] == 1
        assert attribution["per_feature"]["target"]["n_sole_missing"] == 1

    def test_stack_features_only_returns_attribution(self):
        panel, attribution = stack_features_only(self._features())
        assert len(panel) == 4
        assert attribution["rows_dropped"] == 1

    def test_the_panel_itself_is_unchanged_by_the_refactor(self):
        """Attribution is additive: the aligned panel must be exactly what
        it was before, or every downstream hash and metric moves."""
        index = pd.date_range("2022-01-01", periods=5, freq="B")
        panel, _ = stack_long(
            self._features(),
            {"AAA": pd.Series([0.1] * 5, index=index)},
        )
        assert list(panel.columns) == ["date", "entity", "f", "target"]
        assert panel["f"].tolist() == [1.0, 2.0, 3.0, 4.0]


# ── End to end through build_dataset ───────────────────────────────────


class TestAttributionThroughTheBuilder:
    def test_the_long_lookback_feature_is_identified(self, monkeypatch):
        """The case this exists for: a 252-bar feature alongside a 14-bar
        one. Both have missing rows; only one is worth removing."""
        _patch(monkeypatch, lambda s: make_ohlcv(s, n=400))
        built = build_dataset(
            _spec(
                [
                    FeatureSpec(id="technical.rsi"),
                    FeatureSpec(id="risk.rolling_drawdown"),
                ]
            )
        )
        per_feature = built["drop_attribution"]["per_feature"]
        assert per_feature["risk.rolling_drawdown"]["n_sole_missing"] > 0
        # RSI's 14-bar warm-up sits entirely inside the 252-bar one, so
        # removing RSI recovers nothing — even though it has missing rows.
        assert per_feature["technical.rsi"]["n_missing"] > 0
        assert per_feature["technical.rsi"]["n_sole_missing"] == 0

    def test_the_target_is_attributed_separately(self, monkeypatch):
        _patch(monkeypatch, lambda s: make_ohlcv(s, n=400))
        built = build_dataset(_spec([FeatureSpec(id="technical.rsi")], horizon=20))
        per_feature = built["drop_attribution"]["per_feature"]
        assert per_feature["target"]["n_missing"] > 0

    def test_a_longer_horizon_costs_more_rows(self, monkeypatch):
        _patch(monkeypatch, lambda s: make_ohlcv(s, n=400))
        short = build_dataset(_spec([FeatureSpec(id="technical.rsi")], horizon=5))
        long = build_dataset(_spec([FeatureSpec(id="technical.rsi")], horizon=40))
        assert (
            long["drop_attribution"]["per_feature"]["target"]["n_missing"]
            > short["drop_attribution"]["per_feature"]["target"]["n_missing"]
        )

    def test_counts_reconcile_with_the_actual_panel(self, monkeypatch):
        """Guards the numbers themselves: rows_after_alignment must equal
        the panel that was returned, or the report describes a different
        build than the one that happened."""
        _patch(monkeypatch, lambda s: make_ohlcv(s, n=400))
        built = build_dataset(
            _spec(
                [
                    FeatureSpec(id="technical.rsi"),
                    FeatureSpec(id="risk.rolling_drawdown"),
                ]
            )
        )
        attribution = built["drop_attribution"]
        assert attribution["rows_after_alignment"] == len(built["panel"])
        assert attribution["rows_before_alignment"] - attribution[
            "rows_dropped"
        ] == len(built["panel"])


# ── entities now means "in the panel" ──────────────────────────────────


class TestEntitiesReflectsThePanel:
    def _short_history(self, symbol):
        return make_ohlcv(symbol, n=60 if symbol == "CCC" else 400)

    def test_a_dropped_entity_is_not_reported_as_covered(self, monkeypatch):
        """`entities` was the FETCHED list, so a symbol whose history could
        not survive alignment still appeared in the dataset's coverage —
        the model never saw a single row of it."""
        _patch(monkeypatch, self._short_history)
        built = build_dataset(
            _spec([FeatureSpec(id="risk.rolling_drawdown")], universe=["AAA", "CCC"])
        )
        assert "CCC" not in built["entities"]
        assert "CCC" in built["entities_fetched"]
        assert set(built["panel"]["entity"].unique()) == set(built["entities"])

    def test_the_dropped_entity_is_named_in_warnings(self, monkeypatch):
        _patch(monkeypatch, self._short_history)
        built = build_dataset(
            _spec([FeatureSpec(id="risk.rolling_drawdown")], universe=["AAA", "CCC"])
        )
        assert any("CCC" in w and "no rows" in w for w in built["warnings"])

    def test_entities_unchanged_when_everything_survives(self, monkeypatch):
        _patch(monkeypatch, lambda s: make_ohlcv(s, n=400))
        built = build_dataset(
            _spec([FeatureSpec(id="technical.rsi")], universe=["AAA", "BBB"])
        )
        assert built["entities"] == ["AAA", "BBB"]
        assert built["entities"] == built["entities_fetched"]


# ── Warnings ───────────────────────────────────────────────────────────


class TestAlignmentWarnings:
    def test_a_modest_warm_up_is_not_warned_about(self):
        """Every dataset loses its warm-up; warning about all of them would
        train the reader to skip the warnings that matter."""
        attribution = {
            "rows_before_alignment": 1000,
            "rows_dropped": 50,
            "per_feature": {"f": {"n_missing": 50, "n_sole_missing": 50}},
        }
        assert alignment_warnings(attribution, ["AAA"], ["AAA"]) == []

    def test_a_heavy_overall_loss_is_reported_with_a_breakdown(self):
        attribution = {
            "rows_before_alignment": 1000,
            "rows_dropped": 600,
            "per_feature": {
                "big": {"n_missing": 600, "n_sole_missing": 550},
                "small": {"n_missing": 40, "n_sole_missing": 0},
            },
        }
        (message,) = alignment_warnings(attribution, ["AAA"], ["AAA"])
        assert "60%" in message
        assert "big" in message
        assert "small" not in message, "a zero sole-cause feature is not the culprit"

    def test_a_single_expensive_feature_is_named_even_when_totals_look_fine(self):
        """Overall loss of 15% is unremarkable, but one removable feature
        costing 12% of the panel is worth knowing about."""
        attribution = {
            "rows_before_alignment": 1000,
            "rows_dropped": 150,
            "per_feature": {"greedy": {"n_missing": 150, "n_sole_missing": 120}},
        }
        (message,) = alignment_warnings(attribution, ["AAA"], ["AAA"])
        assert "greedy" in message

    def test_overlapping_warm_ups_say_so_instead_of_an_empty_breakdown(self):
        """When no single column is ever the sole cause, an empty list
        would read as a bug rather than as a finding."""
        attribution = {
            "rows_before_alignment": 1000,
            "rows_dropped": 600,
            "per_feature": {
                "a": {"n_missing": 600, "n_sole_missing": 0},
                "b": {"n_missing": 600, "n_sole_missing": 0},
            },
        }
        (message,) = alignment_warnings(attribution, ["AAA"], ["AAA"])
        assert "No single column" in message

    def test_a_dropped_entity_is_reported(self):
        attribution = {
            "rows_before_alignment": 100,
            "rows_dropped": 0,
            "per_feature": {},
        }
        (message,) = alignment_warnings(attribution, ["AAA", "CCC"], ["AAA"])
        assert "CCC" in message

    def test_no_warnings_on_a_clean_build(self):
        attribution = {
            "rows_before_alignment": 100,
            "rows_dropped": 0,
            "per_feature": {},
        }
        assert alignment_warnings(attribution, ["AAA"], ["AAA"]) == []


# ── The dead-end error now explains itself ─────────────────────────────


class TestEmptyPanelError:
    def test_the_error_names_the_expensive_column(self, monkeypatch):
        """ "No rows survive" left the caller to guess which feature was too
        long for the window they asked for."""
        _patch(monkeypatch, lambda s: make_ohlcv(s, n=60))
        with pytest.raises(ValidationError) as excinfo:
            build_dataset(
                _spec([FeatureSpec(id="risk.rolling_drawdown")], universe=["AAA"])
            )
        message = str(excinfo.value)
        assert "risk.rolling_drawdown" in message
        assert "no rows survive" in message


# ── It reaches the caller and is persisted ─────────────────────────────


class TestPropagation:
    def test_tool_result_carries_the_attribution(self, patched_multi_factory):
        result = build_model_dataset(
            BuildModelDatasetInput(spec=_spec([FeatureSpec(id="technical.rsi")]))
        )
        assert result.drop_attribution["rows_after_alignment"] == result.rows
        assert "technical.rsi" in result.drop_attribution["per_feature"]

    def test_it_is_persisted_with_the_dataset(self, patched_multi_factory):
        from standard_quant_tools.modeling import artifacts as _artifacts

        result = build_model_dataset(
            BuildModelDatasetInput(spec=_spec([FeatureSpec(id="technical.rsi")]))
        )
        meta = _artifacts.load_json(
            str(_artifacts.run_dir(result.dataset_id) / "dataset_meta.json")
        )
        assert meta["drop_attribution"] == result.drop_attribution
        assert meta["entities_fetched"] == ["AAA", "BBB"]
