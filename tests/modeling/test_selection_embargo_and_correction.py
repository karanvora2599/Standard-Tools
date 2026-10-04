"""
`select_features` leaves the label horizon between its selection window and
its holdout, and can correct its gate for the number of features tested
(the CHANGELOG entry of 2026-10-04).

A label dated on one of the selection window's last h dates looks h bars
forward, into the dates the holdout is scored on, so the selection IC and
the significance test read outcomes the holdout shares. What these tests
hold:

- `embargo_dates=0` is the selection as it was, to the bit, and the
  library's default;
- `embargo_dates=k` drops the window's last k dates from every number the
  selection computes, leaves the holdout where it was, records what it
  dropped, and on a panel with recorded label ends leaves no selection row
  whose label ends inside the holdout;
- the tool's default is the target horizon from `target_id`, with a note
  when the id names none;
- `correction='bh'` passes a feature on its Benjamini-Hochberg adjusted
  p-value over every representative tested, keeps a subset of what the
  uncorrected gate keeps, and says what it cost; an untestable feature
  enters the family at p = 1;
- the descriptions say why no family-wise correction is offered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as SchemaError

import standard_quant_tools.modeling.analysis.feature_selection as selection_module
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.feature_models import SelectFeaturesInput
from standard_quant_tools.modeling.agent.feature_tools import (
    FEATURE_TOOL_DEFS,
    _selection_embargo,
    select_features,
)
from standard_quant_tools.modeling.agent.models import RegisterExternalPanelInput
from standard_quant_tools.modeling.agent.tools import (
    _load_dataset_panel,
    register_external_panel,
)
from standard_quant_tools.modeling.analysis.feature_report import (
    feature_predictive_stats,
)
from standard_quant_tools.modeling.analysis.feature_selection import (
    select_features as select_features_on,
)
from standard_quant_tools.modeling.analysis.feature_stability import (
    permutation_test_ic,
)
from standard_quant_tools.modeling.validation.comparison import bh_adjust


def _panel(
    *, n_noise: int = 4, n_dates: int = 150, n_entities: int = 15, seed: int = 3
):
    """`signal` loads on the target; the `noise_*` columns do not."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    n = n_dates * n_entities
    target = rng.normal(size=n)
    data = {
        "date": np.repeat(dates.to_numpy(), n_entities),
        "entity": np.tile([f"E{j:02d}" for j in range(n_entities)], n_dates),
        "target": target,
        "signal": 0.3 * target + rng.normal(size=n),
    }
    for k in range(n_noise):
        data[f"noise_{k}"] = rng.normal(size=n)
    return pd.DataFrame(data), ["signal"] + [f"noise_{k}" for k in range(n_noise)]


def _register(frame: pd.DataFrame, tmp_path, horizon=5) -> str:
    path = tmp_path / "panel.parquet"
    frame.to_parquet(path, index=False)
    return register_external_panel(
        RegisterExternalPanelInput(path=str(path), horizon=horizon)
    ).dataset_id


def _planted_p_values(monkeypatch, planted):
    """Stand the permutation test in with planted p-values; a value of None
    makes the feature untestable, as a column the test refuses is."""

    def fake(panel, feature, **_kwargs):
        if planted[feature] is None:
            raise ValidationError(f"{feature} cannot be tested")
        return {"p_value": planted[feature]}

    monkeypatch.setattr(selection_module, "permutation_test_ic", fake)


# ── the embargo ──────────────────────────────────────────────────────────


class TestZeroIsTheOldWindow:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"significance": "none"},
            {"selection_end": "2022-05-31"},
            {"holdout_fraction": 0.0},
            {"max_features": 1, "significance": "none"},
        ],
        ids=["defaults", "no-test", "selection-end", "no-holdout", "cap"],
    )
    def test_it_is_the_default_and_changes_nothing(self, kwargs):
        panel, features = _panel()
        default = select_features_on(panel, features, **kwargs)
        zero = select_features_on(panel, features, embargo_dates=0, **kwargs)
        assert zero == default
        assert zero["embargo_dates"] == 0 and zero["embargo_window"] is None


class TestTheEmbargo:
    def test_it_drops_the_windows_last_dates_and_leaves_the_holdout(self):
        panel, features = _panel()
        before = select_features_on(panel, features)
        after = select_features_on(panel, features, embargo_dates=5)
        assert after["holdout_window"] == before["holdout_window"]
        assert (
            after["selection_window"]["n_dates"]
            == before["selection_window"]["n_dates"] - 5
        )
        dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
        read = dates[dates <= pd.Timestamp(before["selection_window"]["end"])]
        assert after["embargo_dates"] == 5
        assert after["embargo_window"] == {
            "start": str(read[-5].date()),
            "end": str(read[-1].date()),
            "n_dates": 5,
        }
        assert after["selection_window"]["end"] == str(read[-6].date())

    def test_every_number_reads_the_embargoed_window(self):
        """The selection IC and the gate's p-value are what the same
        functions give on the rows through the last date read, and nothing
        on the embargoed dates."""
        panel, features = _panel()
        result = select_features_on(panel, features, embargo_dates=5)
        last = pd.Timestamp(result["selection_window"]["end"])
        window = panel[pd.to_datetime(panel["date"]) <= last]
        stats = feature_predictive_stats(window, features)
        for feature in features:
            assert result["selection_ic"][feature] == stats[feature]["rank_ic_mean"]
        p_value = permutation_test_ic(
            window,
            "signal",
            n_permutations=200,
            method="spearman",
            random_seed=0,
            null="entity_shuffle",
        )["p_value"]
        assert result["selection_p_value"]["signal"] == p_value

    def test_a_named_selection_end_keeps_its_holdout(self):
        panel, features = _panel()
        result = select_features_on(
            panel, features, selection_end="2022-05-31", embargo_dates=3
        )
        assert result["holdout_window"]["start"] == "2022-06-01"
        assert result["embargo_window"]["end"] == "2022-05-31"
        assert result["selection_window"]["end"] == "2022-05-26"
        assert any(
            "were embargoed" in w and "through 2022-05-31" in w
            for w in result["warnings"]
        )

    def test_with_no_holdout_there_is_nothing_to_embargo(self):
        panel, features = _panel()
        result = select_features_on(
            panel, features, holdout_fraction=0.0, embargo_dates=5
        )
        assert result["embargo_dates"] == 0 and result["embargo_window"] is None
        assert result == select_features_on(panel, features, holdout_fraction=0.0)

    def test_an_embargo_that_leaves_nothing_is_refused_by_name(self):
        panel, features = _panel(n_dates=20)
        with pytest.raises(ValidationError, match="embargo_dates=14 would leave"):
            select_features_on(panel, features, embargo_dates=14)
        with pytest.raises(ValidationError, match="embargo_dates=-1"):
            select_features_on(panel, features, embargo_dates=-1)
        with pytest.raises(SchemaError):
            SelectFeaturesInput(dataset_id="ds_x", embargo_dates=-1)


class TestTheToolEmbargoesTheHorizon:
    def test_its_default_is_the_target_horizon(self, tmp_path):
        panel, features = _panel()
        dataset_id = _register(panel, tmp_path, horizon=5)
        result = select_features(SelectFeaturesInput(dataset_id=dataset_id))
        assert result.embargo_dates == 5
        explicit = select_features_on(panel, features, embargo_dates=5)
        assert result.selection_window == explicit["selection_window"]
        assert result.selection_ic == explicit["selection_ic"]
        assert result.selection_p_value == explicit["selection_p_value"]

    def test_no_selection_row_has_a_label_ending_inside_the_holdout(self, tmp_path):
        """The registration records each row's label end, 5 rows ahead on
        the entity's own dates. Without the embargo the window's last five
        dates reach into the holdout; with it none does."""
        panel, _features = _panel()
        dataset_id = _register(panel, tmp_path, horizon=5)
        loaded, _meta, _dir = _load_dataset_panel(dataset_id)
        dates = pd.to_datetime(loaded["date"])
        ends = pd.to_datetime(loaded["label_end_date"])
        for embargo, reaching in ((0, 5), (None, 0)):
            result = select_features(
                SelectFeaturesInput(dataset_id=dataset_id, embargo_dates=embargo)
            )
            start = pd.Timestamp(result.holdout_window["start"], tz=dates.dt.tz)
            last = pd.Timestamp(result.selection_window["end"], tz=dates.dt.tz)
            read = loaded[(dates <= last) & (ends >= start)]
            assert read["date"].nunique() == reaching, embargo

    def test_zero_through_the_tool_is_the_old_window(self, tmp_path):
        panel, features = _panel()
        dataset_id = _register(panel, tmp_path, horizon=5)
        result = select_features(
            SelectFeaturesInput(dataset_id=dataset_id, embargo_dates=0)
        )
        old = select_features_on(panel, features)
        assert result.embargo_dates == 0 and result.embargo_window is None
        assert result.selection_window == old["selection_window"]
        assert result.selection_ic == old["selection_ic"]
        assert result.warnings == old["warnings"]

    def test_a_label_with_no_horizon_is_said_to_get_none(self):
        assert _selection_embargo(None, {"target_id": "forward_return_rank:5"}) == (
            5,
            None,
        )
        assert _selection_embargo(3, {"target_id": "forward_return:20"}) == (3, None)
        embargo, note = _selection_embargo(None, {"target_id": "external:None"})
        assert embargo == 0
        assert "names no horizon" in note and "embargo_dates" in note


# ── the correction ───────────────────────────────────────────────────────


class TestBenjaminiHochberg:
    PLANTED = {
        "signal": 1 / 201,
        "noise_0": 0.03,
        "noise_1": 0.5,
        "noise_2": 0.6,
    }

    def test_it_passes_on_the_adjusted_p_value(self, monkeypatch):
        """Hand-worked over four: 1/201 -> 4 x 1/201 = 0.0199; 0.03 -> 2 x
        0.03 = 0.06; 0.5 -> min(4/3 x 0.5, 0.6) = 0.6; 0.6 -> 0.6. Uncorrected,
        0.03 passes at 0.05; corrected it does not."""
        panel, _ = _panel(n_noise=3)
        features = list(self.PLANTED)
        _planted_p_values(monkeypatch, self.PLANTED)
        plain = select_features_on(panel, features)
        corrected = select_features_on(panel, features, correction="bh")
        assert sorted(plain["selected"]) == ["noise_0", "signal"]
        assert corrected["selected"] == ["signal"]
        assert corrected["selection_p_value"] == plain["selection_p_value"]
        assert corrected["selection_p_value_adjusted"] == pytest.approx(
            {"signal": 4 / 201, "noise_0": 0.06, "noise_1": 0.6, "noise_2": 0.6}
        )
        assert corrected["significance"]["correction"] == "bh"
        assert corrected["significance"]["n_passed"] == 1
        assert corrected["significance"]["n_passed_uncorrected"] == 2
        drop = next(d for d in corrected["dropped"] if d["feature"] == "noise_0")
        assert drop["reason"] == "insignificant" and drop["p_value"] == 0.03
        assert "Benjamini-Hochberg adjusted 0.060 over 4 tested" in drop["detail"]
        assert any(
            "1 of 4 features cleared a Benjamini-Hochberg adjusted p < 0.05" in w
            and "Uncorrected, 2 cleared p < 0.05." in w
            for w in corrected["warnings"]
        )

    def test_the_kept_set_is_a_subset_of_the_uncorrected_one(self):
        panel, features = _panel(n_noise=6)
        plain = select_features_on(panel, features)
        corrected = select_features_on(panel, features, correction="bh")
        assert set(corrected["selected"]) <= set(plain["selected"])
        tested = list(corrected["selection_p_value"])
        assert [corrected["selection_p_value_adjusted"][f] for f in tested] == (
            bh_adjust([corrected["selection_p_value"][f] for f in tested])
        )

    def test_an_untestable_feature_counts_at_one(self, monkeypatch):
        panel, _ = _panel(n_noise=1)
        _planted_p_values(monkeypatch, {"signal": 0.02, "noise_0": None})
        result = select_features_on(panel, ["signal", "noise_0"], correction="bh")
        assert result["selection_p_value_adjusted"] == {
            "signal": pytest.approx(0.04),
            "noise_0": None,
        }
        assert result["selected"] == ["signal"]

    def test_nothing_passing_is_said_beside_what_would_have(self, monkeypatch):
        panel, _ = _panel(n_noise=3)
        planted = {"signal": 0.04, "noise_0": 0.5, "noise_1": 0.7, "noise_2": 0.9}
        _planted_p_values(monkeypatch, planted)
        result = select_features_on(panel, list(planted), correction="bh")
        assert result["selected"] == []
        assert any(
            "No feature cleared a Benjamini-Hochberg adjusted p < 0.05" in w
            and "Uncorrected, 1 cleared" in w
            for w in result["warnings"]
        )

    def test_the_default_corrects_nothing(self):
        panel, features = _panel()
        result = select_features_on(panel, features)
        assert result["selection_p_value_adjusted"] == {}
        assert result["significance"]["correction"] == "none"
        assert (
            result["significance"]["n_passed_uncorrected"]
            == result["significance"]["n_passed"]
        )

    def test_it_is_refused_where_there_is_nothing_to_correct(self):
        panel, features = _panel()
        with pytest.raises(ValidationError, match="significance='none' runs"):
            select_features_on(panel, features, significance="none", correction="bh")
        with pytest.raises(ValidationError, match="correction='holm'"):
            select_features_on(panel, features, correction="holm")
        with pytest.raises(SchemaError):
            SelectFeaturesInput(dataset_id="ds_x", correction="holm")

    def test_through_the_tool(self, tmp_path):
        panel, _features = _panel(n_noise=3)
        dataset_id = _register(panel, tmp_path)
        result = select_features(
            SelectFeaturesInput(dataset_id=dataset_id, correction="bh")
        )
        assert result.significance.correction == "bh"
        assert set(result.selection_p_value_adjusted) == set(result.selection_p_value)
        assert "signal" in result.selected


class TestTheDescriptions:
    def test_the_inputs_say_what_they_do(self):
        fields = SelectFeaturesInput.model_fields
        correction = fields["correction"].description
        assert "1/(n_permutations + 1)" in correction
        assert "more than ten features" in correction
        embargo = fields["embargo_dates"].description
        assert "target horizon" in embargo and "to the bit" in embargo

    def test_the_tool_names_both(self):
        text = {name: text for name, text, _ in FEATURE_TOOL_DEFS}["select_features"]
        assert "embargo_dates" in text
        assert "correction='bh'" in text
