"""
The feature-analysis library refuses what its tool doors refuse, by name.

The feature-lab tools check their inputs in their schemas: a feature named
twice, a `selection_end` or `split_date` that is not a date. The library
functions behind those tools did not, so a direct caller got pandas' own
errors instead -- "the truth value of a DataFrame is ambiguous" from inside
the redundancy clustering, an IndexError from a holdout window with no
dates in it, a DateParseError from the split. An empty `split_date` was
read as "use the median", which is what OMITTING it means, while the tool
refused it. Each is now a ValidationError naming the argument and the
remedy, and the ordinary call beside it still answers. See the CHANGELOG
entry of 2026-09-28.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.analysis.feature_report import (
    _correlation_clusters,
    build_feature_report,
    feature_distribution_stats,
    feature_predictive_stats,
    redundancy_report,
)
from standard_quant_tools.modeling.analysis.feature_selection import (
    _window,
    compare_feature_sets,
    select_features,
    summarize_feature_set,
)
from standard_quant_tools.modeling.analysis.feature_stability import (
    feature_drift,
    feature_stability,
    psi_by_block,
)


def _panel(n_dates: int = 60, n_entities: int = 10, seed: int = 0) -> pd.DataFrame:
    """`a` predicts, `c` restates `a`, `b` is noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for date in pd.bdate_range("2022-01-03", periods=n_dates):
        for j in range(n_entities):
            signal = rng.normal()
            rows.append(
                {
                    "date": date,
                    "entity": f"E{j}",
                    "a": signal,
                    "b": rng.normal(),
                    "c": signal + 0.01 * rng.normal(),
                    "target": 0.3 * signal + rng.normal(),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel()


def _with_unreadable_date(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.loc[3, "date"] = "garbage"
    return out


class TestAFeatureNamedTwice:
    @pytest.mark.parametrize(
        "call",
        [
            lambda p, f: redundancy_report(p, f),
            lambda p, f: build_feature_report(p, f, include_leakage=False),
            lambda p, f: select_features(p, f),
            lambda p, f: summarize_feature_set(p, f),
            lambda p, f: feature_distribution_stats(p, f),
            lambda p, f: feature_predictive_stats(p, f),
        ],
        ids=[
            "redundancy_report",
            "build_feature_report",
            "select_features",
            "summarize_feature_set",
            "feature_distribution_stats",
            "feature_predictive_stats",
        ],
    )
    def test_is_refused_by_name(self, panel, call):
        """The clustering used to raise pandas' ambiguous-truth-value error;
        the two per-feature tables collapsed the repeat without a word."""
        with pytest.raises(ValidationError, match=r"names \['a'\] more than once"):
            call(panel, ["a", "b", "a"])

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_on_either_side_of_a_comparison_is_refused_naming_the_side(
        self, panel, side
    ):
        sets = {"left": ["b"], "right": ["b"], side: ["a", "a"]}
        with pytest.raises(ValidationError, match=rf"{side} names \['a'\]"):
            compare_feature_sets(panel, sets["left"], sets["right"])

    def test_a_panel_with_two_columns_under_one_name_is_refused(self, panel):
        doubled = panel.copy()
        doubled.columns = ["date", "entity", "a", "a", "c", "target"]
        with pytest.raises(ValidationError, match=r"more than one column named"):
            redundancy_report(doubled, ["a", "c"])

    def test_the_clustering_guards_its_own_matrix(self):
        """Any other caller of the clustering meets the same rule."""
        labels = ["a", "b", "a"]
        matrix = pd.DataFrame(np.eye(3), index=labels, columns=labels)
        with pytest.raises(ValidationError, match=r"\['a'\] more than once"):
            _correlation_clusters(matrix, 0.9)

    def test_a_name_on_both_sides_of_a_comparison_is_the_comparison(self, panel):
        """The null case: shared across sides is legal, twice on one is not."""
        result = compare_feature_sets(panel, ["a", "b"], ["a", "c"])
        assert result["shared"] == ["a"]

    def test_names_listed_once_still_cluster(self, panel):
        """The null case: `c` restates `a` and they cluster together."""
        clusters = redundancy_report(panel, ["a", "b", "c"])["clusters"]
        assert sorted(clusters[0]) == ["a", "c"]


class TestASelectionEndThatIsNotADate:
    BAD = ["", "NaT", "not-a-date", "2022-13-45"]

    @pytest.mark.parametrize("bad", BAD)
    def test_select_features_refuses_it_by_name(self, panel, bad):
        """Empty and NaT used to pass the range check (NaT compares false)
        and leave a holdout with no dates, which failed as an IndexError;
        the rest raised pandas' parse error."""
        with pytest.raises(ValidationError, match=r"selection_end=.* is not a date"):
            select_features(panel, ["a", "b"], selection_end=bad)

    @pytest.mark.parametrize("bad", BAD)
    def test_summarize_feature_set_refuses_it_by_name(self, panel, bad):
        with pytest.raises(
            ValidationError, match=r"summarize_feature_set: selection_end"
        ):
            summarize_feature_set(panel, ["a"], selection_end=bad)

    @pytest.mark.parametrize("bad", BAD)
    def test_compare_feature_sets_refuses_it_by_name(self, panel, bad):
        with pytest.raises(
            ValidationError, match=r"compare_feature_sets: selection_end"
        ):
            compare_feature_sets(panel, ["a"], ["b"], selection_end=bad)

    def test_a_date_with_a_zone_the_panel_lacks_is_refused(self, panel):
        """It cannot be compared with the panel's dates at all; pandas
        raised a TypeError from the range check."""
        with pytest.raises(ValidationError, match="time zone"):
            select_features(
                panel, ["a", "b"], selection_end="2022-02-01T00:00:00+00:00"
            )

    def test_an_unreadable_panel_date_is_refused_naming_the_column(self, panel):
        with pytest.raises(ValidationError, match=r"'date' column .*'garbage'"):
            select_features(_with_unreadable_date(panel), ["a", "b"])

    def test_an_empty_window_is_refused_rather_than_indexed(self):
        with pytest.raises(ValidationError, match="no dates"):
            _window(pd.DatetimeIndex([]))

    def test_a_real_date_selects_through_it(self, panel):
        """The null case: the window ends where the caller said."""
        result = select_features(panel, ["a", "b", "c"], selection_end="2022-02-15")
        assert result["selection_window"]["end"] == "2022-02-15"
        assert result["holdout_window"]["n_dates"] > 0

    def test_a_timestamp_is_accepted_as_well_as_a_string(self, panel):
        result = summarize_feature_set(
            panel, ["a", "b"], selection_end=pd.Timestamp("2022-02-15")
        )
        assert result["selection_window"]["end"] == "2022-02-15"


class TestASplitDateThatIsNotADate:
    @pytest.mark.parametrize("bad", ["", "NaT", "not-a-date", "2022-13-45"])
    def test_feature_drift_refuses_it_by_name(self, panel, bad):
        """An empty string used to mean "the median", as omitting it does;
        an unreadable one raised pandas' DateParseError."""
        with pytest.raises(ValidationError, match=r"feature_drift: split_date="):
            feature_drift(panel, "a", split_date=bad)

    def test_a_date_with_a_zone_the_panel_lacks_is_refused(self, panel):
        with pytest.raises(ValidationError, match="time zone"):
            feature_drift(panel, "a", split_date="2022-02-01T00:00:00Z")

    def test_a_plain_date_against_a_zoned_panel_is_refused_with_the_zone(self, panel):
        zoned = panel.assign(date=panel["date"].dt.tz_localize("UTC"))
        with pytest.raises(ValidationError, match="panel's dates are in UTC"):
            feature_drift(zoned, "a", split_date="2022-02-01")

    @pytest.mark.parametrize(
        "call",
        [
            lambda p: feature_drift(p, "a"),
            lambda p: feature_stability(p, "a"),
            lambda p: psi_by_block(p, "a"),
        ],
        ids=["feature_drift", "feature_stability", "psi_by_block"],
    )
    def test_an_unreadable_panel_date_is_refused_naming_the_column(self, panel, call):
        with pytest.raises(ValidationError, match=r"'date' column .*'garbage'"):
            call(_with_unreadable_date(panel))

    def test_no_split_date_is_the_median(self, panel):
        """The null case: omitting it still splits by time at the median."""
        result = feature_drift(panel, "a")
        assert result["n_before"] > 0 and result["n_after"] > 0

    def test_a_real_split_date_splits_there(self, panel):
        result = feature_drift(panel, "a", split_date="2022-02-01")
        assert result["split_date"] == "2022-02-01"

    def test_a_zoned_split_against_a_zoned_panel_splits_there(self, panel):
        zoned = panel.assign(date=panel["date"].dt.tz_localize("UTC"))
        result = feature_drift(zoned, "a", split_date="2022-02-01T00:00:00Z")
        assert result["split_date"] == "2022-02-01"
