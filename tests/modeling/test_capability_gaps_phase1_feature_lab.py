"""
The capability-gaps fixes of 2026-09-21 (CHANGELOG): the feature
lab's two lossy results.

  1C.1  `select_features` discards what it paid for. It calls
        `redundancy_report` to make the decision and then returned only
        `n_clusters` from it, so an agent that wanted "which features were
        dropped as duplicates of what", or the collinearity of what
        survived, had to call `get_feature_redundancy` and buy the same
        correlation matrix a second time. The tests here pin that the two
        tools now agree field for field, so the second call is
        unnecessary rather than merely redundant.

  1C.2  D-8: `summarize_feature_set` summarised the WHOLE panel, so
        `CompareFeatureSetsResult.left/right.mean_abs_rank_ic` was
        in-sample by construction -- and it was the one feature-lab result
        with no `warnings` field to say so, while its sibling
        `select_features` went to considerable trouble over exactly that.
        The tests here plant the failure the silence hides: five noise
        columns cherry-picked out of sixty sit beside a real feature on
        the window that chose them, and collapse on the dates that did not.

Every detector below has a null case beside it: `include_correlation`
off as well as on, a threshold that clusters nothing as well as one that
clusters everything, and a holdout as well as no holdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent import (
    BuildModelDatasetInput,
    build_model_dataset,
)
from standard_quant_tools.modeling.agent.feature_models import (
    CompareFeatureSetsInput,
    FeatureRedundancyInput,
    SelectFeaturesInput,
)
from standard_quant_tools.modeling.agent.feature_tools import (
    FEATURE_TOOL_DISPATCH,
    compare_feature_sets,
    get_feature_redundancy,
    select_features,
)
from standard_quant_tools.modeling.analysis.feature_selection import (
    compare_feature_sets as compare_feature_sets_on,
)
from standard_quant_tools.modeling.analysis.feature_selection import (
    select_features as select_features_on,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    FeatureSpec,
    TargetSpec,
)


def _spec() -> DatasetSpec:
    """The same three-feature, three-name panel `test_feature_tools.py`
    builds: a momentum-ish feature and two risk features, which is enough
    for one cluster to form and for a condition number to be finite."""
    return DatasetSpec(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[
            FeatureSpec(id="technical.rsi"),
            FeatureSpec(id="risk.rolling_beta"),
            FeatureSpec(id="risk.realized_volatility"),
        ],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )


@pytest.fixture
def dataset(patched_multi_factory):
    return build_model_dataset(BuildModelDatasetInput(spec=_spec())).dataset_id


@pytest.fixture
def features(dataset):
    from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

    _panel, meta, _dir = _load_dataset_panel(dataset)
    return list(meta["feature_ids"])


# ── planted panels ───────────────────────────────────────────────────────


def _duplicate_panel(n_dates: int = 120, n_entities: int = 12):
    """
    A panel where the answer is known before the tool runs: `alpha_copy` IS
    `alpha`, to the last decimal, and `beta` is independent of both.

    So the only defensible cluster is {alpha, alpha_copy}, the only
    defensible drop is one of the two, and `duplicate_of` has exactly one
    right value.
    """
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    n = n_dates * n_entities
    alpha = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), n_entities),
            "entity": np.tile([f"E{j}" for j in range(n_entities)], n_dates),
            "target": 0.2 * alpha + rng.normal(size=n),
            "alpha": alpha,
            "alpha_copy": alpha,
            "beta": rng.normal(size=n),
        }
    )
    return frame, ["alpha", "alpha_copy", "beta"]


def _noise_and_signal_panel(
    n_noise: int = 60,
    n_dates: int = 300,
    n_entities: int = 20,
    strength: float = 0.05,
    seed: int = 11,
):
    """
    Sixty columns of pure noise and one real feature, `signal`, built as
    `strength * target + noise` so its cross-sectional IC is real, modest
    and the same on every date.

    This is the D4 shape reproduced offline: the noise is independent of
    the target everywhere, so any IC a noise column shows on the window it
    was picked from is the picking, not the column.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    n = n_dates * n_entities
    target = rng.normal(size=n)
    data = {
        "date": np.repeat(dates.to_numpy(), n_entities),
        "entity": np.tile([f"E{j}" for j in range(n_entities)], n_dates),
        "target": target,
        "signal": strength * target + rng.normal(size=n),
    }
    for k in range(n_noise):
        data[f"noise_{k}"] = rng.normal(size=n)
    return pd.DataFrame(data), [f"noise_{k}" for k in range(n_noise)]


# ── 1C.1 ─────────────────────────────────────────────────────────────────


class TestSelectFeaturesReturnsWhatItPaidFor:
    """
    1C.1 -- `select_features` discards diagnostics it already paid for
    (gaps document section 4, "select_features discards diagnostics").

    The point of these is not that the numbers are right -- they come from
    `redundancy_report`, which `test_feature_report.py` covers -- but that
    they ARRIVE, and that they are the same numbers the redundancy tool
    reports. Two tools that resolved the same panel differently would give
    an agent two drop lists and no way to choose.
    """

    def test_the_clusters_are_the_redundancy_tools_clusters(self, dataset):
        """On the same window and threshold, field for field. A threshold of
        0.0 puts every feature in one cluster, which is the case where the
        representative choice actually has to be made."""
        selected = select_features(
            SelectFeaturesInput(
                dataset_id=dataset, cluster_threshold=0.0, holdout_fraction=0.0
            )
        )
        redundancy = get_feature_redundancy(
            FeatureRedundancyInput(dataset_id=dataset, cluster_threshold=0.0)
        )
        assert selected.clusters == redundancy.clusters
        assert selected.n_clusters == len(redundancy.clusters)
        # And the pin the plan says must keep holding: the selection is the
        # representatives, so the two tools cannot contradict each other.
        assert set(selected.selected) == {c.representative for c in redundancy.clusters}

    def test_the_collinearity_numbers_are_the_redundancy_tools_numbers(
        self, dataset, features
    ):
        """`vif` and `condition_number` were computed to make the decision
        and then dropped. Running the redundancy tool to recover them buys
        the same correlation matrix twice."""
        selected = select_features(
            SelectFeaturesInput(
                dataset_id=dataset, cluster_threshold=0.0, holdout_fraction=0.0
            )
        )
        redundancy = get_feature_redundancy(
            FeatureRedundancyInput(dataset_id=dataset, cluster_threshold=0.0)
        )
        assert set(selected.vif) == set(features)
        for feature in features:
            assert selected.vif[feature] == pytest.approx(redundancy.vif[feature])
        assert selected.condition_number == pytest.approx(redundancy.condition_number)

    def test_a_singleton_threshold_still_reports_every_feature(self, dataset, features):
        """The null case for the clustering detector: at |rho| >= 1.0
        nothing but an exact restatement groups, so every feature is its own
        cluster -- a result, not an omission."""
        selected = select_features(
            SelectFeaturesInput(
                dataset_id=dataset, cluster_threshold=1.0, holdout_fraction=0.0
            )
        )
        assert len(selected.clusters) == len(features)
        assert sorted(c.representative for c in selected.clusters) == sorted(features)
        assert all(c.size == 1 for c in selected.clusters)
        assert all(d.duplicate_of is None for d in selected.dropped)

    def test_correlation_is_absent_unless_it_is_asked_for(self, dataset, features):
        """It is O(n^2) in the payload, so it is off by default. On, it is
        a square matrix over exactly the candidates."""
        without = select_features(
            SelectFeaturesInput(dataset_id=dataset, holdout_fraction=0.0)
        )
        assert without.correlation == {}

        with_it = select_features(
            SelectFeaturesInput(
                dataset_id=dataset, holdout_fraction=0.0, include_correlation=True
            )
        )
        assert set(with_it.correlation) == set(features)
        for row in with_it.correlation.values():
            assert set(row) == set(features)
        for feature in features:
            assert with_it.correlation[feature][feature] == pytest.approx(1.0)
        # Same matrix the redundancy tool publishes, not a second one.
        redundancy = get_feature_redundancy(FeatureRedundancyInput(dataset_id=dataset))
        for a in features:
            for b in features:
                assert with_it.correlation[a][b] == pytest.approx(
                    redundancy.correlation[a][b]
                )

    def test_every_duplicate_names_the_keeper_it_duplicates(self, dataset):
        """'Dropped as a duplicate of what' is a field now, not a sentence
        to parse. At a 0.0 threshold every feature is in one cluster, so
        every redundant drop points at that cluster's representative."""
        selected = select_features(
            SelectFeaturesInput(
                dataset_id=dataset, cluster_threshold=0.0, holdout_fraction=0.0
            )
        )
        by_member = {
            member: cluster.representative
            for cluster in selected.clusters
            for member in cluster.members
        }
        redundant = [d for d in selected.dropped if d.reason == "redundant"]
        assert redundant, "a 0.0 threshold must produce redundant drops"
        for dropped in redundant:
            assert dropped.duplicate_of == by_member[dropped.feature]
            assert dropped.duplicate_of in selected.selected
            # The prose stays: it carries the threshold the drop was made at.
            assert repr(dropped.duplicate_of) in dropped.detail

    def test_a_planted_duplicate_is_dropped_for_its_twin(self):
        """PLANTED: `alpha_copy` is `alpha`. The only right answer is that
        one of the two is dropped and names the other, and that `beta`,
        which is independent of both, is untouched."""
        panel, feature_ids = _duplicate_panel()
        result = select_features_on(panel, feature_ids, holdout_fraction=0.0)

        cluster = next(c for c in result["clusters"] if c["size"] > 1)
        assert cluster["members"] == ["alpha", "alpha_copy"]
        assert cluster["max_abs_correlation"] == pytest.approx(1.0)
        assert ["beta"] in [c["members"] for c in result["clusters"]]

        redundant = [d for d in result["dropped"] if d["reason"] == "redundant"]
        assert len(redundant) == 1
        assert redundant[0]["feature"] in {"alpha", "alpha_copy"}
        assert redundant[0]["duplicate_of"] == cluster["representative"]
        assert redundant[0]["duplicate_of"] != redundant[0]["feature"]
        assert "beta" in result["selected"]

    def test_a_weak_drop_is_not_a_duplicate_of_anything(self):
        """The null case for `duplicate_of`: a feature dropped for failing
        the IC floor was not dropped FOR another feature, and saying so
        would be a different claim."""
        panel, feature_ids = _duplicate_panel()
        result = select_features_on(
            panel, feature_ids, min_abs_rank_ic=1.0, holdout_fraction=0.0
        )
        assert result["selected"] == []
        weak = [d for d in result["dropped"] if d["reason"] == "weak"]
        assert weak
        assert all(d["duplicate_of"] is None for d in weak)


# ── 1C.2 ─────────────────────────────────────────────────────────────────


class TestCompareFeatureSetsSaysWhenItIsInSample:
    """
    1C.2 / D-8 -- `summarize_feature_set` was whole-panel, in-sample and
    silent, and `CompareFeatureSetsResult` had no `warnings` field to say
    so while its sibling `select_features` did.
    """

    def test_the_in_sample_warning_is_unconditional_at_a_zero_fraction(
        self, dataset, features
    ):
        """Not "when it looks suspicious" -- in-sample is a property of how
        the numbers were made, not of how they came out. Zero fraction is
        also the default, so the numbers a caller has today do not move."""
        result = compare_feature_sets(
            CompareFeatureSetsInput(
                dataset_id=dataset, left=features[:1], right=features
            )
        )
        assert result.warnings
        joined = " ".join(result.warnings)
        assert "in-sample by construction" in joined
        assert "+0.045" in joined and "+0.002" in joined
        assert result.left.holdout_window is None
        assert result.left.holdout_mean_abs_rank_ic is None
        assert result.right.holdout_max_abs_rank_ic is None

    def test_identical_sets_give_a_zero_delta_and_still_warn(self, dataset, features):
        """A comparison that found nothing is still a comparison made on
        every date. The warning is about the window, not the verdict."""
        result = compare_feature_sets(
            CompareFeatureSetsInput(dataset_id=dataset, left=features, right=features)
        )
        assert result.delta.n_features == 0
        assert result.delta.n_independent_signals == 0
        assert result.delta.mean_abs_rank_ic == pytest.approx(0.0)
        assert result.delta.condition_number == pytest.approx(0.0)
        assert any("in-sample by construction" in w for w in result.warnings)

    def test_a_holdout_replaces_the_warning_rather_than_keeping_it(
        self, dataset, features
    ):
        """The null case for the in-sample detector: once dates are held
        out the sentence is wrong, so it must not fire."""
        result = compare_feature_sets(
            CompareFeatureSetsInput(
                dataset_id=dataset,
                left=features[:1],
                right=features,
                holdout_fraction=0.3,
            )
        )
        joined = " ".join(result.warnings)
        assert "in-sample by construction" not in joined
        assert "holdout_mean_abs_rank_ic" in joined
        for side in (result.left, result.right):
            assert side.holdout_window is not None
            assert side.selection_window["n_dates"] > side.holdout_window["n_dates"]
            assert side.selection_window["end"] < side.holdout_window["start"]
            assert side.holdout_mean_abs_rank_ic is not None
            assert side.holdout_max_abs_rank_ic is not None

    def test_selection_end_names_the_cutoff_for_both_sides(self, dataset, features):
        result = compare_feature_sets(
            CompareFeatureSetsInput(
                dataset_id=dataset,
                left=features[:1],
                right=features,
                selection_end="2023-06-30",
            )
        )
        assert result.left.selection_window["end"] == "2023-06-30"
        assert result.right.selection_window["end"] == "2023-06-30"
        assert result.left.holdout_window == result.right.holdout_window

    def test_a_one_date_panel_is_refused_by_name(self):
        """`_selection_cutoff`'s refusal, reached through the compare path:
        nothing can be held out of one date, and the message says which
        argument to change rather than which function computed it."""
        panel, noise = _noise_and_signal_panel(n_noise=3, n_dates=2)
        one_date = panel[panel["date"] == panel["date"].min()]
        with pytest.raises(ValidationError, match="fewer than two dates"):
            compare_feature_sets_on(
                one_date, noise[:1], ["signal"], holdout_fraction=0.3
            )
        with pytest.raises(ValidationError, match="compare_feature_sets"):
            compare_feature_sets_on(
                one_date, noise[:1], ["signal"], holdout_fraction=0.3
            )


class TestCherryPickedNoiseSurvivesTheInSampleComparison:
    """
    1C.2 / D-8, planted: sixty noise columns and one real feature. The five
    noise columns picked on the first 70% of the dates are independent of
    the target by construction, so everything they show on that window is
    the picking. Measured on this panel (seed 11, strength 0.05):

        selection window   noise 0.0336   signal 0.0474
        holdout            noise 0.0104   signal 0.0600

    A comparison run on every date reports 0.0205 against 0.0512 and says
    nothing about which of the two it just flattered.
    """

    @staticmethod
    def _sets():
        panel, noise = _noise_and_signal_panel()
        # Chosen on the SELECTION window only, which is the fair version of
        # the cherry-pick: the holdout is untouched, so its collapse is the
        # noise being noise rather than an artifact of having peeked.
        chosen = select_features_on(panel, noise, max_features=5, holdout_fraction=0.3)[
            "selected"
        ]
        assert len(chosen) == 5
        return panel, chosen

    def test_in_sample_the_noise_set_is_not_told_apart_and_the_result_warns(self):
        panel, chosen = self._sets()
        flat = compare_feature_sets_on(panel, chosen, ["signal"], holdout_fraction=0.0)
        noise_ic = flat["left"]["mean_abs_rank_ic"]
        signal_ic = flat["right"]["mean_abs_rank_ic"]
        # Measured 0.0205 against 0.0512: the same order of magnitude, from
        # columns that have no relationship with the target at all.
        assert noise_ic > signal_ic / 3
        assert any("in-sample by construction" in w for w in flat["warnings"])
        assert flat["left"]["holdout_mean_abs_rank_ic"] is None

    def test_the_holdout_collapses_the_noise_and_leaves_the_signal_standing(self):
        panel, chosen = self._sets()
        held = compare_feature_sets_on(panel, chosen, ["signal"], holdout_fraction=0.3)
        noise, signal = held["left"], held["right"]

        # The noise loses most of what the selection window credited it with.
        assert noise["holdout_mean_abs_rank_ic"] < noise["mean_abs_rank_ic"] / 2
        # The real feature does not.
        assert signal["holdout_mean_abs_rank_ic"] > signal["mean_abs_rank_ic"] / 2
        assert (
            signal["holdout_mean_abs_rank_ic"] > 2 * noise["holdout_mean_abs_rank_ic"]
        )

    def test_the_in_sample_view_flatters_the_noise_relative_to_the_holdout(self):
        """The defect in one number. The noise set looks like 40% of the
        real feature when every date is read and 17% when the holdout is
        the one being read -- so the silent comparison overstated a pure-
        noise set by more than a factor of two."""
        panel, chosen = self._sets()
        flat = compare_feature_sets_on(panel, chosen, ["signal"], holdout_fraction=0.0)
        held = compare_feature_sets_on(panel, chosen, ["signal"], holdout_fraction=0.3)

        in_sample_ratio = (
            flat["left"]["mean_abs_rank_ic"] / flat["right"]["mean_abs_rank_ic"]
        )
        honest_ratio = (
            held["left"]["holdout_mean_abs_rank_ic"]
            / held["right"]["holdout_mean_abs_rank_ic"]
        )
        assert in_sample_ratio > 2 * honest_ratio

    def test_the_per_feature_table_reads_the_same_window_as_the_summaries(self):
        """A table measured on every date beside a summary that held dates
        out would be two answers to one question, and the wider one is the
        optimistic one."""
        panel, chosen = self._sets()
        held = compare_feature_sets_on(panel, chosen, ["signal"], holdout_fraction=0.3)
        table = {row["feature"]: row["abs_rank_ic"] for row in held["features"]}
        assert table["signal"] == pytest.approx(held["right"]["mean_abs_rank_ic"])
        assert np.mean([table[f] for f in chosen]) == pytest.approx(
            held["left"]["mean_abs_rank_ic"]
        )


class TestEveryFeatureLabResultCanWarn:
    """
    The convention from the plan's ground rules, checked rather than
    asserted in prose: a typed result with no `warnings` field cannot tell
    an agent anything its numbers do not already say, and an agent should
    not have to know which of nine tools can speak and which cannot.
    """

    def test_every_dispatched_result_model_has_a_warnings_field(self):
        import typing

        missing = []
        for name, (handler, _input_model) in sorted(FEATURE_TOOL_DISPATCH.items()):
            result_model = typing.get_type_hints(handler)["return"]
            if "warnings" not in result_model.model_fields:
                missing.append(f"{name} -> {result_model.__name__}")
        assert not missing, f"feature-lab results that cannot warn: {missing}"

    def test_the_warnings_field_is_a_list_of_strings_everywhere(self):
        import typing

        for _name, (handler, _input_model) in FEATURE_TOOL_DISPATCH.items():
            result_model = typing.get_type_hints(handler)["return"]
            field = result_model.model_fields["warnings"]
            assert field.annotation == typing.List[str]
            assert field.default_factory is list
