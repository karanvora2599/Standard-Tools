"""
The typed feature tools: one question per tool, and an answer with a shape.

WHY THESE EXIST. `analyze_features` returns `report: Dict[str, Any]`. Every
number in it is correct and none of it is promised by a schema, so an agent
asking "is momentum_20d worth keeping" had to profile the whole panel and
then guess key names. That is the failure `extra="forbid"` fixes on the way
IN, left unfixed on the way OUT.

WHAT THESE TESTS ARE FOR. Not the arithmetic -- `test_feature_report.py`
covers that, and none of it changed. These pin the *contract*:

- the typed result carries the same numbers the untyped one did, so the
  split is a re-presentation and not a second implementation that can drift;
- the cluster representative is chosen on merit and is STABLE, because a
  drop-list that reshuffles between identical calls is not a drop list;
- the IC-decay curve is ordered numerically, which the underlying
  string-keyed dict does not guarantee.
"""

import json
import math

import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent import (
    BuildModelDatasetInput,
    build_model_dataset,
)
from standard_quant_tools.modeling.agent.feature_models import (
    AnalyzeFeatureInput,
    CompareFeatureSetsInput,
    FeatureDriftInput,
    FeatureICDecayInput,
    FeatureRedundancyInput,
    FeatureStabilityInput,
    PermutationTestInput,
    SelectFeaturesInput,
)
from standard_quant_tools.modeling.agent.feature_tools import (
    _pick_representative,
    compare_feature_sets,
    feature_dispatch,
    get_feature_drift,
    get_feature_ic_decay,
    get_feature_redundancy,
    get_feature_regime_stability,
    profile_feature,
    run_feature_permutation_test,
    select_features,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    FeatureSpec,
    TargetSpec,
)


def _spec() -> DatasetSpec:
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


class TestAnalyzeFeature:
    def test_it_profiles_one_feature(self, dataset, features):
        result = profile_feature(
            AnalyzeFeatureInput(dataset_id=dataset, feature=features[0])
        )
        assert result.feature == features[0]
        # Typed all the way down: these are attributes, not dict lookups.
        assert 0.0 <= result.distribution.coverage <= 1.0
        assert result.predictive.n_quantiles == 10
        assert isinstance(result.predictive.rank_ic_mean, float)

    def test_it_matches_the_untyped_report(self, dataset, features):
        """The split must be a re-presentation, not a reimplementation. If
        these ever disagree, one of the two paths has grown its own
        arithmetic."""
        from standard_quant_tools.modeling.agent.tools import _load_dataset_panel
        from standard_quant_tools.modeling.analysis.feature_report import (
            build_feature_report,
        )

        panel, _meta, _dir = _load_dataset_panel(dataset)
        report = build_feature_report(panel, features, include_leakage=False)

        def same(typed_value, raw_value):
            """The typed path maps non-finite to None; the library keeps
            NaN. Those are the same statement, so compare them as one."""
            if typed_value is None:
                return raw_value is None or not math.isfinite(raw_value)
            return typed_value == pytest.approx(raw_value)

        for feature in features:
            typed = profile_feature(
                AnalyzeFeatureInput(dataset_id=dataset, feature=feature)
            )
            raw = report["features"][feature]
            assert same(typed.distribution.coverage, raw["coverage"])
            assert same(typed.distribution.turnover, raw["turnover"])
            assert same(typed.predictive.rank_ic_mean, raw["rank_ic_mean"])
            assert same(typed.predictive.monotonicity, raw["monotonicity"])

    def test_an_unknown_feature_is_refused_with_a_suggestion(self, dataset, features):
        """An agent that mistypes a feature name should get the name back,
        not a KeyError from three frames down."""
        typo = features[0][:-1]
        with pytest.raises(ValidationError) as exc:
            profile_feature(AnalyzeFeatureInput(dataset_id=dataset, feature=typo))
        message = str(exc.value)
        assert typo in message
        assert features[0] in message, "the near-match was not offered"

    def test_an_unknown_argument_is_rejected(self, dataset):
        with pytest.raises(Exception):
            AnalyzeFeatureInput(dataset_id=dataset, feature="x", n_quantile=5)  # typo


class TestFeatureRedundancy:
    def test_every_feature_lands_in_exactly_one_cluster(self, dataset, features):
        result = get_feature_redundancy(FeatureRedundancyInput(dataset_id=dataset))
        seen = [m for c in result.clusters for m in c.members]
        assert sorted(seen) == sorted(
            features
        ), "a feature was dropped or double-counted by the clustering"

    def test_the_representative_is_a_member(self, dataset):
        result = get_feature_redundancy(FeatureRedundancyInput(dataset_id=dataset))
        for cluster in result.clusters:
            assert cluster.representative in cluster.members

    def test_the_drop_list_is_everything_but_the_representatives(self, dataset):
        result = get_feature_redundancy(FeatureRedundancyInput(dataset_id=dataset))
        expected = sorted(
            m for c in result.clusters for m in c.members if m != c.representative
        )
        assert result.redundant_features == expected

    def test_a_perfect_duplicate_is_clustered_and_one_is_dropped(self, dataset):
        """The case the tool exists for. Threshold at 0.0 forces every
        feature into one cluster, so exactly one survives."""
        result = get_feature_redundancy(
            FeatureRedundancyInput(dataset_id=dataset, cluster_threshold=0.0)
        )
        assert len(result.clusters) == 1
        assert len(result.redundant_features) == result.n_features - 1
        assert result.warnings, "a total collapse should say so"

    def test_it_is_stable_across_identical_calls(self, dataset):
        """A representative that moved between runs would make every
        downstream drop-list unreproducible."""
        first = get_feature_redundancy(
            FeatureRedundancyInput(dataset_id=dataset, cluster_threshold=0.0)
        )
        second = get_feature_redundancy(
            FeatureRedundancyInput(dataset_id=dataset, cluster_threshold=0.0)
        )
        assert first.redundant_features == second.redundant_features
        assert [c.representative for c in first.clusters] == [
            c.representative for c in second.clusters
        ]

    def test_the_representative_is_the_strongest_signal(self):
        """Picked on merit, not on alphabetical luck."""
        predictive = {
            "aaa_weak": {"rank_ic_mean": 0.01},
            "zzz_strong": {"rank_ic_mean": -0.20},
            "mmm_mid": {"rank_ic_mean": 0.05},
        }
        assert (
            _pick_representative(list(predictive), predictive) == "zzz_strong"
        ), "sign should not matter; magnitude should"

    def test_ties_break_alphabetically(self):
        predictive = {"b": {"rank_ic_mean": 0.1}, "a": {"rank_ic_mean": 0.1}}
        assert _pick_representative(["b", "a"], predictive) == "a"


class TestFeatureICDecay:
    def test_the_curve_is_ordered_numerically(self, dataset, features):
        """The underlying dict is keyed by stringified shifts, where a text
        sort puts '-10' before '-2'."""
        result = get_feature_ic_decay(
            FeatureICDecayInput(dataset_id=dataset, feature=features[0], max_shift=10)
        )
        shifts = [p.shift for p in result.curve]
        assert shifts == sorted(shifts)
        assert shifts == list(range(-10, 11))

    def test_the_named_peak_is_the_actual_peak(self, dataset, features):
        result = get_feature_ic_decay(
            FeatureICDecayInput(dataset_id=dataset, feature=features[0])
        )
        strongest = max(result.curve, key=lambda p: abs(p.ic))
        assert result.peak_shift == strongest.shift

    def test_ic_at_zero_is_the_zero_point_of_the_curve(self, dataset, features):
        result = get_feature_ic_decay(
            FeatureICDecayInput(dataset_id=dataset, feature=features[0])
        )
        zero = next(p for p in result.curve if p.shift == 0)
        assert result.ic_at_zero == pytest.approx(zero.ic)

    def test_it_always_explains_itself(self, dataset, features):
        """`flagged` without a reason is an accusation an agent cannot pass
        to a human."""
        result = get_feature_ic_decay(
            FeatureICDecayInput(dataset_id=dataset, feature=features[0])
        )
        assert result.reason.strip()


class TestTheyAreRealTools:
    @pytest.mark.parametrize(
        "name",
        ["profile_feature", "get_feature_redundancy", "get_feature_ic_decay"],
    )
    def test_dispatchable_by_name(self, name, dataset, features):
        args = {"dataset_id": dataset}
        if name != "get_feature_redundancy":
            args["feature"] = features[0]
        result = feature_dispatch(name, args)
        assert isinstance(result, dict)

    def test_they_belong_to_the_feature_lab_runtime(self):
        """They were built inside `modeling` and moved out once the cluster
        reached the nine-tool floor a runtime needs -- which is the order
        the splitting rule requires: grow it at home, then move it."""
        from standard_quant_tools.agent.runtimes import owner_of

        for name in (
            "profile_feature",
            "get_feature_redundancy",
            "get_feature_ic_decay",
        ):
            assert owner_of(name) == "feature_lab"

    def test_the_move_is_recorded(self):
        """A split is a breaking change. An agent scoped to the donor gets
        a refusal, and a bare one reads as though it had hallucinated the
        name."""
        from standard_quant_tools.agent.runtimes import MOVED_FROM, resolve

        assert MOVED_FROM["profile_feature"] == "modeling"
        try:
            resolve("modeling").dispatch("profile_feature", {})
        except Exception as exc:
            assert "used to be in 'modeling'" in str(exc)
        else:  # pragma: no cover - the dispatch must refuse
            raise AssertionError("modeling still dispatches a moved tool")

    def test_the_history_is_not_told_to_callers_who_never_had_it(self):
        """A research-scoped agent never held these tools, so telling it
        they moved explains a history it was not part of."""
        from standard_quant_tools.agent.runtimes import resolve

        try:
            resolve("research").dispatch("profile_feature", {})
        except Exception as exc:
            assert "used to be" not in str(exc)

    def test_they_are_advertised(self):
        from standard_quant_tools.modeling.agent.feature_tools import (
            get_feature_tools,
        )

        advertised = {d["function"]["name"] for d in get_feature_tools()}
        assert {
            "profile_feature",
            "get_feature_redundancy",
            "get_feature_ic_decay",
        } <= advertised


class TestNonFiniteStatisticsAreJsonSafe:
    """
    A statistic that could not be computed must serialize as `null`.

    This was a live bug rather than a hypothetical. `_safe()` in
    feature_report.py is documented as producing "a float that survives
    JSON" and maps non-finite values to NaN -- but `json.dumps` writes NaN
    as a bare `NaN` token, which is invalid per RFC 8259 and rejected by
    strict parsers. JSON-RPC clients are strict parsers, so
    `analyze_features` could hand an MCP client a body it must reject.

    Reproduced with one entity per date: there is no cross-section, so every
    cross-sectional statistic is undefined and twelve NaNs reach the result.
    """

    @staticmethod
    def _degenerate_panel():
        import pandas as pd

        return pd.DataFrame(
            [
                {"date": d, "entity": "E0", "f": float(i), "target": float(i)}
                for i, d in enumerate(pd.bdate_range("2023-01-02", periods=40))
            ]
        )

    @staticmethod
    def _strict_loads(text: str):
        """json.loads accepts bare NaN by default. A strict parser does
        not, and neither does the protocol, so refuse it here too."""
        return json.loads(
            text,
            parse_constant=lambda c: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant {c!r}")
            ),
        )

    def test_the_library_still_produces_nan(self):
        """NaN is right in memory -- this pins WHERE the conversion
        happens, so nobody 'fixes' it in the numpy pipeline instead."""
        from standard_quant_tools.modeling.analysis.feature_report import (
            build_feature_report,
        )

        report = build_feature_report(
            self._degenerate_panel(), ["f"], include_leakage=False
        )
        assert math.isnan(report["features"]["f"]["ic_mean"])

    def test_the_untyped_tool_result_is_strict_json(self):
        from standard_quant_tools.modeling.agent.tools import _json_safe
        from standard_quant_tools.modeling.analysis.feature_report import (
            build_feature_report,
        )

        report = _json_safe(
            build_feature_report(self._degenerate_panel(), ["f"], include_leakage=False)
        )
        self._strict_loads(json.dumps(report))
        assert report["features"]["f"]["ic_mean"] is None

    def test_the_typed_result_is_strict_json(self):
        from standard_quant_tools.modeling.agent.feature_models import (
            FeatureDistribution,
        )

        stats = FeatureDistribution(
            coverage=1.0,
            n_missing=0,
            mean=float("nan"),
            std=float("inf"),
            skew=float("-inf"),
            kurtosis=0.0,
            outlier_rate=0.0,
            autocorrelation=float("nan"),
            turnover=0.5,
        )
        assert stats.mean is None
        assert stats.std is None
        assert stats.skew is None
        self._strict_loads(json.dumps(stats.model_dump()))

    def test_null_is_not_silently_zero(self):
        """The reason `null` rather than `0.0`. An IC of 0.0 says "no
        signal"; an IC that was never calculable says "no measurement", and
        a caller that cannot tell them apart will size a position on the
        difference."""
        from standard_quant_tools.modeling.agent.feature_models import (
            FeaturePredictive,
        )

        stats = FeaturePredictive(
            ic_mean=float("nan"),
            ic_std=0.0,
            ic_icir=0.0,
            ic_hit_rate=0.0,
            ic_n_dates=0,
            rank_ic_mean=0.0,
            rank_ic_std=0.0,
            rank_ic_icir=0.0,
            rank_ic_hit_rate=0.0,
            rank_ic_n_dates=0,
            n_quantiles=10,
            quantile_spread=0.0,
            monotonicity=0.0,
        )
        assert stats.ic_mean is None
        assert stats.rank_ic_mean == 0.0
        assert stats.ic_mean is not stats.rank_ic_mean


class TestSelectFeatures:
    def test_selected_and_dropped_partition_the_candidates(self, dataset, features):
        result = select_features(SelectFeaturesInput(dataset_id=dataset))
        accounted = set(result.selected) | {d.feature for d in result.dropped}
        assert accounted == set(
            features
        ), "a candidate was neither selected nor explained"
        assert result.n_selected == len(result.selected)

    def test_every_drop_carries_a_reason(self, dataset):
        result = select_features(
            SelectFeaturesInput(dataset_id=dataset, min_abs_rank_ic=0.9)
        )
        assert result.dropped
        for dropped in result.dropped:
            assert dropped.reason in {"redundant", "weak", "capped"}
            assert dropped.detail.strip()

    def test_an_impossible_floor_drops_everything(self, dataset):
        result = select_features(
            SelectFeaturesInput(dataset_id=dataset, min_abs_rank_ic=1.0)
        )
        assert result.selected == []
        assert all(d.reason == "weak" for d in result.dropped)

    def test_max_features_caps_and_says_so(self, dataset):
        result = select_features(
            SelectFeaturesInput(dataset_id=dataset, max_features=1)
        )
        assert len(result.selected) <= 1
        if result.n_clusters > 1:
            assert any(d.reason == "capped" for d in result.dropped)

    def test_it_agrees_with_get_feature_redundancy(self, dataset):
        """The two tools resolve the same clusters on the same panel. If
        they disagreed about which member to keep, an agent following both
        would get contradictory drop lists."""
        selected = select_features(
            SelectFeaturesInput(dataset_id=dataset, cluster_threshold=0.0)
        )
        redundancy = get_feature_redundancy(
            FeatureRedundancyInput(dataset_id=dataset, cluster_threshold=0.0)
        )
        assert set(selected.selected) == {c.representative for c in redundancy.clusters}


class TestCompareFeatureSets:
    def test_it_reports_both_sides_and_the_difference(self, dataset, features):
        result = compare_feature_sets(
            CompareFeatureSetsInput(
                dataset_id=dataset, left=features[:1], right=features
            )
        )
        assert result.left.n_features == 1
        assert result.right.n_features == len(features)
        assert result.delta.n_features == len(features) - 1
        assert result.only_in_right == sorted(set(features) - set(features[:1]))
        assert result.shared == features[:1]

    def test_identical_sets_have_a_zero_delta(self, dataset, features):
        result = compare_feature_sets(
            CompareFeatureSetsInput(dataset_id=dataset, left=features, right=features)
        )
        assert result.delta.n_features == 0
        assert result.delta.n_independent_signals == 0
        assert result.only_in_left == [] and result.only_in_right == []

    def test_independent_signals_never_exceeds_features(self, dataset, features):
        """The whole point of reporting it: it is the honest count of ideas
        where n_features counts columns."""
        result = compare_feature_sets(
            CompareFeatureSetsInput(
                dataset_id=dataset, left=features[:1], right=features
            )
        )
        for side in (result.left, result.right):
            assert side.n_independent_signals <= side.n_features

    def test_an_empty_side_is_rejected_by_the_schema(self, dataset, features):
        with pytest.raises(Exception):
            CompareFeatureSetsInput(dataset_id=dataset, left=[], right=features)


class TestDriftAndStabilityTools:
    def test_drift_splits_the_panel(self, dataset, features):
        result = get_feature_drift(
            FeatureDriftInput(dataset_id=dataset, feature=features[0])
        )
        assert result.n_before > 0 and result.n_after > 0
        assert result.psi_verdict in {"stable", "moderate", "significant"}

    def test_drift_verdict_matches_its_own_thresholds(self, dataset, features):
        from standard_quant_tools.modeling.analysis.feature_stability import (
            PSI_MODERATE,
            PSI_SIGNIFICANT,
        )

        result = get_feature_drift(
            FeatureDriftInput(dataset_id=dataset, feature=features[0])
        )
        if result.psi is None:
            return
        expected = (
            "significant"
            if result.psi >= PSI_SIGNIFICANT
            else "moderate" if result.psi >= PSI_MODERATE else "stable"
        )
        assert result.psi_verdict == expected

    def test_stability_returns_the_blocks_it_was_asked_for(self, dataset, features):
        result = get_feature_regime_stability(
            FeatureStabilityInput(dataset_id=dataset, feature=features[0], n_blocks=3)
        )
        assert result.n_blocks == 3
        assert len(result.blocks) == 3
        assert [b.block for b in result.blocks] == [0, 1, 2]

    def test_permutation_test_is_reproducible(self, dataset, features):
        args = dict(dataset_id=dataset, feature=features[0], n_permutations=30)
        first = run_feature_permutation_test(
            PermutationTestInput(**args, random_seed=4)
        )
        second = run_feature_permutation_test(
            PermutationTestInput(**args, random_seed=4)
        )
        assert first.p_value == second.p_value

    def test_permutation_test_never_reports_p_zero(self, dataset, features):
        result = run_feature_permutation_test(
            PermutationTestInput(
                dataset_id=dataset, feature=features[0], n_permutations=20
            )
        )
        assert result.p_value > 0.0

    def test_the_permutation_budget_is_bounded_by_the_schema(self, dataset, features):
        """Cost is linear in n_permutations, so the ceiling is in the schema
        rather than left to a caller who cannot see the runtime."""
        with pytest.raises(Exception):
            PermutationTestInput(
                dataset_id=dataset, feature=features[0], n_permutations=1_000_000
            )


class TestTheSplitFollowedTheRule:
    """Development/runtime_expansion_plan.md section 3, checked rather than
    asserted in prose."""

    FEATURE_TOOLS = [
        "profile_feature",
        "compare_feature_sets",
        "get_feature_drift",
        "get_feature_ic_decay",
        "get_feature_redundancy",
        "get_feature_regime_stability",
        "run_feature_ablation",
        "run_feature_permutation_test",
        "select_features",
    ]

    def test_the_new_runtime_clears_the_floor(self):
        """Rule 2: a runtime lands at >= 8 tools. A runtime holding two is
        overhead, not isolation."""
        from standard_quant_tools.modeling.agent.feature_tools import (
            FEATURE_TOOL_DISPATCH,
        )

        assert set(FEATURE_TOOL_DISPATCH) == set(self.FEATURE_TOOLS)
        assert len(FEATURE_TOOL_DISPATCH) >= 8

    def test_the_donor_still_clears_it_too(self):
        """Rule 3. The generic floor test already covers this -- it iterates
        every runtime, and a donor is still a runtime after a split -- but
        naming the donor here says WHICH move would have broken it."""
        from standard_quant_tools.modeling.agent import MODELING_TOOL_DISPATCH

        assert (
            len(MODELING_TOOL_DISPATCH) >= 8
        ), f"the split left modeling with {len(MODELING_TOOL_DISPATCH)} tools"

    def test_the_two_runtimes_do_not_overlap(self):
        """A tool duplicated into both would dissolve the boundary exactly
        where it matters."""
        from standard_quant_tools.modeling.agent import MODELING_TOOL_DISPATCH
        from standard_quant_tools.modeling.agent.feature_tools import (
            FEATURE_TOOL_DISPATCH,
        )

        assert not set(MODELING_TOOL_DISPATCH) & set(FEATURE_TOOL_DISPATCH)

    @pytest.mark.parametrize("name", FEATURE_TOOLS)
    def test_each_is_advertised_and_dispatchable(self, name):
        from standard_quant_tools.agent.runtimes import owner_of
        from standard_quant_tools.modeling.agent.feature_tools import (
            FEATURE_TOOL_DISPATCH,
            get_feature_tools,
        )

        assert name in FEATURE_TOOL_DISPATCH
        assert name in {d["function"]["name"] for d in get_feature_tools()}
        assert owner_of(name) == "feature_lab"
