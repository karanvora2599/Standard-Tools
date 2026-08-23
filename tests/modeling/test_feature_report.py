"""
The feature report, and in particular whether its leakage screen works.

Most of this file is ordinary contract testing. The exception is
`TestLeakageScreen`, which is the part that has to earn its place: a
detector that flags honest features is worse than no detector, because an
agent will learn to ignore it. So the screen is tested from both sides —
against a planted leak it must catch, and against honest features with
awkward shapes it must NOT flag.

The awkward shapes are not hypothetical. An earlier version of the screen
asked "did advancing the feature in time improve its IC", which is the right
question for a path-dependent feature (momentum, RSI) and the wrong one for a
slow-moving state feature (realized volatility, ADX) whose predictive content
is about the regime rather than the price path. It produced false positives
on real features in the catalog. The tests below pin the corrected behaviour.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.analysis import (
    build_feature_report,
    feature_distribution_stats,
    feature_predictive_stats,
    lead_lag_ic_curve,
    redundancy_report,
)

HORIZON = 5


def _panel(n_dates=200, n_entities=25, seed=0, freq="B"):
    """
    A panel shaped like a real one: the target is a FORWARD return.

    That detail is load-bearing for the leakage screen, and getting it wrong
    is how the first version of this fixture produced a false failure. The
    screen's whole premise is that `target[t]` spans bars t..t+horizon, so a
    feature evaluated later has seen part of the answer and its IC rises when
    advanced. Against a CONTEMPORANEOUS target — a target that is just a
    function of the same bar — even a perfectly honest feature peaks at shift
    0, because the relationship genuinely lives at lag 0. The screen would be
    right to flag it and the test would be wrong to expect otherwise.

    So `signal` here is momentum-shaped: a trailing sum of returns, plus a
    component that genuinely predicts the forward window. Advancing it lets
    it observe returns that are inside the target, exactly as a real
    path-dependent feature does.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq=freq)
    rows = []
    for entity in range(n_entities):
        returns = rng.normal(0, 1, n_dates + HORIZON)
        # Forward return over the next HORIZON bars, which is what every
        # target in this package actually is.
        target = np.array(
            [returns[i + 1 : i + 1 + HORIZON].sum() for i in range(n_dates)]
        )
        # Trailing 5-bar return: known at t, and at t+k it has observed
        # returns that lie inside target[t].
        trailing = np.array(
            [returns[max(0, i - 4) : i + 1].sum() for i in range(n_dates)]
        )
        signal = trailing + 0.6 * target + rng.normal(0, 2.0, n_dates)
        noise = rng.normal(0, 1, n_dates)
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "entity": f"E{entity:03d}",
                    "signal": signal[i],
                    "noise": noise[i],
                    # a near-duplicate of signal, which redundancy must catch
                    "signal_copy": signal[i] + rng.normal(0, 0.02),
                    "target": target[i],
                }
            )
    return pd.DataFrame(rows)


FEATURES = ["signal", "noise", "signal_copy"]


class TestDistribution:
    def test_coverage_and_moments(self):
        """Moments are compared against pandas rather than against numbers
        baked in from the fixture, so changing the fixture cannot silently
        make this assert something else."""
        panel = _panel()
        stats = feature_distribution_stats(panel, FEATURES)
        assert stats["signal"]["coverage"] == pytest.approx(1.0)
        assert stats["signal"]["n_missing"] == 0
        for key, expected in (
            ("mean", panel["signal"].mean()),
            ("std", panel["signal"].std()),
            ("skew", panel["signal"].skew()),
            ("kurtosis", panel["signal"].kurt()),
        ):
            assert stats["signal"][key] == pytest.approx(expected, abs=1e-12), key

    def test_missing_values_are_counted(self):
        panel = _panel()
        panel.loc[panel.index[:500], "noise"] = np.nan
        stats = feature_distribution_stats(panel, ["noise"])
        assert stats["noise"]["n_missing"] == 500
        assert stats["noise"]["coverage"] < 1.0

    def test_an_entirely_missing_feature_does_not_raise(self):
        panel = _panel()
        panel["empty"] = np.nan
        stats = feature_distribution_stats(panel, ["empty"])
        assert stats["empty"]["coverage"] == 0.0
        assert np.isnan(stats["empty"]["mean"])

    def test_constant_feature_has_no_outliers(self):
        """Zero dispersion means nothing can be an outlier in it — 0.0 is the
        answer, not a division by zero."""
        panel = _panel()
        panel["flat"] = 3.0
        stats = feature_distribution_stats(panel, ["flat"])
        assert stats["flat"]["outlier_rate"] == 0.0

    def test_turnover_separates_a_persistent_feature_from_a_random_one(self):
        """
        The point of reporting turnover at all: two features with the same IC
        and very different turnover are not equally useful, and IC cannot see
        the difference.
        """
        panel = _panel()
        # A slow feature: each entity's value barely changes.
        panel = panel.sort_values(["entity", "date"])
        panel["slow"] = panel.groupby("entity", sort=False).cumcount() * 0.001
        stats = feature_distribution_stats(panel, ["slow", "noise"])
        assert stats["slow"]["turnover"] < stats["noise"]["turnover"]

    def test_autocorrelation_is_within_entity(self):
        """Pooled over the stacked column, autocorrelation would mostly
        measure that consecutive rows belong to different entities."""
        panel = _panel().sort_values(["entity", "date"])
        panel["ramp"] = panel.groupby("entity", sort=False).cumcount().astype(float)
        stats = feature_distribution_stats(panel, ["ramp"])
        assert stats["ramp"]["autocorrelation"] > 0.99


class TestPredictive:
    def test_signal_scores_above_noise(self):
        panel = _panel()
        stats = feature_predictive_stats(panel, FEATURES)
        assert abs(stats["signal"]["rank_ic_mean"]) > abs(
            stats["noise"]["rank_ic_mean"]
        )
        assert stats["signal"]["rank_ic_mean"] > 0.03, stats["signal"]

    def test_quantile_spread_has_the_sign_of_the_relationship(self):
        panel = _panel()
        stats = feature_predictive_stats(panel, ["signal"])
        assert stats["signal"]["quantile_spread"] > 0
        # A cleanly linear relationship should be close to monotone in the
        # deciles, which is what distinguishes "usable" from "real but only
        # in one tail".
        assert stats["signal"]["monotonicity"] > 0.8

    def test_inverted_feature_inverts_the_spread(self):
        panel = _panel()
        panel["inverted"] = -panel["signal"]
        stats = feature_predictive_stats(panel, ["inverted"])
        assert stats["inverted"]["quantile_spread"] < 0
        assert stats["inverted"]["monotonicity"] < -0.8

    def test_ic_matches_the_engine_definition(self):
        """The report's IC must be the same quantity the engine reports on a
        model, computed by the same code — otherwise a feature's standalone
        number and a model's number are not comparable."""
        from standard_quant_tools.modeling.validation.metrics import (
            cross_sectional_ic,
            summarize_cross_sectional_ic,
        )

        panel = _panel()
        stats = feature_predictive_stats(panel, ["signal"])
        expected = summarize_cross_sectional_ic(
            cross_sectional_ic(
                panel["target"].to_numpy(),
                panel["signal"].to_numpy(),
                panel["date"].to_numpy(),
                "spearman",
            ),
            "rank_ic",
        )
        assert stats["signal"]["rank_ic_mean"] == pytest.approx(
            expected["rank_ic_mean"], abs=1e-12
        )

    def test_too_few_entities_for_the_requested_buckets(self):
        """Deciles over eight entities is not deciles. The shape statistics
        report NaN rather than a number built on one entity per bucket."""
        panel = _panel(n_dates=50, n_entities=8)
        stats = feature_predictive_stats(panel, ["signal"], n_quantiles=10)
        assert np.isnan(stats["signal"]["quantile_spread"])


class TestRedundancy:
    def test_near_duplicates_are_clustered(self):
        panel = _panel()
        report = redundancy_report(panel, FEATURES)
        clusters = [set(c) for c in report["clusters"]]
        assert {"signal", "signal_copy"} in clusters
        assert {"noise"} in clusters

    def test_vif_is_high_for_the_duplicate_pair(self):
        panel = _panel()
        report = redundancy_report(panel, FEATURES)
        assert report["vif"]["signal"] > 10
        assert report["vif"]["signal_copy"] > 10
        assert report["vif"]["noise"] < 2

    def test_independent_features_are_their_own_clusters(self):
        rng = np.random.default_rng(1)
        panel = _panel()
        panel["a"] = rng.normal(0, 1, len(panel))
        panel["b"] = rng.normal(0, 1, len(panel))
        report = redundancy_report(panel, ["a", "b"])
        assert sorted(len(c) for c in report["clusters"]) == [1, 1]
        assert report["condition_number"] < 2

    def test_a_single_feature_needs_no_matrix(self):
        report = redundancy_report(_panel(), ["signal"])
        assert report["clusters"] == [["signal"]]

    def test_perfectly_collinear_features_report_infinite_condition_number(self):
        panel = _panel()
        panel["exact_copy"] = panel["signal"]
        report = redundancy_report(panel, ["signal", "exact_copy"])
        assert not np.isfinite(report["condition_number"])


class TestLeakageScreen:
    """
    Both directions, because a detector that cries wolf is worse than none.
    """

    def test_catches_a_feature_that_reads_its_own_target(self):
        panel = _panel()
        rng = np.random.default_rng(2)
        panel["leaky"] = panel["target"] + rng.normal(0, 0.05, len(panel))
        result = lead_lag_ic_curve(panel, "leaky", max_shift=5)
        assert result["flagged"], result["reason"]
        # The signature is a TENT: peak at zero, falling away on both sides.
        curve = {int(k): v for k, v in result["curve"].items()}
        assert curve[0] > curve[-1]
        assert curve[0] > curve[1]
        assert result["peak_ratio"] > 1.0

    def test_does_not_flag_the_same_feature_once_it_is_properly_lagged(self):
        """
        The control that makes the previous test mean something. The same
        leaking column, shifted back so it only knows the past, must come
        back clean — otherwise the screen is detecting "strong feature"
        rather than "leak".
        """
        panel = _panel().sort_values(["entity", "date"])
        rng = np.random.default_rng(2)
        leak = panel["target"] + rng.normal(0, 0.05, len(panel))
        panel["lagged"] = leak.groupby(panel["entity"]).shift(10).reindex(panel.index)
        panel = panel.dropna(subset=["lagged"])
        result = lead_lag_ic_curve(panel, "lagged", max_shift=5)
        assert not result["flagged"], result["reason"]

    def test_does_not_flag_a_pure_noise_feature(self):
        """A feature with no signal has a curve made of noise, where peaks
        appear everywhere and mean nothing. The floor must catch this."""
        panel = _panel()
        result = lead_lag_ic_curve(panel, "noise", max_shift=5)
        assert not result["flagged"]
        assert "below the" in result["reason"]

    def test_does_not_flag_an_honest_predictive_feature(self):
        panel = _panel()
        result = lead_lag_ic_curve(panel, "signal", max_shift=5)
        assert not result["flagged"], result["reason"]

    def test_abstains_on_a_feature_too_persistent_to_judge(self):
        """
        A feature that barely moves across the shift window is being compared
        against a near-copy of itself, so its flat curve says nothing either
        way. The screen must say that rather than guess.
        """
        panel = _panel().sort_values(["entity", "date"])
        # Almost constant within each entity: persistence ~1.
        panel["sticky"] = (
            panel.groupby("entity", sort=False)["target"].transform("mean")
            + panel.groupby("entity", sort=False).cumcount() * 1e-9
        )
        result = lead_lag_ic_curve(panel, "sticky", max_shift=5)
        assert not result["flagged"]
        if np.isfinite(result["persistence"]):
            assert abs(result["persistence"]) > 0.9

    def test_reason_is_always_populated(self):
        panel = _panel()
        for feature in FEATURES:
            result = lead_lag_ic_curve(panel, feature, max_shift=3)
            assert result["reason"], feature

    def test_rejects_a_nonsense_shift(self):
        with pytest.raises(ValidationError, match="max_shift"):
            lead_lag_ic_curve(_panel(), "signal", max_shift=0)


class TestBuildFeatureReport:
    def test_assembles_every_layer(self):
        report = build_feature_report(_panel(), FEATURES)
        assert set(report["features"]) == set(FEATURES)
        assert "redundancy" in report and "leakage" in report
        assert report["n_features"] == 3
        assert report["n_entities"] == 25

    def test_leakage_can_be_skipped(self):
        report = build_feature_report(_panel(), FEATURES, include_leakage=False)
        assert "leakage" not in report

    def test_warns_about_near_duplicate_features(self):
        report = build_feature_report(_panel(), FEATURES, include_leakage=False)
        assert any("near-duplicates" in w for w in report["warnings"])

    def test_warns_about_a_thinly_populated_feature(self):
        panel = _panel()
        panel.loc[panel.index[: int(len(panel) * 0.8)], "noise"] = np.nan
        report = build_feature_report(panel, FEATURES, include_leakage=False)
        assert any("under half the panel" in w for w in report["warnings"])

    def test_warns_when_a_leak_is_flagged(self):
        panel = _panel()
        rng = np.random.default_rng(3)
        panel["leaky"] = panel["target"] + rng.normal(0, 0.05, len(panel))
        report = build_feature_report(panel, ["signal", "leaky"])
        assert report["leakage_flagged"] == ["leaky"]
        assert any(w.startswith("WARNING") for w in report["warnings"])

    def test_rejects_an_unknown_feature(self):
        with pytest.raises(ValidationError, match="no column"):
            build_feature_report(_panel(), ["not_a_column"])

    def test_rejects_a_frame_that_is_not_a_panel(self):
        with pytest.raises(ValidationError, match="missing required column"):
            build_feature_report(pd.DataFrame({"a": [1.0, 2.0]}), ["a"])

    def test_rejects_an_empty_feature_list(self):
        with pytest.raises(ValidationError, match="no features"):
            build_feature_report(_panel(), [])

    def test_result_is_json_safe(self):
        import json

        from standard_quant_tools._jsonsafe import sanitize_for_json

        report = build_feature_report(_panel(), FEATURES)
        json.dumps(sanitize_for_json(report))


class TestAnalyzeFeaturesTool:
    def test_is_registered_with_a_schema(self):
        from standard_quant_tools.modeling.agent.tools import (
            MODELING_TOOL_DISPATCH,
            get_modeling_tools,
        )

        assert "analyze_features" in MODELING_TOOL_DISPATCH
        names = [t["function"]["name"] for t in get_modeling_tools()]
        assert "analyze_features" in names
        # Dispatch and schema list must not drift apart.
        assert sorted(names) == sorted(MODELING_TOOL_DISPATCH)

    def test_runs_end_to_end_from_a_dataset_id(self, patched_multi_factory):
        from standard_quant_tools.modeling.agent.models import (
            AnalyzeFeaturesInput,
            BuildModelDatasetInput,
        )
        from standard_quant_tools.modeling.agent.tools import (
            analyze_features,
            build_model_dataset,
        )
        from standard_quant_tools.modeling.specs import (
            DatasetSpec,
            FeatureSpec,
            TargetSpec,
        )

        built = build_model_dataset(
            BuildModelDatasetInput(
                spec=DatasetSpec(
                    universe=["AAA", "BBB", "CCC", "DDD"],
                    start="2022-01-01",
                    end="2030-01-01",
                    features=[
                        FeatureSpec(id="technical.rsi"),
                        FeatureSpec(id="market.momentum"),
                    ],
                    target=TargetSpec(horizon=5),
                    benchmark="SPY",
                )
            )
        )
        result = analyze_features(
            AnalyzeFeaturesInput(dataset_id=built.dataset_id, leakage_max_shift=2)
        )
        assert result.dataset_id == built.dataset_id
        assert set(result.report["features"]) == {"technical.rsi", "market.momentum"}
        assert "redundancy" in result.report
        assert isinstance(result.warnings, list)

    def test_unknown_dataset_id_is_rejected(self):
        from standard_quant_tools.modeling.agent.models import AnalyzeFeaturesInput
        from standard_quant_tools.modeling.agent.tools import analyze_features

        with pytest.raises(ValidationError, match="no dataset with dataset_id"):
            analyze_features(AnalyzeFeaturesInput(dataset_id="does-not-exist"))
