"""
Redundancy says what VIF says (the CHANGELOG entry of 2026-10-04).

On the live panel `get_feature_redundancy` returned `redundant_features: []`
beside a VIF of 7.46 for rsi_14, with nothing connecting the two: the
clusters join PAIRS at |r| >= 0.9, and rsi_14 is explained by three other
features together (pctb_20 at |r| 0.87 most of all). The condition-number
warning was drawn at 30 -- Belsley's line, which is for the condition
INDEX, the square root -- so it fired at an index of 5.5, while
`analyze_features` drew it at 1000 and `select_features` never. And a
singular matrix reported VIFs below 1 (an exact copy came out at 0.25) and
a finite condition number of ~1e16.

What these tests hold:

- a planted joint dependency with no close pair is named in
  `collinear_features` with the features behind it, and the empty drop list
  beside it carries a sentence saying why;
- `explained_by` is the shortest prefix, strongest partial correlation
  first, whose regression reaches 90% of the full R-squared;
- a singular matrix has an infinite condition number and an infinite VIF
  (None) for each feature in the exact combination, never a VIF below 1;
  every matrix that is not singular takes the old path, to the bit;
- independent features raise no collinearity sentence at all;
- the condition-number sentence is one sentence, at one line, in every tool;
- the redundancy tool and the selection publish the same block on the same
  dates, and a single-member cluster reports no correlation rather than 1.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling.agent.feature_models import (
    FeatureRedundancyInput,
    SelectFeaturesInput,
)
from standard_quant_tools.modeling.agent.feature_tools import (
    FEATURE_TOOL_DEFS,
    get_feature_redundancy,
    select_features,
)
from standard_quant_tools.modeling.agent.models import RegisterExternalPanelInput
from standard_quant_tools.modeling.agent.tools import register_external_panel
from standard_quant_tools.modeling.analysis.feature_report import (
    CONDITION_WARN,
    build_feature_report,
    cluster_records,
    condition_warning,
    redundancy_report,
)
from standard_quant_tools.modeling.analysis.feature_selection import (
    select_features as select_features_on,
)

# ── planted panels ───────────────────────────────────────────────────────


def _frame(columns: dict, *, n_entities: int = 20, seed: int = 0) -> pd.DataFrame:
    """A panel around the given feature columns, with a target that loads
    weakly on the first of them."""
    n = len(next(iter(columns.values())))
    rng = np.random.default_rng(seed)
    n_dates = n // n_entities
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), n_entities)[:n],
            "entity": np.tile([f"E{j:02d}" for j in range(n_entities)], n_dates)[:n],
            **columns,
        }
    )
    first = next(iter(columns.values()))
    frame["target"] = 0.1 * first + rng.normal(size=n)
    return frame


def _joint(seed: int = 0, n: int = 4000, noise: float = 0.3) -> pd.DataFrame:
    """`c` is `a + b` plus a little noise: no pair reaches |r| 0.9 (each is
    about 0.69), and the others explain 96% of `c`'s variance."""
    rng = np.random.default_rng(seed)
    a, b, d = rng.standard_normal((3, n))
    return _frame({"a": a, "b": b, "c": a + b + noise * rng.standard_normal(n), "d": d})


def _independent(seed: int = 1, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _frame({name: rng.standard_normal(n) for name in "wxyz"})


def _register(frame: pd.DataFrame, tmp_path, name: str = "panel") -> str:
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    return register_external_panel(
        RegisterExternalPanelInput(path=str(path), horizon=5)
    ).dataset_id


def _head_vif_and_condition(panel, features):
    """The VIF and condition number as computed before the singular case
    was told apart: eigvalsh for the condition number, the diagonal of
    pinv for the VIF."""
    matrix = panel[features].dropna().corr().to_numpy(dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    smallest, largest = float(np.min(eigenvalues)), float(np.max(eigenvalues))
    condition = largest / smallest if smallest > 0 else float("inf")
    inverse = np.linalg.pinv(matrix)
    return {f: float(inverse[i, i]) for i, f in enumerate(features)}, condition


# ── the block ────────────────────────────────────────────────────────────


class TestAJointDependencyIsNamed:
    def test_the_planted_combination_is_named_with_what_explains_it(self):
        report = redundancy_report(_joint(), list("abcd"))
        assert all(len(cluster) == 1 for cluster in report["clusters"])
        by_feature = {entry["feature"]: entry for entry in report["collinear"]}
        assert set(by_feature) == {"a", "b", "c"}
        # Highest first.
        assert [e["feature"] for e in report["collinear"]][0] == "c"
        c = by_feature["c"]
        assert c["vif"] > 10
        assert c["r_squared"] == pytest.approx(1 - 1 / c["vif"])
        assert {part["feature"] for part in c["explained_by"]} == {"a", "b"}
        assert c["in_cluster"] is False
        # The pairwise view sees almost none of it.
        assert c["vif_from_strongest_pair"] < 2.5 < c["vif"] / 4

    def test_the_drop_list_beside_it_says_why_it_is_empty(self, tmp_path):
        dataset_id = _register(_joint(), tmp_path)
        result = get_feature_redundancy(FeatureRedundancyInput(dataset_id=dataset_id))
        assert result.redundant_features == []
        assert {entry.feature for entry in result.collinear_features} == {"a", "b", "c"}
        action = next(
            w for w in result.warnings if w.startswith("VIF is at or above 10")
        )
        assert "c (" in action and "not individually interpretable" in action
        reconcile = next(
            w for w in result.warnings if "redundant_features lists pairs" in w
        )
        assert "|r| >= 0.90 only" in reconcile
        assert "modelling choice, not a deduplication" in reconcile

    @pytest.mark.parametrize("seed", range(5))
    def test_explained_by_is_the_shortest_prefix_reaching_90_percent(self, seed):
        rng = np.random.default_rng(seed)
        n = 3000
        base = rng.standard_normal((5, n))
        columns = {f"f{k}": base[k] for k in range(5)}
        columns["mix"] = (
            base[0] + 0.6 * base[1] + 0.3 * base[2] + 0.4 * rng.standard_normal(n)
        )
        frame = _frame(columns)
        names = list(columns)
        report = redundancy_report(frame, names)
        matrix = frame[names].corr().to_numpy()
        index = {name: k for k, name in enumerate(names)}
        assert report["collinear"], "the planted mix should clear VIF 5"
        for entry in report["collinear"]:
            i = index[entry["feature"]]
            chosen = [index[part["feature"]] for part in entry["explained_by"]]

            def r2(subset):
                r = matrix[i, subset]
                return float(r @ np.linalg.solve(matrix[np.ix_(subset, subset)], r))

            target = 0.9 * entry["r_squared"]
            assert len(chosen) <= 5
            if len(chosen) < 5:
                assert r2(chosen) >= target - 1e-12
            if len(chosen) > 1:
                assert r2(chosen[:-1]) < target
            partials = [
                abs(part["partial_correlation"]) for part in entry["explained_by"]
            ]
            assert partials == sorted(partials, reverse=True)


class TestASingularMatrixIsOne:
    def test_an_exact_copy_has_infinite_vif_not_a_quarter(self):
        rng = np.random.default_rng(0)
        a, d = rng.standard_normal((2, 2000))
        report = redundancy_report(
            _frame({"a": a, "a_copy": a.copy(), "d": d}), ["a", "a_copy", "d"]
        )
        assert report["condition_number"] == float("inf")
        assert report["vif"]["a"] is None and report["vif"]["a_copy"] is None
        assert report["vif"]["d"] == pytest.approx(1.0, abs=0.01)
        by_feature = {entry["feature"]: entry for entry in report["collinear"]}
        assert set(by_feature) == {"a", "a_copy"}
        assert by_feature["a"]["r_squared"] == 1.0
        assert [p["feature"] for p in by_feature["a"]["explained_by"]] == ["a_copy"]
        assert by_feature["a"]["explained_by"][0]["partial_correlation"] is None

    def test_an_exact_sum_names_the_whole_combination(self):
        """Before: VIF 0.62 / 0.25 and a condition number of 1.8e16."""
        rng = np.random.default_rng(0)
        a, b, d = rng.standard_normal((3, 2000))
        report = redundancy_report(
            _frame({"a": a, "b": b, "ab": a + b, "d": d}), ["a", "b", "ab", "d"]
        )
        assert report["condition_number"] == float("inf")
        assert {f for f, v in report["vif"].items() if v is None} == {"a", "b", "ab"}
        assert report["vif"]["d"] >= 1.0
        ab = next(e for e in report["collinear"] if e["feature"] == "ab")
        assert {p["feature"] for p in ab["explained_by"]} == {"a", "b"}

    @pytest.mark.parametrize("seed", range(6))
    def test_no_vif_is_ever_below_one(self, seed):
        rng = np.random.default_rng(seed)
        a, b, c, d = rng.standard_normal((4, 1500))
        columns = {
            "a": a,
            "b": b,
            "copy": a.copy(),
            "sum": a + b,
            "near": a + 1e-3 * c,
            "d": d,
        }
        report = redundancy_report(_frame(columns), list(columns))
        for value in report["vif"].values():
            assert value is None or value >= 1.0 - 1e-9

    def test_the_warning_names_the_exact_combination(self):
        rng = np.random.default_rng(0)
        a, d = rng.standard_normal((2, 2000))
        report = build_feature_report(
            _frame({"a": a, "a_copy": a.copy(), "d": d}),
            ["a", "a_copy", "d"],
            include_leakage=False,
        )
        singular = next(w for w in report["warnings"] if "is singular:" in w)
        assert "a, a_copy are exact linear combinations" in singular
        assert any("condition number is infinite" in w for w in report["warnings"])

    @pytest.mark.parametrize(
        "builder", [_joint, _independent], ids=["joint", "independent"]
    )
    def test_a_matrix_that_is_not_singular_takes_the_old_path(self, builder):
        frame = builder()
        features = [c for c in frame.columns if c not in ("date", "entity", "target")]
        vif, condition = _head_vif_and_condition(frame, features)
        report = redundancy_report(frame, features)
        assert report["vif"] == vif
        assert report["condition_number"] == condition


class TestIndependentFeaturesSayNothing:
    def test_no_entry_and_no_sentence(self, tmp_path):
        frame = _independent()
        report = redundancy_report(frame, list("wxyz"))
        assert report["collinear"] == []
        dataset_id = _register(frame, tmp_path)
        result = get_feature_redundancy(FeatureRedundancyInput(dataset_id=dataset_id))
        assert result.collinear_features == []
        assert not any("VIF" in w for w in result.warnings)
        assert not any("condition number" in w for w in result.warnings)


class TestOneConditionNumberSentence:
    """The line moved from 30 (the redundancy tool), 1000 (analyze_features)
    and nowhere (select_features) to 1000 in all three, worded once."""

    @staticmethod
    def _ill_conditioned(seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        a, b = rng.standard_normal((2, 4000))
        return _frame(
            {"a": a, "b": b, "near": a + b + 0.02 * rng.standard_normal(4000)}
        )

    def test_every_tool_says_the_same_sentence_past_1000(self, tmp_path):
        frame = self._ill_conditioned()
        features = ["a", "b", "near"]
        condition = redundancy_report(frame, features)["condition_number"]
        assert condition >= CONDITION_WARN
        sentence = condition_warning(condition)
        assert sentence and sentence.startswith(
            "NOTE: the feature correlation matrix has condition number"
        )

        dataset_id = _register(frame, tmp_path)
        report = build_feature_report(frame, features, include_leakage=False)
        redundancy = get_feature_redundancy(
            FeatureRedundancyInput(dataset_id=dataset_id)
        )
        selection = select_features(
            SelectFeaturesInput(
                dataset_id=dataset_id, holdout_fraction=0.0, significance="none"
            )
        )
        for warnings in (report["warnings"], redundancy.warnings, selection.warnings):
            assert sentence in warnings

    def test_a_condition_number_of_36_is_not_warned_about(self, tmp_path):
        """The live panel's 36: a condition INDEX of 6, healthy on
        Belsley's own scale. The redundancy tool used to warn at it."""
        rng = np.random.default_rng(3)
        a, b, c = rng.standard_normal((3, 4000))
        frame = _frame({"a": a, "b": b, "c": a + 0.35 * b + 0.25 * c})
        condition = redundancy_report(frame, ["a", "b", "c"])["condition_number"]
        assert 30 < condition < CONDITION_WARN
        dataset_id = _register(frame, tmp_path)
        result = get_feature_redundancy(FeatureRedundancyInput(dataset_id=dataset_id))
        assert not any("condition number" in w for w in result.warnings)

    def test_the_sentence_is_absent_below_the_line_and_present_at_infinity(self):
        assert condition_warning(999.0) is None
        assert condition_warning(float("nan")) is None
        assert condition_warning(None) is None
        assert "singular" in condition_warning(float("inf"))


class TestTheTwoToolsPublishOneBlock:
    def test_same_dates_same_block(self, tmp_path):
        dataset_id = _register(_joint(), tmp_path)
        redundancy = get_feature_redundancy(
            FeatureRedundancyInput(dataset_id=dataset_id)
        )
        selection = select_features(
            SelectFeaturesInput(
                dataset_id=dataset_id, holdout_fraction=0.0, significance="none"
            )
        )
        assert redundancy.collinear_features
        assert selection.collinear_features == redundancy.collinear_features
        assert selection.clusters == redundancy.clusters

    def test_select_features_reconciles_in_its_own_terms(self, tmp_path):
        result = select_features_on(_joint(), list("abcd"), significance="none")
        sentence = next(w for w in result["warnings"] if "no such pair" in w)
        assert sentence.startswith("The redundancy drops are made for pairs at")

    def test_analyze_features_carries_the_block_untyped(self):
        report = build_feature_report(_joint(), list("abcd"), include_leakage=False)
        assert (
            report["redundancy"]["collinear"]
            == redundancy_report(_joint(), list("abcd"))["collinear"]
        )
        assert any(
            w.startswith("report.redundancy.clusters groups pairs")
            for w in report["warnings"]
        )


class TestASingletonRestatesNothing:
    def test_a_single_member_cluster_reports_no_correlation(self, tmp_path):
        rng = np.random.default_rng(0)
        a, d = rng.standard_normal((2, 2000))
        frame = _frame({"a": a, "a_near": a + 0.05 * rng.standard_normal(2000), "d": d})
        dataset_id = _register(frame, tmp_path)
        result = get_feature_redundancy(FeatureRedundancyInput(dataset_id=dataset_id))
        by_size = {cluster.size: cluster for cluster in result.clusters}
        assert by_size[1].members == ["d"]
        assert by_size[1].max_abs_correlation is None
        assert by_size[2].max_abs_correlation > 0.99
        selection = select_features(
            SelectFeaturesInput(
                dataset_id=dataset_id, holdout_fraction=0.0, significance="none"
            )
        )
        assert selection.clusters == result.clusters

    def test_one_builder_for_both(self):
        predictive = {"x": {"rank_ic_mean": 0.02}, "y": {"rank_ic_mean": float("nan")}}
        records = cluster_records(
            [["y", "x"], ["z"]], {"x": {"y": 0.95}, "y": {"x": 0.95}}, predictive
        )
        assert records[0] == {
            "members": ["x", "y"],
            "representative": "x",
            "max_abs_correlation": 0.95,
            "size": 2,
        }
        assert records[1]["max_abs_correlation"] is None

    def test_the_cluster_warning_says_r(self, tmp_path):
        rng = np.random.default_rng(0)
        a, d = rng.standard_normal((2, 2000))
        frame = _frame({"a": a, "a_near": a + 0.05 * rng.standard_normal(2000), "d": d})
        dataset_id = _register(frame, tmp_path)
        result = get_feature_redundancy(FeatureRedundancyInput(dataset_id=dataset_id))
        sentence = next(w for w in result.warnings if "one signal at" in w)
        assert "|r| >= 0.90" in sentence and "rho" not in sentence


class TestTheRedundancyDescription:
    def test_it_does_not_claim_a_fixed_cluster(self):
        text = {name: t for name, t, _ in FEATURE_TOOL_DEFS}["get_feature_redundancy"]
        assert "are one momentum cluster" not in text
        assert "can form one" in text
        assert "collinear_features" in text
        assert "~30" not in text
