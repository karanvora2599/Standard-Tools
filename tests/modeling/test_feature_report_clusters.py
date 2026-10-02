"""
The redundancy report's clusters and nested matrices, against the code
they replaced.

`_correlation_clusters` was a union-find that made one `.loc` lookup per
pair of features, and `_frame_to_nested` one `.loc` per cell; both were
quadratic in pandas calls (see the CHANGELOG entry of 2026-10-01). The
replacements read the matrix once. The bar is identity, not closeness: the
same clusters with the same members in the same order, and the same keys
in the same order mapping to the same floats. Both originals are kept here,
verbatim, as the reference.
"""

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling.analysis.feature_report import (
    _correlation_clusters,
    _frame_to_nested,
    _safe,
    redundancy_report,
)

# ── The replaced implementations ─────────────────────────────────────────


def _reference_frame_to_nested(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return {
        str(row): {str(col): _safe(frame.loc[row, col]) for col in frame.columns}
        for row in frame.index
    }


def _reference_correlation_clusters(
    correlation: pd.DataFrame, threshold: float
) -> List[List[str]]:
    names = list(correlation.columns)
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= threshold:
                union(left, right)

    groups: Dict[str, List[str]] = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


# ── Helpers ──────────────────────────────────────────────────────────────


def _same_nested(left, right) -> None:
    assert list(left) == list(right)
    for row in left:
        assert list(left[row]) == list(right[row]), row
        for col, a in left[row].items():
            b = right[row][col]
            assert type(a) is type(b) is float
            assert (math.isnan(a) and math.isnan(b)) or a == b, (row, col, a, b)


def _block_correlation(n_features, seed, n_rows=600, permute=True):
    """Features drawn around a handful of latent factors, at random noise
    levels, so a 0.9 threshold leaves some clusters and some singletons."""
    rng = np.random.default_rng(seed)
    n_latent = max(1, n_features // 4)
    latent = rng.normal(size=(n_rows, n_latent))
    columns = {}
    for j in range(n_features):
        sign = 1.0 if rng.random() < 0.7 else -1.0
        noise = rng.uniform(0.05, 1.5)
        columns[f"f{j:03d}"] = sign * latent[:, rng.integers(n_latent)] + noise * (
            rng.normal(size=n_rows)
        )
    frame = pd.DataFrame(columns)
    if permute:
        frame = frame[list(rng.permutation(frame.columns))]
    return frame.corr()


# ── Clusters ─────────────────────────────────────────────────────────────


class TestClustersMatchTheReference:
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize("n_features", [2, 3, 7, 24, 64])
    @pytest.mark.parametrize("threshold", [0.5, 0.8, 0.9, 0.95])
    def test_random_block_matrices(self, seed, n_features, threshold):
        corr = _block_correlation(n_features, seed)
        assert _correlation_clusters(corr, threshold) == (
            _reference_correlation_clusters(corr, threshold)
        )

    def test_planted_clusters(self):
        """Three exact restatements, a pair, and two independents; the
        answer is known before anything runs."""
        rng = np.random.default_rng(0)
        base, other = rng.normal(size=500), rng.normal(size=500)
        frame = pd.DataFrame(
            {
                "zeta": rng.normal(size=500),
                "b2": 2.0 * base + 1.0,
                "pair_y": -other,
                "b1": base,
                "alpha": rng.normal(size=500),
                "pair_x": other,
                "b3": -0.5 * base,
            }
        )
        expected = [["b2", "b1", "b3"], ["pair_y", "pair_x"], ["alpha"], ["zeta"]]
        corr = frame.corr()
        assert _correlation_clusters(corr, 0.9) == expected
        assert _reference_correlation_clusters(corr, 0.9) == expected

    def test_null_case_is_all_singletons(self):
        rng = np.random.default_rng(1)
        corr = pd.DataFrame(rng.normal(size=(2000, 12)), columns=list("abcdefghijkl"))
        corr = corr.corr()
        expected = [[name] for name in "abcdefghijkl"]
        assert _correlation_clusters(corr, 0.9) == expected
        assert _reference_correlation_clusters(corr, 0.9) == expected

    def test_a_single_cluster(self):
        rng = np.random.default_rng(2)
        base = rng.normal(size=300)
        frame = pd.DataFrame(
            {f"x{i}": base + 0.01 * rng.normal(size=300) for i in range(9)}
        )
        corr = frame[[f"x{i}" for i in (4, 0, 8, 2, 6, 1, 7, 3, 5)]].corr()
        assert _correlation_clusters(corr, 0.9) == [list(corr.columns)]
        assert _reference_correlation_clusters(corr, 0.9) == [list(corr.columns)]

    def test_chains_close_transitively(self):
        """a~b and b~c above the threshold, a~c below: one cluster."""
        corr = pd.DataFrame(
            [[1.0, 0.95, 0.5], [0.95, 1.0, 0.95], [0.5, 0.95, 1.0]],
            index=list("abc"),
            columns=list("abc"),
        )
        assert _correlation_clusters(corr, 0.9) == [["a", "b", "c"]]
        assert _reference_correlation_clusters(corr, 0.9) == [["a", "b", "c"]]

    def test_missing_correlations_ties_and_the_boundary(self):
        """NaN is no edge, |corr| equal to the threshold is an edge, and
        a negative correlation counts by its magnitude."""
        names = list("pqrstu")
        values = np.eye(6)
        values[0, 1] = values[1, 0] = np.nan
        values[0, 2] = values[2, 0] = 0.9
        values[3, 4] = values[4, 3] = -0.9
        values[1, 5] = values[5, 1] = 0.8999999999999999
        corr = pd.DataFrame(values, index=names, columns=names)
        for threshold in (0.9, 0.8999999999999999, 0.95, float("nan")):
            assert _correlation_clusters(corr, threshold) == (
                _reference_correlation_clusters(corr, threshold)
            )
        assert _correlation_clusters(corr, 0.9) == [
            ["p", "r"],
            ["s", "t"],
            ["q"],
            ["u"],
        ]

    def test_only_the_upper_triangle_is_read(self):
        """The reference visited each pair once, (earlier, later) in column
        order; an asymmetric matrix shows whether the same entry is read."""
        corr = pd.DataFrame(
            [[1.0, 0.1, 0.1], [0.99, 1.0, 0.1], [0.99, 0.99, 1.0]],
            index=list("xyz"),
            columns=list("xyz"),
        )
        assert _correlation_clusters(corr, 0.9) == [["x"], ["y"], ["z"]]
        assert _reference_correlation_clusters(corr, 0.9) == [["x"], ["y"], ["z"]]
        flipped = corr.T
        assert _correlation_clusters(flipped, 0.9) == [["x", "y", "z"]]
        assert _reference_correlation_clusters(flipped, 0.9) == [["x", "y", "z"]]

    def test_rows_are_read_by_label_not_position(self):
        corr = _block_correlation(10, seed=3)
        shuffled = corr.loc[list(reversed(corr.index))]
        assert _correlation_clusters(shuffled, 0.9) == (
            _reference_correlation_clusters(shuffled, 0.9)
        )
        assert _correlation_clusters(shuffled, 0.9) == _correlation_clusters(corr, 0.9)

    def test_empty_and_one_feature(self):
        empty = pd.DataFrame()
        assert _correlation_clusters(empty, 0.9) == []
        assert _reference_correlation_clusters(empty, 0.9) == []
        one = pd.DataFrame([[1.0]], index=["only"], columns=["only"])
        assert _correlation_clusters(one, 0.9) == [["only"]]
        assert _reference_correlation_clusters(one, 0.9) == [["only"]]


# ── Nested matrices ──────────────────────────────────────────────────────


class TestNestedMatchesTheReference:
    @pytest.mark.parametrize("n_features", [1, 2, 9, 40])
    def test_correlation_matrices(self, n_features):
        corr = _block_correlation(max(n_features, 1), seed=n_features)
        _same_nested(_frame_to_nested(corr), _reference_frame_to_nested(corr))

    def test_non_finite_values_and_signed_zero(self):
        frame = pd.DataFrame(
            [[1.0, np.nan, np.inf], [-np.inf, -0.0, 0.25], [1e-300, -1.5, 2.0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        new, old = _frame_to_nested(frame), _reference_frame_to_nested(frame)
        _same_nested(new, old)
        assert math.copysign(1.0, new["b"]["b"]) == -1.0

    def test_non_string_labels_and_other_dtypes(self):
        """Labels are stringified the same way, and a frame that is not
        float goes through `_safe` cell by cell as before."""
        floats = pd.DataFrame(
            [[0.5, 1.0], [2.0, 3.5]], index=[1, 2.5], columns=[0, "x"]
        )
        _same_nested(_frame_to_nested(floats), _reference_frame_to_nested(floats))
        mixed = pd.DataFrame({"i": [1, 2], "o": ["3.5", None]}, index=["r", "s"])
        _same_nested(_frame_to_nested(mixed), _reference_frame_to_nested(mixed))
        ints = pd.DataFrame([[1, 2], [3, 4]], index=["r", "s"], columns=["u", "v"])
        _same_nested(_frame_to_nested(ints), _reference_frame_to_nested(ints))

    def test_empty(self):
        assert _frame_to_nested(pd.DataFrame()) == {}
        assert _reference_frame_to_nested(pd.DataFrame()) == {}


def test_redundancy_report_end_to_end():
    """The report's clusters and both nested matrices on one panel, against
    the reference applied to the same correlation matrices."""
    rng = np.random.default_rng(5)
    base = rng.normal(size=800)
    panel = pd.DataFrame(
        {
            "a": base,
            "b": base + 0.05 * rng.normal(size=800),
            "c": rng.normal(size=800),
            "d": np.where(rng.random(800) < 0.1, np.nan, rng.normal(size=800)),
        }
    )
    features = ["a", "b", "c", "d"]
    report = redundancy_report(panel, features)
    frame = panel[features].dropna()
    pearson, spearman = frame.corr(), frame.corr(method="spearman")
    assert report["clusters"] == _reference_correlation_clusters(pearson, 0.9)
    assert report["clusters"] == [["a", "b"], ["c"], ["d"]]
    _same_nested(report["correlation"], _reference_frame_to_nested(pearson))
    _same_nested(report["spearman_correlation"], _reference_frame_to_nested(spearman))
