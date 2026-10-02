"""
`detect_change_points` prices every candidate split in one array expression
and chooses the split it chose before.

`_best_split` priced a candidate with a closure called twice per position,
the left side and the right, each a handful of scalar operations on the
prefix sums. Those operations now run over all positions at once (see the
CHANGELOG entry of 2026-10-01). They are the same operations in the same
order, so every cost is the same double -- the reference below is the
closure, kept verbatim, and the split and its gain are required to match
it exactly, not to a tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis import structure
from standard_quant_tools.analysis.structure import detect_change_points


def _reference_best_split(segment: np.ndarray, min_segment: int, penalty: float):
    """`_best_split` as it was, verbatim but for this docstring."""
    n = len(segment)
    if n < 2 * min_segment:
        return None
    # Centred first. The prefix-sum cost is a difference of two large sums
    # (sum of squares minus square of sum over count), and on a series far
    # from zero -- prices, or a segment sitting at a level -- that
    # difference is catastrophic cancellation: the costs of every split
    # were rounding noise and a "gain" appeared from nothing. Centring
    # changes no cost and keeps the sums at the scale of the variation.
    segment = segment - segment.mean()
    total = float((segment**2).sum())

    # Prefix sums make every candidate split O(1) rather than O(n).
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    cumulative_sq = np.concatenate([[0.0], np.cumsum(segment**2)])

    def rss(lo: int, hi: int) -> float:
        count = hi - lo
        if count <= 0:
            return 0.0
        total_ = cumulative[hi] - cumulative[lo]
        total_sq = cumulative_sq[hi] - cumulative_sq[lo]
        return float(total_sq - total_ * total_ / count)

    positions = np.arange(min_segment, n - min_segment + 1)
    if positions.size == 0:
        return None
    costs = np.array([rss(0, p) + rss(p, n) for p in positions])
    best = int(np.argmin(costs))
    gain = total - costs[best]
    # A gain that is a rounding-sized fraction of the segment's own
    # variation is not a split, whatever the penalty says.
    if gain <= penalty or gain <= 1e-12 * total:
        return None
    return int(positions[best]), float(gain)


def _assert_same_split(segment, min_segment, penalty):
    expected = _reference_best_split(segment, min_segment, penalty)
    actual = structure._best_split(segment, min_segment, penalty)
    assert actual == expected, (len(segment), min_segment, penalty)
    if expected is not None:
        assert type(actual[0]) is int and type(actual[1]) is float
        # `==` on the gain is the bit-level check here: a cost that moved
        # in its last bit moves the gain with it.
        assert math.copysign(1.0, actual[1]) == math.copysign(1.0, expected[1])
    return actual


def _segment(kind, n, rng):
    if kind == "noise":
        return rng.normal(0.0, 1.0, n)
    if kind == "step":
        return np.r_[rng.normal(0.0, 1.0, n // 2), rng.normal(2.5, 1.0, n - n // 2)]
    if kind == "ties":
        return np.round(rng.normal(100.0, 0.5, n), 1)
    if kind == "far_from_zero":
        return rng.normal(0.0, 1e-3, n) + 1e4
    if kind == "returns":
        return rng.normal(0.0004, 0.012, n)
    raise AssertionError(kind)


class TestEveryCandidateCostsWhatItDid:
    @pytest.mark.parametrize(
        "kind", ["noise", "step", "ties", "far_from_zero", "returns"]
    )
    @pytest.mark.parametrize("seed", range(8))
    def test_random_segments(self, kind, seed):
        """With the penalty at minus infinity a split is always returned, so
        the comparison reaches the argmin and the gain on every case."""
        rng = np.random.default_rng(seed)
        n = int(rng.integers(4, 900))
        min_segment = int(rng.integers(1, n // 2 + 1))
        segment = _segment(kind, n, rng)
        for penalty in (-np.inf, 0.0, 1.0, 1e12):
            _assert_same_split(segment, min_segment, penalty)

    @pytest.mark.parametrize("min_segment", [0, 1, 2, 10, 19, 20])
    def test_min_segment_at_its_bounds(self, min_segment):
        """0 puts the empty segment at both ends among the candidates --
        the closure's `count <= 0` branch -- and n // 2 leaves exactly one
        candidate."""
        segment = _segment("step", 40, np.random.default_rng(3))
        _assert_same_split(segment, min_segment, -np.inf)

    def test_too_short_a_segment_has_no_candidate(self):
        segment = _segment("noise", 9, np.random.default_rng(0))
        assert _assert_same_split(segment, 5, -np.inf) is None

    def test_a_constant_segment_has_no_split(self):
        """The null case: every cost is the total, so the gain is zero."""
        assert _assert_same_split(np.full(50, 1.23), 5, -np.inf) is None

    def test_a_planted_step_is_found_where_it_was_planted(self):
        segment = np.r_[np.zeros(60), np.ones(40)] + np.tile([0.01, -0.01], 50)
        position, gain = _assert_same_split(segment, 10, 0.0)
        assert position == 60
        assert gain == pytest.approx(60 * 40 / 100, rel=1e-3)


class TestDetectChangePointsIsUnchanged:
    def _both(self, series, monkeypatch, **kwargs):
        actual = detect_change_points(series, **kwargs)
        monkeypatch.setattr(structure, "_best_split", _reference_best_split)
        expected = detect_change_points(series, **kwargs)
        monkeypatch.undo()
        assert actual == expected
        return actual

    @pytest.mark.parametrize("seed", range(6))
    def test_whole_results_match(self, seed, monkeypatch):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(80, 2500))
        values = rng.normal(0.0004, 0.012, n)
        values[n // 3 :] += 0.004 * (seed % 2)
        values[2 * n // 3 :] -= 0.006 * (seed % 3 == 0)
        series = pd.Series(values, index=pd.bdate_range("2018-06-01", periods=n))
        self._both(series, monkeypatch)
        self._both(series, monkeypatch, max_breaks=6, min_segment=5)

    def test_a_planted_break_and_a_null_series(self, monkeypatch):
        rng = np.random.default_rng(11)
        index = pd.bdate_range("2020-01-01", periods=600)
        shifted = pd.Series(
            np.r_[rng.normal(0.0, 0.01, 300), rng.normal(0.03, 0.01, 300)], index=index
        )
        found = self._both(shifted, monkeypatch, max_breaks=1)
        assert [b["index"] for b in found["breaks"]] == [300]

        quiet = pd.Series(rng.normal(0.0, 0.01, 600), index=index)
        assert self._both(quiet, monkeypatch)["n_breaks"] == 0
