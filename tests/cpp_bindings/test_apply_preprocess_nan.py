"""
apply_preprocess_stats against an exact numpy statement of what it computes.

The kernel's per-value NaN test changed from std::isnan, which MSVC compiles
to a call into the C runtime for every value, to an inline compare (the
CHANGELOG entry of 2026-10-02). The answer may not move by a bit, so the
kernel is held here to a numpy mirror of its own arithmetic -- the same
selects in the same order, so equality is exact, not approximate -- and a
NaN is held to the stronger promise the transform makes: the value that
arrives is the value that leaves, sign and payload included.

The shapes straddle the work threshold, so both the serial and the parallel
path answer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="_sqt_core not built")

NAN_BITS = np.array(
    [
        0x7FF8000000000000,  # the quiet NaN numpy makes
        0xFFF8000000000000,  # negative
        0x7FF8000000000123,  # with a payload
        0xFFFFFFFFFFFFFFFF,  # every bit set
        0x7FF0000000000001,  # signalling
    ],
    dtype=np.uint64,
)


def _mirror(values, lo, hi, mean, std):
    """std::min(std::max(v, lo), hi), then (clipped - mean) / std; NaN kept."""
    lifted = np.where(values < lo, lo, values)
    clipped = np.where(hi < lifted, hi, lifted)
    z = (clipped - mean) / std
    return np.where(np.isnan(values), values, z)


def _apply(values, stats):
    return _cpp.apply_preprocess_stats(
        values, stats["lo"], stats["hi"], stats["mean"], stats["std"]
    )


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


class TestEveryValueIsWhatTheMirrorSays:
    @pytest.mark.parametrize(
        "rows, cols",
        [(7, 3), (2_000, 10), (60_000, 10), (20_001, 7)],
        ids=["tiny", "serial-sized", "parallel-sized", "odd-rows"],
    )
    @pytest.mark.parametrize("nan_fraction", [0.0, 0.05, 0.5])
    def test_random_panels(self, rows, cols, nan_fraction):
        rng = np.random.default_rng(rows * 31 + cols + int(nan_fraction * 100))
        values = rng.standard_t(3, (rows, cols))
        values[rng.random(values.shape) < nan_fraction] = np.nan
        stats = _cpp.fit_preprocess_stats(values, 0.05, 0.95)
        got = _apply(values, stats)
        want = _mirror(values, stats["lo"], stats["hi"], stats["mean"], stats["std"])
        assert np.array_equal(_bits(got), _bits(want))

    def test_values_on_and_beyond_the_bounds(self):
        lo, hi, mean, std = (np.array([x]) for x in (-1.0, 2.0, 0.25, 1.5))
        column = np.array(
            [-1.0, 2.0, -0.0, 0.0, np.inf, -np.inf, 1e308, -1e308, 5e-324, 0.25]
        ).reshape(-1, 1)
        got = _cpp.apply_preprocess_stats(column, lo, hi, mean, std)
        want = _mirror(column, lo, hi, mean, std)
        assert np.array_equal(_bits(got), _bits(want))
        assert got[4, 0] == (2.0 - 0.25) / 1.5  # +inf pinned to hi
        assert got[5, 0] == (-1.0 - 0.25) / 1.5  # -inf pinned to lo
        assert got[9, 0] == 0.0  # the mean itself

    def test_an_all_nan_columns_stats_leave_its_values_nan(self):
        """fit_preprocess_stats reports NaN bounds and mean, std 1.0, for a
        column with no values; a value that later arrives in it comes out
        NaN, through the arithmetic rather than the NaN test."""
        nan = np.array([np.nan])
        column = np.array([[1.0], [np.nan], [-3.0]])
        got = _cpp.apply_preprocess_stats(column, nan, nan, nan, np.array([1.0]))
        assert np.isnan(got).all()


class TestANanLeavesAsItArrived:
    def test_every_kind_of_nan_keeps_its_bits(self):
        rng = np.random.default_rng(5)
        values = rng.normal(0.0, 1.0, (40, len(NAN_BITS)))
        stats = _cpp.fit_preprocess_stats(values, 0.01, 0.99)
        rows = rng.choice(40, size=len(NAN_BITS), replace=False)
        planted = values.copy()
        bits = planted.view(np.uint64)
        for col, (row, pattern) in enumerate(zip(rows, NAN_BITS)):
            bits[row, col] = pattern
        got = _apply(planted, stats)
        for col, (row, pattern) in enumerate(zip(rows, NAN_BITS)):
            assert _bits(got)[row, col] == pattern, hex(int(pattern))
        assert np.array_equal(np.isnan(got), np.isnan(planted))

    def test_a_panel_of_nothing_but_nan_comes_back_unchanged(self):
        values = np.tile(NAN_BITS.view(np.float64), (64, 1))
        stats = {
            "lo": np.zeros(len(NAN_BITS)),
            "hi": np.ones(len(NAN_BITS)),
            "mean": np.full(len(NAN_BITS), 0.5),
            "std": np.full(len(NAN_BITS), 2.0),
        }
        assert np.array_equal(_bits(_apply(values, stats)), _bits(values))

    def test_a_panel_without_nan_has_none_after(self):
        values = np.random.default_rng(9).normal(0.0, 1.0, (500, 4))
        stats = _cpp.fit_preprocess_stats(values, 0.01, 0.99)
        assert not np.isnan(_apply(values, stats)).any()
