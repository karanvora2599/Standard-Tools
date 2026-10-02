"""
`detect_stale_prices` finds runs with array passes and reports exactly the
runs its per-bar loop reported.

The loop read two values through `iloc` for every bar of the frame. Runs
are now found from where each bar's run started, carried forward with
`np.maximum.accumulate` (see the CHANGELOG entry of 2026-10-01), and only
the qualifying runs touch pandas. The reference below is the loop, kept
verbatim, and the result is required to be the same list: the same runs in
the same order, the same values, the same types.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.data.quality import detect_stale_prices


def _reference_detect_stale_prices(
    df: pd.DataFrame, n: int = 3
) -> List[Dict[str, Any]]:
    """`detect_stale_prices` as it was, verbatim but for this docstring."""
    close = df["Close"]
    if len(close) == 0:
        return []

    runs: List[Dict[str, Any]] = []
    run_start = 0
    for i in range(1, len(close) + 1):
        changed = i == len(close) or close.iloc[i] != close.iloc[run_start]
        if changed:
            run_length = i - run_start
            if run_length >= n:
                runs.append(
                    {
                        "start": str(close.index[run_start].date()),
                        "end": str(close.index[i - 1].date()),
                        "price": float(close.iloc[run_start]),
                        "run_length": run_length,
                    }
                )
            run_start = i
    return runs


def _identical(a, b) -> bool:
    """Equal and of the same type, all the way down; floats to the bit."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_identical(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_identical(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return a == b


def _frame(close, start="2019-01-02", dtype=None):
    index = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"Close": pd.Series(close, index=index, dtype=dtype)})


def _assert_same(df, n=3):
    expected = _reference_detect_stale_prices(df, n)
    actual = detect_stale_prices(df, n)
    assert _identical(actual, expected), (n, actual, expected)
    return actual


def _prices_with_runs(n_bars, seed, n_runs=8):
    rng = np.random.default_rng(seed)
    close = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.015, n_bars))), 2)
    for _ in range(n_runs):
        start = int(rng.integers(1, max(2, n_bars - 10)))
        close[start : start + int(rng.integers(2, 9))] = close[start - 1]
    return close


class TestEveryRunIsTheLoops:
    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("n", [3, 2, 5])
    def test_seeded_frames(self, seed, n):
        rng = np.random.default_rng(seed)
        _assert_same(_frame(_prices_with_runs(int(rng.integers(20, 2500)), seed)), n)

    @pytest.mark.parametrize("n", [-1, 0, 1, 2, 30, 31])
    def test_the_threshold_at_and_past_its_bounds(self, n):
        """At 1 or below every bar is in a reported run, NaN included."""
        close = _prices_with_runs(30, 3, n_runs=3)
        close[[4, 5, 6]] = np.nan
        _assert_same(_frame(close), n)

    @pytest.mark.parametrize(
        "close",
        [
            [],
            [101.5],
            [101.5, 101.5],
            [np.nan] * 6,
            [100.0] * 12,
            [100.0, 100.0, 100.0, 101.0, 102.0],
            [99.0, 100.0, 101.0, 101.0, 101.0],
            [100.0, np.nan, np.nan, np.nan, 100.0, 100.0, 100.0],
            [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf],
            [-0.0, 0.0, 0.0, 1.0, 0.0, -0.0, -0.0],
            [5.0, 5.0, 5.0, np.nan, 5.0, 5.0, 5.0],
        ],
        ids=[
            "empty",
            "one_row",
            "two_rows",
            "all_nan",
            "constant",
            "run_at_start",
            "run_at_end",
            "nan_between_equal_prices",
            "infinities",
            "signed_zeros",
            "nan_splits_a_run",
        ],
    )
    @pytest.mark.parametrize("n", [1, 3])
    def test_edge_frames(self, close, n):
        """A NaN equals nothing, itself included, so it is a run of one and
        splits a run of equal prices either side of it. -0.0 equals 0.0,
        and a run reports its FIRST bar's value, sign and all."""
        _assert_same(_frame(np.asarray(close, dtype=float)), n)

    @pytest.mark.parametrize("dtype", ["int64", "int32", "float32", "bool", "uint8"])
    def test_numpy_dtypes_other_than_float64(self, dtype):
        close = np.array([1, 1, 1, 0, 0, 2, 2, 2, 2, 1], dtype=float).astype(dtype)
        _assert_same(_frame(close), 2)

    @pytest.mark.parametrize("dtype", ["object", "Float64", "Int64"])
    def test_columns_numpy_cannot_compare_keep_the_loop(self, dtype):
        """Object and nullable columns go through the loop as before."""
        close = [3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0]
        _assert_same(_frame(close, dtype=dtype), 3)

    def test_an_intraday_index(self):
        """Two runs on one calendar day report the same dates, as before."""
        index = pd.date_range("2024-03-01 09:30", periods=12, freq="30min")
        close = [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0, 5.0, 5.0]
        _assert_same(pd.DataFrame({"Close": close}, index=index), 3)

    def test_a_planted_run_is_reported_where_it_was_planted(self):
        close = np.linspace(100.0, 120.0, 40)
        close[10:16] = close[9]
        runs = _assert_same(_frame(close), 3)
        assert len(runs) == 1
        assert runs[0]["run_length"] == 7
        assert runs[0]["price"] == close[9]
        assert runs[0]["start"] == "2019-01-15"

    def test_a_price_that_always_moves_has_no_runs(self):
        """The null case."""
        assert _assert_same(_frame(np.linspace(100.0, 120.0, 500)), 2) == []
