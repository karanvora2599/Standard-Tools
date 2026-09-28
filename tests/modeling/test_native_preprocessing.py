"""
The native preprocessing kernel must be invisible except in the clock.

`fit_preprocessing` and `apply_preprocessing` were measured at 47-56% of a
walk-forward run, so they got a C++ kernel. That kernel replaces a specific
pandas expression, and the bar it has to clear is agreement with pandas at
machine epsilon — including the parts that are pandas CONVENTIONS rather
than mathematical necessity: linearly interpolated quantiles, a ddof=1
standard deviation, NaN skipped by the moments but preserved by the
transform, and infinities not treated as missing.

The Python implementation stays as the reference and the oracle. These tests
compare the two directly by toggling `transforms.HAS_CPP`, so they are
meaningful whether or not the extension is present: with it absent both
sides are the same code and the tests trivially pass, which is the correct
outcome for a machine with no compiler.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.modeling.features import transforms
from standard_quant_tools.modeling.features.transforms import (
    apply_preprocessing,
    fit_preprocessing,
)

pytestmark = pytest.mark.skipif(
    not transforms.HAS_CPP, reason="native extension not built"
)

# Machine epsilon on values of order 1. The kernel sums pairwise precisely so
# it can meet this against numpy; a sequential accumulator missed it by two
# orders of magnitude on return-scale data, which is what forced the issue.
TOL = 1e-14


@pytest.fixture
def python_only(monkeypatch):
    """Force the pure-Python path for the duration of one test."""
    monkeypatch.setattr(transforms, "HAS_CPP", False)


def _both_paths(frame, monkeypatch):
    native_stats = fit_preprocessing(frame)
    native_applied = apply_preprocessing(frame, native_stats)
    monkeypatch.setattr(transforms, "HAS_CPP", False)
    python_stats = fit_preprocessing(frame)
    python_applied = apply_preprocessing(frame, python_stats)
    monkeypatch.undo()
    return (native_stats, native_applied), (python_stats, python_applied)


def _assert_agree(frame, monkeypatch, tol=TOL):
    (n_stats, n_applied), (p_stats, p_applied) = _both_paths(frame, monkeypatch)

    assert list(n_stats) == list(p_stats), "column order differs"
    for column in p_stats:
        for key in ("lo", "hi", "mean", "std"):
            native, python = n_stats[column][key], p_stats[column][key]
            if np.isnan(python):
                assert np.isnan(native), f"{column}.{key}: expected NaN"
                continue
            assert (
                abs(native - python) <= tol
            ), f"{column}.{key}: native {native!r} vs python {python!r}"

    native_values = n_applied.to_numpy()
    python_values = p_applied.to_numpy()
    np.testing.assert_array_equal(
        np.isnan(native_values), np.isnan(python_values), "NaN masks differ"
    )
    finite = ~np.isnan(python_values)
    if finite.any():
        np.testing.assert_allclose(
            native_values[finite], python_values[finite], rtol=0, atol=tol
        )
    assert list(n_applied.columns) == list(p_applied.columns)
    assert n_applied.index.equals(p_applied.index)


class TestNativeMatchesPython:
    @pytest.mark.parametrize("shape", [(100, 3), (5_000, 6), (20_000, 8)])
    def test_ordinary_panels(self, monkeypatch, shape):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            rng.normal(0, 1, shape), columns=[f"f{i}" for i in range(shape[1])]
        )
        _assert_agree(frame, monkeypatch)

    def test_return_scale_data(self, monkeypatch):
        """
        The case that caught a real defect. Daily returns are ~1e-2, and a
        sequential accumulator disagreed with numpy's pairwise summation in
        the 12th significant digit — which propagated to 3.8e-14 on the
        standardized output. The kernel sums pairwise because of this test.
        """
        rng = np.random.default_rng(1)
        frame = pd.DataFrame(
            rng.normal(0.0004, 0.012, (50_000, 6)),
            columns=[f"f{i}" for i in range(6)],
        )
        _assert_agree(frame, monkeypatch, tol=1e-15)

    def test_heavy_ties(self, monkeypatch):
        """A discretized feature puts many equal values on the quantile
        boundary, where an interpolation rule is easiest to get wrong."""
        rng = np.random.default_rng(2)
        frame = pd.DataFrame(
            np.round(rng.normal(0, 1, (20_000, 4)), 1),
            columns=[f"f{i}" for i in range(4)],
        )
        _assert_agree(frame, monkeypatch)

    def test_fat_tailed(self, monkeypatch):
        """Where the winsorize bound actually bites, so lo/hi matter."""
        rng = np.random.default_rng(3)
        frame = pd.DataFrame(
            rng.standard_t(2.0, (30_000, 5)), columns=[f"f{i}" for i in range(5)]
        )
        _assert_agree(frame, monkeypatch)

    def test_constant_and_two_valued_columns(self, monkeypatch):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame(
            {
                "constant": np.full(5_000, 3.5),
                "two_valued": np.where(rng.random(5_000) < 0.5, 1.0, 2.0),
                "normal": rng.normal(0, 1, 5_000),
            }
        )
        _assert_agree(frame, monkeypatch)
        # A column with no dispersion standardizes to exactly zero rather
        # than dividing by zero.
        stats = fit_preprocessing(frame)
        assert stats["constant"]["std"] == 1.0
        assert (apply_preprocessing(frame, stats)["constant"] == 0.0).all()

    def test_nan_present(self, monkeypatch):
        rng = np.random.default_rng(5)
        values = rng.normal(0, 1, (10_000, 4))
        values[rng.random(values.shape) < 0.2] = np.nan
        _assert_agree(
            pd.DataFrame(values, columns=[f"f{i}" for i in range(4)]), monkeypatch
        )

    def test_all_nan_column(self, monkeypatch):
        rng = np.random.default_rng(6)
        values = rng.normal(0, 1, (2_000, 3))
        values[:, 1] = np.nan
        _assert_agree(pd.DataFrame(values, columns=["a", "all_nan", "c"]), monkeypatch)

    def test_infinities_are_not_missing(self, monkeypatch):
        """pandas treats only NaN as missing, so an inf is a real order
        statistic and must participate in the quantile."""
        rng = np.random.default_rng(7)
        values = rng.normal(0, 1, (5_000, 3))
        values[7, 0] = np.inf
        values[11, 1] = -np.inf
        _assert_agree(pd.DataFrame(values, columns=["a", "b", "c"]), monkeypatch)

    @pytest.mark.parametrize("n_rows", [1, 2, 3])
    def test_tiny_panels(self, monkeypatch, n_rows):
        rng = np.random.default_rng(8)
        frame = pd.DataFrame(rng.normal(0, 1, (n_rows, 2)), columns=["a", "b"])
        _assert_agree(frame, monkeypatch)

    def test_single_column(self, monkeypatch):
        rng = np.random.default_rng(9)
        _assert_agree(pd.DataFrame({"only": rng.normal(0, 1, 1_000)}), monkeypatch)


class TestNativePathBoundaries:
    def test_nan_is_preserved_not_clipped_to_a_bound(self):
        """
        Series.clip leaves a missing value missing. Pinning it to a
        winsorize bound would fabricate an observation, and alignment would
        then keep a row it should have dropped.
        """
        frame = pd.DataFrame({"f": [1.0, np.nan, 3.0, 4.0, np.nan]})
        stats = fit_preprocessing(frame)
        out = apply_preprocessing(frame, stats)
        assert out["f"].isna().tolist() == [False, True, False, False, True]

    def test_extra_columns_fall_back_to_the_per_column_path(self):
        """
        The fused kernel transforms the whole matrix, so it only applies
        when the frame is exactly the fitted columns. A frame carrying an
        extra column must still work, with the extra passed through
        untouched — that is a real calling convention, not a corner case.
        """
        rng = np.random.default_rng(10)
        fitted = pd.DataFrame(rng.normal(0, 1, (500, 2)), columns=["a", "b"])
        stats = fit_preprocessing(fitted)
        wider = fitted.copy()
        wider["untouched"] = np.arange(500, dtype=float)
        out = apply_preprocessing(wider, stats)
        np.testing.assert_array_equal(
            out["untouched"].to_numpy(), np.arange(500, dtype=float)
        )
        assert abs(float(out["a"].mean())) < 0.5

    def test_non_numeric_column_does_not_reach_the_kernel(self):
        """A non-float column cannot be handed to the kernel; the Python
        path must take over rather than the conversion raising."""
        frame = pd.DataFrame(
            {"a": np.arange(10, dtype=float), "label": list("abcdefghij")}
        )
        stats = fit_preprocessing(frame[["a"]])
        out = apply_preprocessing(frame, stats)
        assert out["label"].tolist() == list("abcdefghij")

    def test_index_and_columns_survive(self):
        rng = np.random.default_rng(11)
        index = pd.date_range("2020-01-01", periods=300, freq="B")
        frame = pd.DataFrame(
            rng.normal(0, 1, (300, 3)), columns=["x", "y", "z"], index=index
        )
        out = apply_preprocessing(frame, fit_preprocessing(frame))
        assert out.index.equals(index)
        assert list(out.columns) == ["x", "y", "z"]

    def test_python_path_still_reachable(self, python_only):
        """With the extension disabled the module must still work — that is
        the whole premise of an optional fast path."""
        rng = np.random.default_rng(12)
        frame = pd.DataFrame(rng.normal(0, 1, (200, 2)), columns=["a", "b"])
        stats = fit_preprocessing(frame)
        assert set(stats) == {"a", "b"}
        out = apply_preprocessing(frame, stats)
        assert out.shape == frame.shape


class TestQuantileEndpoints:
    """
    The kernel's q=1 bound is the column maximum, and q=0 the minimum.

    The winsorizing callers pass 0.01/0.99, but the binding accepts any q in
    [0, 1]. q=1 used to be read from a buffer slot the previous quantile's
    partition had left there -- right only on columns of 32 values or fewer,
    where the standard library's nth_element happens to insertion-sort --
    and the clipped moments were wrong with it. pandas' quantile, clip, mean
    and std are the oracle.
    """

    @pytest.mark.parametrize("n", [33, 50, 200, 1000])
    @pytest.mark.parametrize("q_low,q_high", [(0.0, 1.0), (0.01, 1.0), (0.0, 0.99)])
    def test_matches_pandas(self, n, q_low, q_high):
        rng = np.random.default_rng(n)
        values = rng.normal(0, 1, (n, 3))
        native = transforms._cpp_core.fit_preprocess_stats(
            np.ascontiguousarray(values), q_low, q_high
        )
        for c in range(values.shape[1]):
            column = pd.Series(values[:, c])
            lo, hi = column.quantile(q_low), column.quantile(q_high)
            clipped = column.clip(lower=lo, upper=hi)
            assert abs(native["lo"][c] - lo) <= TOL
            assert abs(native["hi"][c] - hi) <= TOL
            assert abs(native["mean"][c] - clipped.mean()) <= TOL
            assert abs(native["std"][c] - clipped.std()) <= TOL

    @pytest.mark.parametrize("n", [5, 17, 32])
    def test_short_columns_still_agree(self, n):
        """The null case: the sizes that were right by coincidence stay right."""
        values = np.random.default_rng(n).normal(0, 1, (n, 1))
        native = transforms._cpp_core.fit_preprocess_stats(values, 0.0, 1.0)
        assert native["lo"][0] == values.min()
        assert native["hi"][0] == values.max()

    def test_an_infinite_maximum_is_the_maximum(self):
        """Infinities are order statistics here, not missing values, so the
        q=1 bound of a column holding +inf is +inf. (pandas 2.x answers NaN
        for this one case, because its interpolation forms inf - inf.)"""
        values = np.random.default_rng(3).normal(0, 1, (100, 1))
        values[57, 0] = np.inf
        native = transforms._cpp_core.fit_preprocess_stats(values, 0.01, 1.0)
        assert native["hi"][0] == np.inf
        assert native["lo"][0] == pd.Series(values[:, 0]).quantile(0.01)


class TestStatsThatNoFitProduces:
    """
    `apply_preprocessing` reads its statistics back from a persisted
    preprocessing_stats.json, so they are not necessarily a fit's output.
    std = 0 answered +/-inf and NaN, a negative std flipped every sign, and
    lo > hi was answered differently by the two backends (pandas' clip swaps
    the bounds; the kernel pinned every value to hi). Both paths now refuse
    them with the same error.
    """

    FRAME = pd.DataFrame({"a": [-1.0, 0.0, 1.0], "b": [2.0, 3.0, 4.0]})

    @staticmethod
    def _stats(**column_a):
        good = {"lo": -1.0, "hi": 1.0, "mean": 0.0, "std": 1.0}
        return {
            "a": {**good, **column_a},
            "b": {"lo": 2.0, "hi": 4.0, "mean": 3.0, "std": 1.0},
        }

    @pytest.mark.parametrize(
        "bad",
        [
            dict(std=0.0),
            dict(std=-1.0),
            dict(std=np.nan),
            dict(std=np.inf),
            dict(lo=2.0, hi=1.0),
        ],
    )
    @pytest.mark.parametrize("native", [True, False])
    def test_refused_on_both_paths(self, bad, native, monkeypatch):
        from standard_quant_tools.error import ValidationError

        monkeypatch.setattr(transforms, "HAS_CPP", native)
        with pytest.raises(ValidationError, match="apply_preprocessing"):
            apply_preprocessing(self.FRAME, self._stats(**bad))

    @pytest.mark.parametrize(
        "bad,match",
        [
            (dict(std=0.0), "std"),
            (dict(std=-1.0), "std"),
            (dict(std=np.nan), "std"),
            (dict(std=np.inf), "std"),
            (dict(lo=2.0, hi=1.0), "lo"),
        ],
    )
    def test_refused_by_the_binding(self, bad, match):
        stats = self._stats(**bad)["a"]
        with pytest.raises(ValueError, match=match):
            transforms._cpp_core.apply_preprocess_stats(
                np.array([[0.5]]),
                np.array([stats["lo"]]),
                np.array([stats["hi"]]),
                np.array([stats["mean"]]),
                np.array([stats["std"]]),
            )

    @pytest.mark.parametrize("native", [True, False])
    def test_an_all_nan_columns_stats_still_apply(self, native, monkeypatch):
        """The null case: NaN lo/hi/mean with std 1.0 is what a fit reports
        for a column with no values, and it must keep applying."""
        frame = pd.DataFrame({"a": [np.nan, np.nan, np.nan], "b": [1.0, 2.0, 4.0]})
        monkeypatch.setattr(transforms, "HAS_CPP", native)
        stats = fit_preprocessing(frame)
        assert np.isnan(stats["a"]["lo"]) and stats["a"]["std"] == 1.0
        out = apply_preprocessing(frame, stats)
        assert out["a"].isna().all()
        assert out["b"].notna().all()
