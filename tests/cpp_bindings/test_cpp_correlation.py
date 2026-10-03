"""
`pearson_correlation` is `DataFrame.corr()`, bit for bit.

`hierarchical_risk_parity` spent most of its time in `frame.corr()` --
pandas' `nancorr`, a Welford recursion per pair of columns -- and clusters
on the result with single linkage, whose ties fall on the last bits. So the
kernel replacing it is held to pandas' own output, not to a better formula,
and every comparison here is on the bits (any NaN matching any NaN), across
the cases that exercise each rule of nancorr:

  - complete panels of every shape, in C order, Fortran order and strided
    views, where the kernel shares each column's recursion across its pairs;
  - missing values (NaN, +inf and -inf alike) in every pattern, where it
    runs pandas' loop over each pair's own rows;
  - the NaN rules: a constant column, one row, no rows, an all-missing
    column, and min_periods on either side of a pair's row count;
  - the clip pandas 3 added, which the caller applies exactly when the
    installed pandas does.

Both pandas majors are covered by running this file under each interpreter.
See the CHANGELOG entry of 2026-10-02.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.portfolio import construction

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = hasattr(_cpp, "pearson_correlation")
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(
    not HAS_CPP, reason="_sqt_core.pearson_correlation not built"
)

#: What the installed pandas does to a coefficient outside [-1, 1]: pandas 3
#: clips it, pandas 1.5-2.3 return it as computed.
PANDAS_CLIPS = int(pd.__version__.split(".")[0]) >= 3


@pytest.fixture(autouse=True)
def _kernel_reproduces_this_pandas():
    """A pandas built to fuse a*b+c into one rounding computes other bits
    than the kernel's unfused arithmetic; the library then uses
    frame.corr() itself, and a bit-for-bit comparison has nothing to hold
    the kernel to."""
    if HAS_CPP and construction._native_correlation_mode() == "pandas":
        pytest.skip(f"the kernel does not reproduce this pandas ({pd.__version__})")


def _pandas(values, min_periods: int = 1) -> np.ndarray:
    return pd.DataFrame(values).corr(min_periods=min_periods).to_numpy()


def _kernel(values, min_periods: int = 1) -> np.ndarray:
    out = np.asarray(_cpp.pearson_correlation(values, min_periods))
    if PANDAS_CLIPS:
        np.clip(out, -1.0, 1.0, out=out)
    return out


def _assert_same_bits(got: np.ndarray, want: np.ndarray) -> None:
    assert got.dtype == np.float64 and got.shape == want.shape, (got.shape, want.shape)
    same = (got.view(np.uint64) == want.view(np.uint64)) | (
        np.isnan(got) & np.isnan(want)
    )
    assert same.all(), (
        f"{int((~same).sum())} of {same.size} cells differ, first at "
        f"{np.argwhere(~same)[0].tolist()}: {got[~same][:3]} vs {want[~same][:3]}"
    )


def _check(values, min_periods: int = 1) -> np.ndarray:
    values = np.asarray(values)
    got = _kernel(values, min_periods)
    _assert_same_bits(got, _pandas(values, min_periods))
    return got


def _panel(rows: int, cols: int, seed: int, scale: float = 0.012) -> np.ndarray:
    return np.random.default_rng([rows, cols, seed]).normal(0.0, scale, (rows, cols))


class TestCompletePanels:
    @pytest.mark.parametrize(
        "shape",
        [(2, 1), (2, 2), (3, 3), (5, 40), (17, 9), (60, 8), (500, 40), (731, 97)],
    )
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_seeded_noise(self, shape, order):
        _check(np.asarray(_panel(*shape, seed=1), order=order))

    def test_the_size_hierarchical_risk_parity_was_measured_at(self):
        """2,106 daily returns of 235 names, Fortran-ordered as
        `DataFrame.to_numpy()` hands it over."""
        values = np.asfortranarray(_panel(2_106, 235, seed=2))
        _check(values)

    @pytest.mark.parametrize(
        "name",
        ["offset", "tiny", "mixed scales", "integer ties", "trend"],
    )
    def test_magnitudes_where_the_order_of_operations_shows(self, name):
        rng = np.random.default_rng(7)
        values = {
            "offset": rng.normal(1e8, 1e3, (400, 12)),
            "tiny": rng.normal(0.0, 1e-150, (400, 12)),
            "mixed scales": rng.normal(0, 1, (300, 20))
            * 10.0 ** rng.integers(-8, 8, 20),
            "integer ties": np.round(rng.normal(0, 3, (300, 15))),
            "trend": np.cumsum(rng.normal(0.001, 0.01, (300, 10)), axis=0),
        }[name]
        _check(values)

    def test_perfectly_correlated_columns_where_the_clip_decides(self):
        """A line and its multiples come out a hair beyond +/-1 in Welford
        arithmetic; pandas 3 clips that and pandas 2 does not, and the
        kernel's caller must do exactly what the installed pandas does."""
        base = np.arange(1, 9, dtype=float) / 7.0
        values = np.column_stack([base, base * 0.1 + 1.0, -3.0 * base, base * 7.0])
        got = _check(values)
        raw = _cpp.pearson_correlation(values, 1)
        assert raw[0, 1] > 1.0  # the arithmetic does overshoot
        assert bool(got[0, 1] == 1.0) is PANDAS_CLIPS

    def test_the_result_is_exactly_symmetric(self):
        got = _kernel(_panel(200, 31, seed=3))
        _assert_same_bits(got, got.T.copy())


class TestMissingValues:
    @pytest.mark.parametrize("share", [0.02, 0.1, 0.3, 0.7])
    def test_scattered_gaps_in_every_column(self, share):
        values = _panel(300, 24, seed=int(share * 100))
        values[np.random.default_rng(5).random(values.shape) < share] = np.nan
        _check(values)
        _check(np.asfortranarray(values))

    def test_complete_and_gapped_columns_side_by_side(self):
        """Every kind of pair in one matrix: both complete (the shared
        recursion), one gapped, both gapped (pandas' loop per pair)."""
        values = _panel(400, 30, seed=6)
        rng = np.random.default_rng(6)
        for k in range(0, 30, 3):
            values[rng.random(400) < 0.15, k] = np.nan
        _check(values)

    @pytest.mark.parametrize("fill", [np.inf, -np.inf])
    def test_an_infinity_is_a_missing_value(self, fill):
        """pandas masks with np.isfinite, so +/-inf drops the row from the
        pair exactly as NaN does."""
        values = _panel(120, 6, seed=8)
        values[[3, 50, 51], 2] = fill
        values[77, 4] = fill
        got = _check(values)
        as_nan = values.copy()
        as_nan[~np.isfinite(as_nan)] = np.nan
        _assert_same_bits(got, _kernel(as_nan))

    @pytest.mark.parametrize(
        "pattern", ["leading", "trailing", "all but one", "a whole row", "alternate"]
    )
    def test_gap_patterns(self, pattern):
        values = _panel(90, 7, seed=9)
        if pattern == "leading":
            values[:30, 1] = np.nan
        elif pattern == "trailing":
            values[-25:, 5] = np.nan
        elif pattern == "all but one":
            values[1:, 3] = np.nan
        elif pattern == "a whole row":
            values[44, :] = np.nan
        else:
            values[::2, 0] = np.nan
            values[1::2, 6] = np.nan  # columns 0 and 6 share no row
        _check(values)

    def test_an_all_missing_column(self):
        values = _panel(60, 5, seed=10)
        values[:, 2] = np.nan
        got = _check(values)
        assert np.isnan(got[2]).all() and np.isnan(got[:, 2]).all()


class TestTheNanRules:
    def test_a_constant_column_is_nan_with_everything_including_itself(self):
        values = _panel(80, 4, seed=11)
        values[:, 1] = 0.1
        got = _check(values)
        assert np.isnan(got[1]).all() and np.isnan(got[:, 1]).all()
        assert not np.isnan(got[0, 2])

    def test_one_row_is_all_nan(self):
        got = _check(_panel(1, 5, seed=12))
        assert np.isnan(got).all()

    def test_no_rows_is_all_nan(self):
        got = _check(np.empty((0, 3)))
        assert got.shape == (3, 3) and np.isnan(got).all()

    def test_no_columns_is_an_empty_matrix(self):
        got = _check(np.empty((10, 0)))
        assert got.shape == (0, 0)

    @pytest.mark.parametrize("min_periods", [0, 1, 2, 39, 40, 41, 60, 61])
    def test_min_periods_on_either_side_of_a_pairs_row_count(self, min_periods):
        """Column 1 has 40 rows in common with the others and column 0 has
        all 60, so each threshold cuts the matrix differently."""
        values = _panel(60, 4, seed=13)
        values[40:, 1] = np.nan
        _check(values, min_periods=min_periods)


class TestHowThePanelIsRead:
    def test_strided_views_are_read_in_place(self):
        base = _panel(301, 41, seed=14)
        for view in (base[::2], base[:, ::3], base[::-1], base[:, ::-1], base.T.T):
            _check(view)

    def test_a_stride_that_is_not_a_whole_double_is_copied_first(self):
        packed = np.zeros((50, 3), dtype=[("x", "f8"), ("flag", "i1")])
        packed["x"] = _panel(50, 3, seed=15)
        view = packed["x"]
        assert view.strides[0] % 8 != 0
        _assert_same_bits(_kernel(view), _pandas(np.ascontiguousarray(view)))

    def test_integers_are_read_as_float64_like_pandas(self):
        values = np.random.default_rng(16).integers(-50, 50, (70, 6))
        _check(values)

    @pytest.mark.parametrize("shape", [(5,), (2, 3, 4)])
    def test_anything_but_a_2d_array_is_refused(self, shape):
        with pytest.raises(ValueError, match="2-D"):
            _cpp.pearson_correlation(np.zeros(shape), 1)


class TestTheLibraryUsesIt:
    def test_the_clip_follows_the_installed_pandas(self):
        """Asked of pandas itself on a probe, not read from the version --
        but on the two majors the answer is the version's."""
        assert construction._native_correlation_mode() == (
            "clip" if PANDAS_CLIPS else "raw"
        )

    def test_hrp_reads_the_same_matrix_as_frame_corr(self):
        frame = pd.DataFrame(
            _panel(400, 25, seed=17), columns=[f"N{i:02d}" for i in range(25)]
        )
        _assert_same_bits(
            construction._correlation_matrix(frame), frame.corr().to_numpy()
        )

    def test_hrp_weights_are_the_same_bits_either_way(self, monkeypatch):
        rng = np.random.default_rng(18)
        factor = rng.normal(0, 0.01, (600, 1))
        frame = pd.DataFrame(
            factor * rng.uniform(0.5, 1.5, 30) + rng.normal(0, 0.006, (600, 30)),
            columns=[f"S{i:02d}" for i in range(30)],
        )
        native = construction.hierarchical_risk_parity(frame)
        monkeypatch.setattr(construction, "HAS_CPP", False)
        fallback = construction.hierarchical_risk_parity(frame)
        assert native["cluster_order"] == fallback["cluster_order"]
        assert native["weights"] == fallback["weights"]
        assert native["risk_contributions"] == fallback["risk_contributions"]

    def test_a_pandas_the_kernel_does_not_reproduce_computes_its_own(self, monkeypatch):
        """The probe's verdict is binding: with the kernel ruled out, the
        matrix is frame.corr()'s and the kernel is not called."""

        class _Refuses:
            def pearson_correlation(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("the kernel was called")

        monkeypatch.setattr(construction, "_correlation_mode", "pandas")
        monkeypatch.setattr(construction, "_cpp_core", _Refuses())
        frame = pd.DataFrame(_panel(50, 4, seed=19))
        _assert_same_bits(
            construction._correlation_matrix(frame), frame.corr().to_numpy()
        )

    def test_a_probe_that_disagrees_rules_the_kernel_out(self, monkeypatch):
        class _OffByOneBit:
            def pearson_correlation(self, values, min_periods):
                out = np.asarray(_cpp.pearson_correlation(values, min_periods))
                return np.nextafter(out, np.inf)

        monkeypatch.setattr(construction, "_correlation_mode", None)
        monkeypatch.setattr(construction, "_cpp_core", _OffByOneBit())
        assert construction._native_correlation_mode() == "pandas"
