"""
`cusum_peaks` is the numpy CUSUM loop, bit for bit.

The kernel replaces the scan `analysis.liquidity_events` runs over its
simulated AR(1) null paths -- the step that was 40% of a basis-scan
workload. The loop it replaces is kept below verbatim as the reference, and
every comparison is on the raw bits (`view(np.uint64)`), not on values: the
peaks feed a percentile and a `>=` against a threshold, so a last-bit
difference is a possible different answer, not a rounding detail.

The one place bits are allowed to differ is a NaN's payload in a row that
mixes NaNs from two sources; that case is checked for NaN-ness only, and
the reason is in its test. The library's own inputs are finite, so it never
reaches it. See the CHANGELOG entry of 2026-10-01.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = hasattr(_cpp, "cusum_peaks")
except ImportError:
    HAS_CPP = False

pytestmark = pytest.mark.skipif(not HAS_CPP, reason="_sqt_core.cusum_peaks not built")


def _reference_loop(z: np.ndarray, n_reference: int, slack: float) -> np.ndarray:
    """The scan from `_ar1_null_peaks` as it was before the kernel, verbatim."""
    n_simulations, n = z.shape
    up = np.zeros(n_simulations)
    down = np.zeros(n_simulations)
    peaks = np.zeros(n_simulations)
    for t in range(1, n):
        up = np.maximum(0.0, up + z[:, t] - slack)
        down = np.maximum(0.0, down - z[:, t] - slack)
        if t >= n_reference:
            peaks = np.maximum(peaks, np.maximum(up, down))
    return peaks


def _assert_same_bits(got: np.ndarray, want: np.ndarray) -> None:
    assert got.dtype == np.float64 and got.shape == want.shape
    differ = got.view(np.uint64) != want.view(np.uint64)
    assert not differ.any(), (
        f"{int(differ.sum())} of {differ.size} peaks differ, first at row "
        f"{int(np.flatnonzero(differ)[0])}: {got[differ][:3]} vs {want[differ][:3]}"
    )


def _both(z, n_reference, slack):
    z = np.asarray(z, dtype=float)
    kernel = _cpp.cusum_peaks(z, n_reference, slack)
    return kernel, _reference_loop(z, n_reference, slack)


class TestTheKernelIsTheLoop:
    @pytest.mark.parametrize(
        "shape",
        [(1, 1), (1, 2), (1, 2105), (3, 10), (4, 10), (5, 333), (203, 50), (200, 2105)],
    )
    @pytest.mark.parametrize("slack", [0.0, 0.5, 1.7, -0.25])
    def test_seeded_random_panels(self, shape, slack):
        rng = np.random.default_rng([shape[0], shape[1], int(1000 * slack) % 997])
        z = rng.standard_normal(shape) * rng.uniform(0.3, 3.0) + rng.uniform(-0.5, 0.5)
        n = shape[1]
        for n_reference in sorted({0, 1, 2, n // 3, n - 1, n, n + 7}):
            _assert_same_bits(*_both(z, n_reference, slack))

    def test_the_panel_the_caller_builds(self):
        """AR(1) paths standardized against their own reference window --
        the exact construction in `_ar1_null_peaks` -- at the size that was
        measured: 200 paths of 2,105 steps."""
        rng = np.random.default_rng(20261001)
        n, n_reference = 2105, 631
        for phi in (0.0, 0.45, 0.95, -0.6):
            x = rng.standard_normal((200, n))
            paths = np.empty_like(x)
            paths[:, 0] = x[:, 0] / np.sqrt(max(1.0 - phi * phi, 1e-6))
            for t in range(1, n):
                paths[:, t] = phi * paths[:, t - 1] + x[:, t]
            reference = paths[:, :n_reference]
            scale = reference.std(axis=1, ddof=1, keepdims=True)
            z = (paths - reference.mean(axis=1, keepdims=True)) / scale
            _assert_same_bits(*_both(z, n_reference, 0.5))


class TestPlantedCases:
    def test_a_known_crossing(self):
        """A level shift of 2 sigma after the reference window: up gains
        1.5 per step, so the peak is known in closed form and the row
        crosses the library's threshold of 9.0 many times over."""
        z = np.zeros((1, 100))
        z[0, 60:] = 2.0
        got, want = _both(z, 30, 0.5)
        _assert_same_bits(got, want)
        assert got[0] == 1.5 * 40

    def test_a_null_row_never_crosses(self):
        """Every step inside the slack: both sides stay at +0.0 exactly."""
        rng = np.random.default_rng(5)
        z = rng.uniform(-0.5, 0.5, (7, 300))
        got, want = _both(z, 10, 0.5)
        _assert_same_bits(got, want)
        _assert_same_bits(got, np.zeros(7))

    def test_slack_zero(self):
        z = np.tile([0.0, 1.0, -1.0, 1.0, -1.0], (3, 1))
        got, want = _both(z, 0, 0.0)
        _assert_same_bits(got, want)
        assert list(got) == [1.0, 1.0, 1.0]

    def test_all_negative_increments_drive_only_the_down_side(self):
        z = np.full((2, 11), -1.0)
        got, want = _both(z, 0, 0.5)
        _assert_same_bits(got, want)
        assert list(got) == [5.0, 5.0]

    def test_one_row_and_one_column(self):
        _assert_same_bits(*_both(np.array([[0.0, 3.0, -2.0, 4.0]]), 1, 0.5))
        got, want = _both(np.arange(6.0).reshape(6, 1), 0, 0.5)
        _assert_same_bits(got, want)
        _assert_same_bits(got, np.zeros(6))

    def test_no_rows(self):
        got, want = _both(np.empty((0, 50)), 5, 0.5)
        assert got.shape == want.shape == (0,)

    def test_a_reference_window_past_the_end_scans_nothing(self):
        z = np.random.default_rng(2).standard_normal((4, 20)) * 5
        got, want = _both(z, 20, 0.5)
        _assert_same_bits(got, want)
        _assert_same_bits(got, np.zeros(4))
        _assert_same_bits(*_both(z, 10_000, 0.5))

    def test_negative_zero_anywhere_still_gives_positive_zero(self):
        """The kernel's maximum may break a +0.0/-0.0 tie either way; that
        is safe only because -0.0 cannot get into the recursion."""
        z = np.full((2, 6), -0.0)
        for slack in (0.0, -0.0):
            got, want = _both(z, 0, slack)
            _assert_same_bits(got, want)
            _assert_same_bits(got, np.zeros(2))


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
class TestNanAndInfinity:
    """The caller's panel is finite -- standard normal draws through a
    stable AR(1) -- so none of this reaches the kernel from the library.
    It is checked because the binding takes any array. (The reference loop
    warns on inf - inf; the kernel, like any C++ arithmetic, does not.)"""

    def _planted(self):
        rng = np.random.default_rng(9)
        z = rng.standard_normal((6, 40))
        z[0, 25] = np.nan  # after the reference window
        z[1, 3] = np.nan  # inside it: still poisons the row
        z[2, 0] = np.nan  # column 0 is never read
        z[3, 30] = np.inf
        z[4, 30] = -np.inf
        return z

    @pytest.mark.parametrize("slack", [0.5, 0.0])
    def test_single_source_nan_and_infinity_are_bit_identical(self, slack):
        got, want = _both(self._planted(), 10, slack)
        _assert_same_bits(got, want)
        assert np.isnan(got[0]) and np.isnan(got[1]) and not np.isnan(got[2])
        assert got[3] == np.inf and got[4] == np.inf

    @pytest.mark.parametrize("slack", [np.nan, np.inf, -np.inf])
    def test_a_non_finite_slack_follows_ieee_as_the_loop_does(self, slack):
        z = np.random.default_rng(4).standard_normal((5, 30))
        _assert_same_bits(*_both(z, 3, slack))

    def test_two_nan_sources_in_one_row_agree_on_nan_not_on_payload(self):
        """+inf then -inf makes up = inf - inf, a NaN the hardware makes
        (sign bit set); a later np.nan in z is another NaN (sign bit clear).
        Adding the two, numpy's add keeps the SECOND operand's NaN (measured:
        nan_a + nan_b carries nan_b's bits) and a compiled add is free to
        keep either. Both happen inside the reference window here, so which
        one the scan first sees is the one the add kept. The peak is NaN on
        both paths and its payload is not compared; rows without the mix
        are still compared bit for bit."""
        z = np.random.default_rng(6).standard_normal((3, 40))
        z[0, 20], z[0, 21], z[0, 30] = np.inf, -np.inf, np.nan
        got, want = _both(z, 35, 0.5)
        assert np.isnan(got[0]) and np.isnan(want[0])
        _assert_same_bits(got[1:], want[1:])


class TestTheBinding:
    def test_layout_and_dtype_are_converted_not_misread(self):
        """A Fortran-ordered or strided view, or float32 data, is copied to
        the C-contiguous float64 the kernel reads -- never read with the
        wrong strides."""
        rng = np.random.default_rng(8)
        z = rng.standard_normal((9, 70))
        want = _reference_loop(z, 7, 0.5)
        _assert_same_bits(_cpp.cusum_peaks(np.asfortranarray(z), 7, 0.5), want)
        wide = np.zeros((9, 140))
        wide[:, ::2] = z
        _assert_same_bits(_cpp.cusum_peaks(wide[:, ::2], 7, 0.5), want)
        z32 = z.astype(np.float32)
        _assert_same_bits(
            _cpp.cusum_peaks(z32, 7, 0.5), _reference_loop(z32.astype(float), 7, 0.5)
        )

    def test_concurrent_calls_keep_their_own_panels(self):
        """The scan runs with the GIL released. Eight threads on eight
        panels each get exactly their own panel's peaks back."""
        import threading

        panels = [np.random.default_rng(i).standard_normal((50, 400)) for i in range(8)]
        want = [_reference_loop(p, 100, 0.5) for p in panels]
        got: list = [None] * len(panels)

        def work(i):
            for _ in range(20):
                got[i] = _cpp.cusum_peaks(panels[i], 100, 0.5)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        for g, w in zip(got, want):
            _assert_same_bits(g, w)

    def test_the_input_is_not_modified(self):
        z = np.random.default_rng(1).standard_normal((4, 30))
        before = z.copy()
        _cpp.cusum_peaks(z, 3, 0.5)
        _assert_same_bits(z.ravel(), before.ravel())

    @pytest.mark.parametrize(
        "z", [np.zeros(10), np.zeros((2, 3, 4))], ids=["1-D", "3-D"]
    )
    def test_anything_but_2d_is_refused(self, z):
        with pytest.raises(ValueError, match="2-D"):
            _cpp.cusum_peaks(z, 0, 0.5)

    def test_a_negative_reference_length_is_refused(self):
        with pytest.raises(ValueError, match="n_reference must be >= 0"):
            _cpp.cusum_peaks(np.zeros((2, 5)), -1, 0.5)

    def test_the_docstring_names_the_contract(self):
        doc = _cpp.cusum_peaks.__doc__
        assert "bit for bit" in doc and "ValueError" in doc
