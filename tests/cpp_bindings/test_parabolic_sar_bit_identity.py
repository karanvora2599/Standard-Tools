"""
The native Parabolic SAR returns the Numba reference's bits, on clean and
gapped series, and the wrapper keeps calling it.

Two changes of the CHANGELOG entry of 2026-10-04 meet here. A NaN high or
low is now a missing bar that the state machine skips, in the kernel and in
its Numba fallback alike; and the kernel was respelled so that MSVC compiles
the new-extreme update without a branch, the two prior lows into one
minimum before the SAR meets them, and the output without a separate NaN
pass. None of that is allowed to move a result bit: the Numba kernel keeps
the plain spelling, so the native result must equal it exactly -- compared
as raw 64-bit patterns, which also tells +0.0 from -0.0 -- on every case
below. On the same cases with no missing bar, the respelled kernel returned
the previous build's bits.

The native path stays on the dispatch because, respelled, it runs ahead of
Numba from 20k to 2M bars where it used to trail it (0.82-0.92x before).
`TestTheWrapperCallsTheNativeKernel` pins that choice.
"""

from typing import Any, Iterator, Tuple

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.indicators import trend
from standard_quant_tools.indicators.trend import _psar_numba

_cpp: Any = pytest.importorskip(
    "standard_quant_tools._sqt_core", reason="native extension not built"
)

Case = Tuple[str, np.ndarray, np.ndarray]

AF_SETTINGS = [
    (0.02, 0.02, 0.2),  # the convention
    (0.01, 0.005, 0.5),
    (0.2, 0.0, 0.2),  # no step: af never moves
    (0.05, -0.0, 0.1),  # a negative-zero step is accepted and adds nothing
    (0.02, 0.3, 0.25),  # a step past the cap on the first new extreme
]


def _bits(values: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64)).view(np.uint64)


def _assert_same_bits(native: Any, reference: Any) -> None:
    native, reference = np.asarray(native), np.asarray(reference)
    assert native.shape == reference.shape
    differ = np.flatnonzero(_bits(native) != _bits(reference))
    assert differ.size == 0, (
        f"{differ.size} values differ, first at flat index {differ[:5]}: "
        f"native {native.ravel()[differ[:5]]} vs reference "
        f"{reference.ravel()[differ[:5]]}"
    )


def _walk(n: int, seed: int, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    close = scale * (100.0 + np.cumsum(rng.normal(0.0, 1.0, n)))
    high = close + scale * rng.uniform(0.0, 1.2, n)
    low = close - scale * rng.uniform(0.0, 1.2, n)
    return high, low


def _cases() -> Iterator[Case]:
    # The null case: clean random walks, at magnitudes from 1e-6 to 1e9.
    for seed, scale in enumerate((1e-6, 1.0, 100.0, 1e9)):
        h, l = _walk(2500, seed, scale)
        yield f"clean scale={scale:g}", h, l

    # NaN gaps scattered through each column independently.
    rng = np.random.default_rng(11)
    h, l = _walk(1500, 11)
    for arr in (h, l):
        arr[rng.choice(len(arr), size=120, replace=False)] = np.nan
    yield "scattered gaps", h, l

    # The first bars missing, the second present bar alone, a run of gaps
    # and the last bar missing.
    h, l = _walk(300, 12)
    h[[0, 1, 40, 41, 42, 299]] = np.nan
    l[[3, 150]] = np.nan
    yield "gaps at the start, a run, the end", h, l

    # A leading run of missing bars in both columns.
    h, l = _walk(400, 13)
    h[:30] = l[:30] = np.nan
    yield "leading NaN", h, l

    # Gaps on every other bar, so no two present bars are adjacent.
    h, l = _walk(200, 14)
    h[1::2] = np.nan
    yield "every other bar missing", h, l

    # +/-inf, which a direct kernel caller sees as a missing bar.
    h, l = _walk(400, 15)
    h[[3, 200]] = np.inf
    l[[50, 51]] = [-np.inf, np.inf]
    yield "infinities", h, l

    # A constant series: every comparison is a tie.
    flat = np.full(100, 42.0)
    yield "constant", flat.copy(), flat.copy()

    # Ties and signed zeros, where a minimum or a select could pick the
    # other zero.
    q = np.round(_walk(300, 16)[0] - 100.0)
    h, l = q.copy(), q.copy() - 1.0
    h[::3], l[::3] = -0.0, 0.0
    l[1::5] = -0.0
    yield "ties and signed zeros", h, l

    # Inverted bars (high below low): bad data the kernel must not change
    # its answer on.
    h, l = _walk(300, 17)
    yield "inverted bars", l, h

    # Ranges that overflow, so a SAR step is infinite before the cap.
    big = 1e308
    h = np.tile([big, -big, big, 5.0, big, -big], 10)
    l = np.tile([-big, big, -big, 4.0, -big, big], 10)
    yield "overflowing ranges", h, l

    # No bars, one, two and three, and all missing.
    for n in (0, 1, 2, 3):
        z = np.linspace(1.0, 2.0, n)
        yield f"n={n}", z + 0.5, z - 0.5
    yield "all missing", np.full(5, np.nan), np.arange(5.0)


_CASES = list(_cases())
_IDS = [name for name, *_ in _CASES]


@pytest.mark.parametrize("af", AF_SETTINGS, ids=lambda af: "af=%g,%g,%g" % af)
@pytest.mark.parametrize("name,high,low", _CASES, ids=_IDS)
def test_the_native_kernel_returns_the_reference_bits(name, high, low, af):
    _assert_same_bits(_cpp.parabolic_sar(high, low, *af), _psar_numba(high, low, *af))


@pytest.mark.parametrize("name,high,low", _CASES[:8], ids=_IDS[:8])
def test_the_plain_python_tier_returns_the_same_bits(name, high, low):
    """The tier a machine with neither the extension nor Numba runs."""
    plain = getattr(_psar_numba, "py_func", _psar_numba)
    _assert_same_bits(
        _cpp.parabolic_sar(high, low, 0.02, 0.02, 0.2),
        plain(high, low, 0.02, 0.02, 0.2),
    )


class TestPlantedAnswers:
    """Answers worked by hand, independent of either backend."""

    @pytest.mark.parametrize("kernel", ["native", "numba"])
    def test_a_missing_bar_is_stepped_over(self, kernel):
        fn = _cpp.parabolic_sar if kernel == "native" else _psar_numba
        high = np.array([10.0, 11.0, np.nan, 12.0])
        low = np.array([9.0, 10.0, 10.5, 11.0])
        out = fn(high, low, 0.1, 0.1, 0.5)
        # Bar 0 bootstraps: SAR = low 9, EP = high 10.
        assert out[0].tolist() == [9.0, 1.0]
        # Bar 1: 9 + 0.1 * (10 - 9) = 9.1, capped by low[0] = 9 -> 9; a new
        # high 11 raises af to 0.2.
        assert out[1].tolist() == [9.0, 1.0]
        # Bar 2 is missing.
        assert np.isnan(out[2]).all()
        # Bar 3 follows bar 1 as if bar 2 were not there: 9 + 0.2 * (11 - 9)
        # = 9.4, capped by the two previous PRESENT lows, 10 and 9 -> 9.
        assert out[3].tolist() == [9.0, 1.0]

    @pytest.mark.parametrize("kernel", ["native", "numba"])
    def test_the_bootstrap_is_at_the_first_present_bar(self, kernel):
        fn = _cpp.parabolic_sar if kernel == "native" else _psar_numba
        high = np.array([np.nan, 7.0, 8.0])
        low = np.array([1.0, 5.0, 6.0])
        out = fn(high, low, 0.02, 0.02, 0.2)
        assert np.isnan(out[0]).all()
        assert out[1].tolist() == [5.0, 1.0]
        # One prior present bar only: 5 + 0.02 * (7 - 5) = 5.04, capped by 5.
        assert out[2].tolist() == [5.0, 1.0]


class _Spy:
    """The extension module, counting calls to parabolic_sar."""

    def __init__(self, module: Any) -> None:
        self._module = module
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._module, name)
        if name != "parabolic_sar":
            return attr

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return attr(*args, **kwargs)

        return counted


class TestTheWrapperCallsTheNativeKernel:
    def test_parabolic_sar(self, monkeypatch):
        h, l = _walk(500, 21)
        h[[10, 11, 300]] = np.nan
        spy = _Spy(_cpp)
        monkeypatch.setattr(trend, "_cpp_core", spy)
        monkeypatch.setattr(trend, "HAS_CPP", True)
        result = trend.parabolic_sar(pd.Series(h), pd.Series(l))
        assert spy.calls == 1
        _assert_same_bits(result.to_numpy(), _cpp.parabolic_sar(h, l, 0.02, 0.02, 0.2))
