"""
The native RSI, ADX and Wilder ATR kernels return the Numba reference's
bits, and the wrappers keep calling them.

The three kernels were respelled so that MSVC compiles them without a call
or an unpredictable branch per bar (see the CHANGELOG entry of 2026-10-01):
an inline finiteness test in place of std::isfinite, a two-argument maximum
in place of std::max over an initializer list, and branch-free selects for
the directional moves and the RSI loss. None of that is allowed to move a
result bit. The Numba kernels perform the same operations in the same
order, so the native result must equal theirs exactly -- compared here as
raw 64-bit patterns, which also tells +0.0 from -0.0 and one NaN payload
from another -- on clean data, on every shape of missing data the kernels
accept, and at the edges of their domain. Against the build before the
change the same comparison gave 0 differing bits over 322 such cases,
through the single-series, fused and panel bindings.

The native path stays on the dispatch because, respelled, it matches or
beats Numba from 2k to 2M bars. `TestTheWrappersCallTheNativeKernel` pins
that choice, so moving a wrapper to the fallback has to be a decision made
with a measurement rather than a side effect.
"""

from typing import Any, Iterator, Tuple

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.indicators import momentum, trend, volatility
from standard_quant_tools.indicators.momentum import _rsi_numba
from standard_quant_tools.indicators.trend import _adx_numba
from standard_quant_tools.indicators.volatility import _wilder_atr_kernel

_cpp: Any = pytest.importorskip(
    "standard_quant_tools._sqt_core", reason="native extension not built"
)

Case = Tuple[str, np.ndarray, np.ndarray, np.ndarray, int]


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
    return high, low, close


def _cases() -> Iterator[Case]:
    # The null case: clean random walks, at magnitudes from 1e-6 to 1e9.
    for seed, scale in enumerate((1e-6, 1.0, 100.0, 1e9)):
        h, l, c = _walk(2500, seed, scale)
        for period in (1, 2, 14, 200):
            yield f"clean scale={scale:g}", h, l, c, period

    # NaN gaps scattered through each column independently.
    rng = np.random.default_rng(11)
    h, l, c = _walk(1500, 11)
    for arr in (h, l, c):
        arr[rng.choice(len(arr), size=120, replace=False)] = np.nan
    yield "scattered gaps", h, l, c, 14

    # A gap inside every seed window, and a run of missing bars.
    h, l, c = _walk(300, 12)
    c[[0, 5, 40, 41, 42, 299]] = np.nan
    yield "gaps in the seed and a run", h, l, c, 14

    # A leading run of missing bars, longer than the seed.
    h, l, c = _walk(400, 13)
    h[:30] = l[:30] = c[:30] = np.nan
    yield "leading NaN", h, l, c, 14

    # +/-inf, which a direct kernel caller sees as a missing bar.
    h, l, c = _walk(400, 14)
    c[[3, 50, 51]] = [np.inf, -np.inf, np.inf]
    h[[7, 200]] = np.inf
    l[100] = -np.inf
    yield "infinities", h, l, c, 14

    # A constant series: every move and every range is zero.
    flat = np.full(100, 42.0)
    yield "constant", flat.copy(), flat.copy(), flat.copy(), 14

    # Ties and signed zeros, where a maximum or a select could pick the
    # other zero.
    q = np.round(_walk(300, 15)[2])
    h, l, c = q.copy(), q.copy(), q.copy()
    h[::3], l[::3], c[::3] = -0.0, 0.0, 0.0
    yield "ties and signed zeros", h, l, c, 3

    # Inverted bars (high below low): bad data the kernels must not change
    # their answer on.
    h, l, c = _walk(300, 16)
    yield "inverted bars", l, h, c, 14

    # Ranges that overflow to inf, so a true range is inf but never NaN.
    big = 1e308
    h = np.tile([big, -big, big, 5.0, big, -big], 10)
    l = np.tile([-big, big, -big, 4.0, -big, big], 10)
    c = np.tile([0.0, big, -big, 4.5, big, 0.0], 10)
    for period in (1, 2, 14):
        yield "overflowing ranges", h, l, c, period

    # Shorter than the period, a single bar, no bars, and a bad period.
    for n in (0, 1, 13, 14, 15):
        z = np.linspace(1.0, 2.0, n)
        for period in (14, 0, -1):
            yield f"n={n}", z + 0.5, z - 0.5, z, period


_CASES = list(_cases())
_IDS = [f"{name} period={period}" for name, *_, period in _CASES]


@pytest.mark.parametrize("name,high,low,close,period", _CASES, ids=_IDS)
class TestTheNativeKernelsReturnTheReferenceBits:
    def test_rsi(self, name, high, low, close, period):
        _assert_same_bits(_cpp.rsi(close, period), _rsi_numba(close, period))

    def test_adx(self, name, high, low, close, period):
        _assert_same_bits(
            _cpp.adx(high, low, close, period), _adx_numba(high, low, close, period)
        )

    def test_wilder_atr(self, name, high, low, close, period):
        _assert_same_bits(
            _cpp.wilder_atr(high, low, close, period),
            _wilder_atr_kernel(high, low, close, period),
        )

    def test_the_fused_and_panel_bindings_reach_the_same_kernels(
        self, name, high, low, close, period
    ):
        if len(close) == 0:
            pytest.skip("the panel binding needs at least one bar")
        request = dict(
            compute_rsi=True,
            rsi_period=period,
            compute_adx=True,
            adx_period=period,
            compute_atr=True,
            atr_period=period,
        )
        single = {
            "rsi": _cpp.rsi(close, period),
            "adx": _cpp.adx(high, low, close, period),
            "atr": _cpp.wilder_atr(high, low, close, period),
        }
        fused = _cpp.technical_indicators(high, low, close, **request)
        panel = _cpp.technical_indicators_panel(
            np.vstack([high, high[::-1]]),
            np.vstack([low, low[::-1]]),
            np.vstack([close, close[::-1]]),
            **request,
        )
        for key, expected in single.items():
            _assert_same_bits(fused[key], expected)
            _assert_same_bits(np.asarray(panel[key])[0], expected)


class TestPlantedAnswers:
    """Answers known by construction, independent of either backend."""

    # (high, low, close) per bar, chosen so that each candidate of the true
    # range wins once and each branch of the directional move is taken.
    BARS = np.array(
        [
            [10.0, 8.0, 9.0],
            [12.0, 9.0, 11.0],  # up 2 > down -1: +DM 2; TR = H-L = 3 (ties |H-C|)
            [11.0, 7.0, 8.0],  # down 2 > up -1: -DM 2; TR = H-L = 4 (ties |L-C|)
            [13.0, 5.0, 6.0],  # up 2 == down 2: no DM; TR = H-L = 8
            [12.0, 6.0, 9.0],  # inside bar, both moves negative: no DM; TR 6
            [20.0, 18.0, 19.0],  # gap up: +DM 8; TR = |H-C| = 11
            [10.0, 9.0, 9.5],  # gap down: -DM 9; TR = |L-C| = 10
        ]
    )
    TRUE_RANGE = [2.0, 3.0, 4.0, 8.0, 6.0, 11.0, 10.0]
    PLUS_DM = [None, 2.0, 0.0, 0.0, 0.0, 8.0, 0.0]
    MINUS_DM = [None, 0.0, 2.0, 0.0, 0.0, 0.0, 9.0]

    @pytest.mark.parametrize("kernel", ["native", "numba"])
    def test_wilder_atr_with_period_one_is_the_true_range(self, kernel):
        fn = _cpp.wilder_atr if kernel == "native" else _wilder_atr_kernel
        h, l, c = self.BARS.T.copy()
        _assert_same_bits(fn(h, l, c, 1), np.array(self.TRUE_RANGE))

    @pytest.mark.parametrize("kernel", ["native", "numba"])
    def test_adx_with_period_one_reads_each_bars_directional_move(self, kernel):
        # With period 1 the smoothed sums are this bar's TR and DM exactly
        # (s - s/1 is +0.0), so DI+ and DI- are 100 * DM / TR, bar by bar.
        fn = _cpp.adx if kernel == "native" else _adx_numba
        h, l, c = self.BARS.T.copy()
        out = fn(h, l, c, 1)
        assert np.isnan(out[0]).all()
        for i in range(1, len(self.BARS)):
            tr = self.TRUE_RANGE[i]
            assert _bits(out[i, 0]) == _bits(100.0 * self.PLUS_DM[i] / tr)
            assert _bits(out[i, 1]) == _bits(100.0 * self.MINUS_DM[i] / tr)

    @pytest.mark.parametrize("kernel", ["native", "numba"])
    def test_rsi_is_100_on_every_rise_and_0_on_every_fall(self, kernel):
        fn = _cpp.rsi if kernel == "native" else _rsi_numba
        rising = np.arange(1.0, 41.0)
        assert (fn(rising, 14)[14:] == 100.0).all()
        assert (fn(rising[::-1].copy(), 14)[14:] == 0.0).all()


class _Spy:
    """The extension module, counting calls to the kernels named."""

    def __init__(self, module: Any, names: Tuple[str, ...]) -> None:
        self._module = module
        self.calls = {name: 0 for name in names}

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._module, name)
        if name not in self.calls:
            return attr

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls[name] += 1
            return attr(*args, **kwargs)

        return counted


class TestTheWrappersCallTheNativeKernel:
    @pytest.fixture
    def ohlc(self):
        h, l, c = _walk(500, 21)
        index = pd.RangeIndex(len(c))
        return pd.Series(h, index), pd.Series(l, index), pd.Series(c, index)

    def test_rsi(self, ohlc, monkeypatch):
        spy = _Spy(_cpp, ("rsi",))
        monkeypatch.setattr(momentum, "_cpp_core", spy)
        monkeypatch.setattr(momentum, "HAS_CPP", True)
        result = momentum.rsi(ohlc[2], 14)
        assert spy.calls == {"rsi": 1}
        _assert_same_bits(result.to_numpy(), _cpp.rsi(ohlc[2].to_numpy(), 14))

    def test_adx(self, ohlc, monkeypatch):
        spy = _Spy(_cpp, ("adx",))
        monkeypatch.setattr(trend, "_cpp_core", spy)
        monkeypatch.setattr(trend, "HAS_CPP", True)
        result = trend.adx(*ohlc, period=14)
        assert spy.calls == {"adx": 1}
        h, l, c = (s.to_numpy() for s in ohlc)
        _assert_same_bits(result.to_numpy(), _cpp.adx(h, l, c, 14))

    def test_wilder_atr(self, ohlc, monkeypatch):
        spy = _Spy(_cpp, ("wilder_atr",))
        monkeypatch.setattr(volatility, "_cpp_core", spy)
        monkeypatch.setattr(volatility, "HAS_CPP", True)
        result = volatility.wilder_atr(*ohlc, period=14)
        assert spy.calls == {"wilder_atr": 1}
        h, l, c = (s.to_numpy() for s in ohlc)
        _assert_same_bits(result.to_numpy(), _cpp.wilder_atr(h, l, c, 14))
