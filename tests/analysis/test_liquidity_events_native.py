"""
The AR(1) null behind the CUSUM threshold answers exactly as it did before
it was moved off the Python loop.

`_ar1_null_peaks` simulates 200 AR(1) paths of the channel's length and
scans each with the CUSUM, and it was 40% of a basis-scan workload. Both
halves changed: the paths come from `scipy.signal.lfilter` instead of a
column loop, and the scan from the compiled `cusum_peaks` when the extension
carries it. The peaks feed a 95th percentile (the calibrated threshold) and
a `>=` against the threshold (the false-alarm rate), so the bar is the same
doubles, not close ones.

The implementation before the change is kept below verbatim as the
reference, and every comparison is on the raw bits. The fallback is forced
through the module's `HAS_CPP` flag, so both dispatch paths are held to the
same reference in one run. See the CHANGELOG entry of 2026-10-01.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

from standard_quant_tools.analysis import liquidity_events as le
from standard_quant_tools.delta_one.basis import detect_basis_dislocation

requires_kernel = pytest.mark.skipif(
    not le.HAS_CPP, reason="_sqt_core.cusum_peaks not built"
)


def _ar1_null_peaks_before(rho, n, n_reference, slack, n_simulations, seed):
    """`_ar1_null_peaks` before the change, verbatim."""
    phi = float(np.clip(rho if np.isfinite(rho) else 0.0, -0.95, 0.95))
    rng = np.random.default_rng(seed)
    n_simulations = max(int(n_simulations), 20)
    innovations = rng.standard_normal((n_simulations, n))
    paths = np.empty_like(innovations)
    paths[:, 0] = innovations[:, 0] / math.sqrt(max(1.0 - phi * phi, 1e-6))
    for t in range(1, n):
        paths[:, t] = phi * paths[:, t - 1] + innovations[:, t]
    reference = paths[:, :n_reference]
    scale = reference.std(axis=1, ddof=1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    z = (paths - reference.mean(axis=1, keepdims=True)) / scale
    up = np.zeros(n_simulations)
    down = np.zeros(n_simulations)
    peaks = np.zeros(n_simulations)
    for t in range(1, n):
        up = np.maximum(0.0, up + z[:, t] - slack)
        down = np.maximum(0.0, down - z[:, t] - slack)
        if t >= n_reference:
            peaks = np.maximum(peaks, np.maximum(up, down))
    return peaks


def _bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def _same(a, b) -> bool:
    """Exact equality of two results, NaN equal to NaN, floats by their
    bits -- so 0.1 + 0.2 against 0.3 would fail here, as it should."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or _bits(a) == _bits(b)
    return type(a) is type(b) and a == b


def _null(monkeypatch, *, native: bool, args):
    monkeypatch.setattr(le, "HAS_CPP", native)
    return le._ar1_null_peaks(*args)


_GRID = [
    # (rho, n, n_reference, slack, n_simulations, seed)
    (0.45, 2105, 631, 0.5, 200, 0),  # the measured workload
    (0.0, 10, 5, 0.5, 200, 1),  # the shortest series cusum accepts
    (-0.3, 11, 5, 0.5, 5, 2),  # n_simulations below the floor of 20
    (0.95, 300, 90, 0.5, 200, 3),
    (0.999, 300, 90, 0.5, 50, 4),  # clipped to 0.95
    (-0.999, 300, 90, 0.5, 50, 5),  # clipped to -0.95
    (float("nan"), 120, 36, 0.5, 200, 6),  # no autocorrelation: phi = 0
    (0.67, 1050, 315, 0.0, 200, 7),  # slack 0
    (0.2, 500, 150, 1.25, 64, 8),
    (0.5, 60, 59, 0.5, 33, 9),  # one scanned column
]


class TestTheAr1HalfIsTheColumnLoop:
    @pytest.mark.parametrize("phi", [0.0, -0.0, 0.95, -0.95, 0.45, 1e-300, -0.67])
    @pytest.mark.parametrize("shape", [(20, 2), (200, 2105), (1, 1), (37, 333)])
    def test_lfilter_paths_are_the_loops_bits(self, phi, shape):
        """y[t] = phi * y[t-1] + x[t]: lfilter's other products are by 1.0
        and 0.0, which are exact, so each step rounds as the loop did."""
        x = np.random.default_rng([shape[0], shape[1]]).standard_normal(shape)
        start = x[:, 0] / math.sqrt(max(1.0 - phi * phi, 1e-6))
        loop = np.empty_like(x)
        loop[:, 0] = start
        for t in range(1, shape[1]):
            loop[:, t] = phi * loop[:, t - 1] + x[:, t]
        filtered_input = x.copy()
        filtered_input[:, 0] = start
        filtered = lfilter([1.0], [1.0, -phi], filtered_input, axis=1)
        assert np.array_equal(_bits(filtered), _bits(loop))


class TestTheNullPeaksAreUnchanged:
    @pytest.mark.parametrize("args", _GRID)
    def test_the_fallback_matches_the_old_implementation(self, monkeypatch, args):
        want = _ar1_null_peaks_before(*args)
        got = _null(monkeypatch, native=False, args=args)
        assert np.array_equal(_bits(got), _bits(want))

    @requires_kernel
    @pytest.mark.parametrize("args", _GRID)
    def test_the_kernel_matches_the_old_implementation(self, monkeypatch, args):
        want = _ar1_null_peaks_before(*args)
        got = _null(monkeypatch, native=True, args=args)
        assert np.array_equal(_bits(got), _bits(want))

    @requires_kernel
    def test_the_kernel_is_what_runs_when_it_is_there(self, monkeypatch):
        """The dispatch, held: with HAS_CPP the scan is the kernel's, and
        without it the loop's -- so a parity test above cannot pass by
        comparing the fallback with itself."""
        calls = []
        real = le._cpp_core.cusum_peaks
        monkeypatch.setattr(
            le._cpp_core,
            "cusum_peaks",
            lambda *a: calls.append("kernel") or real(*a),
        )
        loop = le._cusum_peaks_loop
        monkeypatch.setattr(
            le, "_cusum_peaks_loop", lambda *a: calls.append("loop") or loop(*a)
        )
        _null(monkeypatch, native=True, args=_GRID[1])
        _null(monkeypatch, native=False, args=_GRID[1])
        assert calls == ["kernel", "loop"]


def _channel(n=2105, seed=11, *, shift_at=1500):
    """An autocorrelated basis in bps with a sustained shift late on."""
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.7 * noise[t - 1] + rng.normal(0, 4.0)
    noise[shift_at:] += 15.0
    return pd.Series(20.0 + noise, index=pd.date_range("2018-01-01", periods=n))


def _market(seed=0, n=1500):
    """Trades and quotes with the spread widening from 60% through."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2024-03-01 09:30", periods=n, freq="1s")
    on = np.arange(n) >= int(n * 0.6)
    mid = 100 + np.cumsum(rng.normal(0, 0.004, n))
    half = np.where(on, 0.06, 0.01) * rng.lognormal(0, 0.25, n)
    quotes = pd.DataFrame(
        {
            "timestamp": stamps,
            "bid_price": mid - half,
            "ask_price": mid + half,
            "bid_size": 500.0,
            "ask_size": 500.0,
        }
    )
    side = np.where(on, rng.choice([1, 1, 1, -1], n), rng.choice([1, -1], n))
    trades = pd.DataFrame(
        {
            "timestamp": stamps,
            "price": mid + side * half * 0.9,
            "size": rng.integers(80, 300, n).astype(float),
        }
    )
    return trades, quotes


class TestThePublicCallsAnswerTheSame:
    """Before against after, through the functions a caller uses: the
    calibrated threshold, the false-alarm rate and every field around them."""

    def _three_ways(self, monkeypatch, call):
        monkeypatch.setattr(le, "_ar1_null_peaks", _ar1_null_peaks_before)
        before = call()
        monkeypatch.undo()
        monkeypatch.setattr(le, "HAS_CPP", False)
        fallback = call()
        monkeypatch.undo()
        after = call()
        return before, fallback, after

    @pytest.mark.parametrize("calibrate", [False, True])
    def test_cusum(self, monkeypatch, calibrate):
        series = _channel()
        before, fallback, after = self._three_ways(
            monkeypatch, lambda: le.cusum(series, calibrate_threshold=calibrate)
        )
        assert _same(before, fallback) and _same(before, after)
        # The fields the null decides, and that this series exercises them.
        assert before["triggered"] is True
        assert before["false_alarm_rate_at_threshold"] is not None

    def test_detect_basis_dislocation(self, monkeypatch):
        rng = np.random.default_rng(3)
        n = 2105
        spot = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        basis = 0.002 + _channel(n, seed=4).to_numpy() * 1e-5
        futures = spot * (1.0 + basis)
        tte = np.linspace(0.5, 0.25, n)
        before, fallback, after = self._three_ways(
            monkeypatch,
            lambda: detect_basis_dislocation(
                spot=spot, futures=futures, time_to_expiry=tte
            ),
        )
        assert _same(before, fallback) and _same(before, after)

    @pytest.mark.parametrize("calibrate", [False, True])
    def test_detect_liquidity_events(self, monkeypatch, calibrate):
        trades, quotes = _market()
        before, fallback, after = self._three_ways(
            monkeypatch,
            lambda: le.detect_liquidity_events(
                channels=["spread", "mid_return", "trade_intensity", "signed_volume"],
                trades=trades,
                quotes=quotes,
                freq="5s",
                calibrate_threshold=calibrate,
            ),
        )
        assert _same(before, fallback) and _same(before, after)
        assert before["results"], "the market must give the detector channels to run"
