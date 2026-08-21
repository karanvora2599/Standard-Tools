"""
The NaN/Inf DATA contract, pinned at the binding layer for every kernel.

bindings.cpp states this contract in writing, in the comment above its
scalar validators:

    What is deliberately NOT validated here: input DATA [...]. Those
    already have a documented contract in this codebase -- degenerate
    arguments and bad bars yield NaN, not exceptions -- and it exists for
    a reason the tests state outright: build_dataset's finite-value guard
    rejects an ENTIRE panel, so one zero print in one symbol used to fail
    a whole multi-entity build and blame the feature rather than the data.

Nothing tested it. Every existing "returns all NaN" test in this directory
covers a degenerate *parameter* (period <= 0, window > n), never a
non-finite *value*, and two kernels were violating the contract outright:

  * bollinger_bands raised RuntimeError for the whole series on one NaN
    bar, and
  * rolling_hurst(method="dfa") raised RuntimeError where hurst_dfa() on
    the same data returned NaN and rolling_hurst(method="rs") returned NaN.

A third, stochastic_oscillator, did something worse than raising: a NaN
high broke its monotonic-deque invariant and it returned %K values of 125,
166.67 and 250 from a stale window maximum, for an indicator whose range
is 0..100 by definition.

These tests are deliberately at the raw-binding layer. Several Python
wrappers call require_finite_array() before dispatching, so they never
exercise this -- but agent/tools.py's fused technical_indicators() path
does not, and neither does any direct _sqt_core caller.

Run:
    pytest tests/cpp_bindings/test_cpp_nan_data_contract.py -v
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

# ── Extension availability ────────────────────────────────────────────────────

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

requires_cpp = pytest.mark.skipif(not HAS_CPP, reason="_sqt_core not built")

N = 240
RNG = np.random.default_rng(20260821)


def _prices(n: int = N) -> np.ndarray:
    return 100.0 * np.exp(np.cumsum(RNG.normal(0.0, 0.01, n)))


def _ohlc(n: int = N):
    close = _prices(n)
    return close + 0.5, close - 0.5, close


# ── Every kernel: a bad bar must not raise ───────────────────────────────────


@requires_cpp
class TestBadBarsNeverRaise:
    """One NaN and one Inf, into every kernel that takes a price series."""

    @staticmethod
    def _poison(arr: np.ndarray, idx: int, value: float) -> np.ndarray:
        out = arr.copy()
        out[idx] = value
        return out

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_single_series_indicators(self, bad: float) -> None:
        p = self._poison(_prices(), 100, bad)
        # Each of these returns an array; none may raise.
        assert _cpp.rsi(p, 14).shape == (N,)
        assert _cpp.bollinger_bands(p, 20, 2.0).shape == (N, 3)
        assert _cpp.rolling_hurst(p, 100, 10, "dfa", 10).shape == (N,)
        assert _cpp.rolling_hurst(p, 100, 10, "rs", 10).shape == (N,)
        assert _cpp.garch11_variance_recursion(p, 1e-6, 0.1, 0.85).shape == (N,)
        assert np.isscalar(_cpp.garch11_neg_loglik(p, 1e-6, 0.1, 0.85, True))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("series", ["high", "low", "close"])
    def test_ohlc_indicators(self, bad: float, series: str) -> None:
        high, low, close = _ohlc()
        arrs = {"high": high, "low": low, "close": close}
        arrs[series] = self._poison(arrs[series], 100, bad)
        h, low_, c = arrs["high"], arrs["low"], arrs["close"]

        assert _cpp.adx(h, low_, c, 14).shape == (N, 3)
        assert _cpp.wilder_atr(h, low_, c, 14).shape == (N,)
        assert _cpp.stochastic_oscillator(h, low_, c, 14, 3).shape == (N, 2)
        assert _cpp.parabolic_sar(h, low_, 0.02, 0.02, 0.2).shape == (N, 2)
        # Fused path -- the one agent/tools.py uses, with no finite guard
        # in front of it.
        fused = _cpp.technical_indicators(
            h,
            low_,
            c,
            compute_rsi=True,
            compute_adx=True,
            compute_atr=True,
            compute_bollinger=True,
            compute_stochastic=True,
        )
        assert set(fused) == {
            "rsi",
            "adx",
            "atr",
            "bollinger_bands",
            "stochastic_oscillator",
        }

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_two_series_kernels(self, bad: float) -> None:
        y = _prices()
        x = _prices()
        y = self._poison(y, 100, bad)
        assert _cpp.rolling_beta(y, x, 30).shape == (N,)
        assert _cpp.ols2(y, x)["slope"] is not None
        assert _cpp.engle_granger(y, x)["n_obs"] == N
        assert _cpp.kalman_filter_1state(y, x, 0.0001, 0.001)["beta"].shape == (N,)
        assert _cpp.kalman_filter_2state(y, x, 0.0001, 0.001)["beta"].shape == (N,)
        factors = np.column_stack([x, _prices()])
        assert _cpp.rolling_factor_loadings(y, factors, 30).shape == (N, 3)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_backtest_and_signals(self, bad: float) -> None:
        prices = self._poison(_prices(), 100, bad)
        signals = RNG.choice([-1.0, 0.0, 1.0], N)
        assert "sharpe_ratio" in _cpp.run_strategy(prices, signals)
        assert _cpp.batch_run_strategy(prices, signals.reshape(1, -1)).shape == (1, 11)
        vwap = self._poison(_prices(), 100, bad)
        assert _cpp.vwap_reversion_state_machine(prices, vwap, 0.01).shape == (N,)
        assert _cpp.donchian_state_machine(prices, vwap, vwap).shape == (N,)


# ── bollinger_bands: the raise, and the recovery ─────────────────────────────


@requires_cpp
class TestBollingerBadBar:
    def test_nan_bar_does_not_raise(self) -> None:
        p = _prices(60)
        p[30] = np.nan
        out = _cpp.bollinger_bands(p, 10, 2.0)  # used to raise RuntimeError
        assert out.shape == (60, 3)

    def test_matches_pandas_including_where_nan_falls(self) -> None:
        """The half a bare does-not-raise test would miss.

        An O(1) sliding sum cannot un-add a NaN (NaN - NaN is NaN), so the
        sums stayed poisoned until the next periodic refresh and the kernel
        reported NaN for a run of windows holding no bad data at all.
        """
        p = _prices(120)
        p[50] = np.nan
        out = _cpp.bollinger_bands(p, 20, 2.0)

        s = pd.Series(p)
        mid = s.rolling(20).mean()
        sd = s.rolling(20).std()  # ddof=1, as the kernel uses
        ref = np.column_stack([mid + 2 * sd, mid, mid - 2 * sd])

        np.testing.assert_allclose(out, ref, rtol=1e-12, equal_nan=True)

    def test_recovers_on_the_first_clean_window(self) -> None:
        p = _prices(60)
        p[30] = np.nan
        out = _cpp.bollinger_bands(p, 10, 2.0)
        assert np.all(np.isnan(out[30:40, 1])), "windows covering bar 30"
        assert not np.isnan(out[40, 1]), "first window past it must recover"

    def test_leading_nan_does_not_poison_the_series(self) -> None:
        # Bar 0 is also the reference point the shifted sums are centred on.
        p = _prices(60)
        p[0] = np.nan
        out = _cpp.bollinger_bands(p, 10, 2.0)
        assert np.all(np.isnan(out[:10, 1]))
        assert not np.isnan(out[10, 1])


# ── stochastic_oscillator: the deque invariant ───────────────────────────────


@requires_cpp
class TestStochasticBadBar:
    @staticmethod
    def _pandas_reference(high, low, close, k_period, d_period):
        hs, ls, cs = pd.Series(high), pd.Series(low), pd.Series(close)
        ll = ls.rolling(k_period).min()
        hh = hs.rolling(k_period).max()
        rng = hh - ll
        k = (100 * ((cs - ll) / rng.where(rng > 0))).where(rng.isna() | (rng > 0), 0.0)
        return k, k.rolling(d_period).mean()

    def test_k_stays_within_bounds_with_a_nan_high(self) -> None:
        """REGRESSION: returned 125, 166.67 and 250 on this exact shape."""
        n = 20
        close = np.arange(9.0, 9.0 + n)
        high, low = close + 0.5, close - 0.5
        high[5] = np.nan

        out = _cpp.stochastic_oscillator(high, low, close, 5, 3)
        k = out[:, 0]
        finite = k[~np.isnan(k)]
        assert finite.min() >= 0.0 and finite.max() <= 100.0

    def test_matches_pandas_with_a_nan_high(self) -> None:
        n = 20
        close = np.arange(9.0, 9.0 + n)
        high, low = close + 0.5, close - 0.5
        high[5] = np.nan

        out = _cpp.stochastic_oscillator(high, low, close, 5, 3)
        k_ref, d_ref = self._pandas_reference(high, low, close, 5, 3)
        np.testing.assert_allclose(
            out[:, 0], k_ref.to_numpy(), rtol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            out[:, 1], d_ref.to_numpy(), rtol=1e-12, equal_nan=True
        )

    def test_nan_in_one_series_only(self) -> None:
        """min_dq and max_dq are gated on their own series, not on the bar."""
        high, low, close = _ohlc(80)
        low = low.copy()
        low[30] = np.nan
        out = _cpp.stochastic_oscillator(high, low, close, 10, 3)
        k_ref, d_ref = self._pandas_reference(high, low, close, 10, 3)
        np.testing.assert_allclose(
            out[:, 0], k_ref.to_numpy(), rtol=1e-12, equal_nan=True
        )
        np.testing.assert_allclose(
            out[:, 1], d_ref.to_numpy(), rtol=1e-12, equal_nan=True
        )

    def test_one_nan_k_does_not_poison_every_later_d(self) -> None:
        high, low, close = _ohlc(80)
        high = high.copy()
        high[20] = np.nan
        out = _cpp.stochastic_oscillator(high, low, close, 5, 3)
        assert not np.isnan(out[-1, 1]), "%D must recover long before the end"


# ── rolling_hurst: three entry points, one answer ────────────────────────────


@requires_cpp
class TestHurstBadBar:
    def test_all_three_entry_points_agree_on_nan_data(self) -> None:
        a = RNG.normal(0.0, 0.01, 600)
        a[300] = np.nan

        # None of these may raise, and none may claim a real estimate.
        assert np.isnan(_cpp.hurst_dfa(a)["hurst"])
        assert _cpp.hurst_dfa(a)["regime"] == "unknown"
        dfa = _cpp.rolling_hurst(a, 200, 10, "dfa", 10)  # used to raise
        rs = _cpp.rolling_hurst(a, 200, 10, "rs", 10)
        assert dfa.shape == rs.shape == (600,)

    def test_nan_is_local_to_the_windows_that_cover_the_bad_bar(self) -> None:
        a = RNG.normal(0.0, 0.01, 600)
        a[300] = np.nan
        out = _cpp.rolling_hurst(a, 200, 10, "dfa", 10)

        covering = [i for i in range(199, 600, 10) if i - 199 <= 300 <= i]
        clean = [i for i in range(199, 600, 10) if i not in covering]
        assert covering, "test setup must produce covered windows"
        assert all(np.isnan(out[i]) for i in covering)
        assert any(not np.isnan(out[i]) for i in clean), (
            "returning all-NaN would satisfy the covering check while still "
            "throwing away every good window"
        )

    def test_clean_data_is_unaffected(self) -> None:
        a = RNG.normal(0.0, 0.01, 600)
        out = _cpp.rolling_hurst(a, 200, 10, "dfa", 10)
        assert not np.isnan(out[-1])
        # dfa_onepass is a documented reassociation of dfa(), not bit-equal.
        assert out[-1] == pytest.approx(_cpp.hurst_dfa(a[-200:])["hurst"], rel=1e-9)
