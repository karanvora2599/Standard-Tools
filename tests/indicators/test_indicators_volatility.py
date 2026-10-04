"""Tests for volatility indicators: Bollinger Bands, ATR."""

import logging
import types

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.strategies import _bollinger_signals
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators import volatility
from standard_quant_tools.indicators.volatility import (
    atr,
    bollinger_bands,
    collapse_flat_windows,
    flat_window_mask,
    wilder_atr,
)


class TestBollingerBands:
    def test_returns_correct_columns(self, sample_close):
        result = bollinger_bands(sample_close)
        assert set(result.columns) == {"BB_Upper", "BB_Middle", "BB_Lower"}

    def test_output_length_matches_input(self, sample_close):
        result = bollinger_bands(sample_close)
        assert len(result) == len(sample_close)

    def test_upper_greater_than_middle_greater_than_lower(self, sample_close):
        result = bollinger_bands(sample_close, period=20, num_std=2.0)
        valid = result.dropna()
        assert (valid["BB_Upper"] > valid["BB_Middle"]).all()
        assert (valid["BB_Middle"] > valid["BB_Lower"]).all()

    def test_middle_equals_sma(self, sample_close):
        """BB_Middle must exactly equal SMA(period)."""
        from standard_quant_tools.indicators.trend import sma

        period = 20
        result = bollinger_bands(sample_close, period=period)
        expected_middle = sma(sample_close, period)
        pd.testing.assert_series_equal(
            result["BB_Middle"].dropna(),
            expected_middle.dropna(),
            check_names=False,
            rtol=1e-10,
        )

    def test_band_width_equals_2_std_multiples(self, sample_close):
        """Upper - Middle should equal num_std * rolling_std exactly."""
        period, num_std = 20, 2.0
        result = bollinger_bands(sample_close, period=period, num_std=num_std)
        rolling_std = sample_close.rolling(period).std()
        expected_half_width = rolling_std * num_std
        actual_half_width = result["BB_Upper"] - result["BB_Middle"]
        diff = (actual_half_width - expected_half_width).dropna().abs()
        assert diff.max() < 1e-10

    def test_wider_bands_with_higher_num_std(self, sample_close):
        bb2 = bollinger_bands(sample_close, num_std=2.0)
        bb3 = bollinger_bands(sample_close, num_std=3.0)
        width2 = (bb2["BB_Upper"] - bb2["BB_Lower"]).dropna()
        width3 = (bb3["BB_Upper"] - bb3["BB_Lower"]).dropna()
        assert (width3 > width2).all()

    def test_constant_series_yields_zero_width(self):
        """A flat price series has zero volatility → bands collapse to SMA."""
        s = pd.Series([50.0] * 50)
        result = bollinger_bands(s, period=10)
        width = (result["BB_Upper"] - result["BB_Lower"]).dropna()
        assert width.abs().max() < 1e-10

    def test_nan_prefix_length(self, sample_close):
        period = 20
        result = bollinger_bands(sample_close, period=period)
        assert result.iloc[: period - 1].isna().all(axis=None)

    def test_nan_blanks_the_windows_that_hold_it(self, sample_close):
        """A NaN price is a missing bar: every window holding it is NaN and
        the bands resume at the first window past it. It used to be
        refused here while the panel answered it."""
        bad = sample_close.copy()
        bad.iloc[30] = np.nan
        out = bollinger_bands(bad, period=20)
        assert out.iloc[30:50].isna().all(axis=None)
        assert out.iloc[50].notna().all()
        clean = bollinger_bands(sample_close, period=20)
        pd.testing.assert_frame_equal(out.iloc[50:], clean.iloc[50:], rtol=1e-9)

    def test_inf_in_input_raises(self, sample_close):
        bad = sample_close.copy()
        bad.iloc[10] = np.inf
        with pytest.raises(ValidationError, match="infinite"):
            bollinger_bands(bad)

    @pytest.mark.parametrize("period", [1, 0, -3])
    def test_a_period_below_two_is_refused(self, sample_close, period):
        """One bar has no sample standard deviation (0/0). The kernel used
        to answer all-NaN and pandas a middle band equal to the price."""
        with pytest.raises(ValidationError, match="at least 2"):
            bollinger_bands(sample_close, period=period)

    def test_period_two_is_the_smallest_accepted(self, sample_close):
        out = bollinger_bands(sample_close, period=2)
        assert out.iloc[1:].notna().all(axis=None)


class TestWilderATR:
    def test_inf_in_input_raises(self, sample_ohlcv):
        bad_high = sample_ohlcv["High"].copy()
        bad_high.iloc[5] = np.inf
        with pytest.raises(ValidationError, match="infinite"):
            wilder_atr(bad_high, sample_ohlcv["Low"], sample_ohlcv["Close"])

    def test_nan_is_a_missing_bar_the_recursion_skips(self, sample_ohlcv):
        """The ATR of the series with the bar dropped, NaN at the bar."""
        bad_close = sample_ohlcv["Close"].copy()
        bad_close.iloc[7] = np.nan
        out = wilder_atr(sample_ohlcv["High"], sample_ohlcv["Low"], bad_close)
        assert np.isnan(out.iloc[7])
        keep = bad_close.notna()
        dropped = wilder_atr(
            sample_ohlcv["High"][keep], sample_ohlcv["Low"][keep], bad_close[keep]
        )
        pd.testing.assert_series_equal(out[keep], dropped)
        assert np.isfinite(out.iloc[-1])


class TestATR:
    def test_output_length_matches_input(self, sample_ohlcv):
        result = atr(sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"])
        assert len(result) == len(sample_ohlcv)

    def test_atr_is_nonnegative(self, sample_ohlcv):
        result = atr(
            sample_ohlcv["High"], sample_ohlcv["Low"], sample_ohlcv["Close"]
        ).dropna()
        assert (result >= 0).all()

    def test_atr_captures_gaps(self):
        """ATR should spike when a large overnight gap occurs."""
        n = 40
        high = pd.Series([101.0] * n)
        low = pd.Series([99.0] * n)
        close = pd.Series([100.0] * n)
        # Insert a large gap: close drops from 100 to 80 between bar 20 and 21
        close.iloc[20] = 80.0
        high.iloc[20] = 81.0
        low.iloc[20] = 79.0

        result = atr(high, low, close, period=5)
        # ATR near bar 20 should be higher than ATR at bar 0
        atr_before = float(result.iloc[15])
        atr_after = float(result.iloc[22])
        assert atr_after > atr_before

    def test_higher_volatility_yields_higher_atr(self):
        """A high-volatility series should have a larger ATR than a low-vol one."""
        n = 60
        # Low vol
        hi_lo = pd.Series([101.0] * n), pd.Series([99.0] * n), pd.Series([100.0] * n)
        # High vol
        hi_hi = pd.Series([110.0] * n), pd.Series([90.0] * n), pd.Series([100.0] * n)

        atr_lo = atr(*hi_lo, period=14).dropna().mean()
        atr_hi = atr(*hi_hi, period=14).dropna().mean()
        assert atr_hi > atr_lo


def _walk_ohlc(n=300, seed=4):
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    high = close + rng.uniform(0.1, 1.2, n)
    low = close - rng.uniform(0.1, 1.2, n)
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    return (pd.Series(a, index=index) for a in (high, low, close))


class TestATRMissingBars:
    """
    The simple ATR reads a NaN as a missing bar (CHANGELOG entry of
    2026-10-04); it used to refuse it while its sibling wilder_atr, the
    panel and every other indicator read it as a gap. Its true range is
    Wilder's -- NaN at the missing bar, measured against the last present
    close at the bar after it -- and its mean is a window: NaN for every
    window holding a missing bar, resuming at the first that does not.
    """

    @pytest.mark.parametrize("period", [1, 5, 14, 50])
    def test_finite_input_gives_the_previous_bits(self, period):
        """The null case: with every bar present the answer is the one the
        function gave before NaN was accepted, bit for bit."""
        high, low, close = _walk_ohlc()
        prev_close = close.shift(1).to_numpy(dtype=float)
        h, l = high.to_numpy(dtype=float), low.to_numpy(dtype=float)
        tr = np.maximum(
            h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close))
        )
        before = pd.Series(tr, index=close.index).rolling(window=period).mean()
        pd.testing.assert_series_equal(
            atr(high, low, close, period), before, check_exact=True
        )

    @pytest.mark.parametrize("column", ["high", "low", "close"])
    def test_an_infinity_is_refused(self, column):
        cols = dict(zip(("high", "low", "close"), _walk_ohlc()))
        cols[column].iloc[30] = -np.inf
        with pytest.raises(ValidationError, match="infinite"):
            atr(cols["high"], cols["low"], cols["close"])

    @pytest.mark.parametrize("column", ["high", "low", "close"])
    def test_the_windows_holding_a_missing_bar_are_nan(self, column):
        cols = dict(zip(("high", "low", "close"), _walk_ohlc()))
        cols[column].iloc[100] = np.nan
        out = atr(cols["high"], cols["low"], cols["close"], 14)
        assert out.iloc[15:100].notna().all()
        assert out.iloc[100:114].isna().all()
        assert out.iloc[114:].notna().all()

    def test_the_first_window_after_the_gap_measures_from_the_last_present_close(
        self,
    ):
        high, low, close = _walk_ohlc()
        close.iloc[100] = np.nan
        out = atr(high, low, close, 5)
        h, l, c = (s.to_numpy() for s in (high, low, close))

        def tr(i, prev):
            return max(h[i] - l[i], abs(h[i] - c[prev]), abs(l[i] - c[prev]))

        expected = [tr(101, 99)] + [tr(i, i - 1) for i in range(102, 106)]
        assert out.iloc[105] == pytest.approx(np.mean(expected), rel=1e-12)

    def test_period_one_is_wilders_true_range(self):
        """With a one-bar window both ATRs are the true range itself, so
        they agree at every present bar after the first -- where the simple
        ATR has no previous close and Wilder's uses high - low."""
        high, low, close = _walk_ohlc()
        high.iloc[[0, 50, 51]] = np.nan
        close.iloc[[120, 200]] = np.nan
        simple = atr(high, low, close, 1).to_numpy()
        wilder = wilder_atr(high, low, close, 1).to_numpy()
        present = ~(high.isna() | close.isna()).to_numpy()
        first = np.flatnonzero(present)[0]
        assert np.isnan(simple[~present]).all()
        assert np.isnan(simple[first])
        later = present.copy()
        later[first] = False
        np.testing.assert_array_equal(simple[later], wilder[later])

    def test_leading_missing_bars_read_as_a_later_start(self):
        """A ticker whose history starts late in a panel is padded with
        NaN; its ATR is the ATR of the history it has."""
        high, low, close = _walk_ohlc()
        for s in (high, low, close):
            s.iloc[:40] = np.nan
        out = atr(high, low, close, 14)
        assert out.iloc[:54].isna().all()
        pd.testing.assert_series_equal(
            out.iloc[40:],
            atr(high.iloc[40:], low.iloc[40:], close.iloc[40:], 14),
            check_exact=True,
        )


# ── Flat windows, on both backends ───────────────────────────────────────────


@pytest.fixture(params=[True, False], ids=["native", "pandas"])
def bollinger_backend(request, monkeypatch):
    """Run the test once on the native kernel and once on the pandas path."""
    if request.param:
        if not volatility.HAS_CPP:
            pytest.skip("C++ extension not built")
    else:
        monkeypatch.setattr(volatility, "HAS_CPP", False)
    return request.param


#: What an online rolling variance leaves on a window of identical prices at
#: a real price level: pandas 3.x measured a standard deviation of up to
#: 2.6e-5 there, in about half the flat windows.
_RESIDUE = 2.4e-7


def _plant_residue(monkeypatch, native):
    """Make the active backend leave rounding residue on every window, the
    way pandas 3.x does on flat ones, so the flat-window answer is tested
    against a backend that gets it wrong rather than one that happens not
    to on this install."""
    if native:
        real = volatility._cpp_core

        def residue_kernel(prices, period, num_std):
            out = np.array(real.bollinger_bands(prices, period, num_std))
            out[:, 0] += num_std * _RESIDUE
            out[:, 2] -= num_std * _RESIDUE
            return out

        monkeypatch.setattr(
            volatility,
            "_cpp_core",
            types.SimpleNamespace(bollinger_bands=residue_kernel),
        )
    else:
        rolling_cls = pd.core.window.rolling.Rolling
        real_std = rolling_cls.std
        monkeypatch.setattr(
            rolling_cls, "std", lambda self, *a, **k: real_std(self, *a, **k) + _RESIDUE
        )


def _minute_bars_with_a_halt(level=4800.25, n=390, start=200, length=40, seed=3):
    """A walk at an index-future price level with a 40-bar flat stretch."""
    rng = np.random.default_rng(seed)
    walk = np.round(level + np.cumsum(rng.normal(0, level * 5e-4, n)), 2)
    walk[start : start + length] = walk[start - 1]
    return pd.Series(walk)


class TestBollingerOnAFlatWindow:
    """
    A window of identical prices has a mean equal to the price and a
    standard deviation of exactly zero, so all three bands are the price.
    The reversion strategy compares the close with the lower and middle
    bands exactly, so rounding residue there changed which bars traded
    depending on the backend and on the pandas version.
    """

    def test_the_bands_are_the_price_whatever_the_backend_leaves(
        self, bollinger_backend, monkeypatch
    ):
        prices = _minute_bars_with_a_halt()
        _plant_residue(monkeypatch, bollinger_backend)
        bands = bollinger_bands(prices, 20, 2.0)
        flat = (prices.rolling(20).max() == prices.rolling(20).min()).to_numpy()
        # 41 identical prices (the halt and the bar before it): 22 windows.
        assert flat.sum() == 22
        for column in ("BB_Upper", "BB_Middle", "BB_Lower"):
            np.testing.assert_array_equal(
                bands[column].to_numpy()[flat], prices.to_numpy()[flat]
            )

    def test_the_reversion_signals_do_not_depend_on_the_backend(self, monkeypatch):
        if not volatility.HAS_CPP:
            pytest.skip("C++ extension not built")
        frame = pd.DataFrame({"Close": _minute_bars_with_a_halt()})
        native = _bollinger_signals(frame, 20, 2.0)
        monkeypatch.setattr(volatility, "HAS_CPP", False)
        _plant_residue(monkeypatch, native=False)
        fallback = _bollinger_signals(frame, 20, 2.0)
        pd.testing.assert_series_equal(native, fallback)

    def test_the_real_backends_agree_to_the_bit_on_flat_windows(
        self, bollinger_backend
    ):
        prices = _minute_bars_with_a_halt(level=187.23, seed=9)
        bands = bollinger_bands(prices, 20, 2.0)
        flat = flat_window_mask(prices.to_numpy(), 20)
        assert (bands["BB_Upper"].to_numpy()[flat] == prices.to_numpy()[flat]).all()
        assert (bands["BB_Lower"].to_numpy()[flat] == prices.to_numpy()[flat]).all()

    def test_windows_that_are_not_flat_are_untouched(
        self, bollinger_backend, monkeypatch
    ):
        """The null case: only a flat window is set; every other window is
        exactly what the backend computed, residue and all."""
        prices = _minute_bars_with_a_halt()
        _plant_residue(monkeypatch, bollinger_backend)
        bands = bollinger_bands(prices, 20, 2.0)
        if bollinger_backend:
            raw = volatility._cpp_core.bollinger_bands(
                prices.to_numpy(dtype=np.float64), 20, 2.0
            )
        else:
            sma = prices.rolling(20).mean()
            std = prices.rolling(20).std()
            raw = np.column_stack([sma + 2.0 * std, sma, sma - 2.0 * std])
        live = ~flat_window_mask(prices.to_numpy(), 20)
        np.testing.assert_array_equal(bands.to_numpy()[live], raw[live])


class TestFlatWindowMask:
    def test_it_is_rolling_max_equal_to_rolling_min(self):
        rng = np.random.default_rng(0)
        prices = rng.choice([10.0, 10.5, 11.0], size=(4, 300), p=[0.8, 0.1, 0.1])
        for period in (2, 3, 5):
            mask = flat_window_mask(prices, period)
            for row, got in zip(prices, mask):
                s = pd.Series(row)
                want = (s.rolling(period).max() == s.rolling(period).min()).to_numpy()
                np.testing.assert_array_equal(got, want)

    def test_a_window_holding_nan_is_never_flat(self):
        prices = np.array([5.0, np.nan, 5.0, 5.0, 5.0, np.nan, np.nan])
        np.testing.assert_array_equal(
            flat_window_mask(prices, 3), [0, 0, 0, 0, 1, 0, 0]
        )

    def test_a_one_bar_window_is_left_to_the_backend(self):
        """Its sample standard deviation is 0/0, not zero."""
        bands = np.full((4, 3), np.nan)
        out = collapse_flat_windows(np.array([1.0, 2.0, 2.0, 3.0]), bands, 1)
        assert np.isnan(out).all()


class TestBollingerLogging:
    def test_debug_logging_on_a_short_series_does_not_fall_back(
        self, bollinger_backend, caplog
    ):
        """The debug line runs outside the native try and never raises, so
        it cannot turn into a silent switch of backend."""
        with caplog.at_level(logging.DEBUG, logger=volatility.__name__):
            bands = bollinger_bands(pd.Series([4800.25, 4800.5, 4800.0]), 20)
        assert bands.isna().all(axis=None)
        messages = [r.getMessage() for r in caplog.records]
        assert not [m for m in messages if "C++ failed" in m]
        assert any(m.endswith("lower=none") for m in messages)
