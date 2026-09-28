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

    def test_nan_in_input_raises(self, sample_close):
        bad = sample_close.copy()
        bad.iloc[10] = np.nan
        with pytest.raises(ValidationError, match="non-finite"):
            bollinger_bands(bad)


class TestWilderATR:
    def test_nan_in_input_raises(self, sample_ohlcv):
        bad_high = sample_ohlcv["High"].copy()
        bad_high.iloc[5] = np.inf
        with pytest.raises(ValidationError, match="non-finite"):
            wilder_atr(bad_high, sample_ohlcv["Low"], sample_ohlcv["Close"])


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
