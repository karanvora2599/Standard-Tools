"""Tests for volatility indicators: Bollinger Bands, ATR."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.volatility import atr, bollinger_bands, wilder_atr


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
