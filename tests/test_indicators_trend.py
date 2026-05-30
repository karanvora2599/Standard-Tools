"""Tests for trend indicators: SMA, EMA, MACD, ADX, Parabolic SAR, Williams %R."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.indicators.trend import (
    adx, ema, macd, parabolic_sar, sma, williams_r,
)


class TestSMA:
    def test_output_length_matches_input(self, sample_close):
        result = sma(sample_close, 20)
        assert len(result) == len(sample_close)

    def test_first_values_are_nan(self, sample_close):
        period = 20
        result = sma(sample_close, period)
        assert result.iloc[: period - 1].isna().all()

    def test_first_valid_value_is_exact_mean(self, sample_close):
        period = 10
        result = sma(sample_close, period)
        expected = sample_close.iloc[:period].mean()
        assert result.iloc[period - 1] == pytest.approx(expected, rel=1e-10)

    def test_period_1_is_identity(self, sample_close):
        result = sma(sample_close, 1)
        pd.testing.assert_series_equal(result, sample_close, check_names=False)

    def test_constant_series_returns_constant(self):
        s = pd.Series([5.0] * 50)
        result = sma(s, 10)
        assert result.dropna().eq(5.0).all()

    def test_index_is_preserved(self, sample_close):
        result = sma(sample_close, 5)
        pd.testing.assert_index_equal(result.index, sample_close.index)


class TestEMA:
    def test_output_length_matches_input(self, sample_close):
        assert len(ema(sample_close, 12)) == len(sample_close)

    def test_ema_responds_faster_than_sma_to_price_jump(self):
        """Immediately after a price jump, EMA should be closer to the new price than SMA."""
        period = 10
        # 20 bars at 100, then jump to 200
        base = pd.Series([100.0] * 20 + [200.0] * 20)
        # Check just a few bars AFTER the jump (before SMA catches up)
        bar_after_jump = 21  # 2nd bar at 200
        ema_val = float(ema(base, period).iloc[bar_after_jump])
        sma_val = float(sma(base, period).iloc[bar_after_jump])
        # EMA should be higher than SMA right after the jump (responding faster)
        assert ema_val > sma_val

    def test_constant_series_returns_constant(self):
        s = pd.Series([10.0] * 50)
        result = ema(s, 14)
        assert result.dropna().sub(10.0).abs().lt(1e-10).all()


class TestMACD:
    def test_returns_correct_columns(self, sample_close):
        result = macd(sample_close)
        assert set(result.columns) == {'MACD', 'Signal', 'Histogram'}

    def test_histogram_equals_macd_minus_signal(self, sample_close):
        result = macd(sample_close)
        diff = (result['MACD'] - result['Signal'] - result['Histogram']).dropna()
        assert diff.abs().max() < 1e-10

    def test_output_length_matches_input(self, sample_close):
        result = macd(sample_close)
        assert len(result) == len(sample_close)

    def test_custom_periods(self, sample_close):
        result = macd(sample_close, fast=5, slow=10, signal=3)
        assert result['MACD'].notna().any()

    def test_bullish_crossover_positive_histogram(self):
        """Strongly trending up price should produce positive MACD histogram."""
        trend = pd.Series(np.linspace(100, 200, 100))
        result = macd(trend)
        assert float(result['Histogram'].dropna().iloc[-1]) > 0


class TestADX:
    def test_returns_correct_columns(self, sample_ohlcv):
        result = adx(sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close'])
        assert set(result.columns) == {'DI_Plus', 'DI_Minus', 'ADX'}

    def test_output_length_matches_input(self, sample_ohlcv):
        result = adx(sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close'])
        assert len(result) == len(sample_ohlcv)

    def test_adx_bounded_0_to_100(self, sample_ohlcv):
        result = adx(sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close'])
        valid = result['ADX'].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_di_plus_minus_nonnegative(self, sample_ohlcv):
        result = adx(sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close'])
        assert (result['DI_Plus'].dropna() >= 0).all()
        assert (result['DI_Minus'].dropna() >= 0).all()

    def test_strong_trend_yields_high_adx(self):
        """A perfectly linear trend should produce ADX > 25."""
        n = 100
        prices = pd.Series(np.linspace(100, 200, n))
        high = prices + 1.0
        low = prices - 1.0
        result = adx(high, low, prices, period=14)
        assert float(result['ADX'].dropna().iloc[-1]) > 25

    def test_wilder_smoothing_stable(self, sample_ohlcv):
        """ADX should not have extreme spikes — Wilder smoothing keeps it stable."""
        result = adx(sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close'])
        adx_vals = result['ADX'].dropna()
        # No single bar should jump more than 30 points
        assert adx_vals.diff().abs().dropna().max() < 30


class TestParabolicSAR:
    def test_returns_correct_columns(self, sample_ohlcv):
        result = parabolic_sar(sample_ohlcv['High'], sample_ohlcv['Low'])
        assert set(result.columns) == {'SAR', 'Trend'}

    def test_output_length_matches_input(self, sample_ohlcv):
        result = parabolic_sar(sample_ohlcv['High'], sample_ohlcv['Low'])
        assert len(result) == len(sample_ohlcv)

    def test_trend_values_are_plus_or_minus_one(self, sample_ohlcv):
        result = parabolic_sar(sample_ohlcv['High'], sample_ohlcv['Low'])
        trends = result['Trend'].dropna().unique()
        assert set(trends).issubset({1.0, -1.0})

    def test_sar_below_price_on_rising_trend(self, sample_ohlcv):
        """When trend is rising (1), SAR must be below the close price."""
        close = sample_ohlcv['Close']
        result = parabolic_sar(sample_ohlcv['High'], sample_ohlcv['Low'])
        rising_mask = result['Trend'] == 1.0
        # At least 80% of rising-trend bars should have SAR below close
        rising_sar = result.loc[rising_mask, 'SAR']
        rising_close = close.loc[rising_mask]
        fraction_correct = (rising_sar.values < rising_close.values).mean()
        assert fraction_correct > 0.80

    def test_reversals_occur(self, sample_ohlcv):
        """SAR should reverse at least once in 500 bars of real price data."""
        result = parabolic_sar(sample_ohlcv['High'], sample_ohlcv['Low'])
        trend_series: pd.Series = result['Trend']  # type: ignore[assignment]
        trend_changes = int(trend_series.diff().dropna().ne(0).sum())
        assert trend_changes >= 1


class TestWilliamsR:
    def test_output_length_matches_input(self, sample_ohlcv):
        result = williams_r(sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close'])
        assert len(result) == len(sample_ohlcv)

    def test_bounded_minus_100_to_zero(self, sample_ohlcv):
        result = williams_r(sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close'])
        valid = result.dropna()
        assert (valid >= -100).all() and (valid <= 0).all()

    def test_at_period_high_equals_zero(self):
        """When close equals period high, %R should be 0."""
        n = 30
        high = pd.Series([10.0] * n)
        low = pd.Series([5.0] * n)
        close = pd.Series([10.0] * n)  # Close at high
        result = williams_r(high, low, close, period=14)
        assert result.dropna().eq(0.0).all()

    def test_at_period_low_equals_minus_100(self):
        """When close equals period low, %R should be -100."""
        n = 30
        high = pd.Series([10.0] * n)
        low = pd.Series([5.0] * n)
        close = pd.Series([5.0] * n)  # Close at low
        result = williams_r(high, low, close, period=14)
        assert result.dropna().eq(-100.0).all()
