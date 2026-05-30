"""Tests for momentum indicators: RSI, Stochastic Oscillator."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.indicators.momentum import rsi, stochastic_oscillator


class TestRSI:
    def test_output_length_matches_input(self, sample_close):
        result = rsi(sample_close, 14)
        assert len(result) == len(sample_close)

    def test_bounded_0_to_100(self, sample_close):
        result = rsi(sample_close, 14).dropna()
        assert (result >= 0).all() and (result <= 100).all()

    def test_all_gains_yields_100(self):
        """When every bar is up, RSI should reach 100."""
        s = pd.Series(np.linspace(100, 200, 50))
        result = rsi(s, 14)
        assert float(result.dropna().iloc[-1]) == pytest.approx(100.0, abs=0.01)

    def test_all_losses_yields_0(self):
        """When every bar is down, RSI should reach 0."""
        s = pd.Series(np.linspace(200, 100, 50))
        result = rsi(s, 14)
        assert float(result.dropna().iloc[-1]) == pytest.approx(0.0, abs=0.01)

    def test_nan_prefix_length(self, sample_close):
        period = 14
        result = rsi(sample_close, period)
        assert result.iloc[:period].isna().all()

    def test_uptrend_yields_rsi_above_50(self):
        """Consistently rising price should produce RSI above 50."""
        s = pd.Series(np.linspace(100, 150, 60))
        result = rsi(s, 14)
        assert float(result.dropna().iloc[-1]) > 50

    def test_downtrend_yields_rsi_below_50(self):
        """Consistently falling price should produce RSI below 50."""
        s = pd.Series(np.linspace(150, 100, 60))
        result = rsi(s, 14)
        assert float(result.dropna().iloc[-1]) < 50

    def test_custom_period(self, sample_close):
        result = rsi(sample_close, period=7)
        # Shorter period → RSI reacts faster, should have fewer NaN bars
        rsi_7_nan = rsi(sample_close, 7).isna().sum()
        rsi_14_nan = rsi(sample_close, 14).isna().sum()
        assert rsi_7_nan < rsi_14_nan

    def test_wilder_smoothing_stability(self, sample_close):
        """RSI should not jump more than 50 points between consecutive bars."""
        result = rsi(sample_close, 14).dropna()
        max_jump = result.diff().abs().dropna().max()
        assert max_jump < 50

    def test_empty_series_returns_empty(self):
        from standard_quant_tools.error import ValidationError
        with pytest.raises(ValidationError):
            rsi(pd.Series(dtype=float), 14)


class TestStochasticOscillator:
    def test_returns_correct_columns(self, sample_ohlcv):
        result = stochastic_oscillator(
            sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close']
        )
        assert set(result.columns) == {'Stoch_K', 'Stoch_D'}

    def test_output_length_matches_input(self, sample_ohlcv):
        result = stochastic_oscillator(
            sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close']
        )
        assert len(result) == len(sample_ohlcv)

    def test_k_bounded_0_to_100(self, sample_ohlcv):
        result = stochastic_oscillator(
            sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close']
        )
        k = result['Stoch_K'].dropna()
        assert (k >= 0).all() and (k <= 100).all()

    def test_d_bounded_0_to_100(self, sample_ohlcv):
        result = stochastic_oscillator(
            sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close']
        )
        d = result['Stoch_D'].dropna()
        assert (d >= 0).all() and (d <= 100).all()

    def test_close_at_high_yields_k_100(self):
        """When close equals high, %K = 100."""
        n = 30
        high = pd.Series([10.0] * n)
        low = pd.Series([5.0] * n)
        close = pd.Series([10.0] * n)
        result = stochastic_oscillator(high, low, close, k_period=5, d_period=3)
        assert result['Stoch_K'].dropna().eq(100.0).all()

    def test_close_at_low_yields_k_0(self):
        """When close equals low, %K = 0."""
        n = 30
        high = pd.Series([10.0] * n)
        low = pd.Series([5.0] * n)
        close = pd.Series([5.0] * n)
        result = stochastic_oscillator(high, low, close, k_period=5, d_period=3)
        assert result['Stoch_K'].dropna().eq(0.0).all()

    def test_d_is_rolling_mean_of_k(self, sample_ohlcv):
        """%D should equal a 3-period SMA of %K."""
        k_period, d_period = 14, 3
        result = stochastic_oscillator(
            sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close'],
            k_period=k_period, d_period=d_period,
        )
        expected_d = result['Stoch_K'].rolling(d_period).mean()
        pd.testing.assert_series_equal(
            result['Stoch_D'].dropna(), expected_d.dropna(), check_names=False, rtol=1e-10
        )
