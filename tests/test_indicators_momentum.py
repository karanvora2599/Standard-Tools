"""Tests for momentum indicators: RSI, Stochastic Oscillator."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.indicators.momentum import rsi, stochastic_oscillator
from standard_quant_tools.error import ValidationError


class TestRSI:
    def test_output_length_matches_input(self, sample_close):
        result = rsi(sample_close, 14)
        assert len(result) == len(sample_close)

    def test_negative_period_raises(self, sample_close):
        """
        Regression test: the native rsi() kernel indexes result[period],
        which for a negative period wraps to a huge size_t via the
        implicit int->size_t conversion in operator[] — an out-of-bounds
        write, not a Python exception. This validation must run before
        either the C++ or Numba/Python path is reached.
        """
        with pytest.raises(ValidationError, match="period"):
            rsi(sample_close, -1)

    def test_zero_period_raises(self, sample_close):
        with pytest.raises(ValidationError, match="period"):
            rsi(sample_close, 0)

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

    def test_first_bar_is_nan(self, sample_close):
        """The very first bar is always NaN (no prior price to diff from)."""
        result = rsi(sample_close, 14)
        assert pd.isna(result.iloc[0])

    def test_sufficient_bars_have_valid_rsi(self, sample_close):
        """After enough warmup bars, RSI should produce valid values."""
        period = 14
        result = rsi(sample_close, period)
        # At least half the series should have non-NaN RSI
        assert result.notna().sum() > len(sample_close) // 2

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

    def test_custom_period_produces_different_values(self, sample_close):
        """RSI with different periods should produce different indicator values."""
        result_7 = rsi(sample_close, period=7).dropna()
        result_14 = rsi(sample_close, period=14).dropna()
        # They should have at least some different values
        common_idx = result_7.index.intersection(result_14.index)
        diffs = (result_7.loc[common_idx] - result_14.loc[common_idx]).abs()
        assert diffs.max() > 0

    def test_rsi_within_bounds_after_warmup(self, sample_close):
        """After warmup, all RSI values should remain within [0, 100]."""
        result = rsi(sample_close, 14).dropna()
        # Skip the very first few values which may be at extremes with EWM
        result_trimmed = result.iloc[5:]
        assert (result_trimmed >= 0).all() and (result_trimmed <= 100).all()

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
