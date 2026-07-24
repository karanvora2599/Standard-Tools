"""Tests for volume indicators: OBV, VWAP, MFI."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.indicators.volume import mfi, obv, vwap


@pytest.fixture(scope="module")
def vol_data():
    """Small, deterministic OHLCV data for volume indicator tests."""
    np.random.seed(99)
    n = 100
    close = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1, n)))
    spread = np.random.uniform(0.2, 1.0, n)
    high = close + pd.Series(spread)
    low = close - pd.Series(spread)
    volume = pd.Series(np.random.randint(100_000, 1_000_000, n).astype(float))
    return high, low, close, volume


class TestOBV:
    def test_output_length_matches_input(self, vol_data):
        _, _, close, volume = vol_data
        result = obv(close, volume)
        assert len(result) == len(close)

    def test_starts_at_first_bar_volume(self, vol_data):
        """First bar: price diff is NaN → sign = 0 → OBV starts at 0."""
        _, _, close, volume = vol_data
        result = obv(close, volume)
        assert float(result.iloc[0]) == 0.0

    def test_up_bar_increases_obv(self):
        """When price goes up, OBV should increase by the bar's volume."""
        close = pd.Series([100.0, 101.0, 102.0])
        volume = pd.Series([1000.0, 2000.0, 3000.0])
        result = obv(close, volume)
        assert float(result.iloc[1]) == 2000.0
        assert float(result.iloc[2]) == 5000.0

    def test_down_bar_decreases_obv(self):
        """When price goes down, OBV should decrease by the bar's volume."""
        close = pd.Series([102.0, 101.0, 100.0])
        volume = pd.Series([1000.0, 2000.0, 3000.0])
        result = obv(close, volume)
        assert float(result.iloc[1]) == -2000.0
        assert float(result.iloc[2]) == -5000.0

    def test_flat_bar_does_not_change_obv(self):
        """When price is unchanged, OBV must not change (np.sign(0) == 0)."""
        close = pd.Series([100.0, 100.0, 100.0])
        volume = pd.Series([1000.0, 2000.0, 3000.0])
        result = obv(close, volume)
        # All bars flat: sign = 0, so cumsum stays at 0
        assert (result == 0.0).all()

    def test_rising_market_has_rising_obv(self, vol_data):
        """Steadily rising price should produce a generally rising OBV."""
        n = 50
        close = pd.Series(np.linspace(100, 150, n))
        volume = pd.Series([10_000.0] * n)
        result = obv(close, volume)
        # OBV at end should be much higher than at start
        assert float(result.iloc[-1]) > float(result.iloc[10])


class TestVWAP:
    def test_output_length_matches_input(self, vol_data):
        high, low, close, volume = vol_data
        result = vwap(high, low, close, volume)
        assert len(result) == len(close)

    def test_cumulative_vwap_is_between_high_and_low(self, vol_data):
        """VWAP should always be between the min low and max high."""
        high, low, close, volume = vol_data
        result = vwap(high, low, close, volume).dropna()
        assert (result >= low.min()).all()
        assert (result <= high.max()).all()

    def test_rolling_vwap_correct_calculation(self):
        """Manual calculation of rolling VWAP should match the function."""
        n = 20
        high = pd.Series([11.0] * n)
        low = pd.Series([9.0] * n)
        close = pd.Series([10.0] * n)
        volume = pd.Series([1000.0] * n)
        # Typical price = (11+9+10)/3 = 10.0; VWAP = 10*1000 / 1000 = 10.0
        result = vwap(high, low, close, volume, period=5).dropna()
        assert result.sub(10.0).abs().max() < 1e-10

    def test_rolling_vs_cumulative_differ(self, vol_data):
        high, low, close, volume = vol_data
        rolling = vwap(high, low, close, volume, period=10)
        cumulative = vwap(high, low, close, volume)
        # They should produce different values beyond the first window
        assert not rolling.equals(cumulative)


class TestMFI:
    def test_output_length_matches_input(self, vol_data):
        high, low, close, volume = vol_data
        result = mfi(high, low, close, volume)
        assert len(result) == len(close)

    def test_bounded_0_to_100(self, vol_data):
        high, low, close, volume = vol_data
        result = mfi(high, low, close, volume, period=14).dropna()
        assert (result >= 0).all() and (result <= 100).all()

    def test_rising_price_yields_high_mfi(self):
        """Consistently rising typical price with constant volume → MFI near 100."""
        n = 50
        prices = np.linspace(100, 150, n)
        close = pd.Series(prices)
        high = close + 1.0
        low = close - 1.0
        volume = pd.Series([1_000.0] * n)
        result = mfi(high, low, close, volume, period=10).dropna()
        assert float(result.iloc[-1]) > 80

    def test_falling_price_yields_low_mfi(self):
        """Consistently falling typical price → MFI near 0."""
        n = 50
        prices = np.linspace(150, 100, n)
        close = pd.Series(prices)
        high = close + 1.0
        low = close - 1.0
        volume = pd.Series([1_000.0] * n)
        result = mfi(high, low, close, volume, period=10).dropna()
        assert float(result.iloc[-1]) < 20
