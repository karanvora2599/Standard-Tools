"""Tests for realized volatility estimators (Parkinson, Garman-Klass, Yang-Zhang)."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.volatility_estimators import (
    garman_klass_volatility,
    parkinson_volatility,
    yang_zhang_volatility,
)


@pytest.fixture
def two_bar_hl():
    idx = pd.date_range("2023-01-02", periods=2, freq="B")
    high = pd.Series([110.0, 120.0], index=idx)
    low = pd.Series([90.0, 80.0], index=idx)
    return high, low


class TestParkinsonVolatility:
    def test_hand_computed_value(self, two_bar_hl):
        high, low = two_bar_hl
        result = parkinson_volatility(high, low, period=2, periods_per_year=252)
        assert result.iloc[-1] == pytest.approx(3.0497930095233436, rel=1e-9)

    def test_nan_prefix_length(self):
        idx = pd.date_range("2023-01-02", periods=50, freq="B")
        high = pd.Series(np.full(50, 105.0), index=idx)
        low = pd.Series(np.full(50, 95.0), index=idx)
        result = parkinson_volatility(high, low, period=20)
        assert result.iloc[:19].isna().all()
        assert not result.iloc[19:].isna().any()

    def test_wider_range_yields_higher_vol(self):
        idx = pd.date_range("2023-01-02", periods=30, freq="B")
        narrow_high = pd.Series(np.full(30, 101.0), index=idx)
        narrow_low = pd.Series(np.full(30, 99.0), index=idx)
        wide_high = pd.Series(np.full(30, 110.0), index=idx)
        wide_low = pd.Series(np.full(30, 90.0), index=idx)
        narrow = parkinson_volatility(narrow_high, narrow_low, period=10).dropna()
        wide = parkinson_volatility(wide_high, wide_low, period=10).dropna()
        assert (wide > narrow).all()

    @pytest.mark.parametrize("bad_period", [0, 1, -5])
    def test_invalid_period_raises(self, two_bar_hl, bad_period):
        high, low = two_bar_hl
        with pytest.raises(ValidationError, match="period"):
            parkinson_volatility(high, low, period=bad_period)

    def test_empty_series_raises(self):
        empty = pd.Series([], dtype=float)
        with pytest.raises(ValidationError):
            parkinson_volatility(empty, empty, period=5)


class TestGarmanKlassVolatility:
    def test_zero_when_no_intrabar_or_overnight_movement(self):
        """Flat O=H=L=C every bar -> both terms are ln(1)=0 -> zero vol."""
        idx = pd.date_range("2023-01-02", periods=30, freq="B")
        flat = pd.Series(np.full(30, 100.0), index=idx)
        result = garman_klass_volatility(flat, flat, flat, flat, period=10)
        np.testing.assert_allclose(result.dropna().to_numpy(), 0.0, atol=1e-10)

    def test_nan_prefix_length(self):
        idx = pd.date_range("2023-01-02", periods=50, freq="B")
        rng = np.random.default_rng(1)
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, 50))
        open_ = pd.Series(close * 1.001, index=idx)
        high = pd.Series(close * 1.01, index=idx)
        low = pd.Series(close * 0.99, index=idx)
        close = pd.Series(close, index=idx)
        result = garman_klass_volatility(open_, high, low, close, period=20)
        assert result.iloc[:19].isna().all()
        assert not result.iloc[19:].isna().any()

    def test_invalid_period_raises(self, two_bar_hl):
        high, low = two_bar_hl
        with pytest.raises(ValidationError, match="period"):
            garman_klass_volatility(high, high, low, low, period=1)


class TestYangZhangVolatility:
    def test_zero_when_flat(self):
        idx = pd.date_range("2023-01-02", periods=30, freq="B")
        flat = pd.Series(np.full(30, 100.0), index=idx)
        result = yang_zhang_volatility(flat, flat, flat, flat, period=10)
        np.testing.assert_allclose(result.dropna().to_numpy(), 0.0, atol=1e-10)

    def test_captures_overnight_gap_risk_missed_by_parkinson(self):
        """
        A series with large overnight gaps but tiny intraday ranges: Parkinson
        (High/Low only) should report near-zero vol, while Yang-Zhang (which
        includes the overnight close-to-open term) should report much higher
        vol -- this is the whole reason Yang-Zhang exists.
        """
        idx = pd.date_range("2023-01-02", periods=60, freq="B")
        rng = np.random.default_rng(2)
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.03, 60))
        close_s = pd.Series(close, index=idx)
        # Next day's open gaps hard off the prior close, but the bar's own
        # intraday range (High/Low around Open) is razor-thin.
        open_ = close_s.shift(1).fillna(close_s.iloc[0])
        high = open_ * 1.0001
        low = open_ * 0.9999

        parkinson = parkinson_volatility(high, low, period=20).dropna().iloc[-1]
        yz = (
            yang_zhang_volatility(open_, high, low, close_s, period=20)
            .dropna()
            .iloc[-1]
        )
        assert yz > parkinson * 5

    def test_nan_prefix_length(self):
        idx = pd.date_range("2023-01-02", periods=50, freq="B")
        rng = np.random.default_rng(3)
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, 50))
        open_ = pd.Series(close * 1.001, index=idx)
        high = pd.Series(close * 1.01, index=idx)
        low = pd.Series(close * 0.99, index=idx)
        close = pd.Series(close, index=idx)
        result = yang_zhang_volatility(open_, high, low, close, period=20)
        # One bar longer than Parkinson/Garman-Klass's (period-1) prefix: the
        # overnight term needs the PRIOR close (close.shift(1)), which is
        # itself NaN at bar 0, so the first rolling(20) window that avoids
        # touching that NaN starts one bar later than a window with no shift.
        assert result.iloc[:20].isna().all()
        assert not result.iloc[20:].isna().any()

    def test_invalid_period_raises(self, two_bar_hl):
        high, low = two_bar_hl
        with pytest.raises(ValidationError, match="period"):
            yang_zhang_volatility(high, high, low, low, period=0)
