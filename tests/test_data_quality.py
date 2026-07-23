"""Tests for data/quality.py — missing-bar/stale-price/price-jump detection on synthetic OHLCV."""

import pandas as pd
import pytest

from standard_quant_tools.data.quality import (
    detect_missing_bars, detect_stale_prices, detect_price_jumps,
)


def _ohlcv(closes, dates):
    return pd.DataFrame({
        "Open": closes, "High": closes, "Low": closes, "Close": closes,
        "Volume": [1_000_000.0] * len(closes),
    }, index=dates)


class TestDetectMissingBars:
    def test_no_gaps_in_dense_business_day_series(self):
        dates = pd.bdate_range("2023-01-02", periods=10)
        df = _ohlcv([100.0] * 10, dates)
        assert detect_missing_bars(df) == []

    def test_planted_gap_is_detected(self):
        dates = pd.bdate_range("2023-01-02", periods=10)
        # Drop the 5th business day (index 4) to create a gap.
        gapped = dates.delete(4)
        df = _ohlcv([100.0] * 9, gapped)
        gaps = detect_missing_bars(df)
        assert len(gaps) == 1
        assert gaps[0]["date"] == str(dates[4].date())

    def test_fewer_than_two_rows_returns_empty(self):
        dates = pd.bdate_range("2023-01-02", periods=1)
        df = _ohlcv([100.0], dates)
        assert detect_missing_bars(df) == []


class TestDetectStalePrices:
    def test_no_stale_run_below_threshold(self):
        dates = pd.bdate_range("2023-01-02", periods=5)
        df = _ohlcv([100.0, 101.0, 100.0, 102.0, 101.0], dates)
        assert detect_stale_prices(df, n=3) == []

    def test_planted_stale_run_is_detected(self):
        dates = pd.bdate_range("2023-01-02", periods=6)
        df = _ohlcv([100.0, 105.0, 105.0, 105.0, 105.0, 110.0], dates)
        runs = detect_stale_prices(df, n=3)
        assert len(runs) == 1
        assert runs[0]["run_length"] == 4
        assert runs[0]["price"] == pytest.approx(105.0)
        assert runs[0]["start"] == str(dates[1].date())
        assert runs[0]["end"] == str(dates[4].date())

    def test_run_shorter_than_n_not_flagged(self):
        dates = pd.bdate_range("2023-01-02", periods=5)
        df = _ohlcv([100.0, 105.0, 105.0, 110.0, 115.0], dates)
        assert detect_stale_prices(df, n=3) == []

    def test_empty_dataframe_returns_empty(self):
        df = _ohlcv([], pd.DatetimeIndex([]))
        assert detect_stale_prices(df) == []


class TestDetectPriceJumps:
    def test_no_jump_below_threshold(self):
        dates = pd.bdate_range("2023-01-02", periods=3)
        df = _ohlcv([100.0, 105.0, 108.0], dates)
        assert detect_price_jumps(df, threshold=0.15) == []

    def test_planted_jump_is_detected(self):
        dates = pd.bdate_range("2023-01-02", periods=3)
        df = _ohlcv([100.0, 100.0, 50.0], dates)  # -50% single-bar move
        jumps = detect_price_jumps(df, threshold=0.15)
        assert len(jumps) == 1
        assert jumps[0]["date"] == str(dates[2].date())
        assert jumps[0]["pct_change"] == pytest.approx(-0.5)

    def test_fewer_than_two_rows_returns_empty(self):
        dates = pd.bdate_range("2023-01-02", periods=1)
        df = _ohlcv([100.0], dates)
        assert detect_price_jumps(df) == []
