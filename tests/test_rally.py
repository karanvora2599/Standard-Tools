"""Tests for rally detection: analysis.rally.detect_rally."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.rally import detect_rally
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.trend import adx

# ── Shared fixtures ────────────────────────────────────────────────────────────


def _make_ohlcv(close: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(close)
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}, index=dates
    )


_N = 400  # >= default zscore_window(252) + lookback(20)


@pytest.fixture(scope="module")
def rising_ohlcv():
    """Steady mild uptrend, then a sharp accelerating rally in the final
    30 bars -- the case detect_rally is specifically meant to catch."""
    rng = np.random.default_rng(1)
    base = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, _N - 30))
    rally_leg = base[-1] * np.cumprod(1 + rng.normal(0.012, 0.008, 30))
    close = np.concatenate([base, rally_leg])
    return _make_ohlcv(close, seed=1)


@pytest.fixture(scope="module")
def flat_ohlcv():
    """Pure random walk, zero drift -- must not register as a rally."""
    rng = np.random.default_rng(2)
    close = 100 * np.cumprod(1 + rng.normal(0.0, 0.01, _N))
    return _make_ohlcv(close, seed=2)


@pytest.fixture(scope="module")
def falling_ohlcv():
    """Steady, genuinely persistent downtrend -- must not register as a
    rally even though it can trip the trend-strength/regime signals,
    proving the direction check actually gates the result."""
    rng = np.random.default_rng(3)
    close = 100 * np.cumprod(1 + rng.normal(-0.006, 0.01, _N))
    return _make_ohlcv(close, seed=3)


# ── Core detection behavior ───────────────────────────────────────────────────


class TestDetectRallyCoreCases:
    def test_rising_series_is_detected_as_rally(self, rising_ohlcv):
        result = detect_rally(rising_ohlcv)
        assert result["is_rally"] is True
        assert result["trend_direction"] == "bullish"
        assert result["rally_score"] >= 0.6

    def test_flat_series_is_not_a_rally(self, flat_ohlcv):
        result = detect_rally(flat_ohlcv)
        assert result["is_rally"] is False

    def test_falling_series_is_not_a_rally(self, falling_ohlcv):
        """A genuinely persistent downtrend can still trip strong_trend
        and trending_regime, but must not be reported as a rally --
        bullish_direction is False, capping the achievable score."""
        result = detect_rally(falling_ohlcv)
        assert result["is_rally"] is False
        assert result["trend_direction"] == "bearish"

    def test_rally_score_is_fraction_of_five_signals(self, rising_ohlcv):
        result = detect_rally(rising_ohlcv)
        assert result["rally_score"] in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}


# ── Output structure ───────────────────────────────────────────────────────────


class TestOutputStructure:
    def test_has_required_keys(self, rising_ohlcv):
        result = detect_rally(rising_ohlcv)
        expected = {
            "is_rally",
            "rally_score",
            "trailing_return_pct",
            "return_zscore",
            "adx",
            "di_plus",
            "di_minus",
            "trend_direction",
            "hurst",
            "regime",
            "is_new_high",
            "adx_threshold_used",
            "auto_tuned",
            "n_obs",
        }
        assert expected <= set(result.keys())

    def test_trend_direction_is_valid_string(self, rising_ohlcv, flat_ohlcv):
        for result in (detect_rally(rising_ohlcv), detect_rally(flat_ohlcv)):
            assert result["trend_direction"] in {"bullish", "bearish", "neutral"}

    def test_adx_is_nonnegative(self, rising_ohlcv):
        assert detect_rally(rising_ohlcv)["adx"] >= 0.0

    def test_n_obs_matches_input_length(self, rising_ohlcv):
        assert detect_rally(rising_ohlcv)["n_obs"] == len(rising_ohlcv)


# ── Breakout flag: no look-ahead ──────────────────────────────────────────────


class TestNewHighBreakoutNoLookahead:
    def test_matches_hand_computed_shifted_rolling_max(self, rising_ohlcv):
        """is_new_high must compare Close against the PRIOR breakout_period
        bars' High (via .shift(1)), excluding today's own bar -- the same
        look-ahead-safe convention backtest/strategies.py's Donchian
        breakout uses, not the tautology "today's close >= today's own
        rolling max"."""
        result = detect_rally(rising_ohlcv, breakout_period=20)
        breakout_high = rising_ohlcv["High"].rolling(20).max().shift(1)
        expected = bool(rising_ohlcv["Close"].iloc[-1] > breakout_high.iloc[-1])
        assert result["is_new_high"] == expected

    def test_today_bar_alone_cannot_trigger_its_own_breakout(self):
        """A single-bar spike on the very last bar, with High equal to
        Close, must not be able to satisfy its own breakout comparison --
        that would be the tautology bug this convention guards against."""
        rng = np.random.default_rng(5)
        close = 100 * np.cumprod(1 + rng.normal(0.0, 0.005, _N))
        df = _make_ohlcv(close, seed=5)
        # Spike only the very last bar's Close/High together, far above
        # its own prior history.
        df = df.copy()
        df.iloc[-1, df.columns.get_loc("Close")] = df["Close"].iloc[-2] * 1.5
        df.iloc[-1, df.columns.get_loc("High")] = df["Close"].iloc[-1]
        result = detect_rally(df)
        breakout_high = df["High"].rolling(20).max().shift(1)
        # The shifted comparison window excludes the spike bar itself, so
        # is_new_high should be True here (it broke the PRIOR high) --
        # confirm it agrees with the hand-computed shifted value, not a
        # same-bar comparison against its own new High.
        expected = bool(df["Close"].iloc[-1] > breakout_high.iloc[-1])
        assert result["is_new_high"] == expected


# ── Auto-tuned ADX threshold (option 2: no per-asset manual tuning needed) ────


class TestAutoTuneAdxThreshold:
    def test_default_behavior_unchanged_when_auto_tune_off(self, rising_ohlcv):
        """auto_tune_adx_threshold defaults to False -- must be byte-for-byte
        the same result as before this feature existed."""
        result = detect_rally(rising_ohlcv)
        assert result["auto_tuned"] is False
        assert result["adx_threshold_used"] == 25.0

    def test_auto_tuned_threshold_matches_hand_computed_percentile(
        self, rising_ohlcv
    ):
        result = detect_rally(
            rising_ohlcv, auto_tune_adx_threshold=True, auto_tune_percentile=60.0
        )
        adx_df = adx(
            rising_ohlcv["High"], rising_ohlcv["Low"], rising_ohlcv["Close"], period=14
        )
        expected = float(np.percentile(adx_df["ADX"].dropna().to_numpy(), 60.0))
        assert result["adx_threshold_used"] == pytest.approx(expected)
        assert result["auto_tuned"] is True

    def test_higher_percentile_gives_higher_threshold(self, rising_ohlcv):
        low_pct = detect_rally(
            rising_ohlcv, auto_tune_adx_threshold=True, auto_tune_percentile=40.0
        )
        high_pct = detect_rally(
            rising_ohlcv, auto_tune_adx_threshold=True, auto_tune_percentile=90.0
        )
        assert high_pct["adx_threshold_used"] > low_pct["adx_threshold_used"]

    def test_percentile_ignored_when_auto_tune_off(self, rising_ohlcv):
        """A caller-supplied auto_tune_percentile must have zero effect
        unless auto_tune_adx_threshold=True -- including not raising for an
        otherwise-invalid value, since it's simply unused in that path."""
        result = detect_rally(
            rising_ohlcv, auto_tune_adx_threshold=False, auto_tune_percentile=999.0
        )
        assert result["adx_threshold_used"] == 25.0

    def test_out_of_range_percentile_raises_only_when_auto_tuning(self, rising_ohlcv):
        for bad in (0.0, 100.0, -5.0, 150.0):
            with pytest.raises(ValidationError, match="auto_tune_percentile"):
                detect_rally(
                    rising_ohlcv,
                    auto_tune_adx_threshold=True,
                    auto_tune_percentile=bad,
                )


# ── Validation ─────────────────────────────────────────────────────────────────


class TestValidation:
    def test_missing_columns_raises(self, rising_ohlcv):
        with pytest.raises(ValidationError, match="missing required columns"):
            detect_rally(rising_ohlcv.drop(columns=["High"]))

    def test_insufficient_observations_raises(self):
        rng = np.random.default_rng(9)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 50))
        df = _make_ohlcv(close, seed=9)
        with pytest.raises(ValidationError, match="at least 272"):
            detect_rally(df)

    def test_default_min_obs_is_zscore_window_plus_lookback(self):
        rng = np.random.default_rng(9)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, 100))
        df = _make_ohlcv(close, seed=9)
        with pytest.raises(ValidationError, match="zscore_window=252"):
            detect_rally(df, lookback=20, zscore_window=252)
