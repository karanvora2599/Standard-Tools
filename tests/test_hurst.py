"""Tests for Hurst exponent estimation: hurst_exponent and rolling_hurst."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.hurst import hurst_exponent, rolling_hurst


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_returns(seed, n, phi):
    """AR(1) return series with given persistence phi."""
    np.random.seed(seed)
    innov = np.random.normal(0, 1, n)
    ret = np.zeros(n)
    for i in range(1, n):
        ret[i] = phi * ret[i - 1] + innov[i]
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(ret, index=dates)


@pytest.fixture(scope="module")
def iid_returns():
    """Pure iid returns → random walk in prices → H ≈ 0.5."""
    np.random.seed(42)
    n = 2000
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(np.random.normal(0, 1, n), index=dates)


@pytest.fixture(scope="module")
def trending_returns():
    """AR(1) phi=+0.4 → positively autocorrelated → H > 0.55."""
    return _make_returns(seed=42, n=2000, phi=0.4)


@pytest.fixture(scope="module")
def mean_reverting_returns():
    """AR(1) phi=-0.4 → negatively autocorrelated → H < 0.45."""
    return _make_returns(seed=42, n=2000, phi=-0.4)


# ── hurst_exponent — output structure ─────────────────────────────────────────

class TestHurstExponentKeys:
    def test_returns_required_keys(self, iid_returns):
        result = hurst_exponent(iid_returns)
        assert set(result.keys()) == {
            "hurst", "regime", "fit_r_squared", "method", "n_obs"
        }

    def test_hurst_is_float(self, iid_returns):
        assert isinstance(hurst_exponent(iid_returns)["hurst"], float)

    def test_regime_is_string(self, iid_returns):
        assert isinstance(hurst_exponent(iid_returns)["regime"], str)

    def test_method_recorded_correctly(self, iid_returns):
        assert hurst_exponent(iid_returns, method="dfa")["method"] == "dfa"
        assert hurst_exponent(iid_returns, method="rs")["method"] == "rs"

    def test_n_obs_matches_input_length(self, iid_returns):
        result = hurst_exponent(iid_returns)
        assert result["n_obs"] == len(iid_returns)


# ── hurst_exponent — regime detection (DFA) ───────────────────────────────────

class TestHurstDFARegimes:
    def test_iid_returns_near_half(self, iid_returns):
        """iid returns must produce H close to 0.5."""
        h = hurst_exponent(iid_returns, method="dfa")["hurst"]
        assert 0.40 < h < 0.60

    def test_iid_classified_random_walk(self, iid_returns):
        assert hurst_exponent(iid_returns, method="dfa")["regime"] == "random_walk"

    def test_trending_h_above_threshold(self, trending_returns):
        """AR(+0.4) must produce H > 0.55."""
        h = hurst_exponent(trending_returns, method="dfa")["hurst"]
        assert h > 0.55

    def test_trending_classified_correctly(self, trending_returns):
        assert hurst_exponent(trending_returns, method="dfa")["regime"] == "trending"

    def test_mean_reverting_h_below_threshold(self, mean_reverting_returns):
        """AR(-0.4) must produce H < 0.45."""
        h = hurst_exponent(mean_reverting_returns, method="dfa")["hurst"]
        assert h < 0.45

    def test_mean_reverting_classified_correctly(self, mean_reverting_returns):
        result = hurst_exponent(mean_reverting_returns, method="dfa")
        assert result["regime"] == "mean_reverting"

    def test_trending_h_greater_than_mean_reverting(
        self, trending_returns, mean_reverting_returns
    ):
        h_trend = hurst_exponent(trending_returns)["hurst"]
        h_mr = hurst_exponent(mean_reverting_returns)["hurst"]
        assert h_trend > h_mr

    def test_fit_r_squared_high_for_clean_process(self, iid_returns):
        """A clean process should have a good power-law fit (R² > 0.9)."""
        r2 = hurst_exponent(iid_returns, method="dfa")["fit_r_squared"]
        assert r2 > 0.90

    def test_fit_r_squared_bounded_0_to_1(self, iid_returns):
        r2 = hurst_exponent(iid_returns, method="dfa")["fit_r_squared"]
        assert 0.0 <= r2 <= 1.0

    def test_hurst_bounded_0_to_1(self, iid_returns, trending_returns, mean_reverting_returns):
        for s in [iid_returns, trending_returns, mean_reverting_returns]:
            h = hurst_exponent(s)["hurst"]
            assert 0.0 <= h <= 1.5  # clipped at 1.5 in implementation


# ── hurst_exponent — RS method ────────────────────────────────────────────────

class TestHurstRSMethod:
    def test_rs_trending_above_dfa_for_same_series(self, trending_returns):
        """R/S is biased upward; its trending estimate should be >= DFA estimate."""
        h_dfa = hurst_exponent(trending_returns, method="dfa")["hurst"]
        h_rs = hurst_exponent(trending_returns, method="rs")["hurst"]
        # R/S bias → R/S estimate at least as high as DFA
        assert h_rs >= h_dfa - 0.05  # small tolerance for sampling noise

    def test_rs_trending_classified_correctly(self, trending_returns):
        assert hurst_exponent(trending_returns, method="rs")["regime"] == "trending"

    def test_rs_returns_valid_dict(self, iid_returns):
        result = hurst_exponent(iid_returns, method="rs")
        assert not np.isnan(result["hurst"])
        assert result["method"] == "rs"


# ── hurst_exponent — edge cases ───────────────────────────────────────────────

class TestHurstEdgeCases:
    def test_insufficient_data_returns_nan(self):
        """Too few observations to form sub-windows → nan result."""
        tiny = pd.Series(np.random.normal(0, 1, 5))
        result = hurst_exponent(tiny, min_window=10)
        assert np.isnan(result["hurst"])
        assert result["regime"] == "unknown"

    def test_nan_values_dropped_before_computation(self, iid_returns):
        noisy = iid_returns.copy()
        noisy.iloc[::10] = np.nan  # every 10th bar is NaN
        result = hurst_exponent(noisy)
        assert not np.isnan(result["hurst"])
        assert result["n_obs"] == len(noisy.dropna())

    def test_n_components_arg_max_window(self, iid_returns):
        """Explicit max_window should be respected."""
        r1 = hurst_exponent(iid_returns, max_window=50)
        r2 = hurst_exponent(iid_returns, max_window=200)
        # Both should produce valid (non-nan) Hurst values
        assert not np.isnan(r1["hurst"])
        assert not np.isnan(r2["hurst"])

    def test_constant_series_returns_nan(self):
        """A zero-variance series has no R/S scaling → nan."""
        flat = pd.Series(np.ones(500))
        result = hurst_exponent(flat)
        assert np.isnan(result["hurst"])


# ── rolling_hurst ──────────────────────────────────────────────────────────────

class TestRollingHurst:
    def test_returns_series(self, iid_returns):
        assert isinstance(rolling_hurst(iid_returns, window=200), pd.Series)

    def test_output_length_matches_input(self, iid_returns):
        result = rolling_hurst(iid_returns, window=200)
        assert len(result) == len(iid_returns)

    def test_nan_prefix_length(self, iid_returns):
        window = 200
        result = rolling_hurst(iid_returns, window=window)
        assert result.iloc[: window - 1].isna().all()

    def test_no_nan_after_warmup(self, iid_returns):
        window = 200
        result = rolling_hurst(iid_returns, window=window)
        assert not result.iloc[window - 1:].isna().any()

    def test_rolling_values_in_valid_range(self, iid_returns):
        result = rolling_hurst(iid_returns, window=200).dropna()
        assert (result >= 0.0).all()
        assert (result <= 1.5).all()

    def test_step_reduces_computed_points(self, iid_returns):
        """With step=5, only every 5th bar should be non-NaN after warmup."""
        window = 200
        step = 5
        result = rolling_hurst(iid_returns, window=window, step=step)
        valid_mask = ~result.isna()
        valid_indices = np.where(valid_mask)[0]
        # All valid indices should be multiples of step (relative to start)
        if len(valid_indices) > 1:
            diffs = np.diff(valid_indices)
            assert (diffs == step).all()

    def test_series_named_hurst(self, iid_returns):
        result = rolling_hurst(iid_returns, window=200)
        assert result.name == "hurst"

    def test_regime_shift_detected(self):
        """
        A series that transitions from AR(+0.4) to AR(-0.4) midway should
        show higher rolling H in the first half and lower in the second half.
        """
        np.random.seed(7)
        n = 1000
        innov = np.random.normal(0, 1, n)
        ret = np.zeros(n)
        half = n // 2
        for i in range(1, half):
            ret[i] = 0.4 * ret[i - 1] + innov[i]     # trending first half
        for i in range(half, n):
            ret[i] = -0.4 * ret[i - 1] + innov[i]    # mean-reverting second half

        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        s = pd.Series(ret, index=dates)
        rolling = rolling_hurst(s, window=200, step=10)

        # Compare means in second and fourth quarters (avoid transition zone)
        q2_end = 3 * n // 4
        q2_start = n // 4
        mean_early = rolling.iloc[q2_start: half].dropna().mean()
        mean_late = rolling.iloc[q2_end:].dropna().mean()

        assert mean_early > mean_late, (
            f"Expected early H ({mean_early:.3f}) > late H ({mean_late:.3f})"
        )
