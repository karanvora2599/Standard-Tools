"""Tests for cointegration analysis: Engle-Granger test, spread, half-life, z-score."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.cointegration import (
    _kalman_filter_1state,
    cointegration_test,
    compute_spread,
    half_life,
    kalman_hedge_ratio,
    spread_zscore,
)
from standard_quant_tools.error import ValidationError

# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cointegrated_pair():
    """
    Two price series sharing a common random walk.
    True: series_a = 2.0 * common_walk + noise  →  hedge_ratio ≈ 2.0.
    """
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    walk = np.cumsum(np.random.normal(0, 1, n))
    a = pd.Series(2.0 * walk + np.random.normal(0, 0.3, n), index=dates)
    b = pd.Series(walk + np.random.normal(0, 0.3, n), index=dates)
    return a, b


@pytest.fixture(scope="module")
def noncointegrated_pair():
    """Two independent random walks — should not be cointegrated."""
    np.random.seed(7)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rw1 = pd.Series(np.cumsum(np.random.normal(0, 1, n)), index=dates)
    np.random.seed(99)
    rw2 = pd.Series(np.cumsum(np.random.normal(0, 1, n)), index=dates)
    return rw1, rw2


@pytest.fixture(scope="module")
def mean_reverting_spread():
    """AR(1) spread with persistence 0.9 → true half-life ≈ 6.6 bars."""
    np.random.seed(0)
    n = 2000
    ar1 = np.zeros(n)
    for i in range(1, n):
        ar1[i] = 0.9 * ar1[i - 1] + np.random.normal(0, 1)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(ar1, index=dates)


# ── cointegration_test ─────────────────────────────────────────────────────────


class TestCointegrationTestKeys:
    def test_returns_required_keys(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert set(result.keys()) == {
            "cointegrated",
            "hedge_ratio",
            "adf_statistic",
            "p_value",
            "critical_values",
            "half_life_days",
            "n_obs",
        }

    def test_cointegrated_is_bool(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert isinstance(result["cointegrated"], bool)

    def test_critical_values_has_three_levels(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert set(result["critical_values"].keys()) == {"1%", "5%", "10%"}

    def test_critical_values_are_ordered(self, cointegrated_pair):
        """1% critical value must be the most negative (strictest threshold)."""
        a, b = cointegrated_pair
        cv = cointegration_test(a, b)["critical_values"]
        assert cv["1%"] < cv["5%"] < cv["10%"]


class TestCointegrationTestDetection:
    def test_cointegrated_pair_detected(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result["cointegrated"] is True

    def test_cointegrated_p_value_is_low(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result["p_value"] < 0.05

    def test_cointegrated_adf_below_5pct_critical(self, cointegrated_pair):
        """ADF statistic should be more negative than the 5% critical value."""
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result["adf_statistic"] < result["critical_values"]["5%"]

    def test_noncointegrated_pair_not_detected(self, noncointegrated_pair):
        rw1, rw2 = noncointegrated_pair
        result = cointegration_test(rw1, rw2)
        assert result["cointegrated"] is False

    def test_noncointegrated_p_value_is_high(self, noncointegrated_pair):
        rw1, rw2 = noncointegrated_pair
        result = cointegration_test(rw1, rw2)
        # Using 0.20 as threshold — well above 0.05 so this is reliable
        assert result["p_value"] > 0.20


class TestCointegrationTestValues:
    def test_hedge_ratio_close_to_true_value(self, cointegrated_pair):
        """True hedge ratio is 2.0; OLS should recover it within ±0.15."""
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result["hedge_ratio"] == pytest.approx(2.0, abs=0.15)

    def test_p_value_bounded(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_n_obs_matches_overlap(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result["n_obs"] == len(a)

    def test_half_life_positive(self, cointegrated_pair):
        """Cointegrated spread must have a positive finite half-life."""
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result["half_life_days"] > 0
        assert result["half_life_days"] < float("inf")

    def test_index_alignment_partial_overlap(self, cointegrated_pair):
        """Should handle extra rows in either series without raising."""
        a, b = cointegrated_pair
        extra = pd.date_range("2100-01-01", periods=5, freq="B")
        a_ext = pd.concat([a, pd.Series([0.0] * 5, index=extra)])
        result = cointegration_test(a_ext, b)
        assert result["n_obs"] == len(b)


# ── compute_spread ─────────────────────────────────────────────────────────────


class TestComputeSpread:
    def test_returns_series(self, cointegrated_pair):
        a, b = cointegrated_pair
        spread = compute_spread(a, b)
        assert isinstance(spread, pd.Series)

    def test_length_matches_common_index(self, cointegrated_pair):
        a, b = cointegrated_pair
        spread = compute_spread(a, b)
        assert len(spread) == len(a)

    def test_auto_spread_is_near_zero_mean(self, cointegrated_pair):
        """OLS residuals are zero-mean by construction."""
        a, b = cointegrated_pair
        spread = compute_spread(a, b)
        assert abs(spread.mean()) < 0.1

    def test_custom_hedge_ratio_applied(self, cointegrated_pair):
        """spread = a - ratio * b when hedge_ratio is supplied."""
        a, b = cointegrated_pair
        ratio = 2.0
        spread = compute_spread(a, b, hedge_ratio=ratio)
        expected = a.values - ratio * b.values
        np.testing.assert_allclose(spread.values, expected, rtol=1e-9)

    def test_auto_hedge_ratio_matches_cointegration_test(self, cointegrated_pair):
        """Auto-estimated hedge ratio must agree with cointegration_test."""
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        # Spread from compute_spread (OLS) vs from cointegration_test (also OLS)
        spread_auto = compute_spread(a, b)
        spread_manual = compute_spread(a, b, hedge_ratio=result["hedge_ratio"])
        # Both use OLS so they should be very close (may differ by intercept)
        assert spread_auto.std() == pytest.approx(spread_manual.std(), rel=0.05)

    def test_spread_is_stationary_for_cointegrated_pair(self, cointegrated_pair):
        """Spread of a cointegrated pair should have a short half-life."""
        a, b = cointegrated_pair
        spread = compute_spread(a, b)
        hl = half_life(spread)
        assert 0 < hl < 60  # mean-reverts within 60 bars


# ── half_life ──────────────────────────────────────────────────────────────────


class TestHalfLife:
    def test_returns_float(self, mean_reverting_spread):
        hl = half_life(mean_reverting_spread)
        assert isinstance(hl, float)

    def test_ar1_persistence_09_half_life_near_7(self, mean_reverting_spread):
        """AR(1) with persistence 0.9 → true half-life = -ln2/ln(0.9) ≈ 6.6 bars."""
        hl = half_life(mean_reverting_spread)
        assert hl == pytest.approx(6.6, abs=1.5)

    def test_non_mean_reverting_returns_inf(self):
        """Positive AR coefficient (explosive series) → half_life = inf."""
        np.random.seed(1)
        n = 300
        explosive = np.zeros(n)
        for i in range(1, n):
            explosive[i] = 1.05 * explosive[i - 1] + np.random.normal(0, 0.1)
        s = pd.Series(explosive)
        assert half_life(s) == float("inf")

    def test_faster_reversion_gives_shorter_half_life(self):
        """AR(1) with persistence 0.5 should give shorter half-life than 0.9."""
        np.random.seed(5)
        n = 2000
        dates = pd.date_range("2020-01-01", periods=n, freq="B")

        ar_fast = np.zeros(n)
        ar_slow = np.zeros(n)
        for i in range(1, n):
            ar_fast[i] = 0.5 * ar_fast[i - 1] + np.random.normal(0, 1)
            ar_slow[i] = 0.9 * ar_slow[i - 1] + np.random.normal(0, 1)

        hl_fast = half_life(pd.Series(ar_fast, index=dates))
        hl_slow = half_life(pd.Series(ar_slow, index=dates))
        assert hl_fast < hl_slow

    def test_insufficient_data_returns_inf(self):
        s = pd.Series([1.0, 2.0])
        assert half_life(s) == float("inf")


# ── spread_zscore ──────────────────────────────────────────────────────────────


class TestSpreadZscore:
    def test_static_has_zero_mean(self, cointegrated_pair):
        a, b = cointegrated_pair
        spread = compute_spread(a, b)
        z = spread_zscore(spread)
        assert abs(z.mean()) < 1e-10

    def test_static_has_unit_std(self, cointegrated_pair):
        a, b = cointegrated_pair
        spread = compute_spread(a, b)
        z = spread_zscore(spread)
        assert z.std() == pytest.approx(1.0, abs=1e-6)

    def test_rolling_nan_prefix(self, cointegrated_pair):
        a, b = cointegrated_pair
        spread = compute_spread(a, b)
        window = 20
        z = spread_zscore(spread, window=window)
        assert z.iloc[: window - 1].isna().all()

    def test_rolling_no_nan_after_warmup(self, cointegrated_pair):
        a, b = cointegrated_pair
        spread = compute_spread(a, b)
        window = 20
        z = spread_zscore(spread, window=window)
        assert not z.iloc[window - 1 :].isna().any()

    def test_rolling_values_reasonable(self, cointegrated_pair):
        """Rolling z-score of a bounded spread should stay within ±5."""
        a, b = cointegrated_pair
        spread = compute_spread(a, b)
        z = spread_zscore(spread, window=30).dropna()
        assert z.abs().max() < 5.0

    def test_constant_spread_returns_zeros(self):
        """A perfectly flat spread has zero std → z-score should be 0 everywhere."""
        spread = pd.Series([1.5] * 100)
        z = spread_zscore(spread)
        assert (z == 0.0).all()

    def test_returns_series_named_zscore(self, cointegrated_pair):
        a, b = cointegrated_pair
        spread = compute_spread(a, b)
        z = spread_zscore(spread)
        assert z.name == "zscore"


# ── kalman_hedge_ratio ───────────────────────────────────────────────────────


class TestKalmanHedgeRatioOutputStructure:
    def test_returns_expected_columns(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = kalman_hedge_ratio(a, b)
        assert set(result.columns) == {
            "Hedge_Ratio",
            "Intercept",
            "Spread",
            "Kalman_Gain",
        }

    def test_index_matches_common_index(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = kalman_hedge_ratio(a, b)
        assert result.index.equals(a.index.intersection(b.index))

    def test_include_intercept_false_zeroes_intercept(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = kalman_hedge_ratio(a, b, include_intercept=False)
        assert (result["Intercept"] == 0.0).all()

    def test_no_nans(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = kalman_hedge_ratio(a, b)
        assert not result.isna().any().any()


class TestKalmanHedgeRatioConvergence:
    def test_tiny_delta_converges_to_static_ols_hedge_ratio(self, cointegrated_pair):
        """As delta -> 0 the filter should barely adapt, landing close to
        cointegration_test's static OLS hedge_ratio on the same pair — a
        cross-check against already-verified existing code."""
        a, b = cointegrated_pair
        static = cointegration_test(a, b)
        kf = kalman_hedge_ratio(a, b, delta=1e-6, observation_noise=1.0)
        terminal_beta = kf["Hedge_Ratio"].iloc[-1]
        assert terminal_beta == pytest.approx(static["hedge_ratio"], abs=0.15)

    def test_spread_matches_hedge_ratio_and_intercept(self, cointegrated_pair):
        a, b = cointegrated_pair
        common = a.index.intersection(b.index)
        result = kalman_hedge_ratio(a, b)
        expected_spread = (
            a.loc[common] - result["Hedge_Ratio"] * b.loc[common] - result["Intercept"]
        )
        pd.testing.assert_series_equal(
            result["Spread"], expected_spread, check_names=False
        )


class TestKalmanFilter1StateHandComputed:
    def test_matches_hand_computed_two_step_recursion(self):
        y = np.array([2.0, 4.4])
        x = np.array([1.0, 2.0])
        delta, obs_noise = 0.5, 1.0
        beta_path, gain_path, innov_path = _kalman_filter_1state(y, x, delta, obs_noise)

        vw = delta / (1.0 - delta)
        p0 = 1.0e4
        beta_prev, p_prev = 0.0, p0

        r = p_prev + vw
        q = r * x[0] ** 2 + obs_noise
        e0 = y[0] - beta_prev * x[0]
        k0 = r * x[0] / q
        beta0 = beta_prev + k0 * e0
        p0_next = r - k0 * x[0] * r

        assert beta_path[0] == pytest.approx(beta0)
        assert gain_path[0] == pytest.approx(k0)
        assert innov_path[0] == pytest.approx(e0)

        r1 = p0_next + vw
        q1 = r1 * x[1] ** 2 + obs_noise
        e1 = y[1] - beta0 * x[1]
        k1 = r1 * x[1] / q1
        beta1 = beta0 + k1 * e1

        assert beta_path[1] == pytest.approx(beta1)
        assert innov_path[1] == pytest.approx(e1)


class TestKalmanHedgeRatioValidation:
    def test_delta_out_of_bounds_raises(self, cointegrated_pair):
        a, b = cointegrated_pair
        with pytest.raises(ValidationError, match="delta"):
            kalman_hedge_ratio(a, b, delta=1.5)
        with pytest.raises(ValidationError, match="delta"):
            kalman_hedge_ratio(a, b, delta=0.0)

    def test_non_positive_observation_noise_raises(self, cointegrated_pair):
        a, b = cointegrated_pair
        with pytest.raises(ValidationError, match="observation_noise"):
            kalman_hedge_ratio(a, b, observation_noise=0.0)

    def test_too_few_observations_raises(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="B")
        a = pd.Series([1.0, 2.0], index=dates)
        b = pd.Series([1.0, 2.0], index=dates)
        with pytest.raises(ValidationError, match="at least 3"):
            kalman_hedge_ratio(a, b)


@pytest.mark.benchmark
class TestKalmanHedgeRatioScale:
    def test_two_million_points_runs_quickly(self):
        import time

        rng = np.random.default_rng(7)
        n = 2_000_000
        x = np.cumsum(rng.standard_normal(n)) + 100
        y = 1.5 * x + rng.standard_normal(n)
        dates = pd.date_range("2000-01-01", periods=n, freq="min")
        a = pd.Series(y, index=dates)
        b = pd.Series(x, index=dates)

        t0 = time.time()
        result = kalman_hedge_ratio(a, b)
        elapsed = time.time() - t0
        assert elapsed < 10.0, f"2M-point Kalman filter took {elapsed:.2f}s"
        assert len(result) == n
