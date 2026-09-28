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
    scan_cointegrated_pairs,
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
            "half_life_mean_reverting",
            "half_life_t_statistic",
            "half_life_critical_value",
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

    def test_nan_in_series_raises(self, cointegrated_pair):
        a, b = cointegrated_pair
        bad = a.copy()
        bad.iloc[10] = np.nan
        with pytest.raises(ValidationError, match="non-finite"):
            cointegration_test(bad, b)


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

    def test_inf_in_series_raises(self, cointegrated_pair):
        a, b = cointegrated_pair
        bad = b.copy()
        bad.iloc[7] = np.inf
        with pytest.raises(ValidationError, match="non-finite"):
            compute_spread(a, bad)


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

    def test_inf_in_spread_raises(self, mean_reverting_spread):
        bad = mean_reverting_spread.copy()
        bad.iloc[20] = np.inf
        with pytest.raises(ValidationError, match="non-finite"):
            half_life(bad)


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

    def test_rolling_constant_spread_window_yields_nan_not_inf(self):
        """A rolling window with zero variance (e.g. a flat stretch of the
        spread) used to divide by zero -- must produce NaN for that window,
        not inf/-inf that could silently poison downstream math. Unlike the
        static (window=None) branch, NaN is used instead of a literal 0.0
        since a rolling 0.0 would be indistinguishable from a legitimate
        zero z-score mid-series."""
        spread = pd.Series([1.5] * 100)
        window = 20
        z = spread_zscore(spread, window=window)
        assert z.iloc[window - 1 :].isna().all()
        assert not np.isinf(z).any()


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


class TestADegeneratePairHasNoQuestionToAnswer:
    """The two backends returned opposite verdicts on the same input.

    On an exactly affine pair the residual is identically zero, and:

        native (C++)          p=0.2593  adf=-2.546   cointegrated=False
        statsmodels fallback  p=0.0     adf=-inf     cointegrated=True

    Neither is defensible. An ADF statistic asks whether a series reverts to
    its mean; a series that IS its mean has no such question, and -2.546 and
    -inf are both inventions. statsmodels knows -- it emits
    `CollinearityWarning: ... not reliable in this case` and answers anyway.

    So this is refused rather than answered, which is what the package does
    elsewhere for the same shape. An affine pair is not exotic: a dual
    listing, an ETF against its sole holding, or the same column twice in a
    screening universe.
    """

    @staticmethod
    def _series(n=400, seed=7):
        rng = np.random.default_rng(seed)
        index = pd.bdate_range("2021-01-04", periods=n)
        return pd.Series(100 + np.cumsum(rng.normal(0, 0.8, n)), index=index)

    def test_an_exactly_affine_pair_is_refused(self):
        a = self._series()
        with pytest.raises(ValidationError, match="exact linear function"):
            cointegration_test(a, 0.9 * a + 5.0)

    def test_a_constant_series_is_refused(self):
        a = self._series()
        flat = pd.Series(np.full(len(a), 50.0), index=a.index)
        with pytest.raises(ValidationError, match="series_b is constant"):
            cointegration_test(a, flat)

    def test_a_real_relationship_still_answers(self):
        """The guard must not catch a pair with a genuine spread."""
        a = self._series()
        rng = np.random.default_rng(3)
        b = 0.9 * a + 5.0 + rng.normal(0, 0.5, len(a))
        result = cointegration_test(a, b)
        assert result["cointegrated"] is True
        assert result["p_value"] < 0.05

    def test_an_independent_pair_still_answers(self):
        a = self._series(seed=7)
        b = self._series(seed=11)
        result = cointegration_test(a, b)
        assert result["cointegrated"] is False
        assert 0.0 <= result["p_value"] <= 1.0

    def test_a_scan_flags_the_pair_and_keeps_the_rest(self):
        """A single test refuses; a scan must not. One dual listing in a
        universe cannot be allowed to cost the other 4,949 pairs -- but the
        two must agree about WHICH pairs are answerable, which is why they
        share one predicate."""
        base = self._series()
        rng = np.random.default_rng(5)
        frame = pd.DataFrame(
            {
                "AAA": base,
                "BBB": 0.9 * base + 5.0,
                "CCC": 0.9 * base + 5.0 + rng.normal(0, 0.5, len(base)),
                "DDD": self._series(seed=11),
            }
        )
        out = scan_cointegrated_pairs(frame)
        assert len(out) == 6, "pairs were lost"
        flagged = out.xs("AAA", level=0).loc["BBB"]
        assert np.isnan(flagged["p_value"])
        assert bool(flagged["cointegrated"]) is False
        assert int(out["p_value"].notna().sum()) == 5

    def test_the_two_backends_agree_about_what_is_answerable(self, monkeypatch):
        """The property the guard exists to restore. Whether a pair can be
        tested must not depend on whether the extension is built."""
        import standard_quant_tools.analysis.cointegration as coint_module

        a = self._series()
        affine = 0.9 * a + 5.0

        with pytest.raises(ValidationError):
            cointegration_test(a, affine)
        monkeypatch.setattr(coint_module, "HAS_CPP", False)
        with pytest.raises(ValidationError):
            cointegration_test(a, affine)


# ── the minimum sample ─────────────────────────────────────────────────────────


class TestMinimumObservations:
    """
    cointegration_test and scan_cointegrated_pairs share one floor of 20
    aligned observations. Below it the single test answered p=nan with
    cointegrated=False at n=0, p=0.85 at n=8, and at n=1 refused with
    "series_b is constant" -- true, and not the reason.
    """

    @staticmethod
    def _pair(n, seed=3):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        walk = np.cumsum(rng.normal(0, 1, n)) + 50.0
        return (
            pd.Series(1.5 * walk + rng.normal(0, 0.5, n), index=dates),
            pd.Series(walk, index=dates),
        )

    @pytest.mark.parametrize("n", [0, 1, 8, 19])
    def test_fewer_than_twenty_are_refused(self, n):
        a, b = self._pair(max(n, 1))
        with pytest.raises(ValidationError, match="at least 20"):
            cointegration_test(a.iloc[:n], b.iloc[:n])

    def test_twenty_answer(self):
        a, b = self._pair(20)
        result = cointegration_test(a, b)
        assert result["n_obs"] == 20
        assert 0.0 <= result["p_value"] <= 1.0

    def test_the_scan_shares_the_floor(self):
        a, b = self._pair(60)
        frame = pd.DataFrame({"A": a, "B": b})
        with pytest.raises(ValidationError, match="at least 20"):
            scan_cointegrated_pairs(frame.iloc[:19])
        assert len(scan_cointegrated_pairs(frame.iloc[:20])) == 1


# ── half_life_statistics ───────────────────────────────────────────────────────


class TestHalfLifeStatistics:
    """
    half_life() returns a finite number on most random walks: the fitted
    AR(1) coefficient is negative about half the time and a small negative
    coefficient is a long but finite half-life. Measured on 1000 random
    walks of 250 observations, 95.5% finite and 84% inside a 5-126 screen.
    The Dickey-Fuller t-statistic of the same regression is the gate.
    """

    def test_random_walks_are_rarely_called_mean_reverting(self):
        from standard_quant_tools.analysis.cointegration import half_life_statistics

        rng = np.random.default_rng(40)
        flagged = finite = 0
        trials = 200
        for _ in range(trials):
            walk = pd.Series(np.cumsum(rng.normal(0, 1, 250)))
            stats = half_life_statistics(walk)
            flagged += stats["mean_reverting"]
            finite += np.isfinite(stats["half_life"])
        assert finite > trials * 0.5  # what half_life alone would have said
        assert flagged <= trials * 0.08

    def test_a_planted_ar1_is_flagged_with_its_half_life(self):
        """phi = 0.9: a shock halves after log(0.5)/log(0.9) = 6.58 bars."""
        from standard_quant_tools.analysis.cointegration import half_life_statistics

        rng = np.random.default_rng(41)
        values = np.zeros(5000)
        for i in range(1, values.size):
            values[i] = 0.9 * values[i - 1] + rng.normal()
        stats = half_life_statistics(pd.Series(values))
        assert stats["mean_reverting"] is True
        assert stats["half_life"] == pytest.approx(6.58, rel=0.2)
        assert stats["t_statistic"] < stats["critical_value"]

    def test_a_fitted_residual_faces_the_stricter_critical_value(self):
        from standard_quant_tools.analysis.cointegration import half_life_statistics

        spread = pd.Series(np.random.default_rng(42).normal(0, 1, 250))
        plain = half_life_statistics(spread)
        fitted = half_life_statistics(spread, fitted_residual=True)
        assert fitted["critical_value"] < plain["critical_value"] < -2.8

    def test_a_flat_spread_is_not_mean_reverting(self):
        from standard_quant_tools.analysis.cointegration import half_life_statistics

        stats = half_life_statistics(pd.Series([12.3456] * 100))
        assert stats["mean_reverting"] is False
        assert stats["half_life"] == float("inf")

    def test_cointegration_test_reports_the_gate(self, cointegrated_pair):
        a, b = cointegrated_pair
        result = cointegration_test(a, b)
        assert result["half_life_mean_reverting"] is True
        assert result["half_life_t_statistic"] < result["half_life_critical_value"]


# ── the pair scan: both orders and many tests ──────────────────────────────────


class TestScanDirectionAndMultiplicity:
    @staticmethod
    def _walks(k, n=500, seed=50):
        rng = np.random.default_rng(seed)
        return {f"R{i:02d}": np.cumsum(rng.normal(0, 1, n)) + 100.0 for i in range(k)}

    def test_the_new_columns_do_not_depend_on_column_order(self):
        """
        Engle-Granger is not symmetric: swapping the columns flipped 65 of
        276 verdicts on random walks. The pair-level columns are the same
        whichever series comes first.
        """
        frame = pd.DataFrame(self._walks(8))
        forward = scan_cointegrated_pairs(frame)
        backward = scan_cointegrated_pairs(frame[frame.columns[::-1]])
        for (a, b), row in forward.iterrows():
            other = backward.loc[(b, a)]
            assert row["p_value_both"] == pytest.approx(other["p_value_both"])
            assert row["p_value_bh"] == pytest.approx(other["p_value_bh"])
            assert row["p_value"] == pytest.approx(other["p_value_reverse"])
            assert bool(row["cointegrated_fdr"]) is bool(other["cointegrated_fdr"])
            assert bool(row["direction_consistent"]) is bool(
                other["direction_consistent"]
            )

    def test_a_planted_pair_survives_the_false_discovery_control(self):
        data = self._walks(20)
        rng = np.random.default_rng(51)
        data["P"] = 1.3 * data["R00"] + rng.normal(0, 1.0, 500)
        out = scan_cointegrated_pairs(pd.DataFrame(data))
        survivors = list(out.index[out["cointegrated_fdr"]])
        assert ("R00", "P") in survivors
        assert len(survivors) <= 2

    def test_random_walks_alone_leave_at_most_one_survivor(self):
        """
        The null case. 190 unrelated pairs clear 5% about ten times by
        chance in one order; the adjusted screen keeps essentially none.
        """
        out = scan_cointegrated_pairs(pd.DataFrame(self._walks(20, seed=52)))
        assert int(out["cointegrated_fdr"].sum()) <= 1
        assert (out["p_value_bh"] >= out["p_value_both"] - 1e-12).all()

    def test_a_degenerate_pair_is_not_counted_as_a_test(self):
        data = self._walks(4)
        data["DUP"] = 0.9 * data["R00"] + 5.0
        out = scan_cointegrated_pairs(pd.DataFrame(data))
        row = out.loc[("R00", "DUP")]
        assert np.isnan(row["p_value_both"]) and np.isnan(row["p_value_bh"])
        assert bool(row["cointegrated_fdr"]) is False

    def test_benjamini_hochberg_matches_the_textbook_step_up(self):
        from standard_quant_tools.analysis.cointegration import benjamini_hochberg

        p = [0.01, 0.04, 0.03, 0.20, float("nan")]
        # m = 4 answerable. Sorted 0.01, 0.03, 0.04, 0.20 scale to 0.04,
        # 0.06, 0.0533, 0.20; the step-up minimum makes the middle two
        # 0.0533 each.
        adjusted = benjamini_hochberg(p)
        np.testing.assert_allclose(
            adjusted[:4], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.20], rtol=1e-12
        )
        assert np.isnan(adjusted[4])


class TestFlatSpreadZscore:
    def test_a_spread_flat_at_an_awkward_level_is_zero_not_residue(self):
        """
        A spread flat at 12.3456 has a standard deviation of about 7e-15,
        not 0, so the exact `sigma == 0` test passed and the z-score was
        rounding residue over rounding residue. The 0.0 convention holds at
        any level.
        """
        spread = pd.Series([12.3456] * 100)
        assert spread.std() != 0.0
        assert (spread_zscore(spread) == 0.0).all()

    def test_a_flat_rolling_window_is_undefined(self):
        spread = pd.Series([12.3456] * 60)
        z = spread_zscore(spread, window=20)
        assert z.isna().all()
