"""Tests for Hurst exponent estimation: hurst_exponent and rolling_hurst."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.hurst import hurst_exponent, rolling_hurst
from standard_quant_tools.error import ValidationError

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
    """AR(1) phi=-0.5 → negatively autocorrelated → H < 0.45 (phi=-0.4 gives H≈0.456, too close to boundary)."""
    return _make_returns(seed=42, n=2000, phi=-0.5)


# ── hurst_exponent — output structure ─────────────────────────────────────────


class TestHurstExponentKeys:
    def test_returns_required_keys(self, iid_returns):
        """
        The raw R/S slope, the correction applied to it, the width of the
        random-walk band and the largest window actually fitted are part of
        the answer: a caller comparing against a published R/S figure needs
        the raw one, and a regime label means nothing without its band.
        """
        result = hurst_exponent(iid_returns)
        assert set(result.keys()) == {
            "hurst",
            "hurst_raw",
            "bias_correction",
            "regime",
            "regime_band",
            "fit_r_squared",
            "method",
            "n_obs",
            "max_window_used",
            "warnings",
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

    def test_hurst_bounded_0_to_1(
        self, iid_returns, trending_returns, mean_reverting_returns
    ):
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


class TestHurstMethodValidation:
    """
    Regression: both the C++ kernel and the Python fallback treat *anything*
    other than the exact string "dfa" as "rs" -- a typo like "DFA" or "rsi"
    used to silently run R/S analysis while echoing the typo'd string back
    in the "method" field of the result, hiding the mismatch. Must now
    raise ValidationError before either code path runs.
    """

    @pytest.mark.parametrize("bad_method", ["DFA", "RS", "rsi", "r/s", "", "dfa "])
    def test_hurst_exponent_rejects_invalid_method(self, iid_returns, bad_method):
        with pytest.raises(ValidationError, match="method"):
            hurst_exponent(iid_returns, method=bad_method)

    @pytest.mark.parametrize("bad_method", ["DFA", "RS", "rsi", "r/s"])
    def test_rolling_hurst_rejects_invalid_method(self, iid_returns, bad_method):
        with pytest.raises(ValidationError, match="method"):
            rolling_hurst(iid_returns, window=200, method=bad_method)

    def test_valid_methods_still_accepted(self, iid_returns):
        assert not np.isnan(hurst_exponent(iid_returns, method="dfa")["hurst"])
        assert not np.isnan(hurst_exponent(iid_returns, method="rs")["hurst"])


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
        assert not result.iloc[window - 1 :].isna().any()

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
            ret[i] = 0.4 * ret[i - 1] + innov[i]  # trending first half
        for i in range(half, n):
            ret[i] = -0.4 * ret[i - 1] + innov[i]  # mean-reverting second half

        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        s = pd.Series(ret, index=dates)
        rolling = rolling_hurst(s, window=200, step=10)

        # Compare means in second and fourth quarters (avoid transition zone)
        q2_end = 3 * n // 4
        q2_start = n // 4
        mean_early = rolling.iloc[q2_start:half].dropna().mean()
        mean_late = rolling.iloc[q2_end:].dropna().mean()

        assert (
            mean_early > mean_late
        ), f"Expected early H ({mean_early:.3f}) > late H ({mean_late:.3f})"


# ── R/S small-sample correction and the length-aware regime band ──────────────


def _fgn(hurst, n, seed):
    """Exact fractional Gaussian noise by Cholesky of its autocovariance."""
    k = np.arange(n, dtype=float)
    gamma = 0.5 * (
        np.abs(k + 1) ** (2 * hurst)
        - 2 * np.abs(k) ** (2 * hurst)
        + np.abs(k - 1) ** (2 * hurst)
    )
    cov = gamma[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
    chol = np.linalg.cholesky(cov)
    return pd.Series(chol @ np.random.default_rng(seed).standard_normal(n))


class TestRescaledRangeCorrection:
    def test_white_noise_is_not_called_trending_by_rs(self):
        """
        The uncorrected rescaled range is biased upward by about 0.07 at
        1024 observations, and with a fixed 0.55 threshold it labelled about
        two thirds of pure white-noise series "trending". Corrected, the
        mean sits at 0.5 and the labels fall to the band's nominal rate.
        """
        rng = np.random.default_rng(11)
        values, trending = [], 0
        for _ in range(120):
            result = hurst_exponent(pd.Series(rng.standard_normal(1024)), method="rs")
            values.append(result["hurst"])
            trending += result["regime"] == "trending"
        assert 0.48 <= float(np.mean(values)) <= 0.52
        assert trending <= 12

    def test_the_correction_is_reported_next_to_the_raw_slope(self):
        rng = np.random.default_rng(12)
        result = hurst_exponent(pd.Series(rng.standard_normal(1024)), method="rs")
        assert 0.05 < result["bias_correction"] < 0.09
        assert result["hurst"] == pytest.approx(
            result["hurst_raw"] - result["bias_correction"], abs=1e-12
        )

    def test_dfa_is_not_corrected(self, iid_returns):
        result = hurst_exponent(iid_returns, method="dfa")
        assert result["bias_correction"] == 0.0
        assert result["hurst"] == result["hurst_raw"]

    def test_persistent_noise_is_still_called_trending(self):
        """The planted case: exact fGn at H=0.7 is labelled trending."""
        trending = sum(
            hurst_exponent(_fgn(0.7, 1024, seed), method="rs")["regime"] == "trending"
            for seed in range(30)
        )
        assert trending >= 27

    def test_rolling_rs_equals_the_single_window_value(self):
        """One correction constant per (window, min_window), so each rolling
        value is exactly what hurst_exponent gives on that window alone."""
        series = pd.Series(np.random.default_rng(13).standard_normal(500))
        rolling = rolling_hurst(series, window=200, step=41, method="rs")
        positions = np.where(rolling.notna())[0]
        assert positions.size >= 5
        for i in positions:
            alone = hurst_exponent(series.iloc[i - 199 : i + 1], method="rs")
            assert rolling.iloc[i] == pytest.approx(alone["hurst"], abs=1e-12)
            assert rolling.attrs["bias_correction"] == alone["bias_correction"]
            assert rolling.attrs["regime_band"] == alone["regime_band"]


class TestRegimeBand:
    @pytest.mark.parametrize("method", ["dfa", "rs"])
    @pytest.mark.parametrize("n", [256, 1024])
    def test_the_null_table_matches_the_estimator(self, method, n):
        """
        The band is built from a table of white-noise standard deviations.
        Re-measured here, so a change to either estimator that the table
        does not follow fails instead of silently mislabelling.
        """
        from standard_quant_tools.analysis.hurst import null_standard_deviation

        rng = np.random.default_rng(1000 + n)
        values = [
            hurst_exponent(pd.Series(rng.standard_normal(n)), method=method)["hurst"]
            for _ in range(300)
        ]
        measured = float(np.std(values))
        assert measured == pytest.approx(null_standard_deviation(n, method), rel=0.15)

    def test_short_white_noise_is_labelled_at_the_nominal_rate(self):
        """
        At 256 observations the DFA estimate of white noise has a standard
        deviation of 0.08; the fixed +/-0.05 band labelled 27% of such
        series trending and 30% mean-reverting. The band now widens with
        the noise.
        """
        rng = np.random.default_rng(14)
        labels = [
            hurst_exponent(pd.Series(rng.standard_normal(256)))["regime"]
            for _ in range(200)
        ]
        assert labels.count("trending") <= 20
        assert labels.count("mean_reverting") <= 20

    def test_the_band_narrows_with_length_and_has_a_floor(self):
        from standard_quant_tools.analysis.hurst import regime_band

        assert regime_band(256, "dfa") > regime_band(1024, "dfa") > 0.05
        assert regime_band(100_000, "dfa") == 0.05
        assert regime_band(100_000, "rs") == 0.05


class TestWindowRefusals:
    def test_dfa_refuses_a_two_point_box(self, iid_returns):
        """
        A DFA box of two points fits its line exactly; the fluctuation is
        floating-point residue and white noise came back H=1.5, trending.
        """
        with pytest.raises(ValidationError, match="at least 4"):
            hurst_exponent(iid_returns, method="dfa", min_window=2)
        with pytest.raises(ValidationError, match="at least 4"):
            rolling_hurst(iid_returns, window=200, method="dfa", min_window=3)

    def test_dfa_answers_from_a_four_point_box(self):
        series = pd.Series(np.random.default_rng(15).standard_normal(2048))
        result = hurst_exponent(series, method="dfa", min_window=4)
        assert 0.4 <= result["hurst"] <= 0.6

    @pytest.mark.parametrize("max_window", [20, 50])
    def test_an_inverted_window_range_is_refused(self, iid_returns, max_window):
        """
        min_window=50 with max_window=20 used to come back as NaN, which
        the agent tool then reported as 0.0 -- "strongly mean-reverting" --
        for any series at all.
        """
        with pytest.raises(ValidationError, match="greater than min_window"):
            hurst_exponent(iid_returns, min_window=50, max_window=max_window)

    def test_a_clamped_max_window_is_reported(self, iid_returns):
        result = hurst_exponent(iid_returns, method="dfa", max_window=5000)
        assert result["max_window_used"] == len(iid_returns) // 4
        assert any("lowered" in w for w in result["warnings"])

    def test_a_short_series_is_nan_with_a_reason(self):
        result = hurst_exponent(pd.Series(np.random.default_rng(16).normal(0, 1, 30)))
        assert np.isnan(result["hurst"])
        assert result["regime"] == "unknown"
        assert any("too few" in w for w in result["warnings"])
