"""
Python integration tests for ADF Cointegration C++ extension and Python wrapper.

Two execution modes:
  1. _sqt_core NOT built → cpp_* tests are skipped; wrapper tests verify that
     the statsmodels fallback path produces correct results.
  2. _sqt_core IS built  → all tests run and cross-validate C++ vs statsmodels.

Run:
    pytest tests/test_cpp_cointegration.py -v
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

# ── Extension availability ────────────────────────────────────────────────────

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

requires_cpp = pytest.mark.skipif(not HAS_CPP, reason="_sqt_core not built")

from standard_quant_tools.analysis.cointegration import HAS_CPP as COINT_HAS_CPP
from standard_quant_tools.analysis.cointegration import (
    _kalman_filter_1state,
    _kalman_filter_2state,
    cointegration_test,
    compute_spread,
    half_life,
    kalman_hedge_ratio,
)
from standard_quant_tools.analysis.regression import HAS_CPP as REG_HAS_CPP
from standard_quant_tools.analysis.regression import (
    calculate_beta,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
DATE_IDX = pd.date_range("2020-01-01", periods=400, freq="B")


def _make_cointegrated_pair(
    n: int = 400, hedge: float = 2.0, noise_scale: float = 0.05, seed: int = 7
):
    rng = np.random.default_rng(seed)
    rw = rng.standard_normal(n).cumsum()
    noise = rng.standard_normal(n) * noise_scale
    y1 = pd.Series(rw, index=DATE_IDX[:n])
    y0 = pd.Series(hedge * rw + noise, index=DATE_IDX[:n])
    return y0, y1


def _make_independent_rws(n: int = 300, seed: int = 99):
    rng = np.random.default_rng(seed)
    y0 = pd.Series(rng.standard_normal(n).cumsum(), index=DATE_IDX[:n])
    y1 = pd.Series(rng.standard_normal(n).cumsum(), index=DATE_IDX[:n])
    return y0, y1


# ── Wrapper tests (always run) ────────────────────────────────────────────────


class TestCointegrationTestReturnSchema:
    """The public API must return the expected keys regardless of backend."""

    REQUIRED_KEYS = {
        "cointegrated",
        "hedge_ratio",
        "adf_statistic",
        "p_value",
        "critical_values",
        "half_life_days",
        "n_obs",
    }
    CV_KEYS = {"1%", "5%", "10%"}

    def test_keys_present(self):
        y0, y1 = _make_cointegrated_pair()
        result = cointegration_test(y0, y1)
        assert set(result.keys()) == self.REQUIRED_KEYS

    def test_critical_values_keys(self):
        y0, y1 = _make_cointegrated_pair()
        result = cointegration_test(y0, y1)
        assert set(result["critical_values"].keys()) == self.CV_KEYS

    def test_types(self):
        y0, y1 = _make_cointegrated_pair()
        r = cointegration_test(y0, y1)
        assert isinstance(r["cointegrated"], bool)
        assert isinstance(r["hedge_ratio"], float)
        assert isinstance(r["adf_statistic"], float)
        assert isinstance(r["p_value"], float)
        assert isinstance(r["half_life_days"], float)
        assert isinstance(r["n_obs"], int)

    def test_n_obs_matches_input(self):
        n = 250
        y0, y1 = _make_cointegrated_pair(n=n)
        r = cointegration_test(y0, y1)
        assert r["n_obs"] == n

    def test_p_value_in_unit_interval(self):
        y0, y1 = _make_cointegrated_pair()
        r = cointegration_test(y0, y1)
        assert 0.0 <= r["p_value"] <= 1.0

    def test_critical_values_ordered(self):
        y0, y1 = _make_cointegrated_pair()
        r = cointegration_test(y0, y1)
        cv = r["critical_values"]
        assert cv["1%"] < cv["5%"] < cv["10%"] < 0.0


class TestCointegrationTestStatistics:
    def test_cointegrated_pair_detected(self):
        y0, y1 = _make_cointegrated_pair(n=400, noise_scale=0.02)
        r = cointegration_test(y0, y1)
        assert r["cointegrated"] is True
        assert r["p_value"] < 0.05

    def test_hedge_ratio_close_to_true_value(self):
        y0, y1 = _make_cointegrated_pair(n=400, hedge=2.0, noise_scale=0.02)
        r = cointegration_test(y0, y1)
        assert abs(r["hedge_ratio"] - 2.0) < 0.1

    def test_half_life_positive_for_cointegrated_pair(self):
        y0, y1 = _make_cointegrated_pair(n=400, noise_scale=0.05)
        r = cointegration_test(y0, y1)
        assert r["half_life_days"] > 0.0
        assert not np.isinf(r["half_life_days"])

    def test_independent_rws_not_cointegrated(self):
        # Two unrelated random walks — should rarely pass at 1% level
        y0, y1 = _make_independent_rws(n=300)
        r = cointegration_test(y0, y1)
        assert r["p_value"] > 0.01

    def test_bic_autolag(self):
        y0, y1 = _make_cointegrated_pair(n=300)
        r = cointegration_test(y0, y1, autolag="bic")
        assert 0.0 <= r["p_value"] <= 1.0

    def test_index_alignment(self):
        # Misaligned indices — common index should be used
        y0, y1 = _make_cointegrated_pair(n=300)
        y1_shifted = y1.iloc[10:]  # drop first 10 bars
        r = cointegration_test(y0, y1_shifted)
        assert r["n_obs"] == 290


class TestComputeSpread:
    def test_spread_shape(self):
        y0, y1 = _make_cointegrated_pair(n=200)
        spread = compute_spread(y0, y1)
        assert len(spread) == 200

    def test_spread_with_known_hedge(self):
        y0, y1 = _make_cointegrated_pair(n=200, hedge=1.5, noise_scale=0.0)
        spread = compute_spread(y0, y1, hedge_ratio=1.5)
        assert np.allclose(spread.to_numpy(), 0.0, atol=1e-8)

    def test_spread_estimated_hedge_near_zero_mean(self):
        y0, y1 = _make_cointegrated_pair(n=300, noise_scale=0.01)
        spread = compute_spread(y0, y1)
        assert abs(spread.mean()) < 0.2


class TestHalfLife:
    def test_ar1_mean_reverting(self):
        rng = np.random.default_rng(5)
        phi = 0.85
        x = np.zeros(500)
        for i in range(1, 500):
            x[i] = phi * x[i - 1] + rng.standard_normal()
        spread = pd.Series(x)
        hl = half_life(spread)
        # Expected: -ln(2)/ln(phi) = -ln(2)/ln(0.85) ≈ 4.27
        assert 1.0 < hl < 20.0

    def test_non_mean_reverting_returns_inf(self):
        rw = pd.Series(np.random.default_rng(9).standard_normal(200).cumsum())
        hl = half_life(rw)
        # May or may not be inf depending on sample, but should be large
        assert hl > 0.0

    def test_short_series_returns_inf(self):
        spread = pd.Series([1.0, 2.0])
        hl = half_life(spread)
        assert np.isinf(hl)


# ── C++ binding tests (skipped unless _sqt_core is built) ────────────────────


class TestCppEngleGrangerDirect:
    """Direct calls to _sqt_core.engle_granger — bypasses the Python wrapper."""

    @requires_cpp
    def test_basic_call_returns_dict(self):
        y0 = np.random.default_rng(1).standard_normal(200).cumsum()
        y1 = np.random.default_rng(2).standard_normal(200).cumsum()
        result = _cpp.engle_granger(y0, y1)
        assert isinstance(result, dict)

    @requires_cpp
    def test_required_keys(self):
        y0 = np.random.default_rng(3).standard_normal(200).cumsum()
        y1 = 2.0 * y0 + np.random.default_rng(4).standard_normal(200) * 0.01
        result = _cpp.engle_granger(y0, y1)
        expected = {
            "intercept",
            "hedge_ratio",
            "adf_statistic",
            "optimal_lag",
            "p_value",
            "cv_1pct",
            "cv_5pct",
            "cv_10pct",
            "half_life",
            "n_obs",
            "cointegrated",
        }
        assert set(result.keys()) == expected

    @requires_cpp
    def test_cointegrated_pair(self):
        rng = np.random.default_rng(10)
        rw = rng.standard_normal(400).cumsum()
        y0 = 1.8 * rw + rng.standard_normal(400) * 0.03
        y1 = rw
        result = _cpp.engle_granger(y0, y1)
        assert result["cointegrated"] is True
        assert result["p_value"] < 0.05
        assert abs(result["hedge_ratio"] - 1.8) < 0.15

    @requires_cpp
    def test_critical_values_ordered(self):
        rng = np.random.default_rng(20)
        y0 = rng.standard_normal(300).cumsum()
        y1 = rng.standard_normal(300).cumsum()
        result = _cpp.engle_granger(y0, y1)
        assert result["cv_1pct"] < result["cv_5pct"] < result["cv_10pct"] < 0.0

    @requires_cpp
    @pytest.mark.parametrize("n", [40, 51, 100, 250, 601, 1000])
    def test_critical_values_match_statsmodels_exactly(self, n):
        """Ordering was all that was ever asserted about these numbers.

        The kernel shipped MacKinnon (1991) Table 1 coefficients under a
        comment naming MacKinnon (2010) Table 2, and evaluated the response
        surface at nobs rather than the nobs-1 statsmodels' coint() uses.
        Both are monotonic and negative, so `cv_1pct < cv_5pct < cv_10pct <
        0` passed throughout while the values disagreed by up to 0.006 --
        largest on the short samples a walk-forward fold actually uses.
        """
        from statsmodels.tsa.adfvalues import mackinnoncrit

        rng = np.random.default_rng(n)
        y1 = rng.standard_normal(n).cumsum() + 100.0
        y0 = 1.7 * y1 + rng.standard_normal(n)
        result = _cpp.engle_granger(y0, y1)

        expected = mackinnoncrit(N=2, regression="c", nobs=n - 1)
        assert result["cv_1pct"] == pytest.approx(expected[0], abs=1e-12)
        assert result["cv_5pct"] == pytest.approx(expected[1], abs=1e-12)
        assert result["cv_10pct"] == pytest.approx(expected[2], abs=1e-12)

    @requires_cpp
    def test_p_value_in_unit_interval(self):
        rng = np.random.default_rng(30)
        y0 = rng.standard_normal(250).cumsum()
        y1 = rng.standard_normal(250).cumsum()
        result = _cpp.engle_granger(y0, y1)
        assert 0.0 <= result["p_value"] <= 1.0

    @requires_cpp
    def test_n_obs(self):
        n = 180
        y0 = np.arange(n, dtype=float)
        y1 = np.arange(n, dtype=float) * 0.5
        result = _cpp.engle_granger(y0, y1)
        assert result["n_obs"] == n

    @requires_cpp
    def test_mismatched_length_raises(self):
        y0 = np.ones(100)
        y1 = np.ones(99)
        with pytest.raises(Exception):
            _cpp.engle_granger(y0, y1)

    @requires_cpp
    def test_half_life_positive_for_cointegrated(self):
        rng = np.random.default_rng(40)
        rw = rng.standard_normal(400).cumsum()
        y0 = rw + rng.standard_normal(400) * 0.02
        y1 = rw
        result = _cpp.engle_granger(y0, y1)
        assert result["half_life"] > 0.0
        assert not np.isinf(result["half_life"])

    @requires_cpp
    def test_bic_flag(self):
        rng = np.random.default_rng(50)
        y0 = rng.standard_normal(200).cumsum()
        y1 = rng.standard_normal(200).cumsum()
        r_aic = _cpp.engle_granger(y0, y1, use_aic=True)
        r_bic = _cpp.engle_granger(y0, y1, use_aic=False)
        # Both should complete without error and return valid stats
        assert 0.0 <= r_aic["p_value"] <= 1.0
        assert 0.0 <= r_bic["p_value"] <= 1.0

    @requires_cpp
    def test_explicit_max_lag(self):
        rng = np.random.default_rng(60)
        y0 = rng.standard_normal(200).cumsum()
        y1 = rng.standard_normal(200).cumsum()
        result = _cpp.engle_granger(y0, y1, max_lag=3)
        assert result["optimal_lag"] <= 3

    @requires_cpp
    def test_large_baseline_hedge_ratio_recovered(self):
        """engle_granger's step-1 OLS (ols2) at a ~1e9 baseline -- same
        catastrophic-cancellation regression as TestCppOls2Direct's own
        large-baseline test, exercised end-to-end through the full
        two-variable cointegration pipeline."""
        rng = np.random.default_rng(11)
        n = 400
        baseline = 1e9
        rw = rng.standard_normal(n).cumsum()
        y1 = baseline + rw
        y0 = 1.8 * y1 + rng.standard_normal(n) * 0.05
        result = _cpp.engle_granger(y0, y1)
        assert not np.isnan(result["hedge_ratio"])
        assert result["hedge_ratio"] == pytest.approx(1.8, abs=0.1)

    @requires_cpp
    def test_max_lag_above_old_silent_cap_is_honored(self):
        """Regression test: adf_test() used to silently clamp any max_lag
        above 14 (a fixed kMaxK-derived ceiling) to at most 12, with no
        error. kMaxK is now removed -- the loop's own data-driven
        `T < p + 3` break is the sole limiter, so a large series with a
        large explicit max_lag must not raise/crash and must be free to
        select a lag beyond the old ceiling if AIC favors one."""
        rng = np.random.default_rng(71)
        n = 600
        y0 = rng.standard_normal(n).cumsum()
        y1 = rng.standard_normal(n).cumsum()
        # No crash/error at a max_lag far beyond the old silent cap of 14.
        result = _cpp.engle_granger(y0, y1, max_lag=50)
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["n_obs"] == n
        # The candidate lag space genuinely extends past the old cap: the
        # data-driven T >= p+3 bound allows lags well beyond 14 for this n.
        assert result["optimal_lag"] <= 50


class TestCppVsStatsmodels:
    """Cross-validate C++ against the statsmodels fallback on the same data."""

    @requires_cpp
    def test_critical_values_match_coint_across_random_pairs(self):
        """End-to-end: the same call, both backends, on the same data.

        test_critical_values_match_statsmodels_exactly checks the response
        surface against mackinnoncrit directly. This checks the whole
        engle_granger path against coint(), which is what
        cointegration_test() actually dispatches to when the extension is
        not built -- including that both agree on the SAMPLE SIZE fed to the
        surface, the second half of the defect.
        """
        from statsmodels.tsa.stattools import coint

        worst = 0.0
        for seed in range(25):
            rng = np.random.default_rng(seed)
            n = int(rng.integers(40, 600))
            y1 = rng.standard_normal(n).cumsum() + 100.0
            y0 = 1.7 * y1 + rng.standard_normal(n) * rng.uniform(0.5, 4.0)
            result = _cpp.engle_granger(y0, y1)
            _, _, sm_cv = coint(y0, y1, trend="c", autolag="aic")
            worst = max(
                worst,
                abs(result["cv_1pct"] - sm_cv[0]),
                abs(result["cv_5pct"] - sm_cv[1]),
                abs(result["cv_10pct"] - sm_cv[2]),
            )
        assert worst < 1e-12, f"worst critical-value disagreement {worst:.3e}"

    @requires_cpp
    def test_p_value_broadly_consistent(self):
        # p-values from two different implementations won't match exactly,
        # but sign of test and cointegration conclusion should agree.
        y0, y1 = _make_cointegrated_pair(n=400, noise_scale=0.02)
        a_vals = y0.to_numpy(dtype=float)
        b_vals = y1.to_numpy(dtype=float)

        cpp_r = _cpp.engle_granger(a_vals, b_vals)

        # statsmodels reference
        from statsmodels.tsa.stattools import coint

        _, sm_pval, _ = coint(a_vals, b_vals, trend="c", autolag="aic")

        # Both should agree on cointegration at 5% for this strongly cointegrated pair
        assert (cpp_r["p_value"] < 0.05) == (sm_pval < 0.05)

    @requires_cpp
    def test_hedge_ratio_consistent(self):
        y0, y1 = _make_cointegrated_pair(n=400, hedge=1.5, noise_scale=0.02)
        a_vals = y0.to_numpy(dtype=float)
        b_vals = y1.to_numpy(dtype=float)

        cpp_r = _cpp.engle_granger(a_vals, b_vals)

        # statsmodels-path hedge ratio via direct lstsq
        X = np.column_stack([np.ones(len(a_vals)), b_vals])
        beta, *_ = np.linalg.lstsq(X, a_vals, rcond=None)
        sm_hedge = float(beta[1])

        # Same normal equations → should match to machine precision
        assert abs(cpp_r["hedge_ratio"] - sm_hedge) < 0.01

    @requires_cpp
    def test_wrapper_routes_to_cpp(self):
        # When C++ is built, the wrapper should use it (COINT_HAS_CPP = True)
        assert COINT_HAS_CPP is True


class TestMackinnonPValueAccuracy:
    """Regression coverage for the MacKinnon (2010) p-value fix: the
    previous 13-point lookup table was off by up to 0.08 mid-distribution
    despite claiming ±0.01-0.02 accuracy. The replacement is the exact
    regression-surface algorithm statsmodels itself uses (same
    coefficients, same quadratic/cubic branch split, same normal-CDF
    mapping), so it should match statsmodels.tsa.stattools.mackinnonp to
    near machine precision -- not just "broadly consistent" -- for
    engle_granger's own computed adf_statistic, across a real spread of
    statistics (weak to strong mean reversion), not just one lucky point.
    """

    @requires_cpp
    def test_matches_statsmodels_across_a_range_of_statistics(self):
        from statsmodels.tsa.stattools import mackinnonp

        # y1 is (near-)constant so the step-1 OLS just centers y0 -- lets
        # y0's own AR(1) structure directly control the resulting ADF
        # statistic's magnitude via phi, sweeping from barely-distinguishable-
        # from-a-unit-root (phi close to 1) to strongly mean-reverting (phi
        # small), which lands adf_statistic across a wide, realistic range
        # rather than only the extreme-negative region a naive cointegrated
        # random-walk pair tends to produce.
        rng = np.random.default_rng(21)
        n = 400
        y1 = pd.Series(np.full(n, 100.0) + rng.standard_normal(n) * 1e-6)

        seen_mid_range = False
        for phi in (0.995, 0.98, 0.95, 0.9, 0.8, 0.6, 0.3):
            eps = rng.standard_normal(n) * 0.5
            y0 = np.zeros(n)
            y0[0] = eps[0]
            for t in range(1, n):
                y0[t] = phi * y0[t - 1] + eps[t]
            y0 = pd.Series(y0 + 100.0)

            cpp_r = _cpp.engle_granger(
                y0.to_numpy(dtype=float), y1.to_numpy(dtype=float)
            )
            stat = cpp_r["adf_statistic"]
            if np.isnan(stat):
                continue
            expected = mackinnonp(stat, regression="c", N=2)
            if -18.86 < stat < 0.92:
                seen_mid_range = True
            assert cpp_r["p_value"] == pytest.approx(expected, abs=1e-9), (
                f"phi={phi}  adf_statistic={stat}  "
                f"cpp={cpp_r['p_value']}  statsmodels={expected}"
            )

        # Sanity check the sweep actually exercised the interesting part of
        # the distribution (not every phi degenerating to the -18.86/0.92
        # clamp boundaries), otherwise this test would pass trivially.
        assert seen_mid_range

    @requires_cpp
    def test_p05_anchor_point_exact(self):
        # tau_star=-2.62 is the branch boundary; -3.3377 is the classic 5%
        # critical value anchor both the old table and the new algorithm
        # agree on -- confirms the new algorithm didn't regress the one
        # point that was already exact.
        from statsmodels.tsa.stattools import mackinnonp

        expected = mackinnonp(-3.3377, regression="c", N=2)
        assert expected == pytest.approx(0.05, abs=0.001)


# ── C++ ols2 binding tests ────────────────────────────────────────────────────


class TestCppOls2Direct:
    """Direct calls to _sqt_core.ols2 — bypasses the Python wrapper."""

    @requires_cpp
    def test_returns_dict_with_required_keys(self):
        y = np.array([5.0, 7.0, 9.0, 11.0, 13.0])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = _cpp.ols2(y, x)
        assert set(r.keys()) == {"intercept", "slope", "r_squared"}

    @requires_cpp
    def test_perfect_linear_fit(self):
        # y = 3 + 2*x → intercept=3, slope=2, R²=1
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 + 2.0 * x
        r = _cpp.ols2(y, x)
        assert abs(r["intercept"] - 3.0) < 1e-9
        assert abs(r["slope"] - 2.0) < 1e-9
        assert abs(r["r_squared"] - 1.0) < 1e-9

    @requires_cpp
    def test_matches_numpy_lstsq(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(200)
        y = 0.5 + 1.3 * x + rng.standard_normal(200) * 0.2
        r_cpp = _cpp.ols2(y, x)
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        assert abs(r_cpp["intercept"] - beta[0]) < 1e-8
        assert abs(r_cpp["slope"] - beta[1]) < 1e-8

    @requires_cpp
    def test_r_squared_in_unit_interval(self):
        rng = np.random.default_rng(13)
        x = rng.standard_normal(100)
        y = 2.0 * x + rng.standard_normal(100) * 3.0
        r = _cpp.ols2(y, x)
        assert 0.0 <= r["r_squared"] <= 1.0

    @requires_cpp
    def test_mismatched_length_raises(self):
        y = np.ones(50)
        x = np.ones(49)
        with pytest.raises(Exception):
            _cpp.ols2(y, x)

    @requires_cpp
    def test_negative_slope(self):
        x = np.linspace(0, 10, 50)
        y = 5.0 - 1.5 * x
        r = _cpp.ols2(y, x)
        assert r["slope"] < 0.0
        assert abs(r["slope"] - (-1.5)) < 1e-9

    @requires_cpp
    def test_calculate_beta_routes_to_cpp(self):
        assert REG_HAS_CPP is True

    @requires_cpp
    def test_calculate_beta_matches_lstsq(self):
        rng = np.random.default_rng(99)
        bm = pd.Series(
            rng.standard_normal(300), index=pd.date_range("2020-01-01", periods=300)
        )
        asset = pd.Series(
            1.2 * bm.values + rng.standard_normal(300) * 0.1, index=bm.index
        )
        r = calculate_beta(asset, bm)
        assert abs(r["beta"] - 1.2) < 0.05
        assert 0.0 <= r["r_squared"] <= 1.0

    @requires_cpp
    def test_half_life_cpp_path(self):
        # AR(1) with phi=0.8 → half-life ≈ -ln(2)/ln(0.8) ≈ 3.1 bars
        rng = np.random.default_rng(17)
        x = np.zeros(500)
        for i in range(1, 500):
            x[i] = 0.8 * x[i - 1] + rng.standard_normal()
        hl = half_life(pd.Series(x))
        assert 1.0 < hl < 15.0

    @requires_cpp
    def test_compute_spread_cpp_path(self):
        rng = np.random.default_rng(23)
        rw = rng.standard_normal(300).cumsum()
        y0 = pd.Series(
            2.0 * rw + rng.standard_normal(300) * 0.05,
            index=pd.date_range("2020-01-01", periods=300),
        )
        y1 = pd.Series(rw, index=pd.date_range("2020-01-01", periods=300))
        spread = compute_spread(y0, y1)
        assert len(spread) == 300
        assert abs(spread.mean()) < 0.5  # spread near zero for cointegrated pair

    @requires_cpp
    def test_large_baseline_no_catastrophic_cancellation(self):
        """Regression test for ols2's raw-moment cancellation bug (mirrors
        rolling_beta's own large-baseline test in test_cpp_regression.py):
        an x series with a ~1e9 baseline used to make
        det = s1*sxx - sx*sx compute to exactly 0.0 (total cancellation
        between two ~1e20-magnitude terms), falsely declaring the pair
        singular and returning all-NaN for a perfectly well-posed
        regression. The shift-by-reference-point fix must now recover the
        true slope/intercept, matching an independent numpy.polyfit
        reference."""
        rng = np.random.default_rng(3)
        n = 500
        baseline = 1e9
        x = baseline + rng.standard_normal(n)
        true_slope, true_intercept = 1.5, 10.0
        y = true_intercept + true_slope * x + rng.standard_normal(n) * 0.01

        r = _cpp.ols2(y, x)
        assert not np.isnan(r["slope"])
        assert not np.isnan(r["intercept"])

        ref_slope, ref_intercept = np.polyfit(x, y, 1)
        assert r["slope"] == pytest.approx(ref_slope, rel=1e-6)
        # Un-shifted intercept is an extrapolation ~1e9 units from the data
        # (inherently poorly determined -- even numpy's own polyfit differs
        # from the "naive" true_intercept by ~1e5 here, since slope noise
        # gets amplified by the extrapolation distance) -- match the
        # independent numpy reference at moderate relative tolerance, not
        # the naive true_intercept.
        assert r["intercept"] == pytest.approx(ref_intercept, rel=1e-3)


class TestCppKalman1State:
    """Direct calls to _sqt_core.kalman_filter_1state vs. the numba
    reference, at the standard atol=1e-10 precedent."""

    @requires_cpp
    def test_matches_numba_reference(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(300).cumsum()
        y = 1.3 * x + rng.standard_normal(300) * 0.2

        cpp_result = _cpp.kalman_filter_1state(y, x, 1e-4, 1e-3)
        numba_beta, numba_gain, numba_innov = _kalman_filter_1state(y, x, 1e-4, 1e-3)

        np.testing.assert_allclose(cpp_result["beta"], numba_beta, atol=1e-10)
        np.testing.assert_allclose(cpp_result["gain"], numba_gain, atol=1e-10)
        np.testing.assert_allclose(cpp_result["innovation"], numba_innov, atol=1e-10)

    @requires_cpp
    def test_empty_on_bad_delta(self):
        y = np.array([1.0, 2.0, 3.0])
        x = np.array([1.0, 2.0, 3.0])
        for bad_delta in (0.0, 1.0, -0.1, 1.5):
            r = _cpp.kalman_filter_1state(y, x, bad_delta, 1e-3)
            assert len(r["beta"]) == 0

    @requires_cpp
    def test_empty_on_bad_observation_noise(self):
        y = np.array([1.0, 2.0, 3.0])
        x = np.array([1.0, 2.0, 3.0])
        for bad_noise in (0.0, -1.0):
            r = _cpp.kalman_filter_1state(y, x, 1e-4, bad_noise)
            assert len(r["beta"]) == 0


class TestCppKalman2State:
    """Direct calls to _sqt_core.kalman_filter_2state vs. the numba
    reference, at the standard atol=1e-10 precedent."""

    @requires_cpp
    def test_matches_numba_reference(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(300).cumsum()
        y = 5.0 + 1.3 * x + rng.standard_normal(300) * 0.2

        cpp_result = _cpp.kalman_filter_2state(y, x, 1e-4, 1e-3)
        numba_alpha, numba_beta, numba_gain, numba_innov = _kalman_filter_2state(
            y, x, 1e-4, 1e-3
        )

        np.testing.assert_allclose(cpp_result["alpha"], numba_alpha, atol=1e-10)
        np.testing.assert_allclose(cpp_result["beta"], numba_beta, atol=1e-10)
        np.testing.assert_allclose(cpp_result["gain"], numba_gain, atol=1e-10)
        np.testing.assert_allclose(cpp_result["innovation"], numba_innov, atol=1e-10)


class TestKalmanHedgeRatioWrapper:
    """Confirms kalman_hedge_ratio()'s public DataFrame output is column-
    for-column identical whether _sqt_core is built or not."""

    @requires_cpp
    def test_output_identical_with_and_without_cpp_2state(self):
        import standard_quant_tools.analysis.cointegration as coint_module

        rng = np.random.default_rng(2)
        idx = pd.date_range("2020-01-01", periods=250)
        b = pd.Series(rng.standard_normal(250).cumsum() + 100, index=idx)
        a = pd.Series(1.2 * b.values + rng.standard_normal(250) * 0.3, index=idx)

        result_cpp = kalman_hedge_ratio(a, b, include_intercept=True)

        coint_module.HAS_CPP = False
        try:
            result_numba = kalman_hedge_ratio(a, b, include_intercept=True)
        finally:
            coint_module.HAS_CPP = True

        pd.testing.assert_frame_equal(result_cpp, result_numba, atol=1e-10)

    @requires_cpp
    def test_output_identical_with_and_without_cpp_1state(self):
        import standard_quant_tools.analysis.cointegration as coint_module

        rng = np.random.default_rng(3)
        idx = pd.date_range("2020-01-01", periods=250)
        b = pd.Series(rng.standard_normal(250).cumsum() + 100, index=idx)
        a = pd.Series(1.2 * b.values + rng.standard_normal(250) * 0.3, index=idx)

        result_cpp = kalman_hedge_ratio(a, b, include_intercept=False)

        coint_module.HAS_CPP = False
        try:
            result_numba = kalman_hedge_ratio(a, b, include_intercept=False)
        finally:
            coint_module.HAS_CPP = True

        pd.testing.assert_frame_equal(result_cpp, result_numba, atol=1e-10)
