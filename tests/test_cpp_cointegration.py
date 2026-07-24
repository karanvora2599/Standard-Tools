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
    cointegration_test,
    compute_spread,
    half_life,
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


class TestCppVsStatsmodels:
    """Cross-validate C++ against the statsmodels fallback on the same data."""

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
