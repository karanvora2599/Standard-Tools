"""Tests for regression and analysis functions: beta, alpha, R-squared."""

from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis import regression
from standard_quant_tools.analysis.regression import calculate_beta, rolling_beta
from standard_quant_tools.error import ValidationError


class TestCalculateBeta:
    def test_returns_required_keys(self, sample_returns, benchmark_returns):
        result = calculate_beta(sample_returns, benchmark_returns)
        assert set(result.keys()) == {"alpha", "beta", "r_squared"}

    def test_beta_one_when_asset_equals_benchmark(self, sample_returns):
        """When asset and benchmark are identical, beta must be exactly 1."""
        result = calculate_beta(sample_returns, sample_returns)
        assert result["beta"] == pytest.approx(1.0, abs=1e-6)

    def test_alpha_near_zero_when_asset_equals_benchmark(self, sample_returns):
        result = calculate_beta(sample_returns, sample_returns)
        assert result["alpha"] == pytest.approx(0.0, abs=1e-10)

    def test_r_squared_one_when_perfectly_correlated(self, sample_returns):
        result = calculate_beta(sample_returns, sample_returns)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_r_squared_bounded_0_to_1(self, sample_returns, benchmark_returns):
        result = calculate_beta(sample_returns, benchmark_returns)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_beta_two_for_double_leverage(self, sample_returns):
        """An asset with exactly 2x leverage should have beta = 2."""
        leveraged = sample_returns * 2
        result = calculate_beta(leveraged, sample_returns)
        assert result["beta"] == pytest.approx(2.0, abs=1e-6)

    def test_negative_beta_for_inverse_asset(self, sample_returns):
        inverse = -sample_returns
        result = calculate_beta(inverse, sample_returns)
        assert result["beta"] == pytest.approx(-1.0, abs=1e-6)

    def test_low_r_squared_for_uncorrelated_assets(self):
        """Uncorrelated random series should have R² close to 0."""
        np.random.seed(42)
        r1 = pd.Series(np.random.normal(0, 0.01, 500))
        r2 = pd.Series(np.random.normal(0, 0.01, 500))
        result = calculate_beta(r1, r2)
        assert result["r_squared"] < 0.10

    def test_index_alignment_handles_different_lengths(self, sample_returns):
        """calculate_beta should align on common index and not crash."""
        short_bench = sample_returns.iloc[:100]
        result = calculate_beta(sample_returns, short_bench)
        assert isinstance(result["beta"], float)

    def test_nan_in_input_raises(self, sample_returns, benchmark_returns):
        bad = sample_returns.copy()
        bad.iloc[5] = np.nan
        with pytest.raises(ValidationError, match="non-finite"):
            calculate_beta(bad, benchmark_returns)

    def test_minimal_data_is_not_estimable(self):
        """
        A single overlapping point cannot support an OLS fit, so all three
        statistics are NaN — "not estimable".

        This test previously asserted a "safe zero dict" and called that the
        intended behaviour. Zero is not safe here: 0.0 is ALSO a legitimate
        beta (a market-neutral asset), so nothing downstream could tell a
        failed estimate from a real measurement. Two consumers read it the
        wrong way — the screener passed an unestimable ticker through a
        beta_max ceiling, and treynor_ratio turned "no overlapping benchmark
        data" into a plausible-looking risk-adjusted return.
        """
        result = calculate_beta(pd.Series([0.01]), pd.Series([0.01]))
        assert set(result) == {"alpha", "beta", "r_squared"}
        assert all(np.isnan(v) for v in result.values())

    def test_not_estimable_is_distinguishable_from_a_real_zero_beta(self):
        """The property the zero sentinel destroyed: these two states must
        not produce the same number."""
        rng = np.random.default_rng(0)
        idx = pd.date_range("2023-01-02", periods=300, freq="B")
        mkt = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
        independent = pd.Series(rng.normal(0.0005, 0.01, 300), index=idx)
        estimable = calculate_beta(independent, mkt)
        assert np.isfinite(estimable["beta"]), "a real fit stays a number"
        assert np.isnan(calculate_beta(pd.Series([0.01]), pd.Series([0.01]))["beta"])


class TestRollingBeta:
    def test_returns_dataframe_with_rolling_beta_column(
        self, sample_returns, benchmark_returns
    ):
        result = rolling_beta(sample_returns, benchmark_returns, window=60)
        assert "Rolling_Beta" in result.columns

    def test_output_length_matches_input(self, sample_returns, benchmark_returns):
        result = rolling_beta(sample_returns, benchmark_returns, window=60)
        assert len(result) == len(sample_returns)

    def test_nan_prefix_equals_window_minus_one(
        self, sample_returns, benchmark_returns
    ):
        window = 60
        result = rolling_beta(sample_returns, benchmark_returns, window=window)
        assert result["Rolling_Beta"].iloc[: window - 1].isna().all()

    def test_rolling_beta_of_identical_series_is_one(self, sample_returns):
        result = rolling_beta(sample_returns, sample_returns, window=30)
        valid = result["Rolling_Beta"].dropna()
        assert (valid - 1.0).abs().max() < 1e-6

    def test_rolling_beta_of_double_leverage_is_two(self, sample_returns):
        leveraged = sample_returns * 2
        result = rolling_beta(leveraged, sample_returns, window=30)
        valid = result["Rolling_Beta"].dropna()
        assert (valid - 2.0).abs().max() < 1e-6

    def test_rolling_window_does_not_use_future_data(
        self, sample_returns, benchmark_returns
    ):
        """Beta at bar t must only depend on bars [t-window+1 .. t].
        The cov/var rolling approach and OLS may differ slightly; use rel tolerance."""
        window = 30
        result = rolling_beta(sample_returns, benchmark_returns, window=window)
        idx_t = sample_returns.index[window]
        slice_asset = sample_returns.iloc[1 : window + 1]  # window bars ending at t
        slice_bench = benchmark_returns.iloc[1 : window + 1]
        manual = calculate_beta(slice_asset, slice_bench)
        rolling_val = float(result.loc[idx_t, "Rolling_Beta"])
        assert rolling_val == pytest.approx(manual["beta"], rel=0.05)

    def test_constant_benchmark_window_yields_nan_not_inf(self, sample_returns):
        """A window with zero benchmark variance (e.g. a constant
        benchmark) used to divide by zero -- must produce NaN for that
        window, not inf/-inf that could silently poison downstream math."""
        constant_benchmark = pd.Series(1.0, index=sample_returns.index)
        result = rolling_beta(sample_returns, constant_benchmark, window=30)
        valid = result["Rolling_Beta"].dropna()
        assert valid.empty
        assert not np.isinf(result["Rolling_Beta"]).any()


# ── Both backends ────────────────────────────────────────────────────────────


@pytest.fixture(params=[True, False], ids=["native", "numpy"])
def backend(request, monkeypatch):
    """Run the test once on the native kernels and once on the NumPy path."""
    if request.param:
        if not regression.HAS_CPP:
            pytest.skip("C++ extension not built")
    else:
        monkeypatch.setattr(regression, "HAS_CPP", False)
        monkeypatch.setattr(regression, "_cpp_core", None)
    return request.param


def _exact_beta(y, x):
    """The window's OLS slope in exact rational arithmetic."""
    fx = [Fraction(float(v)) for v in x]
    fy = [Fraction(float(v)) for v in y]
    mx, my = sum(fx) / len(fx), sum(fy) / len(fy)
    num = sum((a - mx) * (b - my) for a, b in zip(fx, fy))
    return float(num / sum((a - mx) ** 2 for a in fx))


def _dated(values):
    return pd.Series(values, index=pd.bdate_range("2021-01-04", periods=len(values)))


class TestCalculateBetaDegenerateDesigns:
    """
    One policy for both backends. A benchmark the intercept column spans
    leaves the slope unidentified; the NumPy path used to return lstsq's
    minimum-norm solution there (beta 2e-6, alpha 7e-4 on a constant
    benchmark) while the native path returned NaN.
    """

    @pytest.mark.parametrize("level", [0.01, 250.0, -3.0e-7])
    def test_a_constant_benchmark_has_no_beta(self, backend, level):
        rng = np.random.default_rng(4)
        asset = _dated(rng.normal(0.0005, 0.01, 60))
        result = calculate_beta(asset, _dated(np.full(60, level)))
        assert all(np.isnan(v) for v in result.values()), result

    def test_a_constant_asset_has_a_beta_but_no_r_squared(self, backend):
        rng = np.random.default_rng(5)
        result = calculate_beta(
            _dated(np.full(40, 0.003)), _dated(rng.normal(0, 0.01, 40))
        )
        assert result["beta"] == pytest.approx(0.0, abs=1e-12)
        assert result["alpha"] == pytest.approx(0.003, abs=1e-12)
        assert np.isnan(result["r_squared"])

    def test_one_moved_benchmark_value_is_a_real_fit(self, backend):
        """The null case: the smallest departure from a constant benchmark
        identifies the slope, and both backends report the same one."""
        x = np.full(30, 0.01)
        x[7] = 0.02
        rng = np.random.default_rng(6)
        y = 0.001 + 1.5 * x + rng.normal(0, 1e-4, 30)
        result = calculate_beta(_dated(y), _dated(x))
        assert result["beta"] == pytest.approx(_exact_beta(y, x), rel=1e-9)
        assert 0.0 <= result["r_squared"] <= 1.0


class TestRollingBetaIsExactPerWindow:
    """
    Every window's own beta, on both backends, after a large print has left.

    The native kernel rebuilds its sliding sums when a second moment falls
    four decades below its peak. The fallback used pandas' rolling cov/var,
    which are online too and were not fixed: under pandas 2.x the windows
    after one 1e8 print among 0.01-scale returns were wrong by a median
    factor of 1 and at worst 3.4e3.
    """

    @pytest.mark.parametrize("magnitude", [1e5, 1e8])
    @pytest.mark.parametrize("side", ["asset", "benchmark"])
    def test_every_window_after_the_print_is_the_exact_beta(
        self, backend, magnitude, side
    ):
        window = 60
        for seed in range(6):
            rng = np.random.default_rng(seed)
            x = rng.normal(0, 0.01, 200)
            y = 0.001 + 1.3 * x + rng.normal(0, 0.001, 200)
            (y if side == "asset" else x)[100] = magnitude
            got = rolling_beta(_dated(y), _dated(x), window)["Rolling_Beta"]
            for i in range(160, 200, 3):
                want = _exact_beta(y[i - 59 : i + 1], x[i - 59 : i + 1])
                assert abs(got.iloc[i] - want) <= 1e-9 * abs(want), (seed, i)

    def test_a_flat_benchmark_window_is_nan_and_only_those(self, backend):
        rng = np.random.default_rng(2)
        x = np.r_[rng.normal(0, 0.01, 50), np.full(40, 0.01), rng.normal(0, 0.01, 50)]
        y = 0.7 * x + rng.normal(0, 0.001, 140)
        got = rolling_beta(_dated(y), _dated(x), 20)["Rolling_Beta"].to_numpy()
        flat = np.zeros(140, dtype=bool)
        flat[69:90] = True  # windows ending 69..89 lie wholly inside the run
        warm_up = np.arange(140) < 19
        np.testing.assert_array_equal(np.isnan(got), flat | warm_up)

    def test_a_clean_series_is_the_exact_beta(self, backend):
        """The null case: ordinary returns, every window to rounding."""
        rng = np.random.default_rng(7)
        x = rng.normal(0, 0.01, 300)
        y = 0.8 * x + rng.normal(0, 0.002, 300)
        got = rolling_beta(_dated(y), _dated(x), 60)["Rolling_Beta"]
        assert got.index.equals(_dated(y).index)
        assert got.iloc[:59].isna().all()
        for i in range(59, 300, 11):
            want = _exact_beta(y[i - 59 : i + 1], x[i - 59 : i + 1])
            assert abs(got.iloc[i] - want) <= 1e-12 * abs(want)

    def test_fewer_bars_than_the_window_is_all_nan(self, backend):
        rng = np.random.default_rng(8)
        x = rng.normal(0, 0.01, 10)
        got = rolling_beta(_dated(2 * x), _dated(x), 20)
        assert len(got) == 10 and got["Rolling_Beta"].isna().all()
