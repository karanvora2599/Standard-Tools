"""
A GARCH fit that converged and a GARCH fit that worked are two answers.

`converged` is a statement about L-BFGS-B. It was the only verdict the
result carried, so a fit whose own residuals reject the specification was
indistinguishable from one the sample supports. The standardized residual
z = e / sqrt(sigma2) is what separates them, and the conditional-variance
path it is computed from was already being built on every call and thrown
away except for its last element.

The same shape one module over: `rolling_sharpe_stability` computed a
rolling series, described it in twenty-one scalars, promised it twice in
prose the agent reads -- and returned everything about it except the series.
See the CHANGELOG entry of 2026-09-22.

Every detector here is measured against a planted answer AND against a null
sample it must stay quiet on.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.analysis.diagnostics import rolling_sharpe_stability
from standard_quant_tools.analysis.garch import HAS_SCIPY, garch_volatility_forecast
from standard_quant_tools.error import ValidationError

pytestmark = pytest.mark.skipif(
    not HAS_SCIPY, reason="GARCH MLE fitting requires scipy"
)

TRADING_DAYS = 252
N_SEEDS = 40


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-01", periods=n, freq="B")


def _iid(n: int = 600, seed: int = 0) -> pd.Series:
    """Constant variance, no clustering to find. The null case."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, n), index=_dates(n))


def _cycling_variance(n: int = 600, seed: int = 0, period: int = 20) -> pd.Series:
    """
    Volatility that cycles on a fixed 20-bar period: quiet, loud, quiet.

    GARCH(1,1) models variance as geometric decay from the last shock. A
    cycle is not geometric decay, so the fit converges onto the best
    single-decay approximation of it and the clustering survives into the
    squared standardized residuals. That is the case the diagnostic exists
    for -- the optimizer succeeded and the model is the wrong one.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    vol = 0.01 * (1.0 + 0.95 * np.sin(2 * np.pi * t / period))
    return pd.Series(rng.normal(0.0, 1.0, n) * vol, index=_dates(n))


def _ar1_in_mean(n: int = 600, seed: int = 0, phi: float = 0.25) -> pd.Series:
    """Autocorrelated RETURNS with constant variance: a mean-equation
    problem, not a variance one."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, 0.01, n)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = phi * values[t - 1] + shocks[t]
    return pd.Series(values, index=_dates(n))


class TestTheNullSampleIsNotFlagged:
    def test_an_iid_sample_converges_and_is_not_called_misspecified(self):
        """
        iid Gaussian returns have no clustering, so the detector should fire
        at its nominal 5% and no more. Bounded rather than pinned at zero:
        a test that never fires on the null is a test with no size, and 5%
        of 40 seeds is 2.
        """
        converged = 0
        flagged = 0
        for seed in range(N_SEEDS):
            result = garch_volatility_forecast(_iid(seed=seed))
            converged += result["converged"]
            flagged += result["misspecified"]
        assert converged == N_SEEDS, f"only {converged}/{N_SEEDS} fits converged"
        assert flagged <= 6, (
            f"{flagged}/{N_SEEDS} iid samples were called misspecified; the "
            "nominal rate at the 0.05 threshold is 2"
        )

    def test_the_null_sample_leaves_roughly_normal_residuals(self):
        result = garch_volatility_forecast(_iid(n=2000, seed=7))
        assert abs(result["standardized_skew"]) < 0.5
        assert abs(result["standardized_kurtosis"]) < 1.0


class TestClusteringTheFitCannotRemove:
    def test_a_cycling_variance_is_flagged_while_the_optimizer_succeeds(self):
        result = garch_volatility_forecast(_cycling_variance(seed=0))
        assert result["converged"] is True
        assert result["misspecified"] is True
        assert result["ljung_box_squared_p"] < 0.05

    def test_it_is_flagged_on_every_seed_not_one_lucky_one(self):
        flagged = sum(
            garch_volatility_forecast(_cycling_variance(seed=s))["misspecified"]
            for s in range(N_SEEDS)
        )
        assert flagged >= 36, f"detected on only {flagged}/{N_SEEDS} samples"

    def test_converged_alone_would_have_reported_these_as_healthy(self):
        """The whole point: the two flags are independent, and the one that
        was reported is the one that says nothing about the model."""
        converged = sum(
            garch_volatility_forecast(_cycling_variance(seed=s))["converged"]
            for s in range(N_SEEDS)
        )
        assert converged == N_SEEDS


class TestTheTwoTestsAnswerDifferentQuestions:
    def test_autocorrelated_returns_move_the_raw_test_and_not_the_squared_one(
        self,
    ):
        """
        An AR(1) in the MEAN is not a variance failure. The raw
        standardized-residual test should see it -- this model assumes a
        constant mean -- and the squared test, which is about the variance
        equation, should stay quiet.
        """
        result = garch_volatility_forecast(_ar1_in_mean(seed=0))
        assert result["ljung_box_p"] < 0.01
        assert result["ljung_box_squared_p"] > 0.05
        assert result["misspecified"] is False

    def test_the_separation_holds_across_seeds(self):
        raw_hits = 0
        squared_hits = 0
        for seed in range(N_SEEDS):
            result = garch_volatility_forecast(_ar1_in_mean(seed=seed))
            raw_hits += result["ljung_box_p"] < 0.05
            squared_hits += result["ljung_box_squared_p"] < 0.05
        assert raw_hits == N_SEEDS, f"the mean test missed {N_SEEDS - raw_hits}"
        assert squared_hits <= 12, (
            f"the variance test fired on {squared_hits}/{N_SEEDS} samples whose "
            "variance is constant"
        )


class TestTheVariancePathIsReturned:
    def test_it_is_one_value_per_observation_on_the_returns_index(self):
        returns = _iid(n=400, seed=3)
        result = garch_volatility_forecast(returns)
        series = result["conditional_variance"]
        assert isinstance(series, pd.Series)
        assert len(series) == len(returns) == result["n_obs"]
        assert series.index.equals(returns.index)
        assert (series > 0).all()

    def test_a_ragged_series_is_indexed_by_the_bars_that_survived(self):
        returns = _iid(n=400, seed=4)
        returns.iloc[5] = np.nan
        result = garch_volatility_forecast(returns)
        assert result["n_obs"] == 399
        assert result["conditional_variance"].index.equals(returns.dropna().index)


class TestTheRollingSharpeSeriesIsReturned:
    @staticmethod
    def _returns(n: int = 600, seed: int = 11) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(rng.normal(0.0004, 0.01, n), index=_dates(n))

    def test_it_carries_one_value_per_window(self):
        returns = self._returns()
        window = 60
        result = rolling_sharpe_stability(returns, window=window)
        series = result["rolling_sharpe"]
        assert len(series) == len(returns) - window + 1
        assert len(series) == result["n_windows"]

    def test_each_value_is_that_window_mean_over_its_std_annualized(self):
        returns = self._returns()
        window = 60
        series = rolling_sharpe_stability(returns, window=window)["rolling_sharpe"]
        for position in (0, 137, len(series) - 1):
            bars = returns.iloc[position : position + window]
            expected = bars.mean() / bars.std(ddof=1) * np.sqrt(TRADING_DAYS)
            assert series.iloc[position] == pytest.approx(expected, rel=1e-12)

    def test_each_value_is_labelled_by_the_bar_its_window_ends_on(self):
        returns = self._returns()
        window = 60
        series = rolling_sharpe_stability(returns, window=window)["rolling_sharpe"]
        assert series.index[0] == returns.index[window - 1]
        assert series.index[-1] == returns.index[-1]

    def test_the_scalars_beside_it_summarize_it(self):
        result = rolling_sharpe_stability(self._returns(), window=60)
        series = result["rolling_sharpe"]
        assert result["mean_rolling_sharpe"] == pytest.approx(float(series.mean()))
        assert result["min_rolling_sharpe"] == pytest.approx(float(series.min()))
        assert result["max_rolling_sharpe"] == pytest.approx(float(series.max()))

    def test_a_flat_series_is_still_refused_before_anything_is_built(self):
        """The series being returned does not soften the refusal: a constant
        return stream has no rolling Sharpe to look at."""
        flat = pd.Series(0.001, index=_dates(600))
        with pytest.raises(ValidationError, match="no stability to assess"):
            rolling_sharpe_stability(flat, window=60)

    def test_the_prose_names_what_comes_back(self):
        """Both sentences asserted the series was returned while it was not.
        They now name the key, so the claim is checkable from the text."""
        assert "rolling_sharpe" in rolling_sharpe_stability.__doc__
        warnings = rolling_sharpe_stability(self._returns(), window=60)["warnings"]
        assert any("rolling_sharpe" in w for w in warnings)
