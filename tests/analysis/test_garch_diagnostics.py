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


def _cycling_variance(
    n: int = 600, seed: int = 0, period: int = 10, amplitude: float = 0.9
) -> pd.Series:
    """
    Volatility that cycles on a fixed 10-bar period: quiet, loud, quiet.

    GARCH(1,1) models variance as geometric decay from the last shock. A
    cycle is not geometric decay, so the fit converges onto the best
    single-decay approximation of it and the clustering survives into the
    squared standardized residuals. That is the case the diagnostic exists
    for -- the optimizer succeeded and the model is the wrong one.

    The cycle used to be 20 bars at amplitude 0.95. On that one the
    likelihood's maximum lies at persistence >= 1 on 31 of 40 seeds -- a
    slow cycle looks like a variance that never mean-reverts -- so a fit
    that reaches the maximum rightly reports `converged` False there. It
    read True on all 40 only because the optimizer stopped near its
    starting point and nothing checked the gradient. The 10-bar cycle has
    an interior maximum on every seed measured, which is what these tests
    need: a fit that converged and is still the wrong model.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    vol = 0.01 * (1.0 + amplitude * np.sin(2 * np.pi * t / period))
    return pd.Series(rng.normal(0.0, 1.0, n) * vol, index=_dates(n))


def _simulated_garch(
    n: int = 2000,
    seed: int = 0,
    omega: float = 2e-6,
    alpha: float = 0.15,
    beta: float = 0.80,
) -> pd.Series:
    """GARCH(1,1) with normal innovations: the model's own case."""
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n)
    variance = omega / (1.0 - alpha - beta)
    values = np.empty(n)
    for t in range(n):
        values[t] = np.sqrt(variance) * shocks[t]
        variance = omega + alpha * values[t] ** 2 + beta * variance
    return pd.Series(values, index=_dates(n))


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

        This used to require all 40 fits to converge. They did only because
        the optimizer stopped a few iterations from its start on every
        sample and `converged` never looked at the gradient. A fit that
        reaches the maximum finds that on a few iid samples the likelihood
        is highest at the edge of the parameter space -- alpha at zero, or
        persistence at 1, a sample whose variance drifts a little -- and
        says so. What must hold is that every fit either converged or went
        to that edge, which is where no ARCH effect puts it.
        """
        converged = 0
        flagged = 0
        for seed in range(N_SEEDS):
            result = garch_volatility_forecast(_iid(seed=seed))
            converged += result["converged"]
            flagged += result["misspecified"]
            assert result["converged"] or (
                result["alpha"] < 1e-3 or result["persistence"] >= 1.0
            ), (
                f"seed {seed}: not converged away from the no-ARCH edge "
                f"(alpha {result['alpha']:.4f}, persistence "
                f"{result['persistence']:.4f})"
            )
        assert converged >= 32, f"only {converged}/{N_SEEDS} fits converged"
        assert flagged <= 6, (
            f"{flagged}/{N_SEEDS} iid samples were called misspecified; the "
            "nominal rate at the 0.05 threshold is 2"
        )

    def test_an_iid_sample_finds_no_arch_effect(self):
        """
        The null for the parameters themselves. With no clustering the
        typical fitted alpha is near zero; a fit that stops near its 0.05
        starting point reports a median alpha around 0.05 whatever the
        sample, and that is the number the old fit returned.
        """
        alphas = [
            garch_volatility_forecast(_iid(seed=seed))["alpha"]
            for seed in range(N_SEEDS)
        ]
        assert float(np.median(alphas)) < 0.02
        assert max(alphas) < 0.15

    def test_alpha_on_its_floor_is_named_and_warned_about(self):
        """
        Seed 0 is one of the iid samples whose maximum puts alpha on its
        lower bound. Beta is then not identified -- the variance path is
        the constant omega / (1 - beta) for any beta -- so the reported
        0.99 carries no information, and the result has to say that rather
        than leave a persistence that reads like a strongly clustered
        series.
        """
        result = garch_volatility_forecast(_iid(seed=0))
        assert result["converged"] is True
        assert "alpha" in result["at_bound"]
        assert any(
            "no ARCH effect" in w and "not identified" in w for w in result["warnings"]
        )

    def test_a_clustered_sample_has_no_bound_and_no_such_warning(self):
        result = garch_volatility_forecast(_simulated_garch(seed=3))
        assert result["converged"] is True
        assert result["at_bound"] == []
        assert result["warnings"] == []

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
        was reported is the one that says nothing about the model. Every
        seed of the 10-bar cycle reaches an interior maximum; see
        `_cycling_variance` for why the fixture is no longer the 20-bar
        one."""
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


class TestTheToolCarriesTheFitsOwnVerdict:
    """`run_garch_volatility_forecast` reports the gradient, the bounds and
    the fit's warnings, not only the flags it builds from the residuals."""

    @staticmethod
    def _run(monkeypatch, returns: pd.Series):
        from unittest.mock import MagicMock

        from standard_quant_tools.agent.models import GarchVolatilityForecastInput
        from standard_quant_tools.agent.tools import run_garch_volatility_forecast
        from standard_quant_tools.data.factory import DataFactory

        close = 100.0 * np.cumprod(1.0 + returns)
        bars = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close},
            index=returns.index,
        ).assign(Volume=1e6)
        stub = MagicMock()
        stub.get_ohlcv.return_value = bars
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: stub)
        return run_garch_volatility_forecast(
            GarchVolatilityForecastInput(
                symbol="AAA",
                start_date=str(returns.index[0].date()),
                end_date=str(returns.index[-1].date()),
            )
        )

    def test_alpha_on_its_floor_reaches_the_agent(self, monkeypatch):
        result = self._run(monkeypatch, _iid(seed=0))
        assert result.converged is True
        assert result.gradient_norm < 1e-4
        assert "alpha" in result.at_bound
        assert any("no ARCH effect" in w for w in result.warnings)

    def test_a_clustered_series_carries_no_such_warning(self, monkeypatch):
        result = self._run(monkeypatch, _simulated_garch(seed=3))
        assert result.converged is True
        assert result.gradient_norm < 1e-4
        assert result.at_bound == []
        assert not any(
            "no ARCH effect" in w or "NOT CONVERGED" in w for w in result.warnings
        )
