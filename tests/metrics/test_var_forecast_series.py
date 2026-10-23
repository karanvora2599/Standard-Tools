"""
One-step-ahead VaR and ES, each from the returns before its date.

`forecast_var_series` makes the forecast for date t from the `window`
returns before t by one of six methods. What is held here, each against an
answer computed independently of the code under test:

- THE PAST ONLY. Changing any return at or after t changes no forecast at
  or before t, for every method -- the GARCH ones, whose refits and filters
  are the easiest place for the future to leak in, included.
- `historical` is `var_historical` and `cvar` on each window, to the bit,
  with `cvar_tail_observations`' count.
- `normal` is -mu + sigma z_c and -mu + sigma phi(z_c) / (1 - c) on each
  window's mean and sample standard deviation.
- `ewma` is the RiskMetrics recursion written out by hand.
- `cornish_fisher`: its domain test agrees with a brute-force check that
  the expansion is increasing, its ES with a numerical integral of the
  expansion's quantile function, and a window outside the domain is
  refused rather than forecast.
- `filtered_historical` at a refit date uses exactly the volatility
  `garch_volatility_forecast` forecasts for that window, and `evt_garch`
  is `evt_tail_risk` on the same standardized residuals; too few
  exceedances is refused with the count.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from standard_quant_tools.analysis.garch import garch_volatility_forecast
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import (
    cvar,
    cvar_tail_observations,
    evt_tail_risk,
    var_historical,
)
from standard_quant_tools.metrics.var_forecast import (
    VAR_FORECAST_METHODS,
    cornish_fisher_is_monotone,
    cornish_fisher_tail_mean,
    cornish_fisher_z,
    forecast_var_series,
)


def _garch_returns(n: int = 700, seed: int = 3) -> pd.Series:
    """A GARCH(1,1) path with Student-t shocks: clustered and fat-tailed,
    so every method has something to do."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    out = np.empty(n)
    variance = omega / (1 - alpha - beta)
    for t in range(n):
        out[t] = math.sqrt(variance) * rng.standard_t(6) / math.sqrt(1.5)
        variance = omega + alpha * out[t] ** 2 + beta * variance
    return pd.Series(out, index=pd.bdate_range("2015-01-02", periods=n))


#: Arguments per method that make each one forecast the planted path.
_SETTINGS = {
    "historical": dict(window=250),
    "normal": dict(window=250),
    "cornish_fisher": dict(window=250, confidence=0.95),
    "ewma": dict(window=250),
    "filtered_historical": dict(window=300, refit_every=7),
    "evt_garch": dict(window=300, refit_every=7),
}


class TestOnlyThePastIsRead:
    @pytest.mark.parametrize("method", VAR_FORECAST_METHODS)
    def test_changing_the_future_changes_no_forecast_up_to_it(self, method):
        """The forecast for date t reads the window before t and nothing
        else. Every return from position `cut` on is replaced -- including
        the return ON the cut date, which its own forecast must not see --
        and every forecast at or before the cut is the same bits."""
        returns = _garch_returns()
        settings = _SETTINGS[method]
        before = forecast_var_series(returns, method=method, **settings)
        cut = settings["window"] + 150
        changed = returns.copy()
        # Another path of the same process, so every later window is one
        # each method still forecasts: a much wider shock, or a thin-tailed
        # one, puts Cornish-Fisher's windows outside its domain, which is a
        # refusal rather than a leak.
        other = _garch_returns(seed=11).to_numpy()
        changed.iloc[cut:] = other[cut:]
        after = forecast_var_series(changed, method=method, **settings)
        date = returns.index[cut]
        old = before["forecasts"].loc[:date]
        new = after["forecasts"].loc[:date]
        assert len(old) == cut - settings["window"] + 1
        np.testing.assert_array_equal(old["var"].to_numpy(), new["var"].to_numpy())
        np.testing.assert_array_equal(old["es"].to_numpy(), new["es"].to_numpy())
        # ...and the change is real: forecasts after the cut moved.
        later = slice(returns.index[cut + 5], None)
        assert not np.array_equal(
            before["forecasts"].loc[later, "var"].to_numpy(),
            after["forecasts"].loc[later, "var"].to_numpy(),
        )

    def test_each_row_lines_the_forecast_up_with_the_return_that_followed(self):
        """Row t holds r_t beside the forecast made from r_{t-W..t-1}: the
        first forecast date is the (W+1)-th return's."""
        returns = _garch_returns(400)
        out = forecast_var_series(returns, method="historical", window=250)
        frame = out["forecasts"]
        assert frame.index[0] == returns.index[250]
        assert out["n_forecasts"] == 150 == len(frame)
        np.testing.assert_array_equal(
            frame["realized"].to_numpy(), returns.iloc[250:].to_numpy()
        )
        assert set(frame["confidence"]) == {0.99}


class TestHistorical:
    def test_each_forecast_is_var_historical_and_cvar_on_its_window(self):
        returns = _garch_returns(400)
        out = forecast_var_series(returns, method="historical", window=250)
        frame = out["forecasts"]
        tails = []
        for i, date in enumerate(frame.index[::17]):
            t = returns.index.get_loc(date)
            window = returns.iloc[t - 250 : t]
            assert frame.loc[date, "var"] == var_historical(window, 0.99)
            assert frame.loc[date, "es"] == cvar(window, 0.99)
            tails.append(cvar_tail_observations(window, 0.99))
        assert out["diagnostics"]["min_tail_observations"] <= min(tails)

    def test_the_next_forecast_is_from_the_last_window(self):
        """The forecast after the last return has no date: it is reported
        as next_var / next_es, from the last 250 returns."""
        returns = _garch_returns(400)
        out = forecast_var_series(returns, method="historical", window=250)
        last = returns.iloc[-250:]
        assert out["next_var"] == var_historical(last, 0.99)
        assert out["next_es"] == cvar(last, 0.99)

    def test_a_short_tail_says_how_many_days_es_averages(self):
        """250 returns at 99% leave three at or below the quantile."""
        out = forecast_var_series(_garch_returns(400), method="historical")
        assert out["diagnostics"]["min_tail_observations"] == 3
        assert any("as few as 3 of 250" in w for w in out["warnings"])


class TestNormal:
    def test_each_forecast_is_the_normal_closed_form_on_its_window(self):
        returns = _garch_returns(400)
        out = forecast_var_series(
            returns, method="normal", window=120, confidence=0.975
        )
        frame = out["forecasts"]
        z = stats.norm.ppf(0.975)
        for date in frame.index[::23]:
            t = returns.index.get_loc(date)
            window = returns.iloc[t - 120 : t].to_numpy()
            mu, sigma = window.mean(), window.std(ddof=1)
            assert frame.loc[date, "var"] == pytest.approx(-mu + sigma * z, rel=1e-13)
            assert frame.loc[date, "es"] == pytest.approx(
                -mu + sigma * stats.norm.pdf(z) / 0.025, rel=1e-13
            )


class TestEwma:
    def test_the_forecast_is_the_recursion_written_out(self):
        """sigma^2 for the first forecast date is the mean square of the
        first window; each later one is lambda sigma^2 + (1 - lambda) r^2 of
        the day before. VaR = z sigma and ES = phi(z) sigma / (1 - c), the
        mean taken as zero (RiskMetrics)."""
        returns = _garch_returns(300)
        lam, window = 0.97, 60
        out = forecast_var_series(
            returns, method="ewma", window=window, ewma_lambda=lam, confidence=0.95
        )
        r = returns.to_numpy()
        z = stats.norm.ppf(0.95)
        variance = sum(x * x for x in r[:window]) / window
        expected_var, expected_es = [], []
        for t in range(window, len(r) + 1):
            if t > window:
                variance = lam * variance + (1 - lam) * r[t - 1] ** 2
            expected_var.append(math.sqrt(variance) * z)
            expected_es.append(math.sqrt(variance) * stats.norm.pdf(z) / 0.05)
        frame = out["forecasts"]
        np.testing.assert_allclose(frame["var"], expected_var[:-1], rtol=1e-12)
        np.testing.assert_allclose(frame["es"], expected_es[:-1], rtol=1e-12)
        assert out["next_var"] == pytest.approx(expected_var[-1], rel=1e-12)
        assert out["diagnostics"]["ewma_lambda"] == lam


class TestCornishFisher:
    def test_the_domain_is_where_the_expansion_increases(self):
        """The closed-form region against a brute-force check: the expansion
        evaluated on a fine grid of z is non-decreasing, or it is not."""
        z = np.linspace(-10, 10, 8001)
        rng = np.random.default_rng(0)
        agree = 0
        cases = 3000
        for _ in range(cases):
            s, k = rng.uniform(-2.5, 2.5), rng.uniform(-1.5, 10)
            increasing = bool(np.all(np.diff(cornish_fisher_z(z, s, k)) >= 0))
            agree += increasing == cornish_fisher_is_monotone(s, k)
        # A handful of cases sit within a grid step of the boundary.
        assert agree >= cases - 5

    def test_known_points_inside_and_outside(self):
        assert cornish_fisher_is_monotone(0.0, 0.0)
        assert cornish_fisher_is_monotone(-0.5, 3.0)
        # Negative excess kurtosis makes the cubic term fall: outside.
        assert not cornish_fisher_is_monotone(0.0, -0.5)
        # Skew too large for the kurtosis: outside.
        assert not cornish_fisher_is_monotone(2.0, 1.0)

    @pytest.mark.parametrize("s,k", [(0.0, 0.0), (-0.6, 2.5), (0.4, 1.2)])
    def test_the_tail_mean_is_the_integral_of_the_quantile(self, s, k):
        """(1/p) integral_0^p z_cf(Phi^-1(u)) du, by quadrature, against
        the truncated-normal moments the code uses."""
        p = 0.025
        quad, _err = integrate.quad(
            lambda u: cornish_fisher_z(stats.norm.ppf(u), s, k), 0, p, limit=200
        )
        assert cornish_fisher_tail_mean(p, s, k) == pytest.approx(quad / p, rel=1e-8)

    def test_with_no_skew_or_kurtosis_it_is_the_normal(self):
        p = 0.01
        z = stats.norm.ppf(p)
        assert cornish_fisher_z(z, 0.0, 0.0) == z
        assert cornish_fisher_tail_mean(p, 0.0, 0.0) == pytest.approx(
            -stats.norm.pdf(z) / p, rel=1e-12
        )

    def test_a_window_outside_the_domain_is_refused(self):
        """Mostly flat days with rare large gains: strong positive skew and
        a light left tail, which no monotone expansion describes."""
        rng = np.random.default_rng(4)
        values = rng.uniform(-0.002, 0.002, 400)
        values[::40] = 0.08
        returns = pd.Series(values, index=pd.bdate_range("2020-01-02", periods=400))
        window = returns.iloc[:250].to_numpy()
        centred = window - window.mean()
        s = np.mean(centred**3) / np.mean(centred**2) ** 1.5
        k = np.mean(centred**4) / np.mean(centred**2) ** 2 - 3
        assert not cornish_fisher_is_monotone(s, k)
        with pytest.raises(ValidationError, match="outside the region"):
            forecast_var_series(returns, method="cornish_fisher", window=250)

    def test_inside_the_domain_var_is_the_expansion(self):
        returns = _garch_returns(400)
        out = forecast_var_series(
            returns, method="cornish_fisher", window=250, confidence=0.95
        )
        date = out["forecasts"].index[10]
        t = returns.index.get_loc(date)
        w = returns.iloc[t - 250 : t].to_numpy()
        c = w - w.mean()
        s = np.mean(c**3) / np.mean(c**2) ** 1.5
        k = np.mean(c**4) / np.mean(c**2) ** 2 - 3
        zc = cornish_fisher_z(stats.norm.ppf(0.05), s, k)
        expected = -(w.mean() + w.std(ddof=1) * zc)
        assert out["forecasts"].loc[date, "var"] == pytest.approx(expected, rel=1e-12)


class TestGarchFiltered:
    def test_at_a_refit_date_the_volatility_is_the_garch_forecast(self):
        """The first forecast refits on its window, so its volatility is
        garch_volatility_forecast's one-step-ahead one for that window, and
        the FHS VaR is -mu + sigma times minus the residuals' quantile."""
        returns = _garch_returns(400)
        out = forecast_var_series(
            returns, method="filtered_historical", window=300, refit_every=50
        )
        window = returns.iloc[:300]
        fit = garch_volatility_forecast(window, forecast_horizon=1)
        sigma = fit["current_annualized_vol"] / math.sqrt(252)
        mu = window.mean()
        z = (window - mu) / np.sqrt(fit["conditional_variance"])
        q = np.percentile(z.to_numpy(), 1.0)
        tail = z[z <= q]
        first = out["forecasts"].iloc[0]
        assert first["var"] == pytest.approx(-mu - sigma * q, rel=1e-9)
        assert first["es"] == pytest.approx(-mu - sigma * tail.mean(), rel=1e-9)
        assert out["diagnostics"]["garch_fits"] == math.ceil(101 / 50)

    def test_evt_garch_is_evt_tail_risk_on_the_standardized_residuals(self):
        returns = _garch_returns(400)
        out = forecast_var_series(
            returns, method="evt_garch", window=300, refit_every=50, tail_fraction=0.1
        )
        window = returns.iloc[:300]
        fit = garch_volatility_forecast(window, forecast_horizon=1)
        sigma = fit["current_annualized_vol"] / math.sqrt(252)
        mu = window.mean()
        z = (window - mu) / np.sqrt(fit["conditional_variance"])
        tail = evt_tail_risk(
            pd.Series(z.to_numpy()), confidence=0.99, tail_fraction=0.1
        )
        first = out["forecasts"].iloc[0]
        assert first["var"] == pytest.approx(-mu + sigma * tail["var_evt"], rel=1e-9)
        assert first["es"] == pytest.approx(-mu + sigma * tail["cvar_evt"], rel=1e-9)
        assert out["diagnostics"]["min_exceedances"] == tail["n_exceedances"] == 30

    def test_too_few_exceedances_is_refused_with_the_count(self):
        """150 standardized losses at tail_fraction 0.1 leave 15 above the
        threshold; the tail is fitted to 20 or more."""
        with pytest.raises(ValidationError, match="has 15 standardized losses"):
            forecast_var_series(
                _garch_returns(300), method="evt_garch", window=150, tail_fraction=0.1
            )

    def test_a_level_the_fitted_tail_cannot_reach_is_refused(self):
        with pytest.raises(ValidationError, match="can only say what lies"):
            forecast_var_series(
                _garch_returns(400),
                method="evt_garch",
                window=300,
                confidence=0.85,
                tail_fraction=0.1,
            )


class TestRefusals:
    def test_a_window_too_short_for_the_garch_is_refused(self):
        with pytest.raises(ValidationError, match="at least 100"):
            forecast_var_series(
                _garch_returns(300), method="filtered_historical", window=60
            )

    def test_a_history_no_longer_than_the_window_is_refused(self):
        with pytest.raises(ValidationError, match="at least 251 are needed"):
            forecast_var_series(_garch_returns(250), window=250)

    def test_an_unsorted_or_gapped_series_is_refused(self):
        returns = _garch_returns(300)
        with pytest.raises(ValidationError, match="date order"):
            forecast_var_series(returns.iloc[::-1], window=50)
        gapped = returns.copy()
        gapped.iloc[10] = np.nan
        with pytest.raises(ValidationError, match="non-finite"):
            forecast_var_series(gapped, window=50)

    def test_an_unknown_method_is_refused(self):
        with pytest.raises(ValidationError, match="not one of"):
            forecast_var_series(_garch_returns(300), method="monte_carlo")
