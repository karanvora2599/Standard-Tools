"""
The Kelly leverage of a return stream, held to its closed forms and to
simulation.

f* = mu / sigma^2 for one stream and Sigma^-1 mu for several, each
recomputed here from the sample by hand; its standard error 1 / (sigma
sqrt(T)) against the spread of f-hat over simulated samples; the chance the
estimate is over twice the truth, Phi(-S sqrt(T)), against the share of
simulated estimates that are; the probability of ever losing d at c of
Kelly, (1 - d)^(2/c - 1), against simulated paths monitored continuously;
the empirical Kelly of a two-outcome sample against p/L - q/W, the same
closed form `get_position_size` reports for one trade. A sample built to
have a Sharpe ratio of exactly 0.5 over exactly five years reproduces the
growth shares 0.20 at full Kelly and 0.55 at half.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.allocation.kelly import (
    drawdown_probability,
    empirical_kelly,
    kelly_report,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.position_sizing import kelly_notional_fraction

P = 252


def _exact(n, sharpe, sigma, seed=0, start="2015-01-02"):
    """Daily returns whose sample mean and sd are EXACTLY the annual sharpe
    x sigma and sigma: noise standardized, then scaled."""
    z = np.random.default_rng(seed).standard_normal(n)
    z = (z - z.mean()) / z.std(ddof=1)
    values = sharpe * sigma / P + sigma / math.sqrt(P) * z
    return pd.Series(values, index=pd.bdate_range(start, periods=n), name="s")


def _dated(series):
    return {d.strftime("%Y-%m-%d"): float(v) for d, v in series.items()}


def _call(arguments):
    return resolve("allocation").dispatch("get_kelly_report", arguments)


class TestTheFractionIsTheClosedForm:
    def test_one_stream_is_mu_over_sigma_squared(self):
        """f* = mean excess / variance of excess, both per period (the
        annualization cancels), at a risk-free rate of 3%."""
        r = _exact(1260, 0.7, 0.12, seed=1)
        out = kelly_report(r.to_frame(), periods_per_year=P, risk_free_rate=0.03)
        excess = r.to_numpy() - 0.03 / P
        expected = excess.mean() / excess.var(ddof=1)
        assert out["by_stream"]["s"]["kelly_fraction"] == pytest.approx(
            expected, rel=1e-12
        )

    def test_several_streams_are_sigma_inverse_mu(self):
        """Sigma^-1 mu, recomputed with an explicit inverse, agrees to 1e-12
        on three correlated streams."""
        rng = np.random.default_rng(2)
        mix = np.array([[1.0, 0.0, 0.0], [0.5, 0.9, 0.0], [0.2, -0.3, 0.8]])
        values = rng.normal(0, 0.01, (1500, 3)) @ mix.T + [0.0004, 0.0002, 0.0001]
        frame = pd.DataFrame(
            values,
            index=pd.bdate_range("2014-01-02", periods=1500),
            columns=list("xyz"),
        )
        out = kelly_report(
            frame, periods_per_year=P, risk_free_rate=0.01, n_bootstrap=0
        )
        excess = values - 0.01 / P
        mu = excess.mean(axis=0) * P
        sigma = np.cov(excess, rowvar=False, ddof=1) * P
        expected = np.linalg.inv(sigma) @ mu
        got = np.array([out["by_stream"][c]["kelly_fraction"] for c in "xyz"])
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
        # The standard errors are sqrt of the inverse's diagonal over T.
        se = np.array([out["by_stream"][c]["standard_error"] for c in "xyz"])
        np.testing.assert_allclose(
            se, np.sqrt(np.diag(np.linalg.inv(sigma)) / (1500 / P)), rtol=1e-12
        )
        assert out["sharpe_ratio"] == pytest.approx(math.sqrt(mu @ expected), rel=1e-12)

    def test_a_negative_edge_is_a_negative_fraction_never_zero(self):
        """A stream that loses money has f* < 0 -- Kelly shorts it -- and
        the report says so; zero would be the long-only answer, not this
        one."""
        r = _exact(1260, -0.4, 0.15, seed=3)
        out = kelly_report(r.to_frame(), periods_per_year=P, n_bootstrap=0)
        row = out["by_stream"]["s"]
        assert row["kelly_fraction"] == pytest.approx(-0.4 / 0.15, rel=1e-12)
        assert row["kelly_fraction"] < 0 and row["shrunk_fraction"] < 0
        assert any("SHORTS" in w for w in out["warnings"])
        assert out["by_fraction"][0]["leverage"]["s"] < 0


class TestTheUncertaintyMatchesSimulation:
    S, SIGMA, YEARS, DRAWS = 0.5, 0.10, 5, 4000

    @pytest.fixture(scope="class")
    def estimates(self):
        """f-hat over 4,000 independent five-year daily samples of a stream
        whose true f* is 5, computed as the library computes it for one
        stream (checked against the library on the first sample)."""
        n = self.YEARS * P
        rng = np.random.default_rng(11)
        mu, sigma = self.S * self.SIGMA, self.SIGMA
        samples = rng.normal(mu / P, sigma / math.sqrt(P), (self.DRAWS, n))
        f_hat = samples.mean(axis=1) / samples.var(axis=1, ddof=1)
        first = kelly_report(
            pd.DataFrame(
                {"s": samples[0]}, index=pd.bdate_range("2010-01-04", periods=n)
            ),
            periods_per_year=P,
            n_bootstrap=0,
        )
        assert first["by_stream"]["s"]["kelly_fraction"] == pytest.approx(
            f_hat[0], rel=1e-12
        )
        return f_hat

    def test_the_standard_error_is_the_spread_of_the_estimate(self, estimates):
        """1 / (sigma sqrt(T)) = 4.472 against the simulated sd of f-hat.
        The tolerance is four standard errors of an sd from 4,000 draws
        plus the sample's own error in sigma (1 / sqrt(2n))."""
        r = _exact(self.YEARS * P, self.S, self.SIGMA, seed=4)
        se = kelly_report(r.to_frame(), periods_per_year=P, n_bootstrap=0)["by_stream"][
            "s"
        ]["standard_error"]
        assert se == pytest.approx(1 / (self.SIGMA * math.sqrt(self.YEARS)), rel=1e-12)
        simulated = float(np.std(estimates, ddof=1))
        tolerance = 4 * se / math.sqrt(2 * self.DRAWS) + se / math.sqrt(
            2 * self.YEARS * P
        )
        assert abs(simulated - se) < tolerance

    def test_the_chance_of_twice_the_truth_is_phi_of_minus_s_root_t(self, estimates):
        """Phi(-0.5 sqrt(5)) = 0.132 against the share of the 4,000 simulated
        estimates above 2 f* = 10, within four binomial standard errors;
        the report gives the same formula at the sample's own Sharpe."""
        closed = NormalDist().cdf(-self.S * math.sqrt(self.YEARS))
        assert closed == pytest.approx(0.1318, abs=5e-4)
        share = float(np.mean(estimates > 2 * self.S / self.SIGMA))
        assert abs(share - closed) < 4 * math.sqrt(closed * (1 - closed) / self.DRAWS)
        r = _exact(self.YEARS * P, self.S, self.SIGMA, seed=5)
        out = kelly_report(r.to_frame(), periods_per_year=P, n_bootstrap=0)
        assert out["p_estimate_over_twice_truth"] == pytest.approx(closed, rel=1e-12)
        assert out["by_stream"]["s"]["p_estimate_over_twice_truth"] == pytest.approx(
            closed, rel=1e-12
        )


class TestGrowthAtFractions:
    def test_the_planted_sharpe_gives_twenty_and_fifty_five_percent(self):
        """Sharpe exactly 0.5 over exactly 5 years: full Kelly expects 20%
        of the optimal growth once the estimate's error is counted, half
        Kelly 55%; shrinkage Shat^2 T / (1 + Shat^2 T) = 1.25 / 2.25."""
        r = _exact(5 * P, 0.5, 0.10, seed=6)
        out = kelly_report(r.to_frame(), periods_per_year=P, n_bootstrap=0)
        by_c = {row["fraction_of_kelly"]: row for row in out["by_fraction"]}
        assert by_c[1.0]["share_of_optimal_growth"] == pytest.approx(0.20, abs=1e-12)
        assert by_c[0.5]["share_of_optimal_growth"] == pytest.approx(0.55, abs=1e-12)
        assert by_c[0.25]["share_of_optimal_growth"] == pytest.approx(
            2 * 0.25 - 0.25**2 - 0.25**2 / 1.25, abs=1e-12
        )
        assert by_c[1.0]["excess_growth_rate"] == pytest.approx(0.125, rel=1e-12)
        assert out["optimal_excess_growth_rate"] == pytest.approx(0.125, rel=1e-12)
        assert out["shrink_factor"] == pytest.approx(1.25 / 2.25, rel=1e-12)
        assert out["by_stream"]["s"]["shrunk_fraction"] == pytest.approx(
            5.0 * 1.25 / 2.25, rel=1e-12
        )

    def test_twice_kelly_grows_no_faster_than_cash(self):
        """g(2 f*) = 0 in the model, and beyond it negative."""
        r = _exact(5 * P, 0.8, 0.15, seed=7)
        out = kelly_report(
            r.to_frame(), periods_per_year=P, fractions=[2.0, 3.0], n_bootstrap=0
        )
        assert out["by_fraction"][0]["excess_growth_rate"] == pytest.approx(
            0, abs=1e-15
        )
        assert out["by_fraction"][1]["excess_growth_rate"] < 0


class TestTheDrawdownOdds:
    def test_the_closed_form_is_the_probe_values(self):
        """(1-d)^(2/c-1): a half at full Kelly, an eighth at half Kelly, and
        1 from twice Kelly on."""
        assert drawdown_probability(1.0, 0.5) == pytest.approx(0.5, abs=1e-15)
        assert drawdown_probability(0.5, 0.5) == pytest.approx(0.125, abs=1e-15)
        assert drawdown_probability(0.5, 0.2) == pytest.approx(0.512, abs=1e-15)
        assert drawdown_probability(2.0, 0.1) == 1.0
        assert drawdown_probability(3.0, 0.1) == 1.0

    @pytest.mark.parametrize("c,d", [(1.0, 0.5), (0.5, 0.2), (0.5, 0.5)])
    def test_the_closed_form_against_continuously_monitored_paths(self, c, d):
        """Log wealth at c of Kelly is Brownian with drift S^2 (c - c^2/2) and
        volatility c S. Simulated monthly for 100 years with the Brownian-
        bridge chance of crossing inside each month -- exact for the
        continuous path, which daily monitoring is not (it reads 0.475 for
        a true 0.500) -- the share of paths that ever lose d agrees with
        (1-d)^(2/c-1) within four standard errors of the simulation."""
        S, paths, steps, dt = 1.0, 4000, 1200, 1.0 / 12
        drift, vol = S * S * (c - c * c / 2), c * S
        barrier = math.log(1 - d)
        rng = np.random.default_rng(13)
        x = np.zeros(paths)
        survive = np.ones(paths)
        for _ in range(steps):
            step = drift * dt + vol * math.sqrt(dt) * rng.standard_normal(paths)
            y = x + step
            above = (x > barrier) & (y > barrier)
            cross = np.where(
                above,
                np.exp(-2 * (x - barrier) * (y - barrier) / (vol * vol * dt)),
                1.0,
            )
            survive *= 1 - cross
            x = y
        hit = 1 - survive
        simulated = float(hit.mean())
        tolerance = 4 * float(hit.std(ddof=1)) / math.sqrt(paths)
        assert abs(simulated - drawdown_probability(c, d)) < tolerance

    def test_the_report_carries_them_at_every_fraction(self):
        r = _exact(5 * P, 0.6, 0.1, seed=8)
        out = kelly_report(r.to_frame(), periods_per_year=P, n_bootstrap=0)
        for row in out["by_fraction"]:
            for level, p in row["drawdown_probability"].items():
                assert p == drawdown_probability(row["fraction_of_kelly"], float(level))


class TestTheEmpiricalKelly:
    def test_a_two_outcome_sample_gives_p_over_l_minus_q_over_w(self):
        """55 trades of +6% and 45 of -3%: the leverage that maximises
        realized log growth is p/L - q/W = 10.833, the growth-optimal
        notional `get_position_size` reports for the same statistics."""
        values = np.array([0.06] * 55 + [-0.03] * 45)
        found = empirical_kelly(values[:, None], 0.0)
        assert found is not None
        assert found[0] == pytest.approx(
            kelly_notional_fraction(0.55, 0.06, 0.03), rel=1e-9
        )

    def test_a_sample_that_never_lost_has_no_empirical_kelly(self):
        """More leverage always grew a sample with no losing period, so
        there is no maximum: null with the reason, never a number."""
        r = pd.Series(
            np.abs(np.random.default_rng(9).normal(0.001, 0.001, 300)),
            index=pd.bdate_range("2020-01-02", periods=300),
            name="s",
        )
        out = _call({"returns": {"s": _dated(r)}, "n_bootstrap": 0})
        assert out["empirical_kelly"] is None
        assert out["empirical_kelly_excess_growth_rate"] is None
        assert any("empirical_kelly" in w for w in out["warnings"])


class TestTheBootstrap:
    def test_the_same_seed_gives_the_same_bits_and_another_seed_others(self):
        r = _exact(3 * P, 0.7, 0.12, seed=10)
        args = {"returns": {"s": _dated(r)}, "n_bootstrap": 300}
        one, two = _call(args), _call(args)
        assert one["by_fraction"] == two["by_fraction"]
        other = _call({**args, "random_seed": 1})
        assert (
            other["by_fraction"][2]["max_drawdown_median"]
            != one["by_fraction"][2]["max_drawdown_median"]
        )

    def test_the_drawdowns_deepen_with_leverage(self):
        """Quarter, half and full Kelly on the same draws: each deeper."""
        r = _exact(5 * P, 0.8, 0.12, seed=12)
        out = kelly_report(r.to_frame(), periods_per_year=P, n_bootstrap=400)
        medians = [row["max_drawdown_median"] for row in out["by_fraction"]]
        assert medians[0] > medians[1] > medians[2] > -1
        assert all(
            row["max_drawdown_worst_5pct"] <= row["max_drawdown_median"]
            for row in out["by_fraction"]
        )

    def test_the_in_sample_drawdown_is_the_library_drawdown(self):
        """The path's maximum drawdown is `max_drawdown` of
        `equity_from_returns` at the leverage: the first period counts."""
        from standard_quant_tools.metrics.return_metrics import equity_from_returns
        from standard_quant_tools.metrics.risk_metrics import max_drawdown

        r = _exact(2 * P, 0.6, 0.15, seed=14)
        out = kelly_report(r.to_frame(), periods_per_year=P, n_bootstrap=0)
        for row in out["by_fraction"]:
            levered = row["leverage"]["s"] * r
            assert row["max_drawdown_in_sample"] == pytest.approx(
                max_drawdown(equity_from_returns(levered)), abs=1e-12
            )
            assert math.isnan(row["max_drawdown_median"])  # no draws asked for


class TestTheTool:
    def test_a_published_panel_reads_as_the_inline_streams(self):
        """returns_ref and the same streams inline: the same report."""
        rng = np.random.default_rng(15)
        frame = pd.DataFrame(
            rng.normal(0.0004, 0.01, (600, 2)),
            index=pd.bdate_range("2019-01-02", periods=600),
            columns=["a", "b"],
        )
        ref = handoff.publish(frame, kind="returns_panel", run_id="kelly_ref", name="p")
        by_ref = _call({"returns_ref": ref, "n_bootstrap": 50})
        inline = _call(
            {
                "returns": {c: _dated(frame[c]) for c in frame.columns},
                "n_bootstrap": 50,
            }
        )
        assert by_ref["by_stream"] == inline["by_stream"]
        assert by_ref["by_fraction"] == inline["by_fraction"]
        one = _call({"returns_ref": ref, "columns": ["b"], "n_bootstrap": 0})
        assert one["stream_names"] == ["b"]

    def test_misaligned_streams_meet_on_their_shared_dates(self):
        a = _exact(300, 0.5, 0.1, seed=16)
        b = _exact(300, 0.5, 0.1, seed=17, start="2015-01-09")
        out = _call({"returns": {"a": _dated(a), "b": _dated(b)}, "n_bootstrap": 0})
        assert out["n_periods"] == 295
        assert out["streams"]["n_dates"] == 295
        assert any("aligned" in w for w in out["warnings"])

    @pytest.mark.parametrize(
        "arguments,needle",
        [
            ({}, "exactly one of returns / returns_ref"),
            ({"columns": ["a"]}, "columns"),
            (
                {
                    "returns": None,
                    "returns_ref": "sqt://returns_panel/x/y",
                    "columns": ["a", "a"],
                },
                r"columns names \['a'\] more than once",
            ),
            ({"fractions": [0.0]}, "fractions"),
            ({"drawdown_levels": [1.0]}, "drawdown_levels"),
            ({"risk_free_rate": 1e308}, "risk_free_rate"),
        ],
    )
    def test_the_input_model_refuses_by_name(self, arguments, needle):
        base = {"returns": {"s": _dated(_exact(100, 0.5, 0.1))}}
        if not arguments:
            base = {}
        with pytest.raises(PydanticValidationError, match=needle):
            _call({**base, **arguments})

    def test_too_few_periods_and_a_still_stream_are_refused(self):
        with pytest.raises(ValidationError, match="fewer than min_overlap=20"):
            _call({"returns": {"s": _dated(_exact(15, 0.5, 0.1))}})
        with pytest.raises(ValidationError, match="needs at least 20"):
            kelly_report(_exact(15, 0.5, 0.1).to_frame(), periods_per_year=P)
        flat = pd.Series(0.001, index=pd.bdate_range("2020-01-02", periods=60))
        with pytest.raises(ValidationError, match="never moved"):
            kelly_report(flat.to_frame("s"), periods_per_year=P)

    def test_copies_of_one_stream_are_refused(self):
        """Two copies have a singular covariance: Sigma^-1 mu is not
        determined, and the refusal says why."""
        r = _exact(400, 0.5, 0.1, seed=18)
        frame = pd.DataFrame({"a": r, "b": r * 1.0000000001})
        with pytest.raises(ValidationError, match="condition number"):
            kelly_report(frame, periods_per_year=P)
