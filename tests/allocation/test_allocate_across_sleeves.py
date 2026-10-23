"""
`allocate_across_sleeves`: capital across return streams, and how much of
the answer the sample chose.

THE PLANTED WORLD. Eight sleeves in two correlation clusters (0.5 within,
0.1 across), annual volatilities 5% to 20%, Sharpe ratios 0.2 to 0.8, drawn
as daily normal returns from the known covariance. On it the true
equal-risk-contribution weights and the true inverse-volatility weights are
known exactly, and measured on 40 histories of each length the estimates
approach them as 1/sqrt(T): the ERC weights' mean L1 distance from the
truth is 0.053 at one year and 0.016 at ten, the inverse-volatility ones'
0.032 and 0.009.

Every other planted answer is a closed form or an independent oracle: equal
and inverse-volatility weights by hand, risk contributions equal to 1e-10,
the water-filled capacity by hand, the exact solvers against the critical
line, the James-Stein shrinkage against its chi-squared law and its full
pooling against the long-only maximum-diversification portfolio, and the
bootstrap bands' coverage of the true weight by simulation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pydantic
import pytest
from scipy import integrate, stats

from standard_quant_tools.agent.runtimes import handoff, resolve
from standard_quant_tools.allocation.sleeves import (
    SleeveRule,
    allocate_across_sleeves,
    fit_sleeve_rule,
    shrunk_sharpe_ratios,
    water_fill,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import annualized_sharpe
from standard_quant_tools.portfolio.construction import (
    max_diversification,
    risk_parity,
)
from standard_quant_tools.portfolio.covariance import estimate_covariance
from standard_quant_tools.portfolio.critical_line import critical_line

P = 252
N = 8
NAMES = [f"S{i}" for i in range(N)]
VOL = np.array([0.05, 0.08, 0.10, 0.12, 0.06, 0.15, 0.09, 0.20])
SHARPE = np.array([0.6, 0.3, 0.8, 0.4, 0.5, 0.2, 0.7, 0.5])
CORR = np.full((N, N), 0.1)
CORR[:4, :4] = 0.5
CORR[4:, 4:] = 0.5
np.fill_diagonal(CORR, 1.0)
SIGMA = np.outer(VOL, VOL) * CORR
ROOT = np.linalg.cholesky(SIGMA / P)


def _draw(rng, years, sharpe=SHARPE, periods=P):
    """`years` of the planted world at `periods` a year: daily by default,
    monthly where a long history is wanted cheaply -- a Sharpe ratio's
    sampling error is set by the years, not by how often they are sampled."""
    n = int(round(years * periods))
    root = ROOT if periods == P else np.linalg.cholesky(SIGMA / periods)
    values = (sharpe * VOL) / periods + rng.standard_normal((n, N)) @ root.T
    return pd.DataFrame(
        values, columns=NAMES, index=pd.bdate_range("2000-01-03", periods=n)
    )


@pytest.fixture(scope="module")
def three_years():
    return _draw(np.random.default_rng(20261022), 3)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


def _weights(out):
    return np.array([out["weights"][s] for s in NAMES])


def _covariance(frame, method="ledoit_wolf"):
    matrix = estimate_covariance(frame, method=method)["matrix"]
    return np.array([[matrix[a][b] for b in NAMES] for a in NAMES])


class TestClosedForms:
    def test_equal_weight_is_one_over_n_with_its_own_risk(self, three_years):
        """1/N exactly, and the risk read off the rule's covariance: the
        contributions w_i (Sigma w)_i / sigma sum to sigma = sqrt(w'Sigma w)."""
        out = allocate_across_sleeves(
            three_years, SleeveRule(method="equal"), n_bootstrap=0
        )
        w = np.full(N, 1.0 / N)
        assert _weights(out).tolist() == w.tolist()
        cov = _covariance(three_years)
        sigma = math.sqrt(w @ cov @ w)
        assert out["ex_ante_volatility"] == pytest.approx(sigma, rel=1e-13)
        contributions = np.array([out["risk_contributions"][s] for s in NAMES])
        np.testing.assert_allclose(contributions, w * (cov @ w) / sigma, rtol=1e-12)
        assert contributions.sum() == pytest.approx(sigma, rel=1e-12)
        assert out["effective_n"] == pytest.approx(8.0, rel=1e-14)
        assert out["cash_weight"] == pytest.approx(0.0, abs=1e-15)

    def test_inverse_volatility_is_one_over_each_sample_volatility(self, three_years):
        """On the sample covariance the weights are (1/s_i) / sum_j (1/s_j)
        with s the sample standard deviations; the annualization cancels."""
        out = allocate_across_sleeves(
            three_years,
            SleeveRule(method="inverse_volatility", covariance_method="sample"),
            n_bootstrap=0,
        )
        inverse = 1.0 / three_years.std(ddof=1).to_numpy()
        np.testing.assert_allclose(_weights(out), inverse / inverse.sum(), rtol=1e-12)

    def test_risk_parity_equalizes_the_risk_contributions_to_1e10(self, three_years):
        out = allocate_across_sleeves(
            three_years, SleeveRule(method="risk_parity"), n_bootstrap=0
        )
        shares = np.array([out["risk_shares"][s] for s in NAMES])
        assert np.max(np.abs(shares - 1.0 / N)) <= 1e-10
        assert out["effective_n_risk"] == pytest.approx(8.0, abs=1e-8)

    def test_risk_budgets_are_the_risk_shares(self, three_years):
        budgets = {s: float(i + 1) for i, s in enumerate(NAMES)}
        out = allocate_across_sleeves(
            three_years,
            SleeveRule(method="risk_parity", risk_budgets=budgets),
            n_bootstrap=0,
        )
        target = np.arange(1, N + 1) / np.arange(1, N + 1).sum()
        shares = np.array([out["risk_shares"][s] for s in NAMES])
        assert np.max(np.abs(shares - target)) <= 1e-10

    def test_the_in_sample_sharpe_is_the_allocations_own_on_the_sample(
        self, three_years
    ):
        """Named in-sample, and the annualized Sharpe ratio of sum_i w_i
        (r_i - rf) over the rows the weights were fitted on, to the rounding
        of a product whose summation order follows the frame's layout."""
        out = allocate_across_sleeves(
            three_years, SleeveRule(method="hrp"), cash_rate=0.02, n_bootstrap=0
        )
        excess = (three_years.to_numpy() - 0.02 / P) @ _weights(out)
        assert out["in_sample_sharpe"] == pytest.approx(
            annualized_sharpe(excess, P), rel=1e-13
        )


class TestTheWeightsConvergeToTheTrueOnes:
    @pytest.mark.parametrize(
        "method,one_year,ten_years",
        [("risk_parity", 0.053, 0.016), ("inverse_volatility", 0.032, 0.009)],
    )
    def test_the_l1_distance_falls_as_one_over_root_t(
        self, method, one_year, ten_years
    ):
        """On 40 histories each the mean L1 distance to the weights of the
        TRUE covariance is the measured one at both lengths (its standard
        error over 40 histories is under a tenth of it), and ten times the
        data leaves it sqrt(10) = 3.16 times smaller, within a quarter."""
        truth = (
            np.array(
                list(
                    risk_parity(pd.DataFrame(SIGMA, index=NAMES, columns=NAMES))[
                        "weights"
                    ].values()
                )
            )
            if method == "risk_parity"
            else (1.0 / VOL) / (1.0 / VOL).sum()
        )
        rng = np.random.default_rng(20261022)
        rule = SleeveRule(method=method, covariance_method="sample")
        distance = {}
        for years in (1, 10):
            distance[years] = np.mean(
                [
                    np.abs(
                        fit_sleeve_rule(_draw(rng, years), rule)["weights"] - truth
                    ).sum()
                    for _ in range(40)
                ]
            )
        assert distance[1] == pytest.approx(one_year, rel=0.2)
        assert distance[10] == pytest.approx(ten_years, rel=0.25)
        assert distance[1] / distance[10] == pytest.approx(math.sqrt(10), rel=0.25)


def _expected_shrinkage(k):
    """E[min(1, (k - 3) / Q)] and P(Q <= k - 3) for Q ~ chi-squared(k - 1)."""
    mean = integrate.quad(
        lambda q: min(1.0, (k - 3) / q) * stats.chi2.pdf(q, k - 1),
        0,
        np.inf,
        limit=200,
    )[0]
    return mean, stats.chi2.cdf(k - 3, k - 1)


class TestJamesSteinShrinkage:
    @pytest.mark.parametrize("years,periods", [(1, P), (40, 12)])
    def test_equal_sharpes_follow_the_chi_squared_law_at_any_length(
        self, years, periods
    ):
        """With every true Sharpe ratio 0.5, Q is chi-squared on k - 1 = 7
        degrees of freedom whatever the sample length, so the shrinkage c =
        min(1, 5 / Q) has mean 0.756 and is 1 -- full pooling -- with
        probability 0.340: on one year of days as on forty of months. Over
        400 histories the standard errors are about 0.013 and 0.024; the
        tolerances are three of them."""
        mean, full = _expected_shrinkage(N)
        assert (round(mean, 3), round(full, 3)) == (0.756, 0.34)
        rng = np.random.default_rng(5 + years)
        c = np.array(
            [
                shrunk_sharpe_ratios(
                    _draw(rng, years, np.full(N, 0.5), periods).to_numpy(),
                    CORR,
                    rf_per_period=0.0,
                    periods_per_year=periods,
                )["shrinkage"]
                for _ in range(400)
            ]
        )
        assert c.mean() == pytest.approx(mean, abs=0.04)
        assert (c == 1.0).mean() == pytest.approx(full, abs=0.072)

    def test_full_pooling_grows_with_the_number_of_sleeves(self):
        """E[c] is 0.756 at 8 sleeves and 0.908 at 40: the more sleeves of
        one true Sharpe ratio, the closer the estimate is to pooling them
        all. Simulated at 40 equicorrelated sleeves, 200 histories."""
        k = 40
        mean, _full = _expected_shrinkage(k)
        assert round(mean, 3) == 0.908
        corr = np.full((k, k), 0.3)
        np.fill_diagonal(corr, 1.0)
        root = np.linalg.cholesky(corr * 0.01**2)
        rng = np.random.default_rng(41)
        c = [
            shrunk_sharpe_ratios(
                0.0004 + rng.standard_normal((500, k)) @ root.T,
                corr,
                rf_per_period=0.0,
                periods_per_year=P,
            )["shrinkage"]
            for _ in range(200)
        ]
        assert np.mean(c) == pytest.approx(mean, abs=0.03)

    def test_unequal_sharpes_are_shrunk_less_as_the_sample_grows(self):
        """The planted Sharpe ratios differ, so Q grows with the sample and
        the shrinkage falls toward none: measured 0.70 at one year and 0.20
        at forty (monthly, 100 histories each)."""
        rng = np.random.default_rng(17)

        def mean_shrinkage(years):
            return np.mean(
                [
                    shrunk_sharpe_ratios(
                        _draw(rng, years, periods=12).to_numpy(),
                        CORR,
                        rf_per_period=0.0,
                        periods_per_year=12,
                    )["shrinkage"]
                    for _ in range(100)
                ]
            )

        short, long = mean_shrinkage(1), mean_shrinkage(40)
        assert short > 0.55
        assert long < 0.3

    def test_full_pooling_is_the_maximum_diversification_portfolio(self):
        """At c = 1 every shrunk Sharpe ratio is the pooled one, so the
        excess means are proportional to the volatilities and the tangency
        maximizes sum w_i sigma_i / sigma_p: the long-only maximum
        diversification portfolio, solved exactly by another method."""
        rng = np.random.default_rng(0)
        for _ in range(60):
            frame = _draw(rng, 3, np.full(N, 0.5))
            fit = fit_sleeve_rule(frame, SleeveRule(method="shrunk_sharpe"))
            if fit["details"]["shrinkage"]["shrinkage"] == 1.0:
                break
        else:  # pragma: no cover - P(no pooling in 60) is 1e-11
            pytest.fail("no fully pooled sample in 60")
        assert fit["details"]["shrinkage"]["pooled_sharpe"] > 0
        cov = _covariance(frame)
        oracle = max_diversification(
            pd.DataFrame(cov, index=NAMES, columns=NAMES), long_only=True
        )
        expected = np.array([oracle["weights"][s] for s in NAMES])
        np.testing.assert_allclose(fit["weights"], expected, atol=1e-9)

    def test_fewer_than_four_sleeves_are_refused(self, three_years):
        with pytest.raises(ValidationError, match="at least 4"):
            fit_sleeve_rule(three_years[NAMES[:3]], SleeveRule(method="shrunk_sharpe"))


class TestTheExactSolvers:
    @pytest.mark.parametrize("cap", [None, 0.3])
    def test_min_variance_and_max_sharpe_are_the_critical_lines(self, three_years, cap):
        """The same covariance and means through Markowitz's critical line,
        an independent exact method: its minimum-variance corner and its
        maximum-Sharpe portfolio over the cash rate, long-only and capped."""
        cov = _covariance(three_years)
        mu = three_years.to_numpy().mean(axis=0) * P
        frontier = critical_line(mu, cov, 0.0, np.inf if cap is None else cap)
        low = fit_sleeve_rule(
            three_years, SleeveRule(method="min_variance", max_weight=cap)
        )
        np.testing.assert_allclose(
            low["weights"], frontier.min_variance.weights, atol=1e-10
        )
        top = fit_sleeve_rule(
            three_years,
            SleeveRule(method="max_sharpe", max_weight=cap),
            cash_rate=0.01,
        )
        np.testing.assert_allclose(
            top["weights"], frontier.max_sharpe(0.01).weights, atol=1e-9
        )
        assert top["details"]["certified"] is True

    def test_a_binding_cap_is_reported(self, three_years):
        out = allocate_across_sleeves(
            three_years, SleeveRule(method="max_sharpe", max_weight=0.3), n_bootstrap=0
        )
        capped = {b["sleeve"] for b in out["binding_caps"]}
        held = {s for s, w in out["weights"].items() if w >= 0.3 - 1e-12}
        assert capped == held and held


class TestCapacityAndCaps:
    def test_water_fill_by_hand(self):
        """Five sleeves of 2m each from 10m, capacities S0 1m, S1 1.5m and
        S2 2.2m. The 1.5m cut goes 0.5m to each of S2..S4, which puts S2 at
        2.5m, over its 2.2m: its 0.3m goes to S3 and S4, 2.65m each. The sum
        is the capital and no sleeve is over its capacity."""
        weights, capped, left = water_fill(
            np.full(5, 0.2), np.array([0.1, 0.15, 0.22, np.inf, np.inf])
        )
        np.testing.assert_allclose(weights, [0.1, 0.15, 0.22, 0.265, 0.265], rtol=1e-14)
        assert capped.tolist() == [True, True, True, False, False]
        assert left == pytest.approx(0.0, abs=1e-15)

    @pytest.mark.parametrize("excess", ["cash", "redistribute"])
    def test_capacity_in_currency_sums_to_the_capital_and_respects_each_cap(
        self, three_years, excess
    ):
        names = NAMES[:5]
        rule = SleeveRule(
            method="equal",
            capital=10_000_000.0,
            capacity={"S0": 1_000_000.0, "S1": 1_500_000.0, "S2": 2_200_000.0},
            capacity_excess=excess,
        )
        out = allocate_across_sleeves(three_years[names], rule, n_bootstrap=0)
        amounts = out["capital_by_sleeve"]
        expected = (
            [1e6, 1.5e6, 2.2e6, 2.65e6, 2.65e6]
            if excess == "redistribute"
            else [1e6, 1.5e6, 2e6, 2e6, 2e6]
        )
        np.testing.assert_allclose([amounts[s] for s in names], expected, rtol=1e-13)
        assert amounts["S0"] == 1e6 and amounts["S1"] == 1.5e6
        assert math.fsum(amounts.values()) + out["cash_amount"] == pytest.approx(
            1e7, rel=1e-15
        )
        assert out["cash_amount"] == pytest.approx(
            0.0 if excess == "redistribute" else 1.5e6, abs=1e-6
        )
        assert {(b["sleeve"], b["cap"]) for b in out["binding_caps"]} >= {
            ("S0", "capacity"),
            ("S1", "capacity"),
        }

    def test_capacity_short_of_the_capital_leaves_the_rest_in_cash(self, three_years):
        rule = SleeveRule(
            method="equal",
            capital=1e7,
            capacity={s: 1e6 for s in NAMES[:4]},
            capacity_excess="redistribute",
        )
        out = allocate_across_sleeves(three_years[NAMES[:4]], rule, n_bootstrap=0)
        assert out["cash_amount"] == pytest.approx(6e6, rel=1e-12)
        assert any("stays in cash" in w for w in out["warnings"])

    def test_max_weight_caps_a_heuristic_and_water_fills_the_rest(self, three_years):
        """Inverse volatility gives S0 (5% volatility) about 22%; capped at
        15% its excess goes to the others pro rata, so their ratios to one
        another are the uncapped ones."""
        free = fit_sleeve_rule(three_years, SleeveRule(method="inverse_volatility"))
        capped = fit_sleeve_rule(
            three_years, SleeveRule(method="inverse_volatility", max_weight=0.15)
        )
        w0, w1 = free["weights"], capped["weights"]
        held = w0 > 0.15
        assert held.any()
        assert np.all(w1[held] == 0.15)
        assert w1.sum() == pytest.approx(1.0, abs=1e-14)
        others = ~held & (w1 < 0.15)
        np.testing.assert_allclose(
            w1[others] / w1[others].sum(), w0[others] / w0[others].sum(), rtol=1e-12
        )


class TestTargetVolatility:
    def test_the_gross_is_scaled_to_the_target_and_the_rest_is_cash(self, three_years):
        out = allocate_across_sleeves(
            three_years,
            SleeveRule(method="risk_parity", target_volatility=0.03),
            n_bootstrap=0,
        )
        assert out["ex_ante_volatility"] == pytest.approx(0.03, rel=1e-12)
        assert out["cash_weight"] == pytest.approx(
            1.0 - out["target_volatility_scale"], abs=1e-12
        )

    def test_a_target_past_max_leverage_is_refused_and_allowed_with_more(
        self, three_years
    ):
        rule = dict(method="risk_parity", target_volatility=0.08)
        with pytest.raises(ValidationError, match="above max_leverage=1"):
            fit_sleeve_rule(three_years, SleeveRule(**rule))
        out = fit_sleeve_rule(three_years, SleeveRule(**rule, max_leverage=2.0))
        assert 1.0 < out["weights"].sum() <= 2.0


class TestTheBootstrapBands:
    def test_the_same_seed_gives_the_same_bits_and_another_seed_others(
        self, three_years
    ):
        rule = SleeveRule(method="risk_parity")
        a = allocate_across_sleeves(three_years, rule, n_bootstrap=60, random_seed=3)
        b = allocate_across_sleeves(three_years, rule, n_bootstrap=60, random_seed=3)
        c = allocate_across_sleeves(three_years, rule, n_bootstrap=60, random_seed=4)
        assert (
            a["weight_p5"] == b["weight_p5"] and a["stability_l1"] == b["stability_l1"]
        )
        assert a["weight_p5"] != c["weight_p5"]

    def test_max_sharpe_moves_far_more_than_risk_parity(self, three_years):
        """On the planted world at three years: a
        max-Sharpe allocation's resampled weights move most of the capital,
        risk parity's hardly any."""
        stable = allocate_across_sleeves(
            three_years, SleeveRule(method="risk_parity"), n_bootstrap=100
        )["stability_l1"]
        unstable = allocate_across_sleeves(
            three_years, SleeveRule(method="max_sharpe"), n_bootstrap=100
        )["stability_l1"]
        assert stable < 0.06
        assert unstable > 10 * stable

    @staticmethod
    def _coverage(simulations, draws):
        """How often the true inverse-volatility weight of four sleeves lies
        inside the 5th-95th percentile band, over independent histories of
        250 days."""
        vol = np.array([0.05, 0.10, 0.15, 0.20])
        corr = np.full((4, 4), 0.3)
        np.fill_diagonal(corr, 1.0)
        root = np.linalg.cholesky(np.outer(vol, vol) * corr / P)
        truth = (1.0 / vol) / (1.0 / vol).sum()
        rng = np.random.default_rng(11)
        hits = np.zeros(4)
        rule = SleeveRule(method="inverse_volatility", covariance_method="sample")
        for seed in range(simulations):
            frame = pd.DataFrame(
                0.0002 + rng.standard_normal((250, 4)) @ root.T, columns=list("ABCD")
            )
            out = allocate_across_sleeves(
                frame, rule, n_bootstrap=draws, random_seed=seed
            )
            low = np.array(list(out["weight_p5"].values()))
            high = np.array(list(out["weight_p95"].values()))
            hits += (low <= truth) & (truth <= high)
        return float(hits.mean() / simulations)

    def test_the_90_percent_band_covers_the_true_weight_near_90_percent(self):
        """50 histories, 100 resamples each: 200 sleeve-histories, whose
        coverage has a standard error near 0.025 (measured 0.85 here, and
        0.883 at 200 histories of 200 resamples); the band is three of them
        around the nominal rate, less the handful of draws' own shortfall."""
        assert 0.78 <= self._coverage(50, 100) <= 0.96

    @pytest.mark.slow
    def test_the_band_covers_the_true_weight_near_90_percent_at_length(self):
        """300 histories of 200 resamples: a standard error near 0.012."""
        assert 0.85 <= self._coverage(300, 200) <= 0.94

    def test_no_resamples_no_bands_and_a_reason(self, three_years, runs_dir):
        ref = handoff.publish(three_years, kind="returns_panel", run_id="r", name="p")
        out = resolve("allocation").dispatch(
            "allocate_across_sleeves", {"returns_ref": ref, "n_bootstrap": 0}
        )
        assert out["weight_p5"] is None and out["stability_l1"] is None
        assert any(w.startswith("stability_l1 is null") for w in out["warnings"])


class TestTheTool:
    def test_streams_and_a_panel_give_the_same_allocation(self, three_years, runs_dir):
        """Dated values per sleeve and one published panel are the same
        streams, so the same allocation, bit for bit, with the alignment
        report beside it."""
        ref = handoff.publish(three_years, kind="returns_panel", run_id="t", name="p")
        values = {
            s: {str(d.date()): float(v) for d, v in three_years[s].items()}
            for s in NAMES
        }
        common = {"method": "risk_parity", "max_weight": 0.3, "n_bootstrap": 20}
        tool = resolve("allocation")
        a = tool.dispatch("allocate_across_sleeves", {"returns_ref": ref, **common})
        b = tool.dispatch("allocate_across_sleeves", {"returns": values, **common})
        assert a["weights"] == b["weights"] and a["weight_p95"] == b["weight_p95"]
        assert a["streams"]["n_dates"] == len(three_years)

    @pytest.mark.parametrize(
        "arguments,match",
        [
            ({"method": "risk_parity", "long_only": False}, "long by construction"),
            ({"method": "equal", "risk_budgets": {"S0": 1.0}}, "'risk_parity' only"),
            ({"capacity": {"S0": 1e6}}, "needs the capital"),
            ({"method": "sample", "halflife": 20}, "Input should be"),
            ({"covariance_method": "sample", "halflife": 20}, "ewma covariances only"),
            ({"capacity": {"S0": 0.0}, "capital": 1e7}, "finite and positive"),
        ],
    )
    def test_a_rule_that_cannot_be_is_refused_by_name(
        self, three_years, runs_dir, arguments, match
    ):
        ref = handoff.publish(three_years, kind="returns_panel", run_id="x", name="p")
        with pytest.raises(pydantic.ValidationError, match=match):
            resolve("allocation").dispatch(
                "allocate_across_sleeves", {"returns_ref": ref, **arguments}
            )

    @pytest.mark.parametrize(
        "arguments,match",
        [
            (
                {
                    "returns": {
                        "a": {"2024-01-02": float("nan")},
                        "b": {"2024-01-02": 0.0},
                    }
                },
                "not finite on",
            ),
            (
                {"returns": {"a": {"2024-01-02": 0.01}}, "returns_ref": "x"},
                "exactly one",
            ),
            ({"returns": {"a": {"2024-01-02": 0.01}}, "columns": ["a"]}, "without it"),
            ({"returns": {"a": {"symbol": "AAPL"}}}, "valid number"),
        ],
    )
    def test_an_inline_stream_that_cannot_be_is_refused(self, arguments, match):
        """The inline form is {stream: {ISO date: return}} and nothing else:
        a NaN, a reference beside it, columns without a reference and a
        symbol in place of the dates are refused by the input model."""
        with pytest.raises(pydantic.ValidationError, match=match):
            resolve("allocation").dispatch("allocate_across_sleeves", arguments)

    def test_a_key_that_is_not_a_date_is_refused_by_name(self):
        returns = {
            "a": {"2024-01-02": 0.01, "AAPL": 0.02},
            "b": {"2024-01-02": 0.0, "2024-01-03": 0.01},
        }
        with pytest.raises(ValidationError, match=r"returns\['a'\]"):
            resolve("allocation").dispatch(
                "allocate_across_sleeves", {"returns": returns}
            )

    def test_caps_too_small_for_the_sleeves_are_refused(self, three_years, runs_dir):
        ref = handoff.publish(three_years, kind="returns_panel", run_id="y", name="p")
        with pytest.raises(ValidationError, match="at least 0.125"):
            resolve("allocation").dispatch(
                "allocate_across_sleeves", {"returns_ref": ref, "max_weight": 0.1}
            )

    def test_max_sharpe_with_no_sleeve_over_cash_is_refused(self, three_years):
        with pytest.raises(ValidationError, match="above cash_rate"):
            fit_sleeve_rule(three_years, SleeveRule(method="max_sharpe"), cash_rate=5.0)
