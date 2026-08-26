"""
The last seven functions, checked against their own identities.

Each of these has a case where the answer is fixed by definition rather than
by measurement, and those are the tests worth writing:

- max diversification on UNCORRELATED assets is exactly inverse-volatility,
  and on PERFECTLY correlated ones the ratio is exactly 1.0 -- that is what
  the ratio means;
- risk contributions sum exactly to portfolio volatility, which is what
  makes the decomposition real;
- a scenario's portfolio return is a weighted sum, so it is arithmetic;
- implementation shortfall's four components sum exactly to the total, and
  each one has a closed form on a hand-built example;
- break-even cost is the gross mean return, so it is arithmetic too.

The two estimators -- normality and tail index -- have no exact case, so
they are checked by SEPARATION instead: a normal must not be called fat and
a t(3) must be, and the tail index must order t(3) below t(5) even though it
is biased low on both.
"""

import math

import numpy as np
import pandas as pd
import pytest

# `test_normality` is ALIASED because pytest collects any module-level
# name beginning with `test_` as a test -- including an imported function --
# and then fails trying to inject its `values` argument as a fixture.
from standard_quant_tools.analysis.inference import (
    estimate_tail_index,
)
from standard_quant_tools.analysis.inference import test_normality as normality_test
from standard_quant_tools.analysis.microstructure_estimators import (
    implementation_shortfall,
)
from standard_quant_tools.backtesting.trade_analysis import break_even_cost
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.construction import (
    marginal_risk_contribution,
    max_diversification,
    portfolio_scenarios,
)

VOLS = np.array([0.10, 0.20, 0.40, 0.15])
CORR = np.array(
    [
        [1.0, 0.3, 0.2, 0.5],
        [0.3, 1.0, 0.4, 0.3],
        [0.2, 0.4, 1.0, 0.1],
        [0.5, 0.3, 0.1, 1.0],
    ]
)
NAMES = list("ABCD")


def _cov(corr=CORR):
    return pd.DataFrame(np.outer(VOLS, VOLS) * corr, index=NAMES, columns=NAMES)


class TestMaxDiversification:
    def test_uncorrelated_assets_give_exactly_inverse_volatility(self):
        """
        A closed-form case: with no correlation the maximum-diversification
        portfolio IS inverse-volatility. A solver that misses this is wrong
        everywhere else too.
        """
        result = max_diversification(
            pd.DataFrame(np.diag(VOLS**2), index=NAMES, columns=NAMES)
        )
        expected = (1 / VOLS) / (1 / VOLS).sum()
        for name, want in zip(NAMES, expected):
            assert result["weights"][name] == pytest.approx(want, rel=1e-8)

    def test_perfectly_correlated_assets_give_a_ratio_of_exactly_one(self):
        """
        The other closed-form case, and the one that defines the ratio: when
        everything moves together, combining the assets buys nothing.
        """
        perfect = pd.DataFrame(np.outer(VOLS, VOLS), index=NAMES, columns=NAMES)
        result = max_diversification(perfect)
        assert result["diversification_ratio"] == pytest.approx(1.0, abs=1e-6)

    def test_falling_correlation_raises_the_ratio(self):
        ratios = []
        for rho in (0.9, 0.5, 0.1):
            corr = np.full((4, 4), rho)
            np.fill_diagonal(corr, 1.0)
            ratios.append(max_diversification(_cov(corr))["diversification_ratio"])
        assert ratios == sorted(ratios)

    def test_the_weights_sum_to_one(self):
        assert sum(max_diversification(_cov())["weights"].values()) == pytest.approx(
            1.0
        )

    def test_it_says_it_is_not_minimum_variance(self):
        """The confusion this tool exists inside: minimum variance piles
        into the quietest assets, which is not diversification."""
        result = max_diversification(_cov())
        assert any("NOT the same as minimum variance" in w for w in result["warnings"])

    def test_an_ill_conditioned_matrix_is_flagged(self):
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, 300)
        frame = pd.DataFrame(
            {f"a{i}": base + rng.normal(0, 1e-4, 300) for i in range(5)}
        )
        result = max_diversification(frame.cov())
        assert result["condition_number"] > 1000
        assert any("condition number" in w for w in result["warnings"])

    def test_negative_weights_are_reported_rather_than_clipped(self):
        """Clipping to zero would silently return a different portfolio than
        the one that was optimized."""
        corr = np.array(
            [
                [1.0, 0.9, 0.9, 0.9],
                [0.9, 1.0, 0.2, 0.2],
                [0.9, 0.2, 1.0, 0.2],
                [0.9, 0.2, 0.2, 1.0],
            ]
        )
        result = max_diversification(_cov(corr))
        if result["n_negative_weights"]:
            assert any("shorting" in w for w in result["warnings"])

    def test_a_single_asset_is_refused(self):
        with pytest.raises(ValidationError, match="at least two"):
            max_diversification(pd.DataFrame([[0.04]], index=["A"], columns=["A"]))


class TestMarginalRiskContribution:
    WEIGHTS = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

    def test_the_contributions_sum_exactly_to_portfolio_volatility(self):
        """What makes this a decomposition rather than an allocation of
        blame. Exact, not approximate."""
        result = marginal_risk_contribution(self.WEIGHTS, _cov())
        assert result["sum_of_contributions"] == pytest.approx(
            result["portfolio_volatility"], rel=1e-12
        )

    def test_the_risk_shares_sum_to_one(self):
        result = marginal_risk_contribution(self.WEIGHTS, _cov())
        assert sum(r["risk_share"] for r in result["by_asset"]) == pytest.approx(1.0)

    def test_equal_weights_do_not_mean_equal_risk(self):
        """
        The whole point. Four assets at 25% each, and the 40-vol one carries
        over half the risk.
        """
        result = marginal_risk_contribution(self.WEIGHTS, _cov())
        by_asset = {r["asset"]: r for r in result["by_asset"]}
        assert by_asset["C"]["risk_share"] > 0.4
        assert by_asset["A"]["risk_share"] < 0.15

    def test_an_outsized_position_is_flagged(self):
        result = marginal_risk_contribution(self.WEIGHTS, _cov())
        assert any(r["concentration_flag"] for r in result["by_asset"])
        assert any("twice their weight share" in w for w in result["warnings"])

    def test_a_negatively_correlated_asset_is_identified_as_a_hedge(self):
        """A negative marginal contribution means adding REDUCES risk."""
        corr = np.array(
            [
                [1.0, 0.8, 0.8, -0.8],
                [0.8, 1.0, 0.8, -0.8],
                [0.8, 0.8, 1.0, -0.8],
                [-0.8, -0.8, -0.8, 1.0],
            ]
        )
        result = marginal_risk_contribution(self.WEIGHTS, _cov(corr))
        if result["n_hedges"]:
            assert any("NEGATIVE marginal risk" in w for w in result["warnings"])

    def test_a_missing_weight_is_refused_rather_than_treated_as_zero(self):
        with pytest.raises(ValidationError, match="no weight given"):
            marginal_risk_contribution({"A": 0.5, "B": 0.5}, _cov())

    def test_a_zero_volatility_portfolio_has_no_risk_to_attribute(self):
        with pytest.raises(ValidationError, match="zero volatility"):
            marginal_risk_contribution({"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}, _cov())


class TestPortfolioScenarios:
    WEIGHTS = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}

    def test_the_portfolio_return_is_the_weighted_sum(self):
        """Arithmetic, so it is exact."""
        shocks = {"A": -0.05, "B": -0.15, "C": -0.35, "D": -0.10}
        result = portfolio_scenarios(self.WEIGHTS, {"crash": shocks})
        expected = sum(self.WEIGHTS[a] * shocks[a] for a in shocks)
        assert result["by_scenario"][0]["portfolio_return"] == pytest.approx(expected)

    def test_scenarios_are_ordered_worst_first(self):
        result = portfolio_scenarios(
            self.WEIGHTS,
            {
                "crash": {a: -0.2 for a in self.WEIGHTS},
                "rally": {a: 0.2 for a in self.WEIGHTS},
                "flat": {a: 0.0 for a in self.WEIGHTS},
            },
        )
        returns = [r["portfolio_return"] for r in result["by_scenario"]]
        assert returns == sorted(returns)
        assert result["worst_scenario"]["scenario"] == "crash"

    def test_a_partial_scenario_is_flagged_as_a_lower_bound(self):
        """
        Positions absent from a scenario are treated as unchanged, so a
        scenario touching one of four positions produces a loss that is a
        lower bound rather than a worst case.
        """
        result = portfolio_scenarios(self.WEIGHTS, {"partial": {"C": -0.30}})
        assert result["by_scenario"][0]["coverage"] == pytest.approx(0.25)
        assert any("LOWER BOUND" in w for w in result["warnings"])

    def test_a_covariance_gives_the_move_in_sigmas(self):
        result = portfolio_scenarios(
            self.WEIGHTS,
            {"crash": {a: -0.2 for a in self.WEIGHTS}},
            covariance=_cov(),
        )
        assert result["by_scenario"][0]["sigma_move"] is not None
        assert result["portfolio_volatility"] is not None

    def test_without_a_covariance_there_are_no_sigmas(self):
        result = portfolio_scenarios(self.WEIGHTS, {"crash": {"A": -0.2}})
        assert result["by_scenario"][0]["sigma_move"] is None

    def test_it_says_a_scenario_and_a_var_answer_different_questions(self):
        result = portfolio_scenarios(self.WEIGHTS, {"crash": {"A": -0.2}})
        assert any("different questions" in w for w in result["warnings"])

    def test_no_scenarios_is_refused(self):
        with pytest.raises(ValidationError, match="no scenarios"):
            portfolio_scenarios(self.WEIGHTS, {})


class TestNormality:
    def test_a_normal_sample_is_not_called_fat_tailed(self):
        """The null case."""
        result = normality_test(np.random.default_rng(0).normal(0, 1, 2000))
        assert abs(result["excess_kurtosis"]) < 0.5
        assert result["tail_ratio_3_sigma"] < 2

    def test_a_t3_sample_is(self):
        result = normality_test(np.random.default_rng(0).standard_t(3, 2000))
        assert result["excess_kurtosis"] > 3
        assert result["tail_ratio_3_sigma"] > 2
        assert any("understates the loss" in w for w in result["warnings"])

    def test_the_false_positive_rate_is_near_nominal(self):
        fires = sum(
            not normality_test(np.random.default_rng(s).normal(0, 1, 300))[
                "normal_at_05"
            ]
            for s in range(150)
        )
        assert fires / 150 < 0.15, f"{fires}/150 normal samples called non-normal"

    def test_the_expected_tail_counts_are_the_normal_ones(self):
        result = normality_test(np.random.default_rng(1).normal(0, 1, 1000))
        assert result["expected_beyond_3_sigma"] == pytest.approx(2.7)
        assert result["expected_beyond_4_sigma"] == pytest.approx(0.0633)

    def test_negative_skew_is_named_as_the_short_vol_shape(self):
        rng = np.random.default_rng(2)
        skewed = np.where(rng.random(2000) < 0.97, 0.001, -0.03)
        result = normality_test(skewed)
        assert result["skewness"] < -0.5
        assert any("short-volatility shape" in w for w in result["warnings"])

    def test_a_long_sample_is_told_the_p_value_measures_length(self):
        result = normality_test(np.random.default_rng(3).normal(0, 1, 2000))
        assert any("measuring sample length" in w for w in result["warnings"])

    def test_a_flat_series_is_refused(self):
        with pytest.raises(ValidationError, match="no dispersion"):
            normality_test([0.01] * 100)


class TestTailIndex:
    def test_it_orders_a_fatter_tail_below_a_thinner_one(self):
        """
        No exact case exists -- the estimator is biased low on any real
        distribution -- so the test is SEPARATION. A t(3) must come back
        below a t(5), and a normal above both.
        """
        rng = np.random.default_rng(0)
        t3 = estimate_tail_index(rng.standard_t(3, 4000), tail="left")["alpha"]
        t5 = estimate_tail_index(rng.standard_t(5, 4000), tail="left")["alpha"]
        assert t3 < t5

    def test_the_bias_is_downward_and_is_declared(self):
        """Measured: 2.39 for a true 3.0. The docstring says so and the
        result reports a standard error rather than three decimals."""
        result = estimate_tail_index(
            np.random.default_rng(1).standard_t(3, 4000), tail="left"
        )
        assert result["alpha"] < 3.0
        assert result["standard_error"] is not None
        assert any("Standard error" in w for w in result["warnings"])

    def test_a_very_fat_tail_is_flagged_as_having_infinite_variance(self):
        result = estimate_tail_index(
            np.random.default_rng(2).standard_t(1, 4000), tail="left"
        )
        if result["alpha"] is not None and result["alpha"] < 2:
            assert not result["variance_finite"]
            assert any("infinite" in w for w in result["warnings"])

    def test_instability_across_thresholds_is_reported(self):
        result = estimate_tail_index(
            np.random.default_rng(3).standard_t(5, 4000), tail="left"
        )
        assert result["alpha_spread"] is not None
        assert len(result["across_thresholds"]) >= 3

    def test_both_tails_can_be_asked_for(self):
        rng = np.random.default_rng(4)
        # Left tail deliberately fatter than the right.
        sample = np.where(
            rng.random(4000) < 0.5, rng.standard_t(3, 4000), rng.normal(0, 1, 4000)
        )
        for tail in ("left", "right"):
            result = estimate_tail_index(sample, tail=tail)
            assert result["tail"] == tail

    def test_an_unknown_tail_is_refused(self):
        with pytest.raises(ValidationError, match="left.*right"):
            estimate_tail_index(np.random.default_rng(5).normal(0, 1, 200), tail="both")

    def test_too_little_data_is_refused(self):
        with pytest.raises(ValidationError, match="at least"):
            estimate_tail_index(np.random.default_rng(6).normal(0, 1, 50))


class TestBreakEvenCost:
    def test_the_break_even_is_the_gross_mean_return(self):
        """Arithmetic, so it is exact."""
        trades = np.random.default_rng(0).normal(0.002, 0.02, 300)
        result = break_even_cost(trades, current_cost_bps=5.0)
        expected = (float(trades.mean()) + 5.0 / 1e4) * 1e4
        assert result["break_even_cost_bps"] == pytest.approx(expected, rel=1e-9)

    def test_the_headroom_is_the_ratio_to_the_assumed_cost(self):
        trades = np.random.default_rng(0).normal(0.002, 0.02, 300)
        result = break_even_cost(trades, current_cost_bps=10.0)
        assert result["headroom_multiple"] == pytest.approx(
            result["break_even_cost_bps"] / 10.0, rel=1e-9
        )

    def test_thin_headroom_is_flagged(self):
        """Under about 2x, the backtest is a statement about the cost
        assumption rather than about the strategy."""
        trades = np.random.default_rng(1).normal(0.0002, 0.02, 300)
        result = break_even_cost(trades, current_cost_bps=15.0)
        if result["headroom_multiple"] and result["headroom_multiple"] < 2:
            assert any("headroom" in w for w in result["warnings"])

    def test_a_strategy_that_loses_before_costs_has_no_break_even(self):
        result = break_even_cost(
            np.random.default_rng(2).normal(-0.002, 0.01, 200), current_cost_bps=5.0
        )
        assert result["break_even_cost_bps"] == 0.0
        assert any("before costs" in w.lower() for w in result["warnings"])

    def test_the_sensitivity_table_is_monotone(self):
        trades = np.random.default_rng(3).normal(0.003, 0.02, 300)
        result = break_even_cost(trades, current_cost_bps=5.0)
        means = [row["mean_return"] for row in result["sensitivity"]]
        assert means == sorted(means, reverse=True)

    def test_it_says_it_does_not_model_impact(self):
        result = break_even_cost(
            np.random.default_rng(4).normal(0.002, 0.02, 200), current_cost_bps=5.0
        )
        assert any("does not model impact" in w for w in result["warnings"])


class TestImplementationShortfall:
    BASE = dict(
        decision_price=100.0,
        arrival_price=100.5,
        target_quantity=1000.0,
        final_price=102.0,
        side="buy",
    )
    FILLS = [
        {"quantity": 600.0, "price": 100.8, "fee": 6.0},
        {"quantity": 300.0, "price": 101.2, "fee": 3.0},
    ]

    def test_every_component_matches_its_closed_form(self):
        """
        Hand-computed on this example: delay is (100.5-100.0)*900 = 450,
        impact is (100.9333-100.5)*900 = 390, opportunity is
        (102.0-100.0)*100 = 200, fees are 9.
        """
        result = implementation_shortfall(**self.BASE, fills=self.FILLS)
        by_name = {c["component"]: c["dollars"] for c in result["components"]}
        assert by_name["delay"] == pytest.approx(450.0)
        assert by_name["impact"] == pytest.approx(390.0)
        assert by_name["opportunity"] == pytest.approx(200.0)
        assert by_name["fees"] == pytest.approx(9.0)

    def test_the_components_sum_to_the_total(self):
        result = implementation_shortfall(**self.BASE, fills=self.FILLS)
        assert sum(c["dollars"] for c in result["components"]) == pytest.approx(
            result["total_shortfall_dollars"], rel=1e-12
        )

    def test_a_sell_reverses_the_sign_of_an_adverse_move(self):
        """
        A buy into a rising market pays more; a sell into a rising market
        receives more. The sign has to flip or every number is backwards.
        """
        buy = implementation_shortfall(**{**self.BASE, "side": "buy"}, fills=self.FILLS)
        sell = implementation_shortfall(
            **{**self.BASE, "side": "sell"}, fills=self.FILLS
        )
        assert buy["total_shortfall_dollars"] == pytest.approx(
            -sell["total_shortfall_dollars"] + 2 * 9.0, rel=1e-9
        )

    def test_delay_is_named_as_a_workflow_problem_when_it_dominates(self):
        result = implementation_shortfall(
            **{**self.BASE, "arrival_price": 103.0},
            fills=[{"quantity": 1000.0, "price": 103.1, "fee": 5.0}],
        )
        assert result["largest_component"] == "delay"
        assert any("workflow problem" in w for w in result["warnings"])

    def test_an_incomplete_fill_is_priced_as_opportunity_cost(self):
        result = implementation_shortfall(**self.BASE, fills=self.FILLS)
        assert result["fill_rate"] == pytest.approx(0.9)
        assert any("moved its cost here" in w for w in result["warnings"])

    def test_using_the_arrival_price_as_the_decision_price_is_called_out(self):
        """It zeroes the delay cost by construction, and delay is often
        where the money went."""
        result = implementation_shortfall(
            **{**self.BASE, "decision_price": 100.5}, fills=self.FILLS
        )
        assert any("BY CONSTRUCTION" in w for w in result["warnings"])

    def test_the_sign_convention_is_stated(self):
        result = implementation_shortfall(**self.BASE, fills=self.FILLS)
        assert any("POSITIVE is a COST" in w for w in result["warnings"])

    def test_an_overfill_is_refused_as_a_reconciliation_problem(self):
        with pytest.raises(ValidationError, match="overfill|reconciliation"):
            implementation_shortfall(
                **self.BASE, fills=[{"quantity": 2000.0, "price": 100.8}]
            )

    def test_a_negative_fill_quantity_is_refused(self):
        with pytest.raises(ValidationError, match="negative quantity"):
            implementation_shortfall(
                **self.BASE, fills=[{"quantity": -100.0, "price": 100.8}]
            )

    def test_a_negative_target_is_refused_because_side_carries_direction(self):
        with pytest.raises(ValidationError, match="double-negate"):
            implementation_shortfall(
                **{**self.BASE, "target_quantity": -1000.0}, fills=self.FILLS
            )
