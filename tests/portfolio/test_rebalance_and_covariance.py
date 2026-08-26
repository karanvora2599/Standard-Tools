"""
Getting to a target portfolio, and trusting the matrix you optimized with.

THE BUG WORTH READING ABOUT. `total_cost_bps` originally summed each day's
average impact rate, which added five rates and called the result a cost.
Measured on a transition where liquidity does not bind:

    urgency=1.0   1 day    1.83 bps
    urgency=0.0   5 days   4.08 bps

That says trading everything at once is less than half the cost of spreading
it, which is backwards. Both numbers were small and plausible, and a caller
reading them would have concluded "trade fast, it's cheaper" — the wrong
conclusion in the direction that costs money.

The cost is now total impact dollars over total notional. The test below
checks it against the square-root law's own arithmetic: taking 5x the daily
volume must cost sqrt(5) times the RATE, and it does, to two decimals.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.covariance import estimate_covariance
from standard_quant_tools.portfolio.rebalance import plan_rebalance


def _returns(n_assets=40, n_obs=120, seed=0):
    """A panel with a common factor, so the covariance is genuinely
    ill-conditioned rather than diagonal."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0, 0.01, n_obs)
    return pd.DataFrame(
        {f"A{i}": 0.8 * factor + rng.normal(0, 0.008, n_obs) for i in range(n_assets)},
        index=pd.bdate_range("2024-01-01", periods=n_obs),
    )


class TestTheCostIsACostAndNotASumOfRates:
    LIQUID = {"AAA": 900e6, "BBB": 900e6}
    CURRENT = {"AAA": 0.50, "BBB": 0.50}
    TARGET = {"AAA": 0.20, "BBB": 0.80}

    def _plan(self, urgency, days=5):
        return plan_rebalance(
            self.CURRENT,
            self.TARGET,
            portfolio_value=100e6,
            adv=self.LIQUID,
            urgency=urgency,
            max_days=days,
        )

    def test_trading_fast_costs_more_than_trading_slow(self):
        """The bug, stated as a test. It reported the opposite."""
        fast = self._plan(1.0)
        slow = self._plan(0.0)
        assert fast["total_cost_bps"] > slow["total_cost_bps"], (
            f"trading everything in one day was reported as "
            f"{fast['total_cost_bps']:.2f} bps against {slow['total_cost_bps']:.2f} "
            "for spreading it, which is backwards under a square-root impact law"
        )

    def test_the_rate_ratio_matches_the_square_root_law(self):
        """Five times the daily volume, sqrt(5) times the rate. Checked
        against the law's own arithmetic rather than a recorded number."""
        fast = self._plan(1.0)
        slow = self._plan(0.0)
        assert fast["total_cost_bps"] / slow["total_cost_bps"] == pytest.approx(
            np.sqrt(5.0), rel=0.02
        )

    def test_the_dollars_are_reported_too(self):
        """The quantity with no ambiguity in it. A rate needs a denominator
        stated; dollars do not."""
        fast = self._plan(1.0)
        assert fast["total_cost_dollars"] > 0
        traded = sum(s["traded_notional"] for s in fast["schedule"])
        assert fast["total_cost_bps"] == pytest.approx(
            fast["total_cost_dollars"] / traded * 1e4, rel=1e-6
        )

    def test_both_urgencies_trade_the_same_total(self):
        """Cost differs; turnover must not. If spreading also traded less,
        the comparison above would be measuring the wrong thing."""
        assert self._plan(1.0)["total_turnover"] == pytest.approx(
            self._plan(0.0)["total_turnover"]
        )


class TestItSurfacesWhatAnOptimizerCannot:
    def test_a_target_the_market_cannot_supply_is_named(self):
        """A 60% target in a $2m-ADV name against a $100m book is not a
        position, it is a project. The weight vector is valid, the backtest
        fills at the close, and nothing else in the pipeline notices."""
        result = plan_rebalance(
            {"AAA": 0.30, "BBB": 0.30, "CCC": 0.40},
            {"AAA": 0.10, "BBB": 0.30, "DDD": 0.60},
            portfolio_value=100e6,
            adv={"AAA": 50e6, "BBB": 50e6, "CCC": 40e6, "DDD": 2e6},
            max_days=5,
        )
        assert not result["converged"]
        worst = result["unreachable"][0]
        assert worst["name"] == "DDD"
        assert worst["days_needed"] > 100

    def test_the_warning_says_the_portfolio_is_not_the_optimized_one(self):
        result = plan_rebalance(
            {"AAA": 1.0},
            {"AAA": 0.0, "DDD": 1.0},
            portfolio_value=100e6,
            adv={"AAA": 50e6, "DDD": 1e6},
            max_days=3,
        )
        assert any(
            "not the portfolio that was optimized" in w for w in result["warnings"]
        )

    def test_missing_adv_reports_null_cost_rather_than_zero(self):
        """An unpriced transition is not a free one."""
        result = plan_rebalance(
            {"AAA": 1.0}, {"BBB": 1.0}, portfolio_value=100e6, max_days=3
        )
        assert result["total_cost_bps"] is None
        assert any("not a free one" in w for w in result["warnings"])

    def test_over_participation_is_flagged_as_optimistic(self):
        result = plan_rebalance(
            {"AAA": 0.0},
            {"AAA": 1.0},
            portfolio_value=100e6,
            adv={"AAA": 100e6},
            max_participation=0.9,
            urgency=1.0,
            max_days=2,
        )
        assert any("optimistic" in w for w in result["warnings"])

    def test_already_at_target_trades_nothing(self):
        result = plan_rebalance(
            {"AAA": 0.5, "BBB": 0.5},
            {"AAA": 0.5, "BBB": 0.5},
            portfolio_value=1e6,
        )
        assert result["converged"]
        assert result["total_turnover"] == 0.0
        assert result["schedule"] == []

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"portfolio_value": 0.0},
            {"portfolio_value": 1e6, "urgency": 1.5},
            {"portfolio_value": 1e6, "max_days": 0},
            {"portfolio_value": 1e6, "max_participation": 0.0},
        ],
    )
    def test_nonsense_is_refused(self, kwargs):
        with pytest.raises(ValidationError):
            plan_rebalance({"AAA": 1.0}, {"BBB": 1.0}, **kwargs)


class TestShrinkageAnswersTheConditioningWarning:
    def test_shrinkage_improves_conditioning_and_ewma_worsens_it(self):
        """The claim the module makes, checked rather than asserted. EWMA is
        not a worse shrinkage — it answers a different question and lowers
        the effective sample size doing it."""
        returns = _returns()
        cond = {
            m: estimate_covariance(returns, method=m)["condition_number"]
            for m in ("sample", "ledoit_wolf", "ewma", "ewma_shrunk")
        }
        assert cond["ledoit_wolf"] < cond["sample"]
        assert cond["ewma"] > cond["sample"]
        assert cond["ewma_shrunk"] < cond["ledoit_wolf"]

    def test_it_reports_how_thin_the_estimate_is(self):
        """A covariance over N assets has N(N+1)/2 parameters. That ratio is
        the honest measure and it is not obvious from N and T alone."""
        result = estimate_covariance(_returns(40, 120), method="sample")
        assert result["observations_per_parameter"] == pytest.approx(
            120 * 40 / (40 * 41 / 2), rel=1e-9
        )
        assert any("per estimated parameter" in w for w in result["warnings"])

    def test_the_sample_estimator_warns_when_assets_approach_observations(self):
        result = estimate_covariance(_returns(40, 120), method="sample")
        assert any("error-maximizer" in w for w in result["warnings"])

    def test_a_well_conditioned_panel_does_not_warn(self):
        """A report that warns about everything is ignored."""
        result = estimate_covariance(_returns(3, 500), method="ledoit_wolf")
        assert result["warnings"] == []

    def test_the_matrix_is_annualized(self):
        """Every other risk number here is annualized; a daily covariance
        silently produces a volatility 16 times too small."""
        returns = _returns(3, 500)
        annual = estimate_covariance(returns, method="sample")
        daily_variance = returns["A0"].var(ddof=1)
        assert annual["matrix"]["A0"]["A0"] == pytest.approx(
            daily_variance * 252, rel=1e-6
        )
        assert annual["annualized"]

    def test_ewma_weights_recent_data_more(self):
        """The property it exists for. A late volatility burst must move the
        EWMA estimate more than the sample one."""
        rng = np.random.default_rng(0)
        calm = rng.normal(0, 0.005, 200)
        burst = rng.normal(0, 0.05, 60)
        series = np.concatenate([calm, burst])
        frame = pd.DataFrame(
            {"A": series, "B": rng.normal(0, 0.005, 260)},
            index=pd.bdate_range("2024-01-01", periods=260),
        )
        sample = estimate_covariance(frame, method="sample")["matrix"]["A"]["A"]
        ewma = estimate_covariance(frame, method="ewma", halflife=20)["matrix"]["A"][
            "A"
        ]
        assert ewma > sample

    def test_an_unknown_method_is_refused_by_name(self):
        with pytest.raises(ValidationError, match="unknown covariance method"):
            estimate_covariance(_returns(3, 100), method="oas")

    def test_too_little_history_is_refused_with_the_reason(self):
        with pytest.raises(ValidationError, match="at least 2 complete"):
            estimate_covariance(_returns(3, 100).head(1))


class TestTheMicrostructureSplitHappened:
    """
    This class used to record why the split had NOT happened, and one of its
    tests said so in as many words: "the day microstructure reaches eight
    tools, this fails and says the split is now legal. Do it, and delete
    this test." Microstructure reached eleven, the test failed, and this is
    the follow-through.

    What it pins now is the finished state, and the floors it pins are the
    same ones that blocked it: a runtime holding fewer than eight tools is
    overhead rather than isolation, on either side of a split.
    """

    def test_microstructure_is_its_own_runtime(self):
        from standard_quant_tools.agent.runtimes import resolve

        micro = resolve("microstructure")
        assert len(micro) >= 8, (
            f"microstructure has {len(micro)} tools, below the floor of "
            "eight that makes a runtime worth its own boundary"
        )
        assert "get_microstructure_metrics" in micro.dispatch_table
        assert "estimate_roll_spread" in micro.dispatch_table

    def test_the_donor_still_clears_the_floor(self):
        from standard_quant_tools.agent.runtimes import resolve

        portfolio = resolve("portfolio")
        assert len(portfolio) >= 8, (
            f"portfolio was left with {len(portfolio)} tools after the "
            "split, below the floor"
        )

    def test_no_microstructure_tool_is_still_served_by_portfolio(self):
        """A split that left the tools in both places would dissolve the
        boundary at exactly the point it was drawn."""
        from standard_quant_tools.agent.router import TOOL_CATEGORY
        from standard_quant_tools.agent.runtimes import resolve

        portfolio = resolve("portfolio")
        leaked = [
            n for n in portfolio.tool_names if TOOL_CATEGORY.get(n) == "microstructure"
        ]
        assert not leaked, f"still served by portfolio: {leaked}"

    def test_the_donor_names_the_new_home(self):
        """The break has to be recoverable. A bare 'unknown tool' cannot be
        told apart from a hallucination, and a model receiving one guesses
        again."""
        from standard_quant_tools.agent.runtimes import resolve

        with pytest.raises(ValueError, match="microstructure"):
            resolve("portfolio").dispatch("get_microstructure_metrics", {})
