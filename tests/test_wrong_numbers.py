"""
Six functions that returned a plausible wrong number, and the suite was green.

Every fix in this file was made against a full suite that passed before and
after, which is the reason these tests exist in one place rather than being
scattered into the files that already cover these modules. Those files test
that the functions run and that their outputs are ordered sensibly. None of
them pinned a value against an independent computation, so a wrong value
looked exactly like a right one.

The shared shape: none of these raised, none returned NaN, and four of the
six put a wrong number directly beside a correct one in the same result
object. That is what makes this class worse than a crash.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


class TestFuturesRollIsNotProfit:
    """`run_futures_simulation` computed variation margin before detecting
    the roll, so on a roll day it differenced the NEW contract's price
    against the OLD contract's. That is the calendar spread, not a market
    move, and it was booked as P&L once per roll."""

    def test_a_flat_market_earns_nothing_across_a_roll(self):
        from standard_quant_tools.backtest.futures_engine import (
            run_futures_simulation,
        )

        dates = [f"2024-01-0{i}" for i in range(1, 7)]
        # Contract A prints 100 for three days, B prints 110 for three days.
        # Neither ever moves.
        result = run_futures_simulation(
            prices=dict(zip(dates, [100.0, 100.0, 100.0, 110.0, 110.0, 110.0])),
            target_contracts={dates[0]: 1.0},
            multiplier=1.0,
            initial_capital=1000.0,
            contract_map=dict(zip(dates, ["A", "A", "A", "B", "B", "B"])),
        )
        assert result["n_rolls"] == 1, "the roll must still be detected"
        assert result["total_variation_margin"] == 0.0
        assert result["final_equity"] == pytest.approx(1000.0)
        assert result["total_return_pct"] == pytest.approx(0.0)

    def test_the_sign_was_backwards_for_a_long_in_contango(self):
        """Rolling UP a contango curve costs money. It was credited."""
        from standard_quant_tools.backtest.futures_engine import (
            run_futures_simulation,
        )

        dates = [f"2024-01-0{i}" for i in range(1, 5)]
        result = run_futures_simulation(
            prices=dict(zip(dates, [100.0, 100.0, 115.0, 115.0])),
            target_contracts={dates[0]: 10.0},
            multiplier=50.0,
            initial_capital=1_000_000.0,
            contract_map=dict(zip(dates, ["A", "A", "B", "B"])),
        )
        assert result["total_variation_margin"] <= 0.0

    def test_a_real_move_is_still_booked(self):
        """The fix must not swallow genuine P&L."""
        from standard_quant_tools.backtest.futures_engine import (
            run_futures_simulation,
        )

        dates = [f"2024-01-0{i}" for i in range(1, 7)]
        result = run_futures_simulation(
            prices=dict(zip(dates, [100.0, 100.0, 100.0, 100.0, 105.0, 105.0])),
            target_contracts={dates[0]: 1.0},
            multiplier=1.0,
            initial_capital=1000.0,
        )
        assert result["total_variation_margin"] == pytest.approx(5.0)


class TestSeriesMetricsGetTheSeriesTheyDocument:
    """`cagr` calls `cumulative_return` and `calmar_ratio`'s own parameter is
    named `equity_curve`. Both were registered as wanting returns, so both
    came back sign-flipped -- next to a `max_drawdown` in the same response
    that was correct."""

    def test_cagr_and_calmar_match_the_library(self):
        from standard_quant_tools.agent.runtimes.research.reference_tools import (
            SeriesMetricsInput,
            calculate_series_metrics,
        )
        from standard_quant_tools.metrics.return_metrics import cagr
        from standard_quant_tools.metrics.risk_metrics import (
            calmar_ratio,
            max_drawdown,
        )

        returns = pd.Series(np.random.default_rng(3).normal(0.0002, 0.013, 504))
        equity = (1.0 + returns).cumprod()

        got = calculate_series_metrics(
            SeriesMetricsInput(
                series={"values": returns.tolist()},
                metrics=["cagr", "calmar_ratio", "max_drawdown"],
            )
        ).model_dump()["values"]

        assert got["cagr"] == pytest.approx(float(cagr(equity)), rel=1e-9)
        assert got["calmar_ratio"] == pytest.approx(
            float(calmar_ratio(equity)), rel=1e-9
        )
        assert got["max_drawdown"] == pytest.approx(
            float(max_drawdown(equity)), rel=1e-9
        )

    def test_a_profitable_series_does_not_report_a_loss(self):
        """The failure was this stark: +23.4%/yr came back as -48.8%/yr."""
        from standard_quant_tools.agent.runtimes.research.reference_tools import (
            SeriesMetricsInput,
            calculate_series_metrics,
        )

        returns = pd.Series(np.random.default_rng(3).normal(0.0002, 0.013, 504))
        assert float((1.0 + returns).cumprod().iloc[-1]) > 1.0

        got = calculate_series_metrics(
            SeriesMetricsInput(
                series={"values": returns.tolist()}, metrics=["cagr", "calmar_ratio"]
            )
        ).model_dump()["values"]
        assert got["cagr"] > 0
        assert got["calmar_ratio"] > 0


class TestRhoMatchesTheModelItIsQuotedFor:
    """Black-76's rho was the Black-Scholes formula. Under Black-76 the
    FORWARD is given and does not move with the rate, so the rate enters
    only through the discount factor and every option is worth less when it
    rises: d/dr = -T * price. The call's sign was wrong."""

    @staticmethod
    def _finite_difference(model: str, **kwargs) -> float:
        from standard_quant_tools.analysis.pricing import price_option

        step = 1e-6
        up = price_option(
            **{**kwargs, "risk_free_rate": kwargs["risk_free_rate"] + step},
            model=model,
        )["price"]
        down = price_option(
            **{**kwargs, "risk_free_rate": kwargs["risk_free_rate"] - step},
            model=model,
        )["price"]
        return (up - down) / (2 * step) / 100.0

    @pytest.mark.parametrize("model", ["black_scholes", "black_76"])
    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("time_to_expiry", [0.25, 1.0, 2.0])
    def test_rho_is_the_actual_derivative(self, model, option_type, time_to_expiry):
        from standard_quant_tools.analysis.pricing import price_option

        base = dict(
            spot=100.0,
            strike=100.0,
            time_to_expiry=time_to_expiry,
            volatility=0.2,
            risk_free_rate=0.05,
            option_type=option_type,
        )
        reported = price_option(**base, model=model)["rho"]
        assert reported == pytest.approx(
            self._finite_difference(model, **base), abs=1e-6
        )

    def test_black_76_rho_is_negative_for_both_calls_and_puts(self):
        """The economics: a higher rate discounts the same forward payoff
        harder, so it cannot help either side."""
        from standard_quant_tools.analysis.pricing import price_option

        for option_type in ("call", "put"):
            got = price_option(
                spot=100.0,
                strike=100.0,
                time_to_expiry=1.0,
                volatility=0.2,
                risk_free_rate=0.05,
                option_type=option_type,
                model="black_76",
            )
            assert got["rho"] < 0
            assert got["rho"] == pytest.approx(-1.0 * got["price"] / 100.0, rel=1e-9)


class TestExpectancyCountsBreakevenTradesAsBreakeven:
    """`(1 - win_rate)` is the not-won rate and includes flat trades, so
    every breakeven was priced at `avg_loser`. The streak loop in the same
    function already handled three states."""

    @pytest.mark.parametrize(
        "trades",
        [
            [10.0, 0.0, -5.0],
            [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [1.0, -1.0],
            [0.0, 0.0, 0.0, 5.0],
        ],
    )
    def test_expectancy_is_the_mean_trade(self, trades):
        """Expectancy IS the mean of the trades. Splitting by sign is a
        presentation choice, and it has to reassemble."""
        from standard_quant_tools.metrics.diagnostics import trade_expectancy

        got = trade_expectancy(pd.DataFrame({"return_pct": trades}))
        assert got["expectancy_pct"] == pytest.approx(float(np.mean(trades)), abs=1e-4)

    def test_a_profitable_set_is_not_reported_as_breakeven(self):
        from standard_quant_tools.metrics.diagnostics import trade_expectancy

        got = trade_expectancy(pd.DataFrame({"return_pct": [10.0, 0.0, -5.0]}))
        assert got["expectancy_pct"] > 1.0


class TestRiskContributionsSumToPortfolioVolatility:
    """`_risk_contributions` documents that these sum to sigma_p exactly --
    "a genuine decomposition rather than an allocation of blame". HRP scaled
    only the volatility, so the two disagreed by exactly sqrt(252)."""

    @pytest.mark.parametrize("periods_per_year", [252, 52, 12, 1])
    def test_hierarchical_risk_parity_keeps_the_invariant(self, periods_per_year):
        from standard_quant_tools.portfolio.construction import (
            hierarchical_risk_parity,
        )

        frame = pd.DataFrame(
            np.random.default_rng(2).normal(0.0004, 0.012, (600, 8)),
            columns=list("ABCDEFGH"),
        )
        got = hierarchical_risk_parity(frame, periods_per_year=periods_per_year)
        assert sum(got["risk_contributions"].values()) == pytest.approx(
            got["portfolio_volatility"], rel=1e-9
        )
        assert got["periods_per_year"] == periods_per_year

    def test_risk_parity_keeps_it_too(self):
        """The sibling, which was already consistent -- both figures
        per-period. This pins that they stay that way."""
        from standard_quant_tools.portfolio.construction import risk_parity

        frame = pd.DataFrame(
            np.random.default_rng(2).normal(0.0004, 0.012, (600, 8)),
            columns=list("ABCDEFGH"),
        )
        got = risk_parity(frame.cov().to_numpy())
        assert sum(got["risk_contributions"].values()) == pytest.approx(
            got["portfolio_volatility"], rel=1e-9
        )

    def test_the_annualization_is_not_hardcoded_to_daily(self):
        """252 was baked in, so monthly returns came back 4.583x
        overstated with nothing in the result naming the convention."""
        from standard_quant_tools.portfolio.construction import (
            hierarchical_risk_parity,
        )

        frame = pd.DataFrame(
            np.random.default_rng(2).normal(0.0004, 0.012, (600, 8)),
            columns=list("ABCDEFGH"),
        )
        daily = hierarchical_risk_parity(frame, periods_per_year=252)
        monthly = hierarchical_risk_parity(frame, periods_per_year=12)
        assert daily["portfolio_volatility"] == pytest.approx(
            monthly["portfolio_volatility"] * math.sqrt(252 / 12), rel=1e-9
        )


class TestSeasonalityPValuesAreNotDoubled:
    """`P(F(1, df) > t^2)` is already two-sided -- squaring the statistic is
    what makes it so. The extra factor of 2 made every per-period p-value
    exactly twice too large, and flipped `significant_after_correction`."""

    def test_raw_p_values_match_the_two_sided_t(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        from standard_quant_tools.analysis.diagnostics import seasonality

        n = 500
        index = pd.bdate_range("2022-01-03", periods=n)
        rng = np.random.default_rng(16)
        series = pd.Series(
            rng.normal(0.0002, 0.011, n) + (index.dayofweek.to_numpy() == 0) * 0.003,
            index=index,
        )

        for row in seasonality(series)["by_period"]:
            expected = 2 * scipy_stats.t.sf(abs(row["t_statistic"]), n - 2)
            assert row["p_value_raw"] == pytest.approx(expected, rel=1e-6), row[
                "period"
            ]

    def test_no_p_value_can_exceed_one(self):
        """The doubling could push a p-value above 1.0, which is not a
        probability. It was clipped, so this is the visible half of the
        fault rather than the whole of it."""
        from standard_quant_tools.analysis.diagnostics import seasonality

        n = 400
        index = pd.bdate_range("2022-01-03", periods=n)
        series = pd.Series(
            np.random.default_rng(7).normal(0.0002, 0.011, n), index=index
        )
        for row in seasonality(series)["by_period"]:
            assert 0.0 <= row["p_value_raw"] <= 1.0


class TestArgumentsAreEitherHonouredOrRefused:
    """The library's stated contract is that an argument a tool does not
    take is REJECTED, not ignored -- `PortfolioOptimizationInput` says so in
    a comment, and every top-level input model sets extra="forbid". These
    were the places that took an argument and quietly did nothing with it."""

    def test_the_package_star_import_works(self):
        """Four source comments had been sorted into `__all__` as string
        literals, so `from standard_quant_tools.agent import *` failed with
        a ModuleNotFoundError naming a TEST FILE PATH as a module."""
        import standard_quant_tools.agent as package

        missing = [n for n in package.__all__ if not hasattr(package, n)]
        assert missing == []

    def test_every_advertised_name_is_importable(self):
        """`get_rally_signal` was in `__all__` and never imported, so the
        tool was unreachable from the package facade."""
        from standard_quant_tools.agent import get_rally_signal

        assert callable(get_rally_signal)

    @pytest.mark.parametrize(
        "tool_name",
        [
            "run_buy_and_hold",
            "run_custom_signal_backtest",
            "run_portfolio_simulation",
            "run_pair_trade_backtest",
        ],
    )
    def test_risk_free_rate_reaches_the_metric(self, tool_name):
        """All four declared it "for Sharpe/Sortino" and discarded it. The
        existing guard asserted only that the FIELD EXISTS, and the one test
        that checked behaviour exercised a single tool."""
        import inspect

        from standard_quant_tools.agent.runtimes.backtest import tools

        source = inspect.getsource(getattr(tools, tool_name))
        assert "input_data.risk_free_rate" in source, (
            f"{tool_name} advertises risk_free_rate but never reads it; a "
            f"caller asking for 4.5% gets a Sharpe measured against 0%"
        )

    def test_a_typo_inside_a_nested_spec_is_rejected(self):
        """`validate_model_spec` exists to catch bad specs and certified one
        `valid: True` while the leakage embargo the caller asked for was 0,
        because the nested models did not forbid extras."""
        from pydantic import ValidationError as PydanticValidationError

        from standard_quant_tools.modeling.specs import ValidationSpec

        with pytest.raises(PydanticValidationError):
            ValidationSpec(
                method="walk_forward",
                n_splits=5,
                train_window=252,
                test_window=63,
                embargo_bars=10,  # the real field is `embargo`
            )

        ok = ValidationSpec(
            method="walk_forward",
            n_splits=5,
            train_window=252,
            test_window=63,
            embargo=10,
        )
        assert ok.embargo == 10

    def test_every_nested_input_model_forbids_extras(self):
        """The top-level models all did; the nested ones reachable from them
        did not, which is where a hallucinated argument actually lands."""
        from pydantic import BaseModel

        from standard_quant_tools import modeling
        from standard_quant_tools.agent import models as agent_models
        from standard_quant_tools.modeling import specs

        offenders = []
        for module in (specs,):
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BaseModel)
                    and obj is not BaseModel
                    and obj.model_config.get("extra") != "forbid"
                ):
                    offenders.append(f"{module.__name__}.{name}")
        assert offenders == []

        for name in ("BLViewInput", "CostScenario"):
            model = getattr(agent_models, name)
            assert model.model_config.get("extra") == "forbid", name


class TestThePythonFallbacksActuallyRun:
    """`technical_indicators_panel`'s Bollinger fallback had never worked.

    It selected ["Upper", "Middle", "Lower"] from a frame whose columns are
    BB_Upper/BB_Middle/BB_Lower -- names this module declares itself, 194
    lines above, under a comment promising they are "exactly the ones the
    per-ticker wrappers use". Every call raised KeyError.

    Nobody noticed because the native path is taken wherever the extension
    builds, 507 of the suite's tests are gated on that extension, and CI
    never verifies it loaded. With the extension blocked the suite still
    reports a pass. This test forces the fallback so the gate cannot hide
    it again.
    """

    INDICATORS = ["rsi", "atr", "adx", "bollinger_bands", "stochastic_oscillator"]

    @staticmethod
    def _panel(n: int = 120):
        def frame(seed: int) -> pd.DataFrame:
            rng = np.random.default_rng(seed)
            close = 100 + np.cumsum(rng.normal(0, 1, n))
            return pd.DataFrame(
                {
                    "Open": close,
                    "High": close + 1,
                    "Low": close - 1,
                    "Close": close,
                    "Volume": rng.integers(1e5, 1e6, n).astype(float),
                },
                index=pd.date_range("2024-01-01", periods=n, freq="D"),
            )

        return {"AAA": frame(1), "BBB": frame(2)}

    @pytest.mark.parametrize("indicator", INDICATORS)
    def test_the_fallback_runs_at_all(self, indicator):
        import standard_quant_tools.indicators.panel as panel

        saved_flag, saved_core = panel.HAS_CPP, panel._cpp_core
        try:
            panel.HAS_CPP, panel._cpp_core = False, None
            got = panel.technical_indicators_panel(
                self._panel(), indicators=[indicator]
            )
        finally:
            panel.HAS_CPP, panel._cpp_core = saved_flag, saved_core
        assert indicator in got

    @pytest.mark.parametrize("indicator", INDICATORS)
    def test_the_fallback_agrees_with_the_kernel(self, indicator):
        """Running is necessary, not sufficient -- a fallback that returns
        different numbers is the harder version of the same bug."""
        import standard_quant_tools.indicators.panel as panel

        if not panel.HAS_CPP:
            pytest.skip("no native extension to compare against")

        data = self._panel()
        saved_flag, saved_core = panel.HAS_CPP, panel._cpp_core
        try:
            native = panel.technical_indicators_panel(data, indicators=[indicator])
            panel.HAS_CPP, panel._cpp_core = False, None
            fallback = panel.technical_indicators_panel(data, indicators=[indicator])
        finally:
            panel.HAS_CPP, panel._cpp_core = saved_flag, saved_core

        left = np.asarray(native[indicator], dtype=float)
        right = np.asarray(fallback[indicator], dtype=float)
        assert left.shape == right.shape
        assert np.allclose(left, right, rtol=1e-9, atol=1e-12, equal_nan=True)


class TestTheTwoFuturesCurveMeasuresAgree:
    """`analyze_futures_curve` and `analyze_roll` describe the same two
    contracts and disagreed by 18.25x, because `roll_analysis` annualized
    the step between them by the FRONT'S REMAINING LIFE rather than by the
    gap between the two expiries."""

    def test_roll_yield_does_not_move_with_the_roll_date(self):
        """It used to. Same prices, same cost, rolling at 90 days vs 1 day:
        101 bps against 9,114 bps. The roll date is not an economic
        variable."""
        from standard_quant_tools.delta_one.futures import roll_analysis

        seen = set()
        for days_to_front in (90.0, 30.0, 5.0, 1.0, 0.5):
            got = roll_analysis(
                front_price=6000.0,
                next_price=6015.0,
                contracts_held=10.0,
                multiplier=50.0,
                days_to_front_expiry=days_to_front,
                days_between_expiries=91.0,
            )
            seen.add(round(got["roll_yield_bps"], 8))
        assert len(seen) == 1, f"roll_yield moved with the roll date: {seen}"

    def test_it_agrees_with_the_curve_tool(self):
        from standard_quant_tools.delta_one.futures import (
            futures_curve,
            roll_analysis,
        )

        front_days, gap_days = 5.0, 91.0
        rolled = roll_analysis(
            front_price=6000.0,
            next_price=6015.0,
            contracts_held=10.0,
            multiplier=50.0,
            days_to_front_expiry=front_days,
            days_between_expiries=gap_days,
        )
        curve = futures_curve(
            contracts=[
                {"label": "H5", "price": 6000.0, "time_to_expiry": front_days / 365},
                {
                    "label": "M5",
                    "price": 6015.0,
                    "time_to_expiry": (front_days + gap_days) / 365,
                },
            ]
        )
        carry_bps = curve["calendar_spreads"][0]["forward_carry_rate"] * 1e4
        assert rolled["roll_yield_bps"] == pytest.approx(carry_bps, rel=1e-9)

    def test_roll_yield_is_null_rather_than_annualized_by_the_wrong_period(self):
        from standard_quant_tools.delta_one.futures import roll_analysis

        got = roll_analysis(
            front_price=6000.0,
            next_price=6015.0,
            contracts_held=10.0,
            multiplier=50.0,
            days_to_front_expiry=5.0,
        )
        assert got["roll_yield_bps"] is None
        assert got["net_roll_cost"] is not None, "the cost is still reported"


class TestCurveCurvatureNeedsThreeCarries:
    """`carries[-1] - 2*carries[len//2] + carries[0]` is only a second
    difference when the count is odd. With two carries it collapsed to
    `c0 - c1`, the NEGATIVE of the first difference, so a steepening curve
    reported a negative curvature against a docstring saying positive means
    steepening. Exactly three contracts -- the case the field description
    named as the minimum -- was the broken one."""

    @staticmethod
    def _linear_carry_curve(n_contracts: int):
        times = [(i + 1) * 30 / 365 for i in range(n_contracts)]
        prices = [6000.0]
        for i in range(1, n_contracts):
            carry = 0.02 + 0.01 * (i - 1)  # linear -> zero second difference
            prices.append(prices[-1] * math.exp(carry * (times[i] - times[i - 1])))
        return [
            {"label": f"C{i}", "price": prices[i], "time_to_expiry": times[i]}
            for i in range(n_contracts)
        ]

    @pytest.mark.parametrize("n_contracts", [4, 5, 6, 7])
    def test_a_linear_carry_curve_has_zero_curvature(self, n_contracts):
        from standard_quant_tools.delta_one.futures import futures_curve

        got = futures_curve(contracts=self._linear_carry_curve(n_contracts))
        assert got["curve_curvature"] == pytest.approx(0.0, abs=1e-12)

    def test_three_contracts_report_null_rather_than_a_sign_flip(self):
        from standard_quant_tools.delta_one.futures import futures_curve

        got = futures_curve(contracts=self._linear_carry_curve(3))
        assert got["curve_curvature"] is None

    def test_a_genuinely_convex_curve_reports_positive(self):
        from standard_quant_tools.delta_one.futures import futures_curve

        times = [(i + 1) * 30 / 365 for i in range(5)]
        prices = [6000.0]
        for i in range(1, 5):
            carry = 0.02 + 0.01 * (i - 1) ** 2  # convex in i
            prices.append(prices[-1] * math.exp(carry * (times[i] - times[i - 1])))
        got = futures_curve(
            contracts=[
                {"label": f"C{i}", "price": prices[i], "time_to_expiry": times[i]}
                for i in range(5)
            ]
        )
        assert got["curve_curvature"] > 0
