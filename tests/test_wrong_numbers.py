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


class TestBoundsOnTheProductNotJustTheFactors:
    """`MAX_RATE * MAX_TIME_TO_EXPIRY` is 1,000 and `math.exp` overflows at

    about 709.78, so inputs entirely inside every declared field constraint

    reached `exp` and either raised a bare OverflowError or returned a

    number that was not a price."""

    def test_the_declared_bounds_no_longer_admit_an_overflow(self):
        """Sweeping the tool's OWN field bounds: 12.77% raised a bare

        OverflowError and 8.74% returned a forward of exactly 0.0 for a

        6,000 index, classified `future_rich`."""

        from standard_quant_tools.analysis.derivatives import (
            implied_forward_price,
        )
        from standard_quant_tools.error import ValidationError

        rng = np.random.default_rng(0)

        bad = 0

        for _ in range(3000):

            try:

                forward = implied_forward_price(
                    spot=float(rng.uniform(1e-6, 1e12)),
                    time_to_expiry=float(rng.uniform(1e-8, 100.0)),
                    risk_free_rate=float(rng.uniform(-10, 10)),
                    dividend_yield=float(rng.uniform(-10, 10)),
                    borrow_rate=float(rng.uniform(-10, 10)),
                )["forward"]

                if not np.isfinite(forward) or forward <= 0 or forward > 1e12:

                    bad += 1

            except ValidationError:

                pass  # a refusal that names the input is the correct outcome

            except OverflowError:

                bad += 1

        assert bad == 0

    @pytest.mark.parametrize(
        "rate,time_to_expiry", [(10.0, 100.0), (10.0, 70.0), (-10.0, 100.0)]
    )
    def test_it_refuses_rather_than_returning_a_non_price(self, rate, time_to_expiry):

        from standard_quant_tools.analysis.derivatives import (
            implied_forward_price,
        )
        from standard_quant_tools.error import ValidationError

        with pytest.raises(ValidationError):

            implied_forward_price(
                spot=6000.0,
                time_to_expiry=time_to_expiry,
                risk_free_rate=rate,
                dividend_yield=0.0,
                borrow_rate=0.0,
            )

    def test_ordinary_inputs_are_untouched(self):

        from standard_quant_tools.analysis.derivatives import (
            implied_forward_price,
        )

        got = implied_forward_price(
            spot=6000.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            dividend_yield=0.02,
            borrow_rate=0.01,
        )

        assert got["forward"] == pytest.approx(6000.0 * math.exp(0.02), rel=1e-12)

    def test_the_total_return_future_bound_covers_the_product_too(self):
        """The earlier fix bounded the rate and the sum. Not the product."""

        from standard_quant_tools.delta_one.swaps import total_return_future
        from standard_quant_tools.error import ValidationError

        with pytest.raises(ValidationError):

            total_return_future(
                quote=25.0,
                quote_convention="spread_bps",
                underlying_price=6000.0,
                time_to_expiry=100.0,
                reference_rate=9.99,
            )

    def test_dividend_points_names_the_percent_typo(self):
        """4.3 for 4.3% implies 5,308 index points on a 6,000 index -- 88%

        of the index paid out in six months -- and used to come back with

        only the ordinary "It is a position" line."""

        from standard_quant_tools.delta_one.dividends import dividend_points

        constituents = [
            {
                "symbol": "A",
                "shares": 1.0,
                "dividend_per_share": 20.0,
                "ex_date": "2025-04-15",
            }
        ]

        kwargs = dict(
            divisor=1.0,
            as_of="2025-01-01",
            expiry="2025-06-30",
            spot=6000.0,
            future_price=5940.0,
            time_to_expiry=0.5,
        )

        typo = dividend_points(constituents, financing_rate=4.3, **kwargs)

        assert any("percent rather than a fraction" in w for w in typo["warnings"])

        fine = dividend_points(constituents, financing_rate=0.043, **kwargs)

        assert not any("percent rather than a fraction" in w for w in fine["warnings"])


class TestDeltaOneAgreesWithItsOwnCarryIdentity:
    """`compare_expressions` charged borrow to the LONG. The library's carry

    identity, which `carry.forward_price` and `basis.cash_futures_basis`

    both reach through `implied_forward_price`, is `r - q - b`: a holder of

    the physical can lend it out and earn the borrow, which is why a

    hard-to-borrow name's forward trades below carry."""

    EXPRESSION = {
        "kind": "forward",
        "label": "fwd",
        "financing_rate": 0.045,
        "dividend_yield": 0.015,
        "borrow_rate": 0.030,
    }

    def test_a_fair_future_costs_no_carry(self):
        """r - q - b = 0. This reported 600 bps and $6,000,000 on 100mm."""

        from standard_quant_tools.delta_one.carry import forward_price
        from standard_quant_tools.delta_one.expressions import compare_expressions

        reference = forward_price(
            spot=6000.0,
            time_to_expiry=1.0,
            risk_free_rate=0.045,
            dividend_yield=0.015,
            borrow_rate=0.030,
        )

        assert reference["net_carry_rate"] == pytest.approx(0.0, abs=1e-12)

        got = compare_expressions(
            [self.EXPRESSION], notional=1e8, horizon_years=1.0, direction="long"
        )["expressions"][0]

        assert got["carry_bps"] == pytest.approx(0.0, abs=1e-9)

    def test_borrow_is_earned_by_the_long_and_paid_by_the_short(self):
        """It was the other way round, so the party that actually borrows

        the stock was credited the fee."""

        from standard_quant_tools.delta_one.expressions import compare_expressions

        long = compare_expressions(
            [self.EXPRESSION], notional=1e8, horizon_years=1.0, direction="long"
        )["expressions"][0]

        short = compare_expressions(
            [self.EXPRESSION], notional=1e8, horizon_years=1.0, direction="short"
        )["expressions"][0]

        assert long["borrow_bps"] < 0 < short["borrow_bps"]

        assert long["borrow_bps"] == pytest.approx(-short["borrow_bps"])


class TestTheBasketChangesTheAnswer:
    """`etf.py`'s module header says so in capitals. `basket_value` was

    accepted, warned about, and then ignored by `gross_arbitrage_bps`,

    `net_arbitrage_bps`, `classification`, `action` and

    `arbitrage_survives`, all of which came from NAV alone."""

    def test_the_basket_moves_the_classification(self):
        """Against NAV: 30 bp premium, action `create`, net +14.00 bps.

        Against the holdings the fund is 14.93 bps CHEAP, so that create

        buys at 100.45 and sells at 100.30."""

        from standard_quant_tools.delta_one.etf import etf_fair_value

        rich_basket = etf_fair_value(
            etf_price=100.30,
            nav=100.00,
            basket_value=100.45,
            etf_spread_bps=3.0,
            basket_spread_bps=5.0,
        )

        assert rich_basket["priced_against"] == "basket"

        assert rich_basket["premium_vs_reference_bps"] < 0

        assert rich_basket["action"] != "create"

        assert rich_basket["net_arbitrage_bps"] < 0

    def test_a_basket_equal_to_nav_reproduces_the_nav_answer(self):
        """The control the old code passed trivially, because it returned

        this answer no matter what the basket said."""

        from standard_quant_tools.delta_one.etf import etf_fair_value

        common = dict(
            etf_price=100.30, nav=100.00, etf_spread_bps=3.0, basket_spread_bps=5.0
        )

        with_basket = etf_fair_value(basket_value=100.00, **common)

        without = etf_fair_value(**common)

        assert with_basket["classification"] == without["classification"]

        assert with_basket["action"] == without["action"]

        assert with_basket["net_arbitrage_bps"] == pytest.approx(
            without["net_arbitrage_bps"]
        )

    def test_the_argument_is_not_inert(self):
        """The proof the old version failed: two different baskets, two

        different answers."""

        from standard_quant_tools.delta_one.etf import etf_fair_value

        common = dict(
            etf_price=100.30, nav=100.00, etf_spread_bps=3.0, basket_spread_bps=5.0
        )

        cheap = etf_fair_value(basket_value=99.80, **common)

        rich = etf_fair_value(basket_value=100.45, **common)

        assert cheap["premium_vs_reference_bps"] != rich["premium_vs_reference_bps"]

        assert cheap["net_arbitrage_bps"] != rich["net_arbitrage_bps"]


class TestStatisticsMatchTheirDefinitions:
    """Four functions computing something adjacent to what they claimed."""

    def test_jarque_bera_uses_population_moments(self):
        """It standardized by the SAMPLE deviation, shrinking both moments.

        Worst on short samples -- which is where this function's own

        20-observation minimum puts most of its callers. It flipped the

        normality verdict at n = 20, 30 and 40."""

        scipy_stats = pytest.importorskip("scipy.stats")

        from standard_quant_tools.analysis.inference import test_normality

        for n in (20, 30, 40, 120, 500):

            sample = np.random.default_rng(16).standard_t(6, n)

            got = test_normality(sample.tolist())

            expected, _ = scipy_stats.jarque_bera(sample)

            assert got["jarque_bera"] == pytest.approx(expected, rel=1e-9), n

            assert got["skewness"] == pytest.approx(
                scipy_stats.skew(sample), rel=1e-9
            ), n

    def test_decompose_returns_drops_five_observations_not_five_values(self):
        """`np.isin(array, best_five)` removes every observation EQUAL IN

        VALUE to one of the five. On any rounded or capped series that is

        far more than five: measured, it dropped 8 as "the best 5" and 34 as

        "the worst 5", and reported a negative total for a positive series

        with the "lottery ticket rather than an edge" warning attached."""

        from standard_quant_tools.analysis.inference import decompose_returns

        values = [0.05] * 8 + [-0.04] * 4 + [0.001] * 30

        got = decompose_returns(values)

        array = np.array(values)

        order = np.argsort(array, kind="stable")

        keep = np.ones(array.size, dtype=bool)

        keep[order[-5:]] = False

        expected = float(np.prod(1.0 + array[keep]) - 1.0)

        assert got["total_without_best_5"] == pytest.approx(expected, rel=1e-12)

        assert got["total_without_best_5"] > 0, "the series is profitable"

    def test_buy_volume_fraction_weighs_volume_not_bars(self):
        """`(direction > 0).mean()` is the fraction of up DAYS. The sibling

        in `analysis/microstructure.py` computes it size-weighted under the

        same field name; the two were 506x apart on one tape."""

        from standard_quant_tools.analysis.microstructure_estimators import (
            order_flow_imbalance,
        )

        rng = np.random.default_rng(0)

        n = 300

        close = 100 + np.cumsum(rng.normal(0, 0.4, n))

        change = np.diff(close, prepend=close[0])

        # Up days carry ten times the volume, so the two answers must differ.

        volume = np.where(change > 0, 10_000.0, 1_000.0)

        frame = pd.DataFrame(
            {"close": close, "volume": volume},
            index=pd.date_range("2024-01-01", periods=n, freq="D"),
        )

        got = order_flow_imbalance(frame)["buy_volume_fraction"]

        up = change > 0

        assert got == pytest.approx(volume[up].sum() / volume.sum(), rel=1e-9)

        assert abs(got - up.mean()) > 0.3, "the test data must separate the two"

    def test_the_effective_spread_tool_reports_the_effective_spread(self):
        """`_bps_summary` returned the FIRST column whose name contained

        "bps", and `effective_spread` merges the quoted `spread_bps` in

        before assigning `effective_spread_bps`. So the tool whose promise

        is "what each trade ACTUALLY paid against the prevailing midpoint"

        reported the quoted number -- exactly what its own warning says not

        to charge."""

        import inspect

        from standard_quant_tools.agent.runtimes.microstructure import (
            series_tools,
        )

        source = inspect.getsource(series_tools)

        assert '_bps_summary(series, "effective_spread_bps")' in source

        assert '_bps_summary(series, "spread_bps")' in source

        # And the helper must not be able to guess again.

        assert "for column in frame.columns" not in inspect.getsource(
            series_tools._bps_summary
        )


class TestDrawdownPeaksIncludeTheStartingValue:
    """`(1 + r).cumprod()` never contains 1.0, so a first-bar loss sits AT

    the running maximum and shows no drawdown at all. Two functions had it,

    and in both the wrong number sat beside a correct one."""

    def test_a_first_day_crash_is_a_drawdown(self):
        """Reported max_drawdown_pct 0.0 next to portfolio_return_pct

        -18.39% and worst_day_return_pct -20%, in one result."""

        from standard_quant_tools.backtest.stress_test import (
            replay_stress_scenario,
        )

        got = replay_stress_scenario(pd.DataFrame({"SPY": [-0.20, 0.01, 0.01]}), [1.0])

        assert got["max_drawdown_pct"] == pytest.approx(-0.20, abs=1e-9)

    def test_the_monte_carlo_path_drawdown_is_not_zero(self):
        """Reported 0.0 at the 0TH PERCENTILE of a distribution where no

        path is shallower than -40%, beside a final equity of 0.615."""

        from standard_quant_tools.backtesting.trade_analysis import (
            monte_carlo_trade_paths,
        )

        got = monte_carlo_trade_paths([-0.40] + [0.001] * 24, n_paths=200, seed=0)

        assert got["observed_max_drawdown"] == pytest.approx(-0.40, abs=1e-6)

        assert got["worst_max_drawdown"] <= got["observed_max_drawdown"] + 1e-9

    def test_the_deep_tail_is_the_fifth_percentile(self):
        """These are negative numbers, so p05 is the DEEP tail. "The number

        to size on" was documented on p95, the shallowest, and the warning

        printed p05 while calling it the 95th percentile."""

        from standard_quant_tools.agent.runtimes.backtest.trade_tools import (
            MonteCarloTradesResult,
        )
        from standard_quant_tools.backtesting.trade_analysis import (
            monte_carlo_trade_paths,
        )

        got = monte_carlo_trade_paths(
            np.random.default_rng(3).normal(0.004, 0.03, 120).tolist(),
            n_paths=500,
            seed=1,
        )

        assert got["p05_max_drawdown"] < got["p95_max_drawdown"]

        fields = MonteCarloTradesResult.model_fields

        # The capitalized marker sits on exactly one field, and it is the

        # deep tail. (p95's text says "not to size on", so a bare substring

        # check finds "size on" there too -- which is why this looks for

        # the marker rather than the phrase.)

        marker = "NUMBER TO SIZE ON"

        assert marker in (fields["p05_max_drawdown"].description or "")

        assert marker not in (fields["p95_max_drawdown"].description or "")

        assert "SHALLOW" in (fields["p95_max_drawdown"].description or "")


class TestATotalLossIsNotNegativeEquity:
    """Both engines compounded `1 + r` unguarded, so a bar return at or

    below -100% carried equity NEGATIVE and kept compounding. The C++ kernel

    computes this curve on the default path, so fixing only the Python

    fallback fixed nothing -- the guard sits before the dispatch."""

    @staticmethod
    def _frame(closes):

        index = pd.date_range("2024-01-01", periods=len(closes), freq="D")

        return pd.DataFrame(
            {
                "Open": closes,
                "High": closes,
                "Low": closes,
                "Close": closes,
                "Volume": [1e6] * len(closes),
            },
            index=index,
        )

    def test_a_one_x_short_through_a_tripling_is_refused(self):
        """signal -1.0 is inside the documented {-1, 0, 1}. It produced

        equity [10000, 10000, 10000, -10000, -11666, -12833] -- getting

        MORE negative on bars where the short profits -- and a max_drawdown

        of -2.283, deeper than a total loss."""

        from standard_quant_tools.backtest.engine import run_strategy
        from standard_quant_tools.error import ValidationError

        frame = self._frame([20.0, 20.0, 20.0, 60.0, 50.0, 45.0])

        signals = pd.Series([-1.0] * 6, index=frame.index)

        with pytest.raises(ValidationError, match="total loss"):

            run_strategy(
                frame,
                signals,
                initial_capital=10000.0,
                commission_pct=0.0,
                slippage_pct=0.0,
            )

    def test_an_ordinary_strategy_is_untouched(self):

        from standard_quant_tools.backtest.engine import run_strategy

        frame = self._frame([20.0, 21.0, 22.0, 23.0, 22.0, 24.0])

        signals = pd.Series([1.0] * 6, index=frame.index)

        got = run_strategy(
            frame,
            signals,
            initial_capital=10000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
        )

        assert got["final_equity"] > 0

        assert got["max_drawdown"] >= -1.0


class TestPctChangeDoesNotPadAcrossAGap:
    """pandas pads by default, so a halted name's stale price makes a

    trailing return look real on bars it did not trade. Fixed at 51 sites in

    this repo already; five were left."""

    def test_build_target_does_not_fabricate_a_label(self):
        """close=[100, 101, nan, nan, 90, ...] with horizon=2 gave row 1 a

        forward_return of +0.010000 -- the next real print is -10%. As

        `forward_direction` that is class 1.0, "went up", and those rows are

        not NaN so they survive the downstream dropna. The function's own

        docstring says it exists to prevent this."""

        from standard_quant_tools.modeling.dataset.target import build_target
        from standard_quant_tools.modeling.specs import TargetSpec

        close = pd.Series([100.0, 101.0, np.nan, np.nan, 90.0, 91.0, 92.0, 93.0])

        got = build_target(close, TargetSpec(type="forward_return", horizon=2))

        # Rows 0 and 1 look two bars ahead into the gap: unknowable.

        assert bool(np.isnan(got.iloc[0])) and bool(np.isnan(got.iloc[1]))

    @pytest.mark.parametrize(
        "module,attribute",
        [
            ("standard_quant_tools.analysis.rally", None),
            ("standard_quant_tools.backtest.strategies", None),
            ("standard_quant_tools.modeling.dataset.target", None),
            ("standard_quant_tools.modeling.features.market", None),
        ],
    )
    def test_no_bare_pct_change_survives(self, module, attribute):
        """Parsed with `ast`, not grepped. A docstring naming the call --

        "vectorized pandas.Series.pct_change(periods=lookback) call" -- is

        prose, and every substring filter I tried admitted it."""

        import ast
        import importlib
        import inspect

        source = inspect.getsource(importlib.import_module(module))

        offenders = []

        for node in ast.walk(ast.parse(source)):

            if not isinstance(node, ast.Call):

                continue

            func = node.func

            if not (isinstance(func, ast.Attribute) and func.attr == "pct_change"):

                continue

            if not any(kw.arg == "fill_method" for kw in node.keywords):

                offenders.append(node.lineno)

        assert offenders == [], f"{module}: bare pct_change at lines {offenders}"


class TestValidatedDomainsAreActuallyUsed:
    """Two functions validated an input across a range and then handled only

    part of it."""

    @pytest.mark.parametrize("correlation", [-1.0, -0.75, -0.5, -0.25])
    def test_liquidity_var_uses_negative_correlations(self, correlation):
        """`if correlation <= 0` returned the ZERO answer for the whole

        negative half of the domain the validator accepts, while the result

        echoed back `assumed_correlation: -0.5`. The formula two lines down

        is correct across [-1, 1]."""

        from standard_quant_tools.portfolio.construction import (
            liquidity_adjusted_var,
        )

        common = (
            {"A": 5e6, "B": 5e6},
            {"A": 0.32, "B": 0.32},
            {"A": 4e7, "B": 4e7},
        )

        negative = liquidity_adjusted_var(*common, correlation=correlation)

        zero = liquidity_adjusted_var(*common, correlation=0.0)

        assert negative["liquidity_adjusted_var"] < zero["liquidity_adjusted_var"]

        assert negative["assumed_correlation"] == correlation

    def test_liquidity_var_is_monotone_in_correlation(self):

        from standard_quant_tools.portfolio.construction import (
            liquidity_adjusted_var,
        )

        values = [
            liquidity_adjusted_var(
                {"A": 5e6, "B": 5e6},
                {"A": 0.32, "B": 0.32},
                {"A": 4e7, "B": 4e7},
                correlation=rho,
            )["liquidity_adjusted_var"]
            for rho in (-1.0, -0.5, 0.0, 0.5, 1.0)
        ]

        assert values == sorted(values)

        # Two equal, perfectly offsetting positions net to nothing.

        assert values[0] == pytest.approx(0.0, abs=1e-6)

    def test_variance_ratio_does_not_depend_on_where_a_spread_sits(self):
        """`log(abs(x))` folds a zero-crossing spread onto the positive

        half-line, so the answer moved with the level. Measured VR(2)

        0.620832 centred at zero against 0.855876 for the same series

        shifted by +1000, and a trending spread read as mean-reverting."""

        from standard_quant_tools.analysis.stationarity import variance_ratio

        rng = np.random.default_rng(4)

        n = 800

        noise = rng.normal(0, 1, n)

        spread = np.zeros(n)

        for i in range(1, n):

            spread[i] = 0.85 * spread[i - 1] + noise[i]

        for period in (2, 4, 8):

            centred = variance_ratio(spread, period)["variance_ratio"]

            shifted = variance_ratio(spread + 1000.0, period)["variance_ratio"]

            assert centred == pytest.approx(shifted, rel=0.02), period

    def test_variance_ratio_reports_which_differencing_it_used(self):

        from standard_quant_tools.analysis.stationarity import variance_ratio

        rng = np.random.default_rng(4)

        walk = np.cumsum(rng.normal(0, 1, 400))

        assert variance_ratio(walk - walk.mean(), 2)["differencing"] == "level"

        assert variance_ratio(np.abs(walk) + 100.0, 2)["differencing"] == "log"


class TestModelScoringMeansWhatItSays:

    def test_the_majority_baseline_is_the_largest_class_share(self):
        """`max(p, 1 - p)` is the binary formula and this is reached for the

        3-class triple_barrier target too. On [0]*20 + [1]*30 + [2]*50

        predicted all-2 -- which IS the majority-class predictor, accuracy

        0.5000 -- it reported 0.7000, quoting the baseline 20 points above

        the thing it is a baseline for."""

        from standard_quant_tools.modeling.validation.metrics import (
            classification_metrics,
        )

        truth = np.array([0] * 20 + [1] * 30 + [2] * 50, dtype=float)

        got = classification_metrics(
            truth, np.full(100, 2.0), np.tile([0.2, 0.3, 0.5], (100, 1))
        )

        assert got["majority_class_accuracy"] == pytest.approx(0.50)

        assert got["accuracy"] == pytest.approx(0.50)

        assert got["majority_class_accuracy"] <= got["accuracy"] + 1e-12

    def test_the_binary_case_is_unchanged(self):

        from standard_quant_tools.modeling.validation.metrics import (
            classification_metrics,
        )

        truth = np.array([0] * 30 + [1] * 70, dtype=float)

        got = classification_metrics(truth, np.ones(100), np.tile([0.3, 0.7], (100, 1)))

        assert got["majority_class_accuracy"] == pytest.approx(0.70)

    def test_positive_class_proba_picks_the_positive_class(self):
        """THE OTHER PROVEN MUTATION SURVIVOR: returning `1 - column`

        passes 858 tests across 77 live calls, so every classifier's

        predictions become P(down) with nothing noticing. The function

        exists specifically to avoid hardcoding [:, 1]."""

        from standard_quant_tools.modeling.validation.metrics import (
            positive_class_proba,
        )

        proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])

        class _Estimator:

            def __init__(self, classes):

                self.classes_ = np.array(classes)

            def predict_proba(self, X):

                return proba

        rows = np.zeros((3, 2))

        # Ordinary ordering: class 1 is the second column.

        assert np.allclose(
            positive_class_proba(_Estimator([0, 1]), rows), [0.2, 0.7, 0.6]
        )

        # Reversed ordering: it must follow classes_, not the position.

        assert np.allclose(
            positive_class_proba(_Estimator([1, 0]), rows), [0.8, 0.3, 0.4]
        )

    def test_an_unfillable_exposure_target_is_reported(self):
        """`n_dates_below_target_gross` was counted and left sitting in the
        diagnostics dict with `warnings: None`. A caller asking for gross
        1.0 / net 0.0 got a 100%-LONG book at gross 0.5 -- 60 of 60 dates
        short, mean net +0.34 -- and nothing pointed at it.

        The cause is that `vol_scaled` divides by volatility and normalizes
        gross WITHOUT recentring, so one-sided scores produce no short book
        for `apply_exposure_targets` to fill. That is a property of the
        method, not a bug in it; the silence was the bug.
        """
        from standard_quant_tools.modeling.portfolio_eval import (
            transform_predictions_to_weights,
        )
        from standard_quant_tools.modeling.specs import PredictionTransformSpec

        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        names = [f"N{i}" for i in range(40)]
        returns = pd.DataFrame(rng.normal(0, 0.01, (60, 40)), dates, names)
        spec = PredictionTransformSpec(
            method="cross_sectional_zscore",
            volatility_scale=True,
            gross_exposure=1.0,
            net_exposure=0.0,
            volatility_lookback=20,
        )

        one_sided = pd.DataFrame(np.abs(rng.normal(1, 0.2, (60, 40))), dates, names)
        _, diagnostics = transform_predictions_to_weights(one_sided, spec, returns)
        assert diagnostics["n_dates_below_target_gross"] == 60
        assert diagnostics["warnings"], "the shortfall must be stated"
        assert any("gross exposure" in w for w in diagnostics["warnings"]), diagnostics[
            "warnings"
        ]

    def test_a_met_target_is_not_warned_about(self):
        from standard_quant_tools.modeling.portfolio_eval import (
            transform_predictions_to_weights,
        )
        from standard_quant_tools.modeling.specs import PredictionTransformSpec

        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        names = [f"N{i}" for i in range(40)]
        returns = pd.DataFrame(rng.normal(0, 0.01, (60, 40)), dates, names)
        scores = pd.DataFrame(np.abs(rng.normal(1, 0.2, (60, 40))), dates, names)

        weights, diagnostics = transform_predictions_to_weights(
            scores,
            PredictionTransformSpec(
                method="cross_sectional_zscore",
                gross_exposure=1.0,
                net_exposure=0.0,
            ),
            returns,
        )
        assert diagnostics["n_dates_below_target_gross"] == 0
        assert diagnostics["warnings"] == []
        last = weights.iloc[-1]
        assert last.abs().sum() == pytest.approx(1.0, abs=1e-9)
        assert last.sum() == pytest.approx(0.0, abs=1e-9)


class TestTheOrderBookDoesNotClaimWhatItCannotSee:
    @staticmethod
    def _book(n=100, level_one=True, sizes=(100.0, 100.0)):
        columns = {
            "bid_price_0": [99.0] * n,
            "ask_price_0": [101.0] * n,
            "bid_size_0": [sizes[0]] * n,
            "ask_size_0": [sizes[1]] * n,
        }
        second = (98.0, 102.0, 80.0, 80.0) if level_one else (np.nan,) * 4
        columns.update(
            {
                "bid_price_1": [second[0]] * n,
                "ask_price_1": [second[1]] * n,
                "bid_size_1": [second[2]] * n,
                "ask_size_1": [second[3]] * n,
            }
        )
        return pd.DataFrame(columns)

    def test_an_empty_book_makes_no_directional_claim(self):
        """`np.sign(a) != np.sign(b)` is a BOOL array and `nanmean` never
        skips a bool, so every undefined snapshot counted as a
        disagreement -- `sign(nan) != sign(nan)` is True. On an all-zero-size
        book, where both imbalances are None, it reported "Touch and
        cumulative imbalance point opposite ways in 100% of snapshots ...
        fills badly"."""
        from standard_quant_tools.analysis.order_book import book_metrics

        got = book_metrics(self._book(sizes=(0.0, 0.0)))
        assert not any("opposite ways" in w for w in got["warnings"])

    def test_a_real_disagreement_is_still_reported(self):
        """The guard must not silence the finding it exists for."""
        from standard_quant_tools.analysis.order_book import book_metrics

        rng = np.random.default_rng(0)
        n = 100
        frame = pd.DataFrame(
            {
                "bid_price_0": 99 + rng.normal(0, 0.01, n),
                "ask_price_0": 101 + rng.normal(0, 0.01, n),
                "bid_size_0": rng.integers(50, 500, n).astype(float),
                "ask_size_0": rng.integers(50, 500, n).astype(float),
                "bid_price_1": 98 + rng.normal(0, 0.01, n),
                "ask_price_1": 102 + rng.normal(0, 0.01, n),
                "bid_size_1": rng.integers(50, 500, n).astype(float),
                "ask_size_1": rng.integers(50, 500, n).astype(float),
            }
        )
        warned = [w for w in book_metrics(frame)["warnings"] if "opposite ways" in w]
        assert warned, "a genuinely mixed book must still warn"
        # And it must say how many snapshots the figure is computed over.
        assert "snapshots where both are defined" in warned[0]

    def test_a_level_with_no_prices_is_not_a_level(self):
        """`_levels_present` counted COLUMNS. A two-level book whose level-1
        prices are all NaN -- the commonest malformed shape a vendor
        produces -- reported levels_available 2, emitted no one-level
        warning, and returned a depth_slope computed from the single real
        level."""
        from standard_quant_tools.analysis.order_book import book_metrics

        empty = book_metrics(self._book(level_one=False))
        assert empty["levels_available"] == 1
        assert any("one level" in w.lower() for w in empty["warnings"])

        full = book_metrics(self._book(level_one=True))
        assert full["levels_available"] == 2
        assert not any("one level" in w.lower() for w in full["warnings"])


class TestUserVisibleCountsAreReal:
    """Counts in `--help` text, model-facing runtime descriptions and module
    docstrings had drifted: 178 tools (really 198), 145 across "eight
    runtimes" (really 198 across ten), 62 tools in the execution facade
    (really 172), and a `--enable-long-running` help string naming
    `scan_cointegrated_pairs`, which is a library function and not a tool.

    `tests/docs/test_documentation.py` pins numbers appearing in the
    markdown; nothing pinned them in Python, which is why every one of these
    survived. This is a source scan for the specific stale figures rather
    than a general count checker, so it fails loudly if one comes back.
    """

    # "178 tools" and "all 178" were on this list and have been REMOVED:
    # 178 is now the correct size of the analysis facade, so forbidding the
    # phrase would reject a true statement. A guard that outlaws the current
    # answer is worse than no guard -- it teaches the next person to write
    # something vaguer to get past it.
    STALE = [
        "145 tools",
        "eight runtimes",
        "62 tools regardless",
        "scan_cointegrated_pairs is measured",
    ]

    @pytest.mark.parametrize("phrase", STALE)
    def test_the_stale_phrasing_is_gone(self, phrase):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "src" / "standard_quant_tools"
        offenders = [
            path.relative_to(root).as_posix()
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
            and phrase in path.read_text(encoding="utf-8")
        ]
        assert offenders == [], f"{phrase!r} is back in {offenders}"

    def test_the_real_counts(self):
        """The numbers the corrected text now claims."""
        from standard_quant_tools.agent.runtimes import _build
        from standard_quant_tools.agent.tools import TOOL_CATEGORY
        from standard_quant_tools.mcp.catalog import build_catalog

        runtimes = _build()
        assert len(runtimes) == 10
        assert sum(len(r.tool_names) for r in runtimes.values()) == 207
        assert len(build_catalog()) == 207
        # 207 minus the 20 modeling and 9 feature_lab tools, which are
        # deliberately outside the analysis facade.
        assert len(TOOL_CATEGORY) == 178

    def test_every_long_running_name_is_a_real_tool(self):
        """The help text named one that is not."""
        from standard_quant_tools.mcp.catalog import build_catalog
        from standard_quant_tools.mcp.server import LONG_RUNNING

        catalog = set(build_catalog())
        assert set(LONG_RUNNING) <= catalog, set(LONG_RUNNING) - catalog


class TestEnumeratedParametersAreLiterals:
    """Four input models declared `strategy_type: str` and listed the values
    in prose. `tests/surface/synth.py` synthesizes "a" for a bare `str` and
    the first `get_args(...)` entry for a Literal, so a REQUIRED enumerated
    parameter typed `str` produced "a", the tool refused it, and the tool
    then passed every adversarial-input test without its body ever running.
    run_backtest_compact, get_backtest_diagnostics and run_strategy_matrix
    were fuzzed only on their refusal path."""

    def test_the_literal_matches_the_registry(self):
        """`Literal` cannot be built from a dict at runtime, so it is
        written out. This is what keeps the two in step."""
        from typing import get_args

        from standard_quant_tools.agent.models import RegistryStrategy
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        assert set(get_args(RegistryStrategy)) == set(STRATEGY_REGISTRY)

    @pytest.mark.parametrize(
        "model_name,field",
        [
            ("BacktestCompactInput", "strategy_type"),
            ("BacktestDiagnosticsInput", "strategy_type"),
            ("StrategyMatrixInput", "strategies"),
        ],
    )
    def test_the_schema_carries_an_enum(self, model_name, field):
        """Without it a client doing constrained decoding cannot prevent a
        hallucinated value, and the allowed set lives only in prose."""
        import json

        from standard_quant_tools.agent import models

        schema = json.dumps(getattr(models, model_name).model_json_schema())
        assert "enum" in schema, f"{model_name}.{field} advertises no enum"

    def test_synth_now_produces_a_valid_strategy(self):
        """The mechanism, asserted directly: a Literal field synthesizes its
        first allowed value, where a bare `str` synthesized "a"."""
        from typing import get_args

        from standard_quant_tools.agent.models import RegistryStrategy
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        synthesized = get_args(RegistryStrategy)[0]
        assert synthesized in STRATEGY_REGISTRY
        assert synthesized != "a"


class TestHalfLifeIsTheDiscreteOne:
    """`-log(2) / b` is the continuous-time OU half-life, exact only as
    b -> 0. The regression here gives b from differencing, so phi = 1 + b
    and a shock halves after log(0.5)/log(phi) BARS. The old form was biased
    one way -- always "reverts slower":

        phi     true     -ln2/b    bias
        0.50    1.0000   1.3863    +38.63%
        0.80    3.1063   3.4657    +11.57%
        0.95   13.5134  13.8629     +2.59%

    `agent/models.py` screens pairs on min_half_life=5 / max_half_life=126
    and then RANKS on this number, so the bias changed which pairs passed.
    """

    @pytest.mark.parametrize("phi", [0.5, 0.8, 0.9, 0.95])
    def test_it_matches_the_analytic_value(self, phi):
        from standard_quant_tools.analysis.cointegration import half_life

        rng = np.random.default_rng(0)
        n = 120_000
        series = np.zeros(n)
        noise = rng.normal(0, 1, n)
        for i in range(1, n):
            series[i] = phi * series[i - 1] + noise[i]

        expected = math.log(0.5) / math.log(phi)
        assert half_life(pd.Series(series)) == pytest.approx(expected, rel=0.03)

    def test_a_one_bar_reverter_halves_in_one_bar(self):
        """The clearest case: at phi = 0.5 a shock of 1.0 reaches 0.5 after
        exactly one bar. The old formula said 1.3863."""
        from standard_quant_tools.analysis.cointegration import half_life

        rng = np.random.default_rng(1)
        n = 120_000
        series = np.zeros(n)
        noise = rng.normal(0, 1, n)
        for i in range(1, n):
            series[i] = 0.5 * series[i - 1] + noise[i]
        assert half_life(pd.Series(series)) == pytest.approx(1.0, rel=0.03)

    @pytest.mark.parametrize("seed", [2, 3, 4])
    def test_a_random_walk_is_far_outside_the_tradeable_screen(self, seed):
        """Not `inf`: a finite sample of a random walk estimates a slightly
        NEGATIVE b from sampling noise, so phi is a hair under 1 and the
        half-life is large but finite. `inf` is reserved for an estimated
        b >= 0. What matters is that these land far outside the
        max_half_life=126 screen `agent/models.py` applies -- measured 432
        to 1,629 bars."""
        from standard_quant_tools.analysis.cointegration import half_life

        walk = pd.Series(np.cumsum(np.random.default_rng(seed).normal(0, 1, 5000)))
        assert half_life(walk) > 126.0

    def test_a_trending_series_is_infinite(self):
        """An estimated b >= 0 is genuinely no mean reversion."""
        from standard_quant_tools.analysis.cointegration import half_life

        trending = pd.Series(np.arange(2000, dtype=float))
        assert half_life(trending) == float("inf")

    def test_the_cpp_source_uses_the_same_formula(self):
        """Both backends agreed to 7.11e-15 on the WRONG formula, so
        agreement was never the property worth checking. The compiled
        extension in the tree keeps the old value until it is rebuilt; this
        pins the source so a rebuild lands the fix."""
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src/standard_quant_tools/_cpp/src/cointegration.cpp"
        ).read_text(encoding="utf-8")
        assert "std::log(0.5) / std::log(std::abs(phi))" in source
        assert "return -std::log(2.0) / beta;" not in source


class TestTheDeployedModelIsFitLikeTheValidatedOne:
    def test_the_refit_carries_the_sample_weights(self):
        """`adapter.prepare(..., None)` on the final refit, three lines under
        a comment reading "Refit through the SAME adapter the folds used ...
        fitting it differently from the one that was validated is the
        quietest way to make a validation number describe something else".
        A model validated under weighting.method='time_decay' was deployed
        UNWEIGHTED while the manifest recorded the weighted config."""
        import inspect

        from standard_quant_tools.modeling import engine

        source = inspect.getsource(engine)
        assert "full_weights = _fold_sample_weights(model_spec, panel)" in source
        assert "adapter.prepare(model_spec, panel, full_X, full_y, None)" not in source


class TestBothBackendsAgreeOnAnUndefinedSharpe:
    def test_a_no_trade_strategy_reports_nan(self):
        """The kernel returned 0.0 where Python returns NaN. In
        backtest_grid a no-trade combination then scored 0.0 and sorted
        ABOVE genuinely losing combinations; as NaN it sorts to the bottom.
        The same bug class was found and fixed for CALMAR in that file and
        missed for Sharpe."""
        from standard_quant_tools.backtest.engine import run_strategy

        index = pd.date_range("2024-01-01", periods=40, freq="D")
        flat = [20.0] * 40
        frame = pd.DataFrame(
            {
                "Open": flat,
                "High": flat,
                "Low": flat,
                "Close": flat,
                "Volume": [1e6] * 40,
            },
            index=index,
        )
        got = run_strategy(
            frame,
            pd.Series([0.0] * 40, index=index),
            initial_capital=10000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        assert math.isnan(got["sharpe_ratio"])

    def test_an_ordinary_strategy_still_reports_a_number(self):
        from standard_quant_tools.backtest.engine import run_strategy

        rng = np.random.default_rng(0)
        index = pd.date_range("2024-01-01", periods=60, freq="D")
        close = list(20 + np.cumsum(rng.normal(0, 0.2, 60)))
        frame = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": [1e6] * 60,
            },
            index=index,
        )
        got = run_strategy(
            frame,
            pd.Series([1.0] * 60, index=index),
            initial_capital=10000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        assert math.isfinite(got["sharpe_ratio"])

    def test_the_cpp_source_uses_the_same_convention(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src/standard_quant_tools/_cpp/src/backtest.cpp"
        ).read_text(encoding="utf-8")
        assert "periods_per_year) : 0.0;" not in source
        assert source.count("periods_per_year) : kNaN;") == 2


class TestTheOnlyWritingToolIsContained:
    """`export_audit_bundle` is the one tool in the provenance set that
    writes, and `out_path` was a free string chosen by a model, resolved
    directly. Its own code notes "Overwrote an existing file at {out_path}",
    so a bundle could land anywhere the process can reach and clobber what
    it found. `backtest/artifacts.py` and `modeling/artifacts.py` both guard
    this exact shape with `_resolved_within_runs_dir`; this did not.

    Found the hard way: once `strategy_type` stopped being a bare `str` the
    adversarial sweep began actually executing this tool, and it wrote two
    zips into the REPOSITORY ROOT named from its fuzz values --
    `zzz_not_a_valid_choice` and a Japanese/emoji filename -- which were
    then committed.
    """

    @pytest.mark.parametrize(
        "escape",
        ["../escape.zip", "../../escape.zip", "sub/../../../escape.zip"],
    )
    def test_a_relative_path_climbing_out_is_refused(self, escape):
        from standard_quant_tools.agent.runtimes.meta.tools import (
            _contained_bundle_path,
        )
        from standard_quant_tools.error import ValidationError

        with pytest.raises(ValidationError, match="escapes"):
            _contained_bundle_path(escape)

    def test_an_absolute_path_is_allowed(self, tmp_path):
        """Confining these to the sandbox was my first attempt and it was
        wrong: a bundle exists to be handed to someone outside this
        process, and the existing tests write to a tmp_path."""
        from standard_quant_tools.agent.runtimes.meta.tools import (
            _contained_bundle_path,
        )

        target = tmp_path / "bundle.zip"
        assert _contained_bundle_path(str(target)) == target

    def test_an_absolute_path_into_a_missing_directory_is_refused(self, tmp_path):
        from standard_quant_tools.agent.runtimes.meta.tools import (
            _contained_bundle_path,
        )
        from standard_quant_tools.error import ValidationError

        with pytest.raises(ValidationError, match="does not exist"):
            _contained_bundle_path(str(tmp_path / "nope" / "bundle.zip"))

    def test_an_existing_destination_is_refused_not_overwritten(self, tmp_path):
        """The actual hole. `out_path` is a free string chosen by a model
        and the old code noted "Overwrote an existing file at {out_path}"
        when it clobbered one. An audit bundle is evidence."""
        from standard_quant_tools.agent.runtimes.meta.tools import (
            _contained_bundle_path,
        )
        from standard_quant_tools.error import ValidationError

        existing = tmp_path / "already.zip"
        existing.write_bytes(b"PK")
        with pytest.raises(ValidationError, match="already exists"):
            _contained_bundle_path(str(existing))

    @pytest.mark.parametrize(
        "name", ["ok.zip", "sub/ok.zip", "zzz_not_a_valid_choice", "日本語"]
    )
    def test_a_bare_name_does_not_land_in_the_working_directory(
        self, name, tmp_path, monkeypatch
    ):
        """The symptom that exposed all of this: two of these exact fuzz
        values were written into the repository root and committed."""
        from standard_quant_tools.agent.runtimes.meta.tools import (
            _contained_bundle_path,
        )

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path))
        resolved = _contained_bundle_path(name)
        assert resolved.is_relative_to((tmp_path / "bundles").resolve())

    def test_no_fuzz_artifact_sits_in_the_repository_root(self):
        """Anything the surface tests write into the repo root is a tool
        writing outside where it should."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        strays = [
            p.name
            for p in root.iterdir()
            if p.is_file()
            and p.suffix == ""
            and p.name not in {"LICENSE", "Makefile", "Dockerfile", "a"}
            and p.read_bytes()[:2] == b"PK"
        ]
        assert strays == [], f"zip artifacts written into the repo root: {strays}"
