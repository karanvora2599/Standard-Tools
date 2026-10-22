"""
Idle cash earns the risk-free rate, borrowing pays for itself, and the
Sharpe measures the excess over what cash actually accrued.

Every engine here subtracted a risk-free rate in its Sharpe while the cash
it held earned nothing, its leverage was financed for free and a hedge's
synthetic funding was booked as hedge profit. Each test below plants an
answer that follows from the arithmetic of a cash account -- a book that
holds only cash returns the compounded cash rate, a constant k-times book
returns k*R - (k - 1)*f on every bar, a perfect hedge of a flat market
loses exactly its costs -- and fails on the code before the CHANGELOG entry
of 2026-10-22.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.backtest.engine as engine
import standard_quant_tools.backtest.portfolio_engine as pe
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.backtest.costs import cash_interest, day_count_basis
from standard_quant_tools.backtest.financing import calendar_day_gaps
from standard_quant_tools.backtest.futures_engine import run_futures_simulation
from standard_quant_tools.backtest.futures_hedge_backtest import (
    run_futures_hedge_backtest,
)
from standard_quant_tools.backtest.pairs import run_pair_backtest
from standard_quant_tools.error import ValidationError

# ── shared fixtures ─────────────────────────────────────────────────────


def _frame(close, index):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": np.full(len(close), 1e7),
        },
        index=index,
    )


def _excess_asset(rf=0.04, n=252 * 4, seed=5):
    """An asset that returns rf + 6% a year in excess, with noise."""
    index = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    r = (rf + 0.06) / 252 + rng.normal(0, 0.15 / np.sqrt(252), n)
    return _frame(100 * np.cumprod(1 + r), index)


def _pair_world():
    """A dollar-neutral world: two names sharing a common factor."""
    index = pd.bdate_range("2024-01-02", periods=505)
    rng = np.random.default_rng(0)
    common = rng.normal(0.0003, 0.01, len(index))
    ra = common + rng.normal(0.0002, 0.003, len(index))
    rb = common + rng.normal(-0.0002, 0.003, len(index))
    return {
        "A": _frame(100 * np.cumprod(1 + ra), index),
        "B": _frame(100 * np.cumprod(1 + rb), index),
    }


def _both_paths(fn, module, *args, **kwargs):
    """Run once natively and once through the Python fallback."""
    native = fn(*args, **kwargs)
    saved = module.HAS_CPP
    module.HAS_CPP = False
    try:
        looped = fn(*args, **kwargs)
    finally:
        module.HAS_CPP = saved
    return native, looped


@pytest.fixture
def portfolio_tool(monkeypatch):
    """dispatch('run_portfolio_simulation', ...) over frames we hand it."""
    frames = {}

    def fetch(tickers, *_a, **_k):
        return {t: frames[t] for t in tickers}

    monkeypatch.setattr(
        "standard_quant_tools.agent.runtimes.backtest.tools.fetch_ohlcv_panel_sync",
        fetch,
    )

    def run(data, weights_by_ticker, **kwargs):
        frames.clear()
        frames.update(data)
        first = next(iter(data.values())).index
        payload = {
            "tickers": list(data),
            "start_date": str(first[0].date()),
            "end_date": str(first[-1].date()),
            "target_weights": weights_by_ticker,
            "commission_pct": 0.0,
            "slippage_pct": 0.0,
            "fill_price": "close",
            **kwargs,
        }
        return dispatch("run_portfolio_simulation", payload)

    return run


# ── the futures hedge books no synthetic interest ───────────────────────


class TestTheHedgeBooksNoCollateralInterest:
    """A perfect hedge of a flat market makes nothing and costs its costs.

    The hedge's futures account is funded with twice the peak notional of
    synthetic cash so margin never binds, and the collateral rate was paid
    on all of it and booked as hedge P&L: a flat $100m book at 5% reported
    +$9.94m. It now runs at a zero rate, and what the margin actually posted
    would earn is reported on its own.
    """

    @staticmethod
    def _flat(rate, **kwargs):
        dates = pd.bdate_range("2025-01-02", periods=253)
        return run_futures_hedge_backtest(
            portfolio_values={d: 100_000_000.0 for d in dates},
            future_prices={d: 5000.0 for d in dates},
            multiplier=50.0,
            portfolio_beta=1.0,
            collateral_rate=rate,
            rehedge="monthly",
            **kwargs,
        )

    @pytest.mark.parametrize("rate", [0.0, 0.05, -0.005])
    def test_a_flat_perfect_hedge_loses_exactly_its_costs(self, rate):
        out = self._flat(rate, initial_margin=10_000.0, commission_per_contract=2.0)
        costs = out["total_commission"] + out["total_slippage"]
        # 400 contracts at $2 a contract, once, to put the hedge on.
        assert costs == 800.0
        assert out["combined_pnl"] == -costs
        assert out["hedge_pnl"] == -costs

    def test_no_interest_at_all_without_costs(self):
        out = self._flat(0.05)
        assert out["hedge_pnl"] == 0.0
        assert out["combined_pnl"] == 0.0

    @pytest.mark.parametrize("day_count", ["ACT/365F", "ACT/360"])
    def test_interest_is_reported_apart_on_the_margin_posted(self, day_count):
        dates = pd.bdate_range("2025-01-02", periods=253)
        out = self._flat(
            0.05,
            initial_margin=10_000.0,
            collateral_day_count=day_count,
        )
        days = (dates[-1] - dates[0]).days
        # 400 contracts x $10,000 posted from the first bar to the last.
        expected = 4_000_000.0 * 0.05 * days / day_count_basis(day_count)
        assert out["collateral_interest_on_posted_margin"] == pytest.approx(
            expected, rel=1e-12
        )
        assert out["collateral_interest_base"] == "posted_initial_margin"
        assert out["collateral_day_count"] == day_count
        assert out["hedge_pnl"] == 0.0

    def test_a_nan_rate_is_refused_at_the_library_door(self):
        with pytest.raises(ValidationError, match="collateral_rate"):
            self._flat(float("nan"))


class TestTheFuturesAccountDeclaresItsCollateralBase:
    @staticmethod
    def _run(**kwargs):
        dates = pd.bdate_range("2025-01-02", periods=10)
        return run_futures_simulation(
            prices={d: 100.0 for d in dates},
            target_contracts={dates[0]: 100},
            multiplier=50.0,
            initial_capital=1_000_000.0,
            initial_margin=5000.0,
            **kwargs,
        )

    def test_a_nan_rate_is_refused_rather_than_turning_equity_nan(self):
        with pytest.raises(ValidationError, match="collateral_rate"):
            self._run(collateral_rate=float("nan"))

    def test_act_360_accrues_365_over_360_times_as_much(self):
        """One accrual, over a Friday-to-Monday gap, on the 500,000 of cash
        not posted as margin: 500,000 x 5% x 3 / basis."""
        dates = pd.DatetimeIndex(["2025-01-03", "2025-01-06"])
        for day_count, basis in (("ACT/365F", 365.0), ("ACT/360", 360.0)):
            out = run_futures_simulation(
                prices={d: 100.0 for d in dates},
                target_contracts={dates[0]: 100},
                multiplier=50.0,
                initial_capital=1_000_000.0,
                initial_margin=5000.0,
                collateral_rate=0.05,
                collateral_day_count=day_count,
            )
            assert out["total_collateral_interest"] == pytest.approx(
                500_000.0 * 0.05 * 3 / basis, rel=1e-12
            )

    def test_the_tool_declares_the_base_and_the_day_count(self):
        """Interest accrues on cash NOT posted as margin -- here half the
        equity -- and the result says so, with the day count it used."""
        dates = [str(d.date()) for d in pd.bdate_range("2025-01-02", periods=10)]
        out = dispatch(
            "run_futures_backtest",
            {
                "prices": {d: 100.0 for d in dates},
                "target_contracts": {dates[0]: 100},
                "multiplier": 50.0,
                "initial_capital": 1_000_000.0,
                "initial_margin": 5000.0,
                "collateral_rate": 0.05,
                "collateral_day_count": "ACT/360",
            },
        )
        assert out["collateral_interest_base"] == "cash_excluding_posted_margin"
        assert out["collateral_day_count"] == "ACT/360"
        days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
        # 500,000 of the 1,000,000 is posted; the rest earns the rate.
        assert out["total_collateral_interest"] == pytest.approx(
            500_000.0 * 0.05 * days / 360.0, rel=1e-3
        )


# ── the shared-cash engine credits cash and charges borrowing ───────────


class TestACashBookEarnsTheCompoundedCashRate:
    @pytest.mark.parametrize("day_count", ["ACT/365F", "ACT/360"])
    def test_every_bar_earns_the_rate_on_its_calendar_days(self, day_count):
        data = {"A": _excess_asset()}
        index = data["A"].index
        weights = pd.DataFrame({"A": [0.0]}, index=index[:1])
        for result in _both_paths(
            pe.run_portfolio_simulation,
            pe,
            data,
            weights,
            initial_capital=1_000_000.0,
            cash_rate=0.05,
            day_count=day_count,
        ):
            accrual = 0.05 * (calendar_day_gaps(index) / day_count_basis(day_count))
            expected = 1_000_000.0 * np.cumprod(1.0 + accrual)
            np.testing.assert_allclose(
                result["equity_curve"].to_numpy(), expected, rtol=1e-12
            )
            returns = result["equity_curve"].pct_change().to_numpy()[1:]
            np.testing.assert_allclose(returns, accrual[1:], rtol=0, atol=1e-15)
            assert result["financing"]["interest_credited"] == pytest.approx(
                result["final_equity"] - 1_000_000.0, rel=1e-12
            )

    def test_the_tool_reports_an_excess_return_of_zero(self, portfolio_tool):
        """It reported a NaN Sharpe -- returns of exactly zero -- because the
        cash earned nothing, and a flat book at a 5% rate is 5% short of
        the risk-free asset it is measured against."""
        data = {"A": _excess_asset()}
        first = str(data["A"].index[0].date())
        out = portfolio_tool(data, {"A": {first: 0.0}}, risk_free_rate=0.05)
        assert out["sharpe_ratio"] is not None
        assert abs(out["sharpe_ratio"]) < 1e-9
        gaps = calendar_day_gaps(data["A"].index)
        compounded = float(np.prod(1.0 + 0.05 * gaps / 365.0)) - 1.0
        assert out["total_return"] == pytest.approx(compounded, abs=1e-6)
        assert out["day_count"] == "ACT/365F"
        assert out["financing"]["cash_rate"] == 0.05


class TestHalfInCashScoresLikeFullyInvested:
    """The same asset at 50% and at 100%, rebalanced daily, at a 4% rate.

    The planted Sharpe is the asset's own excess over the rate as cash
    accrues it on this calendar, mean(r - rf * days / 365) / std(r) *
    sqrt(252), computed from the generated returns alone. The tool used to
    report 0.5703 for the 50% book against 0.8375 for 100%: half the book
    sat in cash that earned nothing while the Sharpe charged it the rate.
    """

    def test_both_books_report_the_planted_sharpe(self, portfolio_tool):
        rf = 0.04
        data = {"A": _excess_asset(rf=rf, n=252 * 8)}
        index = data["A"].index
        r = data["A"]["Close"].pct_change().to_numpy()[1:]
        accrued = rf * calendar_day_gaps(index)[1:] / 365.0
        planted = float(np.mean(r - accrued) / np.std(r, ddof=1) * math.sqrt(252))
        assert planted == pytest.approx(0.84, abs=0.02)
        sharpes = {}
        for weight in (1.0, 0.5):
            out = portfolio_tool(
                data,
                {"A": {str(d.date()): weight for d in index}},
                risk_free_rate=rf,
            )
            sharpes[weight] = out["sharpe_ratio"]
        assert sharpes[0.5] == pytest.approx(sharpes[1.0], abs=2e-3)
        assert sharpes[1.0] == pytest.approx(planted, abs=5e-3)


class TestLeverageInvariance:
    """A constant k-times book financed at f returns k*R - (k - 1)*f.

    On every bar, to 1e-12, on both paths, for k below, at and above one:
    cash above zero earns f and cash below zero pays it, so the excess over
    f scales with k and the Sharpe does not move.
    """

    @pytest.mark.parametrize("k", [0.0, 0.5, 2.0, 3.0])
    def test_bar_by_bar(self, k):
        data = {"A": _excess_asset(n=300)}
        index = data["A"].index
        weights = pd.DataFrame({"A": np.full(len(index), k)}, index=index)
        f = 0.05
        accrual = f * calendar_day_gaps(index) / 365.0
        asset = data["A"]["Close"].pct_change().to_numpy()
        for result in _both_paths(
            pe.run_portfolio_simulation,
            pe,
            data,
            weights,
            initial_capital=1_000_000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
            max_gross_leverage=max(k, 1.0),
            max_position_pct=max(k, 1.0),
            cash_rate=f,
            margin_interest_rate=f,
        ):
            got = result["equity_curve"].pct_change().to_numpy()[1:]
            want = k * asset[1:] - (k - 1.0) * accrual[1:]
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


class TestADollarNeutralBookNoLongerCollapses:
    """Long A, short B, rebalanced daily: its cash is its whole equity.

    At a 5% rate its Sharpe went from 0.044 to -0.696 (measured on this
    data drifting rather than rebalanced), because the rate was
    subtracted from a book whose cash earned none of it. Credited, the
    excess over the rate is the spread's own return and the Sharpe stays
    where it was.
    """

    def test_through_the_tool(self, portfolio_tool):
        data = _pair_world()
        dates = [str(d.date()) for d in data["A"].index]
        weights = {"A": {d: 1.0 for d in dates}, "B": {d: -1.0 for d in dates}}
        common = dict(max_gross_leverage=2.0, initial_capital=1_000_000.0)
        at_zero = portfolio_tool(data, weights, **common)
        at_five = portfolio_tool(data, weights, risk_free_rate=0.05, **common)
        assert at_five["sharpe_ratio"] == pytest.approx(
            at_zero["sharpe_ratio"], abs=1e-3
        )
        assert at_five["financing"]["interest_credited"] > 0.0


class TestSegregatedProceeds:
    def test_a_rebate_at_the_cash_rate_is_net_cash(self):
        """Segregated proceeds earning the rebate at the cash rate, with
        negative free cash paying that rate too, is the net-cash account
        in exact arithmetic: (E - L) * f + S * f = (E - L + S) * f."""
        data = _pair_world()
        weights = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=data["A"].index[:1])
        common = dict(
            initial_capital=1_000_000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
            max_gross_leverage=2.0,
            cash_rate=0.05,
            margin_interest_rate=0.05,
        )
        net = pe.run_portfolio_simulation(data, weights, **common)
        seg = pe.run_portfolio_simulation(
            data,
            weights,
            short_proceeds="segregated",
            short_rebate_rate=0.05,
            **common,
        )
        np.testing.assert_allclose(
            seg["equity_curve"].to_numpy(), net["equity_curve"].to_numpy(), rtol=1e-10
        )
        assert seg["financing"]["rebate_credited"] > 0.0

    def test_without_a_rebate_the_proceeds_earn_nothing(self):
        data = _pair_world()
        weights = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=data["A"].index[:1])
        common = dict(
            initial_capital=1_000_000.0, max_gross_leverage=2.0, cash_rate=0.05
        )
        net = pe.run_portfolio_simulation(data, weights, **common)
        seg = pe.run_portfolio_simulation(
            data, weights, short_proceeds="segregated", **common
        )
        assert seg["final_equity"] < net["final_equity"]
        assert seg["financing"]["rebate_credited"] == 0.0

    def test_a_rebate_on_net_cash_proceeds_is_refused(self):
        data = _pair_world()
        weights = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=data["A"].index[:1])
        with pytest.raises(ValidationError, match="short_rebate_rate"):
            pe.run_portfolio_simulation(
                data, weights, max_gross_leverage=2.0, short_rebate_rate=0.03
            )


class TestTheTwoPathsFinanceIdentically:
    """The kernel's financing block is the Python loop's, expression for
    expression. On a one-name book, where no sum is reassociated, the two
    accounts agree to the last bit."""

    @pytest.mark.parametrize("weight", [1.5, -0.8])
    @pytest.mark.parametrize("proceeds", ["net_cash", "segregated"])
    def test_bit_for_bit(self, weight, proceeds):
        if not pe.HAS_CPP:
            pytest.skip("_sqt_core not built")
        data = {"A": _excess_asset(n=200)}
        index = data["A"].index
        weights = pd.DataFrame({"A": [weight, weight]}, index=index[[0, 100]])
        native, looped = _both_paths(
            pe.run_portfolio_simulation,
            pe,
            data,
            weights,
            initial_capital=100_000.0,
            max_gross_leverage=1.5,
            max_position_pct=1.5,
            cash_rate=0.04,
            margin_interest_rate=0.07,
            borrow_fee_bps=80.0,
            short_proceeds=proceeds,
            short_rebate_rate=0.02 if proceeds == "segregated" else 0.0,
            day_count="ACT/360",
        )
        for key in ("equity_curve", "cash_curve"):
            assert np.array_equal(native[key].to_numpy(), looped[key].to_numpy()), key
        assert native["financing"] == looped["financing"]


class TestTheFinancingTermsAreRefusedByName:
    @staticmethod
    def _run(**kwargs):
        data = {"A": _excess_asset(n=30)}
        weights = pd.DataFrame({"A": [0.5]}, index=data["A"].index[:1])
        return pe.run_portfolio_simulation(data, weights, **kwargs)

    @pytest.mark.parametrize(
        "kwargs,name",
        [
            ({"cash_rate": float("nan")}, "cash_rate"),
            ({"cash_rate": 5.0}, "cash_rate"),
            ({"day_count": "30/360"}, "day_count"),
            ({"short_proceeds": "lender"}, "short_proceeds"),
        ],
    )
    def test_refused(self, kwargs, name):
        with pytest.raises(ValidationError, match=name):
            self._run(**kwargs)

    def test_the_model_spec_refuses_a_rebate_on_net_cash_proceeds(self):
        """Caught where `validate_model_spec` can see it, not at evaluation."""
        from pydantic import ValidationError as PydanticValidationError

        from standard_quant_tools.modeling.specs import PortfolioSimSpec

        with pytest.raises(PydanticValidationError, match="short_rebate_rate"):
            PortfolioSimSpec(short_rebate_rate=0.03)
        assert (
            PortfolioSimSpec(
                short_proceeds="segregated", short_rebate_rate=0.03
            ).short_rebate_rate
            == 0.03
        )

    def test_the_tool_bounds_the_cash_rate(self):
        from pydantic import ValidationError as PydanticValidationError

        from standard_quant_tools.agent.models import PortfolioSimulationInput

        base = dict(
            tickers=["A"],
            start_date="2024-01-02",
            end_date="2024-06-28",
            target_weights={"A": {"2024-01-02": 0.5}},
        )
        with pytest.raises(PydanticValidationError, match="cash_rate"):
            PortfolioSimulationInput(**base, cash_rate=5.0)
        assert PortfolioSimulationInput(**base, cash_rate=-0.005).cash_rate == -0.005

    def test_financing_needs_dated_bars(self):
        data = {"A": _excess_asset(n=30).reset_index(drop=True)}
        weights = pd.DataFrame({"A": [0.5]}, index=data["A"].index[:1])
        with pytest.raises(ValidationError, match="DatetimeIndex"):
            pe.run_portfolio_simulation(data, weights, cash_rate=0.05)
        # With every rate at zero nothing accrues, and nothing needs a date.
        assert pe.run_portfolio_simulation(data, weights)["final_equity"] > 0


class TestTheNegativeCashNoteSaysHowMuch:
    """It read "cash went negative at one or more bars", the same words for
    a book financed at zero and one financed at 8%."""

    @staticmethod
    def _levered(**kwargs):
        data = _pair_world()
        weights = pd.DataFrame({"A": [2.0], "B": [0.0]}, index=data["A"].index[:1])
        return pe.run_portfolio_simulation(
            data,
            weights,
            initial_capital=1_000_000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
            max_gross_leverage=2.0,
            max_position_pct=2.0,
            **kwargs,
        )

    def test_financed_for_nothing_names_the_debit(self):
        result = self._levered()
        note = next(w for w in result["warnings"] if w.startswith("cash went negative"))
        cash = result["cash_curve"].to_numpy()
        gaps = calendar_day_gaps(result["cash_curve"].index)[1:]
        debit_years = float((np.maximum(-cash[:-1], 0.0) * gaps / 365.0).sum())
        assert result["financing"]["debit_balance_years"] == pytest.approx(
            debit_years, rel=1e-12
        )
        assert "financed for nothing" in note
        assert f"{debit_years:,.2f} currency-years" in note

    def test_a_charged_rate_names_the_interest(self):
        result = self._levered(margin_interest_rate=0.05)
        note = next(w for w in result["warnings"] if w.startswith("cash went negative"))
        charged = result["financing"]["interest_charged"]
        assert charged > 0.0
        assert f"{charged:,.2f} of interest" in note
        assert "ACT/365F" in note


# ── run_strategy finances what it borrows ───────────────────────────────


def _strategy_prices():
    index = pd.bdate_range("2024-01-02", periods=505)
    rng = np.random.default_rng(0)
    common = rng.normal(0.0003, 0.01, len(index))
    ra = common + rng.normal(0.0002, 0.003, len(index))
    return _frame(100 * np.cumprod(1 + ra), index)


class TestRunStrategyFinancesLeverage:
    """At 5%, a 3x signal read a Sharpe of -0.062 against -0.258 at 1x: the
    two borrowed units cost nothing. Financed at the rate, a constant k-times
    signal returns k*R - (k - 1)*rate/252 on every bar after the first, and
    the Sharpe is the 1x Sharpe."""

    @pytest.mark.parametrize("fill", ["close", "next_open"])
    def test_three_times_is_one_times(self, fill):
        prices = _strategy_prices()
        index = prices.index
        for path in (True, False):
            saved = engine.HAS_CPP
            engine.HAS_CPP = saved and path
            try:
                one = engine.run_strategy(
                    prices,
                    pd.Series(1.0, index=index),
                    commission_pct=0.0,
                    slippage_pct=0.0,
                    risk_free_rate=0.05,
                    fill_price=fill,
                )
                three = engine.run_strategy(
                    prices,
                    pd.Series(3.0, index=index),
                    commission_pct=0.0,
                    slippage_pct=0.0,
                    risk_free_rate=0.05,
                    fill_price=fill,
                )
            finally:
                engine.HAS_CPP = saved
            assert three["sharpe_ratio"] == pytest.approx(one["sharpe_ratio"], abs=1e-3)
            r1 = one["equity_curve"].pct_change().to_numpy()
            r3 = three["equity_curve"].pct_change().to_numpy()
            # From the bar the 3x position is first held over a whole bar.
            start = 1 if fill == "close" else 2
            np.testing.assert_allclose(
                r3[start:], 3.0 * r1[start:] - 2.0 * 0.05 / 252, rtol=0, atol=1e-12
            )

    def test_a_flat_book_earns_the_rate_and_has_no_risk(self):
        prices = _strategy_prices()
        n = len(prices)
        native, looped = _both_paths(
            engine.run_strategy,
            engine,
            prices,
            pd.Series(0.0, index=prices.index),
            risk_free_rate=0.05,
        )
        for result in (native, looped):
            assert result["total_return"] == pytest.approx(
                (1.0 + 0.05 / 252) ** (n - 1) - 1.0, abs=1e-6
            )
            assert math.isnan(result["sharpe_ratio"])
            assert math.isnan(result["sortino_ratio"])

    def test_a_short_pays_its_borrow(self):
        prices = _strategy_prices()
        asset = prices["Close"].pct_change().to_numpy()
        native, looped = _both_paths(
            engine.run_strategy,
            engine,
            prices,
            pd.Series(-1.0, index=prices.index),
            commission_pct=0.0,
            slippage_pct=0.0,
            borrow_fee_bps=100.0,
        )
        for result in (native, looped):
            got = result["equity_curve"].pct_change().to_numpy()[2:]
            np.testing.assert_allclose(got, -asset[2:] - 0.01 / 252, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("fill", ["close", "next_open", "hl2_exploratory"])
    def test_the_two_paths_agree_on_a_financed_book(self, fill):
        prices = _strategy_prices()
        rng = np.random.default_rng(4)
        signal = pd.Series(
            rng.choice([-1.5, -1.0, 0.0, 0.5, 1.0, 2.5], len(prices)),
            index=prices.index,
        )
        native, looped = _both_paths(
            engine.run_strategy,
            engine,
            prices,
            signal,
            risk_free_rate=0.05,
            borrow_fee_bps=40.0,
            fill_price=fill,
        )
        np.testing.assert_allclose(
            native["equity_curve"].to_numpy(),
            looped["equity_curve"].to_numpy(),
            rtol=1e-12,
        )
        for key in ("sharpe_ratio", "sortino_ratio", "total_return"):
            assert native[key] == pytest.approx(looped[key], abs=1e-9), key

    def test_the_grid_is_financed_like_the_single_run(self):
        prices = _strategy_prices()

        def flat(price_data, level):
            return pd.Series(level, index=price_data.index)

        grid = engine.backtest_grid(
            prices, flat, {"level": [0.0, 1.0]}, risk_free_rate=0.05, n_workers=1
        )
        for level in (0.0, 1.0):
            single = engine.run_strategy(
                prices, flat(prices, level), risk_free_rate=0.05
            )
            row = grid[grid["level"] == level].iloc[0]
            assert row["total_return"] == pytest.approx(
                single["total_return"], abs=1e-9
            )


# ── the pair backtest takes the terms its warning names ─────────────────


class TestThePairBacktestTakesItsFinancing:
    @staticmethod
    def _data():
        dates = pd.date_range("2023-01-02", periods=20, freq="B")
        a = [100.0] * 5 + [60.0] * 5 + [100.0] * 5 + [140.0] * 5
        return {"A": _frame(a, dates), "B": _frame([100.0] * 20, dates)}

    def test_borrow_is_charged_and_its_warning_is_gone(self):
        free = run_pair_backtest(
            self._data(),
            "A",
            "B",
            hedge_ratio=1.0,
            entry_z=1.0,
            exit_z=0.3,
            zscore_window=None,
        )
        assert any("borrow_fee_bps=0" in w for w in free["warnings"])
        paid = run_pair_backtest(
            self._data(),
            "A",
            "B",
            hedge_ratio=1.0,
            entry_z=1.0,
            exit_z=0.3,
            zscore_window=None,
            borrow_fee_bps=300.0,
        )
        assert not any("borrow_fee_bps=0" in w for w in paid["warnings"])
        assert paid["financing"]["borrow_charged"] > 0.0
        assert paid["final_equity"] < free["final_equity"]

    def test_every_argument_a_pair_warning_names_is_one_the_tool_takes(self):
        """The simulation's warning said to set borrow_fee_bps on a tool
        that had no such argument, so the advice could not be followed."""
        from standard_quant_tools.agent.models import PairTradeBacktestInput

        fields = set(PairTradeBacktestInput.model_fields)
        for name in ("borrow_fee_bps", "margin_interest_rate", "cash_rate"):
            assert name in fields, name


# ── estimate_trade_cost ─────────────────────────────────────────────────


class TestEstimateTradeCost:
    def test_holding_costs_accrue_once_in_the_breakeven(self):
        """A 90-day short at 7 bps of execution each way and 300 bps of
        borrow must move 2 x 7 + 300 x 90 / 365 = 87.97 bps. It reported
        161.9, the borrow doubled with everything else."""
        out = dispatch(
            "estimate_trade_cost",
            {
                "notional": 1_000_000,
                "side": "sell",
                "commission_model": "pct",
                "commission_pct": 0.0005,
                "spread_model": "fixed_bps",
                "spread_bps": 2.0,
                "short_borrow_bps": 300,
                "holding_days": 90,
            },
        )
        assert out["execution_bps"] == pytest.approx(7.0, abs=1e-9)
        assert out["holding_bps"] == pytest.approx(300 * 90 / 365, abs=1e-4)
        assert out["breakeven_move_bps"] == pytest.approx(
            14.0 + 300 * 90 / 365, abs=1e-4
        )
        assert round(out["breakeven_move_bps"], 1) == 88.0

    def test_a_buy_pays_no_borrow(self):
        out = dispatch(
            "estimate_trade_cost",
            {
                "notional": 1_000_000,
                "side": "buy",
                "commission_model": "none",
                "spread_model": "none",
                "short_borrow_bps": 500,
                "holding_days": 365,
            },
        )
        assert out["legs"] == []
        assert out["total_bps"] == 0.0
        assert out["breakeven_move_bps"] == 0.0
        assert any("borrows no stock" in note for note in out["notes"])

    def test_account_margin_interest_is_not_this_trades_cost(self):
        out = dispatch(
            "estimate_trade_cost",
            {
                "notional": 1_000_000,
                "commission_model": "none",
                "spread_model": "none",
                "margin_cash": -5_000_000,
                "margin_annual_rate": 0.06,
                "holding_days": 30,
            },
        )
        assert out["total_bps"] == 0.0
        assert out["breakeven_move_bps"] == 0.0
        assert out["account_margin_interest"] == pytest.approx(
            5_000_000 * 0.06 * 30 / 365, abs=1e-6
        )


# ── one door for every engine: a cash book is the risk-free asset ───────


class TestEveryDoorCreditsIdleCash:
    """A book that holds nothing but cash at a 5% rate is the risk-free
    asset itself. Every door that reports a Sharpe against that rate used
    to report it short of it -- NaN or below zero -- because its cash
    earned nothing. Each now reports an excess of zero."""

    def test_run_portfolio_simulation(self, portfolio_tool):
        data = {"A": _excess_asset(n=120)}
        first = str(data["A"].index[0].date())
        out = portfolio_tool(data, {"A": {first: 0.0}}, risk_free_rate=0.05)
        assert abs(out["sharpe_ratio"]) < 1e-9

    def test_run_pair_trade_backtest(self, monkeypatch):
        """A dollar-neutral pair holds its equity in cash: at 5% the cash
        earns the rate, so the excess -- and the Sharpe -- is the zero-rate
        one, and the day count and the interest are on the result."""
        data = TestThePairBacktestTakesItsFinancing._data()

        class _Stub:
            def get_ohlcv(self, symbol, *_a, **_k):
                return data[symbol]

        monkeypatch.setattr(
            "standard_quant_tools.agent.runtimes.backtest.tools."
            "DataFactory.get_provider",
            staticmethod(lambda *a, **k: _Stub()),
        )
        payload = {
            "symbol_a": "A",
            "symbol_b": "B",
            "start_date": "2023-01-02",
            "end_date": "2023-01-27",
            "hedge_ratio": 1.0,
            "entry_z": 1.0,
            "exit_z": 0.3,
            "zscore_window": None,
            "commission_pct": 0.0,
            "slippage_pct": 0.0,
        }
        at_zero = dispatch("run_pair_trade_backtest", payload)
        at_five = dispatch(
            "run_pair_trade_backtest", {**payload, "risk_free_rate": 0.05}
        )
        assert at_five["sharpe_ratio"] == pytest.approx(
            at_zero["sharpe_ratio"], abs=2e-3
        )
        assert at_five["day_count"] == "ACT/365F"
        assert at_five["financing"]["cash_rate"] == 0.05
        assert at_five["financing"]["interest_credited"] > 0.0
        assert at_five["total_return"] > at_zero["total_return"]

    def test_run_strategy_both_paths(self):
        prices = _strategy_prices()
        native, looped = _both_paths(
            engine.run_strategy,
            engine,
            prices,
            pd.Series(0.0, index=prices.index),
            risk_free_rate=0.05,
        )
        for result in (native, looped):
            assert result["total_return"] > 0.0
            assert math.isnan(result["sharpe_ratio"])

    def test_evaluate_model_portfolio_summary(self):
        from standard_quant_tools.modeling.portfolio_eval import (
            _spec_financing,
            _summarize_simulation,
        )
        from standard_quant_tools.modeling.specs import PortfolioSimSpec

        spec = PortfolioSimSpec(risk_free_rate=0.05)
        data = {"A": _excess_asset(n=120)}
        weights = pd.DataFrame({"A": [0.0]}, index=data["A"].index[:1])
        simulation = pe.run_portfolio_simulation(
            data, weights, initial_capital=100_000.0, **_spec_financing(spec)
        )
        metrics = _summarize_simulation(
            simulation, 252, spec.risk_free_rate, 0.0, 0.0, day_count=spec.day_count
        )
        assert abs(metrics["sharpe_ratio"]) < 1e-9
        assert metrics["cumulative_return"] > 0.0

    def test_cash_interest_is_the_credit_the_engines_apply(self):
        assert cash_interest(1_000_000.0, 0.05, days=3) == pytest.approx(
            1_000_000.0 * 0.05 * 3 / 365, rel=1e-15
        )
        assert cash_interest(-1.0, 0.05, days=3) == 0.0
