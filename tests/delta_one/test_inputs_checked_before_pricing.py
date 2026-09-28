"""
Inputs that changed the answer without changing whether there was one.

Every case here returned a number, and most returned a plausible one:
two return series of different lengths paired by position and truncated;
a financing rate of 4.5 for 4.5% priced a $100m swap at 450% a year; an
infinite cash component recommended an ETF redemption with a 10,000 bp
edge; a YYYYMMDD integer became 1970-01-01, so a swap across two of them
accrued no financing; negative commissions made a roll a credit; `True`
passed as a price of 1.0; percent-scaled index weights forced $40
trillion of buying on $800bn; and a curve with every contract at one price
was called "mixed" and warned about a kinked segment. Each is planted, and
each has its null case: the input it should still accept, unchanged.

See the CHANGELOG entry of 2026-09-27.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.runtimes.delta_one.models import (
    HedgeEffectivenessInput,
    TotalReturnFutureInput,
    TotalReturnSwapInput,
)
from standard_quant_tools.delta_one._numbers import positive
from standard_quant_tools.delta_one.daycount import to_date, year_fraction
from standard_quant_tools.delta_one.etf import etf_fair_value
from standard_quant_tools.delta_one.futures import futures_curve, roll_analysis
from standard_quant_tools.delta_one.hedging import hedge_effectiveness, tracking_error
from standard_quant_tools.delta_one.rebalance import index_rebalance_flow
from standard_quant_tools.delta_one.swaps import (
    price_total_return_swap,
    total_return_future,
)
from standard_quant_tools.error import ValidationError


class TestPositionalReturnsArePairedOnlyWhenTheyPair:
    @staticmethod
    def _returns(n, seed):
        return np.random.default_rng(seed).normal(0.0, 0.01, n)

    def test_unequal_lists_are_refused_rather_than_truncated(self):
        """500 against 300 was the tracking error of the first 300 of each."""
        with pytest.raises(ValidationError, match="500") as excinfo:
            tracking_error(self._returns(500, 1), self._returns(300, 2))
        assert "300" in str(excinfo.value)

    def test_hedge_effectiveness_refuses_them_too(self):
        """It reported n_observations=200 for 500 against 200, silently."""
        with pytest.raises(ValidationError, match="200"):
            hedge_effectiveness(
                portfolio_returns=self._returns(500, 3),
                hedge_returns=self._returns(200, 4),
                hedge_ratio=-1.0,
            )

    def test_the_schema_refuses_unequal_lists_before_a_call(self):
        with pytest.raises(PydanticValidationError, match="paired by position"):
            HedgeEffectivenessInput(
                portfolio_returns=list(self._returns(50, 5)),
                hedge_returns=list(self._returns(40, 6)),
                hedge_ratio=-1.0,
            )

    def test_dated_series_still_join_on_their_dates(self):
        """The null case: labelled series align on the overlap, as documented."""
        dates = pd.bdate_range("2026-01-01", periods=300)
        portfolio = pd.Series(self._returns(300, 7), index=dates)
        benchmark = pd.Series(self._returns(250, 8), index=dates[50:])
        overlap = dates[50:]
        expected = tracking_error(portfolio[overlap], benchmark[overlap])
        assert tracking_error(portfolio, benchmark) == pytest.approx(expected)
        out = hedge_effectiveness(
            portfolio_returns=portfolio, hedge_returns=benchmark, hedge_ratio=-0.5
        )
        assert out["n_observations"] == 250

    def test_equal_lists_are_unchanged(self):
        a, b = self._returns(250, 9), self._returns(250, 10)
        expected = float(np.std(a - b, ddof=1) * math.sqrt(252))
        assert tracking_error(a, b) == pytest.approx(expected, rel=1e-12)


class TestSwapRatesAreBoundedLikeTheReferenceRate:
    @staticmethod
    def _swap(**over):
        kwargs = dict(
            notional=100e6,
            initial_price=100.0,
            current_price=103.0,
            dividends=1.0,
            financing_rate=0.043,
            spread_bps=45.0,
            time_elapsed=0.5,
        )
        kwargs.update(over)
        return price_total_return_swap(**kwargs)

    def test_a_percent_for_a_fraction_is_named(self):
        """4.5 for 4.5% priced $100m at 450% a year with no word about it."""
        out = self._swap(financing_rate=4.5)
        assert any("percent rather than a fraction" in w for w in out["warnings"])

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e12, -11.0])
    def test_a_non_finite_or_absurd_financing_rate_is_refused(self, value):
        with pytest.raises(ValidationError, match="financing_rate"):
            self._swap(financing_rate=value)

    @pytest.mark.parametrize("value", [float("nan"), float("-inf"), 1e12, True])
    def test_a_non_finite_absurd_or_boolean_spread_is_refused(self, value):
        """1e12 bps was a financing leg of -5e15 on $100m; inf was -inf."""
        with pytest.raises(ValidationError, match="spread_bps"):
            self._swap(spread_bps=value)

    def test_a_decimal_rate_prices_as_before_and_says_nothing(self):
        """The null case: the arithmetic is untouched."""
        out = self._swap()
        assert not any("percent rather than" in w for w in out["warnings"])
        financing = (0.043 + 45.0 / 10_000.0) * 0.5
        assert out["financing_accrued"] == pytest.approx(financing, rel=1e-15)
        assert out["net_pnl"] == pytest.approx(100e6 * (0.04 - financing), rel=1e-12)

    def test_the_schema_bounds_the_rate_at_one_hundred_percent(self):
        """A model is where the percent typo is most likely, so the tool

        surface refuses it outright; a direct call still prices it, with
        the warning above."""
        with pytest.raises(PydanticValidationError, match="financing_rate"):
            TotalReturnSwapInput(
                notional=1e6,
                initial_price=100.0,
                current_price=100.0,
                financing_rate=4.5,
                time_elapsed=0.5,
            )
        TotalReturnSwapInput(
            notional=1e6,
            initial_price=100.0,
            current_price=100.0,
            financing_rate=0.045,
            time_elapsed=0.5,
        )


class TestTotalReturnFutureInputsAreBounded:
    @staticmethod
    def _trf(**over):
        kwargs = dict(
            quote=95.0,
            quote_convention="spread_bps",
            underlying_price=5000.0,
            time_to_expiry=0.5,
            reference_rate=0.043,
        )
        kwargs.update(over)
        return total_return_future(**kwargs)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), 11.0])
    def test_a_non_finite_or_absurd_dividend_yield_is_refused(self, value):
        """NaN made the net carry NaN, with no error."""
        with pytest.raises(ValidationError, match="dividend_yield"):
            self._trf(dividend_yield=value)

    def test_a_percent_dividend_yield_is_named(self):
        """2.0 for 2% took 200% off the net carry, silently."""
        out = self._trf(dividend_yield=2.0)
        assert any(
            "dividend_yield" in w and "percent rather than a fraction" in w
            for w in out["warnings"]
        )

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e12])
    def test_a_non_finite_comparison_spread_is_refused(self, value):
        """An infinite comparison reported a difference of -inf bps."""
        with pytest.raises(ValidationError, match="comparison_spread_bps"):
            self._trf(comparison_spread_bps=value)

    def test_ordinary_inputs_are_unchanged(self):
        out = self._trf(dividend_yield=0.015, comparison_spread_bps=50.0)
        assert out["difference_bps"] == pytest.approx(45.0)
        assert out["comparison_spread_bps"] == 50.0
        assert out["net_carry_rate"] == pytest.approx(0.043 + 0.0095 - 0.015)
        assert not any("percent rather than" in w for w in out["warnings"])

    def test_the_schema_bounds_the_comparison_like_the_quote(self):
        with pytest.raises(PydanticValidationError, match="comparison_spread_bps"):
            TotalReturnFutureInput(
                quote=95.0,
                quote_convention="spread_bps",
                underlying_price=5000.0,
                time_to_expiry=0.5,
                reference_rate=0.043,
                comparison_spread_bps=1e9,
            )


class TestEtfInputsAreFinite:
    _COMMON = dict(etf_price=100.30, nav=100.0, basket_value=100.10)

    @pytest.mark.parametrize("value", [float("inf"), float("nan")])
    def test_a_non_finite_cash_component_is_refused(self, value):
        """inf recommended redeeming at a 10,000 bp edge, survives=True."""
        with pytest.raises(ValidationError, match="cash_component"):
            etf_fair_value(cash_component=value, **self._COMMON)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -5.0])
    def test_the_tolerance_is_checked_as_the_basis_checks_it(self, value):
        """NaN made every discrepancy an arbitrage, inf made all fair."""
        with pytest.raises(ValidationError, match="tolerance_bps"):
            etf_fair_value(tolerance_bps=value, **self._COMMON)

    def test_finite_inputs_are_unchanged(self):
        out = etf_fair_value(cash_component=0.05, tolerance_bps=1.0, **self._COMMON)
        assert out["basket_value_per_share"] == pytest.approx(100.15)
        assert out["premium_vs_reference_bps"] == pytest.approx(
            (100.30 / 100.15 - 1.0) * 10_000.0
        )
        assert out["classification"] == "premium"
        assert out["tolerance_bps"] == 1.0


class TestRollExecutionCostsAreCosts:
    _ROLL = dict(
        front_price=6240.0,
        next_price=6265.0,
        contracts_held=10.0,
        multiplier=50.0,
        days_to_front_expiry=5.0,
    )

    @pytest.mark.parametrize(
        "field, value",
        [
            ("cost_per_contract", -10.0),
            ("cost_per_contract", float("nan")),
            ("spread_ticks", -2.0),
            ("spread_ticks", float("nan")),
            ("tick_value", -12.5),
        ],
    )
    def test_a_negative_or_nan_cost_is_refused(self, field, value):
        """-10 per contract reported an execution cost of -199.01: a

        credit for trading. NaN came back as a NaN cost, no error."""
        kwargs = dict(self._ROLL, spread_ticks=1.0, tick_value=12.5)
        kwargs[field] = value
        with pytest.raises(ValidationError, match=field):
            roll_analysis(**kwargs)

    def test_a_boolean_position_is_not_one_contract(self):
        with pytest.raises(ValidationError, match="contracts_held"):
            roll_analysis(**dict(self._ROLL, contracts_held=True))

    def test_non_negative_costs_are_unchanged(self):
        out = roll_analysis(
            cost_per_contract=2.0, spread_ticks=1.0, tick_value=12.5, **self._ROLL
        )
        next_contracts = 10.0 * 6240.0 / 6265.0
        expected = (10.0 + next_contracts) * (2.0 + 12.5)
        assert out["execution_cost"] == pytest.approx(expected, rel=1e-12)


class TestADateIsNotANumber:
    @pytest.mark.parametrize(
        "value", [20260320, np.int64(20260320), 2026.0320, np.float64(20260320.0)]
    )
    def test_a_number_is_refused_rather_than_read_as_nanoseconds(self, value):
        """pd.Timestamp(20260320) is 1970-01-01 00:00:00.020260320."""
        with pytest.raises(ValidationError, match="is a number, not a date"):
            to_date(value, "expiry")

    def test_a_yyyymmdd_integer_is_told_how_to_pass_it(self):
        with pytest.raises(ValidationError, match="'20260320'"):
            to_date(20260320, "expiry")

    def test_a_swap_across_integer_dates_is_refused_not_free(self):
        """Both dates became 1970-01-01, so no financing accrued at all."""
        with pytest.raises(ValidationError, match="YYYYMMDD"):
            year_fraction(20260102, 20260702)
        with pytest.raises(ValidationError, match="start"):
            price_total_return_swap(
                notional=100e6,
                initial_price=100.0,
                current_price=103.0,
                financing_rate=0.043,
                start_date=20260102,
                valuation_date=20260702,
            )

    @pytest.mark.parametrize(
        "value",
        [
            "20260320",
            "2026-03-20",
            dt.date(2026, 3, 20),
            dt.datetime(2026, 3, 20, 15, 30),
            pd.Timestamp("2026-03-20"),
        ],
    )
    def test_every_date_shape_still_reads(self, value):
        assert to_date(value, "expiry") == dt.date(2026, 3, 20)


class TestPositiveRefusesWhatFiniteRefuses:
    @pytest.mark.parametrize("value", [True, False, "5", np.bool_(True)])
    def test_a_bool_or_a_string_is_not_a_number(self, value):
        """positive(True) was 1.0 and positive('5') was 5.0, while the

        finite and non_negative guards beside it refused both."""
        with pytest.raises(ValidationError, match="must be a number"):
            positive(value, "spot")

    def test_an_etf_priced_off_a_bool_is_refused(self):
        """etf_price='100.3', nav=True reported a 993,000 bp premium."""
        with pytest.raises(ValidationError, match="nav"):
            etf_fair_value(etf_price=100.3, nav=True)
        with pytest.raises(ValidationError, match="etf_price"):
            etf_fair_value(etf_price="100.3", nav=100.0)

    def test_numbers_pass_and_the_range_still_binds(self):
        assert positive(100, "spot") == 100.0
        assert positive(np.float64(1.5), "spot") == 1.5
        assert positive(np.int32(7), "spot") == 7.0
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValidationError, match="spot"):
                positive(bad, "spot")
        with pytest.raises(ValidationError, match="required"):
            positive(None, "spot")


class TestIndexWeightsAreFractions:
    OLD = {"A": 0.30, "B": 0.40, "C": 0.30}
    NEW = {"A": 0.28, "B": 0.37, "XYZ": 0.0035, "C": 0.3465}

    def test_percent_weights_are_refused_with_the_scale(self):
        """On $800bn these forced $40 trillion of buying, turnover 5,000%."""
        with pytest.raises(ValidationError, match="sums to 100") as excinfo:
            index_rebalance_flow(
                old_weights={"A": 40.0, "B": 30.0, "C": 30.0},
                new_weights={"A": 20.0, "B": 30.0, "D": 50.0},
                indexed_assets=800e9,
            )
        assert "percent" in str(excinfo.value)

    def test_basis_point_weights_are_named_as_such(self):
        with pytest.raises(ValidationError, match="basis points"):
            index_rebalance_flow(
                old_weights=self.OLD,
                new_weights={"A": 5_000.0, "B": 5_000.0},
                indexed_assets=800e9,
            )

    def test_a_set_that_is_not_a_whole_index_is_said_to_be(self):
        out = index_rebalance_flow(
            old_weights=self.OLD,
            new_weights={"A": 0.30, "B": 0.37, "C": 0.30},
            indexed_assets=1e9,
        )
        assert any("new_weights sums to 0.97" in w for w in out["warnings"])

    def test_weights_summing_to_one_are_unchanged(self):
        out = index_rebalance_flow(
            old_weights=self.OLD, new_weights=self.NEW, indexed_assets=800e9
        )
        assert not any("sums to" in w for w in out["warnings"])
        assert out["buy_notional"] == pytest.approx((0.0465 + 0.0035) * 800e9)
        assert out["turnover_pct"] == pytest.approx(5.0)


class TestAFlatCurveIsFlat:
    def test_every_contract_at_one_price_is_flat_and_not_kinked(self):
        """It was "mixed", with a warning about a kinked curve hiding the

        segment that is actually dislocated -- of a curve with no segments."""
        out = futures_curve(
            [
                {"time_to_expiry": 0.25, "price": 5000.0},
                {"time_to_expiry": 0.50, "price": 5000.0},
                {"time_to_expiry": 0.75, "price": 5000.0},
            ]
        )
        assert out["shape"] == "flat"
        assert not any("not monotonic" in w for w in out["warnings"])

    def test_a_flat_step_does_not_make_a_rising_curve_non_monotonic(self):
        out = futures_curve(
            [
                {"time_to_expiry": 0.25, "price": 100.0},
                {"time_to_expiry": 0.50, "price": 100.0},
                {"time_to_expiry": 0.75, "price": 101.0},
            ]
        )
        assert out["shape"] == "contango"

    def test_a_kinked_curve_is_still_mixed_and_says_so(self):
        """The null case: a real kink keeps its label and its warning."""
        out = futures_curve(
            [
                {"time_to_expiry": 0.25, "price": 100.0},
                {"time_to_expiry": 0.50, "price": 102.0},
                {"time_to_expiry": 0.75, "price": 101.0},
            ]
        )
        assert out["shape"] == "mixed"
        assert any("not monotonic" in w for w in out["warnings"])
