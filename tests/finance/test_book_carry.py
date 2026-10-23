"""
The carry of each position and of a set of them (Koijen, Moskowitz,
Pedersen & Vrugt 2018), against closed forms.

A FUTURE PRICED OFF THE CARRY IDENTITY, F = S exp((r - q) T), has KMPV
carry (S - F) / (F T) = (exp(-(r - q) T) - 1) / T = (q - r) + (r - q)^2 T/2
+ O(T^2): q - r to FIRST order in T, with an error that halves when T
halves, and a continuously compounded carry ln(S / F) / T that is q - r
exactly. A short earns the opposite of a long. An FX forward priced by
covered interest parity carries the rate differential discounted one
period, (r_f - r_d) / (1 + r_d T), and a foreign deposit funded at home
carries r_f - r_d exactly; the same short forward's carry, scaled from the
forward's notional to the spot's, is the FX hedge tool's pickup.

THE PLANTED SET, base USD: 2 ES at 5,050 (x 50) against a spot of 5,000
with a quarter to expiry, (5,000 - 5,050) / (5,050 x 0.25) = -3.9604% on
505,000 = -20,000 a year; and 1,000 SAP short at 100 EUR (x 1.125 =
112,500), funded at 3.9% with a 2% dividend and 30 bps of borrow,
3.9% - 2% - 0.3% = +1.6%, +1,800 a year. The set carries -18,200 on a
gross of 617,500, -2.9474%.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List

import pydantic
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes.finance.currency_models import (
    BookCarryInput,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.finance import (
    book_carry,
    covered_interest_forward,
    fx_forward_hedge,
)
from standard_quant_tools.mcp.catalog import _reads_market_data
from standard_quant_tools.mcp.schemas import dereference

from ..surface import synth


def _line(instrument: str, quantity: float, price: float, currency: str, **more):
    return {
        "instrument": instrument,
        "quantity": quantity,
        "price": price,
        "currency": currency,
        **more,
    }


FUTURE = _line(
    "ESZ6", 2, 5050.0, "USD", multiplier=50, spot=5000.0, years_to_expiry=0.25
)
FUNDED_SHORT = _line(
    "SAP",
    -1000,
    100.0,
    "EUR",
    income_rate=0.02,
    financing_rate=0.039,
    borrow_fee_bps=30.0,
)
PLANTED: List[Dict[str, Any]] = [FUTURE, FUNDED_SHORT]
RATES = {"EUR": 1.125}


def _one(line: Dict[str, Any], base: str = "USD", rates=None) -> Dict[str, Any]:
    return book_carry([line], base, rates)["lines"][0]


class TestThePlantedSet:
    def test_each_line_and_the_set_carry_the_worked_amounts(self):
        result = book_carry(PLANTED, "USD", RATES)
        future, short = result["lines"]
        assert future["kind"] == "forward" and short["kind"] == "funded"
        assert future["carry"] == pytest.approx(-50 / (5050 * 0.25), rel=1e-15)
        assert future["annual_carry"] == pytest.approx(-20_000.0, rel=1e-14)
        assert short["side"] == "short"
        assert short["carry"] == pytest.approx(0.016, abs=1e-17)
        assert short["annual_carry"] == pytest.approx(1_800.0, rel=1e-14)
        assert result["gross"] == 617_500.0
        assert result["annual_carry"] == pytest.approx(-18_200.0, rel=1e-14)
        assert result["carry"] == pytest.approx(-18_200.0 / 617_500.0, rel=1e-14)
        assert result["annual_carry_by_currency"]["USD"] == future["annual_carry"]
        assert result["annual_carry_by_kind"]["funded"] == short["annual_carry"]

    def test_the_sets_carry_is_the_notional_weighted_carry_of_its_lines(self):
        result = book_carry(PLANTED, "USD", RATES)
        weighted = math.fsum(line["weight"] * line["carry"] for line in result["lines"])
        assert weighted == pytest.approx(result["carry"], rel=1e-15)
        assert math.fsum(line["weight"] for line in result["lines"]) == pytest.approx(
            1.0, rel=1e-15
        )

    def test_the_order_of_the_lines_changes_no_total(self):
        rng = random.Random(11)
        lines = [
            _line(
                f"F{i}",
                rng.choice([-3, 2, 7]),
                rng.uniform(90, 110),
                rng.choice(["USD", "EUR"]),
                spot=100.0,
                years_to_expiry=rng.uniform(0.05, 1.0),
            )
            for i in range(25)
        ]
        base = book_carry(lines, "USD", {"EUR": 1.0837})
        for _ in range(5):
            rng.shuffle(lines)
            shuffled = book_carry(lines, "USD", {"EUR": 1.0837})
            for name in ("annual_carry", "carry", "gross", "net"):
                assert shuffled[name] == base[name]


class TestAFuturePricedOffTheCarryIdentity:
    S, R, Q = 5000.0, 0.053, 0.015

    def _carry(self, years: float) -> Dict[str, Any]:
        forward = self.S * math.exp((self.R - self.Q) * years)
        return _one(_line("F", 1, forward, "USD", spot=self.S, years_to_expiry=years))

    @pytest.mark.parametrize("years", [2.0, 1.0, 0.5, 0.25, 1 / 12])
    def test_its_carry_is_q_minus_r_to_first_order_in_t(self, years):
        """The error is (r - q)^2 T / 2 to leading order: 0.0722% a year at
        one year, 0.006% at a month."""
        error = self._carry(years)["carry"] - (self.Q - self.R)
        leading = (self.R - self.Q) ** 2 * years / 2
        assert error > 0
        assert error == pytest.approx(leading, rel=(self.R - self.Q) * years)

    def test_the_error_halves_when_t_halves(self):
        """O(T): the ratio of the errors at T and T/2 is 2 (1 - aT/3) /
        (1 - aT/6) + O(T^2) with a = r - q: about aT/3 from 2 (1.3% at one
        year), and closer by half at each halving."""
        years, previous = 1.0, None
        for _ in range(8):
            ratio = (self._carry(years)["carry"] - (self.Q - self.R)) / (
                self._carry(years / 2)["carry"] - (self.Q - self.R)
            )
            a = self.R - self.Q
            assert ratio == pytest.approx(
                2 * (1 - a * years / 3) / (1 - a * years / 6), abs=a * a * years**2
            )
            assert abs(ratio - 2) < a * years / 3 * 1.01
            if previous is not None:
                assert abs(ratio - 2) < 0.6 * previous
            previous = abs(ratio - 2)
            years /= 2

    @pytest.mark.parametrize("years", [2.0, 0.25, 1 / 52])
    def test_its_continuously_compounded_carry_is_q_minus_r(self, years):
        assert self._carry(years)["carry_log"] == pytest.approx(
            self.Q - self.R, abs=1e-13
        )


class TestAShortEarnsTheOpposite:
    def test_a_short_future(self):
        long = _one(FUTURE)
        short = _one({**FUTURE, "quantity": -2})
        assert short["carry"] == -long["carry"]
        assert short["carry_log"] == -long["carry_log"]
        assert short["annual_carry"] == -long["annual_carry"]

    def test_a_funded_short_without_borrow(self):
        long = _one(
            {**FUNDED_SHORT, "quantity": 1000, "borrow_fee_bps": None}, rates=RATES
        )
        short = _one({**FUNDED_SHORT, "borrow_fee_bps": 0.0}, rates=RATES)
        assert long["carry"] == pytest.approx(0.02 - 0.039, abs=1e-17)
        assert short["carry"] == -long["carry"]

    def test_a_funded_short_pays_its_borrow(self):
        short = _one(FUNDED_SHORT, rates=RATES)
        assert short["carry"] == pytest.approx(0.039 - 0.02 - 0.003, abs=1e-17)

    def test_a_long_ignores_a_borrow_fee_it_does_not_pay(self):
        """A borrow fee is read only while short, as on every position."""
        long = _one({**FUNDED_SHORT, "quantity": 1000}, rates=RATES)
        assert long["carry"] == pytest.approx(0.02 - 0.039, abs=1e-17)


class TestFxCarry:
    SPOT, R_USD, R_EUR = 1.1, 0.053, 0.039

    def _forward(self, years: float) -> float:
        return covered_interest_forward(
            self.SPOT,
            first_rate=self.R_EUR,
            first_years=years,
            second_rate=self.R_USD,
            second_years=years,
        )

    def _long_eur_forward(self, years: float) -> Dict[str, Any]:
        return _one(
            _line(
                "EURUSD fwd",
                1e6,
                self._forward(years),
                "USD",
                spot=self.SPOT,
                years_to_expiry=years,
            )
        )

    @pytest.mark.parametrize("years", [1.0, 0.25, 1 / 12])
    def test_a_parity_forwards_carry_is_the_differential_discounted(self, years):
        """(S - F)/(F T) with F = S (1 + r_d T)/(1 + r_f T) is (r_f - r_d) /
        (1 + r_d T) exactly: the rate differential, discounted one period."""
        carry = self._long_eur_forward(years)["carry"]
        assert carry * (1 + self.R_USD * years) == pytest.approx(
            self.R_EUR - self.R_USD, rel=1e-13
        )

    def test_a_foreign_deposit_funded_at_home_carries_the_differential(self):
        deposit = _one(
            _line(
                "EUR cash",
                1e6,
                1.0,
                "EUR",
                income_rate=self.R_EUR,
                financing_rate=self.R_USD,
            ),
            rates={"EUR": self.SPOT},
        )
        assert deposit["carry"] == self.R_EUR - self.R_USD
        assert deposit["annual_carry"] == pytest.approx(-0.014 * 1.1e6, rel=1e-14)

    def test_the_forward_and_the_deposit_agree_as_t_shrinks(self):
        """The two differ by r_d (r_f - r_d) T / (1 + r_d T): O(T)."""
        for years in (1.0, 0.1, 0.01, 0.001):
            gap = self._long_eur_forward(years)["carry"] - (self.R_EUR - self.R_USD)
            assert abs(gap) <= self.R_USD * 0.014 * years * 1.0001

    def test_a_short_forward_scaled_to_spot_is_the_hedge_tools_pickup(self):
        """The hedge sells EUR forward: its carry on the forward's notional,
        times F / S, is its carry on the spot value -- what
        price_fx_forward_hedge reports as hedge_carry."""
        years = 0.25
        forward = self._forward(years)
        short = _one(
            _line(
                "EURUSD fwd",
                -1e6,
                forward,
                "USD",
                spot=self.SPOT,
                years_to_expiry=years,
            )
        )
        hedge = fx_forward_hedge(
            pair="EURUSD",
            spot=self.SPOT,
            days=90,
            base_currency="USD",
            base_rate=self.R_USD,
            foreign_rate=self.R_EUR,
            base_day_count="ACT/360",
            foreign_day_count="ACT/360",
        )
        assert hedge["forward"] == forward
        assert short["carry"] * forward / self.SPOT == pytest.approx(
            hedge["hedge_carry"], rel=1e-14
        )


class TestRefusals:
    @pytest.mark.parametrize(
        "change,match",
        [
            ({"income_rate": 0.01}, "at once"),
            ({"spot": None, "years_to_expiry": None}, "no carry terms"),
            ({"years_to_expiry": None}, "without years_to_expiry"),
            ({"price": 0.0}, "undefined at or below zero"),
            ({"quantity": -2, "borrow_fee_bps": 5.0}, "is not borrowed"),
            ({"years_to_expiry": 120.0}, "in YEARS"),
            ({"spot": math.inf}, "spot"),
        ],
    )
    def test_a_future_line_that_cannot_carry_is_refused(self, change, match):
        line = {**FUTURE, **change}
        line = {k: v for k, v in line.items() if v is not None}
        with pytest.raises(ValidationError, match=match):
            book_carry([line], "USD")

    @pytest.mark.parametrize(
        "change,match",
        [
            ({"borrow_fee_bps": None}, "no borrow_fee_bps"),
            ({"financing_rate": None}, "without financing_rate"),
            ({"price": -5.0}, "undefined below zero"),
            ({"income_rate": 25.0}, "beyond 10"),
        ],
    )
    def test_a_funded_line_that_cannot_carry_is_refused(self, change, match):
        line = {**FUNDED_SHORT, **change}
        line = {k: v for k, v in line.items() if v is not None}
        with pytest.raises(ValidationError, match=match):
            book_carry([line], "USD", RATES)

    def test_a_set_with_no_gross_has_no_carry_but_each_line_has_its_own(self):
        """Every price zero: each funded line still has a rate, the set has
        no gross to weight by, and the tool says so beside the null."""
        lines = [
            _line("A", 10, 0.0, "USD", income_rate=0.02, financing_rate=0.05),
            _line("B", 5, 0.0, "USD", income_rate=0.0, financing_rate=0.05),
        ]
        result = resolve("finance").dispatch(
            "analyze_book_carry", {"positions": lines, "base_currency": "USD"}
        )
        assert result["carry"] is None
        assert result["lines"][0]["carry"] == pytest.approx(-0.03, abs=1e-17)
        assert all(line["weight"] is None for line in result["lines"])
        assert any("carry is null" in w for w in result["warnings"])


class TestTheTool:
    def test_it_dispatches_through_the_finance_runtime(self):
        result = resolve("finance").dispatch(
            "analyze_book_carry",
            {"positions": PLANTED, "base_currency": "USD", "fx_rates": RATES},
        )
        assert result["annual_carry"] == pytest.approx(-18_200.0, rel=1e-14)
        assert result["lines"][1]["carry_log"] is None
        assert any("carry_log is null" in w for w in result["warnings"])

    def test_a_line_with_both_kinds_of_terms_is_refused_at_its_location(self):
        lines = copy.deepcopy(PLANTED)
        lines[0]["financing_rate"] = 0.05
        with pytest.raises(pydantic.ValidationError) as caught:
            BookCarryInput(positions=lines, base_currency="USD", fx_rates=RATES)
        assert caught.value.errors()[0]["loc"][:2] == ("positions", 0)
        assert "at once" in str(caught.value)

    def test_a_funded_short_without_borrow_is_refused_at_its_location(self):
        lines = copy.deepcopy(PLANTED)
        del lines[1]["borrow_fee_bps"]
        with pytest.raises(pydantic.ValidationError) as caught:
            BookCarryInput(positions=lines, base_currency="USD", fx_rates=RATES)
        assert caught.value.errors()[0]["loc"][:2] == ("positions", 1)
        assert "no borrow_fee_bps" in str(caught.value)

    def test_it_takes_no_ticker_and_is_synthesizable(self):
        schema = dereference(BookCarryInput.model_json_schema())
        assert not _reads_market_data(schema)
        arguments, reason = synth.synthesize(BookCarryInput)
        assert arguments is not None, reason
