"""
`compute_margin_requirement` and `finance.margin.margin_requirement`: Reg
T initial and FINRA 4210 maintenance margin on a set of positions.

Each planted value is the rule applied by hand -- Reg T 50% long, 150% of a
short with its proceeds (50% as equity); FINRA 4210(c) 25% long and the two
short tiers; 4210(f)(2)(E)(i) for an uncovered short option; the
leverage-factor multiple of Regulatory Notice 09-53 for a leveraged ETF --
and each line must name the rule it applied, because a requirement whose
source is unstated cannot be checked against a broker's statement.
"""

from __future__ import annotations

import random
from typing import Any, Dict

import pydantic
import pytest

from standard_quant_tools.agent.runtimes.finance import compute_margin_requirement
from standard_quant_tools.agent.runtimes.finance.margin_models import (
    MarginRequirementInput,
)
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.error import ValidationError
from standard_quant_tools.finance.margin import margin_requirement


def _line(instrument: str, quantity: float, price: float, **extra: Any) -> Dict:
    return {
        "instrument": instrument,
        "quantity": quantity,
        "price": price,
        "currency": extra.pop("currency", "USD"),
        **extra,
    }


def _option(kind: str, contracts: float, premium: float, strike: float, under: float):
    return _line(
        f"{kind.upper()}{strike:g}",
        contracts,
        premium,
        multiplier=100,
        kind=kind,
        strike=strike,
        underlying_price=under,
    )


def _one(line: Dict, **kwargs: Any):
    return margin_requirement([line], **kwargs)


class TestPlantedAnswers:
    def test_a_long_stock_needs_half_initially_and_a_quarter_to_maintain(self):
        """$100k long: Reg T 50,000 initial, FINRA 4210(c)(1) 25,000
        maintenance, each line citing its paragraph."""
        r = _one(_line("A", 1_000, 100.0))
        assert (r.initial, r.maintenance, r.reg_t) == (50_000.0, 25_000.0, 50_000.0)
        assert r.initial_binding == "reg_t"
        line = r.lines[0]
        assert "220.12(a)" in line.reg_t_rule and "4210(c)(1)" in line.maintenance_rule

    def test_a_short_under_five_dollars_needs_its_whole_value(self):
        """1,000 short at $4: the greater of $2.50 a share (2,500) and 100%
        of the $4,000 value is 4,000, under 4210(c)(2)."""
        r = _one(_line("S", -1_000, 4.0))
        assert r.maintenance == 4_000.0
        assert "4210(c)(2)" in r.lines[0].maintenance_rule

    def test_a_short_at_twenty_dollars_needs_thirty_percent(self):
        """1,000 short at $20: the greater of $5 a share (5,000) and 30% of
        $20,000 (6,000) is 6,000, under 4210(c)(3). Its Reg T initial is
        150% of the value with the proceeds, 50% of it -- 10,000 -- as
        equity."""
        r = _one(_line("S", -1_000, 20.0))
        assert r.maintenance == pytest.approx(6_000.0, rel=1e-15)
        assert r.reg_t == 10_000.0
        assert "4210(c)(3)" in r.lines[0].maintenance_rule
        assert "220.12(c)" in r.lines[0].reg_t_rule

    @pytest.mark.parametrize(
        "price, expected",
        [
            (1.0, 2_500.0),  # $2.50 a share beats 100% of $1,000
            (4.99, 4_990.0),  # 100% of the value beats $2.50 a share
            (5.0, 5_000.0),  # at $5: $5 a share beats 30%
            (16.0, 5_000.0),  # $5 a share beats 30% of $16,000 = 4,800
            (17.0, 5_100.0),  # 30% of $17,000 beats $5 a share
        ],
    )
    def test_the_per_share_floors_bind_where_the_rule_says(self, price, expected):
        """Either side of $5 and either side of where the per-share amount and
        the percentage cross, the greater of the two is the requirement."""
        r = _one(_line("S", -1_000, price))
        assert r.maintenance == pytest.approx(expected, rel=1e-12)

    def test_an_uncovered_short_call_is_premium_plus_twenty_percent_less_otm(self):
        """One contract, underlying 100, strike 110, premium 2: 20% of the
        underlying (20) less the 10 out of the money is 10, which ties the
        10% minimum; plus the premium is 12 a share, 1,200 a contract."""
        r = _one(_option("call", -1, 2.0, 110, 100))
        assert r.maintenance == pytest.approx(1_200.0, rel=1e-12)
        assert r.lines[0].out_of_the_money == pytest.approx(1_000.0, rel=1e-12)
        assert "4210(f)(2)(E)(i)" in r.lines[0].maintenance_rule
        assert r.lines[0].maintenance_rate is None

    @pytest.mark.parametrize(
        "kind, strike, under, premium, per_share",
        [
            # In the money: 20% of 100 = 20, no OTM amount, + premium 12.
            ("call", 90, 100, 12.0, 32.0),
            # Far out: 20% of 100 - 40 OTM < 10% of 100 = 10, + premium 0.5.
            ("call", 140, 100, 0.5, 10.5),
            # A put 20 out: 20% of 100 - 20 = 0 < 10% of the strike 80 = 8.
            ("put", 80, 100, 1.0, 9.0),
            # A put in the money: 20% of 100 = 20, + premium 15.
            ("put", 110, 100, 15.0, 35.0),
        ],
    )
    def test_short_options_follow_the_rule_and_its_minimums(
        self, kind, strike, under, premium, per_share
    ):
        """In and out of the money, a call and a put: 20% of the underlying
        less the out-of-the-money amount, never under 10% of the underlying
        (call) or strike (put), plus the premium."""
        r = _one(_option(kind, -3, premium, strike, under))
        assert r.maintenance == pytest.approx(per_share * 300, rel=1e-12)
        assert r.lines[0].reg_t == r.lines[0].maintenance

    def test_a_long_option_is_paid_in_full(self):
        """A long option has no loan value: its whole premium is the
        requirement, and the result says the 75% past nine months is not
        applied."""
        r = _one(_option("put", 2, 3.5, 95, 100))
        assert r.maintenance == pytest.approx(700.0, rel=1e-15)
        assert any("nine months" in note for note in r.notes)

    def test_a_three_times_etf_held_long_needs_seventy_five_percent(self):
        """A 3x fund: 25% x 3 = 75% maintenance, and since 4210(b) makes the
        initial requirement at least the maintenance one, 75% initially
        too, above Reg T's 50%."""
        r = _one(_line("UPRO", 1_000, 50.0, kind="etf", leverage=3))
        assert r.maintenance == pytest.approx(37_500.0, rel=1e-15)
        assert r.initial == r.maintenance and r.initial_binding == "maintenance"
        assert "09-53" in r.lines[0].maintenance_rule
        assert r.lines[0].maintenance_rate == 0.75

    @pytest.mark.parametrize(
        "side, leverage, rate",
        [(1, 2, 0.50), (1, 5, 1.0), (-1, 2, 0.60), (-1, 3, 0.90), (-1, 4, 1.0)],
    )
    def test_a_leveraged_rate_is_the_factor_times_the_rule_at_most_all(
        self, side, leverage, rate
    ):
        """Long 25% and short 30%, times the fund's factor, capped at 100% of
        its value."""
        r = _one(_line("L", side * 1_000, 50.0, kind="etf", leverage=leverage))
        assert r.lines[0].maintenance_rate == pytest.approx(rate, rel=1e-15)
        assert r.maintenance == pytest.approx(rate * 50_000, rel=1e-12)

    def test_the_account_needs_two_thousand_dollars_or_a_purchase_cost(self):
        """FINRA 4210(b)(4): $2,000 of equity, except that a purchase alone
        needs no more than its cost. A $1,000 purchase needs 1,000, its
        cost; a $3,000 one needs 2,000, above Reg T's 1,500; a $1,000 short
        needs the full 2,000."""
        small = _one(_line("A", 10, 100.0))
        assert (small.initial, small.initial_binding) == (1_000.0, "minimum_equity")
        mid = _one(_line("A", 30, 100.0))
        assert (mid.initial, mid.initial_binding) == (2_000.0, "minimum_equity")
        short = _one(_line("S", -10, 100.0))
        assert short.minimum_equity == 2_000.0 and short.initial == 2_000.0

    def test_lines_add_up_and_order_does_not_matter(self):
        """The account's amounts are the lines' sums, correctly rounded,
        whatever order the lines come in."""
        rng = random.Random(11)
        lines = [
            _line(
                f"N{i}",
                rng.choice([-1, 1]) * rng.randint(1, 5_000),
                rng.uniform(0.5, 300),
            )
            for i in range(50)
        ]
        first = margin_requirement(lines)
        assert first.maintenance == pytest.approx(
            sum(x.maintenance for x in first.lines), rel=1e-12
        )
        rng.shuffle(lines)
        again = margin_requirement(lines)
        assert (first.maintenance, first.reg_t) == (again.maintenance, again.reg_t)

    def test_a_line_in_another_currency_is_read_in_dollars(self):
        """A short at 4.80 EUR with 1 EUR = 1.10 USD is $5.28 a share, so it
        is in the $5-and-over tier: max($5 x 1,000, 30% x $5,280) = 5,000."""
        r = margin_requirement(
            [_line("E", -1_000, 4.80, currency="EUR")], "USD", {"EUR": 1.10}
        )
        assert r.maintenance == pytest.approx(5_000.0, rel=1e-15)
        assert "4210(c)(3)" in r.lines[0].maintenance_rule


class TestHouseRates:
    def test_a_house_rate_at_or_above_the_rule_raises_the_line(self):
        """A house rate above the rule's sets the line's requirement and the
        line says so."""
        r = margin_requirement([_line("A", 1_000, 100.0)], house_rates={"A": 0.40})
        assert r.maintenance == pytest.approx(40_000.0, rel=1e-15)
        assert r.lines[0].house_rate == 0.40
        assert "house rate" in r.lines[0].maintenance_rule

    def test_the_per_share_floor_still_binds_above_a_house_rate(self):
        """35% on a $6 short is $2,100, under the $5-a-share floor of
        $5,000, which binds."""
        r = margin_requirement([_line("S", -1_000, 6.0)], house_rates={"S": 0.35})
        assert r.maintenance == pytest.approx(5_000.0, rel=1e-15)

    @pytest.mark.parametrize(
        "line, rate, minimum",
        [
            (_line("A", 100, 10.0), 0.20, "25%"),
            (_line("A", -100, 10.0), 0.29, "30%"),
            (_line("A", -100, 3.0), 0.50, "100%"),
            (_line("A", 100, 10.0, kind="etf", leverage=2), 0.45, "50%"),
        ],
    )
    def test_a_house_rate_below_the_rule_is_refused_naming_it(
        self, line, rate, minimum
    ):
        """A house rate can only raise a requirement, so one below the line's
        rule is refused with the minimum that rule sets."""
        with pytest.raises(ValidationError) as caught:
            margin_requirement([line], house_rates={"A": rate})
        assert f"below the {minimum} minimum" in str(caught.value)

    def test_a_house_rate_for_an_instrument_not_held_is_refused(self):
        """Instruments match exactly; a key that raises no line is a typo, not
        a rate to ignore."""
        with pytest.raises(ValidationError, match="no position holds"):
            margin_requirement([_line("A", 100, 10.0)], house_rates={"a": 0.5})

    def test_a_house_rate_on_an_option_is_refused(self):
        """The option rule has two rates, so one house rate on market value
        cannot say which it raises."""
        line = _option("call", -1, 2.0, 110, 100)
        with pytest.raises(ValidationError, match="option line"):
            margin_requirement([line], house_rates={line["instrument"]: 0.5})


class TestRefusals:
    @pytest.mark.parametrize(
        "line, needle",
        [
            (_line("A", 1, -1.0), "is negative"),
            (_line("A", 1, 1.0, multiplier=50), "multiplier=50"),
            (_line("A", 1, 1.0, leverage=2), "leverage=2"),
            (_line("A", 1, 1.0, kind="call", strike=1.0), "needs strike and"),
            (_line("A", 1, 1.0, strike=1.0), "Those are an option's"),
            (_line("A", 1, 1.0, kind="future"), "must be one of"),
            (_line("A", 1, 1.0, kind="etf", leverage=11), "must be in [1, 10]"),
        ],
    )
    def test_a_line_margin_cannot_read_is_refused(self, line, needle):
        """Each input the rules cannot read is refused by name, with what to
        give instead."""
        with pytest.raises(ValidationError) as caught:
            margin_requirement([line])
        assert needle in str(caught.value)

    def test_the_base_currency_is_dollars(self):
        """The per-share and minimum amounts are dollars, so another base
        currency would misread them."""
        with pytest.raises(ValidationError, match="base_currency='USD'"):
            margin_requirement([_line("A", 1, 1.0, currency="EUR")], "EUR")

    def test_the_input_model_refuses_an_option_without_its_terms(self):
        """The tool's input refuses an option line with no strike or underlying
        when it is built, before the library runs."""
        with pytest.raises(pydantic.ValidationError) as caught:
            MarginRequirementInput(
                positions=[_line("C", -1, 2.0, multiplier=100, kind="call")],
                base_currency="USD",
            )
        assert "needs strike and underlying_price" in str(caught.value)


class TestTheTool:
    def test_dispatch_reports_each_line_and_the_excess(self):
        """Through the tool: a $100k long and a $20k short need 31,000 to
        maintain and 60,000 initially, and $70k of equity has 10,000 and
        39,000 over them."""
        result = dispatch(
            "compute_margin_requirement",
            {
                "positions": [_line("A", 1_000, 100.0), _line("S", -1_000, 20.0)],
                "base_currency": "USD",
                "equity": 70_000,
            },
        )
        assert result["maintenance"] == pytest.approx(31_000.0, rel=1e-12)
        assert result["initial"] == 60_000.0
        assert result["excess_over_initial"] == pytest.approx(10_000.0, rel=1e-12)
        assert result["excess_over_maintenance"] == pytest.approx(39_000.0, rel=1e-12)
        assert [line["instrument"] for line in result["lines"]] == ["A", "S"]

    def test_without_equity_the_excess_is_null_with_the_reason(self):
        """With no equity there is nothing to compare, so the excess is null
        and the warnings say why."""
        result = compute_margin_requirement(
            MarginRequirementInput(positions=[_line("A", 10, 1.0)], base_currency="USD")
        )
        assert result.excess_over_initial is None
        assert any("no equity was given" in w for w in result.warnings)
