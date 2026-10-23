"""
Positions valued in one base currency, against answers worked by hand.

THE PLANTED BOOK. Four lines in four currencies, at rates that are exact in
binary, so every notional and every total is exact and can be written down:

    ES    +3     x 5,000.25 x 50  USD  x 1          = +750,037.50
    SAP   -100   x 180            EUR  x 1.125      =  -20,250.00
    7203  +1,000 x 2,500          JPY  x 0.0078125  =  +19,531.25
    VOD   -2,000 x 0.75           GBP  x 1.25       =   -1,875.00

net 747,443.75, gross 791,693.75, long 769,568.75, short 22,125.00.

Beyond the planted book: each notional is the correctly rounded product of
its factors, so the order they are multiplied in cannot change a bit, and
every total is `math.fsum`, so the order of the positions cannot either.
Each of those is checked against an exact rational reference, on values
where the naive float arithmetic does depend on the order. Every refusal
is pinned with the argument it names. See the CHANGELOG entry of
2026-10-22.
"""

from __future__ import annotations

import functools
import itertools
import math
import operator
import random
from fractions import Fraction
from typing import Any, Dict, List

import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.finance import (
    ACTIVE_CURRENCIES,
    MAX_POSITIONS,
    WITHDRAWN_CURRENCIES,
    currency_code,
    exact_product,
    exposure_summary,
    nearest_currency,
    position_values,
)

PLANTED: List[Dict[str, Any]] = [
    {
        "instrument": "ES",
        "quantity": 3,
        "price": 5000.25,
        "multiplier": 50,
        "currency": "USD",
    },
    {"instrument": "SAP", "quantity": -100, "price": 180.0, "currency": "EUR"},
    {"instrument": "7203", "quantity": 1000, "price": 2500.0, "currency": "JPY"},
    {"instrument": "VOD", "quantity": -2000, "price": 0.75, "currency": "GBP"},
]
PLANTED_RATES = {"EUR": 1.125, "JPY": 0.0078125, "GBP": 1.25}


def _line(
    instrument: str,
    quantity: float,
    price: float,
    currency: str = "USD",
    multiplier: float = 1.0,
) -> Dict[str, Any]:
    return {
        "instrument": instrument,
        "quantity": quantity,
        "price": price,
        "currency": currency,
        "multiplier": multiplier,
    }


def _random_book(seed: int, n: int = 200) -> List[Dict[str, Any]]:
    """Lines with many-digit prices and rates, in four currencies, so that
    float products and running sums do depend on their order."""
    rng = random.Random(seed)
    codes = ["USD", "EUR", "JPY", "GBP"]
    return [
        _line(
            f"I{i}",
            rng.choice([-1, 1]) * rng.randint(1, 50_000),
            rng.uniform(0.01, 5_000.0),
            codes[i % len(codes)],
            rng.choice([1.0, 50.0, 100.0, 1_000.0, 0.1]),
        )
        for i in range(n)
    ]


RANDOM_RATES = {"EUR": 1.0873, "JPY": 0.006713, "GBP": 1.2719}


def _fraction_notional(line: Dict[str, Any], rate: float) -> float:
    exact = (
        Fraction(line["quantity"])
        * Fraction(line["price"])
        * Fraction(line.get("multiplier", 1.0))
        * Fraction(rate)
    )
    return float(exact)  # Fraction -> float is correctly rounded


def _floats(summary: Any) -> List[str]:
    """Every number of a summary as its exact hex, so equality is bitwise."""
    out = [
        summary.gross.hex(),
        summary.net.hex(),
        summary.long.hex(),
        summary.short.hex(),
    ]
    for c in summary.currencies:
        out += [c.currency, c.fx_rate.hex(), c.local_net.hex(), c.net.hex()]
        out += [c.gross.hex(), c.long.hex(), c.short.hex(), c.share_of_gross.hex()]
    return out


class TestTheNotionalIdentities:
    def test_the_planted_book_values_to_its_hand_worked_notionals(self):
        """Each line is quantity x price x multiplier x rate, exactly."""
        values = position_values(PLANTED, "USD", PLANTED_RATES)
        assert [v.local_notional for v in values] == [
            750_037.5,
            -18_000.0,
            2_500_000.0,
            -1_500.0,
        ]
        assert [v.base_notional for v in values] == [
            750_037.5,
            -20_250.0,
            19_531.25,
            -1_875.0,
        ]
        assert [v.side for v in values] == ["long", "short", "long", "short"]
        assert [v.fx_rate for v in values] == [1.0, 1.125, 0.0078125, 1.25]

    def test_the_planted_book_totals_to_its_hand_worked_sums(self):
        """Net is the signed sum, gross the absolute sum, long and short
        the long and short lines' sums -- and long - short is net."""
        summary = exposure_summary(PLANTED, "USD", PLANTED_RATES)
        assert summary.net == 747_443.75
        assert summary.gross == 791_693.75
        assert summary.long == 769_568.75
        assert summary.short == 22_125.0
        assert summary.long - summary.short == summary.net
        assert (summary.positions, summary.long_positions) == (4, 2)
        assert summary.short_positions == 2

    def test_per_currency_exposure_and_its_share_of_gross(self):
        """Each currency's line, in base, with local_net in its own
        currency and the share its gross is of the holdings' gross."""
        summary = exposure_summary(PLANTED, "USD", PLANTED_RATES)
        by_code = {c.currency: c for c in summary.currencies}
        assert [c.currency for c in summary.currencies] == ["EUR", "GBP", "JPY", "USD"]
        assert by_code["EUR"].local_net == -18_000.0
        assert by_code["EUR"].net == -20_250.0
        assert by_code["EUR"].short == 20_250.0
        assert by_code["EUR"].long == 0.0
        assert by_code["JPY"].local_net == 2_500_000.0
        assert by_code["JPY"].net == 19_531.25
        assert by_code["USD"].share_of_gross == 750_037.5 / 791_693.75
        assert math.fsum(c.share_of_gross for c in summary.currencies) == (
            pytest.approx(1.0, abs=1e-15)
        )
        assert summary.share_of_gross_null_reason is None

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_the_identities_hold_on_a_random_book(self, seed):
        """Net and gross are the correctly rounded sums of the per-line
        notionals; long - short equals net to the rounding of three
        correctly rounded sums."""
        book = _random_book(seed)
        values = position_values(book, "USD", RANDOM_RATES)
        summary = exposure_summary(book, "USD", RANDOM_RATES)
        assert summary.net == math.fsum(v.base_notional for v in values)
        assert summary.gross == math.fsum(abs(v.base_notional) for v in values)
        assert summary.long - summary.short == pytest.approx(
            summary.net, rel=4 * 2**-52, abs=4 * 2**-52 * summary.gross
        )
        assert summary.long + summary.short == pytest.approx(
            summary.gross, rel=4 * 2**-52
        )

    def test_a_base_currency_line_has_a_rate_of_exactly_one(self):
        """Its base notional is its local notional, bit for bit, whether or
        not the table also gives the base's own rate of 1."""
        book = _random_book(7)
        for rates in (RANDOM_RATES, {**RANDOM_RATES, "USD": 1.0}):
            for value in position_values(book, "USD", rates):
                if value.currency == "USD":
                    assert value.fx_rate == 1.0
                    assert value.base_notional.hex() == value.local_notional.hex()

    def test_a_negative_price_is_kept_and_its_notional_takes_the_sign(self):
        """A long at -37.63 (crude, April 2020) has a negative notional and
        is still long: long - short = net holds, long + short is not gross."""
        book = [
            _line("CL", 2, -37.63, multiplier=1000),
            _line("NG", -1, 2.5, multiplier=10_000),
            _line("ZERO", 100, 0.0),
        ]
        values = position_values(book, "USD")
        assert values[0].side == "long"
        assert values[0].base_notional == exact_product(2.0, -37.63, 1000.0)
        assert values[0].base_notional < 0
        assert values[2].base_notional == 0.0
        summary = exposure_summary(book, "USD")
        assert summary.long == values[0].base_notional
        assert summary.short == 25_000.0
        assert summary.net == summary.long - summary.short
        assert summary.gross == 75_260.0 + 25_000.0

    def test_a_book_of_zero_prices_has_shares_that_are_null_with_a_reason(self):
        """No gross, no share of it: null, not 0 and not NaN."""
        summary = exposure_summary([_line("DEAD", 100, 0.0)], "USD")
        assert summary.gross == 0.0
        assert summary.currencies[0].share_of_gross is None
        assert "no gross" in summary.share_of_gross_null_reason


class TestTheMultiplierScalesExactly:
    def test_three_es_contracts_are_three_times_fifty_points(self):
        """3 x 5,000.25 x 50 is 750,037.50 -- exactly, in local and base."""
        (value,) = position_values([PLANTED[0]], "USD")
        assert value.local_notional == 750_037.5
        assert value.base_notional == 750_037.5

    @pytest.mark.parametrize("seed", range(5))
    def test_contracts_times_multiplier_is_the_same_bits_as_the_units(self, seed):
        """(q, p, m) values to the same bits as (q x m, p, 1) whenever q x m
        is exact. Multiplying left to right does not: (q x p) x m and
        (q x m) x p round differently on most many-digit prices, and this
        test fails for that arithmetic."""
        rng = random.Random(seed)
        naive_differs = 0
        for _ in range(200):
            q = rng.randint(1, 10_000) * rng.choice([-1, 1])
            m = rng.choice([50, 20, 100, 1_000, 5_000, 37])
            p = rng.uniform(0.01, 10_000.0)
            rate = rng.uniform(0.001, 200.0)
            (scaled,) = position_values(
                [_line("F", q, p, "EUR", m)], "USD", {"EUR": rate}
            )
            (units,) = position_values(
                [_line("F", q * m, p, "EUR", 1.0)], "USD", {"EUR": rate}
            )
            assert scaled.local_notional.hex() == units.local_notional.hex()
            assert scaled.base_notional.hex() == units.base_notional.hex()
            naive_differs += (q * p) * m != (q * m) * p
        assert naive_differs > 0, "no case where the naive order matters"

    @pytest.mark.parametrize("seed", range(3))
    def test_every_notional_is_the_correctly_rounded_exact_product(self, seed):
        """Against Fraction arithmetic: one rounding of the exact product."""
        book = _random_book(seed)
        for line, value in zip(book, position_values(book, "USD", RANDOM_RATES)):
            rate = 1.0 if line["currency"] == "USD" else RANDOM_RATES[line["currency"]]
            assert value.base_notional == _fraction_notional(line, rate)
            assert value.local_notional == _fraction_notional(line, 1.0)

    def test_doubling_the_multiplier_doubles_every_notional(self):
        book = _random_book(11)
        doubled = [{**line, "multiplier": 2 * line["multiplier"]} for line in book]
        one = exposure_summary(book, "USD", RANDOM_RATES)
        two = exposure_summary(doubled, "USD", RANDOM_RATES)
        assert (two.gross, two.net) == (2 * one.gross, 2 * one.net)
        assert (two.long, two.short) == (2 * one.long, 2 * one.short)


class TestRelabellingACurrencyChangesOnlyTheLabel:
    def test_euros_relabelled_as_swiss_francs_at_the_same_rate(self):
        """Every number is the same bits; only the code differs."""
        relabelled = [
            {**line, "currency": "CHF"} if line["currency"] == "EUR" else line
            for line in PLANTED
        ]
        rates = {("CHF" if k == "EUR" else k): v for k, v in PLANTED_RATES.items()}
        before = exposure_summary(PLANTED, "USD", PLANTED_RATES)
        after = exposure_summary(relabelled, "USD", rates)
        for name in ("gross", "net", "long", "short"):
            assert getattr(after, name).hex() == getattr(before, name).hex()
        old = {c.currency: c for c in before.currencies}
        new = {c.currency: c for c in after.currencies}
        assert set(new) == set(old) - {"EUR"} | {"CHF"}
        for name in ("fx_rate", "local_net", "net", "gross", "long", "short"):
            assert getattr(new["CHF"], name) == getattr(old["EUR"], name)
        assert new["CHF"].share_of_gross == old["EUR"].share_of_gross
        values_before = position_values(PLANTED, "USD", PLANTED_RATES)
        values_after = position_values(relabelled, "USD", rates)
        assert [v.base_notional for v in values_after] == [
            v.base_notional for v in values_before
        ]

    def test_relabelling_the_base_currency_changes_nothing_else(self):
        """The same book reported in CHF with every rate unchanged."""
        book = [
            {**line, "currency": "CHF"}
            for line in _random_book(4)
            if line["currency"] == "USD"
        ]
        in_usd = exposure_summary([{**line, "currency": "USD"} for line in book], "USD")
        in_chf = exposure_summary(book, "CHF")
        assert _floats(in_usd)[:4] == _floats(in_chf)[:4]


class TestOrderChangesNoTotal:
    def test_cancelling_lines_sum_exactly_in_any_order(self):
        """+1e16, +1, -1e16, +1: a running sum gives 0, 1 or 2 by order;
        the exact sum is 2, and so is every total here."""
        book = [
            _line("A", 1e16, 1.0),
            _line("B", 1, 1.0),
            _line("C", -1e16, 1.0),
            _line("D", 1, 1.0),
        ]
        running = {
            functools.reduce(operator.add, (line["quantity"] for line in order))
            for order in itertools.permutations(book)
        }
        assert running == {0.0, 1.0, 2.0}
        for order in itertools.permutations(book):
            summary = exposure_summary(list(order), "USD")
            assert summary.net == 2.0
            assert summary.long == 1e16 + 2.0
            assert summary.short == 1e16

    @pytest.mark.parametrize("seed", range(4))
    def test_shuffling_the_positions_changes_no_total_to_the_bit(self, seed):
        book = _random_book(seed, n=500)
        reference = _floats(exposure_summary(book, "USD", RANDOM_RATES))
        rng = random.Random(100 + seed)
        for _ in range(5):
            shuffled = book[:]
            rng.shuffle(shuffled)
            assert _floats(exposure_summary(shuffled, "USD", RANDOM_RATES)) == reference

    def test_the_per_line_values_follow_the_input_order(self):
        book = _random_book(9, n=20)
        values = position_values(book, "USD", RANDOM_RATES)
        assert [v.instrument for v in values] == [line["instrument"] for line in book]


class TestDeterminism:
    def test_the_same_input_gives_the_same_bits(self):
        book = _random_book(5, n=1_000)
        first = exposure_summary(book, "USD", RANDOM_RATES)
        second = exposure_summary(book, "USD", RANDOM_RATES)
        assert _floats(first) == _floats(second)
        assert first == second
        assert position_values(book, "USD", RANDOM_RATES) == position_values(
            book, "USD", RANDOM_RATES
        )


class TestUnusedRates:
    def test_a_rate_for_a_currency_not_held_is_accepted_and_listed(self):
        """A rate table may cover more than these holdings; the extra rates
        change nothing and are named, so a missing line is visible."""
        with_extra = {**PLANTED_RATES, "CHF": 1.1, "SEK": 0.095}
        summary = exposure_summary(PLANTED, "USD", with_extra)
        assert summary.unused_fx_rates == ("CHF", "SEK")
        assert _floats(summary) == _floats(
            exposure_summary(PLANTED, "USD", PLANTED_RATES)
        )


def _refusal(call, *args, **kwargs) -> str:
    with pytest.raises(ValidationError) as caught:
        call(*args, **kwargs)
    return str(caught.value)


class TestRefusals:
    def test_an_unknown_currency_names_the_nearest_code(self):
        message = _refusal(position_values, [_line("A", 1, 1.0, "EUE")], "USD")
        assert "positions[0].currency='EUE'" in message
        assert "Did you mean 'EUR'?" in message

    def test_an_unknown_base_currency_is_named(self):
        message = _refusal(position_values, [_line("A", 1, 1.0)], "USX")
        assert "base_currency='USX'" in message and "'USD'" in message

    def test_an_unknown_rate_key_is_named(self):
        message = _refusal(position_values, [_line("A", 1, 1.0)], "USD", {"EUE": 1.1})
        assert "fx_rates key='EUE'" in message and "'EUR'" in message

    def test_a_missing_rate_names_the_currency_and_the_lines(self):
        book = [
            _line("A", 1, 1.0, "EUR"),
            _line("B", 1, 1.0),
            _line("C", 2, 1.0, "EUR"),
        ]
        message = _refusal(position_values, book, "USD", {"JPY": 0.0067})
        assert "fx_rates has no rate for EUR" in message
        assert "positions[0] 'A'" in message and "positions[2] 'C'" in message
        assert "{'EUR': <USD per 1 EUR>}" in message

    @pytest.mark.parametrize(
        "field, value",
        [
            ("quantity", math.nan),
            ("quantity", math.inf),
            ("price", math.nan),
            ("price", -math.inf),
            ("multiplier", math.nan),
            ("multiplier", math.inf),
        ],
    )
    def test_a_non_finite_number_is_refused_by_name(self, field, value):
        line = {**_line("A", 1, 1.0), field: value}
        message = _refusal(position_values, [line], "USD")
        assert f"positions[0].{field}={value!r} is not finite" in message

    @pytest.mark.parametrize("rate", [math.nan, math.inf])
    def test_a_non_finite_rate_is_refused_by_name(self, rate):
        message = _refusal(
            position_values, [_line("A", 1, 1.0, "EUR")], "USD", {"EUR": rate}
        )
        assert f"fx_rates['EUR']={rate!r} is not finite" in message

    @pytest.mark.parametrize("multiplier", [0.0, -50.0])
    def test_a_zero_or_negative_multiplier_is_refused_with_its_fix(self, multiplier):
        message = _refusal(
            position_values, [_line("A", 1, 1.0, multiplier=multiplier)], "USD"
        )
        assert f"positions[0].multiplier={multiplier!r} must be greater than 0" in (
            message
        )
        assert "A short is a negative quantity" in message

    @pytest.mark.parametrize("rate", [0.0, -1.1])
    def test_a_rate_of_zero_or_less_is_refused(self, rate):
        message = _refusal(
            position_values, [_line("A", 1, 1.0, "EUR")], "USD", {"EUR": rate}
        )
        assert f"fx_rates['EUR']={rate!r} must be greater than 0" in message

    @pytest.mark.parametrize("rate", [1.01, 0.999999999, 2.0])
    def test_a_base_rate_other_than_one_is_refused(self, rate):
        message = _refusal(position_values, [_line("A", 1, 1.0)], "USD", {"USD": rate})
        assert f"fx_rates['USD']={rate!r} is the base currency's own rate" in message
        assert "exactly 1" in message

    def test_a_base_rate_of_exactly_one_is_accepted(self):
        (value,) = position_values([_line("A", 3, 7.0)], "USD", {"USD": 1})
        assert value.base_notional == 21.0

    def test_a_zero_quantity_is_refused(self):
        message = _refusal(position_values, [_line("A", 0, 1.0)], "USD")
        assert "positions[0].quantity=0 is zero" in message
        assert "Drop the line" in message

    def test_an_instrument_twice_in_one_currency_is_refused(self):
        """Netting is the caller's decision: two lines of one instrument in
        one currency are refused, whatever their sides, and 'aapl' is
        'AAPL'."""
        book = [
            _line("AAPL", 100, 190.0),
            _line("MSFT", 5, 400.0),
            _line("aapl", -40, 190.0),
        ]
        message = _refusal(position_values, book, "USD")
        assert "positions repeats 'AAPL' in USD at positions[0] and positions[2]" in (
            message
        )
        assert "netting is yours to decide" in message

    def test_one_instrument_in_two_currencies_is_two_lines(self):
        """A dual listing -- one identifier quoted in two currencies -- is
        two holdings, not a repeat."""
        book = [_line("SHEL", 100, 27.5, "GBP"), _line("SHEL", 50, 32.0, "EUR")]
        values = position_values(book, "USD", {"GBP": 1.25, "EUR": 1.125})
        assert [v.base_notional for v in values] == [3_437.5, 1_800.0]

    def test_more_lines_than_the_maximum_are_refused(self):
        book = [_line(f"I{i}", 1, 1.0) for i in range(MAX_POSITIONS + 1)]
        message = _refusal(position_values, book, "USD")
        assert "positions has 5,001 lines, above the 5,000" in message

    def test_the_maximum_itself_is_accepted(self):
        book = [_line(f"I{i}", 1, 1.0) for i in range(MAX_POSITIONS)]
        assert exposure_summary(book, "USD").gross == float(MAX_POSITIONS)

    def test_no_positions_is_refused(self):
        assert "positions is empty" in _refusal(position_values, [], "USD")

    def test_a_missing_field_is_named(self):
        message = _refusal(
            position_values,
            [{"instrument": "A", "price": 1.0, "currency": "USD"}],
            "USD",
        )
        assert "positions[0] has no quantity" in message

    @pytest.mark.parametrize(
        "field, value, words",
        [
            ("quantity", 2e18, "beyond 1e+18"),
            ("price", -2e15, "beyond 1e+15"),
            ("multiplier", 2e9, "above 1e+09"),
        ],
    )
    def test_a_magnitude_past_the_bound_is_refused(self, field, value, words):
        """The bounds keep every product and sum inside a float."""
        line = {**_line("A", 1, 1.0), field: value}
        message = _refusal(position_values, [line], "USD")
        assert f"positions[0].{field}={value!r} is {words}" in message

    @pytest.mark.parametrize("rate", [1e-13, 1e13])
    def test_a_rate_far_from_one_is_refused_as_a_unit_error(self, rate):
        message = _refusal(
            position_values, [_line("A", 1, 1.0, "EUR")], "USD", {"EUR": rate}
        )
        assert f"fx_rates['EUR']={rate!r} is outside [1e-12, 1e+12]" in message

    def test_a_negative_borrow_fee_is_refused(self):
        from standard_quant_tools.finance.positions import check_borrow_fee_bps

        assert "is negative" in _refusal(check_borrow_fee_bps, -5.0)
        assert "not finite" in _refusal(check_borrow_fee_bps, math.inf)
        assert "above 100,000 bps" in _refusal(check_borrow_fee_bps, 100_001.0)
        assert check_borrow_fee_bps(None) is None


class TestCurrencyCodes:
    @pytest.mark.parametrize(
        "written, code", [("usd", "USD"), (" eur ", "EUR"), ("Jpy", "JPY")]
    )
    def test_a_code_is_upper_cased(self, written, code):
        assert currency_code(written) == code

    @pytest.mark.parametrize(
        "written, suggested",
        [
            ("EUE", "EUR"),
            ("UDS", "USD"),
            ("USS", "USD"),
            ("GPB", "GBP"),
            ("YEN", "JPY"),
            ("dollar", "USD"),
            ("CHFF", "CHF"),
        ],
    )
    def test_an_unknown_code_suggests_the_nearest(self, written, suggested):
        message = _refusal(currency_code, written)
        assert f"Did you mean '{suggested}'?" in message
        assert nearest_currency(written) == suggested

    @pytest.mark.parametrize(
        "written, code",
        [("GBp", "GBP"), ("GBX", "GBP"), ("ZAc", "ZAR"), ("ILA", "ILS")],
    )
    def test_a_hundredth_of_a_currency_is_refused_with_the_division(
        self, written, code
    ):
        """'GBp' upper-cased is 'GBP': a London price in pence booked as
        pounds is a hundred times its value. It is refused, not upper-cased."""
        message = _refusal(currency_code, written)
        assert f"a hundredth of {code}" in message
        assert "by 100" in message

    @pytest.mark.parametrize(
        "written, successor",
        [
            ("HRK", "EUR"),
            ("BGN", "EUR"),
            ("ANG", "XCG"),
            ("ZWL", "ZWG"),
            ("VEF", "VES"),
            ("DEM", "EUR"),
        ],
    )
    def test_a_withdrawn_code_names_its_successor(self, written, successor):
        assert f"replaced by '{successor}'" in _refusal(currency_code, written)

    @pytest.mark.parametrize("written", ["XXX", "XTS", "BTC", "RMB", "CNH", "", 840])
    def test_codes_that_name_no_currency_a_price_is_in(self, written):
        _refusal(currency_code, written)

    def test_the_table_is_well_formed(self):
        for code in ACTIVE_CURRENCIES | set(WITHDRAWN_CURRENCIES):
            assert len(code) == 3 and code.isascii() and code.isupper(), code
        assert not ACTIVE_CURRENCIES & set(WITHDRAWN_CURRENCIES)
        assert set(WITHDRAWN_CURRENCIES.values()) <= ACTIVE_CURRENCIES
        assert {"USD", "EUR", "JPY", "GBP", "CNY", "XCG", "ZWG", "CLF"} <= (
            ACTIVE_CURRENCIES
        )
        assert not {"XXX", "XTS"} & ACTIVE_CURRENCIES

    def test_every_active_code_is_its_own_nearest(self):
        for code in ACTIVE_CURRENCIES:
            assert nearest_currency(code) == code
