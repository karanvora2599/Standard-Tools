"""
`Position` and `Holdings`, the input models the finance tools take holdings
through: what they accept, what they refuse and under which argument, and
that the surface fuzzer can build them.

A refusal is pinned by its location -- `positions.2.currency`, `fx_rates`
-- and by the value and the fix in its message, because an agent reads
both: the location says which argument to change and the message how. The
models carry no field named `symbol`, `ticker`, `universe` or `book`: the
first three would mark every tool taking holdings as fetching market data
(open-world to MCP, and outside the determinism layer), the last makes the
fuzzer synthesize an order book. See the CHANGELOG entry of 2026-10-22.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pydantic
import pytest
from pydantic import BaseModel, ConfigDict

from standard_quant_tools.agent.runtimes._positions import Holdings, Position
from standard_quant_tools.finance import exposure_summary
from standard_quant_tools.mcp.catalog import _reads_market_data
from standard_quant_tools.mcp.schemas import dereference, property_names

from ..surface import synth


class _HoldingsTool(Holdings):
    """A tool input taking holdings at its top level, as the finance tools
    will: the three fields first, then the tool's own."""

    model_config = ConfigDict(extra="forbid")

    horizon_days: int = 30


class _NestedHoldingsTool(BaseModel):
    """A tool input taking holdings as one field."""

    model_config = ConfigDict(extra="forbid")

    holdings: Holdings


def _line(instrument: str = "A", **fields: Any) -> Dict[str, Any]:
    return {
        "instrument": instrument,
        "quantity": 10,
        "price": 5.0,
        "currency": "USD",
        **fields,
    }


def _book(*lines: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(lines) or [_line()]


def _refused(model: type, **arguments: Any) -> Dict[str, Any]:
    """The first error of a refused construction: its location and text."""
    with pytest.raises(pydantic.ValidationError) as caught:
        model(**arguments)
    error = caught.value.errors()[0]
    return {"loc": error["loc"], "msg": error["msg"]}


class TestWhatIsAccepted:
    def test_codes_are_upper_cased_and_the_identifier_stripped(self):
        holdings = Holdings(
            positions=[
                _line(" ES ", currency="usd", multiplier=50),
                _line("SAP", quantity=-100, currency="Eur"),
            ],
            base_currency="usd",
            fx_rates={"eur": 1.125},
        )
        assert holdings.base_currency == "USD"
        assert holdings.fx_rates == {"EUR": 1.125}
        assert [p.currency for p in holdings.positions] == ["USD", "EUR"]
        assert holdings.positions[0].instrument == "ES"

    def test_a_book_in_the_base_currency_needs_no_rates(self):
        holdings = Holdings(positions=[_line()], base_currency="USD")
        assert holdings.fx_rates == {}
        assert holdings.exposure_summary().gross == 50.0

    def test_zero_and_negative_prices_are_kept(self):
        holdings = Holdings(
            positions=[
                _line("CL", price=-37.63, multiplier=1000),
                _line("DEAD", price=0.0),
            ],
            base_currency="USD",
        )
        values = holdings.position_values()
        assert values[0].base_notional < 0 and values[0].side == "long"
        assert values[1].base_notional == 0.0

    def test_a_borrow_fee_is_carried_on_any_line(self):
        """Read only while short, by the tools that charge it; a long line
        may carry one (a locate file has a rate for every name)."""
        holdings = Holdings(
            positions=[
                _line("A", quantity=-5, borrow_fee_bps=450.0),
                _line("B", borrow_fee_bps=25.0),
            ],
            base_currency="USD",
        )
        assert [p.borrow_fee_bps for p in holdings.positions] == [450.0, 25.0]

    def test_the_model_values_as_the_library_does(self):
        lines = [
            _line("A", currency="EUR", price=101.37),
            _line("B", quantity=-7, price=13.1),
        ]
        holdings = Holdings(
            positions=lines, base_currency="USD", fx_rates={"EUR": 1.0873}
        )
        assert holdings.exposure_summary() == exposure_summary(
            lines, "USD", {"EUR": 1.0873}
        )

    def test_the_same_input_gives_the_same_bits(self):
        arguments = dict(
            positions=[
                _line(
                    f"I{i}",
                    currency=("EUR" if i % 2 else "USD"),
                    price=1.0 + i / 7,
                    quantity=(-1) ** i * (i + 1),
                )
                for i in range(50)
            ],
            base_currency="USD",
            fx_rates={"EUR": 1.0873},
        )
        first = Holdings(**arguments).exposure_summary()
        second = Holdings(**arguments).exposure_summary()
        assert first == second
        assert (first.gross.hex(), first.net.hex()) == (
            second.gross.hex(),
            second.net.hex(),
        )


class TestEachRefusalNamesItsArgument:
    def test_an_unknown_currency_with_its_suggestion(self):
        error = _refused(
            Holdings,
            positions=[_line(), _line("B"), _line("C", currency="EUE")],
            base_currency="USD",
        )
        assert error["loc"] == ("positions", 2, "currency")
        assert "currency='EUE'" in error["msg"]
        assert "Did you mean 'EUR'?" in error["msg"]

    def test_pence_are_refused_rather_than_read_as_pounds(self):
        error = _refused(Position, **_line(currency="GBp"))
        assert error["loc"] == ("currency",)
        assert "a hundredth of GBP" in error["msg"]

    def test_an_unknown_base_currency(self):
        error = _refused(Holdings, positions=_book(), base_currency="UDS")
        assert error["loc"] == ("base_currency",)
        assert "Did you mean 'USD'?" in error["msg"]

    def test_a_missing_rate(self):
        error = _refused(
            Holdings,
            positions=[_line(), _line("SAP", currency="EUR")],
            base_currency="USD",
            fx_rates={"JPY": 0.0067},
        )
        assert error["loc"] == ("fx_rates",)
        assert "no rate for EUR, held by positions[1] 'SAP'" in error["msg"]

    def test_a_missing_rate_when_no_table_is_given(self):
        error = _refused(
            Holdings, positions=[_line(currency="JPY")], base_currency="USD"
        )
        assert error["loc"] == ("fx_rates",)
        assert "no rate for JPY" in error["msg"]

    @pytest.mark.parametrize(
        "field, value",
        [
            ("quantity", math.nan),
            ("quantity", -math.inf),
            ("price", math.nan),
            ("price", math.inf),
            ("multiplier", math.nan),
            ("multiplier", math.inf),
            ("borrow_fee_bps", math.nan),
        ],
    )
    def test_a_non_finite_number(self, field, value):
        error = _refused(
            Holdings, positions=[_line(**{field: value})], base_currency="USD"
        )
        assert error["loc"] == ("positions", 0, field)
        assert f"{field}={value!r} is not finite" in error["msg"]

    @pytest.mark.parametrize("rate", [math.nan, math.inf])
    def test_a_non_finite_rate(self, rate):
        error = _refused(
            Holdings,
            positions=[_line(currency="EUR")],
            base_currency="USD",
            fx_rates={"EUR": rate},
        )
        assert error["loc"] == ("fx_rates",)
        assert f"fx_rates['EUR']={rate!r} is not finite" in error["msg"]

    @pytest.mark.parametrize("multiplier", [0, -50])
    def test_a_zero_or_negative_multiplier(self, multiplier):
        error = _refused(Position, **_line(multiplier=multiplier))
        assert error["loc"] == ("multiplier",)
        assert f"multiplier={multiplier!r} must be greater than 0" in error["msg"]
        assert "A short is a negative quantity" in error["msg"]

    def test_a_zero_quantity(self):
        error = _refused(Position, **_line(quantity=0))
        assert error["loc"] == ("quantity",)
        assert "a zero quantity is not a position" in error["msg"]

    @pytest.mark.parametrize("rate", [1.0000001, 0.5])
    def test_a_base_rate_other_than_one(self, rate):
        error = _refused(
            Holdings, positions=_book(), base_currency="USD", fx_rates={"USD": rate}
        )
        assert error["loc"] == ("fx_rates",)
        assert "the base currency's own rate, which is exactly 1" in error["msg"]

    def test_a_base_rate_of_one_is_accepted(self):
        holdings = Holdings(
            positions=_book(), base_currency="USD", fx_rates={"usd": 1.0}
        )
        assert holdings.fx_rates == {"USD": 1.0}

    def test_a_currency_named_twice_in_the_rates(self):
        error = _refused(
            Holdings,
            positions=[_line(currency="EUR")],
            base_currency="USD",
            fx_rates={"eur": 1.1, "EUR": 1.1},
        )
        assert error["loc"] == ("fx_rates",)
        assert "names EUR twice" in error["msg"]

    def test_an_instrument_twice_in_one_currency(self):
        error = _refused(
            Holdings,
            positions=[_line("AAPL"), _line("MSFT"), _line("AAPL", quantity=-3)],
            base_currency="USD",
        )
        assert error["loc"] == ("positions",)
        assert "repeats 'AAPL' in USD at positions[0] and positions[2]" in (
            error["msg"]
        )

    def test_more_lines_than_the_maximum(self):
        error = _refused(
            Holdings,
            positions=[_line(f"I{i}") for i in range(5_001)],
            base_currency="USD",
        )
        assert error["loc"] == ("positions",)
        assert "5,001 lines, above the 5,000 one call takes" in error["msg"]

    def test_no_lines(self):
        error = _refused(Holdings, positions=[], base_currency="USD")
        assert error["loc"] == ("positions",)
        assert "positions is empty" in error["msg"]

    def test_an_argument_it_does_not_take(self):
        """`extra="forbid"`, in the mixin and in a tool built on it."""
        assert _refused(Position, **_line(symbol="AAPL"))["loc"] == ("symbol",)
        error = _refused(
            _HoldingsTool, positions=_book(), base_currency="USD", book="x"
        )
        assert error["loc"] == ("book",)


class TestTheSurfaceCanFuzzIt:
    @pytest.mark.parametrize("model", [Holdings, _HoldingsTool, _NestedHoldingsTool])
    def test_the_synthesizer_builds_it(self, model):
        """A multi-currency book, every currency with a rate, so a fuzzed
        finance tool reaches its arithmetic rather than its refusals."""
        arguments, reason = synth.synthesize(model)
        assert reason is None, reason
        built = model(**arguments)
        holdings = getattr(built, "holdings", built)
        assert holdings.base_currency == "USD"
        assert len({p.currency for p in holdings.positions}) == len(synth.CURRENCIES)
        summary = holdings.exposure_summary()
        assert summary.gross > 0 and summary.unused_fx_rates == ()

    def test_no_field_marks_it_as_fetching_or_as_an_order_book(self):
        schema = dereference(_HoldingsTool.model_json_schema())
        names = {name.lower() for name in property_names(schema)}
        assert not _reads_market_data(schema)
        assert not names & {
            "symbol",
            "symbols",
            "ticker",
            "tickers",
            "benchmark",
            "universe",
            "book",
        }
