"""
`get_stock_fundamentals` carries what `fetch_financial_ratios` carries.

The two tools answer from the same `get_financial_ratios`, and only one of
them passed on the provider's honesty fields: `definition_notes` (Polygon's
"debt" to equity is liabilities over equity; a trailing ratio refused for
want of a full year), the implausible-value check, and any warning. The
fundamentals tool dropped all three, took no `source`, and did not say the
ratios are today's (see the CHANGELOG entry of 2026-10-22).
"""

from typing import List, Optional

import pytest

from standard_quant_tools.agent.models import FundamentalsInput
from standard_quant_tools.agent.runtimes.research.tools import get_stock_fundamentals
from standard_quant_tools.data.base import FinancialRatios, TickerInfo
from standard_quant_tools.data.factory import DataFactory

NOTES = {
    "debt_to_equity": "Computed as total LIABILITIES / equity, not total DEBT.",
    "trailing_pe": "Not computed: only a quarter's figures were available.",
}


class _Provider:
    def get_ticker_info(self, symbol):
        return TickerInfo(symbol=symbol, name="Planted Co")

    def get_financial_ratios(self, symbol):
        return FinancialRatios(
            trailing_pe=None,
            price_to_book=2.5,
            debt_to_equity=0.5,
            # 4.2 as a decimal fraction is a 420% yield: an unconverted
            # percentage, which the check names.
            dividend_yield=4.2,
            market_cap=1_000_000_000,
            definition_notes=dict(NOTES),
        )


@pytest.fixture
def asked(monkeypatch) -> List[Optional[str]]:
    sources: List[Optional[str]] = []

    def get_provider(source=None, *a, **k):
        sources.append(source)
        return _Provider()

    monkeypatch.setattr(DataFactory, "get_provider", get_provider)
    return sources


class TestTheHonestyFieldsArrive:
    def test_definition_notes_reach_the_result(self, asked):
        out = get_stock_fundamentals(FundamentalsInput(symbol="AAA"))
        assert out.definition_notes == NOTES

    def test_an_implausible_value_is_named(self, asked):
        out = get_stock_fundamentals(FundamentalsInput(symbol="AAA"))
        (flag,) = out.implausible
        assert flag.startswith("dividend_yield=4.2 is implausible")

    def test_every_answer_says_it_is_todays(self, asked):
        out = get_stock_fundamentals(FundamentalsInput(symbol="AAA"))
        assert any("TODAY's snapshot" in w for w in out.warnings)
        assert any("not point-in-time" in w for w in out.warnings)

    def test_the_source_reaches_the_provider_and_is_echoed(self, asked):
        out = get_stock_fundamentals(FundamentalsInput(symbol="AAA", source="polygon"))
        assert asked == ["polygon"]
        assert out.source == "polygon"

    def test_no_source_asks_for_the_default_and_names_it(self, asked):
        out = get_stock_fundamentals(FundamentalsInput(symbol="AAA"))
        assert asked == [None]
        assert out.source == "yfinance"
