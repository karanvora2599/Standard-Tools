"""
Polygon's trailing P/E and return on equity come from a full year, never
from one quarter.

`get_financial_ratios` asked the financials endpoint for the newest filing
with no period filter. After a 10-Q that filing is one quarter, so the
"trailing" P/E was market cap over a quarter's net income -- 40 where the
trailing-twelve-month figure is 10 -- and return on equity a quarter of its
value, with nothing in the result saying so (see the CHANGELOG entry of
2026-10-22). The flows now come from the trailing-twelve-month record,
else the latest annual one, and a name with neither gets no flow ratio and
a note saying why.

`_polygon_get` -- the provider's one network seam -- answers by the
`timeframe` each request names, the way the endpoint filters.
"""

from typing import Any, Dict, List, Optional

import pytest

from standard_quant_tools.data import polygon_provider
from standard_quant_tools.data.polygon_provider import PolygonProvider

MARKET_CAP = 1_000_000_000.0
QUARTER_NET_INCOME = 25_000_000.0
TTM_NET_INCOME = 100_000_000.0
EQUITY = 400_000_000.0


def _record(
    timeframe: str,
    net_income: float,
    revenues: float,
    *,
    fiscal_period: str,
    balance: bool = True,
) -> Dict[str, Any]:
    financials: Dict[str, Any] = {
        "income_statement": {
            "net_income_loss": {"value": net_income},
            "revenues": {"value": revenues},
        }
    }
    if balance:
        financials["balance_sheet"] = {
            "equity": {"value": EQUITY},
            "liabilities": {"value": 200_000_000.0},
        }
    return {
        "timeframe": timeframe,
        "fiscal_period": fiscal_period,
        "fiscal_year": "2026",
        "financials": financials,
    }


TEN_Q = _record("quarterly", QUARTER_NET_INCOME, 125_000_000.0, fiscal_period="Q2")
TTM = _record("ttm", TTM_NET_INCOME, 500_000_000.0, fiscal_period="TTM")
ANNUAL = _record("annual", 90_000_000.0, 450_000_000.0, fiscal_period="FY")


def _serve(monkeypatch, by_timeframe: Dict[Optional[str], List[Dict[str, Any]]]):
    """A financials endpoint holding the records given per timeframe; a
    request with no timeframe gets the newest filing of any kind (the 10-Q
    here, as after any second-quarter filing)."""
    requests: List[Dict[str, Any]] = []

    def fake_get(path: str, params: Dict[str, Any], api_key: str) -> Dict[str, Any]:
        if path.startswith("/v3/reference/tickers/"):
            return {"results": {"market_cap": MARKET_CAP}}
        requests.append(dict(params))
        return {"results": by_timeframe.get(params.get("timeframe"), [])}

    monkeypatch.setattr(polygon_provider, "_polygon_get", fake_get)
    return requests


class TestAQuarterIsNotATrailingYear:
    def test_the_trailing_pe_is_ten_after_a_10q_not_forty(self, monkeypatch):
        """Planted: market cap 1bn, a 10-Q with 25m net income, a TTM record
        with 100m. The old request read the 10-Q: P/E 40, ROE 6.25%."""
        _serve(monkeypatch, {None: [TEN_Q], "ttm": [TTM], "annual": [ANNUAL]})
        ratios = PolygonProvider(api_key="k").get_financial_ratios("AAA")
        assert ratios.trailing_pe == pytest.approx(10.0)
        assert ratios.return_on_equity == pytest.approx(0.25)
        assert ratios.profit_margins == pytest.approx(0.2)
        assert ratios.price_to_book == pytest.approx(2.5)
        assert "trailing_pe" not in ratios.definition_notes

    def test_the_request_names_the_timeframe(self, monkeypatch):
        requests = _serve(monkeypatch, {None: [TEN_Q], "ttm": [TTM]})
        PolygonProvider(api_key="k").get_financial_ratios("AAA")
        assert requests[0]["timeframe"] == "ttm"
        assert requests[0]["sort"] == "period_of_report_date"

    def test_without_ttm_the_latest_annual_figure_is_used_and_said(self, monkeypatch):
        """No TTM record: the 10-K's year, named as such, with the balance
        sheet still the newest one."""
        ten_q_balance_only = _record(
            "quarterly", QUARTER_NET_INCOME, 1.0, fiscal_period="Q2"
        )
        annual_without_balance = _record(
            "annual", 90_000_000.0, 450_000_000.0, fiscal_period="FY", balance=False
        )
        _serve(
            monkeypatch,
            {None: [ten_q_balance_only], "annual": [annual_without_balance]},
        )
        ratios = PolygonProvider(api_key="k").get_financial_ratios("AAA")
        assert ratios.trailing_pe == pytest.approx(MARKET_CAP / 90_000_000.0)
        assert ratios.return_on_equity == pytest.approx(90_000_000.0 / EQUITY)
        note = ratios.definition_notes["trailing_pe"]
        assert "ANNUAL" in note and "FY 2026" in note

    def test_with_only_quarters_the_flow_ratios_are_refused_with_the_reason(
        self, monkeypatch
    ):
        _serve(monkeypatch, {None: [TEN_Q]})
        ratios = PolygonProvider(api_key="k").get_financial_ratios("AAA")
        for field in ("trailing_pe", "return_on_equity", "profit_margins"):
            assert getattr(ratios, field) is None
            assert "four times too high" in ratios.definition_notes[field]
        # The stock ratios need no period and are still there.
        assert ratios.price_to_book == pytest.approx(2.5)
        assert ratios.debt_to_equity == pytest.approx(0.5)

    def test_a_quarter_served_despite_the_filter_is_still_refused(self, monkeypatch):
        """The record's own label wins over the filter that asked for it."""
        _serve(monkeypatch, {None: [TEN_Q], "ttm": [TEN_Q]})
        ratios = PolygonProvider(api_key="k").get_financial_ratios("AAA")
        assert ratios.trailing_pe is None
        assert "'quarterly'" in ratios.definition_notes["trailing_pe"]
