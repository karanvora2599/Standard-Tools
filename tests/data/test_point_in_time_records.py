"""
Point-in-time records from a provider: one row per filing, stamped with
the date it was filed.

The Polygon tests stub `_polygon_get`, the module's single network seam,
and plant a filing that was amended: two results for one period with two
filing dates. What is asserted is that both survive as rows -- the
restatement is a later row, never an overwrite -- that the pages are
walked to the end, that a filing without a filing date is left out and
counted rather than dated to its period end, and that the contract the
provider declares says `unknown` for revisions, because the claim that
restatements arrive as rows is the measurement `observed_revisions` makes
on a pulled history and not a reading of the documentation.
"""

import pandas as pd
import pytest

from standard_quant_tools.data import polygon_provider
from standard_quant_tools.data.base import DataProvider
from standard_quant_tools.data.polygon_provider import (
    PolygonProvider,
    _parse_financials_records,
)
from standard_quant_tools.data.yfinance_provider import YFinanceProvider
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.point_in_time import (
    observed_revisions,
    validate_pit_frame,
)

REVENUES = "income_statement.revenues"
EPS = "income_statement.diluted_earnings_per_share"


def _filing(period_end, filed, *, revenues, eps, fiscal_year, fiscal_period):
    return {
        "start_date": str(pd.Timestamp(period_end) - pd.Timedelta(days=90))[:10],
        "end_date": period_end,
        "filing_date": filed,
        "fiscal_year": str(fiscal_year),
        "fiscal_period": fiscal_period,
        "timeframe": "quarterly",
        "financials": {
            "income_statement": {
                "revenues": {"value": revenues, "unit": "USD"},
                "diluted_earnings_per_share": {"value": eps, "unit": "USD / shares"},
            },
            "balance_sheet": {"assets": {"value": 10.0 * revenues}},
        },
    }


PAGE_ONE = {
    "results": [
        _filing(
            "2023-03-31",
            "2023-04-28",
            revenues=100.0,
            eps=1.00,
            fiscal_year=2023,
            fiscal_period="Q1",
        ),
        _filing(
            "2023-06-30",
            "2023-07-29",
            revenues=120.0,
            eps=1.20,
            fiscal_year=2023,
            fiscal_period="Q2",
        ),
    ],
    "next_url": "https://api.polygon.io/vX/reference/financials?cursor=abc&apiKey=SECRET",
}
PAGE_TWO = {
    "results": [
        # The Q2 filing, amended two weeks later: a second row, not an edit.
        _filing(
            "2023-06-30",
            "2023-08-14",
            revenues=120.0,
            eps=1.05,
            fiscal_year=2023,
            fiscal_period="Q2",
        ),
        # A filing with no filing date cannot be placed in time.
        {
            **_filing(
                "2023-09-30",
                "2023-10-27",
                revenues=130.0,
                eps=1.30,
                fiscal_year=2023,
                fiscal_period="Q3",
            ),
            "filing_date": None,
        },
    ],
}


class TestTheBase:
    def test_a_provider_without_records_refuses_by_name(self):
        provider = YFinanceProvider()
        with pytest.raises(NotImplementedError, match="point-in-time"):
            provider.get_point_in_time_records(
                ["AAPL"], "fundamentals", [REVENUES], "2023-01-01", "2023-12-31"
            )
        assert not provider.get_temporal_contract("fundamentals").pit_safe
        assert hasattr(DataProvider, "get_point_in_time_records")


class TestPolygon:
    @pytest.fixture
    def calls(self, monkeypatch):
        seen = []

        def fake_get(path, params, api_key):
            seen.append((path, dict(params)))
            assert "apiKey" not in params, "the key is appended by the seam itself"
            if params.get("cursor") == "abc":
                return PAGE_TWO
            return PAGE_ONE

        monkeypatch.setattr(polygon_provider, "_polygon_get", fake_get)
        return seen

    def test_filings_become_versioned_records_across_pages(self, calls):
        provider = PolygonProvider(api_key="k")
        records = provider.get_point_in_time_records(
            ["aapl"], "fundamentals", [EPS, REVENUES], "2022-01-01", "2023-12-31"
        )
        assert list(calls[0]) == [
            "/vX/reference/financials",
            {
                "ticker": "AAPL",
                "timeframe": "quarterly",
                "period_of_report_date.gte": "2022-01-01",
                "period_of_report_date.lte": "2023-12-31",
                "limit": 100,
                "sort": "period_of_report_date",
                "order": "asc",
            },
        ]
        assert calls[1][1]["cursor"] == "abc" and "apiKey" not in calls[1][1]
        assert len(records) == 3
        assert records.attrs["n_dropped_without_available_time"] == 1
        validate_pit_frame(records)
        assert (records["entity"] == "aapl").all()
        q2 = records[records["event_time"] == pd.Timestamp("2023-06-30")]
        assert list(q2["available_time"]) == [
            pd.Timestamp("2023-07-29"),
            pd.Timestamp("2023-08-14"),
        ]
        assert list(q2[EPS]) == [1.20, 1.05]
        assert records["fiscal_year"].tolist() == [2023, 2023, 2023]
        assert records["fiscal_period"].tolist() == ["Q1", "Q2", "Q2"]
        # The measurement: this pull shows a restatement arriving as a row.
        assert observed_revisions(records) == {
            "n_facts": 2,
            "n_restated": 1,
            "max_versions": 2,
        }

    def test_the_contract_says_both_timestamps_and_unknown_revisions(self):
        contract = PolygonProvider(api_key="k").get_temporal_contract("fundamentals")
        assert contract.pit_safe and contract.has_event_time
        assert contract.revisions == "unknown" and not contract.reproduces_history
        assert any("observed_revisions" in note for note in contract.notes)
        assert PolygonProvider(api_key="k").get_temporal_contract("bars").pit_safe

    def test_only_fundamentals_are_served_and_a_field_needs_a_statement(self, calls):
        provider = PolygonProvider(api_key="k")
        with pytest.raises(NotImplementedError, match="fundamentals"):
            provider.get_point_in_time_records(
                ["AAPL"], "estimates", [EPS], "2023-01-01", "2023-12-31"
            )
        with pytest.raises(ValidationError, match="statement"):
            _parse_financials_records(PAGE_ONE["results"], "AAPL", ["revenues"])
        with pytest.raises(ValidationError, match="at least one field"):
            provider.get_point_in_time_records(
                ["AAPL"], "fundamentals", [], "2023-01-01", "2023-12-31"
            )

    def test_a_missing_field_is_nan_not_an_error(self):
        frame = _parse_financials_records(
            PAGE_ONE["results"], "AAPL", ["balance_sheet.liabilities", REVENUES]
        )
        assert frame["balance_sheet.liabilities"].isna().all()
        assert frame[REVENUES].tolist() == [100.0, 120.0]
