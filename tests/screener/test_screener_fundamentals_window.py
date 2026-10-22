"""
A screen's fundamental filters: which P/E they compare, and when.

Two defects, one door each (see the CHANGELOG entry of 2026-10-22):

- `pe_ratio_max` compared `forward_pe` alone, which Polygon never fills, so
  every Polygon name failed it whatever its earnings. It now compares the
  forward P/E, else the trailing one, and says per ticker which.
- Every provider's ratios are today's; a screen whose technical window
  ended years ago still admitted names on today's figures, silently. It
  now says so.
"""

import datetime

import pytest

from standard_quant_tools.agent.models import ScreenerInput
from standard_quant_tools.agent.runtimes.research.tools import run_screener
from standard_quant_tools.data.base import FinancialRatios
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.screener.screener import (
    fundamental_snapshot_warnings,
    screen_stocks,
)

RATIOS = {
    # Polygon's shape: no forward P/E at all, a trailing one of 12.
    "TRAIL": FinancialRatios(forward_pe=None, trailing_pe=12.0),
    # Both: the forward figure is the one compared.
    "BOTH": FinancialRatios(forward_pe=20.0, trailing_pe=10.0),
    "NONE": FinancialRatios(forward_pe=None, trailing_pe=None),
}


class _Provider:
    def get_financial_ratios(self, symbol):
        return RATIOS[symbol]


@pytest.fixture(autouse=True)
def provider(monkeypatch):
    stub = _Provider()
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: stub)
    return stub


class TestPeComparesWhatTheProviderHas:
    def test_a_trailing_only_name_passes_on_its_trailing_pe(self):
        """Planted: trailing P/E 12 under a bound of 15 passes, where it
        failed for having no forward figure."""
        out = screen_stocks(["TRAIL"], {"pe_ratio_max": 15})
        assert list(out.index) == ["TRAIL"]
        assert out.loc["TRAIL", "pe_basis"] == "trailing"
        assert out.loc["TRAIL", "trailing_pe"] == 12.0

    def test_forward_pe_is_compared_when_there_is_one(self):
        out = screen_stocks(["BOTH"], {"pe_ratio_max": 15})
        assert out.index.empty
        assert out.attrs["failed_filters"] == {"BOTH": "pe_ratio_max"}
        assert out.attrs["pe_basis"] == {"BOTH": "forward"}

    def test_mixing_forward_and_trailing_is_said(self):
        out = screen_stocks(["TRAIL", "BOTH", "NONE"], {"pe_ratio_max": 25})
        assert out.attrs["pe_basis"] == {
            "TRAIL": "trailing",
            "BOTH": "forward",
            "NONE": "none",
        }
        warnings = out.attrs["warnings"]
        assert any(
            "TRAILING P/E for 1 ticker(s)" in w and "TRAIL" in w for w in warnings
        )
        assert any("means a different thing" in w for w in warnings)
        assert any("neither a forward nor a" in w and "NONE" in w for w in warnings)


class TestTodaysFundamentalsOnAPastWindow:
    def test_a_past_end_date_with_a_fundamental_filter_is_warned(self):
        out = screen_stocks(
            ["TRAIL"],
            {"pe_ratio_max": 15},
            start_date="2020-01-02",
            end_date="2020-12-31",
        )
        (warning,) = [w for w in out.attrs["warnings"] if "TODAY'S" in w]
        assert "pe_ratio_max" in warning and "2020-12-31" in warning

    def test_todays_window_is_not_warned(self):
        out = screen_stocks(["TRAIL"], {"pe_ratio_max": 15})
        assert not any("TODAY'S" in w for w in out.attrs["warnings"])

    def test_the_rule_by_dates(self):
        today = datetime.date(2026, 10, 22)
        filters = {"roe_min": 0.1}
        assert fundamental_snapshot_warnings(filters, "2026-10-21", today)
        assert not fundamental_snapshot_warnings(filters, "2026-10-22", today)
        assert not fundamental_snapshot_warnings(filters, None, today)
        assert not fundamental_snapshot_warnings({"rsi_max": 30}, "2020-01-01", today)

    def test_the_tool_carries_the_warning(self):
        out = run_screener(
            ScreenerInput(
                tickers=["TRAIL"],
                filters={"pe_ratio_max": 15},
                start_date="2020-01-02",
                end_date="2020-12-31",
            )
        )
        assert out.tickers_passed == ["TRAIL"]
        assert any("FUNDAMENTALS ARE TODAY'S" in w for w in out.warnings)
        assert out.results[0]["pe_basis"] == "trailing"
