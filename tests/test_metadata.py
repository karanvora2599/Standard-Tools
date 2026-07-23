"""Tests for data/metadata.py and YFinanceProvider.get_metadata() — honest dataset provenance reporting."""

from standard_quant_tools.data.metadata import DataSetMetadata
from standard_quant_tools.data.yfinance_provider import YFinanceProvider


class TestDataSetMetadata:
    def test_retrieved_at_defaults_to_now(self):
        meta = DataSetMetadata(
            provider="test", adjusted=True, survivorship_free=False,
            point_in_time=False, frequency="1d", timezone="UTC",
        )
        assert meta.retrieved_at  # non-empty ISO string
        assert "T" in meta.retrieved_at


class TestYFinanceProviderMetadata:
    def test_reports_honest_guarantees(self):
        provider = YFinanceProvider()
        meta = provider.get_metadata("AAPL")
        assert meta.provider == "yfinance"
        assert meta.adjusted is True
        # yfinance makes neither guarantee — must be reported as False, not True.
        assert meta.survivorship_free is False
        assert meta.point_in_time is False

    def test_frequency_reflects_requested_interval(self):
        provider = YFinanceProvider()
        meta = provider.get_metadata("AAPL", interval="1wk")
        assert meta.frequency == "1wk"

    def test_default_frequency_is_daily(self):
        provider = YFinanceProvider()
        meta = provider.get_metadata("AAPL")
        assert meta.frequency == "1d"
