"""
A universe fetch that failed because the vendor did says so.

`fetch_ohlcv_panel` fails the whole batch on its first failure and used to
advise, whatever the failure, "drop the symbol that cannot be fetched and
run it again". For a vendor that failed on its side -- a 504, a dropped
connection -- no symbol is at fault and dropping one changes nothing. Such a
failure is now refused as a `VendorUnavailableError` that says the vendor
failed, names the dataset and status it knows, and says to run it later. A
failure about one symbol keeps the old advice. See the CHANGELOG entry of
2026-10-02.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import requests

from standard_quant_tools.agent.runtimes import resolve as resolve_runtime
from standard_quant_tools.data import _cache as cache_module
from standard_quant_tools.data import yfinance_provider
from standard_quant_tools.data.base import DataProvider
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import (
    APIError,
    ValidationError,
    VendorUnavailableError,
)

_INDEX = pd.bdate_range("2023-01-02", periods=30)
ARGS = {
    "tickers": ["AAPL", "MSFT", "NVDA"],
    "start_date": "2023-01-02",
    "end_date": "2023-02-10",
    "name": "bars",
}


def _bars(symbol: str) -> pd.DataFrame:
    close = 100.0 + np.arange(len(_INDEX), dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=_INDEX,
    )


class _Provider(DataProvider):
    """Serves bars; `fail(symbol)` may return an exception to raise."""

    def __init__(self, fail=None) -> None:
        self.fail = fail

    def get_ohlcv(self, symbol, start, end, interval="1d"):
        failure = self.fail(symbol) if self.fail else None
        if failure is not None:
            raise failure
        return _bars(symbol)

    async def get_ohlcv_async(self, symbol, start, end, interval="1d"):
        return self.get_ohlcv(symbol, start, end, interval)

    def get_ticker_info(self, symbol):
        raise NotImplementedError

    def get_financial_ratios(self, symbol):
        raise NotImplementedError

    def get_metadata(self, symbol, interval="1d"):
        raise NotImplementedError


def _panel(provider, run_id: str):
    with patch.object(DataFactory, "get_provider", lambda *a, **k: provider):
        return resolve_runtime("data").dispatch(
            "fetch_ohlcv_panel", {**ARGS, "run_id": run_id}
        )


@pytest.fixture(autouse=True)
def _runs(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr("time.sleep", lambda s: None)
    with cache_module._session_cache_lock:
        cache_module._session_cache.clear()


class TestAVendorOutageIsNotBlamedOnASymbol:
    def test_it_says_the_vendor_failed_and_names_what_it_knows(self):
        outage = VendorUnavailableError(
            "Databento failed on its side: the request for bars for MSFT "
            "(ohlcv-1d) on EQUS.SUMMARY answered HTTP 504 twice.",
            status=504,
            dataset="EQUS.SUMMARY",
        )
        provider = _Provider(fail=lambda s: outage if s == "MSFT" else None)
        with pytest.raises(VendorUnavailableError) as caught:
            _panel(provider, "t_outage")
        text = str(caught.value)
        assert "the data vendor failed on its side (EQUS.SUMMARY, HTTP 504)" in text
        assert "not because of any symbol" in text and "run it again later" in text
        assert "drop the symbol" not in text
        assert (caught.value.status, caught.value.dataset) == (504, "EQUS.SUMMARY")
        assert caught.value.__cause__ is outage

    def test_a_chained_outage_is_found(self):
        def fail(symbol):
            if symbol != "NVDA":
                return None
            try:
                raise VendorUnavailableError("Yahoo Finance failed on its side")
            except VendorUnavailableError as inner:
                wrapped = APIError("fetch failed")
                wrapped.__cause__ = inner
                return wrapped

        with pytest.raises(VendorUnavailableError, match="not because of any symbol"):
            _panel(_Provider(fail=fail), "t_chained")

    def test_the_default_provider_s_dropped_connection_end_to_end(self, monkeypatch):
        """yfinance, stubbed: every history call loses its connection. The
        provider names it after its retries, and the panel says so."""

        class Unreachable:
            def __init__(self, symbol, *a, **k):
                self.ticker = symbol

            def history(self, **_kw):
                raise requests.exceptions.ConnectionError("Connection aborted.")

        monkeypatch.setattr(yfinance_provider.yf, "Ticker", Unreachable)
        provider = yfinance_provider.YFinanceProvider()
        with pytest.raises(VendorUnavailableError) as caught:
            _panel(provider, "t_yahoo")
        text = str(caught.value)
        assert "the data vendor failed on its side" in text
        assert "Connection aborted" in text and "No data found" not in text


class TestAFailureAboutASymbolKeepsItsAdvice:
    def test_null_a_symbol_the_vendor_refused_is_still_to_be_dropped(self):
        provider = _Provider(
            fail=lambda s: RuntimeError("upstream refused") if s == "MSFT" else None
        )
        with pytest.raises(ValidationError) as caught:
            _panel(provider, "t_symbol")
        assert "drop the symbol that cannot be fetched" in str(caught.value)
        assert not isinstance(caught.value, VendorUnavailableError)

    def test_null_a_healthy_universe_is_published(self):
        result = _panel(_Provider(), "t_ok")
        assert sorted(result["entities"]) == ["AAPL", "MSFT", "NVDA"]
