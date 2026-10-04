"""
A universe fetch asks the provider its `source` names.

`fetch_ohlcv_panel` and `fetch_returns_panel` took a `source` and ignored
it: both went through the portfolio helpers, which always built the default
provider, so `source='databento'` fetched Yahoo's bars and published them
under the name the caller asked for. They now build the provider the way
`fetch_ohlcv` does and hand it to the helpers; with no `source` the default
is built exactly as before. A vendor failure names the source that failed.
See the CHANGELOG entry of 2026-10-04.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.runtimes import resolve as resolve_runtime
from standard_quant_tools.data import _cache as cache_module
from standard_quant_tools.data.base import DataProvider
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError, VendorUnavailableError
from standard_quant_tools.portfolio import portfolio

_INDEX = pd.bdate_range("2023-01-02", periods=30)
ARGS = {
    "tickers": ["AAPL", "MSFT"],
    "start_date": "2023-01-02",
    "end_date": "2023-02-10",
    "name": "bars",
}


class _Provider(DataProvider):
    """Serves bars whose Close starts at `level`, so a panel says which
    provider served it; records each symbol asked for."""

    def __init__(self, level: float, fail: Optional[Exception] = None) -> None:
        self.level = level
        self.fail = fail
        self.asked: List[str] = []

    def get_ohlcv(self, symbol, start, end, interval="1d"):
        self.asked.append(symbol)
        if self.fail is not None:
            raise self.fail
        close = self.level + np.arange(len(_INDEX), dtype=float)
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

    async def get_ohlcv_async(self, symbol, start, end, interval="1d"):
        return self.get_ohlcv(symbol, start, end, interval)

    def get_ticker_info(self, symbol):
        raise NotImplementedError

    def get_financial_ratios(self, symbol):
        raise NotImplementedError

    def get_metadata(self, symbol, interval="1d"):
        raise NotImplementedError


class _Factory:
    """`DataFactory.get_provider`, recording what each call named."""

    def __init__(self, providers: Dict[Optional[str], _Provider]) -> None:
        self.providers = providers
        self.calls: List[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        source = args[0] if args else kwargs.get("source")
        return self.providers[source]


@pytest.fixture(autouse=True)
def _runs(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr("time.sleep", lambda s: None)


@pytest.fixture
def factory():
    made = _Factory(
        {
            None: _Provider(100.0),
            "databento": _Provider(500.0),
            "polygon": _Provider(900.0),
        }
    )
    with patch.object(DataFactory, "get_provider", made):
        yield made


def _dispatch(tool: str, **arguments):
    return resolve_runtime("data").dispatch(tool, {**ARGS, **arguments})


def _first_close(result) -> float:
    from standard_quant_tools.agent.runtimes.handoff import resolve as resolve_ref

    return float(resolve_ref(result["ref"])["Close"].iloc[0])


class TestTheRequestedSourceIsAsked:
    @pytest.mark.parametrize("source,level", [("databento", 500.0), ("polygon", 900.0)])
    def test_the_bars_come_from_the_named_provider(self, factory, source, level):
        result = _dispatch("fetch_ohlcv_panel", run_id=f"t_{source}", source=source)
        assert factory.calls == [((source,), {})]
        assert factory.providers[source].asked == ["AAPL", "MSFT"]
        assert factory.providers[None].asked == []
        assert _first_close(result) == level

    def test_the_returns_panel_too(self, factory):
        """fetch_returns_panel ignored `source` the same way."""
        _dispatch("fetch_returns_panel", run_id="t_rets", name="rets", source="polygon")
        assert factory.calls == [(("polygon",), {})]
        assert factory.providers["polygon"].asked == ["AAPL", "MSFT"]
        assert factory.providers[None].asked == []

    def test_null_no_source_builds_the_default_as_before(self, factory):
        """The default is asked with no argument, as the helper asked it."""
        result = _dispatch("fetch_ohlcv_panel", run_id="t_default")
        assert factory.calls == [((), {})]
        assert _first_close(result) == 100.0

    def test_a_source_no_provider_serves_is_refused(self, factory):
        """Refused by the input model, as for fetch_ohlcv, before any
        provider is built."""
        with pytest.raises(ValueError) as caught:
            _dispatch("fetch_ohlcv_panel", run_id="t_bad", source="alpaca")
        assert "source" in str(caught.value) and "'databento'" in str(caught.value)
        assert factory.calls == []


class TestAVendorFailureNamesItsSource:
    def _outage(self) -> VendorUnavailableError:
        return VendorUnavailableError(
            "Databento failed on its side: HTTP 504 twice.",
            status=504,
            dataset="EQUS.SUMMARY",
        )

    def test_the_named_source(self, factory):
        factory.providers["databento"].fail = self._outage()
        with pytest.raises(VendorUnavailableError) as caught:
            _dispatch("fetch_ohlcv_panel", run_id="t_out", source="databento")
        text = str(caught.value)
        assert "fetching 2 ticker(s) from source='databento' failed because" in text
        assert "the data vendor failed on its side (EQUS.SUMMARY, HTTP 504)" in text

    def test_the_default_source(self, factory):
        factory.providers[None].fail = self._outage()
        with pytest.raises(VendorUnavailableError) as caught:
            _dispatch("fetch_ohlcv_panel", run_id="t_out_default")
        assert "from the default source ('yfinance') failed" in str(caught.value)

    def test_null_a_symbol_failure_keeps_its_advice(self, factory):
        factory.providers["polygon"].fail = RuntimeError("upstream refused")
        with pytest.raises(ValidationError, match="drop the symbol"):
            _dispatch("fetch_ohlcv_panel", run_id="t_sym", source="polygon")


class TestTheHelpersKeepTheirDefault:
    def test_null_without_a_provider_the_helpers_build_the_default(self, factory):
        """Every other caller of the portfolio helpers passes no provider and
        gets the default, built with no argument as before."""
        frames = portfolio.fetch_ohlcv_panel_sync(["AAPL"], "2023-01-02", "2023-02-10")
        returns = portfolio.fetch_returns_sync(["AAPL"], "2023-01-02", "2023-02-10")
        assert factory.calls == [((), {}), ((), {})]
        assert float(frames["AAPL"]["Close"].iloc[0]) == 100.0
        assert len(returns) == len(_INDEX) - 1
