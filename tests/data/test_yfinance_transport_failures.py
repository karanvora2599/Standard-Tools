"""
A request that never reached Yahoo is not "no data found".

yfinance swallows the error of a request that got no answer -- a dropped
connection, a timeout, Yahoo's maintenance page, a rate limit hidden in its
timezone lookup -- and returns an empty frame, which the provider reported
as "No data found for 'X'. Verify symbol and date range": a network failure
told as a fact about the symbol. The history call now asks yfinance to raise
instead, and the provider sorts what it raises: a transport failure is
retried and then raised as `VendorUnavailableError`; Yahoo's own empty
answer stays `DataNotFoundError`. See the CHANGELOG entry of 2026-10-02.

yfinance is stubbed; nothing here reaches the network.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, List

import numpy as np
import pandas as pd
import pytest
import requests
from yfinance.exceptions import YFPricesMissingError, YFRateLimitError, YFTzMissingError

from standard_quant_tools.data import _cache as cache_module
from standard_quant_tools.data import yfinance_provider
from standard_quant_tools.data.yfinance_provider import YFinanceProvider
from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    NonRetryableAPIError,
    VendorUnavailableError,
)

WINDOW = ("2024-01-02", "2024-01-31")


def _frame(start: str = "2024-01-02", end: str = "2024-02-01") -> pd.DataFrame:
    index = pd.bdate_range(start, end, inclusive="left", tz="America/New_York")
    close = 100.0 + np.arange(len(index))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.full(len(index), 1_000_000),
        },
        index=index,
    )


class Script:
    """What each history call does, in order; the last step repeats."""

    def __init__(self, *steps: Any) -> None:
        self.steps = list(steps)
        self.calls: List[dict] = []

    def ticker(self, symbol: str, *_a, **_k):
        script = self

        class FakeTicker:
            def __init__(self) -> None:
                self.ticker = symbol.upper()

            def history(self, **kwargs):
                script.calls.append(kwargs)
                step = script.steps.pop(0) if len(script.steps) > 1 else script.steps[0]
                if callable(step) and not isinstance(step, pd.DataFrame):
                    step = step(self.ticker)
                if isinstance(step, BaseException):
                    raise step
                return step.copy()

        return FakeTicker()


@pytest.fixture
def sleeps(monkeypatch) -> List[float]:
    taken: List[float] = []
    monkeypatch.setattr("time.sleep", lambda s: taken.append(s))
    return taken


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch, sleeps):
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")
    with cache_module._session_cache_lock:
        cache_module._session_cache.clear()


def _serve(monkeypatch, script: Script, symbol: str = "AAPL") -> pd.DataFrame:
    monkeypatch.setattr(yfinance_provider.yf, "Ticker", script.ticker)
    return YFinanceProvider().get_ohlcv(symbol, *WINDOW)


def _tz_lookup_failed(reason: str) -> Callable[[str], BaseException]:
    """yfinance's timezone lookup: it logs its own transport error and then
    reports the symbol as possibly delisted."""

    def step(ticker: str) -> BaseException:
        logging.getLogger("yfinance").error(
            f"Failed to get ticker '{ticker}' reason: {reason}"
        )
        return YFTzMissingError(ticker)

    return step


class TestATransportFailureIsTheVendors:
    @pytest.mark.parametrize(
        "failure",
        [
            requests.exceptions.ConnectionError("Connection aborted."),
            requests.exceptions.ReadTimeout("Read timed out. (read timeout=30)"),
            requests.exceptions.ChunkedEncodingError("Connection broken"),
            TimeoutError("timed out"),
            RuntimeError("*** YAHOO! FINANCE IS CURRENTLY DOWN! ***"),
        ],
        ids=["connection", "timeout", "chunked", "socket-timeout", "maintenance"],
    )
    def test_it_is_retried_then_named_never_no_data(self, monkeypatch, sleeps, failure):
        script = Script(failure)
        with pytest.raises(VendorUnavailableError) as caught:
            _serve(monkeypatch, script)
        assert len(script.calls) == 3, "the retry layer's three attempts"
        assert sleeps == [1, 2]
        text = str(caught.value)
        assert "No data found" not in text and "Verify symbol" not in text
        assert "Yahoo Finance failed on its side" in text
        assert "not an answer about the symbol" in text
        assert isinstance(caught.value, NonRetryableAPIError)

    def test_a_failure_that_clears_is_answered(self, monkeypatch):
        script = Script(requests.exceptions.ConnectionError("reset"), _frame())
        frame = _serve(monkeypatch, script)
        assert len(script.calls) == 2 and len(frame) == 22

    def test_a_rate_limit_is_named_with_its_status(self, monkeypatch):
        with pytest.raises(VendorUnavailableError, match="rate-limited") as caught:
            _serve(monkeypatch, Script(YFRateLimitError()))
        assert caught.value.status == 429

    def test_a_timezone_lookup_that_never_reached_yahoo(self, monkeypatch):
        """yfinance reports it as "possibly delisted; no timezone found";
        its log says the lookup failed to connect."""
        script = Script(_tz_lookup_failed("HTTPSConnectionPool: Read timed out."))
        with pytest.raises(VendorUnavailableError, match="timezone lookup") as caught:
            _serve(monkeypatch, script)
        assert "Read timed out" in str(caught.value)

    def test_a_yahoo_5xx_folded_into_no_prices_is_the_vendors(self, monkeypatch):
        failure = YFPricesMissingError(
            "AAPL", "(1d 2024-01-02 -> 2024-02-01)(Yahoo status_code = 503)"
        )
        with pytest.raises(VendorUnavailableError) as caught:
            _serve(monkeypatch, Script(failure))
        assert caught.value.status == 503

    def test_a_non_json_answer_is_the_vendors(self, monkeypatch):
        import json

        failure = json.JSONDecodeError("Expecting value", "<html>", 0)
        with pytest.raises(VendorUnavailableError):
            _serve(monkeypatch, Script(failure))

    def test_the_failure_is_not_cached(self, monkeypatch, tmp_path):
        with pytest.raises(VendorUnavailableError):
            _serve(monkeypatch, Script(requests.exceptions.ConnectionError("x")))
        assert not list((tmp_path / "cache").glob("*.parquet"))
        frame = _serve(monkeypatch, Script(_frame()))
        assert len(frame) == 22

    def test_the_history_call_asks_yfinance_to_raise(self, monkeypatch):
        script = Script(_frame())
        _serve(monkeypatch, script)
        assert script.calls[0].get("raise_errors") is True


class TestYahoosOwnEmptyAnswerIsStillNoData:
    def test_null_possibly_delisted_is_not_found_and_not_retried(self, monkeypatch):
        script = Script(YFTzMissingError("ZZZZ"))
        with pytest.raises(DataNotFoundError, match="No data found for 'ZZZZ'"):
            _serve(monkeypatch, script, symbol="ZZZZ")
        assert len(script.calls) == 1

    def test_null_no_prices_for_the_window_is_not_found(self, monkeypatch):
        failure = YFPricesMissingError("AAPL", "(1d 2024-01-02 -> 2024-02-01)")
        script = Script(failure)
        with pytest.raises(DataNotFoundError, match="Verify symbol and date range"):
            _serve(monkeypatch, script)
        assert len(script.calls) == 1

    def test_null_an_empty_frame_is_not_found(self, monkeypatch):
        script = Script(pd.DataFrame())
        with pytest.raises(DataNotFoundError, match="No data found for 'AAPL'"):
            _serve(monkeypatch, script)
        assert len(script.calls) == 1

    def test_null_another_threads_lookup_failure_is_not_read(self, monkeypatch):
        """The log line is read from the calling thread only: another
        symbol's failure, on another thread, says nothing about this one."""

        def step(ticker: str) -> BaseException:
            other = threading.Thread(
                target=lambda: logging.getLogger("yfinance").error(
                    f"Failed to get ticker '{ticker}' reason: timed out"
                )
            )
            other.start()
            other.join()
            return YFTzMissingError(ticker)

        with pytest.raises(DataNotFoundError):
            _serve(monkeypatch, Script(step))

    def test_null_an_unrecognised_error_is_the_old_api_error(self, monkeypatch):
        script = Script(KeyError("chart"))
        with pytest.raises(APIError, match="Error fetching data") as caught:
            _serve(monkeypatch, script)
        assert not isinstance(caught.value, VendorUnavailableError)
