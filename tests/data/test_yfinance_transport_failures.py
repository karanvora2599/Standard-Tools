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

The requests behind yfinance's own "no data" -- the timezone lookup, whose
error yfinance swallows, and the history request -- are read from
yfinance's data layer, status first: a 5xx, a 408 or a 429 is the vendor's
whatever the error's type or the answer's body, a 404 is Yahoo's answer
about the symbol, and a refusal (401, 403) keeps its old handling. That
holds whatever the process's logging configuration. See the CHANGELOG entry
of 2026-10-04 on the logging configuration and the history status.

Two kinds of stub, neither reaching the network: `TestTheInstalledYfinance`
replaces only yfinance's HTTP layer, so the installed version's own history
code runs; the other classes replace `yf.Ticker`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from typing import Any, Callable, Iterator, List, Optional
from unittest import mock

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


def _swallowed_lookup(ticker: str, failure: BaseException) -> None:
    """yfinance's timezone lookup, its request failing with `failure`: the
    request goes through yfinance's data layer (`cache_get`), and the
    lookup catches the error, logs it and carries on."""
    from yfinance.data import YfData

    def fail(self, url, *_a, **_k):
        raise failure

    with mock.patch.object(YfData, "_make_request", fail):
        try:
            YfData().cache_get(
                url=f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"range": "1d", "interval": "1d"},
                timeout=10,
            )
        except Exception as exc:  # noqa: BLE001 - yfinance's own handling
            logging.getLogger("yfinance").error(
                f"Failed to get ticker '{ticker}' reason: {exc}"
            )


def _tz_lookup_failed(failure: BaseException) -> Callable[[str], BaseException]:
    """yfinance's timezone lookup: it swallows its own transport error and
    then reports the symbol as possibly delisted."""

    def step(ticker: str) -> BaseException:
        _swallowed_lookup(ticker, failure)
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
        the lookup's request timed out."""
        script = Script(
            _tz_lookup_failed(
                requests.exceptions.ReadTimeout("HTTPSConnectionPool: Read timed out.")
            )
        )
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
        """Only the calling thread's requests are read: another symbol's
        failed lookup, on another thread, says nothing about this one."""

        def step(ticker: str) -> BaseException:
            other = threading.Thread(
                target=lambda: _swallowed_lookup(
                    ticker, requests.exceptions.ReadTimeout("timed out")
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


# ── yfinance's own history code, its HTTP layer stubbed ──────────────────


def _client_http_error() -> type:
    """The HTTP error class the installed yfinance's client raises."""
    try:
        from yfinance._http import HTTPError  # yfinance 1.x
    except ImportError:  # yfinance 0.2.x uses curl_cffi's
        from curl_cffi.requests.exceptions import HTTPError
    return HTTPError


class _Response:
    """What yfinance reads from a response: status, body, url, and a
    `raise_for_status` raising its client's error class."""

    REASONS = {200: "OK", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found"}
    REASONS.update({408: "Request Timeout", 429: "Too Many Requests"})
    REASONS.update({500: "Internal Server Error", 502: "Bad Gateway"})
    REASONS.update({503: "Service Unavailable", 504: "Gateway Timeout"})

    def __init__(self, status: int, body: Any, url: str) -> None:
        self.status_code = status
        self.reason = self.REASONS.get(status, "")
        self.text = body if isinstance(body, str) else json.dumps(body)
        self.content = self.text.encode()
        self.url = url
        self.ok = status < 400

    def json(self, **_kw):
        return json.loads(self.text)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise _client_http_error()(
                f"HTTP Error {self.status_code}: {self.reason}", 0, self
            )


def _raised(status: int, client: str = "yfinance") -> BaseException:
    """The HTTP error a client raises for `status`: yfinance's own client
    (curl_cffi's `HTTPError`, a `CurlError`), or requests' -- the client
    yfinance 1.x falls back to without curl_cffi."""
    url = "https://query2.finance.yahoo.com/v8/finance/chart/AAPL"
    response = _Response(status, "", url)
    if client == "requests":
        return requests.exceptions.HTTPError(
            f"{status} Server Error: {response.reason} for url: {url}",
            response=response,
        )
    return _client_http_error()(f"HTTP Error {status}: {response.reason}", 0, response)


TZ_ANSWER = {
    "chart": {
        "result": [
            {"meta": {"symbol": "AAPL", "exchangeTimezoneName": "America/New_York"}}
        ],
        "error": None,
    }
}
MAINTENANCE = "<html><body>Will be right back... Yahoo! Finance is currently down"
REFUSED = {
    "finance": {
        "result": None,
        "error": {"code": "Unauthorized", "description": "Invalid Crumb"},
    }
}


def _chart_error(code: str, description: str) -> dict:
    return {
        "chart": {"result": None, "error": {"code": code, "description": description}}
    }


def _chart() -> dict:
    """Yahoo's chart answer for the window: its 22 sessions, closing at
    100, 101, ..., 121."""
    days = pd.bdate_range(*WINDOW, tz="America/New_York")
    stamps = [
        int((day + pd.Timedelta(hours=9, minutes=30)).timestamp()) for day in days
    ]
    close = [100.0 + i for i in range(len(days))]
    meta = {
        "currency": "USD",
        "symbol": "AAPL",
        "exchangeName": "NMS",
        "instrumentType": "EQUITY",
        "gmtoffset": -18000,
        "timezone": "EST",
        "exchangeTimezoneName": "America/New_York",
        "priceHint": 2,
        "dataGranularity": "1d",
        "range": "",
    }
    quote = {
        "open": close,
        "high": [c + 1 for c in close],
        "low": [c - 1 for c in close],
        "close": close,
        "volume": [1_000_000] * len(days),
    }
    return {
        "chart": {
            "result": [
                {
                    "meta": meta,
                    "timestamp": stamps,
                    "indicators": {"quote": [quote], "adjclose": [{"adjclose": close}]},
                }
            ],
            "error": None,
        }
    }


class Chart:
    """Yahoo's chart endpoint, as the timezone lookup (`range=1d`) and the
    history request reach it: each answers from its steps in turn --
    (status, body), an exception, or a function making one; the last step
    repeats -- and counts the requests it got."""

    def __init__(self, lookup: list, history: list) -> None:
        self.steps = {"lookup": list(lookup), "history": list(history)}
        self.requests = {"lookup": 0, "history": 0}

    def request(self, url: str, params: Optional[dict]) -> _Response:
        kind = "lookup" if (params or {}).get("range") == "1d" else "history"
        self.requests[kind] += 1
        steps = self.steps[kind]
        step = steps.pop(0) if len(steps) > 1 else steps[0]
        if callable(step):
            step = step()
        if isinstance(step, BaseException):
            raise step
        return _Response(*step, url)


@pytest.fixture
def chart(monkeypatch):
    """Install a `Chart` behind the installed yfinance's request layer.
    yfinance keeps each symbol's timezone on disk and, twice per process,
    asks `info` for one its lookup could not get: neither happens here, so
    every history call asks the stub. `cache_get` memoizes answers for the
    process, so it is emptied before and after."""
    from yfinance import base as yf_base
    from yfinance import cache as yf_cache
    from yfinance.data import YfData

    if not (
        callable(getattr(YfData, "_make_request", None))
        and hasattr(getattr(YfData, "cache_get", None), "cache_clear")
        and hasattr(yf_cache, "_TzCacheDummy")
        and hasattr(yf_base, "_tz_info_fetch_ctr")
    ):
        pytest.skip("this yfinance has no request layer of the shape stubbed here")
    monkeypatch.setattr(yf_cache, "get_tz_cache", lambda: yf_cache._TzCacheDummy())
    monkeypatch.setattr(yf_base, "_tz_info_fetch_ctr", 2)

    def install(history=((200, _chart()),), lookup=((200, TZ_ANSWER),)) -> Chart:
        site = Chart(list(lookup), list(history))
        monkeypatch.setattr(
            YfData,
            "_make_request",
            lambda self, url, *a, **k: site.request(url, k.get("params")),
        )
        YfData.cache_get.cache_clear()
        return site

    yield install
    YfData.cache_get.cache_clear()


#: The ways a process silences yfinance's log: its level, `disabled`, a
#: parent's level, and `logging.disable`.
SILENCED = (
    "yfinance-logger-above-error",
    "yfinance-logger-disabled",
    "a-parent-above-error",
    "logging-disable",
)


@contextlib.contextmanager
def _silenced(mode: Optional[str]) -> Iterator[None]:
    """yfinance's log silenced as `mode` says (left alone for None), and
    restored after."""
    yf_log, root = logging.getLogger("yfinance"), logging.getLogger()
    before = (yf_log.level, yf_log.disabled, root.level, logging.root.manager.disable)
    try:
        if mode == "yfinance-logger-above-error":
            yf_log.setLevel(logging.CRITICAL)
        elif mode == "yfinance-logger-disabled":
            yf_log.disabled = True
        elif mode == "a-parent-above-error":
            yf_log.setLevel(logging.NOTSET)
            root.setLevel(logging.CRITICAL)
        elif mode == "logging-disable":
            logging.disable(logging.CRITICAL)
        yield
    finally:
        yf_log.setLevel(before[0])
        yf_log.disabled = before[1]
        root.setLevel(before[2])
        logging.disable(before[3])


VENDOR_STATUSES = (500, 502, 503, 504, 408, 429)


def _get(symbol: str = "AAPL") -> pd.DataFrame:
    return YFinanceProvider().get_ohlcv(symbol, *WINDOW)


@pytest.mark.filterwarnings("ignore:'raise_errors' deprecated:DeprecationWarning")
class TestTheInstalledYfinance:
    """The installed version's own history code, each interpreter in CI
    testing the yfinance it has (0.2.65 under pandas 2, 1.x under pandas
    3). Before, an HTTP error curl_cffi raised was a transport failure
    whatever its status -- a 404 or a 401 was retried three times and named
    as Yahoo's outage -- and a 5xx Yahoo answered with a JSON error body was
    "No data found ... Verify symbol and date range"."""

    def test_null_a_healthy_answer_is_read_as_before(self, chart):
        site = chart()
        frame = _get()
        assert len(frame) == 22 and site.requests == {"lookup": 1, "history": 1}
        assert frame["Close"].tolist() == [100.0 + i for i in range(22)]

    @pytest.mark.parametrize("client", ["yfinance", "requests"])
    @pytest.mark.parametrize("status", VENDOR_STATUSES)
    def test_a_raised_vendor_status_is_the_vendors(self, chart, sleeps, status, client):
        site = chart(history=[lambda: _raised(status, client)])
        with pytest.raises(VendorUnavailableError) as caught:
            _get()
        assert caught.value.status == status
        assert site.requests["history"] == 3 and sleeps == [1, 2]
        assert f"HTTP {status}" in str(caught.value)
        assert "No data found" not in str(caught.value)

    @pytest.mark.parametrize("client", ["yfinance", "requests"])
    def test_null_a_raised_404_is_not_found_and_not_retried(
        self, chart, sleeps, client
    ):
        site = chart(history=[lambda: _raised(404, client)])
        with pytest.raises(DataNotFoundError) as caught:
            _get()
        assert str(caught.value).startswith(
            "No data found for 'AAPL'. Verify symbol and date range. (Yahoo: "
        )
        assert "404" in str(caught.value)
        assert site.requests["history"] == 1 and sleeps == []

    @pytest.mark.parametrize("client", ["yfinance", "requests"])
    @pytest.mark.parametrize("status", [401, 403])
    def test_null_a_raised_refusal_keeps_its_old_handling(
        self, chart, sleeps, status, client
    ):
        """Not the vendor's failure and not an answer about the symbol: the
        old `APIError`, asked three times as any `APIError` is, its message
        naming the status."""
        site = chart(history=[lambda: _raised(status, client)])
        with pytest.raises(APIError, match="Error fetching data for 'AAPL'") as caught:
            _get()
        assert not isinstance(caught.value, (VendorUnavailableError, DataNotFoundError))
        assert str(status) in str(caught.value)
        assert site.requests["history"] == 3

    @pytest.mark.parametrize("status", VENDOR_STATUSES)
    def test_an_answered_vendor_status_is_the_vendors(self, chart, sleeps, status):
        """yfinance turns the answer's JSON error into "no price data
        found"; the status is read first. Each attempt reaches Yahoo:
        yfinance's memo of the failed answer is dropped before the next."""
        site = chart(history=[(status, _chart_error("Internal Server Error", "busy"))])
        with pytest.raises(VendorUnavailableError) as caught:
            _get()
        assert caught.value.status == status
        assert site.requests["history"] == 3 and sleeps == [1, 2]
        assert "No data found" not in str(caught.value)

    def test_a_5xx_answer_that_clears_is_answered(self, chart, sleeps):
        site = chart(history=[(503, _chart_error("x", "busy")), (200, _chart())])
        assert len(_get()) == 22
        assert site.requests["history"] == 2 and sleeps == [1]

    def test_null_an_answered_404_is_not_found_as_before(self, chart, sleeps):
        site = chart(history=[(404, _chart_error("Not Found", "No data found"))])
        with pytest.raises(DataNotFoundError) as caught:
            _get()
        text = str(caught.value)
        assert text.startswith("No data found for 'AAPL'. Verify symbol and date ")
        assert text.endswith('(Yahoo error = "No data found"))') or text.endswith(
            "No data found)"
        )
        assert site.requests["history"] == 1 and sleeps == []

    @pytest.mark.parametrize("status", [401, 403])
    def test_null_an_answered_refusal_is_not_found_and_says_so(
        self, chart, sleeps, status
    ):
        site = chart(history=[(status, REFUSED)])
        with pytest.raises(DataNotFoundError) as caught:
            _get()
        assert f"Yahoo answered HTTP {status} " in str(caught.value)
        assert "may be a refused request" in str(caught.value)
        assert site.requests["history"] == 1 and sleeps == []

    @pytest.mark.parametrize("mode", (None,) + SILENCED)
    @pytest.mark.parametrize(
        "lookup,status",
        [
            (lambda: requests.exceptions.ReadTimeout("Read timed out."), None),
            (lambda: _raised(503), 503),
            ((503, MAINTENANCE), 503),
            ((503, _chart_error("x", "busy")), 503),
            ((200, MAINTENANCE), None),
        ],
        ids=["timeout", "raised-503", "503-page", "503-json", "maintenance-page"],
    )
    def test_a_timezone_lookup_that_failed_is_the_vendors(
        self, chart, sleeps, lookup, status, mode
    ):
        """yfinance swallows the lookup's failure and reports the symbol as
        possibly delisted; the failure is read from its data layer, however
        the process configured its log."""
        site = chart(lookup=[lookup])
        with _silenced(mode):
            with pytest.raises(
                VendorUnavailableError, match="timezone lookup"
            ) as caught:
                _get()
        assert caught.value.status == status
        assert site.requests["lookup"] == 3 and site.requests["history"] == 0
        assert sleeps == [1, 2]

    @pytest.mark.parametrize("mode", (None,) + SILENCED)
    def test_null_an_unknown_symbol_is_not_found(self, chart, sleeps, mode):
        site = chart(lookup=[(404, _chart_error("Not Found", "No data found"))])
        with _silenced(mode):
            with pytest.raises(DataNotFoundError) as caught:
                _get()
        assert str(caught.value) == (
            "No data found for 'AAPL'. Verify symbol and date range. "
            "(Yahoo: $AAPL: possibly delisted; no timezone found)"
        )
        assert site.requests["lookup"] == 1 and sleeps == []

    @pytest.mark.parametrize("mode", [None, "yfinance-logger-above-error"])
    def test_two_threads_each_read_only_their_own_requests(
        self, chart, sleeps, monkeypatch, mode
    ):
        """One symbol's timezone lookup times out while another's, on
        another thread, is answered at the same moment (both threads wait
        for each other inside the lookup): the first is the vendor's
        failure, the second is read as before."""
        from yfinance.data import YfData

        chart()  # the skip check and the emptied memo
        meet = threading.Barrier(2, timeout=10)
        met: set = set()
        lock = threading.Lock()

        def request(self, url, *_a, params=None, **_k):
            symbol = "AAPL" if "AAPL" in url else "MSFT"
            if (params or {}).get("range") == "1d":
                with lock:
                    first = symbol not in met
                    met.add(symbol)
                if first:
                    meet.wait()
                if symbol == "AAPL":
                    raise requests.exceptions.ReadTimeout("Read timed out.")
                return _Response(200, TZ_ANSWER, url)
            return _Response(200, _chart(), url)

        monkeypatch.setattr(YfData, "_make_request", request)
        results: dict = {}

        def read(symbol: str) -> None:
            try:
                results[symbol] = _get(symbol)
            except Exception as exc:  # noqa: BLE001 - asserted below
                results[symbol] = exc

        threads = [threading.Thread(target=read, args=(s,)) for s in ("AAPL", "MSFT")]
        with _silenced(mode):
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
        failed, healthy = results["AAPL"], results["MSFT"]
        assert isinstance(failed, VendorUnavailableError), repr(failed)
        assert "timezone lookup" in str(failed)
        assert isinstance(healthy, pd.DataFrame), repr(healthy)
        assert len(healthy) == 22
