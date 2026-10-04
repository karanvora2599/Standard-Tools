"""
A ticker-info or financial-ratio request that never reached Yahoo is not
"no data found".

`get_ticker_info` and `get_financial_ratios` read yfinance's `ticker.info`,
which cannot be asked to raise. yfinance catches the HTTP error of the
requests behind it, logs "HTTP Error 503: ..." and carries on: 0.2.65 then
fails with a TypeError, and 1.7.0 returns `{'trailingPegRatio': None}` --
the same dict Yahoo's answer for an unknown symbol gives -- which the
provider reported as "No metadata found". A connection error, a timeout, a
rate limit, Yahoo's maintenance page and a 5xx are now retried three times
and raised as `VendorUnavailableError` naming the symbol; Yahoo's own empty
answer stays `DataNotFoundError`. See the CHANGELOG entry of 2026-10-04.

Two kinds of stub, neither reaching the network:

- `TestTheInstalledYfinance` replaces only yfinance's HTTP layer, so the
  installed version's own `info` code runs: each interpreter in CI tests the
  yfinance it has (0.2.65 under pandas 2, 1.x under pandas 3).
- The other classes replace `yf.Ticker` with the shapes each version gives,
  so both versions' answers are held on every interpreter.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, List

import pytest
import requests

from standard_quant_tools.data import _cache as cache_module
from standard_quant_tools.data import yfinance_provider
from standard_quant_tools.data.yfinance_provider import YFinanceProvider
from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    NonRetryableAPIError,
    VendorUnavailableError,
)

yf = yfinance_provider.yf
YF_LOG = logging.getLogger("yfinance")

SUMMARY = {
    "quoteSummary": {
        "result": [
            {
                "assetProfile": {
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "city": "Cupertino",
                    "country": "United States",
                },
                "financialData": {
                    "debtToEquity": 150.5,
                    "returnOnEquity": 1.5,
                    "profitMargins": 0.25,
                },
                "summaryDetail": {"trailingPE": 30.0, "forwardPE": 28.0},
                "defaultKeyStatistics": {"priceToBook": 40.0},
                "quoteType": {"longName": "Apple Inc."},
            }
        ],
        "error": None,
    }
}
QUOTE = {
    "quoteResponse": {
        "result": [{"symbol": "AAPL", "longName": "Apple Inc.", "marketCap": 3e12}],
        "error": None,
    }
}
SERIES = {
    "timeseries": {
        "result": [
            {
                "meta": {"symbol": ["AAPL"], "type": ["trailingPegRatio"]},
                "timestamp": [1],
                "trailingPegRatio": [
                    {"asOfDate": "2024-01-02", "reportedValue": {"raw": 2.1}}
                ],
            }
        ],
        "error": None,
    }
}
UNKNOWN_SUMMARY = {
    "quoteSummary": {
        "result": None,
        "error": {"code": "Not Found", "description": "Quote not found for symbol"},
    }
}
UNKNOWN_QUOTE = {"quoteResponse": {"result": [], "error": None}}
UNKNOWN_SERIES = {
    "timeseries": {
        "result": [{"meta": {"symbol": ["ZZZZ"], "type": ["trailingPegRatio"]}}],
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

# The dict both versions give for a symbol Yahoo does not know -- and 1.7.0
# for a 5xx on both quote requests.
NO_ENTRY = {"trailingPegRatio": None}
FULL = {
    "longName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "city": "Cupertino",
    "debtToEquity": 150.5,
    "forwardPE": 28.0,
    "marketCap": 3e12,
    "trailingPegRatio": 2.1,
}


@pytest.fixture
def sleeps(monkeypatch) -> List[float]:
    taken: List[float] = []
    monkeypatch.setattr("time.sleep", lambda s: taken.append(s))
    return taken


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch, sleeps):
    monkeypatch.setattr(cache_module, "_CACHE_ROOT", tmp_path / "cache")


# ── yfinance's own `info`, its HTTP layer stubbed ────────────────────────


def _http_error_type() -> type:
    try:
        from yfinance._http import HTTPError  # yfinance 1.x
    except ImportError:  # yfinance 0.2.x catches curl_cffi's
        from curl_cffi.requests.exceptions import HTTPError
    return HTTPError


class _Response:
    """What yfinance reads from a response: status, body, url, and a
    `raise_for_status` raising the error class it catches."""

    REASONS = {200: "OK", 401: "Unauthorized", 404: "Not Found"}
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
            raise _http_error_type()(
                f"HTTP Error {self.status_code}: {self.reason}", 0, self
            )


class Yahoo:
    """Yahoo's three info endpoints, by what each answers: (status, body)
    or an exception (or a function making one). `summary_requests` counts
    the reads of `info`."""

    def __init__(self, summary: Any, quote: Any, series: Any) -> None:
        self.routes = {
            "quoteSummary": summary,
            "v7/finance/quote": quote,
            "fundamentals-timeseries": series,
        }
        self.summary_requests = 0

    def request(self, data, url, *_a, **_k):
        for key, step in self.routes.items():
            if key in url:
                if key == "quoteSummary":
                    self.summary_requests += 1
                if callable(step):
                    step = step()
                if isinstance(step, BaseException):
                    raise step
                return _Response(*step, url)
        raise AssertionError(f"unexpected request: {url}")


@pytest.fixture
def yahoo(monkeypatch):
    """Install a `Yahoo` behind the installed yfinance's request layer.
    `cache_get` memoizes responses for the process, so it is emptied before
    and after."""
    from yfinance.data import YfData

    if not callable(getattr(YfData, "_make_request", None)) or not hasattr(
        getattr(YfData, "cache_get", None), "cache_clear"
    ):
        pytest.skip("this yfinance has no request layer of the shape stubbed here")

    def install(summary=(200, SUMMARY), quote=(200, QUOTE), series=(200, SERIES)):
        site = Yahoo(summary, quote, series)
        monkeypatch.setattr(
            YfData, "_make_request", lambda self, url, *a, **k: site.request(self, url)
        )
        YfData.cache_get.cache_clear()
        return site

    yield install
    YfData.cache_get.cache_clear()


def _connection_error() -> BaseException:
    from curl_cffi.requests.exceptions import ConnectionError as CurlConnectionError

    return CurlConnectionError("Failed to connect to query2.finance.yahoo.com")


def _timeout() -> BaseException:
    from curl_cffi.requests.exceptions import Timeout as CurlTimeout

    return CurlTimeout("Operation timed out after 30000 milliseconds")


def _rate_limited() -> BaseException:
    from yfinance.exceptions import YFRateLimitError

    return YFRateLimitError()


class TestTheInstalledYfinance:
    @pytest.mark.parametrize(
        "routes,status",
        [
            (
                dict(
                    summary=(503, MAINTENANCE),
                    quote=(503, MAINTENANCE),
                    series=(503, MAINTENANCE),
                ),
                503,
            ),
            (
                dict(
                    summary=(503, MAINTENANCE),
                    quote=(503, MAINTENANCE),
                    series=(200, UNKNOWN_SERIES),
                ),
                503,
            ),
            (dict(summary=(504, "")), 504),
            (dict(quote=(503, MAINTENANCE)), 503),
            (
                dict(
                    summary=(200, MAINTENANCE),
                    quote=(200, MAINTENANCE),
                    series=(200, MAINTENANCE),
                ),
                None,
            ),
            (dict(summary=_connection_error), None),
            (dict(summary=_timeout), None),
            (dict(summary=_rate_limited), 429),
        ],
        ids=[
            "5xx-everywhere",
            "5xx-on-both-quotes",
            "5xx-on-the-summary",
            "5xx-on-the-quote",
            "maintenance-page",
            "connection",
            "timeout",
            "rate-limit",
        ],
    )
    def test_a_request_yahoo_never_answered_is_the_vendors(
        self, yahoo, sleeps, routes, status
    ):
        """Every kind, through the installed yfinance's own `info`: retried
        three times, then named as Yahoo's failure with its status -- never
        "No metadata found", and never the fields that were left."""
        site = yahoo(**routes)
        with pytest.raises(VendorUnavailableError) as caught:
            YFinanceProvider().get_ticker_info("AAPL")
        assert site.summary_requests == 3 and sleeps == [1, 2]
        text = str(caught.value)
        assert "Yahoo Finance failed on its side for 'AAPL' (ticker info)" in text
        assert "not an answer about the symbol" in text
        assert "No metadata found" not in text
        assert caught.value.status == status
        assert isinstance(caught.value, NonRetryableAPIError)

    def test_the_ratios_are_named_the_same_way(self, yahoo):
        yahoo(summary=(503, MAINTENANCE), quote=(503, MAINTENANCE))
        with pytest.raises(VendorUnavailableError, match=r"'AAPL' \(financial data\)"):
            YFinanceProvider().get_financial_ratios("AAPL")

    def test_a_failure_that_clears_is_answered(self, yahoo, monkeypatch):
        site = yahoo(summary=(503, MAINTENANCE))
        reads = {"n": 0}

        def request(self, url, *a, **k):
            if "quoteSummary" in url:
                reads["n"] += 1
                if reads["n"] > 1:
                    site.routes["quoteSummary"] = (200, SUMMARY)
            return site.request(self, url)

        from yfinance.data import YfData

        monkeypatch.setattr(YfData, "_make_request", request)
        info = YFinanceProvider().get_ticker_info("AAPL")
        assert (info.name, info.sector) == ("Apple Inc.", "Technology")
        assert reads["n"] == 2

    def test_null_an_unknown_symbol_is_not_found_and_not_retried(self, yahoo, sleeps):
        """Yahoo's own answer: a 404 on the summary and no quote."""
        site = yahoo(
            summary=(404, UNKNOWN_SUMMARY),
            quote=(200, UNKNOWN_QUOTE),
            series=(200, UNKNOWN_SERIES),
        )
        provider = YFinanceProvider()
        with pytest.raises(DataNotFoundError) as caught:
            provider.get_ticker_info("ZZZZ")
        assert str(caught.value) == "No metadata found for 'ZZZZ'."
        with pytest.raises(DataNotFoundError) as caught:
            provider.get_financial_ratios("ZZZZ")
        assert str(caught.value) == "No financial data found for 'ZZZZ'."
        assert site.summary_requests == 2 and sleeps == []

    def test_null_a_healthy_answer_is_read_as_before(self, yahoo):
        yahoo(summary=(200, SUMMARY))
        provider = YFinanceProvider()
        info = provider.get_ticker_info("AAPL")
        assert (info.name, info.sector, info.city) == (
            "Apple Inc.",
            "Technology",
            "Cupertino",
        )
        ratios = provider.get_financial_ratios("AAPL")
        assert ratios.debt_to_equity == pytest.approx(1.505)
        assert (ratios.forward_pe, ratios.market_cap) == (28.0, 3e12)

    def test_null_a_refused_request_is_not_found_but_says_so(self, yahoo, sleeps):
        """A 401 (Yahoo's "Invalid Crumb") is not one of the transport
        failures and is not named as the vendor's. 1.7.0's empty answer
        stays not-found and 0.2.65's TypeError the old APIError; either
        message names the status yfinance logged."""
        yahoo(summary=(401, REFUSED), quote=(401, REFUSED), series=(200, SERIES))
        with pytest.raises((DataNotFoundError, APIError)) as caught:
            YFinanceProvider().get_ticker_info("AAPL")
        assert not isinstance(caught.value, VendorUnavailableError)
        assert "HTTP 401" in str(caught.value)


# ── Each version's answer, on every interpreter ──────────────────────────


class FakeTicker:
    """`yf.Ticker` whose `info` runs a step: log lines, then a dict to
    return or an exception to raise. The last step repeats."""

    steps: List[Callable[[], Any]] = []
    reads = 0

    def __init__(self, symbol: str, *_a, **_k) -> None:
        self.ticker = symbol.upper()

    @property
    def info(self) -> Any:
        cls = type(self)
        step = cls.steps.pop(0) if len(cls.steps) > 1 else cls.steps[0]
        cls.reads += 1
        out = step()
        if isinstance(out, BaseException):
            raise out
        return out


@pytest.fixture
def ticker(monkeypatch):
    def install(*steps: Callable[[], Any]) -> type:
        cls = type("Ticker", (FakeTicker,), {"steps": list(steps), "reads": 0})
        monkeypatch.setattr(yf, "Ticker", cls)
        return cls

    return install


def logs(*lines: str, then: Any) -> Callable[[], Any]:
    """A read of `info` that logs `lines` on yfinance's logger, as its
    quote requests do, and then gives `then`."""

    def step() -> Any:
        for line in lines:
            YF_LOG.error(line)
        return dict(then) if isinstance(then, dict) else then

    return step


def _requests_http_error(status: int) -> BaseException:
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(f"{status} Server Error", response=response)


V0265_5XX = logs(
    "HTTP Error 503: Service Unavailable",
    "HTTP Error 503: Service Unavailable",
    then=TypeError("argument of type 'NoneType' is not iterable"),
)
V170_5XX = logs(
    "HTTP Error 503: Service Unavailable<html><body>Will be right back",
    "HTTP Error 503: Service Unavailable<html><body>Will be right back",
    then=NO_ENTRY,
)
V170_5XX_ON_THE_SUMMARY = logs(
    "HTTP Error 502: Bad Gateway<html>", then={"longName": "Apple Inc.", **NO_ENTRY}
)
V_UNKNOWN = logs("HTTP Error 404: Not Found", then=NO_ENTRY)
V0265_UNKNOWN_EMPTY_SERIES = logs(
    "HTTP Error 404: Not Found", then=IndexError("list index out of range")
)


class TestEachVersionsShapeOfAFailure:
    @pytest.mark.parametrize(
        "step,status",
        [
            (V0265_5XX, 503),
            (V170_5XX, 503),
            (V170_5XX_ON_THE_SUMMARY, 502),
            (lambda: _requests_http_error(503), 503),
            (lambda: _requests_http_error(408), 408),
            (lambda: requests.exceptions.ConnectionError("Connection aborted."), None),
            (lambda: requests.exceptions.ReadTimeout("Read timed out."), None),
            (lambda: json.JSONDecodeError("Expecting value", "<html>", 0), None),
        ],
        ids=[
            "0.2.65-typeerror-after-503",
            "1.7.0-no-entry-after-503",
            "1.7.0-partial-after-502",
            "1.x-raised-503",
            "1.x-raised-408",
            "connection",
            "timeout",
            "maintenance-page",
        ],
    )
    def test_it_is_retried_then_named(self, ticker, sleeps, step, status):
        fake = ticker(step)
        with pytest.raises(VendorUnavailableError) as caught:
            YFinanceProvider().get_ticker_info("AAPL")
        assert fake.reads == 3 and sleeps == [1, 2]
        assert caught.value.status == status
        assert "No metadata found" not in str(caught.value)

    def test_the_ratios_too(self, ticker):
        ticker(V170_5XX)
        with pytest.raises(VendorUnavailableError) as caught:
            YFinanceProvider().get_financial_ratios("AAPL")
        assert "has no financial data" in str(caught.value)

    def test_a_failure_that_clears_is_answered(self, ticker, sleeps):
        fake = ticker(V170_5XX, logs(then=FULL))
        info = YFinanceProvider().get_ticker_info("AAPL")
        assert info.name == "Apple Inc." and fake.reads == 2 and sleeps == [1]


class TestYahoosOwnEmptyAnswerIsStillNoData:
    def test_null_the_unknown_symbol_dict_is_not_found(self, ticker, sleeps):
        fake = ticker(V_UNKNOWN)
        with pytest.raises(DataNotFoundError) as caught:
            YFinanceProvider().get_ticker_info("ZZZZ")
        assert str(caught.value) == "No metadata found for 'ZZZZ'."
        assert fake.reads == 1 and sleeps == []

    def test_null_the_ratios_of_an_unknown_symbol_are_not_found(self, ticker):
        """Before, `{'trailingPegRatio': None}` passed the emptiness check and
        came back as FinancialRatios with every field None."""
        ticker(V_UNKNOWN)
        with pytest.raises(DataNotFoundError) as caught:
            YFinanceProvider().get_financial_ratios("ZZZZ")
        assert str(caught.value) == "No financial data found for 'ZZZZ'."

    @pytest.mark.parametrize(
        "step",
        [V0265_UNKNOWN_EMPTY_SERIES, lambda: _requests_http_error(404)],
        ids=["0.2.65-indexerror-after-404", "1.x-raised-404"],
    )
    def test_null_a_404_that_then_fails_is_not_found(self, ticker, sleeps, step):
        fake = ticker(step)
        with pytest.raises(DataNotFoundError, match="No metadata found for 'ZZZZ'"):
            YFinanceProvider().get_ticker_info("ZZZZ")
        assert fake.reads == 1

    def test_null_an_empty_dict_is_not_found(self, ticker):
        ticker(logs(then={}))
        with pytest.raises(DataNotFoundError):
            YFinanceProvider().get_financial_ratios("ZZZZ")

    def test_null_an_unrecognised_error_is_the_old_api_error(self, ticker, sleeps):
        fake = ticker(lambda: KeyError("quoteSummary"))
        with pytest.raises(
            APIError, match="Error fetching ticker info for 'AAPL'"
        ) as c:
            YFinanceProvider().get_ticker_info("AAPL")
        assert not isinstance(c.value, VendorUnavailableError)
        assert fake.reads == 3

    def test_null_another_threads_5xx_is_not_read(self, ticker):
        """The log is read from the calling thread only: another symbol's
        failure, on another thread, says nothing about this answer."""

        def step() -> Any:
            other = threading.Thread(
                target=lambda: YF_LOG.error("HTTP Error 503: Service Unavailable")
            )
            other.start()
            other.join()
            return dict(FULL)

        ticker(step)
        assert YFinanceProvider().get_ticker_info("AAPL").sector == "Technology"
