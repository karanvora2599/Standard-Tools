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

The error yfinance catches is read from its data layer as `get_raw_json`
raises it, not from the line yfinance logs, so a process that silences
yfinance's logger -- its level, `disabled`, a parent's level or
`logging.disable` -- no longer turns a 5xx back into "No metadata found"
(the CHANGELOG entry of 2026-10-04 on the logging configuration).

Two kinds of stub, neither reaching the network:

- `TestTheInstalledYfinance` and `TestTheLoggingConfiguration` replace only
  yfinance's HTTP layer, so the installed version's own `info` code runs:
  each interpreter in CI tests the yfinance it has (0.2.65 under pandas 2,
  1.x under pandas 3).
- The other classes replace `yf.Ticker` with the shapes each version gives,
  their failed requests made through yfinance's data layer, so both
  versions' answers are held on every interpreter.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from typing import Any, Callable, Iterator, List, Tuple
from unittest import mock

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


# ── whatever the process's logging configuration ─────────────────────────

#: The ways a process silences yfinance's log: its level, `disabled`, a
#: parent's level, and `logging.disable`.
SILENCED = (
    "yfinance-logger-above-error",
    "yfinance-logger-disabled",
    "a-parent-above-error",
    "logging-disable",
)


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextlib.contextmanager
def _silenced(mode: str) -> Iterator[None]:
    """yfinance's log silenced as `mode` says, and restored after. Checked
    first: an ERROR on yfinance's logger reaches no handler."""
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
        else:
            logging.disable(logging.CRITICAL)
        probe = _Collect()
        yf_log.addHandler(probe)
        try:
            yf_log.error("HTTP Error 503: Service Unavailable")
        finally:
            yf_log.removeHandler(probe)
        assert probe.records == [], f"{mode} does not silence yfinance's log"
        yield
    finally:
        yf_log.setLevel(before[0])
        yf_log.disabled = before[1]
        root.setLevel(before[2])
        logging.disable(before[3])


def _logging_state() -> tuple:
    """What a caller configured: yfinance's logger and the root logger's
    level, `disabled`, handlers, and the process-wide `disable`."""
    yf_log, root = logging.getLogger("yfinance"), logging.getLogger()
    return (
        yf_log.level,
        yf_log.disabled,
        tuple(yf_log.handlers),
        root.level,
        tuple(root.handlers),
        logging.root.manager.disable,
    )


class TestTheLoggingConfiguration:
    """The installed yfinance's own `info` with its log silenced. Before,
    each of these hid the line the 5xx detection read: a 5xx on both quote
    requests read as "No metadata found" under 1.7.0 and as the old
    `APIError` under 0.2.65."""

    @pytest.mark.parametrize("mode", SILENCED)
    @pytest.mark.parametrize(
        "routes,status",
        [
            (
                dict(
                    summary=(503, MAINTENANCE),
                    quote=(503, MAINTENANCE),
                    series=(200, UNKNOWN_SERIES),
                ),
                503,
            ),
            (dict(summary=(504, "")), 504),
            (dict(quote=(502, "<html>")), 502),
        ],
        ids=["5xx-on-both-quotes", "5xx-on-the-summary", "5xx-on-the-quote"],
    )
    def test_a_5xx_is_the_vendors_however_the_log_is_set(
        self, yahoo, sleeps, mode, routes, status
    ):
        site = yahoo(**routes)
        with _silenced(mode):
            with pytest.raises(VendorUnavailableError) as caught:
                YFinanceProvider().get_ticker_info("AAPL")
        assert caught.value.status == status
        assert site.summary_requests == 3 and sleeps == [1, 2]
        assert "No metadata found" not in str(caught.value)

    @pytest.mark.parametrize("mode", SILENCED)
    def test_the_ratios_too(self, yahoo, mode):
        yahoo(summary=(503, MAINTENANCE), quote=(503, MAINTENANCE))
        with _silenced(mode):
            with pytest.raises(VendorUnavailableError, match=r"\(financial data\)"):
                YFinanceProvider().get_financial_ratios("AAPL")

    @pytest.mark.parametrize("mode", SILENCED)
    def test_null_an_unknown_symbol_is_still_not_found(self, yahoo, sleeps, mode):
        site = yahoo(
            summary=(404, UNKNOWN_SUMMARY),
            quote=(200, UNKNOWN_QUOTE),
            series=(200, UNKNOWN_SERIES),
        )
        with _silenced(mode):
            with pytest.raises(DataNotFoundError) as caught:
                YFinanceProvider().get_ticker_info("ZZZZ")
        assert str(caught.value) == "No metadata found for 'ZZZZ'."
        assert site.summary_requests == 1 and sleeps == []

    @pytest.mark.parametrize("mode", SILENCED)
    def test_null_a_refusal_still_names_its_status(self, yahoo, mode):
        yahoo(summary=(401, REFUSED), quote=(401, REFUSED), series=(200, SERIES))
        with _silenced(mode):
            with pytest.raises((DataNotFoundError, APIError)) as caught:
                YFinanceProvider().get_ticker_info("AAPL")
        assert not isinstance(caught.value, VendorUnavailableError)
        assert "HTTP 401" in str(caught.value)

    @pytest.mark.parametrize("mode", SILENCED)
    def test_null_a_healthy_answer_is_read_as_before(self, yahoo, mode):
        yahoo()
        with _silenced(mode):
            info = YFinanceProvider().get_ticker_info("AAPL")
        assert (info.name, info.sector, info.city) == (
            "Apple Inc.",
            "Technology",
            "Cupertino",
        )

    def test_nothing_is_emitted_and_the_configuration_is_left_alone(
        self, yahoo, sleeps
    ):
        """The provider adds no handler, lowers no level and lifts no
        `disable` to see the error: a caller who silenced yfinance's logger
        gets none of its lines, and the configuration is the caller's while
        each request runs and after."""
        site = yahoo(summary=(503, MAINTENANCE), quote=(503, MAINTENANCE))
        while_requesting: List[tuple] = []
        answer = site.request

        def request(data, url, *a, **k):
            while_requesting.append(_logging_state())
            return answer(data, url, *a, **k)

        site.request = request
        collect = _Collect()
        root = logging.getLogger()
        root.addHandler(collect)
        try:
            with _silenced("yfinance-logger-above-error"):
                configured = _logging_state()
                with pytest.raises(VendorUnavailableError):
                    YFinanceProvider().get_ticker_info("AAPL")
                after = _logging_state()
        finally:
            root.removeHandler(collect)
        # Three reads, each of both quote requests (and 1.x's time series).
        assert site.summary_requests == 3 and len(while_requesting) >= 6
        assert set(while_requesting) == {configured} and after == configured
        assert [r for r in collect.records if r.name.startswith("yfinance")] == []

    @pytest.mark.parametrize("mode", [None, "yfinance-logger-above-error"])
    def test_two_threads_each_read_only_their_own_requests(
        self, yahoo, sleeps, monkeypatch, mode
    ):
        """One symbol's quote requests fail while another's, on another
        thread, are answered at the same moment (both threads wait for each
        other inside their first two requests): the failing symbol is the
        vendor's failure and the healthy one is read as before."""
        from yfinance.data import YfData

        yahoo()  # the skip check and the emptied memo
        meet = {
            "quoteSummary": threading.Barrier(2, timeout=10),
            "v7/finance/quote": threading.Barrier(2, timeout=10),
        }
        met: set = set()
        lock = threading.Lock()

        def request(self, url, *_a, params=None, **_k):
            symbol = "AAPL" if "AAPL" in f"{url}{params}" else "MSFT"
            for key, barrier in meet.items():
                if key in url:
                    with lock:
                        first = (symbol, key) not in met
                        met.add((symbol, key))
                    if first:
                        barrier.wait()
                    if symbol == "AAPL":
                        return _Response(503, MAINTENANCE, url)
                    body = SUMMARY if key == "quoteSummary" else QUOTE
                    return _Response(200, body, url)
            return _Response(200, SERIES, url)

        monkeypatch.setattr(YfData, "_make_request", request)
        results: dict = {}

        def read(symbol: str) -> None:
            try:
                results[symbol] = YFinanceProvider().get_ticker_info(symbol)
            except Exception as exc:  # noqa: BLE001 - asserted below
                results[symbol] = exc

        threads = [threading.Thread(target=read, args=(s,)) for s in ("AAPL", "MSFT")]
        with _silenced(mode) if mode else contextlib.nullcontext():
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
        failed, healthy = results["AAPL"], results["MSFT"]
        assert isinstance(failed, VendorUnavailableError), repr(failed)
        assert failed.status == 503
        assert not isinstance(healthy, BaseException), repr(healthy)
        assert (healthy.symbol, healthy.sector) == ("MSFT", "Technology")


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


def _caught_request(status: int, body: Any) -> None:
    """One quote request answered with `status`, as yfinance makes it: its
    data layer's `get_raw_json` raises the HTTP error, and the quote code
    catches it, logs it and carries on."""
    from yfinance.data import YfData

    def answer(self, url, *_a, **_k):
        return _Response(status, body, url)

    with mock.patch.object(YfData, "_make_request", answer):
        try:
            YfData().get_raw_json(
                "https://query2.finance.yahoo.com/v10/finance/quoteSummary/AAPL"
            )
        except Exception as exc:  # noqa: BLE001 - yfinance's own handling
            YF_LOG.error(str(exc))


def swallows(*answers: Tuple[int, Any], then: Any) -> Callable[[], Any]:
    """A read of `info` whose quote requests are answered with `answers`
    ((status, body) each) and caught inside yfinance, and which then gives
    `then`."""

    def step() -> Any:
        for status, body in answers:
            _caught_request(status, body)
        return dict(then) if isinstance(then, dict) else then

    return step


def _requests_http_error(status: int) -> BaseException:
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(f"{status} Server Error", response=response)


V0265_5XX = swallows(
    (503, MAINTENANCE),
    (503, MAINTENANCE),
    then=TypeError("argument of type 'NoneType' is not iterable"),
)
V170_5XX = swallows((503, MAINTENANCE), (503, MAINTENANCE), then=NO_ENTRY)
V170_5XX_ON_THE_SUMMARY = swallows(
    (502, "<html>"), then={"longName": "Apple Inc.", **NO_ENTRY}
)
V_UNKNOWN = swallows((404, UNKNOWN_SUMMARY), then=NO_ENTRY)
V0265_UNKNOWN_EMPTY_SERIES = swallows(
    (404, UNKNOWN_SUMMARY), then=IndexError("list index out of range")
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
        fake = ticker(V170_5XX, swallows(then=FULL))
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
        ticker(swallows(then={}))
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
        """Only the calling thread's requests are read: another symbol's
        failed request, on another thread, says nothing about this answer."""

        def step() -> Any:
            other = threading.Thread(target=lambda: _caught_request(503, MAINTENANCE))
            other.start()
            other.join()
            return dict(FULL)

        ticker(step)
        assert YFinanceProvider().get_ticker_info("AAPL").sector == "Technology"
