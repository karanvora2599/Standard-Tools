import asyncio
import contextlib
import contextvars
import functools
import inspect
import logging
import re
import sys
import threading
import time
import uuid
import warnings
import weakref
from datetime import datetime
from typing import Optional, Union

from standard_quant_tools.data.ratios import (
    implausible_value_warnings,
    percent_to_fraction,
)

logger = logging.getLogger(__name__)

import pandas as pd
import yfinance as yf

from standard_quant_tools import audit
from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    InvalidSymbolError,
    NonRetryableAPIError,
    ValidationError,
    VendorUnavailableError,
)

from ._cache import (
    _is_historical,
    _norm_cache_bound,
    _norm_date,
    _normalize_ohlcv_index,
    _read_cached_ohlcv,
    _safe_parquet_path,
    _session_cache_get,
    _session_cache_set,
    _write_cached_ohlcv,
    inclusive_end_timestamp,
    trim_to_inclusive_end,
)
from ._retry import retry
from .bar_hygiene import (
    CME_TRADE_DATE,
    US_EQUITY,
    UTC_DAY,
    SessionClock,
    disclose_served,
    drop_unusable_closes,
    local_day,
)
from .base import DataProvider, FinancialRatios, TickerInfo
from .metadata import DataSetMetadata

_VALID_INTERVALS = frozenset(
    (
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    )
)

# ── Yahoo Finance exchange-suffix -> IANA timezone (best-effort, no network
# call) — used by get_metadata() so a non-US listing isn't silently
# mislabeled with the NYSE timezone. Not exhaustive; unsuffixed symbols
# (the common case: US-listed tickers) default to America/New_York.
_EXCHANGE_SUFFIX_TIMEZONES = {
    ".L": "Europe/London",
    ".DE": "Europe/Berlin",
    ".PA": "Europe/Paris",
    ".MI": "Europe/Rome",
    ".AS": "Europe/Amsterdam",
    ".SW": "Europe/Zurich",
    ".ST": "Europe/Stockholm",
    ".HK": "Asia/Hong_Kong",
    ".T": "Asia/Tokyo",
    ".SS": "Asia/Shanghai",
    ".SZ": "Asia/Shanghai",
    ".KS": "Asia/Seoul",
    ".TW": "Asia/Taipei",
    ".NS": "Asia/Kolkata",
    ".BO": "Asia/Kolkata",
    ".AX": "Australia/Sydney",
    ".TO": "America/Toronto",
    ".V": "America/Toronto",
    ".SA": "America/Sao_Paulo",
}

# The exchange calendar each suffix's listings trade on, consulted to decide
# whether a daily bar's session has closed. Shenzhen keeps Shanghai's hours,
# the NSE Bombay's and the TSX Venture the TSX's, and exchange_calendars
# carries the second of each pair.
_EXCHANGE_SUFFIX_CALENDARS = {
    ".L": "XLON",
    ".DE": "XETR",
    ".PA": "XPAR",
    ".MI": "XMIL",
    ".AS": "XAMS",
    ".SW": "XSWX",
    ".ST": "XSTO",
    ".HK": "XHKG",
    ".T": "XTKS",
    ".SS": "XSHG",
    ".SZ": "XSHG",
    ".KS": "XKRX",
    ".TW": "XTAI",
    ".NS": "XBOM",
    ".BO": "XBOM",
    ".AX": "XASX",
    ".TO": "XTSE",
    ".V": "XTSE",
    ".SA": "BVMF",
}

#: Yahoo's crypto pairs: a coin and a quote currency of three or more
#: letters. A share class is one letter ('BRK-B'), so the two never meet.
_CRYPTO_RE = re.compile(r"^[A-Z0-9]+-(USD|USDT|USDC|EUR|GBP|JPY|BTC|ETH)$")


def _session_clock(symbol: str) -> SessionClock:
    """
    How to tell whether a Yahoo symbol's daily bar has closed, from the
    symbol's own convention -- no network call.

    A future ('ES=F') is a CME trade date; a currency pair ('EURUSD=X') a
    whole London day, the zone Yahoo labels its FX bars in; a crypto pair
    ('BTC-USD') a whole UTC day, weekends included; a suffixed listing its
    exchange's calendar, or its local midnight without one. Everything else
    -- US stocks, ETFs and indices -- is the NYSE session, which over-flags
    a foreign index like '^N225' until New York's close rather than ever
    calling a forming bar closed.
    """
    upper = symbol.upper()
    if upper.endswith("=F"):
        return CME_TRADE_DATE
    if upper.endswith("=X"):
        return local_day("Europe/London")
    if _CRYPTO_RE.match(upper):
        return UTC_DAY
    for suffix, tz_name in _EXCHANGE_SUFFIX_TIMEZONES.items():
        if upper.endswith(suffix):
            return local_day(tz_name, _EXCHANGE_SUFFIX_CALENDARS.get(suffix))
    return US_EQUITY


# ── a failure to reach Yahoo is not "no data" ────────────────────────────
#
# yfinance swallows the error of a request that never got an answer -- a
# dropped connection, a timeout, Yahoo's maintenance page -- and returns an
# empty frame, which this provider reported as "No data found for 'X'.
# Verify symbol and date range": a network failure, told as a fact about the
# symbol. The history call is now made with `raise_errors=True`, so the
# failure arrives as itself, and each one is sorted below. A transport
# failure is retried by the shared retry layer and, when every attempt has
# failed, raised as `VendorUnavailableError`. Yahoo's own empty answer stays
# `DataNotFoundError`.

# yfinance 1.x deprecates the argument in favour of a process-wide switch,
# which would change what every other yfinance caller in the process sees.
# The argument still works; only its warning is silenced, by its exact text.
warnings.filterwarnings(
    "ignore", message=r"'raise_errors' deprecated", category=DeprecationWarning
)

#: Yahoo's own words for a service outage, as yfinance raises them.
_YAHOO_DOWN_RE = re.compile(r"yahoo! finance is currently down", re.IGNORECASE)
#: A Yahoo error status yfinance folds into "no price data found".
_YAHOO_STATUS_RE = re.compile(r"status_code\s*=\s*(\d{3})")
#: An HTTP error as yfinance's HTTP client words it: "HTTP Error 503:
#: Service Unavailable".
_YF_HTTP_ERROR_RE = re.compile(r"HTTP Error (\d{3})(?::[ ]?([A-Za-z][A-Za-z' -]*))?")


@functools.lru_cache(maxsize=1)
def _history_raises_errors() -> bool:
    """Whether the installed yfinance's history call takes `raise_errors`."""
    try:
        from yfinance.scrapers.history import PriceHistory

        return "raise_errors" in inspect.signature(PriceHistory.history).parameters
    except Exception:  # noqa: BLE001 - an unknown layout: call it as before
        return False


def _transport_types() -> tuple:
    """Exception types meaning the request got no usable answer: the standard
    library's, and the HTTP clients yfinance may use, when installed."""
    import http.client
    import json

    types: list = [
        ConnectionError,
        TimeoutError,
        http.client.HTTPException,
        json.JSONDecodeError,
    ]
    try:
        import requests

        types += [
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ]
    except ImportError:  # pragma: no cover - yfinance depends on requests
        pass
    curl = sys.modules.get("curl_cffi")
    if curl is not None:
        for name in ("CurlError",):
            if isinstance(getattr(curl, name, None), type):
                types.append(getattr(curl, name))
        exceptions = getattr(getattr(curl, "requests", None), "exceptions", None)
        for name in ("ConnectionError", "Timeout", "ChunkedEncodingError"):
            kind = getattr(exceptions, name, None)
            if isinstance(kind, type):
                types.append(kind)
    return tuple(types)


# ── one reading of an HTTP status, for the history and the info path ────

#: A request Yahoo refused: a missing or stale cookie or crumb (401), or a
#: request it will not serve (403). Not an answer about the symbol.
_REFUSED_STATUSES = frozenset({401, 403})


def _is_vendor_status(status: int) -> bool:
    """An HTTP status that is not an answer about the request: a timeout
    (408), a rate limit (429) or a server failure (5xx)."""
    return status in (408, 429) or 500 <= status <= 599


def _http_status(exc: BaseException) -> Optional[int]:
    """
    The HTTP status a raised error carries -- its response's, or the one
    its message starts with ("HTTP Error 503: ...") -- or None.

    Read before the error's type: curl_cffi's `HTTPError` is a `CurlError`,
    which is also what a dropped connection raises.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status
    found = re.match(r"\s*HTTP Error (\d{3})\b", str(exc))
    return int(found.group(1)) if found else None


def _http_text(exc: BaseException, status: int) -> str:
    """An HTTP error with `status` in words: "HTTP 503 Service Unavailable"."""
    found = _YF_HTTP_ERROR_RE.search(str(exc))
    if found and int(found.group(1)) == status:
        reason = (found.group(2) or "").strip()
    else:
        reason = str(getattr(getattr(exc, "response", None), "reason", "") or "")
    return f"HTTP {status} {reason.strip()}".strip()


# ── what yfinance's data layer handed back ───────────────────────────────
#
# yfinance catches the error of some of its own requests and carries on:
# the timezone lookup behind the history call, and the two quote requests
# behind `ticker.info`. It logs the error on its logger, and that line used
# to be this provider's only trace of it -- so a process that set
# yfinance's logger (or a parent) above ERROR, disabled it, or called
# `logging.disable` turned each of those failures back into "no data".
#
# The provider now reads them where yfinance gets them: its data layer,
# `yfinance.data.YfData`, whose `get`, `cache_get` and `get_raw_json` every
# request here goes through. The three are wrapped once per process. A
# wrapper returns what yfinance's method returns and raises what it raises;
# it records the response or the error only while a provider read is
# running in the calling context (a `ContextVar`, so a thread -- or a task
# -- records only its own requests, and a request made outside a provider
# read is not recorded). Nothing is logged, and yfinance's logger is left as
# the process configured it.

#: The record of the provider read running in this context, or None.
_YF_REQUESTS: contextvars.ContextVar = contextvars.ContextVar(
    "standard_quant_tools_yfinance_requests", default=None
)
_WATCHED_METHODS = ("get", "cache_get", "get_raw_json")
#: The wrappers installed, known by identity: an attribute could be copied
#: onto someone else's replacement by `functools.wraps`.
_WRAPPERS: "weakref.WeakSet" = weakref.WeakSet()
_WATCH_LOCK = threading.Lock()


class _YahooRequests:
    """
    What yfinance's data layer handed back during one provider read in this
    context: each `get`, `cache_get` and `get_raw_json` call, in order, as
    (method, the response -- or `get_raw_json`'s parsed JSON -- or the error
    raised).
    """

    def __init__(self) -> None:
        self.outcomes: list = []

    def add(self, method: str, outcome: object) -> None:
        self.outcomes.append((method, outcome))

    def http_errors(self, propagated: Optional[BaseException] = None) -> list:
        """
        The HTTP errors `get_raw_json` raised, as (status, "HTTP 503
        Service Unavailable") -- the errors yfinance's quote requests catch,
        log and carry on from -- leaving out `propagated`, the one that
        reached the caller.
        """
        found = []
        for method, outcome in self.outcomes:
            if method != "get_raw_json" or not isinstance(outcome, BaseException):
                continue
            if outcome is propagated:
                continue
            status = _http_status(outcome)
            if status is not None:
                found.append((status, _http_text(outcome, status)))
        return found

    def failures(self) -> list:
        """
        Every request that got no usable answer, in order, as (status or
        None, text): an error raised, an answer with a status of 400 or
        more, or a 200 whose body is not JSON (Yahoo's maintenance page).
        """
        found: list = []
        seen: set = set()
        for _method, outcome in self.outcomes:
            if id(outcome) in seen:  # `cache_get` answers through `get`
                continue
            seen.add(id(outcome))
            if isinstance(outcome, BaseException):
                status = _http_status(outcome)
                text = str(outcome) if status is None else _http_text(outcome, status)
                found.append((status, text))
                continue
            status = getattr(outcome, "status_code", None)
            if not isinstance(status, int):
                continue  # `get_raw_json`'s parsed answer
            if status >= 400:
                reason = str(getattr(outcome, "reason", "") or "").strip()
                found.append((status, f"HTTP {status} {reason}".strip()))
                continue
            try:
                outcome.json()
            except ValueError as exc:
                found.append((None, str(exc)))
            except Exception:  # noqa: BLE001 - not a response this reads
                pass
        return found

    def used_memo(self) -> bool:
        """Whether a request went through yfinance's memoizing `cache_get`."""
        return any(method == "cache_get" for method, _ in self.outcomes)


def _watched(method: str, call):
    """`call`, a `YfData` method, recording what it hands back into the
    calling context's provider read, when one is running."""

    @functools.wraps(call)
    def watched(self, *args, **kwargs):
        record = _YF_REQUESTS.get()
        if record is None:
            return call(self, *args, **kwargs)
        try:
            outcome = call(self, *args, **kwargs)
        except Exception as exc:
            record.add(method, exc)
            raise
        record.add(method, outcome)
        return outcome

    _WRAPPERS.add(watched)
    return watched


def _watch_data_layer() -> None:
    """Wrap the installed yfinance's request methods, once. Checked on every
    read, so a method someone replaced and then restored is wrapped
    again."""
    try:
        from yfinance.data import YfData
    except Exception:  # noqa: BLE001 - an unknown layout: errors raised are still read
        return

    def unwatched() -> list:
        return [
            method
            for method in _WATCHED_METHODS
            if callable(YfData.__dict__.get(method))
            and YfData.__dict__[method] not in _WRAPPERS
        ]

    if not unwatched():
        return
    with _WATCH_LOCK:
        for method in unwatched():
            setattr(YfData, method, _watched(method, YfData.__dict__[method]))


@contextlib.contextmanager
def _recording(record: _YahooRequests):
    """Record what yfinance's data layer hands back, in this context, for as
    long as the block runs."""
    _watch_data_layer()
    token = _YF_REQUESTS.set(record)
    try:
        yield record
    finally:
        _YF_REQUESTS.reset(token)


def _forget_memoized_answers(record: _YahooRequests) -> None:
    """
    Drop yfinance's memo of its answers before the retry layer asks again.

    `cache_get` -- the history request for a window that ended more than 30
    minutes ago, the timezone lookup, the info path's time series -- keeps
    each answer for the process, a failed one too: a 503 to a past window
    was served again to every later attempt and every later call, which
    never reached Yahoo. The memo holds no key-by-key eviction, so it is
    emptied; an entry dropped is fetched again when next asked for.
    """
    if not record.used_memo():
        return
    try:
        from yfinance.data import YfData

        clear = getattr(YfData.cache_get, "cache_clear", None)
        if callable(clear):
            clear()
    except Exception:  # noqa: BLE001 - nothing memoized to drop
        pass


def _yahoo_failure(
    exc: BaseException, record: Optional[_YahooRequests] = None
) -> Optional[tuple]:
    """
    (cause, status) when a request failed without an answer from Yahoo --
    or None when the failure is Yahoo's answer about the symbol or window,
    or not known to be the vendor's.

    A status is read first, whatever the error's type: a timeout (408), a
    rate limit (429) or a 5xx is the vendor's; a 404 or a refusal (401,
    403) is Yahoo's answer, left to the caller. yfinance's "no data" errors
    are the vendor's when a request behind them, recorded in `record`, got
    no usable answer.
    """
    from yfinance import exceptions as yfe

    status = _http_status(exc)
    if status is not None:
        if _is_vendor_status(status):
            return f"Yahoo answered {_http_text(exc, status)}", status
        return None
    if isinstance(exc, getattr(yfe, "YFRateLimitError", ())):
        return "Yahoo rate-limited the request (HTTP 429)", 429
    if isinstance(exc, getattr(yfe, "YFTickerMissingError", ())):
        failed = record.failures() if record is not None else []
        if failed:
            status, text = failed[0]
            if status is not None and not _is_vendor_status(status):
                return None
            if isinstance(exc, getattr(yfe, "YFTzMissingError", ())):
                return f"the timezone lookup could not reach Yahoo ({text})", status
            if status is None:
                return f"a request could not reach Yahoo ({text})", None
            return f"Yahoo answered {text}", status
        found = _YAHOO_STATUS_RE.search(str(exc))
        if found and _is_vendor_status(int(found.group(1))):
            return f"Yahoo answered HTTP {found.group(1)}", int(found.group(1))
        return None
    if _YAHOO_DOWN_RE.search(str(exc)):
        return "Yahoo Finance reported that it is down", None
    if isinstance(exc, _transport_types()):
        return f"{type(exc).__name__}: {exc}", None
    return None


def _yahoo_answer_types() -> tuple:
    """yfinance's errors for an answer Yahoo did give: no prices, no
    timezone (possibly delisted), an invalid period."""
    from yfinance import exceptions as yfe

    return tuple(
        kind
        for kind in (
            getattr(yfe, "YFTickerMissingError", None),
            getattr(yfe, "YFInvalidPeriodError", None),
        )
        if isinstance(kind, type)
    )


class _YahooUnreachable(APIError):
    """A history request that got no answer from Yahoo. An `APIError`, so
    the shared retry layer asks again; `get_ohlcv` names the last one as
    the vendor's failure once every attempt has failed."""

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


#: Attempts the retry layer makes on a Yahoo history or info request.
_ATTEMPTS = 3


# ── `ticker.info` has no `raise_errors` ──────────────────────────────────
#
# yfinance reads a symbol's info from three requests -- the quote summary,
# the quote, and a time series for one field (trailingPegRatio) -- and
# cannot be asked to raise. Stubbed at its HTTP layer, the two installed
# versions answer a failed request like this:
#
#   failure                        0.2.65                 1.7.0
#   connection error, timeout      raised                 raised
#   429                            YFRateLimitError       YFRateLimitError
#   200 with a maintenance page    JSONDecodeError        JSONDecodeError
#   5xx on both quote requests     TypeError              {'trailingPegRatio':
#                                                          None}, or the
#                                                          time series' error
#   5xx on one quote request       TypeError, or the      the other request's
#                                  other's fields         fields
#   unknown symbol (Yahoo's 404)   {'trailingPegRatio':   {'trailingPegRatio':
#                                  None}                  None}
#
# A quote request's 5xx is raised by `get_raw_json`, caught inside yfinance
# and logged as "HTTP Error 503: Service Unavailable", and under 1.7.0 the
# dict it leaves is the one Yahoo gives for a symbol it does not know. The
# error is read from the data layer as `get_raw_json` raises it (see above),
# whatever the process's logging configuration.

#: What `ticker.info` holds when Yahoo has no entry for the symbol: the
#: one field yfinance asks the time-series endpoint for, set to None.
_INFO_ONLY_FROM_TIMESERIES = frozenset({"trailingPegRatio"})


def _info_failure(exc: BaseException, caught: list) -> Optional[tuple]:
    """
    (cause, status) when an info read failed without an answer from Yahoo
    -- or None when the failure is Yahoo's answer or not known to be the
    vendor's. `caught` holds the HTTP errors yfinance caught on the way.
    """
    for status, text in caught:
        if _is_vendor_status(status):
            # 0.2.65 raises TypeError once a quote request has failed: the
            # failure is the request's, caught before it.
            return f"Yahoo answered {text}", status
    return _yahoo_failure(exc)


def _read_info(symbol: str, not_found: str, failed: str) -> tuple:
    """
    `yf.Ticker(symbol).info` and the HTTP errors yfinance caught reading
    it, as (status, text).

    Raises `_YahooUnreachable` when a request got no answer from Yahoo --
    raised, or caught and answered with what was left -- so the retry
    layer asks again; `DataNotFoundError(not_found)` when Yahoo's only
    answer was 404; and an `APIError` starting `failed` for anything else
    that went wrong, as before.
    """
    record = _YahooRequests()
    with _recording(record):
        try:
            info = yf.Ticker(symbol).info
        except Exception as exc:  # noqa: BLE001 - sorted below
            caught = record.http_errors(propagated=exc)
            failure = _info_failure(exc, caught)
            if failure is not None:
                cause, status = failure
                _forget_memoized_answers(record)
                raise _YahooUnreachable(
                    f"yfinance could not get an answer from Yahoo for '{symbol}': "
                    f"{cause}.",
                    status=status,
                ) from exc
            statuses = {status for status, _ in caught}
            raised = _http_status(exc)
            if raised is not None:
                statuses.add(raised)
            if statuses == {404}:
                # Yahoo said it has no such symbol; 0.2.65 then fails on the
                # empty answer instead of returning it.
                raise DataNotFoundError(not_found) from exc
            seen = "; ".join(dict.fromkeys(text for _, text in caught))
            raise APIError(
                f"{failed}: {exc}" + (f" (yfinance logged {seen})" if seen else "")
            ) from exc
    caught = record.http_errors()
    for status, text in caught:
        if _is_vendor_status(status):
            _forget_memoized_answers(record)
            raise _YahooUnreachable(
                f"yfinance could not get an answer from Yahoo for '{symbol}': "
                f"Yahoo answered {text}, and yfinance returned what was left "
                f"({len(info or {})} field(s)).",
                status=status,
            )
    return info, caught


def _has_no_entry(info: Optional[dict]) -> bool:
    """Whether `info` is Yahoo's answer for a symbol it does not know: empty,
    or holding only the time-series field yfinance adds to every answer."""
    return not info or set(info) <= _INFO_ONLY_FROM_TIMESERIES


def _not_found(message: str, errors: list) -> str:
    """`message`, with the status Yahoo refused a request with when that
    was not a 404: an empty answer to a refused request is not an answer
    about the symbol."""
    refused = [text for status, text in errors if status != 404]
    if not refused:
        return message
    return (
        f"{message} Yahoo answered {refused[-1]} to the request, so this may be "
        "a refused request rather than an answer about the symbol."
    )


class YFinanceProvider(DataProvider):

    SUPPORTED_INTERVALS = _VALID_INTERVALS

    def __init__(self) -> None:
        # A stable per-instance token for scoping the session cache (see
        # get_ohlcv below) — deliberately NOT id(self): CPython reuses an
        # object's id() once it's garbage collected, so two unrelated,
        # sequentially-created provider instances can end up with the exact
        # same id() if the first is freed before the second is allocated
        # (observed intermittently under full-test-suite memory churn,
        # never in isolation) — a UUID has no such collision risk regardless
        # of allocator behavior.
        self._instance_token = uuid.uuid4()

    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Public entry point. Checks the in-memory session cache itself
        (rather than via a @cached decorator wrapping this whole method) so
        that an open decision record hears of every access — including a
        session-cache hit — instead of missing it whenever the decorator
        short-circuits before the function body executes. Outside a record
        nothing is reported, and the frame is not hashed. Always returns a
        fresh .copy() so a caller mutating the result in place can't corrupt
        the cached object shared with every other caller.
        """
        if not symbol or not isinstance(symbol, str):
            raise InvalidSymbolError(f"Invalid symbol: {symbol}")
        if interval not in _VALID_INTERVALS:
            raise ValidationError(
                f"interval={interval!r} is not supported. Valid intervals: "
                f"{sorted(_VALID_INTERVALS)}"
            )

        # Interval-aware: an intraday request keeps time-of-day in its cache
        # identity, so 09:30->12:00 and 13:00->16:00 on the same day no
        # longer resolve to one file (the second silently serving the
        # first's bars). Daily and coarser produce the same YYYY-MM-DD token
        # as before, so existing cache files stay valid.
        start_str = _norm_cache_bound(start_date, interval)
        end_str = _norm_cache_bound(end_date, interval)
        # Keyed by self._instance_token (not just the call args) to match
        # the previous @cached(_session_cache) decorator's default hashkey,
        # which included self — a fresh provider instance must NOT
        # transparently reuse another instance's cached result (e.g. audit
        # replay constructs a fresh provider specifically to re-read from
        # disk/network and detect tampering; sharing the cache across
        # instances would mask that a cached Parquet file was altered after
        # the original fetch). "yfinance" is included explicitly (rather
        # than relying on it being the only provider using this cache) so
        # the invariant that no two providers can collide on the same entry
        # is visible at every call site, not just in _cache.py's docstring.
        cache_key = (
            "yfinance",
            self._instance_token,
            symbol,
            start_str,
            end_str,
            interval,
        )

        clock = _session_clock(symbol)
        cached_df = _session_cache_get(cache_key)
        if cached_df is not None:
            if audit.recording_data_access():
                audit.record_frame_access(
                    symbol,
                    start_str,
                    end_str,
                    interval,
                    source="session_cache",
                    frame=cached_df,
                )
            return disclose_served(cached_df.copy(), symbol, interval, clock)

        try:
            result = self._fetch_ohlcv_uncached(
                symbol, start_date, end_date, interval, start_str, end_str
            )
        except _YahooUnreachable as exc:
            # Every attempt failed without an answer from Yahoo: the
            # vendor's failure, said as one, never "no data found".
            raise VendorUnavailableError(
                f"Yahoo Finance failed on its side for {symbol!r} ({interval}, "
                f"{start_str} to {end_str}) on all {_ATTEMPTS} attempts: {exc} "
                "This is a network or service failure, not an answer about the "
                "symbol or the window -- it does not mean the symbol is wrong "
                "or has no data. Asking again later may succeed, or ask another "
                "provider (source=...).",
                status=exc.status,
                dataset=None,
                original_exception=exc,
            ) from exc
        _session_cache_set(cache_key, result, end=end_str)
        # Judged on the copy handed out, at the moment it is handed out: a
        # window served from the session cache a minute later may have
        # closed in between.
        return disclose_served(result.copy(), symbol, interval, clock)

    @retry(times=_ATTEMPTS, delay=1)
    def _fetch_ohlcv_uncached(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str,
        start_str: str,
        end_str: str,
    ) -> pd.DataFrame:
        # ── Parquet disk cache (historical ranges only) ────────────────────
        pq_path = _safe_parquet_path(
            symbol, start_str, end_str, interval, provider="yfinance"
        )
        if pq_path is not None and _is_historical(end_date):
            # The shared read: the same column, null-Close and window checks
            # this method makes on a live answer below, and a Windows
            # sharing violation treated as a miss rather than as corruption.
            # interval passed through: without it an intraday cache read
            # normalized every bar to midnight, so the same request
            # answered differently served from cache than served live.
            cached_df = _read_cached_ohlcv(pq_path, interval, start_str, end_str)
            if cached_df is not None:
                logger.debug(
                    "[cache] disk hit  %s  %s → %s  (%s)",
                    symbol,
                    start_str,
                    end_str,
                    pq_path.name,
                )
                if audit.recording_data_access():
                    audit.record_frame_access(
                        symbol,
                        start_str,
                        end_str,
                        interval,
                        source="disk_cache",
                        frame=cached_df,
                    )
                return cached_df

        # ── Fetch from yfinance ────────────────────────────────────────────
        logger.debug(
            "[fetch] yfinance   %s  %s → %s  interval=%s",
            symbol,
            start_str,
            end_str,
            interval,
        )
        t0 = time.perf_counter()
        try:
            ticker = yf.Ticker(symbol)
            # yfinance's `end` is EXCLUSIVE, but this package's get_ohlcv
            # contract defines end_date as an INCLUSIVE observation cutoff
            # (data/base.py) -- the semantics Polygon's aggregates `to` and
            # Bloomberg's `endDate` already had natively. Passing end_date
            # straight through silently dropped its final bar, so
            # get_ohlcv(end="2023-01-01") stopped at Dec 31 here while the
            # other two providers included Jan 1, and score_model(as_of=X)
            # excluded X while still reporting X as the as-of date.
            #
            # Request an exclusive bound past the whole inclusive window,
            # then trim below. Over-fetching at most one day is harmless;
            # under-fetching silently changes the answer.
            request_end = (
                inclusive_end_timestamp(end_date, interval).normalize()
                + pd.Timedelta(days=1)
            ).to_pydatetime()
            # Errors raised, not swallowed into an empty frame (see
            # `_yahoo_failure`); the requests behind yfinance's own "no data"
            # -- the timezone lookup's error, which yfinance swallows, and
            # the status of each answer -- are read from its data layer.
            extra = {"raise_errors": True} if _history_raises_errors() else {}
            record = _YahooRequests()
            with _recording(record):
                try:
                    df = ticker.history(
                        start=start_date,
                        end=request_end,
                        interval=interval,
                        auto_adjust=True,
                        **extra,
                    )
                except Exception as exc:  # noqa: BLE001 - sorted below
                    failure = _yahoo_failure(exc, record)
                    if failure is not None:
                        cause, status = failure
                        _forget_memoized_answers(record)
                        raise _YahooUnreachable(
                            f"yfinance could not get an answer from Yahoo for "
                            f"'{symbol}': {cause}.",
                            status=status,
                        ) from exc
                    if isinstance(exc, _yahoo_answer_types()) or (
                        _http_status(exc) == 404
                    ):
                        # Yahoo answered, with no prices for this symbol and
                        # window: the answer the empty frame below gives. A
                        # refused request (401, 403) is said to be one.
                        refused = [
                            (status, text)
                            for status, text in record.failures()
                            if status in _REFUSED_STATUSES
                        ]
                        raise DataNotFoundError(
                            _not_found(
                                f"No data found for '{symbol}'. Verify symbol and "
                                f"date range. (Yahoo: {exc})",
                                refused[:1],
                            )
                        ) from exc
                    raise

            if df.empty:
                raise DataNotFoundError(
                    f"No data found for '{symbol}'. Verify symbol and date range."
                )

            df.columns = [c.capitalize() for c in df.columns]
            required = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                # The shape of yfinance's answer, which a re-fetch of the
                # same window reproduces: refused once, not three times.
                raise NonRetryableAPIError(
                    f"Incomplete data from yfinance for {symbol}: missing "
                    f"columns {missing} (got {list(df.columns)}). Asking again "
                    "returns the same frame; try another provider (source=...) "
                    "or check that the symbol is a priced instrument."
                )

            # Trim AFTER normalization so the comparison happens in the same
            # tz-naive space the index was just converted into.
            result = trim_to_inclusive_end(
                _normalize_ohlcv_index(df[required], interval), end_date, interval
            )
            # A row with no Close -- the placeholder yfinance appends outside
            # market hours for the next session, or a hole -- is dropped and
            # disclosed rather than refusing the whole series (see the
            # CHANGELOG entry of 2026-10-01). After the trim, so a row past
            # the window is not reported as dropped from it.
            result = drop_unusable_closes(result, symbol, provider="yfinance")

        except (DataNotFoundError, InvalidSymbolError, APIError):
            raise
        except Exception as e:
            raise APIError(
                f"Error fetching data for '{symbol}' from yfinance: {e}"
            ) from e

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("[fetch] ✓ %s  %d rows  %.0fms", symbol, len(result), elapsed_ms)
        if audit.recording_data_access():
            audit.record_frame_access(
                symbol,
                start_str,
                end_str,
                interval,
                source="live_fetch",
                frame=result,
            )

        # ── Persist to Parquet for future sessions ─────────────────────────
        if pq_path is not None and _is_historical(end_date):
            _write_cached_ohlcv(pq_path, result, interval, start_str, end_str)

        return result

    async def get_ohlcv_async(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
    ) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        fn = functools.partial(self.get_ohlcv, symbol, start_date, end_date, interval)
        # run_in_executor (unlike call_soon) does not copy the calling context
        # into the worker thread, so without this the audit contextvars
        # (request id, data-source collector) would silently no-op for every
        # fetch made this way — e.g. the per-ticker legs of a portfolio call.
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(None, lambda: ctx.run(fn))  # type: ignore[arg-type]

    @retry(times=_ATTEMPTS, delay=1)
    def _info_attempts(self, symbol: str, not_found: str, failed: str) -> tuple:
        return _read_info(symbol, not_found, failed)

    def _info(self, symbol: str, what: str, not_found: str, failed: str) -> tuple:
        """`ticker.info` after the retry layer's attempts, with a failure to
        reach Yahoo named as the vendor's (see `_read_info`)."""
        try:
            return self._info_attempts(symbol, not_found, failed)
        except _YahooUnreachable as exc:
            raise VendorUnavailableError(
                f"Yahoo Finance failed on its side for {symbol!r} ({what}) on "
                f"all {_ATTEMPTS} attempts: {exc} This is a network or service "
                "failure, not an answer about the symbol -- it does not mean the "
                f"symbol is wrong or has no {what}. Asking again later may "
                "succeed, or ask another provider (source=...).",
                status=exc.status,
                dataset=None,
                original_exception=exc,
            ) from exc

    def get_ticker_info(self, symbol: str) -> TickerInfo:
        if not symbol:
            raise InvalidSymbolError("Symbol cannot be empty.")
        not_found = f"No metadata found for '{symbol}'."
        failed = f"Error fetching ticker info for '{symbol}'"
        info, caught = self._info(symbol, "ticker info", not_found, failed)
        if _has_no_entry(info) or len(info) < 2:
            raise DataNotFoundError(_not_found(not_found, caught))
        try:
            return TickerInfo(
                symbol=symbol,
                name=info.get("longName", "Unknown"),
                sector=info.get("sector", "Unknown"),
                industry=info.get("industry", "Unknown"),
                full_time_employees=info.get("fullTimeEmployees"),
                city=info.get("city"),
                country=info.get("country"),
                website=info.get("website"),
            )
        except Exception as e:
            raise APIError(f"{failed}: {e}") from e

    def get_metadata(self, symbol: str, interval: str = "1d") -> DataSetMetadata:
        """
        Honest self-report, not aspirational: yfinance auto-adjusts prices
        by default (adjusted=True), but makes no guarantee that delisted
        securities remain queryable (survivorship_free=False) or that
        historical values are never silently revised
        (point_in_time=False) — neither is a yfinance API contract.
        timezone is inferred from the symbol's Yahoo Finance exchange suffix
        (e.g. "SAP.DE" -> Europe/Berlin, "0700.HK" -> Asia/Hong_Kong) via
        _EXCHANGE_SUFFIX_TIMEZONES when present, so a non-US listing isn't
        silently mislabeled with the NYSE timezone. Unsuffixed symbols (the
        common case: US-listed tickers) default to America/New_York. This is
        a local, no-network heuristic based on ticker convention, not a
        provider-verified exchange timezone — yfinance doesn't expose a
        reliable per-symbol timezone through this provider's interface.
        """
        timezone = "America/New_York"
        upper_symbol = symbol.upper()
        for suffix, tz_name in _EXCHANGE_SUFFIX_TIMEZONES.items():
            if upper_symbol.endswith(suffix):
                timezone = tz_name
                break
        return DataSetMetadata(
            provider="yfinance",
            adjusted=True,
            survivorship_free=False,
            point_in_time=False,
            frequency=interval,
            timezone=timezone,
            notes=[
                f"The index is naive: daily bars are session dates in {timezone}, "
                "intraday bars UTC instants with the zone stripped. Localising it "
                "to the zone above before a join aligns nothing."
            ],
        )

    def get_financial_ratios(self, symbol: str) -> FinancialRatios:
        if not symbol:
            raise InvalidSymbolError("Symbol cannot be empty.")
        not_found = f"No financial data found for '{symbol}'."
        failed = f"Error fetching financials for '{symbol}'"
        info, caught = self._info(symbol, "financial data", not_found, failed)
        # Yahoo's answer for a symbol it does not know holds one field set
        # to None, which used to pass as financial data with every ratio
        # missing.
        if _has_no_entry(info):
            raise DataNotFoundError(_not_found(not_found, caught))
        try:
            # yfinance's units, field by field. debtToEquity is the one
            # that differs from this package's canonical unit: yfinance
            # reports it as a PERCENTAGE (150.5 meaning debt is 1.505x
            # equity), while every other provider and every consumer here
            # expects a plain ratio. Left unconverted it made a
            # `debt_equity_max=2.0` screen admit essentially the entire
            # universe.
            #
            # returnOnEquity and profitMargins are already decimal fractions
            # in yfinance and are passed through unchanged.
            ratios = FinancialRatios(
                forward_pe=info.get("forwardPE"),
                trailing_pe=info.get("trailingPE"),
                price_to_book=info.get("priceToBook"),
                debt_to_equity=percent_to_fraction(info.get("debtToEquity")),
                return_on_equity=info.get("returnOnEquity"),
                profit_margins=info.get("profitMargins"),
                dividend_yield=info.get("dividendYield"),
                market_cap=info.get("marketCap"),
            )
            # yfinance changed dividendYield from a fraction to a percentage
            # between releases, so which one arrives depends on the installed
            # version rather than on anything this package controls. Reported
            # rather than auto-corrected: silently rescaling would rewrite a
            # genuine outlier, and a warning names the problem where a wrong
            # number would not.
            for warning in implausible_value_warnings(ratios):
                logger.warning("[yfinance:%s] %s", symbol, warning)
            return ratios
        except Exception as e:
            raise APIError(f"{failed}: {e}") from e
