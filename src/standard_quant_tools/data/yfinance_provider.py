import asyncio
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


class _TzLookupFailures(logging.Handler):
    """Records yfinance's "Failed to get ticker 'X' reason: ..." lines from
    THIS thread. yfinance's timezone lookup swallows its own transport
    error, logs that line, and then reports the symbol as possibly delisted
    ("no timezone found"): the line is the only trace of the real cause."""

    def __init__(self, symbol: str) -> None:
        super().__init__(level=logging.ERROR)
        self._thread = threading.get_ident()
        self._needle = f"failed to get ticker '{symbol}' reason:".lower()
        self.reasons: list = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread != self._thread:
            return
        try:
            text = record.getMessage()
        except Exception:  # noqa: BLE001 - an unformattable line says nothing
            return
        lowered = text.lower()
        if self._needle in lowered:
            self.reasons.append(text[lowered.index("reason:") + 7 :].strip())


def _yahoo_failure(exc: BaseException, tz_failures: list) -> Optional[str]:
    """
    What went wrong, when a history call failed without Yahoo answering --
    or None when the failure is Yahoo's answer about the symbol or window.
    """
    from yfinance import exceptions as yfe

    if isinstance(exc, getattr(yfe, "YFRateLimitError", ())):
        return "Yahoo rate-limited the request (HTTP 429)"
    if isinstance(exc, getattr(yfe, "YFTzMissingError", ())) and tz_failures:
        return f"the timezone lookup could not reach Yahoo ({tz_failures[-1]})"
    if isinstance(exc, getattr(yfe, "YFTickerMissingError", ())):
        status = _YAHOO_STATUS_RE.search(str(exc))
        if status and (int(status.group(1)) >= 500 or status.group(1) == "429"):
            return f"Yahoo answered HTTP {status.group(1)}"
        return None
    if _YAHOO_DOWN_RE.search(str(exc)):
        return "Yahoo Finance reported that it is down"
    if isinstance(exc, _transport_types()):
        return f"{type(exc).__name__}: {exc}"
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


#: Attempts the retry layer makes on a Yahoo history request.
_ATTEMPTS = 3


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
                audit.record_data_access(
                    symbol,
                    start_str,
                    end_str,
                    interval,
                    source="session_cache",
                    content_hash=audit.hash_dataframe(cached_df),
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
                    audit.record_data_access(
                        symbol,
                        start_str,
                        end_str,
                        interval,
                        source="disk_cache",
                        content_hash=audit.hash_dataframe(cached_df),
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
            # `_yahoo_failure`); the timezone lookup's own swallowed error is
            # read from yfinance's log.
            extra = {"raise_errors": True} if _history_raises_errors() else {}
            tz_lookup = _TzLookupFailures(symbol)
            yf_logger = logging.getLogger("yfinance")
            yf_logger.addHandler(tz_lookup)
            try:
                df = ticker.history(
                    start=start_date,
                    end=request_end,
                    interval=interval,
                    auto_adjust=True,
                    **extra,
                )
            except Exception as exc:  # noqa: BLE001 - sorted below
                cause = _yahoo_failure(exc, tz_lookup.reasons)
                if cause is not None:
                    status = re.search(r"HTTP (\d{3})", cause)
                    raise _YahooUnreachable(
                        f"yfinance could not get an answer from Yahoo for "
                        f"'{symbol}': {cause}.",
                        status=int(status.group(1)) if status else None,
                    ) from exc
                if isinstance(exc, _yahoo_answer_types()):
                    # Yahoo answered, with no prices for this symbol and
                    # window: the answer the empty frame below gives.
                    raise DataNotFoundError(
                        f"No data found for '{symbol}'. Verify symbol and date "
                        f"range. (Yahoo: {exc})"
                    ) from exc
                raise
            finally:
                yf_logger.removeHandler(tz_lookup)

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
            audit.record_data_access(
                symbol,
                start_str,
                end_str,
                interval,
                source="live_fetch",
                content_hash=audit.hash_dataframe(result),
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

    @retry(times=3, delay=1)
    def get_ticker_info(self, symbol: str) -> TickerInfo:
        if not symbol:
            raise InvalidSymbolError("Symbol cannot be empty.")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info or len(info) < 2:
                raise DataNotFoundError(f"No metadata found for '{symbol}'.")
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
        except (DataNotFoundError, InvalidSymbolError):
            raise
        except Exception as e:
            raise APIError(f"Error fetching ticker info for '{symbol}': {e}") from e

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

    @retry(times=3, delay=1)
    def get_financial_ratios(self, symbol: str) -> FinancialRatios:
        if not symbol:
            raise InvalidSymbolError("Symbol cannot be empty.")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info:
                raise DataNotFoundError(f"No financial data found for '{symbol}'.")
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
        except (DataNotFoundError, InvalidSymbolError):
            raise
        except Exception as e:
            raise APIError(f"Error fetching financials for '{symbol}': {e}") from e
