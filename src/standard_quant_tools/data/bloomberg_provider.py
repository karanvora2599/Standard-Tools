"""
Bloomberg Desktop API (DAPI) data provider — talks to a locally running,
logged-in Bloomberg Terminal via `blpapi` over `localhost:8194` by default.

Unlike YFinanceProvider, there is no API key/secret: Desktop API
authenticates via the Terminal login itself, not a credential this process
holds. What IS configurable — host/port, and only meaningful if you proxy
DAPI to a non-default address — is read from the environment (optionally via
a local `.env` file through `standard_quant_tools.config.load_env()`, or
real environment variables injected by CI/CD secrets) rather than hardcoded,
consistent with every other provider-facing config in this package
(`SQT_CACHE_DIR`, `SQT_AUDIT_DIR`, `SQT_RUNS_DIR`):

    SQT_BLOOMBERG_HOST   default "localhost"
    SQT_BLOOMBERG_PORT   default "8194"

`blpapi` is an optional dependency (`pip install standard_quant_tools[bloomberg]`)
distributed by Bloomberg, not a hard requirement of this package — every
other provider, and the package as a whole, must keep working when it isn't
installed. Constructing `BloombergProvider()` without it installed raises a
clear `APIError` explaining how to install it, the same "graceful, explicit
failure" contract `_sqt_core`'s C++ extension uses elsewhere in this codebase.

Only daily/weekly/monthly bars are supported (Bloomberg's
`HistoricalDataRequest`). Intraday intervals would require a structurally
different request (`IntradayBarRequest`, with its own history-depth limits
and event-based bar semantics) that isn't implemented here — `get_ohlcv`
raises a clear `ValidationError` for an intraday `interval` rather than
silently returning wrong data.
"""

import asyncio
import contextvars
import functools
import logging
import os
from datetime import date as _date
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from standard_quant_tools import audit
from standard_quant_tools.config import load_env
from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    InvalidSymbolError,
    ValidationError,
)

from ._retry import retry
from .base import DataProvider, FinancialRatios, TickerInfo
from .metadata import DataSetMetadata

logger = logging.getLogger(__name__)

# ── Optional blpapi dependency ────────────────────────────────────────────────
_blpapi: Any = None
HAS_BLPAPI = False
try:
    import blpapi as _blpapi  # type: ignore[import-not-found]

    HAS_BLPAPI = True
except ImportError:
    pass

_REFDATA_SERVICE = "//blp/refdata"

# interval (this package's convention, matching YFinanceProvider) ->
# Bloomberg's HistoricalDataRequest periodicitySelection. Intraday strings
# are intentionally absent — see module docstring.
_PERIODICITY = {
    "1d": "DAILY",
    "1wk": "WEEKLY",
    "1mo": "MONTHLY",
}

# Bloomberg "yellow key" market-sector suffixes get_ohlcv/get_ticker_info
# recognize as "already a fully-qualified Bloomberg ticker" — appended
# automatically otherwise (see _to_bloomberg_ticker).
_MARKET_SECTORS = (
    "Equity",
    "Govt",
    "Corp",
    "Curncy",
    "Comdty",
    "Index",
    "Mtge",
    "Muni",
    "Pfd",
)

# Bloomberg exchange/country "yellow key" -> IANA timezone. Best-effort, no
# network call — mirrors YFinanceProvider's _EXCHANGE_SUFFIX_TIMEZONES in
# spirit: not exhaustive, and a ticker with no recognized (or no) yellow key
# defaults to America/New_York (the common case: bare US tickers normalized
# to "<SYMBOL> US Equity" by _to_bloomberg_ticker).
_YELLOW_KEY_TIMEZONES = {
    "US": "America/New_York",
    "LN": "Europe/London",
    "GR": "Europe/Berlin",
    "GY": "Europe/Berlin",
    "FP": "Europe/Paris",
    "IM": "Europe/Rome",
    "NA": "Europe/Amsterdam",
    "SW": "Europe/Zurich",
    "SS": "Europe/Stockholm",
    "HK": "Asia/Hong_Kong",
    "JP": "Asia/Tokyo",
    "JT": "Asia/Tokyo",
    "CH": "Asia/Shanghai",
    "KS": "Asia/Seoul",
    "KP": "Asia/Seoul",
    "TT": "Asia/Taipei",
    "IN": "Asia/Kolkata",
    "AU": "Australia/Sydney",
    "CN": "America/Toronto",
    "BZ": "America/Sao_Paulo",
}


def _require_blpapi() -> None:
    if not HAS_BLPAPI:
        raise APIError(
            "blpapi is not installed. Bloomberg Desktop API requires Bloomberg's "
            "own Python SDK, which isn't a hard dependency of this package. "
            "Install it with `pip install standard_quant_tools[bloomberg]` "
            "(or `pip install blpapi` directly; if that fails to resolve, use "
            "Bloomberg's own package index: `pip install --index-url "
            "https://blpapi.bloomberg.com/repository/releases/python/simple/ "
            "blpapi`). A running, logged-in Bloomberg Terminal on this machine "
            "is also required — Desktop API has no separate credential."
        )


def _resolve_bloomberg_config(
    host: Optional[str] = None, port: Optional[int] = None
) -> "tuple[str, int]":
    """
    Resolve (host, port), preferring explicit constructor args, then
    SQT_BLOOMBERG_HOST/SQT_BLOOMBERG_PORT (loaded from a local .env file if
    present, or real environment variables set by CI/CD secrets), then the
    Desktop API default (localhost:8194). Pure function — independent of
    whether blpapi itself is installed, so it's testable without it.

    Raises:
        ValidationError: SQT_BLOOMBERG_PORT is set but isn't a valid integer.
    """
    load_env()
    resolved_host = host or os.environ.get("SQT_BLOOMBERG_HOST", "localhost")
    port_str = os.environ.get("SQT_BLOOMBERG_PORT")
    if port is not None:
        resolved_port = port
    elif port_str is not None:
        try:
            resolved_port = int(port_str)
        except ValueError:
            raise ValidationError(
                f"SQT_BLOOMBERG_PORT={port_str!r} is not a valid integer port."
            )
    else:
        resolved_port = 8194
    return resolved_host, resolved_port


def _to_bloomberg_ticker(symbol: str) -> str:
    """
    Normalize a plain symbol (e.g. "AAPL") to a fully-qualified Bloomberg
    ticker (e.g. "AAPL US Equity"). A symbol that already ends in a
    recognized market-sector keyword (e.g. "VOD LN Equity", "EURUSD Curncy")
    is passed through unchanged — this is a convenience heuristic for the
    common case (bare US equity tickers), not a guarantee; callers with
    non-US or non-equity instruments should pass the fully-qualified
    Bloomberg ticker themselves.
    """
    stripped = symbol.strip()
    if not stripped:
        raise InvalidSymbolError("Symbol cannot be empty.")
    last_token = stripped.rsplit(" ", 1)[-1]
    if last_token in _MARKET_SECTORS:
        return stripped
    return f"{stripped} US Equity"


def _bloomberg_timezone(bloomberg_ticker: str) -> str:
    """Best-effort IANA timezone from a fully-qualified Bloomberg ticker's
    yellow key (the token before the market sector) — see
    _YELLOW_KEY_TIMEZONES. Defaults to America/New_York."""
    parts = bloomberg_ticker.split()
    if len(parts) >= 3:
        yellow_key = parts[-2].upper()
        if yellow_key in _YELLOW_KEY_TIMEZONES:
            return _YELLOW_KEY_TIMEZONES[yellow_key]
    return "America/New_York"


def _parse_historical_bars(bars: List[Dict[str, Any]], symbol: str) -> pd.DataFrame:
    """
    Build the standard OHLCV DataFrame from already-extracted historical
    bars (each a plain dict with "date"/"PX_OPEN"/"PX_HIGH"/"PX_LOW"/
    "PX_LAST"/"PX_VOLUME" keys — see _extract_historical_bars for the
    blpapi-Element-consuming step that produces this shape). Pure function,
    independent of blpapi, so it's directly unit-testable.

    Raises:
        DataNotFoundError: bars is empty.
        APIError: a bar is missing a required field.
    """
    if not bars:
        raise DataNotFoundError(
            f"No data found for '{symbol}'. Verify symbol and date range."
        )
    required = ("date", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST")
    rows = []
    for bar in bars:
        missing = [f for f in required if bar.get(f) is None]
        if missing:
            raise APIError(
                f"Incomplete historical bar for '{symbol}' on "
                f"{bar.get('date')!r}: missing {missing}."
            )
        rows.append(
            {
                "Open": float(bar["PX_OPEN"]),
                "High": float(bar["PX_HIGH"]),
                "Low": float(bar["PX_LOW"]),
                "Close": float(bar["PX_LAST"]),
                "Volume": float(bar.get("PX_VOLUME") or 0.0),
            }
        )
    index = pd.DatetimeIndex([pd.Timestamp(bar["date"]) for bar in bars]).normalize()
    df = pd.DataFrame(rows, index=index)
    df.index.name = None
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _parse_ticker_info(symbol: str, fields: Dict[str, Any]) -> TickerInfo:
    """Map raw ReferenceDataRequest field values to TickerInfo. Pure
    function — see _extract_reference_fields for the blpapi-consuming step."""
    return TickerInfo(
        symbol=symbol,
        name=fields.get("LONG_COMP_NAME") or fields.get("NAME") or "Unknown",
        sector=fields.get("GICS_SECTOR_NAME", "Unknown"),
        industry=fields.get("GICS_INDUSTRY_NAME", "Unknown"),
        full_time_employees=fields.get("NUM_OF_EMPLOYEES"),
        city=fields.get("CIE_ADDRESS_CITY"),
        country=fields.get("CNTRY_OF_DOMICILE"),
        website=None,  # not exposed via the standard reference-data fields used here
    )


def _parse_financial_ratios(fields: Dict[str, Any]) -> FinancialRatios:
    """Map raw ReferenceDataRequest field values to FinancialRatios. Pure
    function — see _extract_reference_fields for the blpapi-consuming step."""
    return FinancialRatios(
        forward_pe=fields.get("BEST_PE_RATIO"),
        trailing_pe=fields.get("PE_RATIO"),
        price_to_book=fields.get("PX_TO_BOOK_RATIO"),
        debt_to_equity=fields.get("TOT_DEBT_TO_TOT_EQY"),
        return_on_equity=fields.get("RETURN_COM_EQY"),
        profit_margins=fields.get("PROF_MARGIN"),
        dividend_yield=fields.get("EQY_DVD_YLD_IND"),
        market_cap=fields.get("CUR_MKT_CAP"),
    )


class BloombergProvider(DataProvider):
    """
    DataProvider backed by a local Bloomberg Terminal via Desktop API.

    Args:
        host, port: Override SQT_BLOOMBERG_HOST/SQT_BLOOMBERG_PORT (and the
            localhost:8194 default) for this instance specifically.

    Raises:
        APIError: blpapi is not installed (see _require_blpapi's message
            for install instructions).
    """

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        _require_blpapi()
        self._host, self._port = _resolve_bloomberg_config(host, port)

    # ── Session plumbing ────────────────────────────────────────────────────

    def _open_session(self) -> Any:
        options = _blpapi.SessionOptions()
        options.setServerHost(self._host)
        options.setServerPort(self._port)
        session = _blpapi.Session(options)
        if not session.start():
            raise APIError(
                f"Could not start a Bloomberg session at {self._host}:{self._port} — "
                "is the Bloomberg Terminal running and logged in on this machine?"
            )
        if not session.openService(_REFDATA_SERVICE):
            session.stop()
            raise APIError(f"Could not open Bloomberg service {_REFDATA_SERVICE!r}.")
        return session

    def _send_request(self, request: Any) -> List[Any]:
        """Open a session, send one request, collect every message across
        every event until the final RESPONSE event, then close the session.
        A fresh session per request trades a little latency for never
        holding a stale/half-broken session across calls."""
        session = self._open_session()
        try:
            session.sendRequest(request)
            messages: List[Any] = []
            while True:
                event = session.nextEvent(30_000)  # 30s per-event timeout
                for msg in event:
                    messages.append(msg)
                if event.eventType() == _blpapi.Event.RESPONSE:
                    break
                if event.eventType() == _blpapi.Event.TIMEOUT:
                    raise APIError(
                        f"Bloomberg request timed out waiting for a response "
                        f"from {self._host}:{self._port}."
                    )
            return messages
        finally:
            session.stop()

    # ── Historical data ─────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval not in _PERIODICITY:
            raise ValidationError(
                f"interval={interval!r} is not supported by the Bloomberg provider "
                f"— only {sorted(_PERIODICITY)} (daily/weekly/monthly bars via "
                "HistoricalDataRequest). Intraday bars would need a separate "
                "IntradayBarRequest integration that isn't implemented here."
            )
        return self._get_ohlcv_uncached(symbol, start_date, end_date, interval)

    @retry(times=3, delay=1)
    def _get_ohlcv_uncached(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str,
    ) -> pd.DataFrame:
        bbg_ticker = _to_bloomberg_ticker(symbol)
        start_str = _to_bbg_date(start_date)
        end_str = _to_bbg_date(end_date)

        session = self._open_session()
        try:
            service = session.getService(_REFDATA_SERVICE)
            request = service.createRequest("HistoricalDataRequest")
            request.getElement("securities").appendValue(bbg_ticker)
            for field in ("PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"):
                request.getElement("fields").appendValue(field)
            request.set("periodicitySelection", _PERIODICITY[interval])
            request.set("startDate", start_str)
            request.set("endDate", end_str)
            request.set("adjustmentSplit", True)

            session.sendRequest(request)
            bars, error = _drain_historical_response(session, bbg_ticker)
        finally:
            session.stop()

        if error is not None:
            raise error

        result = _parse_historical_bars(bars, symbol)
        audit.record_data_access(
            symbol,
            start_str,
            end_str,
            interval,
            source="live_fetch",
            content_hash=audit.hash_dataframe(result),
        )
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
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(None, lambda: ctx.run(fn))  # type: ignore[arg-type]

    # ── Reference data ──────────────────────────────────────────────────────

    @retry(times=3, delay=1)
    def get_ticker_info(self, symbol: str) -> TickerInfo:
        fields = self._fetch_reference_fields(
            symbol,
            (
                "LONG_COMP_NAME",
                "NAME",
                "GICS_SECTOR_NAME",
                "GICS_INDUSTRY_NAME",
                "NUM_OF_EMPLOYEES",
                "CIE_ADDRESS_CITY",
                "CNTRY_OF_DOMICILE",
            ),
        )
        return _parse_ticker_info(symbol, fields)

    @retry(times=3, delay=1)
    def get_financial_ratios(self, symbol: str) -> FinancialRatios:
        fields = self._fetch_reference_fields(
            symbol,
            (
                "BEST_PE_RATIO",
                "PE_RATIO",
                "PX_TO_BOOK_RATIO",
                "TOT_DEBT_TO_TOT_EQY",
                "RETURN_COM_EQY",
                "PROF_MARGIN",
                "EQY_DVD_YLD_IND",
                "CUR_MKT_CAP",
            ),
        )
        return _parse_financial_ratios(fields)

    def _fetch_reference_fields(
        self, symbol: str, fields: "tuple[str, ...]"
    ) -> Dict[str, Any]:
        bbg_ticker = _to_bloomberg_ticker(symbol)
        session = self._open_session()
        try:
            service = session.getService(_REFDATA_SERVICE)
            request = service.createRequest("ReferenceDataRequest")
            request.getElement("securities").appendValue(bbg_ticker)
            for field in fields:
                request.getElement("fields").appendValue(field)

            session.sendRequest(request)
            extracted, error = _drain_reference_response(session, bbg_ticker)
        finally:
            session.stop()

        if error is not None:
            raise error
        return extracted

    def get_metadata(self, symbol: str, interval: str = "1d") -> DataSetMetadata:
        """
        Honest self-report: `adjustmentSplit=True` is always requested for
        historical bars (adjusted=True), but Desktop API makes no guarantee
        that delisted securities remain queryable (survivorship_free=False)
        or that historical values are never revised (point_in_time=False) —
        a real point-in-time/survivorship-free feed needs Bloomberg's PORT/
        enterprise data products, not plain DAPI. timezone is inferred from
        the ticker's yellow key (see _bloomberg_timezone) — a local,
        no-network heuristic, not a provider-verified exchange timezone.
        """
        bbg_ticker = _to_bloomberg_ticker(symbol)
        return DataSetMetadata(
            provider="bloomberg",
            adjusted=True,
            survivorship_free=False,
            point_in_time=False,
            frequency=interval,
            timezone=_bloomberg_timezone(bbg_ticker),
        )


def _to_bbg_date(d: Union[str, datetime, _date]) -> str:
    """YYYY-MM-DD (or datetime/date) -> Bloomberg's YYYYMMDD request format."""
    normalized = str(d)[:10]
    try:
        return datetime.strptime(normalized, "%Y-%m-%d").strftime("%Y%m%d")
    except ValueError:
        raise ValidationError(
            f"date must be in YYYY-MM-DD format, got {d!r} (normalized: {normalized!r})"
        )


def _drain_historical_response(
    session: Any, bbg_ticker: str
) -> "tuple[List[Dict[str, Any]], Optional[Exception]]":
    """
    Consume every message/event until the final RESPONSE, extracting
    HistoricalDataResponse bars into plain dicts (see _parse_historical_bars
    for the pure downstream parsing step). Returns (bars, error) — error is
    non-None on a security-level failure (e.g. invalid ticker), letting the
    caller raise it outside the session's `finally: session.stop()` block.
    """
    bars: List[Dict[str, Any]] = []
    error: Optional[Exception] = None
    while True:
        event = session.nextEvent(30_000)
        for msg in event:
            if msg.messageType() != _blpapi.Name("HistoricalDataResponse"):
                continue
            security_data = msg.getElement("securityData")
            if security_data.hasElement("securityError"):
                sec_error = security_data.getElement("securityError")
                message = sec_error.getElementAsString("message")
                error = InvalidSymbolError(
                    f"Bloomberg rejected '{bbg_ticker}': {message}"
                )
                continue
            field_data = security_data.getElement("fieldData")
            for i in range(field_data.numValues()):
                bar_element = field_data.getValueAsElement(i)
                bar: Dict[str, Any] = {
                    "date": bar_element.getElementAsDatetime("date"),
                }
                for field in ("PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"):
                    if bar_element.hasElement(field):
                        bar[field] = bar_element.getElementAsFloat(field)
                bars.append(bar)
        if event.eventType() == _blpapi.Event.RESPONSE:
            break
        if event.eventType() == _blpapi.Event.TIMEOUT:
            error = APIError(
                f"Bloomberg historical data request for '{bbg_ticker}' timed out."
            )
            break
    return bars, error


def _drain_reference_response(
    session: Any, bbg_ticker: str
) -> "tuple[Dict[str, Any], Optional[Exception]]":
    """Consume every message/event until the final RESPONSE, extracting
    ReferenceDataResponse field values into a plain dict (see
    _parse_ticker_info/_parse_financial_ratios for the pure downstream
    mapping steps). Returns (fields, error), same contract as
    _drain_historical_response."""
    fields: Dict[str, Any] = {}
    error: Optional[Exception] = None
    while True:
        event = session.nextEvent(30_000)
        for msg in event:
            if msg.messageType() != _blpapi.Name("ReferenceDataResponse"):
                continue
            security_data_array = msg.getElement("securityData")
            for i in range(security_data_array.numValues()):
                security_data = security_data_array.getValueAsElement(i)
                if security_data.hasElement("securityError"):
                    sec_error = security_data.getElement("securityError")
                    message = sec_error.getElementAsString("message")
                    error = InvalidSymbolError(
                        f"Bloomberg rejected '{bbg_ticker}': {message}"
                    )
                    continue
                field_data = security_data.getElement("fieldData")
                for j in range(field_data.numElements()):
                    element = field_data.getElement(j)
                    name = str(element.name())
                    fields[name] = (
                        element.getValueAsString()
                        if element.isNull() is False
                        else None
                    )
        if event.eventType() == _blpapi.Event.RESPONSE:
            break
        if event.eventType() == _blpapi.Event.TIMEOUT:
            error = APIError(
                f"Bloomberg reference data request for '{bbg_ticker}' timed out."
            )
            break
    if not fields and error is None:
        error = DataNotFoundError(f"No reference data found for '{bbg_ticker}'.")
    return fields, error
