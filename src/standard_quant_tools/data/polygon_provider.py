"""
Polygon.io REST API data provider — https://polygon.io.

Unlike BloombergProvider, there is no vendor SDK to install: Polygon is a
plain HTTPS/JSON REST API, so this module talks to it directly via the
standard library's `urllib.request` rather than pulling in a new optional
dependency (`polygon-api-client` or otherwise). The only thing required to
use it is an API key.

Auth is a bearer-style API key, read from the environment (optionally via a
local `.env` file through `standard_quant_tools.config.load_env()`, or a
real environment variable injected by CI/CD secrets) — the same convention
every other provider-facing secret in this package uses:

    SQT_POLYGON_API_KEY   (no default — required)

Get a free key at https://polygon.io/dashboard/api-keys. The free tier is
rate-limited (5 requests/minute at the time of writing) and daily-bars-only
for most endpoints; this module doesn't special-case that beyond the shared
`retry` decorator's generic backoff — a 429 is treated like any other
transient `APIError`.

Scope, stated explicitly (same "honest, not aspirational" spirit as
BloombergProvider):
- Only plain equity tickers are exercised end-to-end here (crypto `X:`/forex
  `C:` prefixes may work against the same aggs endpoint but are untested).
- `get_ohlcv` uses the Aggregates (Bars) endpoint with a single page
  (`limit=50000`). A request whose true result set exceeds one page (mostly
  a risk for long intraday ranges) is NOT paginated — the response's
  `next_url` presence is detected and logged as a warning so truncation is
  visible rather than silent, but the remaining pages are not fetched here.
- `get_financial_ratios` has no direct analogue to yfinance's `.info`
  ratios in Polygon's reference data. `market_cap` comes straight from
  Ticker Details v3. The rest (`trailing_pe`, `price_to_book`,
  `debt_to_equity`, `return_on_equity`, `profit_margins`) are derived from
  the most recent filing on the Financials vX endpoint combined with
  `market_cap` (e.g. `trailing_pe ~= market_cap / net_income`, avoiding a
  separate per-share price fetch since both market_cap and net_income scale
  with share count). `forward_pe` (no forward estimates in this data) and
  `dividend_yield` (would need a separate dividends-history aggregation)
  are always `None` — missing, not wrong.
- Same two-tier cache as YFinanceProvider (in-memory session TTL cache +
  persistent Parquet disk cache), sharing the hardened implementation in
  `data/_cache.py` rather than reaching Polygon on every call — a real
  concern given the free tier's 5-requests/minute limit.
"""

import asyncio
import contextvars
import functools
import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote as _urlquote
from urllib.parse import urlencode

import pandas as pd

from standard_quant_tools import audit
from standard_quant_tools.config import load_env
from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    InvalidSymbolError,
    NonRetryableAPIError,
    ValidationError,
)

from ._cache import (
    _is_historical,
    _norm_cache_bound,
    _norm_date,
    _normalize_ohlcv_index,
    _safe_parquet_path,
    _session_cache_get,
    _session_cache_set,
    _write_parquet_atomic,
    trim_to_inclusive_end,
)
from ._retry import retry
from .base import DataProvider, FinancialRatios, TickerInfo
from .metadata import DataSetMetadata

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.polygon.io"

# interval (this package's convention, matching YFinanceProvider) -> Polygon
# Aggregates (multiplier, timespan). Only the subset Polygon's aggs endpoint
# supports natively is listed; anything else raises ValidationError rather
# than silently guessing a mapping.
_TIMESPAN_MAP = {
    "1m": (1, "minute"),
    "5m": (5, "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "60m": (1, "hour"),
    "1h": (1, "hour"),
    "1d": (1, "day"),
    "1wk": (1, "week"),
    "1mo": (1, "month"),
    "3mo": (3, "month"),
}

# Timespans whose bars represent a full calendar day (or coarser) — their
# timestamps are normalized to a tz-naive midnight date. Anything finer
# (minute/hour) keeps its intraday time component, converted to US/Eastern
# (Polygon's primary market coverage) then stripped of tzinfo, matching the
# convention YFinanceProvider's _normalize_ohlcv_index establishes for every
# other provider in this package.
_DAY_OR_COARSER = frozenset({"day", "week", "month"})


def _resolve_polygon_api_key(api_key: Optional[str] = None) -> str:
    """
    Resolve the Polygon API key, preferring an explicit constructor arg,
    then SQT_POLYGON_API_KEY (loaded from a local .env file if present, or a
    real environment variable set by CI/CD secrets).

    Raises:
        APIError: no key was found anywhere.
    """
    load_env()
    resolved = api_key or os.environ.get("SQT_POLYGON_API_KEY")
    if not resolved:
        raise APIError(
            "No Polygon.io API key found. Pass api_key= explicitly, set "
            "SQT_POLYGON_API_KEY (via a local .env file or a real "
            "environment variable), or get a free key at "
            "https://polygon.io/dashboard/api-keys."
        )
    return resolved


def _polygon_get(path: str, params: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    GET one Polygon.io endpoint and return the parsed JSON body.

    Pure side effect (network), deliberately kept as a single seam so tests
    can monkeypatch `urllib.request.urlopen` without touching the parsing
    logic. The API key is appended to the query string (Polygon's own auth
    convention) but never logged.

    Raises:
        APIError: non-2xx HTTP response other than 404, invalid JSON, or a
            body-level {"status": "ERROR", ...} payload.
        DataNotFoundError: HTTP 404.
    """
    query = dict(params)
    query["apiKey"] = api_key
    url = f"{_BASE_URL}{path}?{urlencode(query)}"
    logger.debug("[fetch] polygon  GET %s%s", _BASE_URL, path)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        if e.code == 404:
            raise DataNotFoundError(
                f"Polygon.io returned 404 for {path}: {detail}"
            ) from e
        if e.code in (401, 403):
            # Permanent -- an invalid/expired API key will never succeed no
            # matter how many times it's retried, unlike 429/5xx below.
            raise NonRetryableAPIError(
                f"Polygon.io rejected the request (HTTP {e.code}) — check "
                f"SQT_POLYGON_API_KEY is valid: {detail}"
            ) from e
        if e.code == 429:
            raise APIError(
                f"Polygon.io rate limit exceeded (HTTP 429): {detail}"
            ) from e
        raise APIError(f"Polygon.io HTTP {e.code} error for {path}: {detail}") from e
    except urllib.error.URLError as e:
        raise APIError(f"Polygon.io request failed for {path}: {e}") from e

    try:
        payload: Dict[str, Any] = json.loads(body)
    except json.JSONDecodeError as e:
        raise APIError(f"Polygon.io returned invalid JSON for {path}: {e}") from e

    if payload.get("status") == "ERROR":
        raise APIError(
            f"Polygon.io error for {path}: {payload.get('error') or payload}"
        )
    return payload


def _parse_aggs(
    results: List[Dict[str, Any]], symbol: str, timespan: str
) -> pd.DataFrame:
    """
    Build the standard OHLCV DataFrame from a Polygon Aggregates response's
    `results` list (each a dict with "o"/"h"/"l"/"c"/"v"/"t" keys). Pure
    function, independent of the network call that produced it.

    Raises:
        APIError: a bar is missing a required field.
    """
    rows = []
    index = []
    for bar in results:
        required = ("o", "h", "l", "c", "t")
        missing = [f for f in required if bar.get(f) is None]
        if missing:
            raise APIError(
                f"Incomplete aggregate bar for '{symbol}': missing {missing}."
            )
        rows.append(
            {
                "Open": float(bar["o"]),
                "High": float(bar["h"]),
                "Low": float(bar["l"]),
                "Close": float(bar["c"]),
                "Volume": float(bar.get("v") or 0.0),
            }
        )
        ts = pd.Timestamp(bar["t"], unit="ms", tz="UTC")
        if timespan in _DAY_OR_COARSER:
            index.append(ts.tz_localize(None).normalize())
        else:
            index.append(ts.tz_convert("America/New_York").tz_localize(None))
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(index))
    df.index.name = None
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _parse_ticker_info(symbol: str, fields: Dict[str, Any]) -> TickerInfo:
    """Map a Polygon Ticker Details v3 `results` object to TickerInfo. Pure
    function. Polygon's reference data has one classification field
    (`sic_description`) rather than separate sector/industry taxonomies, so
    both fields fall back to it — a known coarser mapping than yfinance's."""
    address = fields.get("address") or {}
    sic_description = fields.get("sic_description")
    return TickerInfo(
        symbol=symbol,
        name=fields.get("name") or "Unknown",
        sector=sic_description or "Unknown",
        industry=sic_description or "Unknown",
        full_time_employees=fields.get("total_employees"),
        city=address.get("city"),
        country=address.get("country")
        or ("US" if fields.get("locale") == "us" else None),
        website=fields.get("homepage_url"),
    )


def _financial_value(section: Dict[str, Any], key: str) -> Optional[float]:
    """Extract a numeric `.value` from one field of a Polygon financials
    statement section (each field is `{"value": ..., "unit": ..., ...}`),
    or None if the field/value is absent."""
    entry = section.get(key)
    if not entry or entry.get("value") is None:
        return None
    return float(entry["value"])


def _parse_financial_ratios(
    details: Dict[str, Any], financials: Dict[str, Any]
) -> FinancialRatios:
    """
    Derive FinancialRatios from a Ticker Details v3 `results` object plus
    the most recent Financials vX filing's `financials` object. Pure
    function — see module docstring for exactly which ratios are computed
    vs. always None, and why.
    """
    income = financials.get("income_statement") or {}
    balance = financials.get("balance_sheet") or {}

    net_income = _financial_value(income, "net_income_loss")
    revenues = _financial_value(income, "revenues")
    equity = _financial_value(balance, "equity")
    liabilities = _financial_value(balance, "liabilities")
    market_cap = details.get("market_cap")

    trailing_pe = None
    if market_cap and net_income:
        trailing_pe = market_cap / net_income

    price_to_book = None
    if market_cap and equity:
        price_to_book = market_cap / equity

    debt_to_equity = None
    if liabilities is not None and equity:
        debt_to_equity = liabilities / equity

    return_on_equity = None
    if net_income is not None and equity:
        return_on_equity = net_income / equity

    profit_margins = None
    if net_income is not None and revenues:
        profit_margins = net_income / revenues

    return FinancialRatios(
        forward_pe=None,
        trailing_pe=trailing_pe,
        price_to_book=price_to_book,
        debt_to_equity=debt_to_equity,
        return_on_equity=return_on_equity,
        profit_margins=profit_margins,
        dividend_yield=None,
        market_cap=int(market_cap) if market_cap else None,
    )


class PolygonProvider(DataProvider):
    """
    DataProvider backed by the Polygon.io REST API.

    Args:
        api_key: Overrides SQT_POLYGON_API_KEY for this instance specifically.

    Raises:
        APIError: no API key found anywhere (see _resolve_polygon_api_key).
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = _resolve_polygon_api_key(api_key)
        # Stable per-instance token for the session cache -- deliberately
        # NOT id(self): CPython reuses an object's id() once it's garbage
        # collected, so two unrelated, sequentially-created instances could
        # collide on it (see YFinanceProvider.__init__ for the original
        # fix this mirrors).
        self._instance_token = uuid.uuid4()

    # ── Historical data ─────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
    ) -> pd.DataFrame:
        if not symbol or not isinstance(symbol, str):
            raise InvalidSymbolError(f"Invalid symbol: {symbol}")
        if interval not in _TIMESPAN_MAP:
            raise ValidationError(
                f"interval={interval!r} is not supported by the Polygon provider "
                f"— only {sorted(_TIMESPAN_MAP)} (Aggregates/Bars endpoint)."
            )
        # Interval-aware cache identity -- see YFinanceProvider for why a
        # blanket date truncation collided two different intraday ranges.
        start_str = _norm_cache_bound(start_date, interval)
        end_str = _norm_cache_bound(end_date, interval)

        # "polygon" is included explicitly in the cache key so this can
        # never collide with another provider's entry for the "same"
        # symbol/date/interval -- different providers can have different
        # adjustment conventions or data revisions.
        cache_key = (
            "polygon",
            self._instance_token,
            symbol,
            start_str,
            end_str,
            interval,
        )
        cached_df = _session_cache_get(cache_key)
        if cached_df is not None:
            audit.record_data_access(
                symbol,
                start_str,
                end_str,
                interval,
                source="session_cache",
                content_hash=audit.hash_dataframe(cached_df),
            )
            return cached_df.copy()

        result = self._get_ohlcv_uncached(symbol, start_str, end_str, interval)
        _session_cache_set(cache_key, result)
        return result.copy()

    @retry(times=3, delay=1)
    def _get_ohlcv_uncached(
        self, symbol: str, start_str: str, end_str: str, interval: str
    ) -> pd.DataFrame:
        # ── Parquet disk cache (historical ranges only) ────────────────────
        pq_path = _safe_parquet_path(
            symbol, start_str, end_str, interval, provider="polygon"
        )
        if pq_path is not None and _is_historical(end_str) and pq_path.exists():
            try:
                # interval passed through: Polygon's live _parse_aggs already
                # preserves intraday timestamps, so normalizing them to
                # midnight here made cached and uncached reads of the SAME
                # request disagree — a cache/non-cache parity bug, not just
                # a formatting difference.
                cached_df = _normalize_ohlcv_index(pd.read_parquet(pq_path), interval)
            except Exception as exc:
                logger.warning(
                    "[cache] disk read failed for %s (%s) — evicting and "
                    "refetching: %s",
                    symbol,
                    pq_path.name,
                    exc,
                )
                pq_path.unlink(missing_ok=True)
            else:
                logger.debug(
                    "[cache] disk hit  %s  %s → %s  (%s)",
                    symbol,
                    start_str,
                    end_str,
                    pq_path.name,
                )
                audit.record_data_access(
                    symbol,
                    start_str,
                    end_str,
                    interval,
                    source="disk_cache",
                    content_hash=audit.hash_dataframe(cached_df),
                )
                return cached_df

        multiplier, timespan = _TIMESPAN_MAP[interval]
        ticker = _urlquote(symbol.strip().upper(), safe=":")
        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
        payload = _polygon_get(
            path,
            {"adjusted": "true", "sort": "asc", "limit": 50000},
            self._api_key,
        )
        results = payload.get("results") or []
        if not results:
            raise DataNotFoundError(
                f"No data found for '{symbol}' from Polygon.io. Verify symbol "
                "and date range."
            )
        if payload.get("next_url"):
            logger.warning(
                "[fetch] polygon  %s returned more results than one page "
                "(limit=50000) — truncating; pagination isn't implemented",
                symbol,
            )
        # Polygon's aggregates `to` is already inclusive, so this is a
        # no-op on a well-behaved response. Applied anyway so the
        # inclusive-end contract holds by CONSTRUCTION for every provider
        # rather than by trusting each vendor's documented boundary -- a
        # vendor changing or mis-documenting its own semantics then cannot
        # silently move the window.
        result = trim_to_inclusive_end(
            _parse_aggs(results, symbol, timespan), end_str, interval
        )
        audit.record_data_access(
            symbol,
            start_str,
            end_str,
            interval,
            source="live_fetch",
            content_hash=audit.hash_dataframe(result),
        )

        if pq_path is not None and _is_historical(end_str):
            _write_parquet_atomic(pq_path, result)

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
        if not symbol:
            raise InvalidSymbolError("Symbol cannot be empty.")
        details = self._fetch_ticker_details(symbol)
        return _parse_ticker_info(symbol, details)

    @retry(times=3, delay=1)
    def get_financial_ratios(self, symbol: str) -> FinancialRatios:
        if not symbol:
            raise InvalidSymbolError("Symbol cannot be empty.")
        details = self._fetch_ticker_details(symbol)
        financials_payload = _polygon_get(
            "/vX/reference/financials",
            {
                "ticker": symbol.strip().upper(),
                "limit": 1,
                "sort": "filing_date",
                "order": "desc",
            },
            self._api_key,
        )
        financials_results = financials_payload.get("results") or []
        financials = (
            financials_results[0].get("financials", {}) if financials_results else {}
        )
        return _parse_financial_ratios(details, financials)

    def _fetch_ticker_details(self, symbol: str) -> Dict[str, Any]:
        ticker = _urlquote(symbol.strip().upper(), safe=":")
        payload = _polygon_get(f"/v3/reference/tickers/{ticker}", {}, self._api_key)
        details = payload.get("results")
        if not details:
            raise DataNotFoundError(
                f"No ticker details found for '{symbol}' from Polygon.io."
            )
        return details

    def get_metadata(self, symbol: str, interval: str = "1d") -> DataSetMetadata:
        """
        Honest self-report: `adjusted=True` because every `get_ohlcv` call
        requests `adjusted=true`, but Polygon makes no guarantee in this
        provider's scope that a delisted security remains queryable
        (survivorship_free=False) or that financials/aggregates are never
        restated after the fact (point_in_time=False). timezone defaults to
        America/New_York — Polygon's primary coverage is US exchanges, and
        this provider doesn't do a per-symbol exchange lookup.
        """
        return DataSetMetadata(
            provider="polygon",
            adjusted=True,
            survivorship_free=False,
            point_in_time=False,
            frequency=interval,
            timezone="America/New_York",
        )
