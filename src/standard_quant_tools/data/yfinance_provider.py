import asyncio
import contextvars
import functools
import logging
import time
import uuid
from datetime import datetime
from typing import Union

logger = logging.getLogger(__name__)

import pandas as pd
import yfinance as yf

from standard_quant_tools import audit
from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    InvalidSymbolError,
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
)
from ._retry import retry
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


class YFinanceProvider(DataProvider):

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
        that audit.record_data_access() always runs — including on a
        session-cache hit — instead of being skipped whenever the decorator
        short-circuits before the function body executes. Always returns a
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

        result = self._fetch_ohlcv_uncached(
            symbol, start_date, end_date, interval, start_str, end_str
        )
        _session_cache_set(cache_key, result)
        return result.copy()

    @retry(times=3, delay=1)
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
        if pq_path is not None and _is_historical(end_date) and pq_path.exists():
            try:
                # interval passed through: without it an intraday cache read
                # normalized every bar to midnight, so the same request
                # answered differently served from cache than served live.
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
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
            )

            if df.empty:
                raise DataNotFoundError(
                    f"No data found for '{symbol}'. Verify symbol and date range."
                )

            df.columns = [c.capitalize() for c in df.columns]
            required = ["Open", "High", "Low", "Close", "Volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise APIError(
                    f"Incomplete data from yfinance. Missing columns: {missing}"
                )
            if df["Close"].isnull().any():
                raise APIError(f"Data for {symbol} contains NaNs in Close column.")

            result = _normalize_ohlcv_index(df[required], interval)

        except (DataNotFoundError, InvalidSymbolError, APIError):
            raise
        except Exception as e:
            raise APIError(
                f"Error fetching data for '{symbol}' from yfinance: {e}"
            ) from e

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("[fetch] ✓ %s  %d rows  %.0fms", symbol, len(result), elapsed_ms)
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
            return FinancialRatios(
                forward_pe=info.get("forwardPE"),
                trailing_pe=info.get("trailingPE"),
                price_to_book=info.get("priceToBook"),
                debt_to_equity=info.get("debtToEquity"),
                return_on_equity=info.get("returnOnEquity"),
                profit_margins=info.get("profitMargins"),
                dividend_yield=info.get("dividendYield"),
                market_cap=info.get("marketCap"),
            )
        except (DataNotFoundError, InvalidSymbolError):
            raise
        except Exception as e:
            raise APIError(f"Error fetching financials for '{symbol}': {e}") from e
