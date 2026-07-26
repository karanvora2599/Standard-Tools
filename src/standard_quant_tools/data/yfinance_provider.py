import asyncio
import contextvars
import functools
import logging
import os
import re
import threading
import time
import uuid
from datetime import date as _date
from datetime import datetime
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

import pandas as pd
import yfinance as yf
from cachetools import TTLCache

from standard_quant_tools import audit
from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    InvalidSymbolError,
    ValidationError,
)

from ._retry import retry
from .base import DataProvider, FinancialRatios, TickerInfo
from .metadata import DataSetMetadata

# Permissive enough for realistic ticker formats (BRK.B, BRK/B, 0700.HK,
# ^GSPC, EURUSD=X) while rejecting ".." (parent-directory traversal),
# backslashes, drive-letter colons, and null bytes — the same slug-plus-
# resolved-containment approach artifacts.py uses for run_id/name, applied
# here since start_date/end_date/interval are just as LLM-reachable
# (get_ohlcv's own parameters) and were previously used unsanitized in the
# cache filename. A lone "/" is allowed (some tickers, e.g. BRK/B, use it)
# but still replaced with "-" before building the path, same as before.
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9./\-^=]+$")
_DATE_STR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
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

# ── In-process session cache (avoids repeated network calls in the same run) ──
_session_cache = TTLCache(maxsize=100, ttl=3600)

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

# ── Persistent Parquet disk cache ─────────────────────────────────────────────
# Historical OHLCV bars are stored permanently on disk once a date range is in
# the past. Note "historical" here means "not still forming today" — it does
# NOT mean the values are guaranteed never to change again: we fetch with
# auto_adjust=True, so a later corporate action (split, special dividend) can
# retroactively change the adjusted Close/Open/High/Low for dates that were
# already cached. This cache trades that small staleness risk for avoiding
# repeated network calls; callers who need post-corporate-action-accurate
# history for a symbol that's had a recent action should clear/bypass the
# cache (SQT_CACHE_DIR) rather than assume it self-heals.
# The cache directory can be overridden with the SQT_CACHE_DIR env variable.
_CACHE_ROOT = Path(
    os.environ.get(
        "SQT_CACHE_DIR",
        str(Path.home() / ".cache" / "standard_quant_tools" / "ohlcv"),
    )
)


def _norm_date(d: Union[str, datetime, _date]) -> str:
    """
    Normalise any date-like value to a YYYY-MM-DD string.

    Validates the result actually looks like a date rather than blindly
    truncating: a real datetime/date object always stringifies to a valid
    YYYY-MM-DD prefix, but an arbitrary caller-supplied string (start_date/
    end_date are LLM-reachable via get_ohlcv) does not, and this value
    feeds directly into the Parquet cache filename (_parquet_path) — a
    truncated-but-unvalidated string could still contain '..' or path
    separators after slicing to 10 characters.
    """
    norm = str(d)[:10]
    if not _DATE_STR_RE.match(norm):
        raise ValidationError(
            f"date must be in YYYY-MM-DD format, got {d!r} (normalized: {norm!r})"
        )
    return norm


def _normalize_ohlcv_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance attaches the listing exchange's own timezone to its returned
    index (even for daily bars) — e.g. tz-aware 'America/New_York' for a
    plain US ticker. Every downstream consumer (agent/tools.py's
    pd.Timestamp(iso_date) signal/target-weight keys, portfolio_engine.py's
    per-ticker index intersection, reindex-based signal_fill_policy, etc.)
    builds or compares against tz-naive, midnight-normalized timestamps —
    intersecting/reindexing a tz-naive index against a tz-aware one either
    raises or (via .reindex(), which doesn't raise) silently produces an
    all-NaN result, so a signal/target-weight dict keyed by plain ISO dates
    can appear to have "no matching market data" for every date, or
    silently zero out. Strip tz and drop any intraday time component here,
    once, at the single choke point every OHLCV consumer goes through,
    rather than requiring every call site to defend against it.
    """
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()
    df = df.copy()
    df.index = idx
    return df


def _parquet_path(symbol: str, start: str, end: str, interval: str) -> Path:
    """
    Build the Parquet cache path for (symbol, start, end, interval).

    All four components are LLM-reachable via get_ohlcv's own parameters,
    and used to go straight into the filename unsanitized except for a
    single "/" -> "-" replacement on symbol — start/end/interval were not
    checked at all. Validates each against an allow-list/pattern (the same
    slug-plus-resolved-containment approach artifacts.py uses for run_id/
    name) and confirms the resulting path actually resolves inside
    _CACHE_ROOT before returning it, as defense in depth.

    Raises:
        ValidationError: symbol/interval don't match their allowed pattern/
            set, or the resolved path would escape _CACHE_ROOT.
    """
    if not symbol or ".." in symbol or not _SYMBOL_RE.match(symbol):
        raise ValidationError(
            f"symbol={symbol!r} is not a valid identifier for caching — only "
            "letters, digits, '.', '/', '-', '^', '=' are allowed, and '..' "
            "is never allowed."
        )
    for value, name in ((start, "start"), (end, "end")):
        if not _DATE_STR_RE.match(value):
            raise ValidationError(
                f"{name}={value!r} must already be normalized to YYYY-MM-DD "
                "before building a cache path (call _norm_date first)."
            )
    if interval not in _VALID_INTERVALS:
        raise ValidationError(
            f"interval={interval!r} is not supported. Valid intervals: "
            f"{sorted(_VALID_INTERVALS)}"
        )
    safe = symbol.replace("/", "-").upper()
    path = _CACHE_ROOT / f"{safe}_{start}_{end}_{interval}.parquet"
    root = _CACHE_ROOT.resolve()
    resolved = path.resolve()
    # On Windows, Path.resolve() calls into GetFinalPathNameByHandle for a
    # path that actually exists on disk, which returns the "\\?\"-prefixed
    # extended-length form — but for a path that doesn't exist yet (or is
    # short enough), it's returned without that prefix. _CACHE_ROOT and the
    # full file path can therefore disagree on the prefix even though they
    # denote the same location, causing a false-positive "escapes cache
    # dir" rejection. Compare with the prefix stripped from both sides;
    # still return the real `resolved` path (the prefix is harmless to the
    # filesystem APIs that consume it).
    root_cmp = Path(str(root).removeprefix("\\\\?\\"))
    resolved_cmp = Path(str(resolved).removeprefix("\\\\?\\"))
    if not resolved_cmp.is_relative_to(root_cmp):
        raise ValidationError(
            f"resolved cache path {resolved} escapes SQT_CACHE_DIR ({root})"
        )
    return resolved


def _is_historical(end_date: Union[str, datetime, _date]) -> bool:
    """Return True when end_date is strictly before today (bar is fully formed,
    so it's eligible for the disk cache — see the cache-root comment above for
    why "historical" doesn't mean the adjusted values can never change)."""
    try:
        return _norm_date(end_date) < _date.today().isoformat()
    except Exception:
        return False


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

        start_str = _norm_date(start_date)
        end_str = _norm_date(end_date)
        # Keyed by self._instance_token (not just the call args) to match
        # the previous @cached(_session_cache) decorator's default hashkey,
        # which included self — a fresh provider instance must NOT
        # transparently reuse another instance's cached result (e.g. audit
        # replay constructs a fresh provider specifically to re-read from
        # disk/network and detect tampering; sharing the cache across
        # instances would mask that a cached Parquet file was altered after
        # the original fetch).
        cache_key = (self._instance_token, symbol, start_str, end_str, interval)

        cached_df = _session_cache.get(cache_key)
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
        _session_cache[cache_key] = result
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
        pq_path = _parquet_path(symbol, start_str, end_str, interval)
        if _is_historical(end_date) and pq_path.exists():
            try:
                cached_df = _normalize_ohlcv_index(pd.read_parquet(pq_path))
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

            result = _normalize_ohlcv_index(df[required])

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
        if _is_historical(end_date):
            try:
                _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
                # Write to a per-PID-and-thread temp file then atomically
                # replace the target so concurrent processes AND concurrent
                # threads within the same process don't collide on the same
                # temp filename (os.getpid() alone isn't unique across threads).
                tmp = pq_path.with_name(
                    f"{pq_path.stem}.{os.getpid()}.{threading.get_ident()}."
                    f"{uuid.uuid4().hex[:8]}.tmp.parquet"
                )
                result.to_parquet(tmp)
                tmp.replace(pq_path)  # atomic on all platforms
                logger.debug("[cache] disk write %s  → %s", symbol, pq_path.name)
            except Exception as cache_exc:
                logger.warning(
                    "[cache] disk write failed for %s: %s", symbol, cache_exc
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
