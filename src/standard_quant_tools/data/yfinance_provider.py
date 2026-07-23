import asyncio
import contextvars
import functools
import logging
import os
import time
from datetime import date as _date, datetime
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

import pandas as pd
import yfinance as yf

from standard_quant_tools import audit
from .base import DataProvider, TickerInfo, FinancialRatios
from .metadata import DataSetMetadata
from standard_quant_tools.error import APIError, DataNotFoundError, InvalidSymbolError
from cachetools import TTLCache, cached

# ── In-process session cache (avoids repeated network calls in the same run) ──
_session_cache = TTLCache(maxsize=100, ttl=3600)

# ── Persistent Parquet disk cache ─────────────────────────────────────────────
# Historical OHLCV data never changes, so we store it permanently on disk.
# The cache directory can be overridden with the SQT_CACHE_DIR env variable.
_CACHE_ROOT = Path(
    os.environ.get(
        "SQT_CACHE_DIR",
        str(Path.home() / ".cache" / "standard_quant_tools" / "ohlcv"),
    )
)


def _norm_date(d: Union[str, datetime, _date]) -> str:
    """Normalise any date-like value to a YYYY-MM-DD string."""
    return str(d)[:10]


def _parquet_path(symbol: str, start: str, end: str, interval: str) -> Path:
    safe = symbol.replace("/", "-").upper()
    return _CACHE_ROOT / f"{safe}_{start}_{end}_{interval}.parquet"


def _is_historical(end_date: Union[str, datetime, _date]) -> bool:
    """Return True when end_date is strictly before today (data is immutable)."""
    try:
        return _norm_date(end_date) < _date.today().isoformat()
    except Exception:
        return False


def retry(times: int = 3, delay: float = 1, backoff: float = 2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t_delay = delay
            last_exc = None
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except (InvalidSymbolError, DataNotFoundError):
                    raise  # definitive errors — never retry or re-wrap
                except (ValueError, APIError) as e:
                    last_exc = e
                    if i == times - 1:
                        raise
                    logger.warning(
                        "[retry] %s attempt %d/%d failed: %s — retrying in %.1fs",
                        func.__name__, i + 1, times, e, t_delay,
                    )
                    time.sleep(t_delay)
                    t_delay *= backoff
                except Exception as e:
                    raise APIError(
                        f"Unexpected error in {func.__name__}: {e}"
                    ) from e
            if last_exc:
                raise last_exc
        return wrapper
    return decorator


class YFinanceProvider(DataProvider):

    @cached(_session_cache)
    @retry(times=3, delay=1)
    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
    ) -> pd.DataFrame:
        if not symbol or not isinstance(symbol, str):
            raise InvalidSymbolError(f"Invalid symbol: {symbol}")

        start_str = _norm_date(start_date)
        end_str = _norm_date(end_date)

        # ── Parquet disk cache (historical ranges only) ────────────────────
        pq_path = _parquet_path(symbol, start_str, end_str, interval)
        if _is_historical(end_date) and pq_path.exists():
            logger.debug("[cache] disk hit  %s  %s → %s  (%s)", symbol, start_str, end_str, pq_path.name)
            cached_df = pd.read_parquet(pq_path)
            audit.record_data_access(
                symbol, start_str, end_str, interval,
                source="disk_cache", content_hash=audit.hash_dataframe(cached_df),
            )
            return cached_df

        # ── Fetch from yfinance ────────────────────────────────────────────
        logger.debug("[fetch] yfinance   %s  %s → %s  interval=%s", symbol, start_str, end_str, interval)
        t0 = time.perf_counter()
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date, end=end_date,
                interval=interval, auto_adjust=True,
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

            result = df[required]

        except (DataNotFoundError, InvalidSymbolError, APIError):
            raise
        except Exception as e:
            raise APIError(
                f"Error fetching data for '{symbol}' from yfinance: {e}"
            ) from e

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("[fetch] ✓ %s  %d rows  %.0fms", symbol, len(result), elapsed_ms)
        audit.record_data_access(
            symbol, start_str, end_str, interval,
            source="live_fetch", content_hash=audit.hash_dataframe(result),
        )

        # ── Persist to Parquet for future sessions ─────────────────────────
        if _is_historical(end_date):
            try:
                _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
                # Write to a per-PID temp file then atomically replace the
                # target so concurrent processes don't corrupt each other.
                tmp = pq_path.with_name(
                    f"{pq_path.stem}.{os.getpid()}.tmp.parquet"
                )
                result.to_parquet(tmp)
                tmp.replace(pq_path)  # atomic on all platforms
                logger.debug("[cache] disk write %s  → %s", symbol, pq_path.name)
            except Exception as cache_exc:
                logger.warning("[cache] disk write failed for %s: %s", symbol, cache_exc)

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
        timezone is a fixed NYSE default since yfinance doesn't expose a
        reliable per-symbol timezone through this provider's interface.
        """
        return DataSetMetadata(
            provider="yfinance",
            adjusted=True,
            survivorship_free=False,
            point_in_time=False,
            frequency=interval,
            timezone="America/New_York",
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
