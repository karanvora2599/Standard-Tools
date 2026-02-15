from datetime import datetime
from typing import Union
import pandas as pd
import yfinance as yf
from .base import DataProvider, TickerInfo, FinancialRatios
from standard_quant_tools.error import APIError, DataNotFoundError, InvalidSymbolError
import asyncio
from cachetools import TTLCache, cached
import time
import functools

# Cache data for 1 hour
cache = TTLCache(maxsize=100, ttl=3600)

def retry(times=3, delay=1, backoff=2):
    """
    Retry decorator with exponential backoff and specific exception handling.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t_delay = delay
            last_exception = None
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except (ValueError, APIError) as e:
                    # Retry on transient errors
                    last_exception = e
                    if i == times - 1:
                        raise e
                    time.sleep(t_delay)
                    t_delay *= backoff
                except Exception as e:
                    # Don't retry on unexpected errors (fail fast)
                    raise APIError(f"Unexpected error in {func.__name__}: {str(e)}") from e
            if last_exception:
                raise last_exception
        return wrapper
    return decorator

class YFinanceProvider(DataProvider):
    """
    Implementation of DataProvider using yfinance.
    """
    
    @cached(cache)
    @retry(times=3, delay=1)
    def get_ohlcv(
        self, 
        symbol: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime], 
        interval: str = "1d"
    ) -> pd.DataFrame:
        if not symbol or not isinstance(symbol, str):
            raise InvalidSymbolError(f"Invalid symbol: {symbol}")

        try:
            ticker = yf.Ticker(symbol)
            # Use 'auto_adjust=True' for HFT-like analysis
            df = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
            
            if df.empty:
                # yfinance returns empty DF for invalid symbols or no data
                raise DataNotFoundError(f"No data found for symbol '{symbol}' using yfinance. Verify symbol and date range.")
            
            df.columns = [c.capitalize() for c in df.columns]
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                 raise APIError(f"Incomplete data from yfinance. Missing columns: {missing}")

            # Basic data quality check
            if df['Close'].isnull().any():
                 raise APIError(f"Data for {symbol} contains NaNs in Close column.")

            return df[required_cols]

        except (DataNotFoundError, InvalidSymbolError, APIError):
            raise
        except Exception as e:
            raise APIError(f"Error fetching data for '{symbol}' from yfinance: {str(e)}") from e

    async def get_ohlcv_async(
        self, 
        symbol: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime], 
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Async fetches by offloading to a thread.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_ohlcv, symbol, start_date, end_date, interval)

    @retry(times=3, delay=1)
    def get_ticker_info(self, symbol: str) -> TickerInfo:
        if not symbol:
             raise InvalidSymbolError("Symbol cannot be empty.")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # yfinance often returns an empty dict or minimal dict if symbol is invalid but doesn't raise
            if not info or len(info) < 2:
                 raise DataNotFoundError(f"No metadata found for symbol '{symbol}'.")

            return TickerInfo(
                symbol=symbol,
                name=info.get('longName', 'Unknown'),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                full_time_employees=info.get('fullTimeEmployees'),
                city=info.get('city'),
                country=info.get('country'),
                website=info.get('website')
            )
        except (DataNotFoundError, InvalidSymbolError):
            raise
        except Exception as e:
             raise APIError(f"Error fetching ticker info for '{symbol}': {str(e)}") from e

    @retry(times=3, delay=1)
    def get_financial_ratios(self, symbol: str) -> FinancialRatios:
        if not symbol:
             raise InvalidSymbolError("Symbol cannot be empty.")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                 raise DataNotFoundError(f"No financial data found for symbol '{symbol}'.")

            return FinancialRatios(
                forward_pe=info.get('forwardPE'),
                trailing_pe=info.get('trailingPE'),
                price_to_book=info.get('priceToBook'),
                debt_to_equity=info.get('debtToEquity'),
                return_on_equity=info.get('returnOnEquity'),
                profit_margins=info.get('profitMargins'),
                dividend_yield=info.get('dividendYield'),
                market_cap=info.get('marketCap')
            )
        except (DataNotFoundError, InvalidSymbolError):
             raise
        except Exception as e:
            raise APIError(f"Error fetching financials for '{symbol}': {str(e)}") from e
