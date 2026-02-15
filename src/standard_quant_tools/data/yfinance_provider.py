from datetime import datetime
from typing import Union
import pandas as pd
import yfinance as yf
from .base import DataProvider, TickerInfo, FinancialRatios
import asyncio
from cachetools import TTLCache, cached
import time
import functools

# Cache data for 1 hour
cache = TTLCache(maxsize=100, ttl=3600)

def retry(times=3, delay=1, backoff=2):
    """
    Retry decorator with exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t_delay = delay
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if i == times - 1:
                        raise e
                    time.sleep(t_delay)
                    t_delay *= backoff
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
        try:
            ticker = yf.Ticker(symbol)
            # Use 'auto_adjust=True' for HFT-like analysis
            df = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
            
            if df.empty:
                raise ValueError(f"No data found for symbol '{symbol}' using yfinance.")
            
            df.columns = [c.capitalize() for c in df.columns]
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                 missing = [c for c in required_cols if c not in df.columns]
                 raise ValueError(f"Incomplete data. Missing: {missing}")

            return df[required_cols]

        except Exception as e:
            raise ValueError(f"Error fetching data for '{symbol}': {str(e)}")

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
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
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
        except Exception as e:
             raise ValueError(f"Error fetching ticker info for '{symbol}': {str(e)}")

    @retry(times=3, delay=1)
    def get_financial_ratios(self, symbol: str) -> FinancialRatios:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
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
        except Exception as e:
            raise ValueError(f"Error fetching financials for '{symbol}': {str(e)}")
