from datetime import datetime
from typing import Union
import pandas as pd
import yfinance as yf
from .base import DataProvider, TickerInfo, FinancialRatios

class YFinanceProvider(DataProvider):
    """
    Implementation of DataProvider using yfinance.
    """
    
    def get_ohlcv(
        self, 
        symbol: str, 
        start_date: Union[str, datetime], 
        end_date: Union[str, datetime], 
        interval: str = "1d"
    ) -> pd.DataFrame:
        try:
            # yfinance expects date strings or datetime objects
            # Ensure proper string formatting if passed as string
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data found for symbol '{symbol}' using yfinance.")
            
            # Standardize column names (Title Case)
            df.columns = [c.capitalize() for c in df.columns]
            
            # Ensure required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                 # Handle cases where some columns might be missing or named differently
                 # For yfinance, Dividends and Stock Splits are also returned, we can keep them or drop them.
                 # We strictly need OHLCV for most analysis.
                 missing = [c for c in required_cols if c not in df.columns]
                 raise ValueError(f"Incomplete data from yfinance. Missing columns: {missing}")

            return df[required_cols]

        except Exception as e:
            raise ValueError(f"Error fetching data for '{symbol}' from yfinance: {str(e)}")

    def get_ticker_info(self, symbol: str) -> TickerInfo:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Map yfinance info dict to TickerInfo model
            # Use .get() to handle potential missing keys safely
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
             # Depending on desired behavior, we could return a default object or raise
             # For agent robustness, raising a clear error might be better so it knows it failed
             raise ValueError(f"Error fetching ticker info for '{symbol}': {str(e)}")

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
