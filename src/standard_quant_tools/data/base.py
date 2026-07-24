import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel

from standard_quant_tools.data.metadata import DataSetMetadata


class TickerInfo(BaseModel):
    symbol: str
    name: str = "Unknown"
    sector: str = "Unknown"
    industry: str = "Unknown"
    full_time_employees: Optional[int] = None
    city: Optional[str] = None
    country: Optional[str] = None
    website: Optional[str] = None


class FinancialRatios(BaseModel):
    forward_pe: Optional[float] = None
    trailing_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    debt_to_equity: Optional[float] = None
    return_on_equity: Optional[float] = None
    profit_margins: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[int] = None


class DataProvider(ABC):
    """
    Abstract Base Class for Data Providers.
    Ensures all providers return data in a standard format.
    """

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV data.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL').
            start_date: Start date (YYYY-MM-DD or datetime).
            end_date: End date (YYYY-MM-DD or datetime).
            interval: Data interval (e.g., '1d', '1h').

        Returns:
            pd.DataFrame: A DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
                          and a DatetimeIndex.

        Raises:
            ValueError: If data cannot be fetched or symbol is invalid.
        """
        pass

    @abstractmethod
    async def get_ohlcv_async(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Async version of get_ohlcv.
        """
        pass

    @abstractmethod
    def get_ticker_info(self, symbol: str) -> TickerInfo:
        """
        Fetches basic company information.
        """
        pass

    @abstractmethod
    def get_financial_ratios(self, symbol: str) -> FinancialRatios:
        """
        Fetches key financial ratios.
        """
        pass

    @abstractmethod
    def get_metadata(self, symbol: str, interval: str = "1d") -> DataSetMetadata:
        """
        Reports this provider's dataset guarantees (or lack thereof) for a
        given symbol/interval — see DataSetMetadata's docstring. Every
        provider must answer honestly, not aspirationally.
        """
        pass
