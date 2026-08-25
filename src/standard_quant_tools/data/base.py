from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

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
    """
    Fundamental ratios in ONE canonical unit and definition, whichever
    provider served them.

    The shared field names used to imply an interchangeability that did not
    exist. yfinance reports `debtToEquity` as a PERCENTAGE (150.5) while
    Polygon computes a plain RATIO (1.505), so a screen written as
    `debt_equity_max=2.0` admitted nearly every company on one provider and
    nearly none on the other — with nothing in either result saying which
    convention was in force.

    The canonical units:

    | field | unit |
    |---|---|
    | `forward_pe`, `trailing_pe`, `price_to_book`, `debt_to_equity` | plain ratio |
    | `return_on_equity`, `profit_margins`, `dividend_yield` | decimal fraction (0.15 == 15%) |
    | `market_cap` | absolute units of the reporting currency |

    See `standard_quant_tools.data.ratios` for the per-field formula and the
    per-provider conversions.

    `definition_notes` carries any field whose FORMULA (not merely its unit)
    departs from the canonical one — a unit difference is mechanical and is
    converted, a definition difference is not and is declared. The clearest
    case is `debt_to_equity`: Polygon derives it from total LIABILITIES,
    which include payables and deferred revenue, so it is systematically
    higher than a debt-based ratio for reasons unrelated to leverage. The
    value is still returned, because a liabilities-to-equity ratio is useful
    when you know that is what it is.
    """

    forward_pe: Optional[float] = None
    trailing_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    debt_to_equity: Optional[float] = None
    return_on_equity: Optional[float] = None
    profit_margins: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[int] = None
    definition_notes: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "field -> how this provider's definition departs from the "
            "canonical one. Empty when every populated field is canonical."
        ),
    )


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
            start_date: Start date (YYYY-MM-DD or datetime), INCLUSIVE.
            end_date: End date (YYYY-MM-DD or datetime), **INCLUSIVE** — the
                returned frame contains observations up to and including this
                date. A bare date means "through the end of that day" at every
                interval; an explicit intraday timestamp means exactly that
                instant.

                This is a contract every provider must honor, not a
                pass-through of whatever its upstream API happens to do.
                The underlying vendors disagree: Polygon's aggregates `to`
                and Bloomberg's `endDate` are inclusive, but yfinance's
                `ticker.history(end=...)` is EXCLUSIVE. Passing the caller's
                date straight through therefore returned a different window
                depending only on which provider served it, and silently
                dropped the final bar on the default provider.

                Implementations convert to whatever their API expects and
                trim the result (see data/_cache.py's
                inclusive_end_timestamp / trim_to_inclusive_end), so the
                contract holds by construction rather than by trusting each
                vendor's documented boundary.
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

    # ── Tick-level data (optional capability) ────────────────────────────
    #
    # Not abstract, deliberately. Every method above is something all three
    # shipped providers can do; these two are not. Marking them abstract
    # would break yfinance and Bloomberg at import time to express a fact
    # better expressed by a clear error at the point of use -- the same
    # choice the modeling runtime makes for lightgbm/xgboost, where a
    # missing capability is reported rather than fatal.
    #
    # The bar methods above are the library's whole world today. These exist
    # so that a caller who needs the microstructure layer gets a specific
    # answer about THIS provider rather than an AttributeError, and so that
    # a provider gaining the capability has an obvious place to put it.

    def get_trades(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Individual trades (ticks) for one symbol over a time range.

        Returns a DataFrame indexed by timestamp with at least `price` and
        `size` columns; providers may add exchange and condition codes.

        Raises:
            NotImplementedError: this provider has no tick feed. Bars are
                not a substitute and this deliberately does not synthesize
                one -- a "trade" derived from an OHLCV row is a fiction that
                every downstream microstructure measure would treat as fact.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide tick-level trades. "
            "Only PolygonProvider does, and it needs a plan tier that "
            "includes trades (see Documentation/01_data_fetching.md). Bar "
            "data cannot substitute: spreads and signed order flow are not "
            "recoverable from an OHLCV row."
        )

    def get_quotes(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Best bid/offer quotes for one symbol over a time range.

        Returns a DataFrame indexed by timestamp with at least `bid_price`,
        `bid_size`, `ask_price` and `ask_size`.

        This is TOP OF BOOK only. No shipped provider offers depth, so
        nothing in this library sees the order book, and anything needing
        queue position or resting size at a level is out of reach rather
        than approximated.

        Raises:
            NotImplementedError: this provider has no quote feed.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide quotes. Only "
            "PolygonProvider does, and it needs a plan tier that includes "
            "quotes (see Documentation/01_data_fetching.md). The "
            "Corwin-Schultz and Amihud estimators in "
            "`analysis`/`get_liquidity_metrics` exist precisely because this "
            "data is usually absent -- they are proxies, and they say so."
        )
