from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, FrozenSet, Optional, Union

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

    #: Bar intervals this provider accepts, or None when it declares no
    #: set. Every provider already validates `interval` against its own
    #: private module constant; this makes that vocabulary askable without
    #: a caller reaching into another module's underscore-prefixed global.
    SUPPORTED_INTERVALS: Optional[FrozenSet[str]] = None

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

    #: Canonical column layout for a depth frame, so every consumer of an
    #: order book agrees on what one looks like before any provider serves
    #: one. Levels are numbered from the touch: bid_price_0 is the best bid.
    ORDER_BOOK_COLUMNS = (
        "timestamp",
        "bid_price_{level}",
        "bid_size_{level}",
        "ask_price_{level}",
        "ask_size_{level}",
    )

    def get_order_book(
        self,
        symbol: str,
        start_date,
        end_date,
        levels: int = 5,
        limit=None,
    ):
        """
        L2 depth snapshots: price and resting size at each level, per update.

        NOT IMPLEMENTED BY ANY PROVIDER IN THIS LIBRARY. The refusal is
        explicit and by name, exactly as `get_trades` refuses, because the
        alternative -- returning top-of-book twice and calling it depth --
        would silently produce a book with one level and an imbalance of
        zero, which reads as a balanced market rather than as missing data.

        Declared before any implementation on purpose. The analysis that
        consumes a book (microprice, order-flow imbalance, depth slope) can
        be written and tested against synthetic books now, so that when a
        source arrives the correctness-critical part already exists rather
        than being invented under deadline. That is the same sequencing
        `point_in_time.py` used for the availability join, and for the same
        reason.

        Columns, when a provider does implement it: `timestamp`, then
        `bid_price_{i}` / `bid_size_{i}` / `ask_price_{i}` / `ask_size_{i}`
        for i in 0..levels-1, level 0 being the touch.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not serve L2 order book data. No "
            "provider in this library does yet. Call "
            "describe_data_capabilities to see what this provider can "
            "serve; do not substitute top-of-book quotes, which have one "
            "level and would report every book as perfectly balanced."
        )

    def get_temporal_contract(self, frame_kind: str = "bars"):
        """
        What this provider can say about WHEN its facts became knowable.

        The base implementation is the honest default rather than a
        placeholder. Bars are safe by construction -- a bar is knowable at
        its own close -- and every other frame kind is declared UNSUPPORTED,
        because no provider in this library currently supplies availability
        timestamps for filings, estimates or macro releases. A provider that
        can should override this and say so per kind.

        Declaring it rather than inferring it is the point. A heuristic that
        guessed availability from an event date would be right for prices,
        wrong for every filing, and wrong in the direction that makes a
        backtest look prescient.
        """
        from standard_quant_tools.data.temporal import (
            TemporalContract,
            price_contract,
        )

        name = type(self).__name__
        if frame_kind == "bars":
            return price_contract(name)
        return TemporalContract(
            source=name,
            frame_kind=frame_kind,
            has_event_time=False,
            has_available_time=False,
            revisions="unknown",
            notes=[
                f"{name} does not serve {frame_kind!r} with availability "
                "timestamps. This is a statement about the provider, not "
                "about the frame kind -- the point-in-time join is built and "
                "tested, and works as soon as a source supplies the column.",
            ],
        )

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
