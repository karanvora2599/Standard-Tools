from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, FrozenSet, Optional, Sequence, Tuple, Union

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

        IMPLEMENTED BY `DatabentoProvider` ONLY, from its depth dataset.
        Every other provider refuses explicitly and by name, exactly as
        `get_trades` refuses -- because the alternative, returning
        top-of-book twice and calling it depth, would silently produce a
        book with one level and an imbalance of zero, which reads as a
        balanced market rather than as missing data.

        Declared before any implementation on purpose, and the sequencing
        paid off. The analysis that consumes a book (microprice,
        order-flow imbalance, depth slope) was written and tested against
        synthetic books shaped to THIS contract, so when a source finally
        arrived the correctness-critical part already existed rather than
        being invented under deadline. `point_in_time.py` used the same
        sequencing for the availability join.

        Columns, when a provider does implement it: `timestamp`, then
        `bid_price_{i}` / `bid_size_{i}` / `ask_price_{i}` / `ask_size_{i}`
        for i in 0..levels-1, level 0 being the touch.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not serve L2 order book data. "
            "DatabentoProvider does, from its depth dataset -- use "
            "provider='databento' with DATABENTO_API_KEY set. Call "
            "describe_data_capabilities to see what THIS provider can "
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

    def get_point_in_time_records(
        self,
        symbols: Sequence[str],
        frame_kind: str,
        fields: Sequence[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Records of `frame_kind` for `symbols`, each stamped with when it
        became knowable.

        Returns a frame in the `modeling.dataset.point_in_time` schema: one
        row per VERSION of a fact, with `entity`, `event_time` (what period
        the record describes), `available_time` (when anyone could act on
        it) and one column per requested field. A restatement is a second
        row with a later `available_time`, never an overwrite -- that is
        what lets a past decision be reproduced from the frame.

        `start_date`/`end_date` bound the records' EVENT times; a caller
        joining onto a panel widens `start_date` by the staleness it will
        accept, so the panel's first rows have a record to read.

        Raises:
            NotImplementedError: this provider does not serve point-in-time
                records. Its `get_temporal_contract(frame_kind)` says the
                same thing first, without a fetch, and the dataset builder
                asks that before calling this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not serve point-in-time "
            f"{frame_kind!r} records. PolygonProvider serves 'fundamentals' "
            "from its financials endpoint, stamped with each filing's date; "
            "a frame joined on the period it describes rather than on when "
            "it was filed would put weeks of hindsight in every row, which "
            "is why nothing here substitutes get_financial_ratios."
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
            "PolygonProvider (on a plan tier that includes trades) and "
            "DatabentoProvider (from the venue tape) do -- source='polygon' "
            "or source='databento' (see Documentation/01_data_fetching.md). Bar "
            "data cannot substitute: spreads and signed order flow are not "
            "recoverable from an OHLCV row."
        )

    #: Canonical column layout for an ORDER feed, the same way
    #: ORDER_BOOK_COLUMNS does for depth. `order_id` and `action` are what
    #: make it an order feed: aggregated depth cannot say whether 5,000
    #: shares is one order or two hundred, nor tell a cancel from a fill,
    #: and those mean opposite things about who wanted to trade.
    ORDER_EVENT_COLUMNS = (
        "timestamp",
        "order_id",
        "action",
        "side",
        "price",
        "size",
    )

    def get_order_events(
        self,
        symbol: str,
        start_date,
        end_date,
        limit=None,
    ):
        """
        Order-by-order events: every add, cancel, modify and fill.

        IMPLEMENTED BY `DatabentoProvider` ONLY, from its depth dataset's
        market-by-order schema. This is a strictly deeper feed than
        `get_order_book`, and the difference is not depth but IDENTITY: a
        book snapshot aggregates size per price level, and that aggregation
        is what makes queue position, order lifetime and a true
        cancellation rate impossible to recover.

        Columns, when a provider implements it: `timestamp`, `order_id`,
        `action` (A add, C cancel, M modify, F fill, T trade, R clear),
        `side` (B/A), `price`, `size`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not serve order-by-order data. "
            "DatabentoProvider does, through its market-by-order schema -- "
            "use provider='databento' with DATABENTO_API_KEY set. Depth "
            "snapshots are not a substitute: aggregated size per level "
            "cannot say how much of it is ahead of your order, nor "
            "distinguish a cancel from a fill."
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

        This is TOP OF BOOK only, whichever provider serves it. Depth is
        a different call -- `get_order_book`, which DatabentoProvider
        implements. Queue position and per-order resting size are in
        NEITHER: they need an order-level feed, and inferring them from
        aggregated depth would be a guess wearing a measurement's clothes.

        Raises:
            NotImplementedError: this provider has no quote feed.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide quotes. "
            "PolygonProvider (on a plan tier that includes quotes) and "
            "DatabentoProvider do -- source='polygon' or source='databento' "
            "(see Documentation/01_data_fetching.md). The "
            "Corwin-Schultz and Amihud estimators in "
            "`analysis`/`get_liquidity_metrics` exist precisely because this "
            "data is usually absent -- they are proxies, and they say so."
        )

    # ── what a request would cost before it is made ──────────────────
    #
    # THESE TWO ARE FREE AND THE FETCHES THEY DESCRIBE ARE NOT. A metered
    # feed prices a window by the bytes it would transfer, and the only
    # honest way to find out is to ask before asking -- five minutes of one
    # active name at ten depth levels is tens of megabytes, and a caller
    # who learns that from the invoice learned it too late. Declared on the
    # contract rather than on one provider so a tool can ask the question
    # without knowing which provider answers it, the same way the depth
    # contract was declared before anything served it.

    def get_dataset_coverage(
        self, datasets: Optional[Sequence[str]] = None
    ) -> Dict[str, Tuple[str, str]]:
        """
        Which window each vendor dataset actually published.

        Returns `{dataset: (first, last)}` with both bounds as ISO-8601
        UTC strings, for the datasets named or for every dataset this
        provider would route a request to.

        A dataset the subscription cannot see, or whose range the vendor
        declines to report, is ABSENT from the mapping rather than present
        with a guessed window. That asymmetry is the point: a fabricated
        coverage window is worse than no window at all, because it is
        exactly the thing a caller would plan a request around, and the
        request would then fail with a vendor error naming no dataset.

        Raises:
            NotImplementedError: this provider publishes no coverage
                metadata. DatabentoProvider does, from the same endpoint it
                clamps its own requests against.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not report dataset coverage. "
            "DatabentoProvider does, from the vendor's free metadata "
            "endpoint -- use source='databento'. There is no coverage "
            "window to infer from a provider that serves one feed: "
            "'whatever get_ohlcv returned' is a fact about the request, "
            "not about what the vendor holds."
        )

    def get_billable_size(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        schema: str,
        *,
        dataset: Optional[str] = None,
    ) -> int:
        """
        Bytes the vendor would bill for this exact request, without making it.

        BYTES RATHER THAN MONEY, deliberately. A subscription that includes
        a feed prices every request on it at zero, so a cost in dollars is
        `0.00` for the one caller who most needs to know that the request
        is forty megabytes. The byte count is the quantity that is true for
        every account.

        `schema` names the vendor's own product -- daily bars, trades, top
        of book, depth, order-by-order -- and `dataset` pins the feed when
        the caller has one in mind; left None, the provider routes the
        request exactly as the matching fetch would.

        Raises:
            NotImplementedError: this provider does not price requests.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not price a request before it is "
            "made. DatabentoProvider does, from the vendor's free metadata "
            "endpoint -- use source='databento'. A provider with a flat "
            "subscription and no metered feed has no per-request size to "
            "report, and returning zero would read as 'this is free' "
            "rather than as 'nobody asked'."
        )
