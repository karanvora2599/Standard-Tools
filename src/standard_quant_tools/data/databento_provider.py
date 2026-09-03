"""
Databento as a first-class provider — and the first depth feed this
library has ever had.

WHY THIS IS THE ONE THAT MATTERS. `DataProvider.get_order_book` has been a
declared contract with canonical columns and no implementation since it was
written: "NOT IMPLEMENTED BY ANY PROVIDER IN THIS LIBRARY." Every depth
measure in `analysis/order_book.py` — microprice, touch and cumulative
imbalance, the depth profile, the depth slope — was written and tested
against synthetic books because nothing could serve a real one. This is the
provider that serves one.

WHAT WAS BORROWED RATHER THAN REDISCOVERED. A sibling project has run
Databento in production and its provider carries operational knowledge that
is expensive to learn twice. Its own docstring says upstream "is the
cleaner home" for this, and that it monkey-patches only because it cannot
edit here. Reused deliberately:

  - Dataset PREFERENCE, not a single name. The consolidated feed is the
    best answer where it reaches, the venue feed covers what it does not,
    and the depth venue's bars reach furthest back.
  - `end` anchored to the dataset's own available edge rather than to
    `datetime.now()`. A Saturday request against wall-clock now asks for
    data that was never published and 422s; against the edge it returns
    Friday's tail, which is what the caller meant.
  - The daily finalization lag. `ohlcv-1d` finalizes a day or two behind
    the live feed while `metadata.get_dataset_range` reports the LIVE edge,
    so a naive end lands in the unfinalized tail and fails. The request
    walks the end back a day at a time on that specific error.
  - Entitlement denials remembered. A 403 for one dataset is a fact about
    the subscription, not about the request, and re-asking on every call
    turns one refusal into a per-call latency cost.

WHAT IS NOT BORROWED. Prices. That project scales by a magnitude test on
the LAST row -- `1e9 if close > 1e7 else 1.0` -- which reads one value to
decide the units of a whole frame. `data/databento.py` decides from the
dtype and cross-checks the magnitude in both directions, masks the
int64-max sentinels BEFORE scaling (after it, a sentinel is just a large
float), and reports which timestamp it used. That module also reads the
vendor's own `F_MAYBE_BAD_BOOK` flag, which nothing in that project does.

CREDENTIALS COME FROM THE ENVIRONMENT. `DATABENTO_API_KEY`, never from a
spec or a tool argument: a `DatasetSpec` is persisted to disk, hashed into
a model's lineage and written into decision records, so a key passed
through one would land in all three.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from standard_quant_tools.data.base import DataProvider, FinancialRatios, TickerInfo
from standard_quant_tools.data.databento import (
    CONSOLIDATED_START,
    DATASET_CONSOLIDATED,
    DATASET_DEPTH,
    DATASET_NASDAQ_BASIC,
    normalize_book,
    normalize_mbo,
    normalize_quotes,
    normalize_trades,
)
from standard_quant_tools.data.metadata import DataSetMetadata
from standard_quant_tools.error import APIError, ValidationError

logger = logging.getLogger(__name__)

#: Bar interval -> Databento aggregate schema. Databento publishes
#: aggregates at these four; weekly and monthly are derived from daily by
#: the caller rather than requested, because Databento has no such schema
#: and inventing one here would hide that.
BAR_SCHEMAS: Dict[str, str] = {
    "1s": "ohlcv-1s",
    "1m": "ohlcv-1m",
    "1h": "ohlcv-1h",
    "1d": "ohlcv-1d",
}

#: A plain US equity ticker, and the share-class form that needs mapping.
_EQUITY_RE = re.compile(r"^[A-Z]{1,5}$")
_CLASS_RE = re.compile(r"^([A-Z]{1,5})[.\-]([A-Z])$")

#: Substrings that mean "your subscription does not cover this", as opposed
#: to "this request was wrong". The distinction matters because the first
#: is worth remembering and the second is not.
_DENIAL_MARKERS = ("403", "license", "entitlement", "auth", "not_entitled")

#: The daily-schema finalization error, by the text Databento returns.
_UNFINALIZED_MARKERS = ("available_end", "not_fully_available")

#: How many days to walk the end back before giving up on the daily lag.
_FINALIZATION_ATTEMPTS = 6


def _to_utc(value: Union[str, datetime], *, end_of_day: bool) -> datetime:
    """
    Parse a boundary, honouring this library's INCLUSIVE end-date contract.

    A bare `YYYY-MM-DD` end means "through the end of that day" everywhere
    in this library, and Databento's range is half-open, so the bare form is
    pushed to the next midnight. Passing a date and receiving nothing from
    it is the failure this prevents.
    """
    if isinstance(value, datetime):
        moment = value
        bare = False
    else:
        text = str(value).strip()
        bare = len(text) == 10
        try:
            moment = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValidationError(
                f"{value!r} is not an ISO date or datetime: {exc}"
            ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    if end_of_day and (
        bare or (moment.hour, moment.minute, moment.second) == (0, 0, 0)
    ):
        moment = moment + timedelta(days=1)
    return moment


class DatabentoProvider(DataProvider):
    """Databento Historical, honouring this library's provider contract."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Any = None,
        dataset: Optional[str] = None,
        depth_dataset: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("DATABENTO_API_KEY", "").strip()
        # Injectable so the operational logic above -- dataset preference,
        # the finalization walk-back, denial memory, symbol mapping -- is
        # testable without a key, a network or an entitlement. Those are
        # exactly the parts that are expensive to get wrong and impossible
        # to exercise against a live API in a test suite.
        self._client = client
        self._client_failed = False
        self._lock = threading.Lock()
        self._ranges: Dict[str, Tuple[datetime, datetime]] = {}
        self._denied: Set[str] = set()
        self._dataset = (
            dataset
            or os.environ.get("DATABENTO_DATASET", "").strip()
            or DATASET_NASDAQ_BASIC
        )
        self._depth_dataset = (
            depth_dataset
            or os.environ.get("DATABENTO_DEPTH_DATASET", "").strip()
            or DATASET_DEPTH
        )

    # ── client and datasets ──────────────────────────────────────────
    def _get_client(self) -> Any:
        with self._lock:
            if self._client is not None or self._client_failed:
                if self._client is None:
                    raise APIError(
                        "the Databento client could not be constructed earlier "
                        "in this process and is not retried."
                    )
                return self._client
            if not self._api_key:
                self._client_failed = True
                raise APIError(
                    "DATABENTO_API_KEY is not set. This provider reads its "
                    "credential from the environment and never from a spec "
                    "or a tool argument, because a DatasetSpec is persisted "
                    "to disk, hashed into a model's lineage and written into "
                    "decision records."
                )
            try:
                import databento as db
            except ImportError as exc:
                self._client_failed = True
                raise APIError(
                    "the `databento` package is not installed. Install it "
                    "with `pip install databento` to use provider "
                    "'databento'."
                ) from exc
            try:
                self._client = db.Historical(self._api_key)
            except Exception as exc:  # noqa: BLE001 - one refusal, not a trace
                self._client_failed = True
                raise APIError(f"Databento client construction failed: {exc}") from exc
            return self._client

    @staticmethod
    def _is_denial(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(marker in text for marker in _DENIAL_MARKERS)

    def _available_range(self, dataset: str) -> Optional[Tuple[datetime, datetime]]:
        """
        What the dataset actually covers, resolved once and remembered.

        Every request is clamped to this. Without it a request whose end is
        wall-clock `now` asks for data Databento has not published -- which
        is every request made on a weekend, and it fails rather than
        returning the last session.
        """
        with self._lock:
            if dataset in self._ranges:
                return self._ranges[dataset]
        client = self._get_client()
        try:
            meta = client.metadata.get_dataset_range(dataset=dataset)
        except Exception as exc:  # noqa: BLE001
            logger.warning("databento range lookup failed for %s: %s", dataset, exc)
            if self._is_denial(exc):
                with self._lock:
                    self._denied.add(dataset)
            return None
        try:
            start = datetime.fromisoformat(str(meta["start"]).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(meta["end"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("databento range for %s is unreadable: %s", dataset, exc)
            return None
        with self._lock:
            self._ranges[dataset] = (start, end)
        return (start, end)

    def _bar_datasets(self) -> List[str]:
        """
        Bar datasets in preference order.

        The consolidated feed first, because every lit venue on one tape is
        the best answer wherever it reaches. Then the venue feed. Then the
        DEPTH venue, whose bars reach furthest back -- so a multi-year
        request that the consolidated feed cannot cover still has somewhere
        to go instead of failing.
        """
        override = os.environ.get("DATABENTO_OHLCV_DATASET", "").strip()
        candidates = (
            [override, self._depth_dataset]
            if override
            else [DATASET_CONSOLIDATED, self._dataset, self._depth_dataset]
        )
        seen: List[str] = []
        for name in candidates:
            if name and name not in seen and name not in self._denied:
                seen.append(name)
        return seen

    @staticmethod
    def to_raw_symbol(symbol: str) -> str:
        """
        This library's ticker as Databento's `raw_symbol`.

        Share classes are the whole job: `BRK.B` and `BRK-B` are `BRKB` on
        the Nasdaq feeds. Note that the LIVE gateway uses the dotted form
        for the same instrument, so a live path needs the inverse map and
        not this one -- a difference worth stating because getting it
        backwards produces an empty result rather than an error.
        """
        text = str(symbol).strip().upper()
        if _EQUITY_RE.match(text):
            return text
        match = _CLASS_RE.match(text)
        if match:
            return f"{match.group(1)}{match.group(2)}"
        raise ValidationError(
            f"{symbol!r} is not a US-equity symbol this provider can map to "
            "a Databento raw_symbol. Expected a 1-5 letter ticker, or a "
            "share class like 'BRK.B'."
        )

    # ── the request itself ───────────────────────────────────────────
    def _range(
        self,
        dataset: str,
        start: datetime,
        end: datetime,
    ) -> Optional[Tuple[datetime, datetime]]:
        """Clamp a request to what the dataset published, or decline it."""
        span = self._available_range(dataset)
        if span is None:
            return None
        first, last = span
        if start < first:
            # Not an error: a deeper dataset may cover it, and the caller
            # gets whichever one does.
            return None
        clamped_end = min(end, last)
        if clamped_end <= start:
            return None
        return start, clamped_end

    def _get_range(
        self, dataset: str, schema: str, raw: str, start: datetime, end: datetime
    ) -> Optional[pd.DataFrame]:
        """One request, with the daily-finalization walk-back."""
        client = self._get_client()

        def _call(request_end: datetime, fmt: str) -> pd.DataFrame:
            store = client.timeseries.get_range(
                dataset=dataset,
                schema=schema,
                symbols=[raw],
                stype_in="raw_symbol",
                start=start.strftime(fmt),
                end=request_end.strftime(fmt),
            )
            return store.to_df()

        if schema == "ohlcv-1d":
            # Databento finalizes daily bars a day or two behind the live
            # feed, while the dataset range reports the LIVE edge -- so the
            # honest end lands in the unfinalized tail and is refused. Walk
            # it back, but only on THAT error: anything else is a real
            # failure and re-asking would hide it.
            attempt = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
            attempt += timedelta(days=1)
            for _ in range(_FINALIZATION_ATTEMPTS):
                if attempt <= start:
                    return None
                try:
                    return _call(attempt, "%Y-%m-%d")
                except Exception as exc:  # noqa: BLE001
                    text = str(exc).lower()
                    if any(marker in text for marker in _UNFINALIZED_MARKERS):
                        attempt -= timedelta(days=1)
                        continue
                    raise
            return None
        return _call(end, "%Y-%m-%dT%H:%M:%S")

    def _fetch(
        self,
        schema: str,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        *,
        datasets: Optional[List[str]] = None,
        what: str = "data",
    ) -> Tuple[pd.DataFrame, str]:
        """Try each dataset in order; return the first that answers."""
        raw = self.to_raw_symbol(symbol)
        start = _to_utc(start_date, end_of_day=False)
        end = _to_utc(end_date, end_of_day=True)
        if end <= start:
            raise ValidationError(
                f"empty window: start {start_date!r} is not before end "
                f"{end_date!r} (the end date is INCLUSIVE, so a same-day "
                "request is valid and this is not one)."
            )

        tried: List[str] = []
        for dataset in datasets if datasets is not None else self._bar_datasets():
            if dataset in self._denied:
                continue
            if dataset == DATASET_CONSOLIDATED and start < CONSOLIDATED_START:
                # The consolidated feed does not exist before this date, so
                # asking is a guaranteed miss and a wasted round trip.
                continue
            window = self._range(dataset, start, end)
            if window is None:
                continue
            tried.append(dataset)
            try:
                frame = self._get_range(dataset, schema, raw, *window)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "databento %s %s on %s failed: %s", symbol, schema, dataset, exc
                )
                if self._is_denial(exc):
                    with self._lock:
                        self._denied.add(dataset)
                continue
            if frame is not None and len(frame):
                return frame, dataset

        raise APIError(
            f"Databento returned no {what} for {symbol} "
            f"({schema}) between {start:%Y-%m-%d} and {end:%Y-%m-%d}. "
            + (
                f"Datasets tried: {tried}."
                if tried
                else "No dataset covers that range, or every one was declined "
                "by the subscription."
            )
        )

    # ── the contract ─────────────────────────────────────────────────
    def get_ohlcv(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        OHLCV bars, UNADJUSTED.

        Databento serves what the venue published, so a split is a real
        -50% bar and a dividend is a real gap. That is the correct raw
        record and the wrong input for a momentum feature, which is why
        `get_metadata` reports `adjusted=False` rather than leaving a caller
        to discover it in a return series.
        """
        schema = BAR_SCHEMAS.get(interval)
        if schema is None:
            raise ValidationError(
                f"interval={interval!r} is not one Databento aggregates. It "
                f"publishes {sorted(BAR_SCHEMAS)}; weekly and monthly are "
                "resampled from daily by the caller rather than requested, "
                "because Databento has no such schema and inventing one here "
                "would hide that."
            )
        frame, _dataset = self._fetch(schema, symbol, start_date, end_date, what="bars")
        return self._to_ohlcv(frame, symbol)

    @staticmethod
    def _to_ohlcv(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
        columns = {c.lower(): c for c in frame.columns}
        missing = [
            c for c in ("open", "high", "low", "close", "volume") if c not in columns
        ]
        if missing:
            raise APIError(
                f"Databento bars for {symbol} are missing {missing}; got "
                f"{list(frame.columns)[:12]}"
            )
        out = pd.DataFrame(index=frame.index)
        # Prices are fixed-point in the raw store and float dollars from
        # `.to_df()`. Decided from the dtype rather than from a magnitude
        # test on one row -- see data/databento.py::_decide_price_scale.
        from standard_quant_tools.data.databento import _decide_price_scale

        price_cols = [columns[c] for c in ("open", "high", "low", "close")]
        factor, _note = _decide_price_scale(frame, price_cols, "auto")
        for lower, target in (
            ("open", "Open"),
            ("high", "High"),
            ("low", "Low"),
            ("close", "Close"),
        ):
            out[target] = pd.to_numeric(frame[columns[lower]], errors="coerce") * factor
        out["Volume"] = pd.to_numeric(frame[columns["volume"]], errors="coerce")
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
        out = out[out.index.notna()]
        if out["Close"].isna().any():
            raise APIError(
                f"Databento bars for {symbol} contain a null Close, which no "
                "downstream return calculation can use."
            )
        return out.sort_index()

    async def get_ohlcv_async(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
    ) -> pd.DataFrame:
        import asyncio

        return await asyncio.to_thread(
            self.get_ohlcv, symbol, start_date, end_date, interval
        )

    def get_trades(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Individual trades, in this library's `price`/`size` contract."""
        frame, _dataset = self._fetch(
            "trades",
            symbol,
            start_date,
            end_date,
            datasets=[self._depth_dataset, self._dataset],
            what="trades",
        )
        out, _notes = normalize_trades(frame)
        if limit is not None and len(out) > limit:
            out = out.head(int(limit))
        return out.set_index("timestamp") if "timestamp" in out.columns else out

    def get_quotes(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Top-of-book quotes, in this library's `bid_price`/`ask_price` contract."""
        frame, _dataset = self._fetch(
            "mbp-1",
            symbol,
            start_date,
            end_date,
            datasets=[self._dataset, self._depth_dataset],
            what="quotes",
        )
        out, _notes = normalize_quotes(frame)
        if limit is not None and len(out) > limit:
            out = out.head(int(limit))
        return out.set_index("timestamp") if "timestamp" in out.columns else out

    def get_order_book(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        levels: int = 5,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        L2 depth snapshots — the first implementation of this contract.

        Served from the DEPTH dataset only. The consolidated and Nasdaq
        Basic feeds are top of book, and returning one of those here would
        hand back a book of one level whose imbalance is zero by
        construction — which reads as a balanced market rather than as
        missing depth. That is the exact substitution the base class refuses
        to make, and it is refused here too.

        `levels` caps how deep to read; mbp-10 carries ten. The frame comes
        back in `ORDER_BOOK_COLUMNS`, so `get_order_book_metrics` and
        `analysis/order_book.py` read it without knowing where it came from.
        """
        if levels < 1 or levels > 10:
            raise ValidationError(
                f"levels={levels}: mbp-10 carries ten levels, so 1-10 is the "
                "range this provider can answer."
            )
        frame, _dataset = self._fetch(
            "mbp-10",
            symbol,
            start_date,
            end_date,
            datasets=[self._depth_dataset],
            what="depth",
        )
        out, notes = normalize_book(frame, levels=levels)
        for note in notes:
            if note.startswith("WARNING"):
                logger.warning("databento book %s: %s", symbol, note)
        if limit is not None and len(out) > limit:
            out = out.head(int(limit))
        return out

    def get_order_events(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Market-by-order: every add, cancel, modify and fill, with its id.

        The deepest feed here, and the only one from which queue position
        and a true cancellation rate can be computed. Depth aggregates size
        per level; this does not aggregate at all, which is the whole
        difference.

        BE AWARE OF THE VOLUME. MBO is one record per order event, so an
        active name produces millions in a session where mbp-10 produces
        thousands. A window that is comfortable for `get_order_book` can be
        two orders of magnitude larger here -- pass `limit`, or pull it
        once, write it, and register it with `register_external_dataset`
        rather than re-fetching a metered feed.
        """
        frame, _dataset = self._fetch(
            "mbo",
            symbol,
            start_date,
            end_date,
            datasets=[self._depth_dataset],
            what="order events",
        )
        out, notes = normalize_mbo(frame)
        for note in notes:
            if note.startswith("WARNING"):
                logger.warning("databento mbo %s: %s", symbol, note)
        if limit is not None and len(out) > limit:
            out = out.head(int(limit))
        return out

    def get_ticker_info(self, symbol: str) -> TickerInfo:
        """
        Databento serves market data, not company reference data.

        Returned rather than raised, with the fields it genuinely knows,
        because `get_ticker_info` is called incidentally by several paths
        and an exception there would make a market-data provider unusable
        for market data.
        """
        return TickerInfo(symbol=str(symbol).upper())

    def get_financial_ratios(self, symbol: str) -> FinancialRatios:
        raise ValidationError(
            "Databento is a market-data provider and publishes no "
            "fundamentals, so there are no ratios to report for "
            f"{symbol!r}. Use provider 'yfinance' or 'polygon' for those; "
            "reporting empty ratios here would be indistinguishable from a "
            "company that genuinely has none."
        )

    def get_metadata(self, symbol: str, interval: str = "1d") -> DataSetMetadata:
        """
        An honest self-report, and two fields worth reading before trusting
        a backtest built on this.

        `adjusted=False` because Databento serves what the venue published.
        A split is a real -50% bar. Every other provider here reports True,
        so this is the one that will surprise someone.

        `point_in_time=False` because Databento does reprocess and correct
        data. The corrections are announced rather than silent, which is
        better than most, but "announced" is not the guarantee this field
        asks about.
        """
        return DataSetMetadata(
            provider="databento",
            adjusted=False,
            # The archive is organized by publication, so an instrument that
            # stopped trading stays queryable over the window it traded in.
            survivorship_free=True,
            point_in_time=False,
            frequency=interval,
            timezone="UTC",
        )


__all__ = ["BAR_SCHEMAS", "DatabentoProvider"]
