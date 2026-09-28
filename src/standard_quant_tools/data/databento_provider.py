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
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from standard_quant_tools import audit
from standard_quant_tools._env import env_str
from standard_quant_tools.data._cache import (
    _is_historical,
    _norm_cache_bound,
    _normalize_ohlcv_index,
    _read_cached_ohlcv,
    _safe_parquet_path,
    _session_cache_get,
    _session_cache_set,
    _write_cached_ohlcv,
    trim_to_inclusive_end,
)
from standard_quant_tools.data._retry import retry
from standard_quant_tools.data.base import DataProvider, FinancialRatios, TickerInfo
from standard_quant_tools.data.databento import (
    CONSOLIDATED_START,
    DATASET_CONSOLIDATED,
    DATASET_DEPTH,
    DATASET_FUTURES,
    DATASET_NASDAQ_BASIC,
    DATASET_OPTIONS,
    DATASET_SUMMARY,
    SUMMARY_SCHEMAS,
    SUMMARY_START,
    normalize_book,
    normalize_mbo,
    normalize_quotes,
    normalize_trades,
    print_counts,
    timestamp_source,
)
from standard_quant_tools.data.metadata import DataSetMetadata
from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    InvalidSymbolError,
    NonRetryableAPIError,
    ValidationError,
)

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

#: A plain US equity ticker, and a share class. The class is sent to the
#: vendor in its dotted spelling (`BRK.B`) whichever separator the caller
#: used: '-' is Yahoo's class separator and '/' Bloomberg's.
_EQUITY_RE = re.compile(r"^[A-Z]{1,5}$")
_CLASS_RE = re.compile(r"^([A-Z]{1,5})[.\-/]([A-Z])$")
#: A ticker with a DOTTED exchange suffix, the Yahoo and Reuters convention
#: (`GOOG.L`, `0700.HK`, `IBM.N`). The root may be digits (`7203.T`).
_EXCHANGE_SUFFIX_RE = re.compile(r"^([A-Z0-9]{1,8})\.([A-Z]{1,3})$")
#: Dotted suffixes that name a non-US listing, and the venue each names.
#: A single letter that is also a US share-class letter is deliberately
#: absent -- BRK.A, BF.B, MKC.V and units such as `.U` are US securities this
#: provider serves -- which is why Toronto's venture board (`.V`) and the
#: Reuters code for NYSE American (`.A`) are not refused here.
_FOREIGN_SUFFIXES: Dict[str, str] = {
    "L": "London Stock Exchange",
    "IL": "London Stock Exchange's international order book",
    "T": "Tokyo Stock Exchange",
    "F": "Frankfurt Stock Exchange",
    "DE": "Xetra",
    "S": "SIX Swiss Exchange",
    "SW": "SIX Swiss Exchange",
    "PA": "Euronext Paris",
    "AS": "Euronext Amsterdam",
    "BR": "Euronext Brussels",
    "LS": "Euronext Lisbon",
    "IR": "Euronext Dublin",
    "MI": "Borsa Italiana",
    "MC": "Bolsa de Madrid",
    "ST": "Nasdaq Stockholm",
    "CO": "Nasdaq Copenhagen",
    "HE": "Nasdaq Helsinki",
    "OL": "Oslo Bors",
    "VI": "Vienna Stock Exchange",
    "WA": "Warsaw Stock Exchange",
    "HK": "Hong Kong Stock Exchange",
    "SS": "Shanghai Stock Exchange",
    "SZ": "Shenzhen Stock Exchange",
    "TW": "Taiwan Stock Exchange",
    "KS": "Korea Exchange",
    "KQ": "KOSDAQ",
    "NS": "National Stock Exchange of India",
    "BO": "Bombay Stock Exchange",
    "SI": "Singapore Exchange",
    "AX": "Australian Securities Exchange",
    "NZ": "New Zealand Exchange",
    "TO": "Toronto Stock Exchange",
    "SA": "B3 (Sao Paulo)",
    "MX": "Mexican Stock Exchange",
    "JO": "Johannesburg Stock Exchange",
    "TA": "Tel Aviv Stock Exchange",
}
#: Reuters instrument-code suffixes for US venues: the listing IS one this
#: provider serves, under its plain ticker.
_US_RIC_SUFFIXES: Dict[str, str] = {"N": "NYSE", "O": "Nasdaq", "OQ": "Nasdaq"}
#: The derivatives grammar. `ES.c.0` / `ES.n.1` / `ES.v.0` are Databento's
#: continuous symbols (calendar, open-interest and volume rolls), `ES.FUT`
#: and `ES.OPT` the parent symbols, `ESZ6` / `ESZ26` a contract, and an
#: OSI string (`AAPL  240119C00190000`) an option. A bare root such as `ES`
#: matches the equity ticker rule too, which is exactly the ambiguity
#: this grammar exists to refuse rather than resolve.
_CONTINUOUS_RE = re.compile(r"^([A-Z0-9]{1,4})\.([CNV])\.(\d{1,2})$")
_PARENT_RE = re.compile(r"^([A-Z0-9]{1,4})\.(FUT|OPT)$")
#: A contract: root, month code, year. The root may START with a digit (the
#: CME currency roots `6E`, `6J`, `6B`, ...) or carry one inside (`M2K`,
#: `SR3`); it used to have to start with a letter, so `6EZ5` and `M2KZ5`
#: were refused while `6E.c.0` resolved -- with a message naming "a
#: contract like 'ESZ6'", the shape that had just been given.
_CONTRACT_RE = re.compile(r"^([0-9]?[A-Z][A-Z0-9]{0,3})([FGHJKMNQUVXZ])(\d{1,2})$")
_OSI_RE = re.compile(r"^([A-Z]{1,6})\s*(\d{6})([CP])(\d{8})$")
#: Futures roots that are also plausible equity tickers. A bare one of these
#: is refused as ambiguous; `ES~equity` names the equity reading on purpose.
_FUTURES_ROOTS = frozenset(
    {
        "ES",
        "NQ",
        "YM",
        "RTY",
        "MES",
        "MNQ",
        "M2K",
        "MYM",
        "CL",
        "NG",
        "HO",
        "RB",
        "BZ",
        "MCL",
        "GC",
        "SI",
        "HG",
        "PL",
        "PA",
        "MGC",
        "SIL",
        "ZB",
        "ZN",
        "ZF",
        "ZT",
        "UB",
        "TN",
        "SR3",
        "ZQ",
        "ZC",
        "ZS",
        "ZW",
        "ZM",
        "ZL",
        "KE",
        "LE",
        "HE",
        "GF",
        "6E",
        "6J",
        "6B",
        "6A",
        "6C",
        "6S",
        "6M",
        "6N",
        "VX",
        "BTC",
        "ETH",
        "MBT",
    }
)


class SymbolRoute(NamedTuple):
    """What a symbol resolves to: the vendor spelling, the symbology it is
    in, and the family that decides which datasets can answer."""

    raw: str
    stype_in: str
    family: str


#: How a vendor failure is read when it carries no HTTP status -- a stub, or
#: an error raised before a response. The client's HTTP errors carry
#: `http_status`, and that is read first: matching words in the message
#: made a 401 an entitlement denial (the marker "auth"), and a 500 whose
#: request id happened to contain "403" a permanent one. Whole words only,
#: for the same reason. See the CHANGELOG entry of 2026-09-28.
_AUTH_TEXT_RE = re.compile(
    r"\b401\b|\bunauthori[sz]ed\b|\bauth_authentication_failed\b"
    r"|\bauthentication failed\b"
)
_DENIAL_TEXT_RE = re.compile(
    r"\b403\b|\bforbidden\b|\bnot_entitled\b|\bentitlement\b|\blicen[cs]e\b"
)

#: The daily-schema finalization error, by the text Databento returns.
_UNFINALIZED_MARKERS = ("available_end", "not_fully_available")

#: How many days to walk the end back before giving up on the daily lag.
_FINALIZATION_ATTEMPTS = 6

#: The width of one intraday bar. The vendor's range is half-open and this
#: library's end is inclusive, so an explicit intraday end is extended by
#: one bar before it is sent: a 14:00 to 14:10 request at one minute used
#: to stop at 14:09, because the 14:10 bar lies at the excluded edge, and
#: no trim can restore a bar the vendor never sent. A bare-date end is
#: already pushed to the next midnight and needs nothing. Tick schemas stay
#: half-open, as documented.
_BAR_WIDTH: Dict[str, timedelta] = {
    "ohlcv-1s": timedelta(seconds=1),
    "ohlcv-1m": timedelta(minutes=1),
    "ohlcv-1h": timedelta(hours=1),
}

#: CME Globex trade dates. The trading day opens at 17:00 Chicago time on
#: the evening before the date it belongs to and closes by 16:00 on it, so
#: shifting a Chicago instant forward seven hours lands every print on its
#: trade date, across both daylight-saving transitions.
_CME_TIMEZONE = "America/Chicago"
_CME_ROLL = pd.Timedelta(hours=7)
_CME_SESSION_LABEL = (
    "CME trade date: 17:00 America/Chicago on the prior evening to 16:00 "
    "on the date, aggregated from hourly bars"
)


def _failure_kind(exc: BaseException) -> str:
    """
    'auth', 'denied' or 'other' for a failed vendor call.

    A 401 is a bad credential and fails every dataset alike, so it is
    raised at once and named. A 403 is the subscription declining one
    dataset, which is worth remembering so the next request goes straight
    to a feed that answers. Everything else -- a 5xx, a timeout, a 422 --
    says nothing permanent, and remembering it as a denial would retire a
    healthy feed for the life of the provider.
    """
    status = getattr(exc, "http_status", None)
    try:
        code = int(status) if status is not None else None
    except (TypeError, ValueError):
        code = None
    if code == 401:
        return "auth"
    if code == 403:
        return "denied"
    if code is not None:
        return "other"
    text = str(exc).lower()
    if _AUTH_TEXT_RE.search(text):
        return "auth"
    if _DENIAL_TEXT_RE.search(text):
        return "denied"
    return "other"


def _rejected_key(exc: BaseException, where: str) -> NonRetryableAPIError:
    """The refusal for a credential the vendor rejected, naming the variable."""
    return NonRetryableAPIError(
        f"Databento rejected DATABENTO_API_KEY (HTTP 401) on {where}: {exc}. "
        "The key is missing, mistyped, revoked or expired; set a valid one "
        "in the environment. This is not a coverage or entitlement answer "
        "-- no dataset was judged -- and retrying with the same key cannot "
        "succeed."
    )


def _not_found_symbols(store: Any) -> Set[str]:
    """The request's symbols the vendor's symbology could not resolve.

    Read from the store's own symbology report, which the client keeps
    beside the records and which this provider used to discard with the
    store. Defensive, because a stub or an older client may carry neither
    `symbology` nor `metadata.not_found`.
    """
    found: Any = None
    try:
        symbology = getattr(store, "symbology", None)
        if isinstance(symbology, Mapping):
            found = symbology.get("not_found")
        if found is None:
            found = getattr(getattr(store, "metadata", None), "not_found", None)
    except Exception:  # noqa: BLE001 - a report that cannot be read says nothing
        return set()
    try:
        return {str(s) for s in (found or ())}
    except TypeError:
        return set()


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


def _record(
    symbol: str,
    start_date: Any,
    end_date: Any,
    what: str,
    dataset: str,
    frame: pd.DataFrame,
) -> None:
    """One line per fetch into the open decision record, if there is one:
    which dataset answered and a digest of what it said, so a replay can
    tell a restated feed from a changed tool."""
    audit.record_data_access(
        symbol,
        str(start_date),
        str(end_date),
        what,
        source=f"databento:{dataset}",
        content_hash=audit.hash_dataframe(frame),
    )


def _refuse_exchange_suffix(symbol: str, text: str) -> None:
    """
    Refuse a ticker carrying a dotted exchange suffix, naming the listing.

    Before this refusal, a single-letter suffix read as a share class and
    was folded into the ticker: `GOOG.L` came back as Alphabet class A
    (`GOOGL`) and `BP.L` as `BPL`, another company, with no warning. The
    suffix names a venue, not a class, and this provider's equity datasets
    are US venues, so the honest answer is a refusal that says which
    listing was asked for and how to ask for the one this provider has.
    """
    match = _EXCHANGE_SUFFIX_RE.match(text)
    if match is None:
        return
    root, suffix = match.groups()
    if suffix in _US_RIC_SUFFIXES:
        raise ValidationError(
            f"{symbol!r} is a Reuters code for the {_US_RIC_SUFFIXES[suffix]} "
            f"listing of {root!r}, and this provider serves US listings under "
            f"their plain ticker: ask for {root!r}. The '.{suffix}' exchange "
            "suffix is refused rather than guessed at, because reading it as "
            "a share class named a different symbol."
        )
    venue = _FOREIGN_SUFFIXES.get(suffix)
    if venue is None:
        return
    remedies = [
        f"for the {venue} listing use a provider that carries it (provider "
        "'yfinance' takes Yahoo exchange suffixes)",
        "for a US share class use its class letter, as in 'BRK.B'",
    ]
    if _EQUITY_RE.match(root):
        remedies.insert(0, f"for the US listing ask for {root!r}, a separate security")
    remedy = "; ".join(remedies)
    raise ValidationError(
        f"{symbol!r} names a listing on the {venue} (the '.{suffix}' exchange "
        "suffix), and this provider serves US listings only. It will not read "
        "the suffix as a share class, which folded 'GOOG.L' into 'GOOGL', "
        f"another security. {remedy[0].upper()}{remedy[1:]}."
    )


def _with_attrs(frame: pd.DataFrame, attrs: Dict[str, Any]) -> pd.DataFrame:
    """`DataFrame.copy()` keeps attrs in recent pandas and not in older ones;
    the served dataset must survive either way."""
    frame.attrs.update(dict(attrs))
    return frame


def _cme_trade_date_bars(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Hourly bars (a tz-aware UTC index) aggregated into CME trade dates.

    The vendor's `ohlcv-1d` is a UTC day, and a Globex session is not one:
    it opens at 17:00 Chicago on the evening before its date and closes by
    16:00, so a UTC day holds the tail of one session and the start of the
    next. A week came back as six bars -- a Sunday-evening fragment at a
    couple of percent of a day's volume among them -- and each close was
    the price two or three hours into the following session, not the
    session's last trade. A continuous series rolled on a Sunday.

    Each hourly bar is placed on its trade date by shifting its Chicago
    wall-clock time forward seven hours (so 17:00 opens the next date,
    across both daylight-saving changes), and a Saturday or Sunday date --
    a print after Friday's close -- is carried to Monday, the date that
    session belongs to. A holiday's abbreviated session keeps the date the
    clock gives it. Open is the first bar's open, High and Low the extremes,
    Close the LAST TRADE of the date -- not the settlement price, which is
    a separate publication -- and Volume the sum.
    """
    if hourly.empty:
        return hourly
    local = pd.DatetimeIndex(hourly.index).tz_convert(_CME_TIMEZONE).tz_localize(None)
    dates = (local + _CME_ROLL).normalize()
    weekday = np.asarray(dates.weekday)
    carry = np.where(weekday == 5, 2, np.where(weekday == 6, 1, 0))
    dates = dates + pd.to_timedelta(carry, unit="D")
    grouped = hourly.groupby(dates, sort=True)
    out = pd.DataFrame(
        {
            "Open": grouped["Open"].first(),
            "High": grouped["High"].max(),
            "Low": grouped["Low"].min(),
            "Close": grouped["Close"].last(),
            "Volume": grouped["Volume"].sum(),
        }
    )
    out.index = pd.DatetimeIndex(out.index)
    out.index.name = None
    return out


class DatabentoProvider(DataProvider):
    """Databento Historical, honouring this library's provider contract."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        client: Any = None,
        dataset: Optional[str] = None,
        depth_dataset: Optional[str] = None,
    ) -> None:
        # The settings below are read through env_str, like every setting in
        # the library: blank is unset, and a local .env is loaded first, so
        # a key or dataset supplied there is honoured here too.
        self._api_key = api_key or env_str("DATABENTO_API_KEY") or ""
        # Injectable so the operational logic above -- dataset preference,
        # the finalization walk-back, denial memory, symbol mapping -- is
        # testable without a key, a network or an entitlement. Those are
        # exactly the parts that are expensive to get wrong and impossible
        # to exercise against a live API in a test suite.
        self._client = client
        self._client_failed = False
        self._client_error: Optional[str] = None
        self._lock = threading.Lock()
        self._ranges: Dict[str, Tuple[datetime, datetime]] = {}
        self._denied: Set[str] = set()
        # Why a range lookup last failed, per dataset: a refusal that says
        # "no dataset covers that range" when the lookup itself failed is
        # the wrong answer to act on.
        self._range_errors: Dict[str, str] = {}
        self._dataset = dataset or env_str("DATABENTO_DATASET") or DATASET_NASDAQ_BASIC
        self._depth_dataset = (
            depth_dataset or env_str("DATABENTO_DEPTH_DATASET") or DATASET_DEPTH
        )
        self._futures_dataset = env_str("DATABENTO_FUTURES_DATASET") or DATASET_FUTURES
        self._options_dataset = env_str("DATABENTO_OPTIONS_DATASET") or DATASET_OPTIONS
        # The session cache is keyed per instance, like yfinance's: a fresh
        # provider never reuses another's result, which is what lets a
        # replay construct one to re-read from disk and detect tampering.
        self._instance_token = uuid.uuid4()

    # ── client and datasets ──────────────────────────────────────────
    @property
    def is_configured(self) -> bool:
        """Whether a key is present. The constructor defers the key check to
        the first fetch, so `describe_data_capabilities` reported an
        unconfigured provider as available (findings, the plumbing)."""
        return bool(self._api_key)

    @property
    def unconfigured_reason(self) -> Optional[str]:
        if self._api_key:
            return None
        return (
            "DATABENTO_API_KEY is not set. The provider constructs without a "
            "key and fails on its first fetch; set the variable to use it."
        )

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is not None or self._client_failed:
                if self._client is None:
                    # The reason it failed the first time, every time: the
                    # retry layer above re-asks, and the answer must not
                    # decay into "it failed earlier".
                    raise APIError(
                        self._client_error
                        or "the Databento client could not be constructed."
                    )
                return self._client
            if not self._api_key:
                self._client_failed = True
                self._client_error = (
                    "DATABENTO_API_KEY is not set. This provider reads its "
                    "credential from the environment and never from a spec "
                    "or a tool argument, because a DatasetSpec is persisted "
                    "to disk, hashed into a model's lineage and written into "
                    "decision records."
                )
                raise APIError(self._client_error)
            try:
                import databento as db
            except ImportError as exc:
                self._client_failed = True
                self._client_error = (
                    "the `databento` package is not installed. Install it "
                    "with `pip install databento` to use provider "
                    "'databento'."
                )
                raise APIError(self._client_error) from exc
            try:
                self._client = db.Historical(self._api_key)
            except Exception as exc:  # noqa: BLE001 - one refusal, not a trace
                self._client_failed = True
                self._client_error = f"Databento client construction failed: {exc}"
                raise APIError(self._client_error) from exc
            return self._client

    @staticmethod
    def _is_denial(exc: Exception) -> bool:
        """Whether a failure is the subscription declining the dataset (a
        403), the only failure worth remembering. See `_failure_kind`."""
        return _failure_kind(exc) == "denied"

    def _available_range(self, dataset: str) -> Optional[Tuple[datetime, datetime]]:
        """
        What the dataset actually covers, resolved once and remembered.

        Every request is clamped to this. Without it a request whose end is
        wall-clock `now` asks for data Databento has not published -- which
        is every request made on a weekend, and it fails rather than
        returning the last session.

        A REJECTED KEY IS RAISED HERE, BY NAME. This free lookup is the
        first call every fetch makes, and it used to swallow every failure:
        a 401 was remembered as an entitlement denial on each dataset in
        turn and the caller was told "No dataset covers that range" -- the
        commonest misconfiguration, reported as a date problem.
        """
        with self._lock:
            if dataset in self._ranges:
                return self._ranges[dataset]
        client = self._get_client()
        try:
            meta = client.metadata.get_dataset_range(dataset=dataset)
        except Exception as exc:  # noqa: BLE001
            kind = _failure_kind(exc)
            if kind == "auth":
                raise _rejected_key(exc, f"the range lookup for {dataset}") from exc
            logger.warning("databento range lookup failed for %s: %s", dataset, exc)
            with self._lock:
                if kind == "denied":
                    self._denied.add(dataset)
                self._range_errors[dataset] = str(exc)
            return None
        try:
            start = datetime.fromisoformat(str(meta["start"]).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(meta["end"]).replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("databento range for %s is unreadable: %s", dataset, exc)
            with self._lock:
                self._range_errors[dataset] = f"an unreadable range ({exc})"
            return None
        with self._lock:
            self._ranges[dataset] = (start, end)
            self._range_errors.pop(dataset, None)
        return (start, end)

    def _bar_datasets(
        self, schema: str = "ohlcv-1d", start: Optional[datetime] = None
    ) -> List[str]:
        """
        Bar datasets in preference order, for a schema and a window.

        DAILY: the summary feed first wherever it reaches, because it is the
        consolidated close and volume exactly; then the consolidated-symbol
        sample feed, then the venue feed, then the depth venue whose bars
        reach furthest back. INTRADAY: the venue feeds only, in the same
        order the tick methods use, so the minute bars and the trades for
        one window come from ONE tape (D11) -- the sample feed is not a tape
        and a minute of it reconciles with nothing.
        """
        override = env_str("DATABENTO_OHLCV_DATASET")
        if override:
            candidates = [override, self._depth_dataset]
        elif schema in SUMMARY_SCHEMAS:
            candidates = [
                DATASET_SUMMARY,
                DATASET_CONSOLIDATED,
                self._dataset,
                self._depth_dataset,
            ]
            if start is not None and start < SUMMARY_START:
                candidates.remove(DATASET_SUMMARY)
        else:
            candidates = [self._dataset, self._depth_dataset]
        seen: List[str] = []
        for name in candidates:
            if name and name not in seen and name not in self._denied:
                seen.append(name)
        return seen

    def _datasets_for(
        self, family: str, schema: str, start: Optional[datetime] = None
    ) -> List[str]:
        """The datasets that can answer a symbol family, in preference order."""
        if family == "future":
            return [self._futures_dataset]
        if family == "option":
            return [self._options_dataset]
        return self._bar_datasets(schema, start)

    def _tick_datasets(self, symbol: str) -> List[str]:
        """Trades and quotes: the same venue order as intraday bars."""
        route = self.resolve_symbol(symbol)
        if route.family != "equity":
            return self._datasets_for(route.family, "trades")
        return [
            d for d in (self._dataset, self._depth_dataset) if d not in self._denied
        ]

    def _depth_datasets(self, symbol: str) -> List[str]:
        route = self.resolve_symbol(symbol)
        if route.family != "equity":
            return self._datasets_for(route.family, "mbp-10")
        return [self._depth_dataset]

    #: Schemas whose routing is NOT the bar routing. Depth and
    #: order-by-order come from the depth venue alone; the tape schemas
    #: follow the same venue order intraday bars do, so bars and ticks for
    #: one window come from one tape.
    DEPTH_SCHEMAS = ("mbp-10", "mbo")
    TICK_SCHEMAS = ("trades", "tbbo", "mbp-1", "bbo-1s", "bbo-1m")

    def datasets_for_schema(
        self,
        symbol: str,
        schema: str,
        start_date: Optional[Union[str, datetime]] = None,
    ) -> List[str]:
        """
        The datasets a request would be offered to, in preference order.

        The same three routers the fetch methods use, chosen by schema
        rather than by method name, plus the two window guards `_fetch`
        applies before it asks anyone: the sample feed does not exist
        before its start date and the summary feed answers daily bars
        only. What is NOT applied is the live coverage check -- that costs
        a metadata round trip per dataset and belongs to the caller that
        wants it, which is why `get_dataset_coverage` is a separate call.

        This is a PREVIEW of the routing, so it can say which feed would
        answer without transferring anything. Preferring the first entry
        blind would be wrong for a window the preferred feed does not
        reach; pair it with the coverage windows to choose.
        """
        if schema in self.DEPTH_SCHEMAS:
            return self._depth_datasets(symbol)
        if schema in self.TICK_SCHEMAS:
            return self._tick_datasets(symbol)
        route = self.resolve_symbol(symbol)
        start = (
            _to_utc(start_date, end_of_day=False) if start_date is not None else None
        )
        candidates = self._datasets_for(route.family, schema, start)
        return [name for name in candidates if self._window_admits(name, schema, start)]

    @staticmethod
    def _window_admits(dataset: str, schema: str, start: Optional[datetime]) -> bool:
        """
        The two window rules applied before anyone is asked.

        The sample feed does not exist before `CONSOLIDATED_START`, so
        asking is a guaranteed miss and a wasted round trip; the summary
        feed answers daily bars only, from `SUMMARY_START`. One definition
        for the fetch, the routing preview and the disk-cache lookup, so the
        three cannot disagree about which feed answers a window. A missing
        start (a preview with no window) is admitted.
        """
        if (
            dataset == DATASET_CONSOLIDATED
            and start is not None
            and start < CONSOLIDATED_START
        ):
            return False
        if dataset == DATASET_SUMMARY and (
            schema not in SUMMARY_SCHEMAS
            or (start is not None and start < SUMMARY_START)
        ):
            return False
        return True

    def _first_to_ask(
        self, family: str, schema: str, start: datetime, end: datetime
    ) -> Optional[str]:
        """
        The dataset the routing asks first for this window, from what this
        instance knows WITHOUT a network call: the window rules, remembered
        entitlement denials and, for a dataset whose range lookup has
        succeeded, its published coverage.

        This is the feed that would answer the request now, and the only
        one whose disk-cache entry may answer it before anyone is asked.
        A coverage this instance has not yet looked up counts as covering:
        on a cache miss `_fetch` looks it up, and a dataset it turns out
        not to cover is passed over there.
        """
        for dataset in self._datasets_for(family, schema, start):
            if dataset in self._denied or not self._window_admits(
                dataset, schema, start
            ):
                continue
            with self._lock:
                span = self._ranges.get(dataset)
            if span is not None and self._clamp(span, start, end) is None:
                continue
            return dataset
        return None

    def _known_datasets(self) -> List[str]:
        """Every dataset this provider is configured to reach, deduplicated."""
        ordered = [
            DATASET_SUMMARY,
            DATASET_CONSOLIDATED,
            self._dataset,
            self._depth_dataset,
            self._futures_dataset,
            self._options_dataset,
        ]
        seen: List[str] = []
        for name in ordered:
            if name and name not in seen:
                seen.append(name)
        return seen

    @staticmethod
    def resolve_symbol(symbol: str) -> SymbolRoute:
        """
        What a symbol names: its vendor spelling, symbology and family.

        SHARE CLASSES KEEP THE DOT. `BRK.B`, `BRK-B`, `BRK/B` and `brk.b`
        all resolve to raw `BRK.B`, which is how Databento's Historical
        symbology spells a class share (`BRKB` is `not_found` there). This
        provider used to send the undotted concatenation, so every share
        class failed after a dozen requests. Futures and options are named
        by their own grammar -- continuous, parent, contract, OSI -- and
        routed to their own datasets.

        AN EXCHANGE SUFFIX IS REFUSED BY NAME. `GOOG.L`, `BP.L`, `SONY.T`,
        `0700.HK` and `RY.TO` name non-US listings, and `IBM.N` / `AAPL.O`
        are Reuters codes for US ones. A single-letter suffix used to read
        as a share class and fold into the ticker, so `GOOG.L` returned
        Alphabet class A. The refusal names the listing and the spelling
        this provider does serve; see `_refuse_exchange_suffix`.

        A BARE FUTURES ROOT IS REFUSED. `ES` and `CL` are equity tickers as
        well as roots, and this provider used to resolve them to the
        equity: `get_ohlcv("CL")` returned Colgate-Palmolive at 87 where
        crude was at 70-90 a barrel, with no warning (D5). Spell the
        instrument -- `ES.c.0` for the front continuous contract, `ESZ6`
        for a contract, `ES.FUT` for the parent, or `ES~equity` for the
        ticker -- and it is one thing.
        """
        text = str(symbol).strip().upper()
        if text.endswith("~EQUITY"):
            text = text[: -len("~EQUITY")]
            # The escape hatch names the equity reading of a futures root;
            # it does not make an exchange suffix a share class.
            _refuse_exchange_suffix(symbol, text)
            match = _CLASS_RE.match(text)
            if match:
                return SymbolRoute(
                    f"{match.group(1)}.{match.group(2)}", "raw_symbol", "equity"
                )
            if _EQUITY_RE.match(text):
                return SymbolRoute(text, "raw_symbol", "equity")
            raise ValidationError(
                f"{symbol!r} names an equity by suffix but {text!r} is not a "
                "1-5 letter ticker or a share class such as 'BRK.B'."
            )
        if _CONTINUOUS_RE.match(text):
            return SymbolRoute(
                text.replace(".C.", ".c.").replace(".N.", ".n.").replace(".V.", ".v."),
                "continuous",
                "future",
            )
        if _PARENT_RE.match(text):
            return SymbolRoute(
                text, "parent", "option" if text.endswith(".OPT") else "future"
            )
        if _OSI_RE.match(text.replace(" ", "")) or _OSI_RE.match(text):
            match = _OSI_RE.match(text.replace(" ", ""))
            assert match is not None
            root, date, kind, strike = match.groups()
            return SymbolRoute(f"{root:<6}{date}{kind}{strike}", "raw_symbol", "option")
        if _CONTRACT_RE.match(text) and not _EQUITY_RE.match(text):
            return SymbolRoute(text, "raw_symbol", "future")
        if text in _FUTURES_ROOTS and _EQUITY_RE.match(text):
            raise ValidationError(
                f"{symbol!r} is ambiguous: a futures root and a US equity ticker "
                "at once, and this provider will not pick one for you. Spell "
                f"the instrument: '{text}.c.0' (front continuous contract), "
                f"'{text}Z6'-style for a contract, '{text}.FUT' (parent), or "
                f"'{text}~equity' for the ticker."
            )
        if _EQUITY_RE.match(text):
            return SymbolRoute(text, "raw_symbol", "equity")
        # Before the share-class rule: `GOOG.L` matches it, and reading the
        # suffix as a class is the mistake this refusal exists to prevent.
        _refuse_exchange_suffix(symbol, text)
        match = _CLASS_RE.match(text)
        if match:
            return SymbolRoute(
                f"{match.group(1)}.{match.group(2)}", "raw_symbol", "equity"
            )
        raise ValidationError(
            f"{symbol!r} is not a symbol this provider can map to a Databento "
            "raw_symbol. Expected a 1-5 letter ticker, a share class like "
            "'BRK.B', a continuous future like 'ES.c.0', a contract like "
            "'ESZ6' or '6EZ5', a parent like 'ES.FUT', or an OSI option "
            "string."
        )

    @staticmethod
    def to_raw_symbol(symbol: str) -> str:
        """This library's symbol as Databento's raw spelling, any family."""
        return DatabentoProvider.resolve_symbol(symbol).raw

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
        return self._clamp(span, start, end)

    @staticmethod
    def _clamp(
        span: Tuple[datetime, datetime], start: datetime, end: datetime
    ) -> Optional[Tuple[datetime, datetime]]:
        """A window clamped to a published span, or None when it starts
        before the span or ends where the span has nothing."""
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
        self,
        dataset: str,
        schema: str,
        raw: str,
        start: datetime,
        end: datetime,
        stype_in: str = "raw_symbol",
        not_found: Optional[Set[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """One request, with the daily-finalization walk-back.

        `not_found`, when given, collects the symbols the vendor's
        symbology reported it could not resolve for this request -- the
        one fact that tells an empty answer about a bad symbol from an
        empty answer about a quiet window.
        """
        client = self._get_client()

        def _call(request_end: datetime, fmt: str) -> pd.DataFrame:
            store = client.timeseries.get_range(
                dataset=dataset,
                schema=schema,
                symbols=[raw],
                stype_in=stype_in,
                start=start.strftime(fmt),
                end=request_end.strftime(fmt),
            )
            if not_found is not None:
                not_found.update(_not_found_symbols(store))
            return store.to_df()

        if schema == "ohlcv-1d":
            # Databento finalizes daily bars a day or two behind the live
            # feed, while the dataset range reports the LIVE edge -- so the
            # honest end lands in the unfinalized tail and is refused. Walk
            # it back, but only on THAT error: anything else is a real
            # failure and re-asking would hide it.
            # The end is already the next midnight for a bare date (see
            # `_to_utc`), and Databento's day-granular end is EXCLUSIVE, so
            # the honest daily request ends there: round UP to a whole day
            # and add nothing. Adding a day here asked for one bar past the
            # inclusive end, and that bar was tomorrow (D1) -- and cost a
            # guaranteed 422 on every daily request before the walk-back.
            attempt = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
            if attempt < end:
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
        stored: Optional[Callable[[str], Optional[pd.DataFrame]]] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Try each dataset in order; return the first that answers.

        `stored`, when given, is asked for a dataset's stored answer at the
        moment that dataset is the one this request would be sent to --
        past the window rules, the remembered denials and its published
        coverage -- and a frame it returns is served instead of a request.
        That is the only point at which a lesser feed's stored answer can
        be the right one: every better feed has just been passed over.
        """
        route = self.resolve_symbol(symbol)
        raw = route.raw
        start = _to_utc(start_date, end_of_day=False)
        end = _to_utc(end_date, end_of_day=True)
        if end <= start:
            raise ValidationError(
                f"empty window: start {start_date!r} is not before end "
                f"{end_date!r} (the end date is INCLUSIVE, so a same-day "
                "request is valid and this is not one)."
            )
        width = _BAR_WIDTH.get(schema)
        if width is not None and end == _to_utc(end_date, end_of_day=False):
            # An explicit intraday end, sent to a half-open range: one bar
            # more, so the bar AT the end is served (see `_BAR_WIDTH`). It
            # is added before the edge clamp in `_range`, which still pulls
            # it back to what the dataset has published.
            end = end + width

        tried: List[str] = []
        passed: Dict[str, str] = {}
        candidates = (
            datasets
            if datasets is not None
            else self._datasets_for(route.family, schema, start)
        )
        for dataset in candidates:
            if dataset in self._denied:
                passed[dataset] = "declined by the subscription (HTTP 403)"
                continue
            if not self._window_admits(dataset, schema, start):
                continue
            window = self._range(dataset, start, end)
            if window is None:
                passed[dataset] = self._why_passed_over(dataset)
                continue
            if stored is not None:
                answer = stored(dataset)
                if answer is not None:
                    return answer, dataset
            tried.append(dataset)
            unresolved: Set[str] = set()
            try:
                frame = self._get_range(
                    dataset,
                    schema,
                    raw,
                    *window,
                    stype_in=route.stype_in,
                    not_found=unresolved,
                )
            except Exception as exc:  # noqa: BLE001
                kind = _failure_kind(exc)
                if kind == "auth":
                    raise _rejected_key(
                        exc, f"a {schema} request to {dataset}"
                    ) from exc
                logger.warning(
                    "databento %s %s on %s failed: %s", symbol, schema, dataset, exc
                )
                if kind == "denied":
                    with self._lock:
                        self._denied.add(dataset)
                    passed[dataset] = "declined by the subscription (HTTP 403)"
                continue
            if frame is not None and len(frame):
                return frame, dataset
            if raw in unresolved:
                # THE VENDOR SAID THE SYMBOL DOES NOT EXIST, and asking the
                # next feed, then the retry layer asking all of them twice
                # more, turned that one answer into a dozen requests and a
                # generic error. The US equity feeds share one symbology,
                # and every other family has a single dataset.
                raise InvalidSymbolError(
                    f"{symbol!r} (sent to Databento as {raw!r}, symbology "
                    f"{route.stype_in}) did not resolve on {dataset} between "
                    f"{start:%Y-%m-%d} and {end:%Y-%m-%d}: the vendor's "
                    "symbology reports it not found. Check the spelling -- a "
                    "share class is dotted ('BRK.B'), a futures contract "
                    "carries a one-digit year ('ESZ6') -- and that the "
                    "instrument was listed in that window."
                )

        span = f"({schema}) between {start:%Y-%m-%d} and {end:%Y-%m-%d}"
        reasons = "; ".join(f"{name}: {why}" for name, why in passed.items())
        if tried:
            raise APIError(
                f"Databento returned no {what} for {symbol} {span}. Datasets "
                f"tried: {tried}." + (f" Passed over: {reasons}." if reasons else "")
            )
        if passed and all("HTTP 403" in why for why in passed.values()):
            # An entitlement answer, not a coverage one, and asking again
            # cannot change it: every dataset that could answer is one the
            # subscription has declined.
            raise NonRetryableAPIError(
                f"Databento's subscription declined every dataset that could "
                f"serve {what} for {symbol} {span}: {sorted(passed)} (HTTP "
                "403). This is an entitlement problem, not a date-range one: "
                "add the dataset to the subscription, or point the provider "
                "at one it includes (DATABENTO_DATASET, "
                "DATABENTO_DEPTH_DATASET, DATABENTO_OHLCV_DATASET)."
            )
        raise APIError(
            f"Databento returned no {what} for {symbol} {span}. "
            + (
                f"No dataset was asked: {reasons}."
                if reasons
                else "No dataset covers that range."
            )
        )

    def _why_passed_over(self, dataset: str) -> str:
        """Why `_range` declined a dataset, for the refusal that names it."""
        with self._lock:
            span = self._ranges.get(dataset)
            error = self._range_errors.get(dataset)
            denied = dataset in self._denied
        if denied:
            return "declined by the subscription (HTTP 403)"
        if span is not None:
            first, last = span
            return (
                f"it publishes {first:%Y-%m-%d} to {last:%Y-%m-%d}, which does "
                "not contain the window's start"
            )
        return f"its coverage could not be looked up ({error or 'no answer'})"

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
        # The same seams every other provider passes: the session cache
        # (keyed per instance), the retry layer, the Parquet cache keyed by
        # the dataset that answered, and the index normaliser. Databento
        # bypassed all four, which is why its index was the only tz-aware
        # one in the library (D2), three identical requests were three
        # metered fetches, and a daily request kept one bar too many (D1).
        start_str = _norm_cache_bound(start_date, interval)
        end_str = _norm_cache_bound(end_date, interval)
        key = (
            "databento",
            self._instance_token,
            str(symbol).strip().upper(),
            start_str,
            end_str,
            interval,
        )
        cached = _session_cache_get(key)
        if cached is not None:
            _record(
                symbol,
                start_date,
                end_date,
                interval,
                f"{cached.attrs.get('dataset', '?')}:session_cache",
                cached,
            )
            return _with_attrs(cached.copy(), cached.attrs)
        result, preferred = self._fetch_ohlcv_uncached(
            symbol, start_date, end_date, interval, schema, start_str, end_str
        )
        if preferred:
            # An answer from a lesser feed because a better one failed (or
            # returned nothing) is not kept for the session: the next call
            # on this instance asks the better feed again rather than
            # repeating the degraded answer for an hour.
            _session_cache_set(key, result, end=end_str)
        return _with_attrs(result.copy(), result.attrs)

    @retry(times=3, delay=1)
    def _fetch_ohlcv_uncached(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str,
        schema: str,
        start_str: str,
        end_str: str,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        The bars, and whether they came from the feed the routing prefers
        for this window now (False when a better feed failed and a lesser
        one answered in its place).

        THE DISK CACHE IS READ FOR THE FEED THAT WOULD ANSWER, AND NO OTHER.
        Entries are keyed by the dataset that answered, so the same window
        served by two feeds 30x apart in volume is two files. The lookup
        used to walk every candidate and serve the first file it found, so
        one transient failure on the summary feed -- or one run with
        `DATABENTO_OHLCV_DATASET` set -- left a sample-feed file that every
        later call served for a window the summary feed covered all along.
        Now the preferred feed's entry is read before anything is asked; a
        lesser feed's entry is read only inside `_fetch`, at the moment
        every better feed has been passed over (denied, not covering the
        window, or failing right now), which is when a live request would
        be answered by that lesser feed too.

        A window before `SUMMARY_START` is served by the sample feed by
        policy, every time: its file is the preferred feed's entry for that
        window, and reading it is not a fallback. It is disclosed, though,
        when the summary feed covers part of it (see `_disclose`).

        A FUTURE'S DAILY BAR IS A CME TRADE DATE, built from hourly bars
        (see `_cme_trade_date_bars`), because the vendor's `ohlcv-1d` is a
        UTC day: six bars a week, one of them a Sunday-evening fragment, and
        every close taken two or three hours into the next session.
        """
        route = self.resolve_symbol(symbol)
        start = _to_utc(start_date, end_of_day=False)
        end = _to_utc(end_date, end_of_day=True)
        trade_dates = route.family == "future" and interval == "1d"
        request_schema = "ohlcv-1h" if trade_dates else schema
        # A trade date opens at 17:00 Chicago on the evening before it,
        # which is the previous UTC day: ask from a day earlier and cut the
        # partial first date off after aggregating.
        fetch_start = start - timedelta(days=1) if trade_dates else start_date
        served: Dict[str, pd.DataFrame] = {}

        def _disclosed(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
            return self._disclose(
                frame, symbol, route, interval, schema, start, end, dataset
            )

        def _stored(dataset: str) -> Optional[pd.DataFrame]:
            path = self._bar_cache_path(route, dataset, start_str, end_str, interval)
            # The shared read: an entry that is not a plausible answer is
            # evicted, and a Windows sharing violation is a miss, not a
            # reason to delete a valid entry (see _cache._read_cached_ohlcv).
            frame = _read_cached_ohlcv(path, interval, start_str, end_str)
            if frame is None:
                return None
            frame.attrs["dataset"] = dataset
            frame.attrs["provider"] = "databento"
            _record(
                symbol, start_date, end_date, interval, f"{dataset}:disk_cache", frame
            )
            served[dataset] = frame
            return frame

        first = self._first_to_ask(route.family, schema, start, end)
        if first is not None and _stored(first) is not None:
            return _disclosed(served[first], first), True
        raw_frame, dataset = self._fetch(
            request_schema, symbol, fetch_start, end_date, what="bars", stored=_stored
        )
        # Asked again after the fetch, which has learned denials and
        # coverage: a feed passed over for either is not a failure, and the
        # dataset that answered is then the preferred one.
        preferred = self._first_to_ask(route.family, schema, start, end)
        if dataset != preferred:
            logger.warning(
                "databento %s %s was answered by %s because the preferred %s "
                "failed or returned nothing; not kept for the session, so the "
                "next request asks %s again",
                symbol,
                schema,
                dataset,
                preferred,
                preferred,
            )
        if dataset in served:
            return _disclosed(served[dataset], dataset), dataset == preferred
        out = self._shape_bars(
            raw_frame,
            symbol,
            interval,
            end_date,
            trade_dates_from=start_date if trade_dates else None,
        )
        if trade_dates and out.empty:
            raise DataNotFoundError(
                f"Databento served hourly {symbol} bars around {start_date} to "
                f"{end_date}, and no CME trade date falls inside that window: "
                "a trade date runs from 17:00 Chicago on the evening before "
                "it, so a window of a weekend or a holiday holds none."
            )
        out.attrs["dataset"] = dataset
        out.attrs["provider"] = "databento"
        _disclosed(out, dataset)
        # Into the open decision record, like every other provider's bars.
        _record(symbol, start_date, end_date, interval, dataset, out)
        # Written under the dataset that answered, even when it answered in
        # a failing feed's place: the entry is that dataset's true answer,
        # and the lookup above reads it only when that dataset is the one
        # that would answer.
        path = self._bar_cache_path(route, dataset, start_str, end_str, interval)
        if path is not None and _is_historical(end_date):
            # Never raises: a failed write is logged and its temp removed.
            _write_cached_ohlcv(path, out, interval, start_str, end_str)
        return out, dataset == preferred

    @staticmethod
    def _bar_cache_path(
        route: SymbolRoute, dataset: str, start_str: str, end_str: str, interval: str
    ):
        """
        The disk-cache entry for one dataset's answer to one window.

        Named by the symbol AS SENT TO THE VENDOR, not as the caller spelled
        it: `BRK.B`, `BRK-B` and `BRK/B` are one request and share one
        entry, and a mapping change moves the name with it instead of
        leaving a file named for one spelling holding another instrument's
        bars. The spaces in an OSI option symbol are dropped, because the
        cache refuses a space in a filename and the root and the date
        cannot run together ambiguously.
        """
        return _safe_parquet_path(
            route.raw.replace(" ", ""),
            start_str,
            end_str,
            interval,
            provider=f"databento-{dataset}",
        )

    def _known_start(self, dataset: str, schema: str) -> Optional[datetime]:
        """The first instant a dataset can answer `schema`, from what this
        instance knows without a network call: the two fixed start dates,
        and a published coverage it has already looked up."""
        starts: List[datetime] = []
        if dataset == DATASET_SUMMARY:
            if schema not in SUMMARY_SCHEMAS:
                return None
            starts.append(SUMMARY_START.to_pydatetime())
        if dataset == DATASET_CONSOLIDATED:
            starts.append(CONSOLIDATED_START.to_pydatetime())
        with self._lock:
            span = self._ranges.get(dataset)
        if span is not None:
            starts.append(span[0])
        return max(starts) if starts else None

    def _coverage_downgrade(
        self,
        family: str,
        schema: str,
        start: datetime,
        end: datetime,
        served: str,
    ) -> Optional[Dict[str, str]]:
        """
        The better feed a window was passed over for, because it starts
        inside the window -- or None.

        ONE FEED PER FRAME, AND THE WINDOW'S START CHOOSES IT. A feed that
        begins after the window's start is not asked at all, so a window
        opening one week before the summary feed's first date was served
        whole by the sample feed: the same symbol over overlapping dates,
        27x apart in volume, chosen by where the window started and with
        nothing said. The answer is not changed here -- clamping the start
        would return a shorter window than was asked, silently, and
        stitching the two would put a volume step at the seam -- but it is
        disclosed, with the date to split the request at.
        """
        for dataset in self._datasets_for(family, schema, None):
            if dataset == served:
                return None
            if dataset in self._denied:
                continue
            first = self._known_start(dataset, schema)
            if first is not None and start < first < end:
                return {
                    "preferred": dataset,
                    "covers_from": first.date().isoformat(),
                    "served": served,
                    "advice": (
                        f"split the request at {first.date().isoformat()}: "
                        f"{dataset} answers from that date on, and {served} "
                        "answered the whole window because it starts before"
                    ),
                }
        return None

    def coverage_downgrade(
        self,
        symbol: str,
        schema: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        served: str,
    ) -> Optional[Dict[str, str]]:
        """
        Whether a request served by `served` passes over a better feed that
        covers part of its window -- the disclosure a fetch attaches as
        `attrs["coverage_downgrade"]`, askable before the fetch is made.
        """
        route = self.resolve_symbol(symbol)
        start = _to_utc(start_date, end_of_day=False)
        end = _to_utc(end_date, end_of_day=True)
        return self._coverage_downgrade(route.family, schema, start, end, served)

    def _disclose(
        self,
        frame: pd.DataFrame,
        symbol: str,
        route: SymbolRoute,
        interval: str,
        schema: str,
        start: datetime,
        end: datetime,
        dataset: str,
    ) -> pd.DataFrame:
        """What a bar frame is, on the frame, however it was served: the
        session a futures daily bar spans, and a coverage downgrade."""
        if route.family == "future" and interval == "1d":
            frame.attrs["session"] = _CME_SESSION_LABEL
            frame.attrs["bars_from"] = "ohlcv-1h"
        downgrade = self._coverage_downgrade(route.family, schema, start, end, dataset)
        if downgrade is None:
            frame.attrs.pop("coverage_downgrade", None)
            return frame
        frame.attrs["coverage_downgrade"] = downgrade
        logger.warning(
            "databento %s %s: the window starts before %s covers it (from %s), "
            "so %s answered all of it; %s",
            symbol,
            schema,
            downgrade["preferred"],
            downgrade["covers_from"],
            dataset,
            downgrade["advice"],
        )
        return frame

    def _shape_bars(
        self,
        raw_frame: pd.DataFrame,
        symbol: str,
        interval: str,
        end_date: Union[str, datetime],
        *,
        trade_dates_from: Optional[Union[str, datetime]] = None,
    ) -> pd.DataFrame:
        """The library's column contract, a naive index, integer volume, and
        the inclusive end enforced -- the same shaping every provider does.

        With `trade_dates_from`, `raw_frame` holds hourly bars that are
        aggregated into CME trade dates, and the dates before that start --
        the partial one the widened request reached into -- are cut off.
        """
        bars = self._to_ohlcv(raw_frame, symbol)
        if trade_dates_from is not None:
            bars = _cme_trade_date_bars(bars)
            first_date = pd.Timestamp(
                _to_utc(trade_dates_from, end_of_day=False).date()
            )
            bars = bars[bars.index >= first_date]
        out = _normalize_ohlcv_index(bars, interval)
        # What the venue published: no split or dividend adjustment. The
        # backtest engine's split screen reads this to phrase its warning.
        out.attrs["adjusted"] = False
        if out["Volume"].notna().all():
            # uint64 from the vendor: `Volume.diff()` on it returned
            # 1.8e19 instead of -1,150,414. int64 like every other provider.
            out["Volume"] = out["Volume"].astype("int64")
        return trim_to_inclusive_end(out, end_date, interval)

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
        dataset: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Individual trades, in this library's `price`/`size` contract.

        Indexed by `timestamp`, which is `ts_recv` when the feed carries it
        (`attrs["timestamp_source"]` says which); both vendor stamps are
        kept as columns. `dataset` pins the venue -- pass the same one to
        `get_quotes` for a same-venue pair. Zero-size and sub-penny prints
        are kept and counted in `attrs["print_counts"]`.
        """
        return self._tape(
            "trades", normalize_trades, symbol, start_date, end_date, limit, dataset
        )

    def get_quotes(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        limit: Optional[int] = None,
        dataset: Optional[str] = None,
    ) -> pd.DataFrame:
        """Top-of-book quotes, in this library's `bid_price`/`ask_price`
        contract. `dataset` pins the venue, as on `get_trades`."""
        return self._tape(
            "mbp-1", normalize_quotes, symbol, start_date, end_date, limit, dataset
        )

    def _tape(
        self,
        schema: str,
        normalize: Callable[..., Tuple[pd.DataFrame, List[str]]],
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        limit: Optional[int],
        dataset: Optional[str],
    ) -> pd.DataFrame:
        """
        Trades or quotes, with what the vendor frame said kept on the result.

        THE TWO ARE ROUTED SEPARATELY, and that is the hazard. Each takes
        the first dataset that answers ITS schema, so a venue feed that
        serves trades but not top-of-book quotes answers the tape while the
        next feed answers the quotes, and a spread computed from the pair
        compares one venue's trades with another's quotes. Nothing said so:
        both frames carried a dataset name that no consumer compared. Now a
        frame answered by any dataset but the first the routing asks
        carries a WARNING saying the pair may be cross-venue, and `dataset`
        pins both to one venue. The default routing is unchanged.

        THE VENDOR'S NOTES ARE KEPT. The normalizer reports the flag
        warnings (`F_MAYBE_BAD_BOOK` among them), the sentinel count, the
        price-scale decision and the timestamp it used, and this used to
        discard all of them. They are on `attrs["vendor_notes"]`, and each
        WARNING is logged, the way the depth path does it.
        """
        what = "trades" if schema == "trades" else "quotes"
        if dataset is not None:
            name = str(dataset).strip()
            if not name:
                raise ValidationError(
                    "dataset='' names no dataset. Pass a Databento dataset "
                    f"such as {self._dataset!r} or {self._depth_dataset!r}, "
                    "or leave it unset for the default routing."
                )
            candidates = [name]
        else:
            candidates = self._tick_datasets(symbol)
        frame, served = self._fetch(
            schema, symbol, start_date, end_date, datasets=candidates, what=what
        )
        if limit is not None and len(frame) > limit:
            # Cut BEFORE normalizing, so the counts and warnings the
            # normalizer reports describe the rows that are returned.
            frame = frame.head(int(limit))
        stamp = timestamp_source(frame)
        out, notes = normalize(frame)
        notes = list(notes)
        out = out.set_index("timestamp") if "timestamp" in out.columns else out
        attrs: Dict[str, Any] = {
            "dataset": served,
            "provider": "databento",
            "adjusted": False,
            "schema": schema,
            "timestamp_source": stamp,
        }
        if dataset is None and candidates and served != candidates[0]:
            skipped = candidates[: candidates.index(served)]
            attrs["fallback_from"] = skipped
            other = "quotes" if what == "trades" else "trades"
            notes.append(
                f"WARNING: these {what} come from {served}, not {skipped[0]}, "
                f"the first dataset the tape and quote fetches ask: "
                f"{skipped[0]} did not serve {schema} for this window. The "
                f"{other} for the same window are answered by the first "
                f"dataset that serves them, which may be {skipped[0]} -- and a "
                "spread measured across two venues compares prices that never "
                f"met in one book. Pass dataset={served!r} to both fetches for "
                "a same-venue pair."
            )
        if schema == "trades":
            counts = print_counts(out)
            attrs["print_counts"] = {
                key: value for key, value in counts.items() if key != "n"
            }
        attrs["vendor_notes"] = notes
        for note in notes:
            if note.startswith("WARNING"):
                logger.warning("databento %s %s: %s", what, symbol, note)
        _record(symbol, start_date, end_date, what, served, out)
        # WHICH TAPE ANSWERED, on the frame, the way the bars path does it.
        # The tick datasets are single-venue, so a volume here is that
        # venue's share and not the market's -- and which venue it was is
        # chosen by the request unless the caller pins it.
        return _with_attrs(out, attrs)

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
            datasets=self._depth_datasets(symbol),
            what="depth",
        )
        stamp = timestamp_source(frame)
        out, notes = normalize_book(frame, levels=levels)
        for note in notes:
            if note.startswith("WARNING"):
                logger.warning("databento book %s: %s", symbol, note)
        if limit is not None and len(out) > limit:
            out = out.head(int(limit))
        _record(symbol, start_date, end_date, f"mbp-10:{int(levels)}", _dataset, out)
        return _with_attrs(
            out,
            {
                "dataset": _dataset,
                "provider": "databento",
                "adjusted": False,
                "schema": "mbp-10",
                "timestamp_source": stamp,
                # Kept on the frame, not only logged: the fetch tools read
                # them into their results, where an agent can see them.
                "vendor_notes": list(notes),
            },
        )

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
            datasets=self._depth_datasets(symbol),
            what="order events",
        )
        stamp = timestamp_source(frame)
        out, notes = normalize_mbo(frame)
        for note in notes:
            if note.startswith("WARNING"):
                logger.warning("databento mbo %s: %s", symbol, note)
        if limit is not None and len(out) > limit:
            out = out.head(int(limit))
        _record(symbol, start_date, end_date, "mbo", _dataset, out)
        return _with_attrs(
            out,
            {
                "dataset": _dataset,
                "provider": "databento",
                "adjusted": False,
                "schema": "mbo",
                "timestamp_source": stamp,
                "vendor_notes": list(notes),
            },
        )

    def get_dataset_coverage(
        self, datasets: Optional[Sequence[str]] = None
    ) -> Dict[str, Tuple[str, str]]:
        """
        What each dataset published, from the endpoint that costs nothing.

        The same `_available_range` every fetch already clamps itself
        against, made askable. It is memoized per dataset per provider, so
        a caller that asks about six feeds pays six lookups once and none
        afterwards.

        A dataset that answers nothing -- not entitled, unknown, or a
        range the vendor returns unreadable -- is LEFT OUT of the mapping.
        Returning a window for it would be inventing the one number a
        caller plans around.
        """
        names = (
            [str(d).strip() for d in datasets]
            if datasets is not None
            else self._known_datasets()
        )
        coverage: Dict[str, Tuple[str, str]] = {}
        for name in names:
            if not name or name in coverage:
                continue
            span = self._available_range(name)
            if span is None:
                continue
            first, last = span
            coverage[name] = (first.isoformat(), last.isoformat())
        return coverage

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
        Bytes this exact request would bill, asked before it is made.

        Routed exactly as the matching fetch would route it and clamped to
        the same published window, so the number describes the request the
        caller is about to make rather than a nearby one. Each candidate is
        tried in preference order; a dataset the subscription declines is
        remembered as denied, the way a declined fetch is.

        The vendor's own cost endpoint reports `0.00` on a subscription
        that already includes the feed, which is why this returns BYTES.
        """
        start = _to_utc(start_date, end_of_day=False)
        end = _to_utc(end_date, end_of_day=True)
        if end <= start:
            raise ValidationError(
                f"empty window: start {start_date!r} is not before end "
                f"{end_date!r} (the end date is INCLUSIVE, so a same-day "
                "request is valid and this is not one)."
            )
        route = self.resolve_symbol(symbol)
        candidates = (
            [str(dataset)]
            if dataset
            else self.datasets_for_schema(symbol, schema, start_date)
        )
        # What the fetch would actually send: a future's daily bars are
        # built from hourly bars over a window a day wider (see
        # `_cme_trade_date_bars`), so that is the request priced.
        request_schema = schema
        if route.family == "future" and schema == "ohlcv-1d":
            request_schema = "ohlcv-1h"
            start = start - timedelta(days=1)
        client = self._get_client()
        fmt = "%Y-%m-%d" if request_schema == "ohlcv-1d" else "%Y-%m-%dT%H:%M:%S"
        refused: List[str] = []
        for name in candidates:
            if not name or name in self._denied:
                continue
            window = self._range(name, start, end) or (start, end)
            try:
                size = client.metadata.get_billable_size(
                    dataset=name,
                    schema=request_schema,
                    symbols=[route.raw],
                    stype_in=route.stype_in,
                    start=window[0].strftime(fmt),
                    end=window[1].strftime(fmt),
                )
            except Exception as exc:  # noqa: BLE001 - one refusal, not a trace
                kind = _failure_kind(exc)
                if kind == "auth":
                    raise _rejected_key(
                        exc, f"the billable-size lookup for {name}"
                    ) from exc
                logger.warning(
                    "databento billable size for %s (%s) on %s failed: %s",
                    symbol,
                    schema,
                    name,
                    exc,
                )
                if kind == "denied":
                    with self._lock:
                        self._denied.add(name)
                refused.append(f"{name}: {exc}")
                continue
            return int(size)
        raise APIError(
            f"Databento would not price {schema} for {symbol} between "
            f"{start:%Y-%m-%d} and {end:%Y-%m-%d}. "
            + (
                f"Datasets tried: {refused}."
                if refused
                else "No dataset in this entitlement routes that schema."
            )
        )

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
        try:
            family = self.resolve_symbol(symbol).family
        except ValidationError:
            family = "unknown"
        if family == "future":
            feed = (
                f"{self._futures_dataset} (CME Globex; continuous, parent and "
                "contract symbols)"
            )
            if interval == "1d":
                feed += (
                    ". A daily bar is a CME TRADE DATE -- 17:00 Chicago on the "
                    "prior evening to 16:00 on the date -- aggregated from "
                    "hourly bars, not the vendor's UTC-day bar; Close is the "
                    "date's last trade, not the settlement price, which the "
                    "statistics schema publishes separately"
                )
        elif family == "option":
            feed = f"{self._options_dataset} (OPRA; OSI option symbols)"
        elif interval == "1d":
            feed = (
                f"{DATASET_SUMMARY} from {SUMMARY_START.date()} (the consolidated close "
                f"and volume exactly); before that {DATASET_CONSOLIDATED}, a SAMPLE "
                "feed -- 2-4% of consolidated volume and a UTC-day close that is "
                f"often an after-hours print -- then {self._dataset} and "
                f"{self._depth_dataset}, single-venue tapes"
            )
        else:
            feed = (
                f"{self._dataset} then {self._depth_dataset}: single-venue tapes, "
                "the same order trades and quotes use, so bars and ticks for one "
                "window come from one tape. No consolidated intraday feed exists "
                "in this entitlement; volumes are the venue's share of the market."
            )
        return DataSetMetadata(
            provider="databento",
            adjusted=False,
            # The archive is organized by publication, so an instrument that
            # stopped trading stays queryable over the window it traded in.
            survivorship_free=True,
            point_in_time=False,
            frequency=interval,
            timezone="UTC",
            notes=[
                f"Served by: {feed}. The dataset that answered a fetch is on "
                "the frame as attrs['dataset'].",
                "The index is normalised like every provider's: daily bars are "
                "naive session dates, intraday bars naive UTC instants. Do not "
                "tz-localize before joining.",
                "A futures root that is also an equity ticker (ES, CL, GC, ...) "
                "is refused as ambiguous; spell ES.c.0, ESZ6, ES.FUT or ES~equity.",
            ],
        )

    def get_temporal_contract(self, frame_kind: str = "bars"):
        """
        Bars, with `revisions='unknown'`: Databento reprocesses and corrects
        data (which is why `get_metadata` says `point_in_time=False`), so
        the base contract's claim that bars are never restated would
        contradict the metadata on the same object.
        """
        if frame_kind != "bars":
            return super().get_temporal_contract(frame_kind)
        from standard_quant_tools.data.temporal import price_contract

        contract = price_contract(type(self).__name__)
        return contract.model_copy(
            update={
                "revisions": "unknown",
                "notes": [
                    *contract.notes,
                    "Databento announces reprocessing and corrections; a bar can "
                    "be restated after publication, so revisions are 'unknown' "
                    "rather than 'none'.",
                ],
            }
        )


__all__ = ["BAR_SCHEMAS", "DatabentoProvider"]
