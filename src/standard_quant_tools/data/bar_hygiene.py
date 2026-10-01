"""
Bars a provider cannot serve as they came, and a last bar that is not final
yet: what is dropped, what is flagged, and how both reach a tool's warnings.

A BAR WITH NO CLOSE IS DROPPED, NOT REFUSED. Vendors list a row for a session
before it has traded -- yfinance appends one outside market hours for the
next session -- and a single such row used to refuse the whole series, as an
error the retry layer then repeated three times on rows that could not
change. Every provider now drops a row whose Close is missing and says so on
the frame and in the log:

    attrs["dropped_placeholder_bars"]  rows after the last bar with a Close
    attrs["dropped_missing_bars"]      rows before it: holes in the series

Dropping a row is the answer the native indicator recursions already give a
gap -- they step over a NaN bar -- without handing a NaN to the tools that
have no notion of one (returns, risk, Hurst, tail risk). Those see a shorter
series, and a return across a dropped day spans both days. Only a window
with no Close at all is refused, as `NonRetryableAPIError`: asking again
returns the same rows.

A LAST BAR THAT IS STILL FORMING IS KEPT AND FLAGGED. During a session the
last daily bar is the session so far -- its Close is the latest price and its
volume a fraction of a day's -- and every daily tool used to read it as a
complete session. When its session has not closed, the frame carries
`attrs["partial_last_bar"] = True`, the session's date and the instant it
closes. Whether it has closed is read from the exchange calendar when the
optional `exchange_calendars` package is installed (an early close is then an
early close), and otherwise from the venue's regular close: 16:00
America/New_York for US equities, the CME trade date's 16:00 America/Chicago
for futures. Only daily and coarser bars are judged; an intraday bar is a
complete interval of its own.

The flag is decided when a frame is SERVED, not when it is fetched, so a
frame held in a cache carries the answer for the moment it is handed out.

ONE COLLECTOR FOR EVERY TOOL. A tool reads what was dropped or flagged from
the frames it was served (`collect_served_bars`), so a panel assembled from
several frames -- whose attrs pandas drops on a concat or a column pick --
still reports what each of its symbols disclosed.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import re
from dataclasses import dataclass
from datetime import datetime, time, timezone
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.error import NonRetryableAPIError

from ._cache import is_intraday_interval

logger = logging.getLogger(__name__)

#: Rows after the last bar with a Close, dropped: sessions a vendor listed
#: before they traded.
PLACEHOLDER_KEY = "dropped_placeholder_bars"
#: Rows with no Close before the last bar with one, dropped as missing bars.
MISSING_KEY = "dropped_missing_bars"
#: True when the last bar's session had not closed when the frame was served.
PARTIAL_KEY = "partial_last_bar"
#: The partial bar's label, as an ISO date.
PARTIAL_SESSION_KEY = "partial_last_bar_session"
#: The instant (UTC, ISO 8601) the partial bar's session closes.
PARTIAL_CLOSE_KEY = "partial_last_bar_closes_at"

_PARTIAL_KEYS = (PARTIAL_KEY, PARTIAL_SESSION_KEY, PARTIAL_CLOSE_KEY)
#: Every attrs key this module writes.
DISCLOSURE_KEYS = (PLACEHOLDER_KEY, MISSING_KEY) + _PARTIAL_KEYS

#: How many dates a warning names before it summarises the rest.
_DATES_NAMED = 5


def _utc_now() -> pd.Timestamp:
    """The current instant, tz-aware UTC. A seam, so a test can stand in a
    session that is still trading."""
    return pd.Timestamp(datetime.now(timezone.utc))


def _label(value: Any) -> str:
    """A bar label as text: the date for a midnight label, the full
    timestamp otherwise, so two intraday bars on one day stay distinct."""
    ts = pd.Timestamp(value)
    if ts == ts.normalize():
        return ts.date().isoformat()
    return ts.isoformat()


def _listed(dates: Sequence[str]) -> str:
    shown = ", ".join(dates[:_DATES_NAMED])
    more = len(dates) - _DATES_NAMED
    return shown + (f" and {more} more" if more > 0 else "")


# ── Bars with no Close ────────────────────────────────────────────────────────


def drop_unusable_closes(
    frame: pd.DataFrame, symbol: str, *, provider: str
) -> pd.DataFrame:
    """
    `frame` without its rows whose Close is missing, each disclosed.

    A row after the last bar that has a Close is a PLACEHOLDER -- a session
    listed before it traded -- and one before it is a MISSING bar. Both are
    dropped, named in `attrs` (`dropped_placeholder_bars`,
    `dropped_missing_bars`) and logged as a warning. A frame with no such
    row is returned untouched, attrs included.

    Raises:
        NonRetryableAPIError: no row has a Close. The vendor's answer for
            this window has nothing in it to compute with, and a re-fetch
            returns the same rows.
    """
    if frame is None or len(frame) == 0 or "Close" not in frame.columns:
        return frame
    null = frame["Close"].isna().to_numpy()
    if not null.any():
        return frame
    labels = pd.DatetimeIndex(frame.index)
    if null.all():
        raise NonRetryableAPIError(
            f"{provider} returned {len(frame)} bar(s) for {symbol} from "
            f"{_label(labels.min())} to {_label(labels.max())}, and none of them "
            "has a Close: every row is a session listed before it traded, or a "
            "bar the vendor has no price for. This is the vendor's answer for "
            "the window, so asking again returns the same rows. Widen the "
            "window to include a session that has traded, check that the "
            "symbol still trades, or ask another provider (source=...)."
        )
    last_priced = labels[~null].max()
    placeholder = null & np.asarray(labels > last_priced)
    missing = null & ~placeholder
    kept = frame.loc[~null].copy()
    # Set explicitly: `.copy()` keeps attrs in recent pandas only.
    kept.attrs = dict(frame.attrs)
    if placeholder.any():
        dates = [_label(x) for x in labels[placeholder]]
        kept.attrs[PLACEHOLDER_KEY] = dates
        logger.warning(
            "[%s] %s: dropped %d trailing bar(s) with no Close (%s) -- a session "
            "listed before it traded; the series ends at %s",
            provider,
            symbol,
            len(dates),
            _listed(dates),
            _label(last_priced),
        )
    if missing.any():
        dates = [_label(x) for x in labels[missing]]
        kept.attrs[MISSING_KEY] = dates
        logger.warning(
            "[%s] %s: dropped %d bar(s) with no Close inside the window as "
            "missing bars (%s)",
            provider,
            symbol,
            len(dates),
            _listed(dates),
        )
    return kept


# ── When a session closes ─────────────────────────────────────────────────────


@lru_cache(maxsize=None)
def _exchange_calendar(code: str) -> Any:
    """The `exchange_calendars` calendar for `code`, or None when the
    package is absent or does not know the code. Built once per process:
    building one takes a fifth of a second."""
    try:
        import exchange_calendars as xcals
    except ImportError:
        return None
    try:
        return xcals.get_calendar(code)
    except Exception:  # noqa: BLE001 - an unknown code is the fixed close
        return None


@dataclass(frozen=True)
class SessionClock:
    """
    When a bar labelled with a date stops changing.

    `calendar` is an `exchange_calendars` code, consulted first when the
    package is installed and the date is one of its sessions, so a holiday's
    early close is honoured. Otherwise the session closes at `close` on the
    date in `timezone` -- or, with `day_end`, at the end of the date, for a
    market whose daily bar is a whole day.
    """

    name: str
    timezone: str
    close: time = time(0, 0)
    calendar: Optional[str] = None
    day_end: bool = False
    weekends: bool = False

    def session_close(self, day: pd.Timestamp) -> pd.Timestamp:
        """The instant (tz-aware UTC) the session labelled `day` closes."""
        day = pd.Timestamp(day).normalize()
        if self.calendar:
            cal = _exchange_calendar(self.calendar)
            if cal is not None:
                try:
                    if cal.is_session(day):
                        return pd.Timestamp(cal.session_close(day)).tz_convert("UTC")
                except Exception:  # noqa: BLE001 - outside the calendar's range
                    pass
        base = day + pd.Timedelta(days=1) if self.day_end else day
        wall = pd.Timestamp(datetime.combine(base.date(), self.close))
        local = wall.tz_localize(
            self.timezone, ambiguous=False, nonexistent="shift_forward"
        )
        return local.tz_convert("UTC")

    def last_session(self, first: pd.Timestamp, last: pd.Timestamp) -> pd.Timestamp:
        """The last session from `first` through `last`: the day a weekly
        or monthly bar labelled `first` stops changing."""
        if self.calendar:
            cal = _exchange_calendar(self.calendar)
            if cal is not None:
                try:
                    sessions = cal.sessions_in_range(first, last)
                    if len(sessions):
                        return pd.Timestamp(sessions[-1]).tz_localize(None)
                except Exception:  # noqa: BLE001 - outside the calendar's range
                    pass
        day = pd.Timestamp(last).normalize()
        if not self.weekends:
            while day.weekday() >= 5 and day > first:
                day -= pd.Timedelta(days=1)
        return day


#: US equities and ETFs: the NYSE calendar, else 16:00 New York.
US_EQUITY = SessionClock(
    name="the US equity close",
    timezone="America/New_York",
    close=time(16, 0),
    calendar="XNYS",
)
#: A futures daily bar is a CME trade date, which runs from 17:00 Chicago on
#: the prior evening to 16:00 on the date -- the convention the Databento
#: provider aggregates its futures daily bars by.
CME_TRADE_DATE = SessionClock(
    name="the CME trade date's close",
    timezone="America/Chicago",
    close=time(16, 0),
)
#: A daily bar that is a whole UTC day: crypto, and a vendor's UTC-day bar.
#: Weekends count, so a weekly bar is not called closed on a Friday while
#: its Saturday and Sunday are still trading.
UTC_DAY = SessionClock(
    name="the end of the UTC day", timezone="UTC", day_end=True, weekends=True
)


def local_day(timezone_name: str, calendar: Optional[str] = None) -> SessionClock:
    """A listing judged by its exchange's calendar when one is known and
    installed, and otherwise as forming until its local midnight -- later
    than any exchange's close, so a closed session can be flagged for a few
    hours but a forming one is never called closed."""
    return SessionClock(
        name=f"the end of the {timezone_name} day",
        timezone=timezone_name,
        calendar=calendar,
        day_end=True,
    )


_PERIOD_RE = re.compile(r"^(\d+)\s*(d|day|wk|w|week|mo|month)s?$", re.IGNORECASE)


def _period_last_day(
    label: pd.Timestamp, interval: str, anchor: str
) -> Optional[pd.Timestamp]:
    """The last calendar day a bar labelled `label` covers, or None for an
    interval this cannot place. A bar labelled by the end of its period
    (`anchor="end"`, Bloomberg's) covers through its label."""
    if anchor == "end":
        return label
    match = _PERIOD_RE.match(str(interval).strip())
    if match is None:
        return None
    count = max(int(match.group(1)), 1)
    unit = match.group(2).lower()
    if unit in ("d", "day"):
        return label + pd.offsets.BDay(count - 1) if count > 1 else label
    if unit in ("wk", "w", "week"):
        return label + pd.Timedelta(days=7 * count - 1)
    return label + pd.DateOffset(months=count) - pd.Timedelta(days=1)


def flag_partial_last_bar(
    frame: pd.DataFrame,
    symbol: str,
    interval: str,
    clock: Optional[SessionClock],
    *,
    anchor: str = "start",
    now: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Flag, in place, a last bar whose session has not closed.

    Any earlier flag is removed first: the answer belongs to the moment the
    frame is served, and a frame held in a cache may have been flagged when
    its session was still trading. A closed series is left as it was.
    Intraday intervals are never flagged.
    """
    attrs = frame.attrs
    for key in _PARTIAL_KEYS:
        attrs.pop(key, None)
    if clock is None or len(frame) == 0 or is_intraday_interval(interval):
        return frame
    try:
        label = pd.Timestamp(pd.DatetimeIndex(frame.index).max()).normalize()
    except (TypeError, ValueError):
        return frame
    if pd.isna(label):
        return frame
    if label.tzinfo is not None:
        label = label.tz_localize(None)
    last_day = _period_last_day(label, interval, anchor)
    if last_day is None:
        return frame
    now = _utc_now() if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    # Every venue's session has closed by the end of the day after its date,
    # so an older bar is settled without asking a calendar.
    if now.tz_convert("UTC").tz_localize(None) >= last_day + pd.Timedelta(days=2):
        return frame
    session = last_day if last_day == label else clock.last_session(label, last_day)
    closes_at = clock.session_close(session)
    if now < closes_at:
        attrs[PARTIAL_KEY] = True
        attrs[PARTIAL_SESSION_KEY] = label.date().isoformat()
        attrs[PARTIAL_CLOSE_KEY] = closes_at.isoformat()
        logger.warning(
            "%s: the last %s bar (%s) is a session that has not closed -- it "
            "closes at %s (%s); its Close is the latest price and its volume "
            "a partial session's",
            symbol,
            interval,
            label.date().isoformat(),
            closes_at.isoformat(),
            clock.name,
        )
    return frame


# ── What a frame discloses, in words ──────────────────────────────────────────


def disclosed_attrs(frame: Any) -> Dict[str, Any]:
    """The disclosure keys this module wrote on `frame`, and nothing else."""
    attrs = getattr(frame, "attrs", None)
    if not isinstance(attrs, Mapping):
        return {}
    return {k: attrs[k] for k in DISCLOSURE_KEYS if attrs.get(k)}


def describe_disclosures(symbol: str, attrs: Mapping[str, Any]) -> List[str]:
    """One warning per condition a frame's attrs disclose, naming `symbol`."""
    notes: List[str] = []
    placeholder = list(attrs.get(PLACEHOLDER_KEY) or [])
    if placeholder:
        notes.append(
            f"{symbol}: {len(placeholder)} bar(s) at the end of the window had "
            f"no Close and were dropped ({_listed(placeholder)}). A vendor lists "
            "a session before it trades -- outside market hours the next "
            "session is a row with no price -- so the series ends at the last "
            "bar that has one."
            + (
                " More than one such bar usually means the symbol has not "
                "traded since, not merely that the next session is pending."
                if len(placeholder) > 1
                else ""
            )
        )
    missing = list(attrs.get(MISSING_KEY) or [])
    if missing:
        notes.append(
            f"{symbol}: {len(missing)} bar(s) inside the window had no Close and "
            f"were dropped as missing bars ({_listed(missing)}). A return across "
            "a dropped bar spans both sessions as one observation; indicators "
            "step over the bar the way they step over any gap."
        )
    if attrs.get(PARTIAL_KEY):
        session = attrs.get(PARTIAL_SESSION_KEY) or "the last bar"
        closes = attrs.get(PARTIAL_CLOSE_KEY)
        notes.append(
            f"{symbol}: the last bar ({session}) is a session that has not "
            "closed"
            + (f" -- it closes at {closes}" if closes else "")
            + ". Its Close is the latest price, not the session's close, and "
            "its volume covers part of the session, yet every figure computed "
            "from it treats it as a complete session. Re-run after the close "
            "for settled numbers, or end the window on the previous session."
        )
    return notes


def bar_warnings(frame: Any, symbol: str) -> List[str]:
    """The warnings one served frame carries, for a caller that holds it."""
    return describe_disclosures(symbol, disclosed_attrs(frame))


# ── Collecting what was served ────────────────────────────────────────────────


class ServedBars:
    """What the bar frames served inside one `collect_served_bars` block
    disclosed, by symbol."""

    def __init__(self) -> None:
        self._entries: List[Tuple[str, Dict[str, Any]]] = []

    def _add(self, symbol: str, disclosed: Dict[str, Any]) -> None:
        self._entries.append((symbol, disclosed))

    def warnings(self) -> List[str]:
        """Every disclosure as a warning, each stated once, in the order the
        frames were served."""
        seen: Dict[str, None] = {}
        for symbol, disclosed in list(self._entries):
            for note in describe_disclosures(symbol, disclosed):
                seen.setdefault(note, None)
        return list(seen)

    def __len__(self) -> int:
        return len(self._entries)


_COLLECTOR: contextvars.ContextVar[Optional[ServedBars]] = contextvars.ContextVar(
    "standard_quant_tools_served_bars", default=None
)


@contextlib.contextmanager
def collect_served_bars() -> Iterator[ServedBars]:
    """
    Collect the disclosures of every bar frame a provider serves inside the
    block, including frames fetched on worker threads by an async panel
    fetch -- every provider copies its context into the thread it runs on.

    A block opened inside another hands what it collected to the outer one
    when it closes, so a tool calling a helper that collects for itself
    still sees its fetches.
    """
    parent = _COLLECTOR.get()
    served = ServedBars()
    token = _COLLECTOR.set(served)
    try:
        yield served
    finally:
        _COLLECTOR.reset(token)
        if parent is not None:
            parent._entries.extend(served._entries)


def disclose_served(
    frame: pd.DataFrame,
    symbol: str,
    interval: str,
    clock: Optional[SessionClock],
    *,
    anchor: str = "start",
) -> pd.DataFrame:
    """
    The last step of every provider's `get_ohlcv`: decide whether the last
    bar is still forming, and record what the frame discloses for the
    tool that asked. Returns `frame`, flagged in place.
    """
    flag_partial_last_bar(frame, symbol, interval, clock, anchor=anchor)
    collector = _COLLECTOR.get()
    if collector is not None:
        disclosed = disclosed_attrs(frame)
        if disclosed:
            collector._add(str(symbol), disclosed)
    return frame


__all__ = [
    "CME_TRADE_DATE",
    "DISCLOSURE_KEYS",
    "MISSING_KEY",
    "PARTIAL_CLOSE_KEY",
    "PARTIAL_KEY",
    "PARTIAL_SESSION_KEY",
    "PLACEHOLDER_KEY",
    "SessionClock",
    "ServedBars",
    "US_EQUITY",
    "UTC_DAY",
    "bar_warnings",
    "collect_served_bars",
    "describe_disclosures",
    "disclose_served",
    "disclosed_attrs",
    "drop_unusable_closes",
    "flag_partial_last_bar",
    "local_day",
]
