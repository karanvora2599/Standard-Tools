"""
Shared two-tier OHLCV cache for data providers: an in-process TTL session
cache plus a persistent Parquet disk cache. Originally built for
YFinanceProvider only; extracted here so BloombergProvider and
PolygonProvider can reuse the exact same hardened logic (path-traversal
defenses, atomic writes, TTL eviction) instead of having no caching at all
— every provider re-fetching from scratch on every call is a real
practical-deployment cost (redundant network/Terminal load, avoidable
latency, and for a rate-limited API like Polygon's free tier, avoidably
burning through the request budget).

Every cache key (session and disk) includes an explicit `provider` name so
providers can never collide on the same entry for the "same" symbol/date/
interval, even though different providers can have different adjustment
conventions or data revisions for it.
"""

import logging
import os
import re
import threading
import uuid
from datetime import date as _date
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
from cachetools import TTLCache

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

# Permissive enough for realistic ticker formats (BRK.B, BRK/B, 0700.HK,
# ^GSPC, EURUSD=X) while rejecting ".." (parent-directory traversal),
# backslashes, drive-letter colons, and null bytes — the same slug-plus-
# resolved-containment approach artifacts.py uses for run_id/name.
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9./\-^=]+$")
_DATE_STR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Cache-path date bound: either a plain date (daily and coarser) or a
# date-plus-time-of-day (intraday). Both forms are filesystem-safe by
# construction -- digits, hyphens and a single 'T' -- so neither can carry
# a path separator or '..' past _parquet_path's containment check.
_BOUND_STR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{6})?$")
# Bounded, generic allow-list for the interval token — deliberately NOT
# tied to any one provider's specific interval vocabulary (yfinance's
# "1m".."3mo", Bloomberg's DAILY/WEEKLY/MONTHLY, Polygon's "1m".."3mo" are
# all different sets) since each provider's own get_ohlcv already validates
# interval against its own supported set before ever reaching this cache
# layer. This just needs to reject path-traversal/injection attempts.
_INTERVAL_RE = re.compile(r"^[A-Za-z0-9]{1,10}$")

# Cache-format generation, embedded in every cache filename.
#
# Bumped when the MEANING of a cached frame changes, so files written under
# the old meaning are simply never looked up again (they age out with the
# directory) instead of being served as though they matched the new one.
#
# v2: end_date became an inclusive observation cutoff (see
# inclusive_end_timestamp). Every v1 file was written under yfinance's
# exclusive-`end` behavior and is therefore missing its final bar -- serving
# one on a cache hit would answer the same request differently than a live
# fetch, which is precisely the cache/live parity failure this layer exists
# to avoid.
# v3: intraday timestamps are canonicalized to UTC before tz-stripping (see
# _normalize_ohlcv_index). Every v2 intraday file holds LOCAL wall-clock
# times, so serving one on a cache hit would answer the same request with a
# different instant than a live fetch — the cache/live parity failure this
# layer exists to prevent. Old files are never looked up again rather than
# migrated; they age out with the directory.
_CACHE_FORMAT_VERSION = "v3"

# ── In-process session cache (avoids repeated network calls in the same run) ──
_session_cache = TTLCache(maxsize=100, ttl=3600)
# cachetools' cache classes do no internal locking of their own (that's why
# its own @cached/cachedmethod decorators accept an explicit lock= param) --
# _session_cache is read/written directly from multiple threads (every
# provider's get_ohlcv_async dispatches to asyncio's default
# ThreadPoolExecutor via run_in_executor), so every access must go through
# this lock via _session_cache_get/_session_cache_set below, never touching
# _session_cache directly. Kept as a plain Lock (not RLock): no call site
# re-enters the cache from inside an already-held lock.
_session_cache_lock = threading.Lock()


def _session_cache_get(key):
    with _session_cache_lock:
        return _session_cache.get(key)


def _session_cache_set(key, value) -> None:
    with _session_cache_lock:
        _session_cache[key] = value


# ── Persistent Parquet disk cache ─────────────────────────────────────────────
# Historical OHLCV bars are stored permanently on disk once a date range is in
# the past. Note "historical" here means "not still forming today" — it does
# NOT mean the values are guaranteed never to change again: adjusted prices
# can be retroactively revised by a later corporate action (split, special
# dividend) for dates that were already cached. This cache trades that small
# staleness risk for avoiding repeated network calls; callers who need
# post-corporate-action-accurate history for a symbol that's had a recent
# action should clear/bypass the cache (SQT_CACHE_DIR) rather than assume it
# self-heals. The cache directory can be overridden with SQT_CACHE_DIR.
_CACHE_ROOT = Path(
    os.environ.get(
        "SQT_CACHE_DIR",
        str(Path.home() / ".cache" / "standard_quant_tools" / "ohlcv"),
    )
)


def _norm_date(d: Union[str, datetime, _date]) -> str:
    """
    Normalise any date-like value to a YYYY-MM-DD string.

    Validates the result actually looks like a date rather than blindly
    truncating: a real datetime/date object always stringifies to a valid
    YYYY-MM-DD prefix, but an arbitrary caller-supplied string (start_date/
    end_date are LLM-reachable via every provider's get_ohlcv) does not,
    and this value feeds directly into the Parquet cache filename
    (_parquet_path) — a truncated-but-unvalidated string could still
    contain '..' or path separators after slicing to 10 characters.
    """
    norm = str(d)[:10]
    if not _DATE_STR_RE.match(norm):
        raise ValidationError(
            f"date must be in YYYY-MM-DD format, got {d!r} (normalized: {norm!r})"
        )
    return norm


# Sub-daily bar intervals, across every provider's own vocabulary
# (yfinance "1m".."90m"/"1h"; Polygon "1m"/"5m"/"1h"; etc.). Anchored so
# the daily-and-coarser tokens that merely START with a digit and 'm'
# ("1mo", "3mo") do NOT match — misclassifying a monthly bar as intraday
# would skip the date normalization every downstream consumer depends on.
_INTRADAY_INTERVAL_RE = re.compile(r"^\d+\s*(m|min|minute|h|hour)s?$", re.IGNORECASE)


def is_intraday_interval(interval: str) -> bool:
    """True for sub-daily bar intervals ("1m", "15m", "1h"), False for
    daily and coarser ("1d", "5d", "1wk", "1mo", "3mo")."""
    return bool(_INTRADAY_INTERVAL_RE.match(str(interval).strip()))


def inclusive_end_timestamp(
    end_date: Union[str, datetime, _date], interval: str = "1d"
) -> "pd.Timestamp":
    """
    The last observation timestamp an `end_date` is defined to include.

    `DataProvider.get_ohlcv`'s `end_date` is an INCLUSIVE observation
    cutoff (see data/base.py). Providers disagreed natively -- yfinance's
    `ticker.history(end=...)` is exclusive, while Polygon's aggregates `to`
    and Bloomberg's `endDate` are inclusive -- so the same call returned a
    different window depending only on which provider served it, and
    score_model(as_of=X) silently excluded X on the default provider while
    still reporting X as the as-of date.

    A bare date means "through the end of that day" at every interval; an
    explicit intraday timestamp means exactly that instant.
    """
    ts = pd.Timestamp(end_date)
    if pd.isna(ts):
        raise ValidationError(
            f"end_date must be parseable as a timestamp, got {end_date!r}"
        )
    if (ts.hour, ts.minute, ts.second, ts.microsecond) == (0, 0, 0, 0):
        return ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def trim_to_inclusive_end(
    df: pd.DataFrame, end_date: Union[str, datetime, _date], interval: str = "1d"
) -> pd.DataFrame:
    """
    Enforce the inclusive-end contract on a provider's returned frame.

    Applied to EVERY provider rather than trusting each vendor's documented
    boundary: the contract then holds by construction, so a vendor changing
    (or mis-documenting) its own semantics can't silently move the window.
    Cheap -- one boolean mask on an already-materialized frame.
    """
    if df.empty:
        return df
    bound = inclusive_end_timestamp(end_date, interval)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        bound = bound.tz_localize(idx.tz) if bound.tzinfo is None else bound
    return df[idx <= bound]


def _norm_cache_bound(d: Union[str, datetime, _date], interval: str = "1d") -> str:
    """
    Normalise a start/end bound into a cache-path token.

    Daily and coarser collapse to YYYY-MM-DD (unchanged — existing cache
    files keep their names and stay valid). Intraday keeps time-of-day as
    YYYY-MM-DDTHHMMSS, because _norm_date's blanket 10-character truncation
    made two genuinely different intraday requests on the same day —
    09:30→12:00 and 13:00→16:00 — resolve to the same cache file, so the
    second silently served the first's bars.

    A bare date under an intraday interval is left as a date rather than
    padded to midnight: it means "the whole day", which is a different
    request from "the day starting at 00:00:00", and conflating them would
    reintroduce the same collision from the other direction.
    """
    if not is_intraday_interval(interval):
        return _norm_date(d)
    if isinstance(d, str) and _DATE_STR_RE.match(d):
        return d
    ts = pd.Timestamp(d)
    if pd.isna(ts):
        raise ValidationError(f"date must be parseable as a timestamp, got {d!r}")
    if ts.tz is not None:
        ts = ts.tz_convert(None) if ts.tzinfo is not None else ts
    if (ts.hour, ts.minute, ts.second) == (0, 0, 0):
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m-%dT%H%M%S")


def _normalize_ohlcv_index(df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    """
    Strip tz-awareness from an OHLCV DataFrame's index, and — for daily and
    coarser bars only — drop the time component, at a single choke point
    every provider's disk-cache-read path (and yfinance's live-fetch path,
    which attaches tz-aware timestamps even for daily bars) goes through.
    Every downstream consumer builds or compares against tz-naive
    timestamps, and mixing tz-aware/tz-naive indices either raises or (via
    .reindex(), which doesn't raise) silently produces an all-NaN result.

    `interval` is REQUIRED to be accurate for intraday data. This function
    used to call .normalize() unconditionally, which set every timestamp to
    midnight — so a 4-bar hourly series collapsed to four copies of the same
    date and lost its time-series identity entirely. That ran on yfinance's
    live fetch AND on both providers' Parquet cache reads, which also made
    the same request answer differently depending on whether it was served
    live or from cache (Polygon's live parser preserves intraday timestamps;
    the cache read did not).

    Defaults to "1d" so any caller not passing an interval keeps the exact
    previous behavior rather than silently gaining time-of-day it isn't
    prepared for; daily output is unchanged bit-for-bit.

    INTRADAY TIMESTAMPS ARE CONVERTED TO UTC before the timezone is dropped.
    Stripping tz-awareness without converting first keeps the LOCAL wall
    clock, which silently makes bars from different exchanges look
    simultaneous:

        London  15:00 BST  (14:00 UTC) -> naive 15:00
        New York 15:00 EDT (19:00 UTC) -> naive 15:00

    Those two bars are five hours apart, and after normalization their
    indexes are equal — so a join, a correlation, a PCA or a cross-sectional
    panel silently pairs a London afternoon with a New York afternoon as one
    instant. Nothing raises; the numbers are simply about a market state
    that never existed. UTC is the canonical instant, so it is what survives
    the strip.

    DAILY AND COARSER DELIBERATELY DO NOT CONVERT. A daily bar is identified
    by its LOCAL TRADING DATE, and converting first would shift it: Tokyo
    2024-06-03 00:00 JST is 2024-06-02 15:00 UTC, which normalizes to the
    WRONG DAY. The two cases genuinely differ — an intraday bar is an
    instant, a daily bar is a session — so they are handled differently on
    purpose rather than by oversight.
    """
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        if is_intraday_interval(interval):
            idx = idx.tz_convert("UTC")
        idx = idx.tz_localize(None)
    if not is_intraday_interval(interval):
        # The `interval` default is "1d" for back-compat, which means a caller
        # who simply FORGETS to pass it gets the old collapsing behaviour on
        # intraday data -- the exact bug this function was rewritten to fix,
        # reachable again by omission rather than by intent. Every call site in
        # this package passes it; this warning is here so a future one that
        # does not fails loudly in the log instead of silently flattening a
        # time series to a single date.
        if len(idx) and (idx != idx.normalize()).any():
            logger.warning(
                "[_normalize_ohlcv_index] interval=%r is daily-or-coarser but "
                "the index carries a time component, so %d timestamp(s) are "
                "about to be flattened to midnight. If this is intraday data, "
                "pass the real interval — otherwise the series loses its "
                "time-series identity and several bars collapse onto one date.",
                interval,
                int((idx != idx.normalize()).sum()),
            )
        idx = idx.normalize()
    df = df.copy()
    df.index = idx
    return df


def _parquet_path(
    symbol: str, start: str, end: str, interval: str, provider: str = "yfinance"
) -> Path:
    """
    Build the Parquet cache path for (provider, symbol, start, end, interval).

    `provider` is prefixed into the filename (not a subdirectory — keeps
    the layout flat, which callers/tests rely on) so different providers
    can never collide on the same cache entry for the "same" symbol/date/
    interval — they can have different adjustment conventions or data
    revisions. Defaults to "yfinance" so existing callers/tests that
    predate multi-provider caching don't need to change.

    All inputs are LLM-reachable via some provider's get_ohlcv parameters
    and go straight into the filename — validates each against an allow-
    list/pattern (the same slug-plus-resolved-containment approach
    artifacts.py uses for run_id/name) and confirms the resulting path
    actually resolves inside _CACHE_ROOT before returning it, as defense
    in depth.

    Raises:
        ValidationError: symbol/interval/provider don't match their
            allowed pattern/set, or the resolved path would escape
            _CACHE_ROOT.
    """
    if not symbol or ".." in symbol or not _SYMBOL_RE.match(symbol):
        raise ValidationError(
            f"symbol={symbol!r} is not a valid identifier for caching — only "
            "letters, digits, '.', '/', '-', '^', '=' are allowed, and '..' "
            "is never allowed."
        )
    for value, name in ((start, "start"), (end, "end")):
        if not _BOUND_STR_RE.match(value):
            raise ValidationError(
                f"{name}={value!r} must already be normalized to YYYY-MM-DD "
                "(or YYYY-MM-DDTHHMMSS for an intraday interval) before "
                "building a cache path (call _norm_cache_bound first)."
            )
    if not _INTERVAL_RE.match(interval):
        raise ValidationError(
            f"interval={interval!r} is not a valid cache-path token — only "
            "alphanumeric strings up to 10 characters are allowed."
        )
    if not _SYMBOL_RE.match(provider) or ".." in provider:
        raise ValidationError(f"provider={provider!r} is not a valid identifier.")

    # "/" is not usable in a filename, but a plain replace with "-" made
    # "BRK/B" and "BRK-B" — two genuinely different symbols in real ticker
    # vocabularies — collide on one cache entry, so one symbol could be
    # served the other's bars. Use a token that _SYMBOL_RE itself rejects, so
    # no real symbol can ever produce it by other means.
    safe = symbol.replace("/", "__SLASH__").upper()
    path = (
        _CACHE_ROOT
        / f"{_CACHE_FORMAT_VERSION}_{provider}_{safe}_{start}_{end}_{interval}.parquet"
    )
    root = _CACHE_ROOT.resolve()
    resolved = path.resolve()
    # On Windows, Path.resolve() calls into GetFinalPathNameByHandle for a
    # path that actually exists on disk, which returns the "\\?\"-prefixed
    # extended-length form — but for a path that doesn't exist yet (or is
    # short enough), it's returned without that prefix. _CACHE_ROOT and the
    # full file path can therefore disagree on the prefix even though they
    # denote the same location, causing a false-positive "escapes cache
    # dir" rejection. Compare with the prefix stripped from both sides;
    # still return the real `resolved` path (the prefix is harmless to the
    # filesystem APIs that consume it).
    root_cmp = Path(str(root).removeprefix("\\\\?\\"))
    resolved_cmp = Path(str(resolved).removeprefix("\\\\?\\"))
    if not resolved_cmp.is_relative_to(root_cmp):
        raise ValidationError(
            f"resolved cache path {resolved} escapes SQT_CACHE_DIR ({root})"
        )
    return resolved


def _safe_parquet_path(
    symbol: str, start: str, end: str, interval: str, provider: str = "yfinance"
) -> "Path | None":
    """
    Same as _parquet_path, but returns None instead of raising when the
    inputs can't be safely encoded into a cache path (e.g. a symbol
    containing characters _SYMBOL_RE rejects). Caching is an optimization,
    not a correctness requirement — a symbol a provider's own live-fetch
    path can still handle safely (already escaped via urllib.parse.quote
    or similar at that layer) should not have the entire call fail just
    because it can't ALSO be cached; it should just skip caching for that
    call and fall through to a live fetch, same as a disk-cache read
    failure already does.
    """
    try:
        return _parquet_path(symbol, start, end, interval, provider=provider)
    except ValidationError:
        logger.debug(
            "[cache] %r is not a valid cache-path symbol for provider=%r — "
            "skipping disk cache for this call",
            symbol,
            provider,
        )
        return None


def _is_historical(end_date: Union[str, datetime, _date]) -> bool:
    """Return True when end_date is strictly before today (bar is fully formed,
    so it's eligible for the disk cache — see the cache-root comment above for
    why "historical" doesn't mean the adjusted values can never change)."""
    try:
        return _norm_date(end_date) < _date.today().isoformat()
    except Exception:
        return False


def _write_parquet_atomic(path: Path, df: pd.DataFrame) -> None:
    """
    Write `df` to `path` atomically: write to a per-PID-and-thread temp
    file then atomically replace the target, so concurrent processes AND
    concurrent threads within the same process don't collide on the same
    temp filename (os.getpid() alone isn't unique across threads).
    Failures are logged and swallowed — a failed cache write should never
    fail the caller's actual data fetch, which already succeeded.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(
            f"{path.stem}.{os.getpid()}.{threading.get_ident()}."
            f"{uuid.uuid4().hex[:8]}.tmp.parquet"
        )
        df.to_parquet(tmp)
        tmp.replace(path)  # atomic on all platforms
        logger.debug("[cache] disk write → %s", path.name)
    except Exception as cache_exc:
        logger.warning("[cache] disk write failed for %s: %s", path.name, cache_exc)
