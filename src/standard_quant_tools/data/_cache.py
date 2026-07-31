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
# Bounded, generic allow-list for the interval token — deliberately NOT
# tied to any one provider's specific interval vocabulary (yfinance's
# "1m".."3mo", Bloomberg's DAILY/WEEKLY/MONTHLY, Polygon's "1m".."3mo" are
# all different sets) since each provider's own get_ohlcv already validates
# interval against its own supported set before ever reaching this cache
# layer. This just needs to reject path-traversal/injection attempts.
_INTERVAL_RE = re.compile(r"^[A-Za-z0-9]{1,10}$")

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


def _normalize_ohlcv_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip any tz-awareness and drop any intraday time component from an
    OHLCV DataFrame's index, once, at a single choke point every provider's
    disk-cache-read path (and yfinance's live-fetch path, which attaches
    tz-aware timestamps even for daily bars) goes through — every
    downstream consumer builds or compares against tz-naive, midnight-
    normalized timestamps, and mixing tz-aware/tz-naive indices either
    raises or (via .reindex(), which doesn't raise) silently produces an
    all-NaN result.
    """
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
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
        if not _DATE_STR_RE.match(value):
            raise ValidationError(
                f"{name}={value!r} must already be normalized to YYYY-MM-DD "
                "before building a cache path (call _norm_date first)."
            )
    if not _INTERVAL_RE.match(interval):
        raise ValidationError(
            f"interval={interval!r} is not a valid cache-path token — only "
            "alphanumeric strings up to 10 characters are allowed."
        )
    if not _SYMBOL_RE.match(provider) or ".." in provider:
        raise ValidationError(f"provider={provider!r} is not a valid identifier.")

    safe = symbol.replace("/", "-").upper()
    path = _CACHE_ROOT / f"{provider}_{safe}_{start}_{end}_{interval}.parquet"
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
