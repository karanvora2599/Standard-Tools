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

ONE READ AND ONE WRITE FOR EVERY PROVIDER. `_read_cached_ohlcv` and
`_write_cached_ohlcv` are the only way a provider touches the disk tier.
The three providers each carried their own copy of the read, and the copies
drifted: none of them checked what the live path checks, so a readable
Parquet file holding the wrong thing was served as a hit, and each treated
a Windows sharing violation -- a reader opening a file another process is
replacing -- as corruption and deleted a valid entry.
"""

import contextlib
import io
import logging
import re
import threading
import time
from datetime import date as _date
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from cachetools import TTLCache

from standard_quant_tools._containment import require_within
from standard_quant_tools._env import env_path
from standard_quant_tools.artifact_store import write_bytes_atomically
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

# Per-provider generation bumps, ON TOP of the shared version above.
#
# A change to what ONE provider's cached frame means should retire that
# provider's files and nobody else's: throwing away the Polygon cache
# because Databento's keys changed would spend a rate-limited budget
# refetching bars that were right. The value is an offset rather than an
# absolute name so a later shared bump still moves every provider -- with
# an absolute "v4" here, a shared bump to v4 would leave these files
# current by coincidence. A provider without an entry is at the shared
# version, so every existing yfinance and Polygon file keeps its name.
#
# databento +1: a single-letter exchange suffix used to be folded into the
# ticker, so a file named for `GOOG.L` holds Alphabet class A's bars, and
# files are now named by the symbol as sent to the vendor rather than as
# the caller spelled it. See the CHANGELOG entry of 2026-09-27.
# databento +2: an intraday window with an explicit end was cached one bar
# short (the vendor's range is half-open and the bar AT the end was never
# asked for), a future's daily file holds UTC-day bars where a daily bar is
# now a CME trade date, and one-second bars were read as daily -- their
# window bounds collapsed to dates and their index to midnights -- until
# "1s" was counted as intraday. Each keeps its name, so only a bump stops it
# being served. See the CHANGELOG entry of 2026-09-28.
_PROVIDER_GENERATION_BUMPS = {"databento": 2}


def _provider_family(provider: str) -> str:
    """The provider a cache token belongs to: `databento-EQUS.SUMMARY` is
    Databento's, whatever dataset answered."""
    return str(provider).split("-", 1)[0].lower()


def cache_generation(provider: str) -> str:
    """The generation prefix this provider's cache files are written and
    looked up under: the shared version plus the provider's own bumps."""
    base = int(_CACHE_FORMAT_VERSION.lstrip("v"))
    bump = _PROVIDER_GENERATION_BUMPS.get(_provider_family(provider), 0)
    return f"v{base + bump}"


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


#: How long a window whose last bar is still forming may be served from
#: the session cache. The cache's own TTL is an hour, which served an
#: unsettled bar as final for up to an hour (findings, the plumbing); a
#: minute keeps three identical requests in one run to one metered fetch
#: without pretending a live bar is history.
_UNSETTLED_TTL_SECONDS = 60.0


def _session_cache_get(key):
    with _session_cache_lock:
        entry = _session_cache.get(key)
    if entry is None:
        return None
    value, expires_at = entry
    if expires_at is not None and time.monotonic() >= expires_at:
        with _session_cache_lock:
            _session_cache.pop(key, None)
        return None
    return value


def _session_cache_set(key, value, *, end=None, complete: bool = True) -> None:
    """Store a fetched window. `end` is the window's inclusive end bound;
    a window that is not yet historical is kept only for
    _UNSETTLED_TTL_SECONDS, because its last bar is still moving. So is an
    answer the provider knows is short of its window (`complete=False`):
    the vendor had not published all of it, and will."""
    expires_at = None
    if not complete or (end is not None and not _is_historical(end)):
        expires_at = time.monotonic() + _UNSETTLED_TTL_SECONDS
    with _session_cache_lock:
        _session_cache[key] = (value, expires_at)


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


def _default_cache_root() -> Path:
    return Path.home() / ".cache" / "standard_quant_tools" / "ohlcv"


def cache_root() -> Path:
    """
    The directory the disk tier reads and writes, resolved once, at first
    use, and fixed for the rest of the process.

    RESOLVED AT FIRST USE, NOT AT IMPORT. The root used to be read when this
    module was imported, before a local `.env` had been loaded, so a
    `SQT_CACHE_DIR` set there was never honoured; and it was read with a
    bare `os.environ.get`, so `SQT_CACHE_DIR=` (empty) meant the working
    directory. It now goes through `env_path`: blank is the default under
    the home directory, a relative path is refused by name rather than
    anchored to whatever directory the process happens to be in, and the
    refusal reaches the caller instead of being mistaken for a symbol the
    cache cannot encode.

    FIXED ONCE RESOLVED, so a later change to the environment does not move
    the cache under a running process. A test that relocates the cache
    assigns `_CACHE_ROOT`; that assignment is honoured.
    """
    root = globals().get("_CACHE_ROOT")
    if root is None:
        root = env_path("SQT_CACHE_DIR", _default_cache_root)
        globals()["_CACHE_ROOT"] = root
    return Path(root)


def __getattr__(name: str):
    # `_CACHE_ROOT` is read directly by the tools that describe the cache
    # and by the external-data fence. Until the first use resolves it, the
    # name is not bound, and reading it resolves it here -- so every reader
    # sees the root the cache itself uses, never a placeholder.
    if name == "_CACHE_ROOT":
        return cache_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # The shape is not enough: '2019-13-45' matches it, reached the vendor,
    # failed there, and the retry layer asked twice more for a date that
    # does not exist. Refused here, by name, before any request.
    try:
        datetime.strptime(norm, "%Y-%m-%d")
    except ValueError as exc:
        raise ValidationError(
            f"date {d!r} is not a calendar date ({exc}). Dates are YYYY-MM-DD "
            "with a real month and day."
        ) from exc
    return norm


# Sub-daily bar intervals, across every provider's own vocabulary
# (yfinance "1m".."90m"/"1h"; Polygon "1m"/"5m"/"1h"; Databento "1s"; etc.).
# Anchored so the daily-and-coarser tokens that merely START with a digit
# and 'm' ("1mo", "3mo") do NOT match — misclassifying a monthly bar as
# intraday would skip the date normalization every downstream consumer
# depends on. Seconds are intraday too: without them Databento's one-second
# bars were read as daily and every bar of a day was flattened onto its
# midnight.
_INTRADAY_INTERVAL_RE = re.compile(
    r"^\d+\s*(s|sec|second|m|min|minute|h|hour)s?$", re.IGNORECASE
)


def is_intraday_interval(interval: str) -> bool:
    """True for sub-daily bar intervals ("1s", "1m", "15m", "1h"), False for
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
    # Pinned to nanoseconds: Parquet reads an index back at the resolution
    # it stored (`[s]`, `[ms]`), and `hash_dataframe` saw a different
    # frame for byte-identical data, so a replay reported `data_changed`.
    if hasattr(idx, "as_unit"):
        idx = idx.as_unit("ns")
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
    actually resolves inside the cache root before returning it, as defense
    in depth.

    Raises:
        ValidationError: symbol/interval/provider don't match their
            allowed pattern/set, the resolved path would escape the cache
            root, or SQT_CACHE_DIR is set to something it cannot be.
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
    generation = cache_generation(provider)
    base = cache_root()
    path = base / f"{generation}_{provider}_{safe}_{start}_{end}_{interval}.parquet"
    root = base.resolve()
    resolved = path.resolve()
    # The extended-length prefix Windows puts on a resolved path that
    # exists is handled inside require_within, once, for every root this
    # library writes under.
    return require_within(
        resolved,
        root,
        f"resolved cache path {resolved} escapes SQT_CACHE_DIR ({root})",
    )


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

    A misconfigured SQT_CACHE_DIR is NOT such a symbol: it is resolved
    before the path is built, so its refusal reaches the caller by name
    instead of quietly turning the disk cache off for every call.
    """
    cache_root()
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


def _utc_today() -> _date:
    """Today's UTC date: the one clock the disk tier's "historical" guard and
    a provider's settled-prefix rule both read. A seam, so a test can stand
    in a day without moving the system clock."""
    return datetime.now(timezone.utc).date()


def _is_historical(end_date: Union[str, datetime, _date]) -> bool:
    """Return True when end_date is strictly before today (bar is fully formed,
    so it's eligible for the disk cache — see the cache-root comment above for
    why "historical" doesn't mean the adjusted values can never change).

    TODAY IS THE UTC DATE. The comparison used the local date, so east of
    UTC+5:30 a session still trading was already 'yesterday' and its
    mid-session bar was written to the disk cache permanently (findings,
    the plumbing). Every provider's bars are on the UTC clock after
    normalisation, and so is this guard."""
    try:
        return _norm_date(end_date) < _utc_today().isoformat()
    except Exception:
        return False


_GENERATION_RE = re.compile(r"^v\d+_")
#: A generation prefix and the provider token after it. Provider tokens
#: cannot contain '_' (`_SYMBOL_RE` refuses it), so the first underscore
#: after the generation ends the token.
_GENERATION_PROVIDER_RE = re.compile(r"^(v\d+)_([^_]+)_")


def _is_dead_generation(name: str) -> bool:
    """Whether a cache filename carries a generation its provider no longer
    reads. A name with a generation but no provider token is compared with
    the shared version, as every file was before generations were per
    provider."""
    if not _GENERATION_RE.match(name):
        return False
    match = _GENERATION_PROVIDER_RE.match(name)
    if match is None:
        return not name.startswith(f"{_CACHE_FORMAT_VERSION}_")
    return match.group(1) != cache_generation(match.group(2))


def dead_generations(*, dry_run: bool = True) -> "list[Path]":
    """
    Cache files written under an earlier format version, which are never
    read again: the reader looks up the current version's name only.
    Returned sorted; deleted when `dry_run` is False. Nothing else in the
    directory is touched -- a file that does not carry a generation prefix
    is not this cache's to remove. The live cache held 1,574 files, 501 of
    them a dead generation (findings, the plumbing).

    CURRENT IS PER PROVIDER. A provider whose own generation was bumped
    (see `_PROVIDER_GENERATION_BUMPS`) has its previous files listed here,
    while another provider's files at the shared version stay current.

    With `dry_run=False` the files actually removed are returned: one that
    another process holds open on Windows cannot be deleted, and is left
    for the next collection rather than stopping this one.
    """
    root = cache_root()
    if not root.exists():
        return []
    dead = sorted(p for p in root.glob("*.parquet") if _is_dead_generation(p.name))
    if dry_run:
        return dead
    return [p for p in dead if _remove(p)]


def _remove(path: Path) -> bool:
    """Delete one file; False when the platform refuses (held open, gone)."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("[cache] could not remove %s: %s", path.name, exc)
        return False
    return True


# ── Orphaned temp files ───────────────────────────────────────────────────────
#
# A write puts its bytes under a temp name and renames it over the entry. A
# write that dies between the two -- the process killed, the disk full, a
# rename Windows refused -- can leave the temp behind. The current writer
# removes its temp in a `finally`, so only a killed process leaves one now;
# the writer before it (see the CHANGELOG entry of 2026-09-28) left one on
# every failed rename, and named it with the current generation prefix, so
# the dead-generation collection never saw it.

#: The temp names this cache's writers have used: the current one
#: (`artifact_store.write_bytes_atomically`: `.<entry>.<32 hex>.tmp`) and
#: the one before it (`<entry stem>.<pid>.<thread>.<8 hex>.tmp.parquet`).
_ORPHAN_NAME_RES = (
    re.compile(r"^\.v\d+_[^\\/]+\.parquet\.[0-9a-f]{32}\.tmp$"),
    re.compile(r"^v\d+_[^\\/]+\.\d+\.\d+\.[0-9a-f]{8}\.tmp\.parquet$"),
)

#: A temp file younger than this is left alone: a writer may still own it.
#: A cache write holds its temp for the milliseconds between writing the
#: bytes and renaming them, so an hour is far past any live write.
ORPHAN_MIN_AGE_SECONDS = 3600.0


def _is_orphan_name(name: str) -> bool:
    return any(pattern.match(name) for pattern in _ORPHAN_NAME_RES)


def orphaned_temps(
    *,
    dry_run: bool = True,
    min_age_seconds: float = ORPHAN_MIN_AGE_SECONDS,
) -> "list[Path]":
    """
    Temp files a cache write left behind, older than `min_age_seconds`.
    Returned sorted; deleted when `dry_run` is False, in which case the
    files actually removed are returned.

    NEVER A FILE A LIVE WRITER MAY STILL OWN. Only names a cache writer
    produces are candidates, and only once their last modification is
    older than the threshold, so a write in progress in another thread or
    process is not collected underneath it. A temp that carries a dead
    generation is listed by `dead_generations` and not again here.
    """
    if min_age_seconds < 0:
        raise ValidationError(
            f"min_age_seconds must be >= 0, got {min_age_seconds!r}: a "
            "negative age would collect a temp file a writer still owns."
        )
    root = cache_root()
    if not root.exists():
        return []
    cutoff = time.time() - float(min_age_seconds)
    found: List[Path] = []
    for path in root.iterdir():
        name = path.name
        if not _is_orphan_name(name) or _is_dead_generation(name):
            continue
        try:
            if not path.is_file() or path.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        found.append(path)
    found.sort()
    if dry_run:
        return found
    return [p for p in found if _remove(p)]


# ── The one write ─────────────────────────────────────────────────────────────


def _write_parquet_atomic(path: Path, df: pd.DataFrame) -> bool:
    """
    Write `df` to `path` so no reader ever sees a partial file, through the
    library's one atomic writer (`artifact_store.write_bytes_atomically`):
    the bytes go to a uniquely named temp file beside the entry and are
    renamed over it, and the temp is removed in a `finally`, so a failed
    rename -- or a KeyboardInterrupt in the middle -- leaves nothing behind.
    This used to be a second implementation that removed its temp only on
    success, and it left one behind on every rename Windows refused.

    A rename Windows refuses because a reader has the entry open is retried
    by that writer, briefly. Failures are logged and swallowed: a failed
    cache write should never fail the caller's data fetch, which already
    succeeded. An interrupt still propagates, after the cleanup. Returns
    whether the entry was written.
    """
    try:
        buffer = io.BytesIO()
        df.to_parquet(buffer)
        write_bytes_atomically(path, buffer.getvalue())
    except Exception as exc:  # noqa: BLE001 - caching is an optimisation
        logger.warning("[cache] disk write failed for %s: %s", path.name, exc)
        return False
    logger.debug("[cache] disk write → %s", path.name)
    return True


# ── The one read, and what an entry must look like to be served ───────────────

#: The columns every provider's live path guarantees, in the library's order.
REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

#: The attrs key a writer sets on an entry it KNOWS is settled: the vendor
#: served the whole window, published through its last day, so the entry is
#: the complete answer and may stand as the stored first part of a longer
#: window. Its value is the window `[start, end]` the entry answers. An
#: entry without it is served as before for its own window and never used
#: as a part. The reader removes the key, so no served frame carries it.
SETTLED_ATTR = "sqt_settled_window"

#: A read Windows refuses with a sharing violation -- another process is
#: renaming a new version over the entry, or has it open -- is tried this
#: many times, with a short, growing pause between attempts, before the
#: call is treated as a miss. Both causes last milliseconds.
_SHARING_ATTEMPTS = 3
_SHARING_BACKOFF_SECONDS = 0.02

_DAYS_PER_UNIT = {"d": 1, "day": 1, "wk": 7, "w": 7, "week": 7, "mo": 31, "month": 31}
_PERIOD_RE = re.compile(r"^(\d+)\s*(d|day|wk|w|week|mo|month)s?$", re.IGNORECASE)


def _bound_timestamp(token: str) -> pd.Timestamp:
    """A cache-path bound token back to the instant it names."""
    if "T" in token:
        return pd.Timestamp(datetime.strptime(token, "%Y-%m-%dT%H%M%S"))
    return pd.Timestamp(token)


def _window(
    interval: str, start_str: str, end_str: str
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    The earliest and latest bar label a correct answer to this request can
    carry.

    The end is the inclusive end every provider trims to before it writes.
    The start is widened by one bar period for daily and coarser bars,
    because a weekly or monthly bar is labelled by the start of its period,
    which can precede the requested start. An intraday request is widened a
    day either way: its bounds are naive and in the caller's zone, while
    the bars are UTC instants, and the two can differ by up to a day.
    """
    lower = _bound_timestamp(start_str)
    upper = inclusive_end_timestamp(_bound_timestamp(end_str), interval)
    if is_intraday_interval(interval):
        return lower - pd.Timedelta(days=1), upper + pd.Timedelta(days=1)
    match = _PERIOD_RE.match(str(interval).strip())
    days = int(match.group(1)) * _DAYS_PER_UNIT[match.group(2).lower()] if match else 92
    return lower - pd.Timedelta(days=days + 3), upper


def _cached_frame_problem(
    frame: pd.DataFrame, interval: str, start_str: str, end_str: str
) -> Optional[str]:
    """
    Why a frame is not a plausible answer to this request, or None.

    The checks the live paths make before they return a frame: the five
    OHLCV columns, numeric; at least one bar; no null Close. And one only a
    cache needs: every bar inside the requested window. A readable Parquet
    file holding a single `Close` column, or another window's bars, was
    served as a hit because the read path checked none of this.

    NO NULL CLOSE, BECAUSE NO LIVE PATH SERVES ONE. Every provider drops a
    bar with no Close before it returns or writes a frame, and records the
    dates it dropped in the frame's attrs, which Parquet keeps -- so what is
    cached is what was served, disclosures included. A file that still
    holds a null Close was not written that way, and is evicted rather than
    served with a hole the live path would have reported.
    """
    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in frame.columns]
    if missing:
        return f"it has no {missing} column(s)"
    non_numeric = [
        c for c in REQUIRED_OHLCV_COLUMNS if not pd.api.types.is_numeric_dtype(frame[c])
    ]
    if non_numeric:
        return f"its {non_numeric} column(s) are not numeric"
    if frame.empty:
        return "it holds no bars"
    if frame["Close"].isna().any():
        return "it has a null Close"
    try:
        index = pd.DatetimeIndex(frame.index)
        if index.tz is not None:
            return "its index carries a timezone, which no normalised frame does"
        if index.hasnans:
            return "it has a missing timestamp"
        lower, upper = _window(interval, start_str, end_str)
        first, last = index.min(), index.max()
    except Exception as exc:  # noqa: BLE001 - an index that cannot be judged
        return f"its index cannot be checked against the window ({exc})"
    if first < lower or last > upper:
        return (
            f"its bars run from {first} to {last}, outside the requested "
            f"window {start_str} to {end_str}"
        )
    return None


def _evict(path: Path, reason: str) -> None:
    """Remove an entry that is not a plausible answer. A removal the
    platform refuses is not an error: the entry is refetched either way,
    and the next write replaces it."""
    logger.warning("[cache] evicting %s and refetching: %s", path.name, reason)
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _read_cached_ohlcv(
    path: Optional[Path],
    interval: str,
    start_str: str,
    end_str: str,
    *,
    settled_only: bool = False,
) -> Optional[pd.DataFrame]:
    """
    The entry at `path` as a normalised OHLCV frame, or None when there is
    nothing to serve.

    With `settled_only`, an entry that does not carry the settled marker
    for exactly this window (`SETTLED_ATTR`) is a miss -- not evicted: it
    is a valid answer for its own window, written before the marker existed
    or by a fetch the vendor had not finished publishing. The marker is
    removed from the frame either way.

    THE SAME CHECKS THE LIVE PATH MAKES. A file that reads but is not a
    plausible answer to this request (`_cached_frame_problem`) is evicted,
    and the caller fetches live, exactly as for a file that does not read.

    A SHARING VIOLATION IS NOT CORRUPTION. On Windows, opening an entry
    another process is renaming a new version over, or has open, raises
    `PermissionError`. Every provider used to treat that as a corrupt file
    and delete a valid entry -- and the unguarded delete could itself raise,
    which the retry layer turned into an APIError with no network call
    made. The read is retried briefly; if it still cannot open, the call is
    a miss and the entry is kept. Content that does not parse (pyarrow's
    errors, a ValueError or an OSError that is not a permission error) is
    evicted as before.

    Duplicate or out-of-order bar labels are logged, not evicted: no writer
    here produces them, so they came from the vendor, and a refetch would
    bring them back.
    """
    if path is None:
        return None
    try:
        if not path.exists():
            return None
    except OSError:
        return None
    raw = None
    refusal: Optional[BaseException] = None
    for attempt in range(1, _SHARING_ATTEMPTS + 1):
        try:
            raw = pd.read_parquet(path)
            break
        except FileNotFoundError:
            return None
        except PermissionError as exc:
            refusal = exc
            if attempt < _SHARING_ATTEMPTS:
                time.sleep(_SHARING_BACKOFF_SECONDS * attempt)
        except Exception as exc:  # noqa: BLE001 - unparseable content
            _evict(path, f"it does not read as Parquet ({type(exc).__name__}: {exc})")
            return None
    if raw is None:
        logger.warning(
            "[cache] %s could not be opened (%s); fetching live and keeping "
            "the entry, which another process is writing or reading",
            path.name,
            refusal,
        )
        return None
    try:
        frame = _normalize_ohlcv_index(raw, interval)
    except Exception as exc:  # noqa: BLE001 - an index that is not time
        _evict(path, f"its index is not a timestamp index ({exc})")
        return None
    marker = frame.attrs.pop(SETTLED_ATTR, None)
    if settled_only and _settled_marker(marker) != (start_str, end_str):
        return None
    problem = _cached_frame_problem(frame, interval, start_str, end_str)
    if problem is not None:
        _evict(path, problem)
        return None
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        logger.warning(
            "[cache] %s has duplicate or out-of-order bar labels; served as "
            "stored, since a refetch would bring them back",
            path.name,
        )
    return frame


def _settled_marker(value: object) -> Optional[Tuple[str, str]]:
    """The window a settled marker names, or None for anything else."""
    try:
        start, end = value  # type: ignore[misc]
    except (TypeError, ValueError):
        return None
    return (start, end) if isinstance(start, str) and isinstance(end, str) else None


def _write_cached_ohlcv(
    path: Optional[Path],
    df: pd.DataFrame,
    interval: str,
    start_str: str,
    end_str: str,
    *,
    settled: bool = False,
) -> bool:
    """
    Persist a live answer, unless the read would refuse it. Returns whether
    the entry was written.

    A frame `_read_cached_ohlcv` would evict is not written: storing it
    would only cost a write now and an eviction and a refetch on every later
    call, and the answer is served live either way.

    With `settled`, the entry is marked as the complete answer for its
    window (`SETTLED_ATTR`). Only a caller that knows the vendor served the
    whole window, published through its last day, passes it. The frame
    passed in is not changed.
    """
    if path is None:
        return False
    problem = _cached_frame_problem(df, interval, start_str, end_str)
    if problem is not None:
        logger.warning("[cache] not caching %s: %s", path.name, problem)
        return False
    if settled:
        marked = df.copy(deep=False)
        marked.attrs = {**df.attrs, SETTLED_ATTR: [start_str, end_str]}
        df = marked
    return _write_parquet_atomic(path, df)
