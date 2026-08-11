"""
Concurrent universe fetch for build_dataset.

The builder previously fetched with a serial dict comprehension, so a
50-symbol universe paid 50 sequential network round-trips end to end.
Every provider in this package already exposes `get_ohlcv_async`, and the
rest of the library (portfolio.fetch_ohlcv_panel_async, the signal-panel
backtester) already fetches universes concurrently — modeling was the
outlier, not the async path.

Three things this module is careful about that a bare
`asyncio.gather(*[provider.get_ohlcv_async(s, ...) for s in symbols])`
gets wrong:

1. **Error attribution.** `gather` without `return_exceptions=True`
   propagates the FIRST exception and abandons the rest, so a universe
   with three bad tickers reports one of them, in nondeterministic order,
   with no indication that the others are also bad. Every symbol is
   awaited to completion and all failures are reported together, sorted,
   in one message — the caller fixes their universe in one pass instead
   of one ticker per run.

2. **Bounded concurrency.** The providers implement `get_ohlcv_async` by
   handing the blocking call to the default thread executor, so an
   unbounded gather over a large universe queues every symbol at once and
   leans on the executor's size as an accidental rate limit. A semaphore
   makes the limit explicit and tunable
   (`SQT_MODELING_FETCH_CONCURRENCY`, default 8) — high enough to be a
   real speedup, low enough not to look like abuse to a public endpoint.

3. **Being called from inside a running event loop.** `asyncio.run`
   raises outright if a loop is already running in this thread, which
   would make build_dataset unusable from a notebook, an async agent
   runtime, or any web handler. Detected explicitly and served by a
   sequential fetch on the caller's own thread rather than by failing
   (the alternative — starting a second loop in a worker thread — is
   possible but adds a failure mode for a path whose cost is dominated by
   the provider's own cache anyway).
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

_DEFAULT_CONCURRENCY = 8
_CONCURRENCY_ENV = "SQT_MODELING_FETCH_CONCURRENCY"


def _max_concurrency() -> int:
    """Read the concurrency limit from the environment, falling back to the
    default on anything unusable — a malformed env var must not take down a
    fetch that would otherwise work, the same tolerance
    audit.retention.py applies to SQT_AUDIT_RETENTION_DAYS."""
    raw = os.environ.get(_CONCURRENCY_ENV)
    if raw is None:
        return _DEFAULT_CONCURRENCY
    try:
        value = int(raw)
    except ValueError:
        logger.warning("[modeling] %s=%r is not an integer", _CONCURRENCY_ENV, raw)
        return _DEFAULT_CONCURRENCY
    if value < 1:
        logger.warning("[modeling] %s=%r must be >= 1", _CONCURRENCY_ENV, raw)
        return _DEFAULT_CONCURRENCY
    return value


def _describe_failure(symbol: str, exc: BaseException) -> str:
    return f"{symbol!r}: {type(exc).__name__}: {exc}"


def _validate_frame(symbol: str, df: Any) -> pd.DataFrame:
    """An empty frame is a failure, not an empty dataset: every downstream
    feature would produce all-NaN and the panel would come out empty with
    no indication of which symbol caused it."""
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(
            f"provider returned {type(df).__name__}, not a DataFrame, for {symbol!r}"
        )
    if df.empty:
        raise ValidationError(f"no OHLCV data returned for {symbol!r}")
    return df


async def _fetch_one(
    provider: Any,
    symbol: str,
    start: str,
    end: str,
    interval: str,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame:
    async with semaphore:
        df = await provider.get_ohlcv_async(symbol, start, end, interval)
    return _validate_frame(symbol, df)


async def _fetch_all_async(
    provider: Any,
    symbols: List[str],
    start: str,
    end: str,
    interval: str,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    semaphore = asyncio.Semaphore(_max_concurrency())
    results = await asyncio.gather(
        *(
            _fetch_one(provider, symbol, start, end, interval, semaphore)
            for symbol in symbols
        ),
        # Every symbol runs to completion so ALL failures can be reported
        # at once. Without this the first exception wins and the remaining
        # tasks are abandoned mid-flight.
        return_exceptions=True,
    )
    frames: Dict[str, pd.DataFrame] = {}
    failures: List[str] = []
    for symbol, outcome in zip(symbols, results):
        if isinstance(outcome, BaseException):
            failures.append(_describe_failure(symbol, outcome))
        else:
            frames[symbol] = outcome
    return frames, failures


def _fetch_all_sequential(
    provider: Any,
    symbols: List[str],
    start: str,
    end: str,
    interval: str,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Fallback for when a loop is already running in this thread. Same
    all-failures-reported contract as the async path, so the error a caller
    sees does not depend on which of the two ran."""
    frames: Dict[str, pd.DataFrame] = {}
    failures: List[str] = []
    for symbol in symbols:
        try:
            frames[symbol] = _validate_frame(
                symbol, provider.get_ohlcv(symbol, start, end, interval)
            )
        except Exception as exc:  # noqa: BLE001 — reported, not swallowed
            failures.append(_describe_failure(symbol, exc))
    return frames, failures


def _in_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def fetch_universe_ohlcv(
    provider: Any,
    symbols: List[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for every symbol, concurrently where possible.

    Returns {symbol: DataFrame} for the symbols requested, in the order
    given.

    Raises:
        ValidationError: naming EVERY symbol that failed and why, not just
        the first one encountered.
    """
    if not symbols:
        return {}

    # DataProvider declares get_ohlcv_async abstract, so every provider in
    # this package has one — but build_dataset accepts whatever
    # DataFactory hands back, and a duck-typed or partially-stubbed
    # provider that implements only the sync call should degrade to a
    # slower fetch rather than an AttributeError from inside a coroutine.
    supports_async = callable(getattr(provider, "get_ohlcv_async", None))

    if _in_running_loop() or not supports_async:
        logger.debug(
            "[modeling] fetching %d symbol(s) sequentially "
            "(running_loop=%s, async_support=%s)",
            len(symbols),
            _in_running_loop(),
            supports_async,
        )
        frames, failures = _fetch_all_sequential(
            provider, symbols, start, end, interval
        )
    else:
        frames, failures = asyncio.run(
            _fetch_all_async(provider, symbols, start, end, interval)
        )

    if failures:
        raise ValidationError(
            f"build_model_dataset: {len(failures)} of {len(symbols)} symbol(s) failed to "
            f"fetch ({start} to {end}, interval={interval!r}) — "
            + "; ".join(sorted(failures))
        )

    # Preserve the caller's order: dict comprehension over `symbols` rather
    # than returning `frames` directly, whose order follows completion for
    # the async path.
    return {symbol: frames[symbol] for symbol in symbols}
