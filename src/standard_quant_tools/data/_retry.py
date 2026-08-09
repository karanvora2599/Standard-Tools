"""Shared retry decorator for data-provider network calls."""

import functools
import logging
import time

from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    InvalidSymbolError,
    NonRetryableAPIError,
    QuantError,
)

logger = logging.getLogger(__name__)

# Definitive failures — the same call will fail identically no matter how many
# times it is repeated, so retrying only wastes wall-clock time and (for a
# rate-limited provider) request budget. Listed before the retryable clause
# below because NonRetryableAPIError IS an APIError and Python matches except
# clauses in order.
_NEVER_RETRY = (InvalidSymbolError, DataNotFoundError, NonRetryableAPIError)


def retry(times: int = 3, delay: float = 1, backoff: float = 2):
    """
    Retry a provider call on transient failures.

    Retries: APIError (transient 429/5xx), ValueError, and any non-QuantError
    exception — the latter is what a raw network stack actually raises
    (ConnectionError, TimeoutError, socket.gaierror, aiohttp/requests client
    errors). Those are the single most common transient failure mode, so they
    must be retried rather than wrapped and re-raised on the first attempt.

    Never retried, and re-raised with their own type intact:
      - InvalidSymbolError / DataNotFoundError / NonRetryableAPIError —
        definitive provider answers.
      - Every other QuantError (notably ValidationError) — a caller error in
        the arguments, not a transient condition. Re-wrapping these as
        APIError would also destroy the type callers catch on.

    Args:
        times: Total attempts (not retries-after-the-first). Must be >= 1.
        delay: Seconds to wait before the second attempt.
        backoff: Multiplier applied to `delay` after each failed attempt.

    Raises:
        ValueError: times < 1 — a zero/negative count would otherwise skip the
            wrapped call entirely and silently return None.
    """
    if times < 1:
        raise ValueError(f"retry(times=...) must be >= 1, got {times}")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t_delay = delay
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Classified inside a single handler rather than across
                    # several `except` clauses: the hierarchy overlaps
                    # (NonRetryableAPIError < APIError < DataProviderError <
                    # QuantError), so clause ordering alone silently decided
                    # the wrong outcome for the middle case.
                    if isinstance(e, _NEVER_RETRY):
                        raise
                    if isinstance(e, QuantError) and not isinstance(e, APIError):
                        # A non-transient library error (e.g. ValidationError
                        # on a bad date range). Propagate unchanged — retrying
                        # cannot help, and re-wrapping would hide the real
                        # type callers catch on.
                        raise
                    if i == times - 1:
                        # Preserve the original type for APIError and its
                        # subclasses (call sites catch on those); wrap a raw
                        # network/stdlib exception so callers still only have
                        # to handle this package's own error hierarchy.
                        if isinstance(e, APIError):
                            raise
                        raise APIError(
                            f"{func.__name__} failed after {times} attempt(s): {e}"
                        ) from e
                    logger.warning(
                        "[retry] %s attempt %d/%d failed: %s — retrying in %.1fs",
                        func.__name__,
                        i + 1,
                        times,
                        e,
                        t_delay,
                    )
                    time.sleep(t_delay)
                    t_delay *= backoff
            # Unreachable: the loop either returns or raises on the final
            # attempt. Kept so a future edit to the loop bounds can't silently
            # reintroduce the "returns None without calling anything" bug.
            raise AssertionError("retry loop exited without returning or raising")

        return wrapper

    return decorator
