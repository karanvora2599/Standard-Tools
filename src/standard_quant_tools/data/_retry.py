"""Shared retry decorator for data-provider network calls."""

import functools
import logging
import time

from standard_quant_tools.error import (
    APIError,
    DataNotFoundError,
    InvalidSymbolError,
    NonRetryableAPIError,
)

logger = logging.getLogger(__name__)


def retry(times: int = 3, delay: float = 1, backoff: float = 2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t_delay = delay
            last_exc = None
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except (InvalidSymbolError, DataNotFoundError, NonRetryableAPIError):
                    # Definitive errors — never retry or re-wrap. Must be
                    # caught before the broader `except (..., APIError)`
                    # below, since NonRetryableAPIError IS an APIError and
                    # Python matches except clauses in order — a permanent
                    # failure (e.g. an invalid API key) would otherwise be
                    # retried identically to a transient one (429/5xx).
                    raise
                except (ValueError, APIError) as e:
                    last_exc = e
                    if i == times - 1:
                        raise
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
                except Exception as e:
                    raise APIError(f"Unexpected error in {func.__name__}: {e}") from e
            if last_exc:
                raise last_exc

        return wrapper

    return decorator
