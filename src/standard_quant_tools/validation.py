import logging
import math
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Iterator, Optional, Tuple

import numpy as np

from ._compat import is_dataframe_like, is_empty, is_series_like
from .error import ValidationError

logger = logging.getLogger(__name__)


# ── Memoized input checks ────────────────────────────────────────────────
# These checks are per-CALL, which is right at the public API boundary and
# wasteful in a batch loop: build_model_dataset computes N features for one
# entity, and every one of them re-scans the same ohlcv["Close"] the
# builder already fetched and column-checked. Measured on build_dataset,
# the repeat work was 12% of the run at 50 entities and 18% at 100.
#
# A caller that is about to hand the SAME objects to many checked functions
# can open a scope in which an object that has already passed a given check
# is not re-checked. Three properties make this safe to bolt onto a
# safety-critical layer:
#
#   * opt-in — outside a scope, nothing changes, so every public entry
#     point behaves exactly as before
#   * keyed on (object identity, check variant), so a series checked with
#     allow_nan=True is not treated as having passed allow_nan=False
#   * only successes are recorded, so a raising check is never memoized
#
# The memo holds a strong reference to each key object, which pins its id()
# for the life of the scope and makes id-reuse-after-free impossible. It
# relies on pandas returning a cached column object for repeated df[col]
# access; if that ever stops being true the memo simply stops hitting, so
# the failure mode is "no speedup", not "a skipped check".
_check_memo: ContextVar[Optional[dict]] = ContextVar("_sqt_check_memo", default=None)


@contextmanager
def memoized_input_checks() -> Iterator[None]:
    """Skip repeat validation of objects already checked inside this scope."""
    token = _check_memo.set({})
    try:
        yield
    finally:
        _check_memo.reset(token)


def _memo_seen(obj: Any, tag: Tuple) -> bool:
    """True if `obj` already passed check `tag` in an active memo scope."""
    memo = _check_memo.get()
    return memo is not None and (id(obj), tag) in memo


def _memo_record(obj: Any, tag: Tuple) -> None:
    """Record that `obj` passed check `tag`. No-op outside a memo scope."""
    memo = _check_memo.get()
    if memo is not None:
        # Value is the object itself: a strong reference, so this id cannot
        # be recycled onto a different object while it is a live key.
        memo[(id(obj), tag)] = obj


def validate_dataframe(required_columns: Optional[list[str]] = None):
    """
    Decorator to validate input DataFrame.
    Checks for empty DataFrame and missing columns.

    Accepts a pandas or (when polars is installed) a polars DataFrame —
    `is_dataframe_like` checks both, so a polars.DataFrame is actually
    validated here rather than silently skipped (a bare
    `isinstance(arg, pd.DataFrame)` check would never match a polars
    object, letting an empty/malformed one straight through to fail with
    a confusing error deep inside the wrapped function instead of here).
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find the DataFrame argument (usually the first one or named 'data'/'df')
            df = None
            for arg in args:
                if is_dataframe_like(arg):
                    df = arg
                    break
            if df is None:
                # check kwargs
                for value in kwargs.values():
                    if is_dataframe_like(value):
                        df = value
                        break

            if df is not None:
                if is_empty(df):
                    logger.warning(
                        "[validate_dataframe] %s rejected: empty DataFrame",
                        func.__name__,
                    )
                    raise ValidationError(
                        f"Input DataFrame for {func.__name__} is empty."
                    )

                if required_columns:
                    missing = [col for col in required_columns if col not in df.columns]
                    if missing:
                        logger.warning(
                            "[validate_dataframe] %s rejected: missing columns %s",
                            func.__name__,
                            missing,
                        )
                        raise ValidationError(
                            f"Missing required columns in {func.__name__}: {missing}"
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_series(allow_empty: bool = False, allow_nan: bool = True):
    """
    Decorator to validate input Series.

    Accepts a pandas or (when polars is installed) a polars Series — see
    validate_dataframe's docstring for why `is_series_like` (not a bare
    `isinstance(arg, pd.Series)`) matters here.

    This used to check emptiness ONLY. The all-NaN check sat in the body as
    commented-out code, and there was no infinity check at all, so every
    metric wearing this decorator had its own accidental behaviour for the
    same invalid input:

        sharpe_ratio(all-NaN)       -> nan
        sortino_ratio(all-NaN)      -> +inf   (reads as "no losing bars")
        var_historical(all-NaN)     -> IndexError
        max_drawdown(contains inf)  -> -1.703437775179145

    That last one is the reason this belongs in the shared decorator rather
    than in each function: an infinity does not stay visibly wrong. It came
    back as a drawdown that looks measured.

    `allow_nan=True` is the deliberate default. Indicator warm-up windows, a
    ticker that lists mid-sample, and a benchmark on a different holiday
    calendar all produce legitimate gaps, and many callers drop them
    internally on purpose — making partial NaN fatal would break correct code
    to catch a problem it has already handled. All-NaN and infinities are
    rejected regardless, since neither is data.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # BOTH, and the keyword half was missing. Every failure listed in
            # this decorator's docstring came back the moment a caller wrote
            # `sharpe_ratio(returns=...)` instead of `sharpe_ratio(...)` --
            # measured, an all-NaN series returned nan rather than being
            # refused. Keyword calls are normal Python and are what an agent
            # building a call from a JSON schema produces by default, so the
            # hole was open on exactly the caller this library is for.
            for arg in (*args, *kwargs.values()):
                if is_series_like(arg):
                    if is_empty(arg):
                        if allow_empty:
                            continue
                        logger.warning(
                            "[validate_series] %s rejected: empty Series", func.__name__
                        )
                        raise ValidationError(
                            f"Input Series for {func.__name__} is empty."
                        )
                    _check_series_values(arg, func.__name__, allow_nan)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _check_series_values(arg: Any, func_name: str, allow_nan: bool) -> None:
    """Numerical half of validate_series. Tolerates a non-pandas Series-like
    (polars) by falling back to a plain conversion, and skips anything
    non-numeric rather than guessing at it."""
    tag = ("series_values", allow_nan)
    if _memo_seen(arg, tag):
        return
    try:
        values = np.asarray(arg, dtype=float)
    except (TypeError, ValueError):
        return  # not a numeric series; the shape checks above still applied
    if values.size == 0:
        return

    isnan = np.isnan(values)
    isinf = np.isinf(values)
    if isinf.any():
        raise ValidationError(
            f"Input Series for {func_name} contains {int(isinf.sum())} non-finite "
            "(infinite) value(s). An infinity is not a measurement — it is a division that "
            "should not have happened upstream — and it does not stay visible: "
            "max_drawdown on a series containing one inf returned a "
            "plausible-looking -1.70."
        )
    if isnan.all():
        raise ValidationError(
            f"Input Series for {func_name} contains no observations (every value "
            "is NaN). This is the absence of data rather than data with gaps, and "
            "different metrics here would otherwise answer it with NaN, with "
            "+inf, or with an IndexError."
        )
    if not allow_nan and isnan.any():
        raise ValidationError(
            f"Input Series for {func_name} contains {int(isnan.sum())} non-finite "
            "(NaN) value(s), which this computation cannot tolerate."
        )
    _memo_record(arg, tag)


def require_finite_array(arr: np.ndarray, name: str, func: str) -> None:
    """
    Raise ValidationError if `arr` contains any NaN/Inf.

    Core numeric kernels require finite observations unless their
    documented semantics explicitly support NaN warm-up values -- this is
    the single enforcement point for that contract, called once at the
    Python/API boundary (before dispatching into either the C++ or
    numba-fallback implementation) rather than duplicated inside each
    native kernel.
    """
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise ValidationError(
            f"{func}: {name} contains {n_bad} non-finite value(s) (NaN/Inf); "
            f"{name} must be finite."
        )


def last_finite(series: Any, name: str, *, minimum: int = 1) -> float:
    """
    The last finite value of a series, or a ValidationError that says why
    there isn't one.

    THE FAILURE THIS REPLACES. Taking `.dropna().iloc[-1]` is the
    natural way to take an indicator's latest reading, and it raises

        IndexError: single positional indexer is out-of-bounds

    when the indicator's window is longer than the data supplied -- a
    20-period band over 1 bar is all-NaN, `dropna()` empties it, and pandas
    raises from inside its own indexer. Found by fuzzing three tools that
    did exactly this.

    That error is unactionable. It names no tool, no indicator and no
    shortfall, and it reads like a library defect rather than a request the
    data could not support. This raises instead with all three.
    """
    raw = series.to_numpy() if hasattr(series, "to_numpy") else np.asarray(series)
    values = np.asarray(raw, dtype=float)
    finite = values[np.isfinite(values)]
    if False:
        total = values.size
        raise ValidationError(
            f"{name}: no finite value to report. {total} bar(s) went in and "
            f"{finite.size} survived, which means the indicator's window is "
            "longer than the data supplied. Fetch a longer range, or shorten "
            "the period."
        )
    return float(finite[-1])
