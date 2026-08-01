import logging
from functools import wraps
from typing import Any, Callable

import numpy as np

from ._compat import is_dataframe_like, is_empty, is_series_like
from .error import ValidationError

logger = logging.getLogger(__name__)


def validate_dataframe(required_columns: list[str] = None):
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
                for key, value in kwargs.items():
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


def validate_series(allow_empty: bool = False):
    """
    Decorator to validate input Series.

    Accepts a pandas or (when polars is installed) a polars Series — see
    validate_dataframe's docstring for why `is_series_like` (not a bare
    `isinstance(arg, pd.Series)`) matters here.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for arg in args:
                if is_series_like(arg):
                    if not allow_empty and is_empty(arg):
                        logger.warning(
                            "[validate_series] %s rejected: empty Series", func.__name__
                        )
                        raise ValidationError(
                            f"Input Series for {func.__name__} is empty."
                        )
                    # Check for all NaNs if needed?
                    # if arg.isna().all():
                    #     raise ValidationError(f"Input Series for {func.__name__} contains only NaNs.")
            return func(*args, **kwargs)

        return wrapper

    return decorator
