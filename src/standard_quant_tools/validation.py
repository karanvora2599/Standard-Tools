import logging

import pandas as pd
import numpy as np
from functools import wraps
from typing import Callable, Any
from .error import ValidationError

logger = logging.getLogger(__name__)

def validate_dataframe(required_columns: list[str] = None):
    """
    Decorator to validate input DataFrame.
    Checks for empty DataFrame and missing columns.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find the DataFrame argument (usually the first one or named 'data'/'df')
            df = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    break
            if df is None:
                # check kwargs
                for key, value in kwargs.items():
                    if isinstance(value, pd.DataFrame):
                        df = value
                        break
            
            if df is not None:
                if df.empty:
                    logger.warning("[validate_dataframe] %s rejected: empty DataFrame", func.__name__)
                    raise ValidationError(f"Input DataFrame for {func.__name__} is empty.")

                if required_columns:
                    missing = [col for col in required_columns if col not in df.columns]
                    if missing:
                        logger.warning("[validate_dataframe] %s rejected: missing columns %s", func.__name__, missing)
                        raise ValidationError(f"Missing required columns in {func.__name__}: {missing}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_series(allow_empty: bool = False):
    """
    Decorator to validate input Series.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for arg in args:
                if isinstance(arg, pd.Series):
                    if not allow_empty and arg.empty:
                        logger.warning("[validate_series] %s rejected: empty Series", func.__name__)
                        raise ValidationError(f"Input Series for {func.__name__} is empty.")
                    # Check for all NaNs if needed?
                    # if arg.isna().all():
                    #     raise ValidationError(f"Input Series for {func.__name__} contains only NaNs.")
            return func(*args, **kwargs)
        return wrapper
    return decorator
