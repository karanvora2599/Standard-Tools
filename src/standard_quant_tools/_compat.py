"""
Optional Polars interop — this codebase is pandas-first and pandas stays
the default and required backend; Polars support is additive and opt-in
(`pip install standard_quant_tools[polars]`), following the same lazy-
import-guard convention as every other optional dependency here
(`portfolio/optimize.py`'s HAS_SCIPY, `bloomberg_provider.py`'s HAS_BLPAPI).

Scope, stated explicitly (same "honest, not aspirational" spirit as the
data providers' docstrings): this module only detects and validates
Polars objects. It does not attempt to make every function in this
library accept Polars input — see Documentation/14_polars_support.md for
exactly which functions do, and which raise a clear "not supported yet"
error instead of a confusing crash on a pandas-only method deep inside.

Polars deliberately mirrors much of pandas' Series API (`.to_numpy()`,
`.mean()`, `.std()`, `.rolling_mean()`, `.diff()`, `.shift()`), but not
all of it: notably, `polars.Series`/`polars.DataFrame` have no `.empty`
property (use `len(obj) == 0` instead) and no `.dropna()` (Polars calls it
`.drop_nulls()`), and pandas' implicit index-based alignment
(`.index.intersection()`, `.loc[...]`) has no Polars equivalent at all —
Polars has no index concept.

Lives at the top level (sibling to validation.py/error.py/config.py), not
under data/, since it's needed by validation.py and (in later phases)
metrics/analysis/indicators — none of which otherwise depend on data/.
"""

from typing import Any

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    pl = None  # type: ignore[assignment]
    HAS_POLARS = False


def is_series_like(obj: Any) -> bool:
    """True for a pandas Series, or (when polars is installed) a polars Series."""
    if isinstance(obj, pd.Series):
        return True
    return HAS_POLARS and isinstance(obj, pl.Series)


def is_dataframe_like(obj: Any) -> bool:
    """True for a pandas DataFrame, or (when polars is installed) a polars DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return True
    return HAS_POLARS and isinstance(obj, pl.DataFrame)


def is_empty(obj: Any) -> bool:
    """
    True if a pandas or polars Series/DataFrame has zero rows.

    Needed because polars.Series/polars.DataFrame have no `.empty`
    property (unlike pandas) — `len(obj) == 0` works identically for both
    backends, so this is the one generic check callers should use instead
    of `obj.empty`.
    """
    return len(obj) == 0


def to_clean_numpy(series_like: Any, dtype=float) -> np.ndarray:
    """
    Drop missing/NaN values and return a plain numpy array — the one
    conversion nearly every function in this library's "thin wrapper over
    numpy/numba/C++" tier needs at its very first line, for either a
    pandas or a polars Series.

    This is NOT just `series_like.dropna().to_numpy(dtype=dtype)` for
    both backends: polars.Series has no `.dropna()` at all, and its
    closest equivalent, `.drop_nulls()`, only drops `null` (missing)
    entries — NOT floating-point `NaN` values, which polars treats as a
    distinct concept from pandas (pandas' `.dropna()` drops both in one
    call). Matching pandas' actual behavior requires
    `.drop_nulls().drop_nans()` on the polars side.
    """
    if HAS_POLARS and isinstance(series_like, pl.Series):
        return series_like.drop_nulls().drop_nans().to_numpy().astype(dtype)
    return series_like.dropna().to_numpy(dtype=dtype)


def require_polars(context: str) -> None:
    """
    Raise a clear ValidationError if polars isn't installed.

    Use this only where Polars is specifically required (e.g. an explicit
    Polars output-conversion utility) — not as a general gate, since most
    functions in this library work with EITHER backend and shouldn't
    require polars just because a caller happens to pass a polars object.
    """
    if not HAS_POLARS:
        raise ValidationError(
            f"{context} requires polars, which is not installed. Install "
            "it with `pip install standard_quant_tools[polars]`, or pass "
            "a pandas object instead — pandas remains fully supported "
            "without any extra install."
        )
