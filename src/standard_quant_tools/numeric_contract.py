"""
One numerical input contract, shared by every public boundary.

`require_finite_array` already existed and was applied rigorously by the
native-backed kernels. Everything else — the older pandas metrics, the cost
primitives, the diagnostics — validated ad hoc or not at all, so the SAME
invalid input produced a `ValidationError` in one function and a
plausible-looking number in another. That inconsistency is itself the bug
this module exists to remove; the individual symptoms were only its
expression.

The three rules, and why each is drawn where it is:

  NON-FINITE INPUT IS NEVER INFORMATION. `+/-inf` in a return or price series
  has no economic reading. It is a division that should not have happened
  upstream, and it does not stay obvious: `max_drawdown` on a series
  containing one `inf` returned **-1.703437775179145**, a drawdown that looks
  measured. Rejected everywhere.

  ALL-NaN IS NOT A SERIES. It carries no observation at all, and functions
  disagreed wildly about what to do with it: `sharpe_ratio` returned NaN,
  `sortino_ratio` returned **+inf** (indistinguishable from a strategy with
  no losing bars), `var_historical` raised `IndexError`. Rejected.

  PARTIAL NaN IS ALLOWED BY DEFAULT. This is the deliberate limit of the
  contract. Indicator warm-up windows, a ticker that lists mid-sample, a
  benchmark with a different holiday calendar — all legitimately produce
  gaps, and many callers drop them internally on purpose. Making them fatal
  would break correct code to catch a problem those callers have already
  handled. Pass `allow_nan=False` where a gap genuinely cannot be tolerated.

A price series gets a stronger rule than a return series: prices must be
strictly POSITIVE. `run_strategy` checked finiteness only, so a `Close` of
`-5.0` passed and produced a total return of **+0.397914** — a plausible
profit computed through a negative price — while a `Close` of `0.0` produced
a silent total wipeout. Both are finite; neither is a price.
"""

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

__all__ = [
    "require_finite_series",
    "require_finite_series_frame",
    "require_positive_price_series",
    "require_positive_start_level",
    "require_aligned",
    "require_positive_int",
    "require_finite_scalar",
    "require_periods_per_year",
    "require_finite_covariance",
]


def _describe(bad: pd.Series, limit: int = 3) -> str:
    """Name the offending positions rather than only their count — on a
    2,000-bar series, 'contains 1 non-finite value' is not actionable."""
    labels = [str(x) for x in bad.index[:limit]]
    more = "" if len(bad) <= limit else f", ... (+{len(bad) - limit} more)"
    return ", ".join(labels) + more


def require_finite_series(
    series: pd.Series,
    name: str,
    func: str,
    allow_nan: bool = True,
    allow_empty: bool = False,
) -> pd.Series:
    """
    Enforce the baseline contract for a numeric Series input.

    Rejects: infinities (always), an all-NaN series (always), an empty series
    (unless `allow_empty`), and — only when `allow_nan=False` — any NaN.

    Returns the series unchanged so this can be used inline.
    """
    if not isinstance(series, pd.Series):
        raise ValidationError(
            f"{func}: {name} must be a pandas Series, got {type(series).__name__}"
        )
    if series.empty:
        if allow_empty:
            return series
        raise ValidationError(f"{func}: {name} is empty")

    values = pd.to_numeric(series, errors="coerce")
    isna = values.isna()

    infinite = values.isin([np.inf, -np.inf])
    if infinite.any():
        raise ValidationError(
            f"{func}: {name} contains {int(infinite.sum())} non-finite (infinite) "
            f"value(s) at "
            f"{_describe(values[infinite])}. An infinity is not a measurement — "
            "it is a division that should not have happened upstream (commonly a "
            "zero or negative price reaching pct_change), and it does not stay "
            "visible: it can produce a finite, plausible-looking result several "
            "steps later."
        )
    if isna.all():
        raise ValidationError(
            f"{func}: {name} contains no observations (every value is NaN). "
            "This is not a series with missing data — it is the absence of data, "
            "and different functions here would otherwise answer it with NaN, "
            "with +inf, or with an IndexError."
        )
    if not allow_nan and isna.any():
        raise ValidationError(
            f"{func}: {name} contains {int(isna.sum())} non-finite (NaN) value(s) at "
            f"{_describe(values[isna])}, and this computation cannot tolerate "
            "gaps. Drop or fill them explicitly so the choice is yours rather "
            "than implicit."
        )
    return series


def require_positive_price_series(
    series: pd.Series, name: str, func: str, allow_nan: bool = True
) -> pd.Series:
    """
    A price series must be finite AND strictly positive.

    Finiteness alone is not enough, which is the whole point of a separate
    helper: `0.0` and `-5.0` are perfectly finite and are not prices. Both
    survive a finite check and then corrupt everything derived from them —
    `pct_change` against a zero divides by zero, and a negative denominator
    silently flips the sign of a return.
    """
    require_finite_series(series, name, func, allow_nan=allow_nan)
    values = pd.to_numeric(series, errors="coerce")
    nonpositive = values.notna() & (values <= 0)
    if nonpositive.any():
        raise ValidationError(
            f"{func}: {name} contains {int(nonpositive.sum())} non-positive "
            f"value(s) at {_describe(values[nonpositive])}. A price must be > 0: "
            "zero makes a return calculation divide by zero, and a negative "
            "price flips the sign of every return derived from it while "
            "remaining perfectly finite — measured on run_strategy, a single "
            "-5.0 close produced a total return of +0.397914."
        )
    return series


def require_positive_start_level(series: pd.Series, name: str, func: str) -> pd.Series:
    """
    An equity/level series must START strictly positive.

    Deliberately weaker than require_positive_price_series, and the
    difference matters: a leveraged position CAN be wiped out, so an equity
    curve legitimately reaches zero or goes negative at its tail —
    run_strategy applies no bankruptcy floor and cagr() already has documented
    handling for exactly that case. Demanding positivity everywhere would
    reject a real, representable outcome.

    What must hold is that the DENOMINATOR is positive. Both
    cumulative_return (`last / first`) and the drawdown ratio
    (`(series - cummax) / cummax`) divide by a quantity fixed by the opening
    level: cummax is non-decreasing, so a positive first value keeps it
    positive for the whole series. A starting level of 0.0 divided by zero;
    a negative one silently flipped the sign of every result derived from it.
    """
    require_finite_series(series, name, func, allow_nan=True)
    first = pd.to_numeric(series, errors="coerce").dropna()
    if first.empty:
        raise ValidationError(f"{func}: {name} has no usable observations")
    start = float(first.iloc[0])
    if start <= 0:
        raise ValidationError(
            f"{func}: {name} starts at {start}, but the opening level is the "
            "denominator of every quantity derived from it (cumulative return "
            "divides by it; the drawdown ratio divides by a running maximum "
            "seeded from it). Zero divides by zero and a negative start flips "
            "the sign of the result while looking entirely ordinary. Note this "
            "constrains only the START — a wiped-out curve that reaches zero or "
            "goes negative later is a real outcome and stays supported."
        )
    return series


def require_aligned(
    left: pd.Series, right: pd.Series, left_name: str, right_name: str, func: str
) -> None:
    """
    Two paired series must share an index, not merely a length.

    Equal length is not alignment. These have the same length and describe
    different days:

        left  index: Jan 1, Jan 2, Jan 3
        right index: Jan 2, Jan 3, Jan 4

    pandas operations label-align, while NumPy/native paths index
    positionally — so the same two inputs mean different things depending on
    which execution path happens to run, which is exactly the backend
    divergence class this codebase pins tests against.
    """
    if len(left) != len(right):
        raise ValidationError(
            f"{func}: {left_name} has {len(left)} rows but {right_name} has "
            f"{len(right)}"
        )
    if not left.index.equals(right.index):
        overlap = len(left.index.intersection(right.index))
        raise ValidationError(
            f"{func}: {left_name} and {right_name} have the same length but "
            f"different indexes ({overlap} of {len(left)} labels in common). "
            "Equal length is not alignment: pandas would label-align these "
            "while a NumPy or native path would pair them positionally, so the "
            "same inputs would mean different things on different execution "
            "paths. Align them explicitly first."
        )


def require_positive_int(
    value: Any, name: str, func: str, maximum: Optional[int] = None
) -> int:
    """A window/period/count must be a positive whole number."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer)):
        raise ValidationError(
            f"{func}: {name} must be a positive whole number, got "
            f"{type(value).__name__} ({value!r})"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{func}: {name} must be finite, got {value!r}")
    if not number.is_integer():
        raise ValidationError(
            f"{func}: {name} counts whole periods and must be an integer, got {value!r}"
        )
    result = int(number)
    if result < 1:
        raise ValidationError(
            f"{func}: {name} must be >= 1, got {result}. A negative period is not "
            "merely invalid — pandas reads a negative shift/pct_change period as a "
            "FORWARD window, which reads future data."
        )
    if maximum is not None and result > maximum:
        raise ValidationError(f"{func}: {name}={result} exceeds the maximum {maximum}")
    return result


def require_finite_scalar(
    value: Any,
    name: str,
    func: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """
    A scalar must be a finite number, checked BEFORE any range comparison.

    Order is the point. Range guards are written as comparisons, and every
    comparison against NaN is False — so `if rate <= 0: raise` never fires
    for NaN, and the NaN flows on to produce a result that carries a success
    flag and no numbers.
    """
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValidationError(
            f"{func}: {name} must be a number, got {type(value).__name__} ({value!r})"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(
            f"{func}: {name} must be finite, got {value!r}. NaN compares False "
            "against every bound, so it passes range checks written as "
            "comparisons and corrupts the result silently."
        )
    if minimum is not None and number < minimum:
        raise ValidationError(f"{func}: {name} must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ValidationError(f"{func}: {name} must be <= {maximum}, got {number}")
    return number


def require_periods_per_year(value: Any, func: str) -> int:
    """
    The annualization factor must be a positive integer.

    Left unchecked it produces confidently wrong numbers rather than errors:
    `periods_per_year=-252` returned a CAGR of **-0.5350151890419428**, which
    reads as a perfectly ordinary annual loss. Zero raised a bare
    `ZeroDivisionError` from inside the arithmetic, and NaN propagated.
    """
    return require_positive_int(value, "periods_per_year", func, maximum=31_536_000)


def require_finite_covariance(
    cov: np.ndarray, name: str, func: str, check_symmetry: bool = True
) -> np.ndarray:
    """
    A covariance matrix must be square, finite, and (by default) symmetric.

    A NaN here is especially quiet: the usual positive-definiteness guard is
    `if portfolio_variance <= 0`, which NaN does not satisfy, so a NaN
    covariance passed the very check meant to catch a degenerate one and
    propagated through every iteration that followed.
    """
    array = np.asarray(cov, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValidationError(
            f"{func}: {name} must be a square 2-D matrix, got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        n_bad = int(np.sum(~np.isfinite(array)))
        raise ValidationError(
            f"{func}: {name} contains {n_bad} non-finite entr(ies). A NaN "
            "covariance does not trip the usual `variance <= 0` degeneracy "
            "check — NaN satisfies no comparison — so it would flow through "
            "the whole solve and emerge as NaN weights."
        )
    if check_symmetry and not np.allclose(array, array.T, rtol=1e-9, atol=1e-12):
        worst = float(np.max(np.abs(array - array.T)))
        raise ValidationError(
            f"{func}: {name} is not symmetric (largest |A - A'| = {worst:.3e}). "
            "A covariance matrix is symmetric by definition; an asymmetric one "
            "means it was not built as a covariance, and eigenvalue-based code "
            "downstream would silently use only one triangle."
        )
    return array


def require_finite_series_frame(
    frame: "pd.DataFrame", name: str, func: str, allow_nan: bool = True
) -> "pd.DataFrame":
    """
    Frame-level counterpart to require_finite_series.

    Matrix routines fail differently from scalar ones: an infinity reaching
    numpy.linalg.svd raises a bare `LinAlgError: SVD did not converge`, which
    names neither the input nor the offending column, and a caller reads it as
    an algorithmic failure rather than as bad data. Checked at the boundary so
    the message names the column instead.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError(
            f"{func}: {name} must be a pandas DataFrame, got {type(frame).__name__}"
        )
    if frame.empty:
        raise ValidationError(f"{func}: {name} is empty")
    values = frame.to_numpy(dtype=float)
    infinite = np.isinf(values)
    if infinite.any():
        cols = [str(c) for c, flag in zip(frame.columns, infinite.any(axis=0)) if flag]
        raise ValidationError(
            f"{func}: {name} contains {int(infinite.sum())} non-finite (infinite) "
            f"value(s) in column(s) {cols}. Reaching a matrix decomposition, this "
            "surfaces as a bare LinAlgError that names neither the input nor the "
            "column responsible."
        )
    isnan = np.isnan(values)
    if isnan.all():
        raise ValidationError(f"{func}: {name} contains no observations")
    if not allow_nan and isnan.any():
        raise ValidationError(
            f"{func}: {name} contains {int(isnan.sum())} NaN value(s) and this "
            "computation cannot tolerate gaps"
        )
    return frame
