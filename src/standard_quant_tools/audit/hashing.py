"""Content-fingerprint hashing shared by every other module in this package:
`hash_payload` for JSON-serializable objects (decision records, chain-index
entries), `hash_dataframe` for OHLCV data provenance, and `round_floats`,
which an output passes through before its rounded hash is taken."""

import hashlib
import json
import math
from typing import Any

#: Significant digits an output's rounded hash keeps. Twelve because that is
#: what the reproducibility contract promises across native builds and
#: instruction-set paths: the AVX2+FMA and scalar reductions differ by a few
#: units in the last place (measured worst case 6e-15 relative on
#: rolling_beta), three orders of magnitude inside the twelfth digit.
ROUNDED_SIGNIFICANT_DIGITS = 12


def hash_dataframe(df: Any) -> str:
    """
    Content fingerprint of a DataFrame (columns + values + index), stable
    across runs.

    Covers the COLUMN NAMES and dtypes as well as the values. Hashing values
    alone (pd.util.hash_pandas_object is a per-row digest that never sees the
    column labels) meant two frames holding identical numbers under entirely
    different column names produced the same fingerprint -- e.g. a
    Close/Open frame and a Volume/Adj frame collided, which defeats the point
    of a provenance hash whose whole job is to tell different data apart.

    NOTE (format change): fingerprints produced here differ from those written
    by versions before this fix. Replaying a decision record captured by an
    older version will report a data_source mismatch even when the underlying
    data is unchanged. Only the `content_hash` values inside `data_sources`
    are affected -- the tamper-evident record chain is built by `hash_payload`
    (below), which is unchanged for all normal records, so existing audit
    trails still verify.
    """
    import numpy as np
    import pandas as pd

    hashed = pd.util.hash_pandas_object(df, index=True)
    values_digest = hashlib.sha256(np.asarray(hashed.values).tobytes()).hexdigest()
    # Column identity, in the frame's own column order (a reordering is a
    # genuinely different frame for provenance purposes).
    schema = json.dumps(
        (
            [[str(c), str(dtype)] for c, dtype in zip(df.columns, df.dtypes)]
            if hasattr(df, "columns")
            else []
        ),
        sort_keys=False,
    )
    combined = f"{schema}|{values_digest}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def _canonical_default(obj: Any) -> Any:
    """
    Fallback encoder for objects `json` can't serialize natively.

    Plain `default=str` silently routed numpy arrays (and anything else with
    an abbreviating __str__) through a LOSSY repr: numpy truncates with '...'
    past ~1000 elements, so two different large arrays produced byte-identical
    canonical forms and therefore the same hash. Anything array-like is
    converted to its full element list here instead; genuinely opaque objects
    still fall back to str(), which is fine for the scalars (datetime, Path,
    Decimal) that actually reach this path in practice.
    """
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):  # numpy ndarray / scalar, pandas Series/Index
        return tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    return str(obj)


def round_floats(obj: Any, digits: int = ROUNDED_SIGNIFICANT_DIGITS) -> Any:
    """
    `obj` with every float rounded to `digits` significant decimal digits,
    for a hash that survives a change in the last bits and nothing more.

    Exactly:

    - a finite, non-zero float becomes the double nearest its value
      correctly rounded to `digits` significant digits, ties to even --
      what `format(x, '.11e')` prints for twelve, read back. The rounding
      is decimal and relative, so 1234.56789012345 and 1.23456789012345e-9
      both keep twelve digits;
    - 0.0 and -0.0 both become 0.0: the sign of a zero is a last-bit
      difference, and JSON spells the two apart;
    - NaN, inf and -inf are kept. Their JSON tokens carry no low bits, so
      there is nothing to round, and an infinity keeps its sign;
    - int and bool are untouched, at any size: an integer is a count or an
      index, and rounding one would hide a real difference;
    - str, None and dict KEYS are untouched; dict values, list and tuple
      items are rounded at any depth (a tuple comes back as a list, which
      JSON spells the same way);
    - a numpy array or scalar is converted with `tolist()` first, as the
      exact hash's encoder converts it, then rounded;
    - anything else is returned as it is and hashed as before.

    What it cannot do: a value that is zero in exact arithmetic but comes
    out as rounding noise (1e-17 on one path, -3e-18 on another) agrees to
    no number of significant digits, so the rounded hash differs there. And
    two values a last bit apart that straddle a twelfth-digit rounding
    boundary round apart; the chance is about the size of their difference
    relative to the twelfth digit, measured at none in 7,231 rolling betas.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return obj
        if obj == 0.0:
            return 0.0
        return float(format(obj, f".{digits - 1}e"))
    if isinstance(obj, dict):
        return {key: round_floats(value, digits) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(value, digits) for value in obj]
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):  # numpy ndarray / scalar, pandas Series/Index
        return round_floats(tolist(), digits)
    return obj


def hash_payload(obj: Any) -> str:
    """
    Content fingerprint of a JSON-serializable object (dict/list/scalar).

    Output is unchanged for objects made only of native JSON types (which is
    every DecisionRecord / chain-index entry), so the tamper-evident record
    chain built on this function stays valid across this change -- only the
    previously-lossy non-JSON fallback path behaves differently.
    """
    canonical = json.dumps(obj, sort_keys=True, default=_canonical_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
